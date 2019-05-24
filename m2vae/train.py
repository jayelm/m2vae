"""
Train an m2vae model.
"""

import os
import sys
from collections import defaultdict
from datetime import datetime

import torch

import numpy as np
from tqdm import tqdm

import data
import mvae
import models
import util
import io_util
import wrappers


def fast_f1(true, pred):
    """
    This is faster than detaching to cpu and using
    sklearn.metrics.f1_score
    """
    true = true.view(-1)
    pred = pred.view(-1)

    hits = true == pred
    misses = true != pred

    true_positives = (pred * hits).sum().float()
    false_positives = ((1 - pred) * (misses)).sum().float()
    false_negatives = (pred * (misses)).sum().float()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    try:
        return (2 * precision * recall / (precision + recall))
    except ZeroDivisionError:
        return torch.zeros(())


def init_meters(*metrics):
    """Return an averagemeter for each metric passed"""
    return {m: util.AverageMeter() for m in metrics}


def init_metrics():
    metrics = defaultdict(list)
    metrics['best_f1'] = -10
    metrics['best_loss'] = float('inf')
    metrics['best_epoch'] = 0
    return metrics


def compute_metrics(meters):
    """
    Compute averages from meters. Handle tensors vs floats (always return a
    float)
    """
    metrics = {m: vs.avg for m, vs in meters.items()}
    metrics = {m: v if isinstance(v, float) else v.item() for m, v in metrics.items()}
    return metrics


def compute_kl_annealing_factor(batch, epoch, n_batches, annealing_epochs):
    return (float(batch + (epoch - 1) * n_batches + 1) /
            float(annealing_epochs * n_batches))


def train(epoch, model, optimizer, loss, dataloader, args):
    model.train()
    report_f1 = (epoch % args.f1_interval == 0)
    if report_f1:
        meters = init_meters('loss', 'annealing_factor', 'recon_loss', 'kl_divergence', 'f1')
    else:
        meters = init_meters('loss', 'annealing_factor', 'recon_loss', 'kl_divergence')

    for i, tracks in enumerate(dataloader):
        if args.no_kl:
            annealing_factor = 0.0
        elif epoch < args.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = compute_kl_annealing_factor(i, epoch, len(dataloader),
                                                           args.annealing_epochs)
        else:
            annealing_factor = 1.0
        # tracks: [batch_size, n_bar, n_timesteps, n_pitches, n_tracks]
        tracks = tracks[:, :, :, :, :args.n_tracks]
        if args.cuda:
            tracks = tracks.cuda()
        batch_size = tracks.shape[0]

        # Refresh the optimizer
        optimizer.zero_grad()

        total_loss = 0
        n_elbo_terms = 0

        # Forward pass - all data
        tracks_recon, mu, logvar = model(tracks)

        this_loss, recon_loss, kl_divergence = loss(tracks_recon, tracks, mu, logvar,
                                                    annealing_factor=annealing_factor)

        total_loss += this_loss
        n_elbo_terms += 1

        meters['loss'].update(total_loss, batch_size)
        meters['annealing_factor'].update(annealing_factor, batch_size)
        meters['recon_loss'].update(recon_loss, batch_size)
        meters['kl_divergence'].update(kl_divergence, batch_size)

        # All data - F1 score
        if report_f1:
            f1 = fast_f1(tracks.type(torch.ByteTensor),
                         (tracks_recon > 0).type(torch.ByteTensor))
            meters['f1'].update(f1, batch_size)

        total_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                epoch, i * batch_size, len(dataloader.dataset),
                100. * i / len(dataloader), meters['loss'].avg, annealing_factor))

    metrics = compute_metrics(meters)
    if not report_f1:
        metrics['f1'] = -1.0
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('{} Epoch: {}\ttrain {}'.format(
        dt, epoch, ' '.join('{}: {:.4f}'.format(m, v) for m, v in metrics.items())
    ))
    return metrics


def test(epoch, model, optimizer, loss, dataloader, args):
    model.eval()
    report_f1 = (epoch % args.f1_interval == 0)
    if report_f1:
        meters = init_meters('loss', 'recon_loss', 'kl_divergence', 'f1')
    else:
        meters = init_meters('loss', 'recon_loss', 'kl_divergence')

    with torch.no_grad():
        for i, tracks in enumerate(dataloader):
            tracks = tracks[:, :, :, :, :args.n_tracks]
            if args.cuda:
                tracks = tracks.cuda()
            batch_size = tracks.shape[0]

            total_loss = 0
            n_elbo_terms = 0

            # Forward pass - all data
            tracks_recon, mu, logvar = model(tracks)

            # Note: here total val loss assumes no annealing
            this_loss, recon_loss, kl_divergence = loss(tracks_recon, tracks, mu, logvar,
                                                        annealing_factor=0.0 if args.no_kl else 1.0)

            total_loss += this_loss
            n_elbo_terms += 1

            # All data - F1 score
            if report_f1:
                f1 = fast_f1(tracks.type(torch.ByteTensor),
                             (tracks_recon > 0).type(torch.ByteTensor))
                meters['f1'].update(f1, batch_size)

            meters['loss'].update(total_loss, batch_size)
            meters['recon_loss'].update(recon_loss, batch_size)
            meters['kl_divergence'].update(kl_divergence, batch_size)

        metrics = compute_metrics(meters)
        if not report_f1:
            metrics['f1'] = -1.0
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} Epoch: {}\tval {}'.format(
            dt, epoch, ' '.join('{}: {:.4f}'.format(m, v) for m, v in metrics.items())
        ))

    return metrics


if __name__ == '__main__':
    args = io_util.parse_args('train', desc=__doc__)

    # Make experiment directory
    resumable = args.resume and util.is_resumable(args.exp_dir)
    os.makedirs(args.exp_dir, exist_ok=True)
    if not resumable:
        util.save_args(args, args.exp_dir)

    # Seed
    random = np.random.RandomState(args.seed)

    dataloaders, pos_prop = wrappers.load_data(args, random_state=random,
                                               use_random_transpose=True)
    model, optimizer, loss = wrappers.build_mvae(args, pos_prop=pos_prop)

    # If resume, load metrics; otherwise init metrics
    if resumable:
        util.restore_checkpoint(model, optimizer, args.exp_dir)

        metrics = util.load_metrics(args.exp_dir)
        start_epoch = metrics['current_epoch'] + 1
        print("Resuming from epoch {}".format(metrics['current_epoch']))
    else:
        metrics = init_metrics()
        start_epoch = 1

    if start_epoch > args.epochs:
        raise RuntimeError("start_epoch {} > total epochs {}".format(
            start_epoch, args.epochs))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train(epoch, model, optimizer, loss, dataloaders['train'], args)
        val_metrics = test(epoch, model, optimizer, loss, dataloaders['val'], args)

        for metric, value in train_metrics.items():
            metrics['train_{}'.format(metric)].append(value)
        for metric, value in val_metrics.items():
            metrics['val_{}'.format(metric)].append(value)
        metrics['current_epoch'] = epoch

        is_best = val_metrics['f1'] > metrics['best_f1']
        if is_best:
            metrics['best_f1'] = val_metrics['f1']
            metrics['best_loss'] = val_metrics['loss']
            metrics['best_epoch'] = epoch

        # Save model
        util.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch': epoch
        }, is_best, args.exp_dir)

        # Save metrics
        util.save_metrics(metrics, args.exp_dir)
