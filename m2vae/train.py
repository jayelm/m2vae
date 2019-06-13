"""
Train an m2vae model.
"""

import os
import sys
from collections import defaultdict
import contextlib
from itertools import combinations

import torch

import numpy as np
from tqdm import tqdm

import pretty_midi

import data
import mvae
import models
import util
import io_util
import wrappers

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


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


def enumerate_combinations(n):
    """Enumerate entire pool of combinations.

    We use this to define the domain of ELBO terms,
    (the pool of 2^19 ELBO terms).
    @param n: integer
              number of features (19 for Celeb19)
    @return: a list of ALL permutations
    """
    combos = []
    for i in range(2, n):  # 1 to n - 1
        _combos = list(combinations(range(n), i))
        combos  += _combos

    combos_np = np.zeros((len(combos), n))
    for i in range(len(combos)):
        for idx in combos[i]:
            combos_np[i][idx] = 1

    combos_np = combos_np.astype(np.bool)
    return combos_np


def sample_combinations(pool, random_state=None, size=1):
    """Return boolean list of which data points to use to compute a modality.
    Ignore combinations that are all True or only contain a single True.
    @param pool: np.array
                 enumerating all possible combinations.
    @param size: integer (default: 1)
                 number of combinations to sample.
    """
    if random_state is None:
        random_state = np.random

    n_modalities = pool.shape[1]
    pool_size    = len(pool)
    pool_sums    = np.sum(pool, axis=1)
    pool_dist    = np.bincount(pool_sums)
    pool_space   = np.where(pool_dist > 0)[0]

    sample_pool  = random_state.choice(pool_space, size, replace=True)
    sample_dist  = np.bincount(sample_pool)
    if sample_dist.size < n_modalities:
        zeros_pad   = np.zeros(n_modalities - sample_dist.size).astype(np.int)
        sample_dist = np.concatenate((sample_dist, zeros_pad))

    sample_combo = []
    for ix in range(n_modalities):
        if sample_dist[ix] > 0:
            pool_i  = pool[pool_sums == ix]
            combo_i = random_state.choice(range(pool_i.shape[0]),
                                          size=sample_dist[ix],
                                          replace=False)
            sample_combo.append(pool_i[combo_i])

    sample_combo = np.concatenate(sample_combo)
    return sample_combo


def run(split, epoch, model, optimizer, loss, dataloaders, m_combos, args,
        random_state=None):
    """
    Run the model for a single epoch.
    """
    training = split == 'train'
    dataloader = dataloaders[split]
    if training:
        model.train()
        context = contextlib.suppress
    else:
        model.eval()
        context = torch.no_grad

    report_f1 = (epoch % args.f1_interval == 0)
    measures = ['loss', 'annealing_factor', 'recon_loss', 'kl_divergence']
    if report_f1:
        measures.append('f1')
    meters = init_meters(*measures)

    with context():
        for batch_i, (tracks, notes) in enumerate(dataloader):
            if args.no_kl:
                annealing_factor = 0.0
            elif training and epoch < args.annealing_epochs:
                annealing_factor = compute_kl_annealing_factor(batch_i, epoch, len(dataloader),
                                                               args.annealing_epochs)
            else:
                annealing_factor = 1.0  # No annealing at val/test
            # tracks: [batch_size, n_bar, n_timesteps, n_pitches, n_tracks]
            tracks = tracks[:, :, :, :, :args.n_tracks]
            if args.cuda:
                tracks = tracks.cuda()
                if notes is not None:
                    notes = notes.cuda()
            batch_size = tracks.shape[0]

            # Split tracks into list so we can zero them out
            tracks = [tracks[:, :, :, :, i] for i in range(args.n_tracks)]

            # Refresh the optimizer
            if training:
                optimizer.zero_grad()

            total_loss = 0
            total_recon_loss = 0
            total_kl_divergence = 0

            # Forward pass - all data
            tracks_recon, mu, logvar = model(tracks, notes)

            this_loss, recon_loss, kl_divergence = loss(tracks_recon, tracks, mu, logvar,
                                                        annealing_factor=annealing_factor)
            total_loss += this_loss
            total_recon_loss += recon_loss
            total_kl_divergence += kl_divergence

            if training:
                # Additional forward passes
                # Individual tracks
                if not args.no_single_pass:
                    for i in range(args.n_tracks):
                        tracks_single = [tracks[t] if t == i else None
                                         for t in range(args.n_tracks)]
                        tracks_single_recon, mu, logvar = model(tracks_single, notes)

                        this_loss, recon_loss, kl_divergence = loss(tracks_single_recon, tracks_single, mu, logvar,
                                                                    annealing_factor=annealing_factor)
                        total_loss += this_loss
                        total_recon_loss += recon_loss
                        total_kl_divergence += kl_divergence

                # Subsampled combinations of tracks
                if args.approx_m > 0:
                    sample_combos = sample_combinations(m_combos, random_state=random_state, size=args.approx_m)
                    for sample_combo in sample_combos:
                        tracks_samp = [track if i else None for
                                       i, track in zip(sample_combo, tracks)]
                        tracks_samp_recon, mu, logvar = model(tracks_samp, notes)

                        this_loss, recon_loss, kl_divergence = loss(tracks_samp_recon, tracks_samp, mu, logvar,
                                                                    annealing_factor=annealing_factor)
                        total_loss += this_loss
                        total_recon_loss += recon_loss
                        total_kl_divergence += kl_divergence

                # SGD step
                total_loss.backward()
                optimizer.step()

            meters['loss'].update(total_loss, batch_size)
            meters['annealing_factor'].update(annealing_factor, batch_size)
            meters['recon_loss'].update(total_recon_loss, batch_size)
            meters['kl_divergence'].update(total_kl_divergence, batch_size)

            # All data - F1 score
            if report_f1:
                f1 = sum(
                    fast_f1(t.type(torch.ByteTensor), (tr > 0).type(torch.ByteTensor))
                    for t, tr in zip(tracks, tracks_recon))
                f1 /= args.n_tracks
                meters['f1'].update(f1, batch_size)

            if training and batch_i % args.log_interval == 0:
                logging.info('Epoch {}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_i * batch_size, len(dataloader.dataset),
                    100 * batch_i / len(dataloader), meters['loss'].avg, annealing_factor))

    metrics = compute_metrics(meters)
    if not report_f1:
        metrics['f1'] = -1.0
    logging.info('Epoch {}\t{} {}'.format(
        epoch, split.upper(), ' '.join('{}: {:.4f}'.format(m, v) for m, v in metrics.items())
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
                                               use_random_transpose=True,
                                               note_condition=args.note_condition)
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

    # Enumerate subsampled modality combinations
    m_combos = enumerate_combinations(args.n_tracks)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run('train', epoch, model, optimizer, loss, dataloaders, m_combos, args, random_state=random)
        val_metrics = run('val', epoch, model, optimizer, loss, dataloaders, m_combos, args, random_state=random)

        for metric, value in train_metrics.items():
            try:
                metrics['train_{}'.format(metric)].append(value)
            except KeyError:
                pass  # Could be missing due to resuming from older code
        for metric, value in val_metrics.items():
            try:
                metrics['val_{}'.format(metric)].append(value)
            except KeyError:
                pass
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
