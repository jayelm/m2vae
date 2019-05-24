"""
Train the M2VAE
"""

import os
import sys
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from tqdm import tqdm

import data
import mvae
import models
import util


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
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Train M2VAE',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_file', default='data/train_x_lpd_5_phr.npz',
                        help='Cleaned/processed LP5 dataset')
    parser.add_argument('--exp_dir', default='exp/debug/',
                        help='Experiment directory')
    parser.add_argument('--activation', default='relu', choices=['swish', 'lrelu', 'relu'],
                        help='Nonlinear activation in encoders/decoders')
    parser.add_argument('--durations', action='store_true', help='Train the model where hits/durations predicted separately')
    parser.add_argument('--hits_only', action='store_true', help='Predict hits only in the durations model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of multitrack embeddings')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--no_kl', action='store_true', help="Don't use KL (vanilla autoencoder)")
    parser.add_argument('--n_tracks', type=int, default=5, help='Number of tracks (between 1 and 5 inclusive)')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--annealing_epochs', type=int, default=250, help='Annealing epochs')
    parser.add_argument('--mse_factor', type=float, default=1.00, help='Weight on MSE loss')
    parser.add_argument('--kl_factor', type=float, default=0.001, help='Constant weight on KL divergence')
    parser.add_argument('--max_train', type=int, default=None, help='Maximum training examples to train on')
    parser.add_argument('--resume', action='store_true', help='Try to resume from checkpoint')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Load data into CUDA-pinned memory')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--f1_interval', type=int, default=5, help='How often to calculate f1 score')
    parser.add_argument('--log_interval', type=int, default=100, help='How often to log progress (in batches)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    parser.add_argument('--debug', action='store_true', help='Load tiny data file')

    args = parser.parse_args()

    if args.hits_only and not args.durations:
        parser.error('--hits_only requires --durations')

    if args.n_tracks < 1 or args.n_tracks > 5:
        parser.error('--n_tracks must be between 1 and 5 inclusive')

    if args.durations and 'durations' not in args.data_file:
        parser.error('Load the durations datafile to run durations model')

    # Make experiment directory
    resumable = args.resume and util.is_resumable(args.exp_dir)
    os.makedirs(args.exp_dir, exist_ok=True)
    if not resumable:
        util.save_args(args, args.exp_dir)

    # Seed
    random = np.random.RandomState(args.seed)

    lpd_file = args.data_file
    if args.debug and 'debug' not in lpd_file:
        lpd_file = lpd_file.replace('.npz', '_debug.npz')

    if args.debug or args.durations:
        lpd_raw = np.load(lpd_file)['arr_0']
    else:
        lpd_raw = data.load_data_from_npz(lpd_file)

    if args.hits_only:
        lpd_raw = lpd_raw > 0

    if args.max_train is not None:
        lpd_raw = lpd_raw[:args.max_train]

    # Calcualte mean positive weight
    pos_prop = lpd_raw.mean()
    lpds = data.train_val_test_split(lpd_raw, random_state=random)
    del lpd_raw
    dataloaders = {}
    for split, dataset in lpds.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers, shuffle=True,
            pin_memory=args.pin_memory
        )

    def encoder_func():
        bar_encoder = models.ConvBarEncoder()
        track_encoder = models.RNNTrackEncoder(bar_encoder, output_size=args.hidden_size)
        muvar_encoder = models.StdMuVarEncoder(track_encoder, input_size=args.hidden_size, hidden_size=args.hidden_size, output_size=args.hidden_size)
        return muvar_encoder

    def decoder_func():
        bar_decoder = models.RNNTrackDecoder(input_size=args.hidden_size)
        note_decoder = models.ConvBarDecoder(bar_decoder)
        return note_decoder

    # Model
    model = mvae.MVAE(encoder_func, decoder_func, n_tracks=args.n_tracks, hidden_size=args.hidden_size)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss
    loss = mvae.ELBOLoss(pos_weight=torch.tensor(1 / pos_prop),
                         kl_factor=args.kl_factor,
                         mse_factor=args.mse_factor)

    if args.cuda:
        model = model.cuda()
        loss = loss.cuda()

    # If resume, load metrics; otherwise init metrics
    if resumable:
        ckpt = util.load_checkpoint(args.exp_dir)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

        metrics = util.load_metrics(args.exp_dir)
        start_epoch = metrics['current_epoch'] + 1
        print("Resuming from epoch {}".format(metrics['current_epoch']))
    else:
        metrics = defaultdict(list)
        metrics['best_f1'] = -10
        metrics['best_loss'] = float('inf')
        metrics['best_epoch'] = 0
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
