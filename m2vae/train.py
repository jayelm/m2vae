"""
Train the M2VAE
"""

import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

import data
import mvae
import models
import util


def train(epoch, model, optimizer, loss, dataloader, args):
    model.train()
    loss_meter = util.AverageMeter()
    #  f1_meter = util.AverageMeter()


    for i, tracks in enumerate(dataloader):
        if epoch < args.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = (float(i + (epoch - 1) * len(dataloader) + 1) /
                                float(args.annealing_epochs * len(dataloader)))
        else:
            annealing_factor = 1.0
        # tracks: [batch_size, n_bar, n_timesteps, n_pitches, n_tracks]
        if args.cuda:
            tracks = tracks.cuda()
        batch_size = tracks.shape[0]
        # Refresh the optimizer
        optimizer.zero_grad()

        total_loss = 0
        n_elbo_terms = 0

        # Forward pass - all data
        tracks_recon, mu, logvar = model(tracks)

        total_loss += loss(tracks_recon, tracks, mu, logvar,
                           annealing_factor=annealing_factor)
        n_elbo_terms += 1

        # All data - F1 score
        #  tracks_np = tracks.detach().cpu().numpy().flatten().astype(np.bool)
        #  tracks_recon_np = tracks_recon.detach().cpu().numpy().flatten() > 1
        #  train_f1 = f1_score(tracks_np, tracks_recon_np)

        # Forward pass - single modalities
        #  for t in range(tracks.shape[-1]):
            #  track = tracks[t]

        loss_meter.update(total_loss, batch_size)
        #  f1_meter.update(train_f1)

        total_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                epoch, i * batch_size, len(dataloader.dataset),
                100. * i / len(dataloader), loss_meter.avg, annealing_factor))

    metrics = {
        'loss': loss_meter.avg.item(),
        #  'f1': f1_meter.avg.item()
    }
    print('====> Epoch: {}\ttrain {}'.format(
        epoch, ' '.join('{}: {:.4f}'.format(m, v) for m, v in metrics.items())
    ))
    return metrics


def test(epoch, model, optimizer, loss, dataloader, args):
    model.eval()
    loss_meter = util.AverageMeter()
    f1_meter = util.AverageMeter()

    for i, tracks in enumerate(dataloader):
        if args.cuda:
            tracks = tracks.cuda()
        batch_size = tracks.shape[0]

        total_loss = 0
        n_elbo_terms = 0

        # Forward pass - all data
        with torch.no_grad():
            tracks_recon, mu, logvar = model(tracks)

        total_loss += loss(tracks_recon, tracks, mu, logvar,
                           annealing_factor=1.0)
        n_elbo_terms += 1

        # All data - F1 score
        tracks_np = tracks.detach().cpu().numpy().flatten().astype(np.bool)
        tracks_recon_np = tracks_recon.detach().cpu().numpy().flatten() > 1
        f1 = f1_score(tracks_np, tracks_recon_np)

        # Forward pass - single modalities
        #  for t in range(tracks.shape[-1]):
            #  track = tracks[t]

        loss_meter.update(total_loss, batch_size)
        f1_meter.update(f1, batch_size)

    metrics = {
        'loss': loss_meter.avg.item(),
        'f1': f1_meter.avg.item()
    }
    print('====> Epoch: {}\tval {}'.format(
        epoch, ' '.join('{}: {:.4f}'.format(m, v) for m, v in metrics.items())
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
                        help='Cleaned/processed LP5 dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--annealing_epochs', type=int, default=20, help='Annealing epochs')
    parser.add_argument('--resume', action='store_true', help='Try to resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Load data into CUDA-pinned memory')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=100, help='How often to log progress (in batches)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    parser.add_argument('--debug', action='store_true', help='Load tiny data file')

    args = parser.parse_args()

    # Make experiment directory
    resumable = args.resume and util.is_resumable(args.exp_dir)
    os.makedirs(args.exp_dir, exist_ok=True)
    if resumable:
        # Make sure args are the same
        old_args = util.load_args(args.exp_dir)
        old_args['resume'] = True
        assert old_args == vars(args)
    else:
        util.save_args(args, args.exp_dir)

    # Seed
    random = np.random.RandomState(args.seed)

    if args.debug:
        lpd_raw = np.load(args.data_file + '.debug')['arr_0']
    else:
        lpd_raw = data.load_data_from_npz(args.data_file)
    lpds = data.train_val_test_split(lpd_raw, random_state=random)
    del lpd_raw
    dataloaders = {}
    for split, dataset in lpds.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True,
            pin_memory=args.pin_memory
        )

    def encoder_func():
        bar_encoder = models.ConvBarEncoder()
        track_encoder = models.RNNTrackEncoder(bar_encoder)
        muvar_encoder = models.StdMuVarEncoder(track_encoder)
        return muvar_encoder

    def decoder_func():
        bar_decoder = models.RNNBarDecoder()
        note_decoder = models.ConvNoteDecoder(bar_decoder)
        return note_decoder

    # Model
    model = mvae.MVAE(encoder_func, decoder_func, n_tracks=5, hidden_size=256)

    if args.cuda:
        model = model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss
    loss = mvae.ELBOLoss()

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
        metrics['best_loss'] = float('inf')
        metrics['best_epoch'] = 0
        start_epoch = 1

    if start_epoch > args.epochs:
        raise RuntimeError("start_epoch {} > total epochs {}".format(
            start_epoch, args.epoch))


    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train(epoch, model, optimizer, loss, dataloaders['train'], args)
        val_metrics = test(epoch, model, optimizer, loss, dataloaders['val'], args)

        for metric, value in train_metrics.items():
            metrics['train_{}'.format(metric)].append(value)
        metrics['current_epoch'] = epoch

        is_best = train_metrics['loss'] < metrics['best_loss']
        if is_best:
            metrics['best_loss'] = train_metrics['loss']
            metrics['best_epoch'] = epoch

        # Save model
        util.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch': epoch
        }, is_best, args.exp_dir)

        # Save metrics
        util.save_metrics(metrics, args.exp_dir)
