"""
Train the M2VAE
"""

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import data
import mvae
import models
import util


def train(epoch, model, optimizer, loss, dataloader, args):
    model.train()
    train_loss_meter = util.AverageMeter()

    # Compute KL annealing
    annealing_factor = 1.0

    for i, tracks in enumerate(dataloader):
        # tracks: [batch_size, n_bar, n_timesteps, n_pitches, n_tracks]
        if args.cuda:
            tracks = tracks.cuda()
        batch_size = tracks.shape[0]
        # Refresh the optimizer
        optimizer.zero_grad()

        train_loss = 0
        n_elbo_terms = 0

        # Forward pass - all data
        tracks_recon, mu, logvar = model(tracks)

        train_loss += loss(tracks_recon, tracks, mu, logvar,
                           annealing_factor=annealing_factor)
        n_elbo_terms += 1

        # Forward pass - single modalities
        #  for t in range(tracks.shape[-1]):
            #  track = tracks[t]
        train_loss_meter.update(train_loss)

        train_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                epoch, i * batch_size, len(dataloader.dataset),
                100. * i / len(dataloader), train_loss_meter.avg, annealing_factor))

    print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Train M2VAE',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_file', default='data/train_x_lpd_5_phr.npz',
                        help='Cleaned/processed LP5 dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=1, help='How often to log progress (in batches)')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    parser.add_argument('--debug', action='store_true', help='Load tiny data file')

    args = parser.parse_args()

    lpd = data.LPD(args.data_file + ('.debug' if args.debug else ''))
    lpd_loader = DataLoader(lpd, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=True)

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

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, loss, lpd_loader, args)
