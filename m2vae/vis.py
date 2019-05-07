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

from pypianoroll import Track
import matplotlib.pyplot as plt

import data
import mvae
import models
import util


def to_track(track_np, is_drum=False, name='piano'):
    if len(track_np.shape) != 3:
        raise ValueError("Pass in an array of shape [n_bar, n_timesteps, n_pitches]")
    n_bar = track_np.shape[0]
    n_timesteps = track_np.shape[1]
    tot_timesteps = n_bar * n_timesteps
    n_pitches = track_np.shape[2]
    track_np_flat = track_np.reshape(tot_timesteps, -1)
    padding_amt = (128 - n_pitches) // 2
    note_padding = np.zeros((tot_timesteps,  padding_amt), dtype=np.bool)
    track_np_flat = np.concatenate((note_padding, track_np_flat, note_padding), axis=1)
    track = Track(pianoroll=track_np_flat,
                  is_drum=is_drum, name=name)

    return track


def plot_track(track):
    track.plot()
    plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Visualize M2VAE',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_file', default='data/train_x_lpd_5_phr.npz',
                        help='Cleaned/processed LP5 dataset')
    parser.add_argument('--hidden_size', default=128, type=int,
                        help='Hidden size')
    parser.add_argument('--exp_dir', default='exp/debug/',
                        help='Cleaned/processed LP5 dataset')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda')
    parser.add_argument('--debug', action='store_true',
                        help='Load shortened version')

    args = parser.parse_args()

    if not os.path.exists(args.exp_dir):
        raise RuntimeError("Can't find {}".format(args.exp_Dir))

    exp_args = util.load_args(args.exp_dir)
    for arg, val in exp_args.items():
        if not arg in args:
            args.__setattr__(arg, val)

    # Seed
    random = np.random.RandomState(args.seed)

    if args.debug:
        lpd_raw = np.load(args.data_file.replace('.npz', '_debug.npz'))['arr_0']
    else:
        lpd_raw = data.load_data_from_npz(args.data_file)
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
        muvar_encoder = models.StdMuVarEncoder(track_encoder, input_size=args.hidden_size, hidden_size=args.hidden_size)
        return muvar_encoder

    def decoder_func():
        bar_decoder = models.RNNTrackDecoder(input_size=args.hidden_size)
        note_decoder = models.ConvBarDecoder(bar_decoder)
        return note_decoder

    # Model
    model = mvae.MVAE(encoder_func, decoder_func, n_tracks=args.n_tracks, hidden_size=args.hidden_size)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss = mvae.ELBOLoss(pos_weight = torch.tensor(1 / pos_prop))

    if args.cuda:
        model = model.cuda()
        loss = loss.cuda()

    ckpt = util.load_checkpoint(args.exp_dir,
                                cuda=args.cuda,
                                filename='model_best.pth')
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])

    model.eval()

    for i, tracks in enumerate(dataloaders['val']):
        tracks = tracks[:, :, :, :, :args.n_tracks]
        tracks_recon, mu, logvar = model(tracks)

        tracks_np = tracks.detach().cpu().numpy().astype(np.bool)
        tracks_recon_np = (tracks_recon.detach().cpu().numpy() > 0.0)

        print(f1_score(tracks_np.flatten(), tracks_recon_np.flatten()))

        for track, track_recon in zip(tracks_np, tracks_recon_np):
            track = to_track(track[:, :, :, 0])
            track_recon = to_track(track_recon[:, :, :, 0])

            plot_track(track)
            plot_track(track_recon)

            x = input('Continue?')
