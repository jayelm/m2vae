"""
Visualize a trained m2vae model.
"""

import os
import sys
from collections import defaultdict

import torch

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from scipy.interpolate import interp1d

from pypianoroll import Track, plot_pianoroll
import matplotlib.pyplot as plt

import data
import mvae
import models
import util
import io_util
import wrappers


def interpolate(mu1, mu2, steps=3, method='linear'):
    all_steps = list(range(1, steps + 1))
    linfit = interp1d([1, steps + 1], np.vstack([mu1, mu2]), axis=0)
    return linfit(all_steps)


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
    args = io_util.parse_args('vis', desc=__doc__)
    util.restore_args(args, args.exp_dir)

    # Seed
    random = np.random.RandomState(args.seed)

    dataloaders, pos_prop = wrappers.load_data(args, random_state=random)
    model, optimizer, loss = wrappers.build_mvae(args, pos_prop=pos_prop)

    util.restore_checkpoint(model, optimizer, args.exp_dir,
                            cuda=args.cuda,
                            filename='model_best.pth')

    model.eval()

    for i, tracks in enumerate(dataloaders['val']):
        tracks = tracks[:, :, :, :, :args.n_tracks]
        tracks_recon, z = model(tracks, return_z=True)

        tracks_np = tracks.detach().cpu().numpy().astype(np.bool)
        tracks_recon_np = (tracks_recon.detach().cpu().numpy() > 0.0)

        print(f1_score(tracks_np.flatten(), tracks_recon_np.flatten()))

        #  # Interpolate
        #  z1 = z[0].detach().cpu().numpy()
        #  z2 = z[1].detach().cpu().numpy()
        #  z_interp = interpolate(z1, z2, 5)

        #  f, axarr = plt.subplots(ncols=5, figsize=(20, 4))

        #  for i in range(5):
            #  if i == 0 or i == 5:
                #  # Just plot the track
                #  track = to_track(tracks[0 if i == 0 else 1, :, :, :, 0])
                #  plot_pianoroll(axarr[i], track.pianoroll)
            #  else:
                #  zi = z_interp[i]
                #  zit = torch.tensor(zi).to(z.device).float().unsqueeze(0)
                #  track_recon = model.decode(zit).detach().cpu().numpy()
                #  track_recon = to_track(track_recon[0, :, :, :, 0])
                #  plot_pianoroll(axarr[i], track_recon.pianoroll)

        #  plt.show()
        #  raise Exception

        for track, track_recon in zip(tracks_np, tracks_recon_np):
            track = to_track(track[:, :, :, 0])
            track_recon = to_track(track_recon[:, :, :, 0])

            f, axarr = plt.subplots(ncols=2, figsize=(20, 4))

            axarr[0].set_title('Original')
            axarr[1].set_title('Reconstructed')

            plot_pianoroll(axarr[0], track.pianoroll)
            plot_pianoroll(axarr[1], track_recon.pianoroll)

            plt.show()
            x = input('Continue?')
