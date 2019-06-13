"""
Functions and utilities for visualizaing a trained m2vae model.
"""

import os
import sys
from collections import defaultdict

import torch

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from scipy.interpolate import interp1d

import pypianoroll as ppr
import matplotlib.pyplot as plt
import pretty_midi

from IPython.display import HTML, Audio, display

import data
import mvae
import models
import util
import io_util
import wrappers


DEFAULT_RATE = 22050
DOWNBEATS = np.array([0, 48, 96, 144])
DOWNBEATS_ONEHOT = np.zeros(192, dtype=np.bool_)
DOWNBEATS_ONEHOT[DOWNBEATS] = 1


N2MIDI = {
    0: 0,  # acoustic grand piano; and set is_drum = True
    1: 0,  # acoustic grand piano
    2: 24,  # acoustic guitar (nylon)
    3: 32,  # acoustic bass
    4: 48,  # string ensemble 1 (nylon)
}
N2NAME = {
    0: 'drum',
    1: 'piano',
    2: 'guitar',
    3: 'bass',
    4: 'strings'
}


def interpolate(mu1, mu2, steps=3, method='linear'):
    all_steps = list(range(1, steps + 1))
    linfit = interp1d([1, steps + 1], np.vstack([mu1, mu2]), axis=0)
    return linfit(all_steps)


def to_track(track_np, n=None, is_drum=False, name='piano', program=0):
    """
    If n is given, use the values in N2MIDI/N2NAME and ignore name/program
    """
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

    if n is not None:
        program = N2MIDI[n]
        name = N2NAME[n]
        is_drum = n == 0
    track = ppr.Track(pianoroll=track_np_flat,
                      is_drum=is_drum, name=name, program=program)

    return track


def track_is_empty(track):
    return (track.sum() == 0)


def to_multitrack(mt, n=None):
    """
    Create a multitrack output out of a model tensor
    Input is [n_bars, n_timesteps, n_pitches, n_tracks] tensor.
    If n is given, it's a list of length n_tracks, detailing the LPD-5 number
    for each track.
    TODO: Support custom programs/names just like to_track.
    """
    n_tracks = len(mt)
    if n is not None and len(n) != n_tracks:
        raise ValueError("Must supply n == n_tracks")
    tracks = []
    for i in range(n_tracks):
        if n is None:
            this_n = i
        else:
            this_n = n[i]
        tracks.append(to_track(mt[i], n=i))
    return ppr.Multitrack(tracks=tracks, beat_resolution=12,
                          downbeat=DOWNBEATS_ONEHOT)


def synthesize(mt, rate=DEFAULT_RATE):
    midi = mt.to_pretty_midi()
    audio = midi.fluidsynth(fs=rate)
    return audio


def plot_track(track):
    track.plot()
    plt.show()


def ipy_display(audio, title=None, rate=DEFAULT_RATE):
    if title is not None:
        display(HTML('<h2>{}</h2>'.format(title)))
    display(Audio(audio, rate=rate))
