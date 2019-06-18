"""
Borrowed from https://github.com/salu133445/musegan/blob/master/src/musegan/data.py
"""

from torch.utils.data import Dataset

import numpy as np
from sklearn.model_selection import train_test_split


def load_data_from_npz(filename):
    """Load and return the training data from a npz file (sparse format)."""
    with np.load(filename) as f:
        data = np.zeros(f['shape'], np.uint8)
        data[[x for x in f['nonzero']]] = 1
    return data


def random_transpose(pianoroll):
    """
    Randomly transpose a pianoroll with [-5, 6] semitones.
    Input:
    pianoroll: torch.Tensor of shape (n_bars, n_timesteps, n_pitches, n_tracks)
    """
    semitone = np.random.randint(-5, 6)
    if semitone > 0:
        # 1: skips drums
        pianoroll[:, :, semitone:, 1:] = pianoroll[:, :, :-semitone, 1:]
        pianoroll[:, :, :semitone, 1:] = 0
    elif semitone < 0:
        pianoroll[:, :, :semitone, 1:] = pianoroll[:, :, -semitone:, 1:]
        pianoroll[:, :, semitone:, 1:] = 0
    return pianoroll


class LPD(Dataset):
    """
    LPD dataset
    """
    def __init__(self, data_file=None, data=None, use_random_transpose=True, max_size=None):
        if data_file is None and data is None:
            raise ValueError("Must supply one of data_file or data")
        if data_file is not None and data is not None:
            raise ValueError("Can't supply data_file and data")

        if data is not None:
            self.data = data
        else:
            if data_file.endswith('debug'):
                self.data = np.load(data_file)['arr_0']
            else:
                self.data = load_data_from_npz(data_file)
        if max_size is not None:
            self.data = self.data[:max_size]
        self.use_random_transpose = use_random_transpose

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        p = self.data[i]
        # Convert to float32
        p = p.astype(np.float32)
        # Apply random transpose
        if self.use_random_transpose:
            p = random_transpose(p)
        notes = extract_note_distribution(p)
        return p, notes


def extract_note_distribution(p):
    """
    Input: pianoroll of shape (n_bars, n_timesteps, n_pitches, n_tracks)
    """
    p = p[:, :, :, 1:]  # Remove drum track
    n_bars, n_timesteps, n_pitches, n_tracks = p.shape
    p_flat = p.reshape((n_bars * n_timesteps, n_pitches, n_tracks))
    # Concat along tracks
    p_combined = p_flat.transpose(2, 0, 1).reshape((n_tracks * n_bars * n_timesteps, n_pitches))
    notes = np.zeros(12, dtype=np.float32)
    for pitch in range(12):
        notes[pitch] = p_combined[:, pitch::12].sum()
    if notes.sum() == 0:  # Empty track?
        return notes
    return notes / notes.sum()


def train_val_test_split(data, val_size=0.1, test_size=0.1, random_state=None,
                         **kwargs):
    idx = np.arange(data.shape[0])
    idx_train, idx_valtest = train_test_split(idx, test_size=val_size + test_size, random_state=random_state, shuffle=True)
    idx_val, idx_test = train_test_split(idx_valtest, test_size=test_size / (val_size + test_size), random_state=random_state, shuffle=True)
    return {
        'train': LPD(data=data[idx_train], **kwargs),
        'val': LPD(data=data[idx_val], **kwargs),
        'test': LPD(data=data[idx_test], **kwargs)
    }
