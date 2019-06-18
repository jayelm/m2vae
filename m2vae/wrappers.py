"""
Utilities to build models and MVAEs from script arguments
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import models
import mvae
import data


def load_data(args, random_state=None, use_random_transpose=False):
    lpd_file = args.data_file
    if args.debug and 'debug' not in lpd_file:
        lpd_file = lpd_file.replace('.npz', '_debug.npz')

    if args.debug:
        lpd_raw = np.load(lpd_file)['arr_0']
    else:
        lpd_raw = data.load_data_from_npz(lpd_file)

    if args.hits_only:
        if lpd_raw.max() == 1:
            print("WARNING: data does not seem to contain durations")
        lpd_raw = lpd_raw > 0

    if args.max_train is not None:
        lpd_raw = lpd_raw[:args.max_train]

    # Calcualte mean positive weight
    pos_prop = lpd_raw.mean()
    lpds = data.train_val_test_split(lpd_raw, random_state=random_state,
                                     use_random_transpose=use_random_transpose)
    del lpd_raw
    dataloaders = {}
    for split, dataset in lpds.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers, shuffle=True,
            pin_memory=args.pin_memory
        )
    return dataloaders, pos_prop


def build_mvae(args, pos_prop=1):
    def encoder_func():
        bar_encoder = models.ConvBarEncoder()
        track_encoder = models.RNNTrackEncoder(bar_encoder, output_size=args.hidden_size)
        muvar_encoder = models.StdMuVarEncoder(track_encoder, input_size=args.hidden_size, hidden_size=args.hidden_size, output_size=args.hidden_size)
        return muvar_encoder

    def decoder_func():
        hidden_size = args.hidden_size
        if args.note_condition:
            hidden_size = hidden_size + 12  # Add dimensionality for semitones
        bar_decoder = models.RNNTrackDecoder(input_size=hidden_size)
        note_decoder = models.ConvBarDecoder(bar_decoder)
        return note_decoder

    # Model
    model = mvae.MVAE(encoder_func, decoder_func, n_tracks=args.n_tracks, hidden_size=args.hidden_size, note_condition=args.note_condition)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss = mvae.ELBOLoss(pos_weight = torch.tensor(1 / pos_prop),
                         kl_factor=args.kl_factor,
                         mse_factor=args.mse_factor)

    if args.cuda:
        model = model.cuda()
        loss = loss.cuda()

    return model, optimizer, loss
