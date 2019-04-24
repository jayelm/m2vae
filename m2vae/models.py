"""
Music encoder/decoders
"""

import torch.nn as nn
import torch


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class MultiTrackEncoder(nn.Module):
    """
    Wraps around a single track encoder to produce a
    multi track encoder.
    """
    def __init__(self, encoder, n_tracks=5):
        super(MultiTrackEncoder, self).__init__()
        self.encoders = nn.ModuleList([encoder() for _ in range(n_tracks)])
        self.n_tracks = n_tracks

    def forward(self, x):
        # Loop through tracks (final dim)
        assert x.shape[-1] == self.n_tracks
        x_enc = []
        for t in range(self.n_tracks):
            encoder = self.encoders[t]
            xt = x[:, :, :, :, t]
            xt_enc = encoder(xt)
            x_enc.append(xt_enc)
        import ipdb; ipdb.set_trace()
        x_enc = torch.stack(x_enc, dim=3)


class TrackEncoder(nn.Module):
    def __init__(self, bar_encoder):
        super(TrackEncoder, self).__init__()
        self.bar_encoder = bar_encoder

    def forward(self, x):
        x_bar_enc = self.bar_encoder(x)
        x_track_enc = self.track_encode(x_bar_enc)
        return x_track_enc

    def track_encode(self, x):
        """
        Inputs:
            x: [batch_size, n_bar, bar_hidden_size]

        Ouputs:
            x_enc: [batch_size, output_size]
        """
        raise NotImplementedError


class ConcatTrackEncoder(TrackEncoder):
    """
    A track encoder which just concatenates the bars.
    """
    def track_encode(self, x):
        """
        Inputs:
            x: [batch_size, n_bar, bar_hidden_size]

        Ouputs:
            x_enc: [batch_size, n_bar * bar_hidden_size]
        """
        return x.view(x.shape[0], -1)


class RNNTrackEncoder(TrackEncoder):
    """
    Use a bidirectional RNN to concatenate the bars
    """
    def __init__(self, encoder, n_bars=4, bar_hidden_size=64, output_size=256,
                 rnn_type='lstm'):
        super(RNNTrackEncoder, self).__init__(encoder)
        assert output_size % 2 == 0
        self.hidden_size = output_size // 2
        self.output_size = output_size
        self.rnn_type = rnn_type
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(bar_hidden_size, self.hidden_size, bidirectional=True,
                               batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(bar_hidden_size, self.hidden_size, bidirectional=True,
                              batch_first=True)
        else:
            raise NotImplementedError('rnn_type = {}'.format(self.rnn_type))


    def track_encode(self, x):
        """
        Inputs:
            x: [batch_size, n_bar, bar_hidden_size]
        Outputs:
        x_enc: [batch_size, output_size]
        """
        x_enc, _ = self.rnn(x)
        return x_enc[:, -1, :]


class FCTrackEncoder(TrackEncoder):
    """
    A track encoder that puts concatenated bars into
    FC + Swish + FC
    """
    def __init__(self, encoder, n_bars=4, bar_hidden_size=64, output_size=256):
        super(FCTrackEncoder, self).__init__(encoder)
        self.output_size = output_size
        self.trunk = nn.Sequential(
            nn.Linear(bar_hidden_size * n_bars, output_size),
            Swish(),
            nn.Linear(output_size, output_size)
        )

    def track_encode(self, x):
        """
        Inputs:
            x: [batch_size, n_bar, bar_hidden_size]

        Ouputs:
            x_enc: [batch_size, output_size]
        """
        x = x.view(x.shape[0], -1)
        return self.trunk(x)


class BarEncoder(nn.Module):
    pass


class ConvBarEncoder(BarEncoder):
    """
    A single bar encoder.
    """
    def __init__(self):
        super(ConvBarEncoder, self).__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 12), stride=(1, 12)),
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=(1, 7), stride=(1, 7)),
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            Swish(),
            nn.Conv2d(16, 16, kernel_size=(2, 1), stride=(2, 1)),
        )


    def forward(self, x):
        # Flaten out bars
        batch_size = x.shape[0]
        n_bars = x.shape[1]
        rest = x.shape[2:]
        # (batch_size * n_bars, 1 [channel],
        # n_timesteps, n_pitches)
        x = x.view(batch_size * n_bars, *rest).unsqueeze(1)
        x_enc = self.trunk(x)
        x_enc = x_enc.view(batch_size, n_bars, -1)
        return x_enc


class MuVarEncoder(nn.Module):
    def __init__(self, encoder):
        super(MuVarEncoder, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        x_enc = self.encoder(x)
        return self.parameterize(x_enc)


    def parameterize(self, x):
        raise NotImplementedError


class StdMuVarEncoder(MuVarEncoder):
    """
    Wrap around an encoder with FC + Swish + Dropout
    + FC to split an embedding into mu/logvar
    vectors
    """
    def __init__(self, encoder, input_size=256, hidden_size=256, output_size=256, dropout=0.1):
        super(StdMuVarEncoder, self).__init__(encoder)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            Swish(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size * 2)
        )

    def parameterize(self, x):
        x_enc = self.trunk(x)
        mu = x_enc[:, :self.output_size]
        logvar = x_enc[:, self.output_size:]
        return mu, logvar


class BarDecoder(nn.Module):
    pass


class RNNBarDecoder(BarDecoder):
    """
    A single bar decoder.
    """
    def __init__(self, input_size=256, output_size=256, n_bars=4, rnn_type='lstm'):
        super(RNNBarDecoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_bars = n_bars
        self.rnn_type = rnn_type

        self.init_h = nn.Linear(self.input_size, self.output_size)
        if self.rnn_type == 'lstm':
            self.rnn_cell = nn.LSTMCell(self.output_size * 2, self.output_size)
            self.init_c = nn.Linear(self.input_size, self.output_size)
        elif self.rnn_type == 'gru':
            self.rnn_cell = nn.GRUCell(self.input_size, self.output_size)
        else:
            raise NotImplementedError('rnn_type = {}'.format(self.rnn_type))

    def forward(self, x):
        """
        Generate inputs autoregressively as in
        https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/lstm_models.py#L1059-L1067
        """
        batch_size = x.shape[0]
        if self.rnn_type == 'lstm':
            hidden = (self.init_h(x), self.init_c(x))
        elif self.rnn_type == 'gru':
            hidden = self.init_h(x)

        x_enc = []
        for t in range(self.n_bars):
            if self.rnn_type == 'lstm':
                rnn_input = torch.cat((hidden[0], x), dim=1)
            elif self.rnn_type == 'gru':
                rnn_input = torch.cat((hidden, x), dim=1)

            hidden = self.rnn_cell(rnn_input, hidden)

            if self.rnn_type == 'lstm':
                x_enc.append(hidden[0])
            elif self.rnn_type == 'gru':
                x_enc.append(hidden)

        return torch.stack(x_enc, dim=1)


class NoteDecoder(nn.Module):
    def __init__(self, bar_decoder):
        super(NoteDecoder, self).__init__()
        self.bar_decoder = bar_decoder

    def forward(self, x):
        x_bar = self.bar_decoder(x)  # (batch_size, n_bars, bar_hidden_size)
        return self.note_decode(x_bar)

    def note_decode(self, x_bar):
        raise NotImplementedError


class ConvNoteDecoder(NoteDecoder):
    """
    A single note decoder.
    """
    def __init__(self, encoder):
        super(ConvNoteDecoder, self).__init__(encoder)

        self.trunk = nn.Sequential(
            nn.ConvTranspose2d(256, 1024, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(1024),
            Swish(),
            nn.ConvTranspose2d(1024, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            Swish(),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            Swish(),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            Swish(),
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 7), stride=(1, 7)),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 1, kernel_size=(1, 12), stride=(1, 12)),
            nn.Sigmoid()
        )


    def note_decode(self, x_bar):
        """
        Args:
            x_bar: tensor of shape [batch_size, n_bar, bar_hidden_dim]

        Returns:
            x_enc: tensor of shape [batch_size, n_bar, n_timesteps, n_pitches]
        """
        # Flaten out bars
        batch_size = x_bar.shape[0]
        n_bars = x_bar.shape[1]
        rest = x_bar.shape[2:]
        x_bar = x_bar.view(batch_size * n_bars, *rest).unsqueeze(2).unsqueeze(3)
        x_enc = self.trunk(x_bar)
        x_enc = x_enc.view(batch_size, n_bars, *x_enc.shape[2:])
        return x_enc
