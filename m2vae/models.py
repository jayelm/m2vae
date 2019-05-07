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


ACTIVATIONS = {
    'swish': Swish,
    'lrelu': nn.LeakyReLU,
    'relu': nn.ReLU
}


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
    def __init__(self, encoder, n_bars=4, bar_hidden_size=128, output_size=256,
                 rnn_type='gru'):
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
    def __init__(self, encoder, n_bars=4, bar_hidden_size=128, output_size=128, activation='relu'):
        super(FCTrackEncoder, self).__init__(encoder)
        self.output_size = output_size
        self.trunk = nn.Sequential(
            nn.Linear(bar_hidden_size * n_bars, output_size),
            ACTIVATIONS[activation](),
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


class ConvBarEncoder(nn.Module):
    """
    A single bar encoder.
    """
    def __init__(self, activation='relu'):
        super(ConvBarEncoder, self).__init__()

        act = ACTIVATIONS[activation]

        self.trunk = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 12), stride=(1, 12)),
            nn.BatchNorm2d(64),
            act(),
            nn.Conv2d(64, 128, kernel_size=(1, 7), stride=(1, 7)),
            nn.BatchNorm2d(128),
            act(),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2d(256),
            act(),
            nn.Conv2d(256, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            act(),
            nn.Conv2d(256, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            act(),
            nn.Conv2d(256, 1024, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(1024),
            act(),
            nn.Conv2d(1024, 128, kernel_size=(2, 1), stride=(2, 1)),
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
    Wrap around an encoder with FC + Activation + Dropout
    + FC to split an embedding into mu/logvar
    vectors
    """
    def __init__(self, encoder, input_size=128, hidden_size=128, output_size=128, dropout=0.1, activation='relu'):
        super(StdMuVarEncoder, self).__init__(encoder)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            ACTIVATIONS[activation](),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size * 2)
        )

    def parameterize(self, x):
        x_enc = self.trunk(x)
        mu = x_enc[:, :self.output_size]
        logvar = x_enc[:, self.output_size:]
        return mu, logvar


class TrackDecoder(nn.Module):
    pass


class RNNTrackDecoder(TrackDecoder):
    """
    A single bar decoder.
    """
    def __init__(self, input_size=256, output_size=128, n_bars=4, rnn_type='gru'):
        super(RNNTrackDecoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_bars = n_bars
        self.rnn_type = rnn_type

        self.init_h = nn.Linear(self.input_size, self.output_size)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.output_size, batch_first=True)
            self.init_c = nn.Linear(self.input_size, self.output_size)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.input_size, self.output_size, batch_first=True)
        else:
            raise NotImplementedError('rnn_type = {}'.format(self.rnn_type))

    def forward(self, x):
        """
        Generate inputs autoregressively as in
        https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/lstm_models.py#L1059-L1067
        """
        batch_size = x.shape[0]
        if self.rnn_type == 'lstm':
            hidden = (self.init_h(x).unsqueeze(0), self.init_c(x).unsqueeze(0))
        elif self.rnn_type == 'gru':
            hidden = self.init_h(x).unsqueeze(0)

        # Just give it the embedding over and over again as input
        rnn_input = x.unsqueeze(1).expand(batch_size, self.n_bars, -1)
        outputs, _ = self.rnn(rnn_input, hidden)

        return outputs.contiguous()


class BarDecoder(nn.Module):
    def __init__(self, bar_decoder):
        super(BarDecoder, self).__init__()
        self.bar_decoder = bar_decoder

    def forward(self, x):
        x_bar = self.bar_decoder(x)  # (batch_size, n_bars, bar_hidden_size)
        return self.note_decode(x_bar)

    def note_decode(self, x_bar):
        raise NotImplementedError


class ConvBarDecoder(BarDecoder):
    """
    A single bar decoder.
    """
    def __init__(self, encoder, activation='relu'):
        super(ConvBarDecoder, self).__init__(encoder)

        act = ACTIVATIONS[activation]

        self.trunk = nn.Sequential(
            nn.ConvTranspose2d(128, 1024, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(1024),
            act(),
            nn.ConvTranspose2d(1024, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            act(),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            act(),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            act(),
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=(3, 1)),
            nn.BatchNorm2d(128),
            act(),
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 7), stride=(1, 7)),
            nn.BatchNorm2d(64),
            act(),
            nn.ConvTranspose2d(64, 1, kernel_size=(1, 12), stride=(1, 12)),
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
