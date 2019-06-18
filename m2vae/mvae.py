"""
Multitmodal variational autoencoder, borrowed from
https://github.com/mhw32/multimodal-vae-public
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self,
                 encoder_func,
                 decoder_func,
                 n_tracks=5, hidden_size=256,
                 durations=False, note_condition=False):
        super(MVAE, self).__init__()
        self.encoders = nn.ModuleList([encoder_func() for _ in range(n_tracks)])
        self.decoders = nn.ModuleList([decoder_func() for _ in range(n_tracks)])
        self.n_tracks = n_tracks
        self.hidden_size = hidden_size
        self.durations = durations
        self.note_condition = note_condition
        self.experts = ProductOfExperts()

    def reparametrize(self, mu, logvar, override_training=False):
        if self.training or override_training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, tracks, notes, return_z=False):
        """Forward pass through the MVAE.

        @param tracks: list of ?PyTorch.Tensors

        @return image_recon: PyTorch.Tensor
        @return attr_recons: list of PyTorch.Tensors (N_ATTRS length)
        """
        mu, logvar = self.encode(tracks)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        if self.note_condition:
            z = torch.cat((z, notes), 1)
        tracks_recon = self.decode(z, targets=[t is not None for t in tracks])
        if return_z:
            return tracks_recon, z
        else:
            return tracks_recon, mu, logvar

    def encode(self, tracks):
        assert len(tracks) == self.n_tracks
        # Batch size
        batch_size = get_batch_size(tracks)

        cuda = next(self.parameters()).is_cuda  # check if CUDA

        # Row 0 is prior expert: N(0, 1).
        present_tracks = [(t, track) for t, track in enumerate(tracks) if track is not None]
        all_mu = torch.zeros((1 + len(present_tracks), batch_size, self.hidden_size))
        all_logvar = torch.zeros((1 + len(present_tracks), batch_size, self.hidden_size))
        if cuda:
            all_mu = all_mu.cuda()
            all_logvar = all_logvar.cuda()

        for i, (t, track) in enumerate(present_tracks):
            encoder = self.encoders[t]
            track_mu, track_logvar = encoder(track)

            all_mu[i + 1] = track_mu
            all_logvar[i + 1] = track_logvar

        # product of experts to combine gaussians
        mu, logvar = self.experts(all_mu, all_logvar)
        return mu, logvar

    def decode(self, z, targets=None):
        if targets is None:
            targets = [True for _ in range(self.n_tracks)]
        batch_size = z.shape[0]
        decoded = []
        for t, decode_this in zip(range(self.n_tracks), targets):
            if decode_this:
                decoder = self.decoders[t]
                track = decoder(z)
                decoded.append(track)
            else:
                decoded.append(None)
        return decoded


def get_batch_size(tracks):
    """
    If tracks is a track-major list of possibly None tracks, get the batch size
    """
    return get_shape(tracks)[0]


def get_shape(tracks):
    """
    If tracks is a track-major list of possibly None tracks, get the batch size
    """
    for t in tracks:
        if t is not None:
            return t.shape
    raise ValueError("Cannot pass all None")


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class ELBOLoss(nn.Module):
    """
    ELBO Loss for MVAE
    """
    def __init__(self, pos_weight=None, kl_factor=0.01, mse_factor=0.00):
        super(ELBOLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.mse_factor = mse_factor
        if self.mse_factor > 0:
            self.mse_loss = nn.MSELoss()
        self.kl_factor = kl_factor

    def forward(self, all_recon, all_data, mu, logvar, annealing_factor=1.0):
        """Compute the ELBO for an arbitrary number of data modalities.
        @param recon: list of torch.Tensors/Variables
                      Contains one for each modality.
        @param data: list of torch.Tensors/Variables
                     Size much agree with recon.
        @param mu: Torch.Tensor
                   Mean of the variational distribution.
        @param logvar: Torch.Tensor
                       Log variance for variational distribution.
        @param annealing_factor: float [default: 1]
                                 Beta - how much to weight the KL regularizer.
        """
        assert len(all_recon) == len(all_data)
        batch_size, n_bars = get_shape(all_recon)[:2]

        # ==== RECONSTRUCTION LOSS ====
        all_bce = 0
        all_mse = 0
        n_recons = 0
        for recon, data in zip(all_recon, all_data):
            if recon is None or data is None:
                assert recon is None and data is None
                continue
            n_recons += 1
            recon_2d = recon.view(batch_size, -1)
            data_2d = data.view(batch_size, -1)

            # Binary cross entropy
            all_bce += self.bce_loss(recon_2d, data_2d)
            # Mean squared error
            if self.mse_factor > 0:
                recon_bar_major = recon.view(batch_size * n_bars, -1)
                data_bar_major = data.view(batch_size * n_bars, -1)

                recon_bar_major_values = torch.sigmoid(recon_bar_major)
                all_mse += self.mse_loss(recon_bar_major_values, data_bar_major)
        recon_loss = all_bce + (self.mse_factor * all_mse)
        recon_loss /= n_recons
        # ==== KL DIVERGENCE ====
        kl_divergence  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_divergence = kl_divergence.mean()
        elbo = recon_loss + self.kl_factor * annealing_factor * kl_divergence
        return elbo, recon_loss, kl_divergence
