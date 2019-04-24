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
                 n_tracks=5, hidden_size=256):
        super(MVAE, self).__init__()
        self.encoders = nn.ModuleList([encoder_func() for _ in range(n_tracks)])
        self.decoders = nn.ModuleList([decoder_func() for _ in range(n_tracks)])
        self.n_tracks = n_tracks
        self.hidden_size = hidden_size
        self.experts = ProductOfExperts()

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, tracks):
        """Forward pass through the MVAE.

        @param tracks: list of ?PyTorch.Tensors

        @return image_recon: PyTorch.Tensor
        @return attr_recons: list of PyTorch.Tensors (N_ATTRS length)
        """
        mu, logvar = self.encode(tracks)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        tracks_recon = self.decode(z)
        return tracks_recon, mu, logvar

    def encode(self, tracks):
        assert tracks.shape[-1] == self.n_tracks
        # Batch size
        batch_size = tracks.shape[0]

        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        mu, logvar = prior_expert((1, batch_size, self.hidden_size),
                                  use_cuda=use_cuda)

        for t in range(self.n_tracks):
            track = tracks[:, :, :, :, t]
            encoder = self.encoders[t]
            track_mu, track_logvar = encoder(track)

            mu = torch.cat((mu, track_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, track_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

    def decode(self, z):
        tracks = []
        for t in range(self.n_tracks):
            decoder = self.decoders[t]
            track = decoder(z)
            tracks.append(track)
        return torch.stack(tracks, dim=4)

def get_batch_size(tracks):
    """
    If tracks is a track-major list of possibly None tracks, get the batch size
    """
    for t in tracks:
        if t is not None:
            return t.shape[0]
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


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class ELBOLoss(nn.Module):
    """
    ELBO Loss for MVAE
    """
    def __init__(self):
        super(ELBOLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, recon, data, mu, logvar, annealing_factor=1.0):
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
        assert len(recon) == len(data), "must supply ground truth for every modality."
        n_modalities = len(recon)
        batch_size = mu.size(0)

        bce = 0  # reconstruction cost
        for ix in range(n_modalities):
            bce += self.bce_loss(recon[ix], data[ix])
        kl_divergence  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elbo = torch.mean(bce + annealing_factor * kl_divergence)
        return elbo
