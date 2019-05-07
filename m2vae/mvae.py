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

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA

        # Row 0 is prior expert: N(0, 1).
        all_mu = torch.zeros((1 + self.n_tracks, batch_size, self.hidden_size))
        all_logvar = torch.zeros((1 + self.n_tracks, batch_size, self.hidden_size))
        if use_cuda:
            all_mu = all_mu.cuda()
            all_logvar = all_logvar.cuda()

        for t in range(self.n_tracks):
            track = tracks[:, :, :, :, t]
            encoder = self.encoders[t]
            track_mu, track_logvar = encoder(track)

            all_mu[t + 1] = track_mu
            all_logvar[t + 1] = track_logvar

        # product of experts to combine gaussians
        mu, logvar = self.experts(all_mu, all_logvar)
        return mu, logvar

    def decode(self, z):
        batch_size = z.shape[0]
        decoded = torch.zeros((batch_size, 4, 48, 84, self.n_tracks),
                              dtype=torch.float32).to(z.device)
        tracks = []
        for t in range(self.n_tracks):
            decoder = self.decoders[t]
            track = decoder(z)
            decoded[:, :, :, :, t] = track
        return decoded

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


class ELBOLoss(nn.Module):
    """
    ELBO Loss for MVAE
    """
    def __init__(self, pos_weight=None, kl_factor=0.01):
        super(ELBOLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.kl_factor = kl_factor

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
        assert recon.shape[4] == data.shape[4], "must supply ground truth for every modality."

        bce = self.bce_loss(recon.view(-1), data.view(-1))
        kl_divergence  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_divergence = kl_divergence.mean()
        elbo = bce + self.kl_factor * annealing_factor * kl_divergence
        return elbo, bce, kl_divergence
