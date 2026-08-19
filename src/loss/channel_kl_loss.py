"""
Channel-Aware KL Regularization — DARSLP
==========================================
Implements the channel-prior KL divergence loss used in DARSLP Phase 2
training. Each latent channel is treated as an independent Gaussian, and its
prior distribution (mean, std) is precomputed from ground-truth pose encodings
produced by DisentangledAE on the training set.

During training, the predicted latent distribution is aligned with these
per-channel priors, reducing regression to the mean and improving
motion diversity without requiring gloss supervision.

Workflow:
  1. Run compute_channel_priors.py on the training set to extract per-channel
     statistics → saves a .npy/.npz file with 'Mean' and 'StdDev' arrays.
  2. Load priors at training time via load_channel_priors().
  3. Pass to channel_kl_divergence() inside the training step.

Reference: Taşyürek et al., "Disentangle and Regularize: Sign Language Production
with Articulator-Based Disentanglement and Channel-Aware Regularization," WACV 2026.
"""

import numpy as np
import torch
import torch.nn as nn


class ChannelKLLoss(nn.Module):
    """KL divergence between a predicted Gaussian and a precomputed channel prior."""

    def __init__(self, prior_mu, prior_logvar):
        super().__init__()
        self.register_buffer('prior_mu', prior_mu)
        self.register_buffer('prior_logvar', prior_logvar)

    def forward(self, mu, logvar):
        kl_div = 0.5 * torch.sum(
            torch.exp(logvar - self.prior_logvar) +
            (mu - self.prior_mu) ** 2 / torch.exp(self.prior_logvar) -
            1 - logvar + self.prior_logvar,
            dim=1,
        )
        return torch.mean(kl_div)


def load_channel_priors(npy_file, latent_dim):
    """
    Load per-channel prior (mean, logvar) from a .npy or .npz stats file.

    The file must contain 'Mean' and 'StdDev' arrays of length latent_dim,
    produced by compute_channel_priors.py.
    """
    if npy_file.endswith(".npz"):
        f = np.load(npy_file)
        mean_array = f["Mean"]
        std_array  = f["StdDev"]
    else:
        stats = np.load(npy_file, allow_pickle=True).item()
        if "Mean" not in stats or "StdDev" not in stats:
            raise KeyError("Prior file must contain 'Mean' and 'StdDev' keys.")
        mean_array = stats["Mean"]
        std_array  = stats["StdDev"]

    if len(mean_array) != latent_dim or len(std_array) != latent_dim:
        raise ValueError(
            f"Prior dimension {len(mean_array)} does not match latent_dim={latent_dim}."
        )

    prior_mu     = torch.tensor(mean_array, dtype=torch.float32)
    prior_logvar = torch.log(torch.tensor(std_array, dtype=torch.float32).clamp(min=1e-8) ** 2)
    return prior_mu, prior_logvar


def channel_kl_divergence(mu, logvar, prior_mu, prior_logvar):
    """Scalar KL divergence between N(mu, exp(logvar)) and the precomputed prior."""
    kl_div = 0.5 * torch.sum(
        torch.exp(logvar - prior_logvar) +
        (mu - prior_mu) ** 2 / torch.exp(prior_logvar) -
        1 - logvar + prior_logvar
    )
    return torch.mean(kl_div)
