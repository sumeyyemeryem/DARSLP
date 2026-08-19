"""
Stage 1 — Standard (Non-Disentangled) Pose Autoencoder
=======================================================
Encodes all 176 joints jointly into a single latent vector using a linear
encoder and decoder. Used as the non-disentangled ablation baseline, paired
with StandardGenerator (Stage 2).

Unlike DisentangledAE, this model does not partition articulators into separate
latent subspaces. It is useful for ablation studies that isolate the contribution
of structural disentanglement: keeping everything else equal, replacing
DisentangledAE + DARSLPGenerator with StandardAE + StandardGenerator measures
how much the disentangled representation matters.

Input:  pose sequence [B, F, 178, 3]  (wrist duplicates at indices 2 and 5
        are removed internally, resulting in 176 joints)
Output: single latent code z [B, F, latent_dim] per frame
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.utils.helpers import create_mask


class StandardAE(pl.LightningModule):
    def __init__(
        self,
        num_joints=178,
        num_feats=3,
        max_frame_len=475,
        latent_dim=64,
        l1_lambda=1e-4,
        base_learning_rate=2.0e-4,
        dropout_rate=0.2,
        loss_type="l1",
        scheduler_config=None,
        ckpt_path=None,
        ignore_keys=[],
        **kwargs,
    ):
        super().__init__()

        self.latent_dim    = latent_dim
        self.loss_type     = loss_type
        self.learning_rate = base_learning_rate
        self.num_joints    = num_joints
        self.num_feats     = num_feats
        self.max_frame_len = max_frame_len
        self.l1_lambda     = l1_lambda
        self.dropout_rate  = dropout_rate

        self.training_step_outputs = []
        self.valid_step_outputs    = []
        self.total_valid_losses    = []
        self.total_train_losses    = []

        # 176 joints after removing wrist duplicates (indices 2 and 5)
        in_dim = 176 * num_feats
        self.encoder = nn.Linear(in_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, in_dim)

    def _remove_wrist_duplicates(self, x):
        """Remove body-wrist duplicate joints at indices 2 and 5."""
        keep = [i for i in range(x.shape[2]) if i not in (2, 5)]
        return x[:, :, keep, :]  # [B, F, 176, 3]

    def forward(self, x, mask):
        x      = self._remove_wrist_duplicates(x)
        x_flat = rearrange(x, 'b f v c -> b f (v c)')
        z      = self.encoder(x_flat)
        recon  = rearrange(self.decoder(z), 'b f (v c) -> b f v c', v=176, c=self.num_feats)
        recon  = recon * mask.unsqueeze(-1).unsqueeze(-1)
        return recon

    def get_input(self, batch):
        keypoints     = batch["poses"]
        frame_lengths = batch["frame_lengths"]
        mask = create_mask(frame_lengths, self.max_frame_len, device=self.device)
        keypoints = self._remove_wrist_duplicates(keypoints.clone())
        return keypoints, mask

    def share_step(self, inputs, mask, split="train"):
        reconstructions = self(inputs, mask)
        recon_loss = (
            F.l1_loss(reconstructions, inputs, reduction='none') *
            mask.unsqueeze(-1).unsqueeze(-1)
        ).sum() / mask.sum()
        self.log(f"{split}/total_loss", recon_loss.detach(),
                 prog_bar=True, sync_dist=True, batch_size=inputs.shape[0])
        return recon_loss

    def training_step(self, batch, batch_idx):
        inputs, mask = self.get_input(batch)
        total_loss   = self.share_step(inputs, mask, split="train")
        l1_reg = sum(p.abs().sum() for p in self.encoder.parameters())
        total_loss = total_loss + self.l1_lambda * l1_reg
        self.training_step_outputs.append(total_loss)
        return total_loss

    def on_train_epoch_end(self):
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        self.total_train_losses.append(epoch_loss.item())
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, mask = self.get_input(batch)
        total_loss   = self.share_step(inputs, mask, split="valid")
        self.valid_step_outputs.append(total_loss)

    def on_validation_epoch_end(self):
        epoch_loss = torch.stack(self.valid_step_outputs).mean()
        self.total_valid_losses.append(epoch_loss.item())
        self.valid_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
