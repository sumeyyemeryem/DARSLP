"""
Stage 1 — Structurally Disentangled Pose Autoencoder (DARSLP)
==============================================================
Encodes each pose frame into four separate latent subspaces — upper body,
right hand, left hand, and face — rather than a single holistic vector.
This structural disentanglement introduces articulator-aware inductive biases
and allows region-specific training emphasis (hand regions are weighted more
heavily during reconstruction to counter their suppression by dominant body motion).

Input:  pose sequence [B, F, 176, 3]  (178 keypoints minus the two wrist
        duplicates at indices 2 and 5, which are later restored at inference)
Output: four latent codes per frame (z_upper, z_right, z_left, z_face)
        whose concatenation forms the regression target for DARSLPGenerator.

The trained model is saved as a .pth file and loaded frozen during Stage 2
training. Latent dimensions are allocated proportionally to joint count:
  upper body  → 6  joints  → latent_upper  channels
  right hand  → 21 joints  → latent_right  channels
  left  hand  → 21 joints  → latent_left   channels
  face        → 128 joints → face_latent_dim channels (configurable separately)

Reference: Taşyürek et al., "Disentangle and Regularize: Sign Language Production
with Articulator-Based Disentanglement and Channel-Aware Regularization," WACV 2026.

Two hand/face encoder-decoder architectures are supported via
--hand_face_arch, matching how the two shipped checkpoints were actually
trained (the underlying research code changed between the two runs):
  mlp    (default) — 2-layer MLP (Linear -> PReLU -> Linear) per region.
         Matches models/ae_csl_disentangled.pth.
  linear — single nn.Linear per region, no hidden layer.
         Matches models/ae_phoenix_disentangled.pth.
Loading a checkpoint with the wrong --hand_face_arch will fail with a
state_dict shape/key mismatch.
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


class DisentangledAE(pl.LightningModule):
    def __init__(
        self,
        num_joints=176,
        num_feats=3,
        max_frame_len=475,
        latent_dim=64,
        face_latent_dim=16,
        l1_lambda=1e-4,
        base_learning_rate=2.0e-4,
        dropout_rate=0.1,
        loss_type="l1",
        hand_face_arch="mlp",
        scheduler_config=None,
        ckpt_path=None,
        ignore_keys=[],
        **kwargs,
    ):
        super().__init__()

        self.latent_dim      = latent_dim
        self.face_latent_dim = face_latent_dim
        self.loss_type       = loss_type
        self.learning_rate   = base_learning_rate
        self.num_joints      = num_joints
        self.num_feats       = num_feats
        self.max_frame_len   = max_frame_len
        self.l1_lambda       = l1_lambda
        self.dropout_rate    = dropout_rate
        self.hand_face_arch  = hand_face_arch

        self.training_step_outputs = []
        self.valid_step_outputs    = []
        self.total_valid_losses    = []
        self.total_train_losses    = []

        self.training_step_recon_manuel    = []
        self.valid_step_recon_manuel       = []
        self.recon_valid_losses_manuel     = []
        self.recon_train_losses_manuel     = []
        self.training_step_recon_nonmanuel = []
        self.valid_step_recon_nonmanuel    = []
        self.recon_valid_losses_nonmanuel  = []
        self.recon_train_losses_nonmanuel  = []

        def mlp(in_dim, mid_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, mid_dim), nn.PReLU(), nn.Linear(mid_dim, out_dim))

        if hand_face_arch == "mlp":
            def hand_face_block(in_dim, mid_dim, out_dim):
                return mlp(in_dim, mid_dim, out_dim)
        elif hand_face_arch == "linear":
            def hand_face_block(in_dim, mid_dim, out_dim):
                return nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown hand_face_arch '{hand_face_arch}'. Choose: mlp, linear")

        # Latent split proportional to joint count (num_joints excludes face)
        body_joints  = num_joints - 128  # upper + hands = 48
        latent_upper = round(latent_dim * (6  / body_joints))
        latent_right = round(latent_dim * (21 / body_joints))
        latent_left  = round(latent_dim * (21 / body_joints))

        self.encoder_upper_body = nn.Linear(6   * num_feats, latent_upper)
        self.encoder_right_hand = hand_face_block(21  * num_feats, 40, latent_right)
        self.encoder_left_hand  = hand_face_block(21  * num_feats, 40, latent_left)
        self.encoder_face       = hand_face_block(128 * num_feats, 96, face_latent_dim)

        self.decoder_upper_body = nn.Linear(latent_upper,    6   * num_feats)
        self.decoder_right_hand = hand_face_block(latent_right,  40, 21  * num_feats)
        self.decoder_left_hand  = hand_face_block(latent_left,   40, 21  * num_feats)
        self.decoder_face       = hand_face_block(face_latent_dim, 96, 128 * num_feats)

    def encode(self, x):
        x_upper = x[:, :, [0, 1, 3, 4, 6, 7], :]
        x_right = x[:, :, 8:29,  :]
        x_left  = x[:, :, 29:50, :]
        x_face  = x[:, :, 50:,   :]

        z_upper = self.encoder_upper_body(rearrange(x_upper, 'b f v c -> b f (v c)'))
        z_right = self.encoder_right_hand(rearrange(x_right, 'b f v c -> b f (v c)'))
        z_left  = self.encoder_left_hand( rearrange(x_left,  'b f v c -> b f (v c)'))
        z_face  = self.encoder_face(      rearrange(x_face,  'b f v c -> b f (v c)'))
        return z_upper, z_right, z_left, z_face

    def decode(self, z_upper, z_right, z_left, z_face):
        return (
            self.decoder_upper_body(z_upper),
            self.decoder_right_hand(z_right),
            self.decoder_left_hand(z_left),
            self.decoder_face(z_face),
        )

    def forward(self, x, mask):
        z_upper, z_right, z_left, z_face = self.encode(x)
        recon_upper, recon_right, recon_left, recon_face = self.decode(z_upper, z_right, z_left, z_face)

        recon_upper = rearrange(recon_upper, 'b f (v c) -> b f v c', v=6,   c=self.num_feats)
        recon_right = rearrange(recon_right, 'b f (v c) -> b f v c', v=21,  c=self.num_feats)
        recon_left  = rearrange(recon_left,  'b f (v c) -> b f v c', v=21,  c=self.num_feats)
        recon_face  = rearrange(recon_face,  'b f (v c) -> b f v c', v=128, c=self.num_feats)

        recon_x = torch.cat([recon_upper, recon_right, recon_left, recon_face], dim=2)
        recon_x = recon_x * mask.unsqueeze(-1).unsqueeze(-1)
        return recon_x, recon_upper, recon_right, recon_left, recon_face

    def get_input(self, batch):
        keypoints    = batch["poses"]
        frame_lengths = batch["frame_lengths"]
        mask = create_mask(frame_lengths, self.max_frame_len, device=self.device)
        return keypoints, mask

    def share_step(self, inputs, mask, split="train"):
        _, recon_upper, recon_right, recon_left, recon_face = self(inputs, mask)

        w_upper, w_right, w_left, w_face = 0.5, 1.5, 1.5, 1.0

        def masked_l1(pred, gt):
            return (F.l1_loss(pred, gt, reduction='none') * mask.unsqueeze(-1).unsqueeze(-1)).sum() / mask.sum()

        loss_upper = w_upper * masked_l1(recon_upper, inputs[:, :, [0, 1, 3, 4, 6, 7], :])
        loss_right = w_right * masked_l1(recon_right, inputs[:, :, 8:29,  :])
        loss_left  = w_left  * masked_l1(recon_left,  inputs[:, :, 29:50, :])
        loss_face  = w_face  * masked_l1(recon_face,  inputs[:, :, 50:,   :])
        total_loss = loss_upper + loss_right + loss_left + loss_face

        self.log_dict({
            f"{split}/loss_upper": loss_upper.detach(),
            f"{split}/loss_right": loss_right.detach(),
            f"{split}/loss_left":  loss_left.detach(),
            f"{split}/loss_face":  loss_face.detach(),
            f"{split}/total_loss": total_loss.detach(),
        }, prog_bar=True, sync_dist=True, batch_size=inputs.shape[0])
        return total_loss

    def training_step(self, batch, batch_idx):
        inputs, mask = self.get_input(batch)
        total_loss = self.share_step(inputs, mask, split="train")
        l1_reg = (
            sum(p.abs().sum() for p in self.encoder_upper_body.parameters()) +
            sum(p.abs().sum() for p in self.encoder_right_hand.parameters()) +
            sum(p.abs().sum() for p in self.encoder_left_hand.parameters())
        )
        total_loss = total_loss + self.l1_lambda * l1_reg
        self.training_step_outputs.append(total_loss)
        return total_loss

    def on_train_epoch_end(self):
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        self.total_train_losses.append(epoch_loss.item())
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, mask = self.get_input(batch)
        total_loss = self.share_step(inputs, mask, split="valid")
        self.valid_step_outputs.append(total_loss)

    def on_validation_epoch_end(self):
        epoch_loss = torch.stack(self.valid_step_outputs).mean()
        self.total_valid_losses.append(epoch_loss.item())
        self.valid_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
