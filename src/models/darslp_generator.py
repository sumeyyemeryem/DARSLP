"""
Stage 2 — Non-Autoregressive Transformer Generator (DARSLP)
============================================================
Predicts articulator-wise latent embeddings directly from text (BERT embeddings),
bypassing sequential decoding and eliminating autoregressive error accumulation.

Architecture:
  • Transformer encoder over tokenized text → masked mean pooling
  • Length predictor: a linear head that estimates the output-length ratio
  • Static reference-pose queries projected into decoder dimension
  • Transformer decoder → per-frame latent codes
  • Linear projection → split into (z_upper, z_right, z_left, z_face) matching
    DisentangledAE's latent structure

Training (two phases):
  Phase 1 — latent regression only: L1 loss between predicted and ground-truth
    DisentangledAE encodings + length prediction loss.
  Phase 2 — add channel-aware KL regularization: aligns the predicted per-channel
    distribution with disentangled-region priors precomputed from the training set,
    reducing regression to the mean and improving motion diversity.

Requires a pretrained, frozen DisentangledAE (Stage 1) for decoding latents back
to pose sequences at inference time.

Reference: Taşyürek et al., "Disentangle and Regularize: Sign Language Production
with Articulator-Based Disentanglement and Channel-Aware Regularization," WACV 2026.
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

from src.loss.channel_kl_loss import load_channel_priors, channel_kl_divergence
from src.models.positional import PositionalEncoding
from src.utils.helpers import create_mask


class DARSLPGenerator(pl.LightningModule):
    def __init__(
        self,
        cfg=None,
        args=None,
        len_train_dataloader=None,
        ae_model=None,
        text_vocab=None,
        kl_loss=False,
        pose_dim=96,
        max_frame_len=300,
        num_joints=176,
        num_feats=3,
        first_stage_trainable=False,
        dim_feedforward_e=1024,
        dim_feedforward_d=1024,
        encoder_dim=512,
        decoder_dim=512,
        intermediate_dim=128,
        decoder_n_heads=4,
        encoder_n_heads=4,
        decoder_n_layers=3,
        encoder_n_layers=3,
        dropout_e=0.1,
        dropout_d=0.1,
        activation_e="relu",
        activation_d="relu",
        emb_dim=512,
        base_learning_rate=1.0e-4,
        label_smoothing=0.1,
        RH_weight=1,
        LH_weight=1,
        KL_RH_weight=1,
        KL_LH_weight=1,
        **kwargs,
    ):
        super().__init__()

        self.cfg                  = cfg
        self.args                 = args
        self.len_train_dataloader = len_train_dataloader
        self.text_vocab           = text_vocab
        self.max_frame_len        = max_frame_len
        self.num_joints           = num_joints
        self.num_feats            = num_feats
        self.kl_loss              = kl_loss
        self.pose_dim             = pose_dim
        self.RH_weight            = RH_weight
        self.LH_weight            = LH_weight
        self.KL_RH_weight         = KL_RH_weight
        self.KL_LH_weight         = KL_LH_weight

        def _cfg(section, key, default):
            return cfg[section].get(key, default) if cfg and section in cfg else default

        self.first_stage_trainable = _cfg('model', 'first_stage_trainable', first_stage_trainable)
        self.encoder_n_layers  = _cfg('model', 'encoder', {}).get('n_layers',      encoder_n_layers)
        self.encoder_n_heads   = _cfg('model', 'encoder', {}).get('n_head',        encoder_n_heads)
        self.encoder_dim       = _cfg('model', 'encoder', {}).get('encoder_dim',   encoder_dim)
        self.dim_feedforward_e = _cfg('model', 'encoder', {}).get('dim_feedforward', dim_feedforward_e)
        self.activation_e      = _cfg('model', 'encoder', {}).get('activation',    activation_e)
        self.dropout_e         = _cfg('model', 'encoder', {}).get('dropout',       dropout_e)

        self.decoder_n_layers  = _cfg('model', 'decoder', {}).get('n_layers',      decoder_n_layers)
        self.decoder_n_heads   = _cfg('model', 'decoder', {}).get('n_head',        decoder_n_heads)
        self.decoder_dim       = _cfg('model', 'decoder', {}).get('decoder_dim',   decoder_dim)
        self.intermediate_dim  = _cfg('model', 'decoder', {}).get('intermediate_dim', intermediate_dim)
        self.dim_feedforward_d = _cfg('model', 'decoder', {}).get('dim_feedforward', dim_feedforward_d)
        self.activation_d      = _cfg('model', 'decoder', {}).get('activation',    activation_d)
        self.dropout_d         = _cfg('model', 'encoder', {}).get('dropout',       dropout_d)
        self.emb_dim           = _cfg('model', 'embeddings', {}).get('embedding_dim', emb_dim)
        self.learning_rate     = _cfg('training', 'base_learning_rate', base_learning_rate)
        self.label_smoothing   = _cfg('training', 'label_smoothing',    label_smoothing)

        self.training_step_outputs = []
        self.valid_step_outputs    = []
        self.valid_losses          = []
        self.train_losses          = []

        self.text_emb = nn.Linear(768, self.emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            self.encoder_dim, self.encoder_n_heads, self.dim_feedforward_e,
            self.dropout_e, self.activation_e, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.encoder_n_layers)

        self.length_predictor = nn.Linear(self.encoder_dim, 1)

        self.pose_projection = nn.Linear(num_joints * num_feats, self.decoder_dim)
        _default_ref = os.path.join(REPO_ROOT, "data", "reference_pose.pt")
        reference_pose_path = kwargs.get("reference_pose_path", _default_ref)
        reference_pose = torch.load(reference_pose_path, map_location="cpu", weights_only=False)
        keep = torch.ones(reference_pose.shape[0], dtype=torch.bool)
        keep[[2, 5]] = False
        self.reference_pose = reference_pose[keep]

        dec_layer = nn.TransformerDecoderLayer(
            self.decoder_dim, self.decoder_n_heads, self.dim_feedforward_d,
            self.dropout_d, self.activation_d, batch_first=True)
        self.decoder     = nn.TransformerDecoder(dec_layer, num_layers=self.decoder_n_layers)
        self.pos_encoding = PositionalEncoding(self.decoder_dim)
        self.projection   = nn.Linear(self.decoder_dim, pose_dim)

        if not self.first_stage_trainable:
            self.first_stage_model = ae_model.eval()
            for p in self.first_stage_model.parameters():
                p.requires_grad = False
        else:
            self.first_stage_model = ae_model

    def _latent_dims(self):
        def out_features(enc):
            return enc.out_features if hasattr(enc, 'out_features') else enc[-1].out_features
        return (
            out_features(self.first_stage_model.encoder_upper_body),
            out_features(self.first_stage_model.encoder_right_hand),
            out_features(self.first_stage_model.encoder_left_hand),
            out_features(self.first_stage_model.encoder_face),
        )

    def encode(self, src, mask):
        embed = self.pos_encoding(self.text_emb(src))
        return self.encoder(embed, src_key_padding_mask=~mask)

    def decode(self, enc_outs, trg, mask):
        embed = self.pos_encoding(trg)
        return self.decoder(tgt=embed, memory=enc_outs, tgt_key_padding_mask=~mask)

    def _time_queries(self, batch_size, seq_len):
        flat  = self.reference_pose.view(-1)
        proj  = self.pose_projection(flat.unsqueeze(0).expand(batch_size, -1).to(self.device))
        return proj.unsqueeze(1).expand(-1, seq_len, -1)

    def forward(self, text, text_mask, keypoints_mask):
        batch_size, seq_len = keypoints_mask.shape
        enc_outs  = self.encode(text, text_mask)
        valid_cnt = text_mask.sum(dim=1, keepdim=True)
        mean_pool = (enc_outs * text_mask.unsqueeze(-1)).sum(dim=1) / valid_cnt
        length_ratio = torch.sigmoid(self.length_predictor(mean_pool))
        dec_outs     = self.decode(enc_outs, self._time_queries(batch_size, seq_len), keypoints_mask)
        reduced      = self.projection(dec_outs)
        dims = self._latent_dims()
        z_upper, z_right, z_left, z_face = torch.split(reduced, list(dims), dim=-1)
        return z_upper, z_right, z_left, z_face, length_ratio

    def predict(self, text, text_mask, keypoints_mask):
        batch_size, seq_len = keypoints_mask.shape
        enc_outs  = self.encode(text, text_mask)
        valid_cnt = text_mask.sum(dim=1, keepdim=True)
        mean_pool = (enc_outs * text_mask.unsqueeze(-1)).sum(dim=1) / valid_cnt
        length_ratio = torch.sigmoid(self.length_predictor(mean_pool))
        dec_outs     = self.decode(enc_outs, self._time_queries(batch_size, seq_len), keypoints_mask)
        reduced      = self.projection(dec_outs)
        dims = self._latent_dims()
        z_upper, z_right, z_left, z_face = torch.split(reduced, list(dims), dim=-1)

        recon_upper, recon_right, recon_left, recon_face = self.first_stage_model.decode(
            z_upper, z_right, z_left, z_face)
        recon_upper = rearrange(recon_upper, 'b f (v c) -> b f v c', v=6,   c=self.num_feats)
        recon_right = rearrange(recon_right, 'b f (v c) -> b f v c', v=21,  c=self.num_feats)
        recon_left  = rearrange(recon_left,  'b f (v c) -> b f v c', v=21,  c=self.num_feats)
        recon_face  = rearrange(recon_face,  'b f (v c) -> b f v c', v=128, c=self.num_feats)
        reconstructions = torch.cat([recon_upper, recon_right, recon_left, recon_face], dim=2)
        reconstructions = reconstructions * keypoints_mask.unsqueeze(-1).unsqueeze(-1)
        return reconstructions, length_ratio

    def get_inputs(self, batch):
        text          = batch["text_embeddings"]
        frame_lengths = batch["frame_lengths"]
        gt_encodings  = batch["pose_encodings"]

        actual_ratio = torch.clamp(frame_lengths / self.max_frame_len, max=1)
        kp_mask      = create_mask(frame_lengths, self.max_frame_len, device=self.device)
        text_mask    = (text != self.text_vocab["[PAD]"])[:, :, 0].to(self.device)
        text         = text.to(self.device)
        kp_mask      = kp_mask.to(self.device)
        return (text, text_mask), kp_mask, gt_encodings, actual_ratio

    def share_step(self, text, text_mask, kp_mask, gt_encodings, actual_ratio, split):
        z_upper, z_right, z_left, z_face, pred_ratio = self(text, text_mask, kp_mask)
        length_loss = F.l1_loss(pred_ratio.squeeze(), actual_ratio)

        dims = self._latent_dims()
        gt_upper, gt_right, gt_left, gt_face = torch.split(gt_encodings, list(dims), dim=-1)

        def masked_l1(pred, gt, w=1.0):
            return w * (F.l1_loss(pred, gt, reduction='none') * kp_mask.unsqueeze(-1)).sum() / kp_mask.sum()

        loss_upper = masked_l1(z_upper, gt_upper)
        loss_right = masked_l1(z_right, gt_right, self.RH_weight)
        loss_left  = masked_l1(z_left,  gt_left,  self.LH_weight)
        loss_face  = masked_l1(z_face,  gt_face)
        pose_loss  = loss_upper + loss_right + loss_left + loss_face

        log = {
            f"{split}/loss_upper":  loss_upper.detach(),
            f"{split}/loss_right":  loss_right.detach(),
            f"{split}/loss_left":   loss_left.detach(),
            f"{split}/loss_face":   loss_face.detach(),
            f"{split}/length_loss": length_loss.detach(),
        }

        if self.kl_loss:
            def batch_kl(z, gt_z, prior_mu, prior_lv, w=1.0):
                mu     = z.mean(dim=(0, 1))
                logvar = z.var(dim=(0, 1), unbiased=False).log()
                return w * channel_kl_divergence(mu, logvar, prior_mu, prior_lv)

            pu, pr, pl, pf = torch.split(self.prior_mu.to(self.device),     list(dims), dim=-1)
            lvu, lvr, lvl, lvf = torch.split(self.prior_logvar.to(self.device), list(dims), dim=-1)
            kl_total = (batch_kl(z_upper, gt_upper, pu, lvu) +
                        batch_kl(z_right, gt_right, pr, lvr, self.KL_RH_weight) +
                        batch_kl(z_left,  gt_left,  pl, lvl, self.KL_LH_weight) +
                        batch_kl(z_face,  gt_face,  pf, lvf))
            total_loss = pose_loss + length_loss + kl_total
            log[f"{split}/kl_total"] = kl_total.detach()
        else:
            total_loss = pose_loss + length_loss

        log[f"{split}/total_loss"] = total_loss.detach()
        return total_loss, log

    def training_step(self, batch, batch_idx):
        (text, text_mask), kp_mask, gt_enc, ratio = self.get_inputs(batch)
        loss, log = self.share_step(text, text_mask, kp_mask, gt_enc, ratio, "train")
        self.log_dict(log, prog_bar=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
                 prog_bar=True, logger=True, sync_dist=True)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        self.train_losses.append(torch.stack(self.training_step_outputs).mean().item())
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        (text, text_mask), kp_mask, gt_enc, ratio = self.get_inputs(batch)
        loss, log = self.share_step(text, text_mask, kp_mask, gt_enc, ratio, "valid")
        self.log_dict(log, prog_bar=True, sync_dist=True)
        self.valid_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        self.valid_losses.append(torch.stack(self.valid_step_outputs).mean().item())
        self.valid_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg['training']['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, threshold=1e-4,
            patience=40, verbose=True, min_lr=1e-12,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'valid/total_loss',
                                 'interval': 'epoch', 'frequency': 1}}
