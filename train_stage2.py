"""
DARSLP — Stage 2: Non-Autoregressive Transformer Generator Training
====================================================================
Trains the Stage 2 transformer (text → latent embeddings) on top of a frozen
Stage 1 autoencoder. Two variants are supported:

  darslp (default)
      DARSLPGenerator — full disentangled pipeline. Predicts separate latent
      codes for upper body, right hand, left hand, and face. Requires a
      DisentangledAE checkpoint from train_stage1.py --model disentangled.

      Training runs in two phases:
        Phase 1: latent L1 regression + length prediction loss.
        Phase 2: add channel-aware KL regularization (pass --prior_file).

  standard
      StandardGenerator — non-disentangled ablation. Predicts a single unified
      latent vector. Requires a StandardAE checkpoint from
      train_stage1.py --model standard.

Requires a pretrained Stage 1 checkpoint (--ae_ckpt) saved by train_stage1.py.

Prerequisite — precompute pose encodings (run once before Stage 2 training):
    python precompute_encodings.py \\
        --model disentangled --ae_ckpt models/ae_phoenix_disentangled.pth \\
        --poses /data/phoenix/train.pt \\
        --output_dir data/pose_encodings/phoenix_train_disentangled_80dim
    python precompute_encodings.py \\
        --model disentangled --ae_ckpt models/ae_phoenix_disentangled.pth \\
        --poses /data/phoenix/dev.pt \\
        --output_dir data/pose_encodings/phoenix_dev_disentangled_80dim

Usage — DARSLP disentangled (PHOENIX14T, Phase 1):
    python train_stage2.py \\
        --model           darslp \\
        --config          configs/train_phoenix.yaml \\
        --ae_ckpt         models/ae_phoenix_disentangled.pth \\
        --train_pt        /data/phoenix/train.pt \\
        --train_embeddings /data/phoenix/text_embeddings/train \\
        --train_encodings  data/pose_encodings/phoenix_train_disentangled_80dim \\
        --dev_pt          /data/phoenix/dev.pt \\
        --dev_embeddings  /data/phoenix/text_embeddings/dev \\
        --dev_encodings   data/pose_encodings/phoenix_dev_disentangled_80dim \\
        --pose_dim        80 --RH_weight 7 --LH_weight 5

Usage — DARSLP Phase 2 (add KL regularization):
    python train_stage2.py \\
        ... (same as above) \\
        --prior_file data/channel_priors/channel_priors_phoenix_80dim.npy

Usage — Non-disentangled ablation (PHOENIX14T):
    python train_stage2.py \\
        --model           standard \\
        --config          configs/train_phoenix.yaml \\
        --ae_ckpt         models/ae_phoenix_standard.pth \\
        --train_pt        /data/phoenix/train.pt \\
        --train_embeddings /data/phoenix/text_embeddings/train \\
        --train_encodings  data/pose_encodings/phoenix_train_standard_80dim \\
        --dev_pt          /data/phoenix/dev.pt \\
        --dev_embeddings  /data/phoenix/text_embeddings/dev \\
        --dev_encodings   data/pose_encodings/phoenix_dev_standard_80dim \\
        --pose_dim        80
"""

import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.data.SLPDataset import SLPDataset
from src.models.darslp_generator import DARSLPGenerator
from src.models.standard_generator import StandardGenerator
from src.utils.helpers import load_config


def build_generator(args, cfg, ae_model, vocab, len_train):
    shared = dict(
        cfg=cfg, args=args, len_train_dataloader=len_train,
        ae_model=ae_model, text_vocab=vocab,
        num_joints=176, num_feats=3, pose_dim=args.pose_dim,
    )
    if args.model == "darslp":
        extra = {}
        if args.reference_pose:
            extra["reference_pose_path"] = args.reference_pose
        return DARSLPGenerator(
            **shared,
            RH_weight=args.RH_weight,  LH_weight=args.LH_weight,
            KL_RH_weight=args.KL_RH_weight, KL_LH_weight=args.KL_LH_weight,
            kl_loss=bool(args.prior_file),
            **extra,
        )
    elif args.model == "standard":
        return StandardGenerator(**shared)
    else:
        raise ValueError(f"Unknown --model '{args.model}'. Choose: darslp, standard")


def main(args):
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg       = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab     = tokenizer.vocab

    train_dataset = SLPDataset(
        pt_file_path=args.train_pt,
        text_embeddings_path=args.train_embeddings,
        texts_max_length=cfg["data"]["max_text_len"],
        poses_max_length=cfg["data"]["max_frame_len"],
        encodings_path=args.train_encodings,
    )
    dev_dataset = SLPDataset(
        pt_file_path=args.dev_pt,
        text_embeddings_path=args.dev_embeddings,
        texts_max_length=cfg["data"]["max_text_len"],
        poses_max_length=cfg["data"]["max_frame_len"],
        encodings_path=args.dev_encodings,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True,
                              persistent_workers=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False,
                              persistent_workers=True)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    ae_model = torch.load(args.ae_ckpt, map_location=device, weights_only=False)

    model = build_generator(args, cfg, ae_model, vocab, len(train_loader))

    # Attach channel priors for Phase 2 KL regularization (darslp only)
    if args.model == "darslp" and args.prior_file:
        from src.loss.channel_kl_loss import load_channel_priors
        prior_mu, prior_logvar = load_channel_priors(args.prior_file, args.pose_dim)
        model.prior_mu     = prior_mu
        model.prior_logvar = prior_logvar

    best_ckpt = ModelCheckpoint(monitor="valid/total_loss", mode="min",
                                save_top_k=1, filename="best-val-{epoch}")
    last_ckpt = ModelCheckpoint(filename="last-checkpoint-epoch={epoch}",
                                save_last=False, auto_insert_metric_name=False)
    early_stop = EarlyStopping(monitor="valid/total_loss", mode="min",
                               patience=150, verbose=True)
    logger = CSVLogger(REPO_ROOT, name=f"logs_stage2_{args.model}")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[best_ckpt, last_ckpt, early_stop],
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        strategy="ddp_find_unused_parameters_false" if args.gpus > 1 else "auto",
    )

    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="darslp", choices=["darslp", "standard"],
                        help="Generator variant: 'darslp' (disentangled) or 'standard' (ablation)")
    # Required
    parser.add_argument("--config",           required=True)
    parser.add_argument("--ae_ckpt",          required=True,
                        help="Path to pretrained Stage 1 .pth model")
    parser.add_argument("--train_pt",         required=True)
    parser.add_argument("--train_embeddings", required=True)
    parser.add_argument("--dev_pt",           required=True)
    parser.add_argument("--dev_embeddings",   required=True)
    # Optional data
    parser.add_argument("--train_encodings",  default=None,
                        help="Dir with precomputed DisentangledAE encodings (train)")
    parser.add_argument("--dev_encodings",    default=None,
                        help="Dir with precomputed DisentangledAE encodings (dev)")
    # Optional model / training
    parser.add_argument("--tokenizer",        default="dbmdz/bert-base-german-uncased")
    parser.add_argument("--pose_dim",         type=int,   default=96)
    parser.add_argument("--RH_weight",        type=float, default=7.0)
    parser.add_argument("--LH_weight",        type=float, default=5.0)
    parser.add_argument("--KL_RH_weight",     type=float, default=1.0)
    parser.add_argument("--KL_LH_weight",     type=float, default=1.0)
    parser.add_argument("--prior_file",       default=None,
                        help="Path to .npy/.npz prior file for Phase 2 KL regularization")
    parser.add_argument("--reference_pose",   default=None,
                        help="Path to reference_pose.pt (default: data/reference_pose.pt in repo root)")
    parser.add_argument("--epochs",           type=int,   default=800)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--num_workers",      type=int,   default=8)
    parser.add_argument("--gpus",             type=int,   default=4)
    parser.add_argument("--seed",             type=int,   default=42)
    main(parser.parse_args())
