"""
DARSLP — Stage 1: Autoencoder Training
=======================================
Trains the Stage 1 pose autoencoder. Two variants are supported:

  disentangled (default)
      DisentangledAE — Structurally Disentangled Pose Autoencoder.
      Encodes each frame into four separate latent subspaces: upper body,
      right hand, left hand, face. This is the Stage 1 used by the full
      DARSLP pipeline (train_stage2.py --model darslp).

  standard
      StandardAE — Non-disentangled baseline. Encodes all 176 joints into
      a single unified latent vector. Used for ablation studies alongside
      train_stage2.py --model standard.

The trained model is saved as a .pth file and consumed by train_stage2.py
via --ae_ckpt.

Usage — DisentangledAE (PHOENIX14T):
    python train_stage1.py \\
        --model           disentangled \\
        --train_pt        /data/phoenix/train.pt \\
        --train_embeddings /data/phoenix/text_embeddings/train \\
        --dev_pt          /data/phoenix/dev.pt \\
        --dev_embeddings  /data/phoenix/text_embeddings/dev \\
        --output          models/ae_phoenix_disentangled.pth \\
        --latent_dim      64 --face_latent_dim 16 --max_frame_len 300

Usage — StandardAE (PHOENIX14T, ablation):
    python train_stage1.py \\
        --model           standard \\
        --train_pt        /data/phoenix/train.pt \\
        --train_embeddings /data/phoenix/text_embeddings/train \\
        --dev_pt          /data/phoenix/dev.pt \\
        --dev_embeddings  /data/phoenix/text_embeddings/dev \\
        --output          models/ae_phoenix_standard.pth \\
        --latent_dim      64
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

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.data.SLPDataset import SLPDataset
from src.models.disentangled_ae import DisentangledAE
from src.models.standard_ae import StandardAE


def build_ae(args):
    if args.model == "disentangled":
        return DisentangledAE(
            num_joints=176,
            num_feats=3,
            max_frame_len=args.max_frame_len,
            latent_dim=args.latent_dim,
            face_latent_dim=args.face_latent_dim,
            l1_lambda=args.l1_lambda,
            base_learning_rate=args.lr,
        )
    elif args.model == "standard":
        return StandardAE(
            num_joints=178,
            num_feats=3,
            max_frame_len=args.max_frame_len,
            latent_dim=args.latent_dim,
            l1_lambda=args.l1_lambda,
            base_learning_rate=args.lr,
        )
    else:
        raise ValueError(f"Unknown --model '{args.model}'. Choose: disentangled, standard")


def main(args):
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = SLPDataset(
        pt_file_path=args.train_pt,
        text_embeddings_path=args.train_embeddings,
        texts_max_length=args.max_text_len,
        poses_max_length=args.max_frame_len,
    )
    dev_dataset = SLPDataset(
        pt_file_path=args.dev_pt,
        text_embeddings_path=args.dev_embeddings,
        texts_max_length=args.max_text_len,
        poses_max_length=args.max_frame_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True,
                              persistent_workers=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False,
                              persistent_workers=True)

    model = build_ae(args)

    best_ckpt = ModelCheckpoint(monitor="valid/total_loss", mode="min",
                                save_top_k=1, filename="best-stage1-{epoch}")
    last_ckpt = ModelCheckpoint(filename="last-stage1-epoch={epoch}", save_last=False)
    early_stop = EarlyStopping(monitor="valid/total_loss", mode="min",
                               patience=100, verbose=True)
    logger = CSVLogger(REPO_ROOT, name=f"logs_stage1_{args.model}")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[best_ckpt, last_ckpt, early_stop],
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        strategy="ddp_find_unused_parameters_false" if args.gpus > 1 else "auto",
    )

    trainer.fit(model, train_loader, dev_loader)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(model, args.output)
    print(f"Stage 1 model saved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="disentangled",
                        choices=["disentangled", "standard"],
                        help="AE variant: 'disentangled' (DisentangledAE) or 'standard' (StandardAE)")
    parser.add_argument("--train_pt",         required=True)
    parser.add_argument("--dev_pt",           required=True)
    parser.add_argument("--train_embeddings", required=True)
    parser.add_argument("--dev_embeddings",   required=True)
    parser.add_argument("--output",           required=True,
                        help="Output path for saved .pth AE model")
    parser.add_argument("--latent_dim",       type=int,   default=64)
    parser.add_argument("--face_latent_dim",  type=int,   default=16,
                        help="Face latent dim (disentangled model only)")
    parser.add_argument("--max_frame_len",    type=int,   default=300)
    parser.add_argument("--max_text_len",     type=int,   default=52)
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--num_workers",      type=int,   default=8)
    parser.add_argument("--gpus",             type=int,   default=1)
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--l1_lambda",        type=float, default=1e-4)
    parser.add_argument("--seed",             type=int,   default=42)
    main(parser.parse_args())
