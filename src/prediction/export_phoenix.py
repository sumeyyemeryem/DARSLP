"""
Export DARSLP Predictions — PHOENIX-2014T Test Set
====================================================
Runs inference with a trained DARSLPGenerator + DisentangledAE checkpoint on the
PHOENIX-2014T test split and saves all predictions to a .npz archive.

The output file is compatible with the SLRTP back-translation evaluation pipeline
(https://github.com/walsharry/SLRTP-Sign-Production-Evaluation).

Output .npz keys:
    names      — (N,)  dataset-specific sample IDs
    texts      — (N,)  spoken-language sentences
    glosses    — (N,)  gloss sequences
    speakers   — (N,)  signer IDs
    pred_poses — (N,)  object array; pred_poses[i] has shape (F_pred, 178, 3)
    gt_poses   — (N,)  object array; gt_poses[i]   has shape (F_gt,   178, 3)

Pose layout (178 joints):
    0–1   upper body (shoulders/elbows)
    2     right wrist  (restored from right-hand wrist)
    3–4   upper body (continued)
    5     left wrist   (restored from left-hand wrist)
    6–26  right hand fingers
    27–47 left hand fingers
    48–175 face (128 landmarks)

Usage:
    python src/prediction/export_phoenix.py \\
        --config          configs/train_phoenix.yaml \\
        --ckpt            /path/to/best-val-epoch=780.ckpt \\
        --ae_ckpt         /path/to/ae_model.pth \\
        --text_embeddings /data/phoenix/text_embeddings/test \\
        --poses           /data/phoenix/test.pt \\
        --output          exports/predictions_DARSLP_phoenix_test.npz
"""

import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.data.SLPDataset import SLPDataset
from src.models.darslp_generator import DARSLPGenerator
from src.utils.helpers import load_config

MAX_TEXT_LEN  = 52
MAX_FRAME_LEN = 300


def run_inference(model, dataloader, vocab, device):
    pred_poses, gt_poses = [], []
    names_list, texts_list, glosses_list, speakers_list = [], [], [], []

    for batch in dataloader:
        source_embeddings = batch['text_embeddings']
        names             = batch['name']
        seq_lengths       = batch['frame_lengths']

        gt_text_tensor = torch.tensor(source_embeddings, dtype=torch.float32).to(device)
        text_mask      = torch.tensor((source_embeddings != vocab["[PAD]"])[:, :, 0]).to(device)
        kp_mask        = torch.ones(len(names), MAX_FRAME_LEN, dtype=torch.bool, device=device)

        with torch.no_grad():
            reconstructions, length_ratio = model.predict(gt_text_tensor, text_mask, kp_mask)

        pred_lengths = (length_ratio * model.max_frame_len).round().long()

        # Restore wrist joints: RHWrist(6) → RWrist(2), LHWrist(27) → LWrist(5)
        generated_pose = torch.cat([
            reconstructions[:, :, :2, :],
            reconstructions[:, :, 6,  :].unsqueeze(2),
            reconstructions[:, :, 2:4, :],
            reconstructions[:, :, 27, :].unsqueeze(2),
            reconstructions[:, :, 4:, :],
        ], dim=2).cpu()

        for i in range(len(names)):
            f_pred = pred_lengths[i].item()
            f_gt   = seq_lengths[i]
            pred_poses.append(generated_pose[i][:f_pred].numpy())
            gt_poses.append(batch['poses'][i][:f_gt].numpy())
            names_list.append(names[i])
            texts_list.append(batch['text'][i])
            glosses_list.append(batch['gloss'][i])
            speakers_list.append(batch['speaker'][i])

    return dict(
        names     =np.array(names_list),
        texts     =np.array(texts_list),
        glosses   =np.array(glosses_list),
        speakers  =np.array(speakers_list),
        pred_poses=np.array(pred_poses, dtype=object),
        gt_poses  =np.array(gt_poses,   dtype=object),
    )


def main(args):
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = load_config(args.config)
    vocab  = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased").vocab

    dataset = SLPDataset(
        text_embeddings_path=args.text_embeddings,
        pt_file_path=args.poses,
        texts_max_length=MAX_TEXT_LEN,
        poses_max_length=MAX_FRAME_LEN,
    )
    dataloader = DataLoader(dataset, batch_size=32, num_workers=1, shuffle=False,
                            persistent_workers=True)

    ae_model = torch.load(args.ae_ckpt, weights_only=False)
    ckpt_kwargs = dict(
        ae_model=ae_model,
        text_vocab=vocab,
        max_frame_len=MAX_FRAME_LEN,
        cfg=cfg,
        args=args,
        num_joints=176,
        num_feats=3,
        pose_dim=80,
    )
    if args.reference_pose:
        ckpt_kwargs["reference_pose_path"] = args.reference_pose
    model = DARSLPGenerator.load_from_checkpoint(args.ckpt, **ckpt_kwargs)
    model.to(device).eval()

    results = run_inference(model, dataloader, vocab, device)
    np.savez(args.output, **results)
    print(f"Saved {len(results['names'])} samples → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default=r"configs\train_phoenix.yaml")
    parser.add_argument("--ckpt",            required=True)
    parser.add_argument("--ae_ckpt",         required=True)
    parser.add_argument("--text_embeddings", required=True)
    parser.add_argument("--poses",           required=True)
    parser.add_argument("--output",          default=r"exports\predictions_DARSLP_phoenix_test.npz")
    parser.add_argument("--reference_pose",  default=None,
                        help="Path to reference_pose.pt (default: data/reference_pose.pt in repo root)")
    main(parser.parse_args())
