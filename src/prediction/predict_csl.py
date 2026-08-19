"""
DARSLP Inference — CSL-Daily
==============================
Runs inference with a trained DARSLPGenerator + DisentangledAE checkpoint on any
CSL-Daily split (train / dev / test) and saves the results as a .pt file.

Output .pt structure (dict keyed by sample ID):
    {
      "<sample_id>": {
          "name":             str,          e.g. "S000020_P0004_T00"
          "text":             str,
          "gloss":            str,
          "speaker":          str,
          "poses_3d":         Tensor [F_pred, 178, 3],   # predicted pose
          "gt_pose_sequence": Tensor [F_gt,   178, 3],   # ground-truth pose
      },
      ...
    }

Note on wrist slot convention (CSL-Daily):
    CSL hand layout swaps which wrist index represents which hand compared to
    PHOENIX-2014T. The joint insertion here mirrors the original CSL pipeline:
      LHWrist (index 27) → RWrist slot (index 2)
      RHWrist (index  6) → LWrist slot (index 5)

The .pt output is used by the visualization scripts in src/visualization/.
For back-translation evaluation use export_csl.py instead (outputs .npz).

Usage:
    python src/prediction/predict_csl.py \\
        --config          configs/train_CSL.yaml \\
        --ckpt            /path/to/best-val-epoch=446.ckpt \\
        --ae_ckpt         /path/to/ae_model_csl.pth \\
        --text_embeddings /data/csl/text_embeddings/test \\
        --poses           /data/csl/test.pt \\
        --output          predictions/predictions_csl_test.pt
"""

import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.data.SLPDataset import SLPDataset
from src.models.darslp_generator import DARSLPGenerator
from src.utils.helpers import load_config

MAX_TEXT_LEN  = 43
MAX_FRAME_LEN = 350


def generate_predictions(model, dataloader, device):
    """Run inference over the dataloader and return a dict of results."""
    results = {}

    for batch in dataloader:
        poses             = rearrange(batch['poses'], 'b f v c -> b f (v c)')
        source_embeddings = batch['text_embeddings']
        names             = batch['name']
        seq_lengths       = batch['frame_lengths']
        texts             = batch['text']
        glosses           = batch['gloss']
        speakers          = batch['speaker']

        gt_text_tensor = torch.tensor(source_embeddings, dtype=torch.float32).to(device)
        text_mask      = torch.tensor((source_embeddings != vocab["[PAD]"])[:, :, 0]).to(device)
        kp_mask        = torch.arange(MAX_FRAME_LEN, device=device)[None, :] < \
                         torch.full((len(names),), MAX_FRAME_LEN, device=device)[:, None]

        with torch.no_grad():
            reconstructions, length_ratio = model.predict(gt_text_tensor, text_mask, kp_mask)

        pred_lengths = (length_ratio * model.max_frame_len).round().long()

        # CSL wrist convention: LHWrist(27) → RWrist(2), RHWrist(6) → LWrist(5)
        generated_pose = torch.cat([
            reconstructions[:, :, :2,  :],
            reconstructions[:, :, 27,  :].unsqueeze(2),
            reconstructions[:, :, 2:4, :],
            reconstructions[:, :, 6,   :].unsqueeze(2),
            reconstructions[:, :, 4:,  :],
        ], dim=2).cpu()

        for i in range(len(names)):
            f_pred = pred_lengths[i].item()
            f_gt   = seq_lengths[i]
            results[names[i]] = {
                'name':             names[i],
                'text':             texts[i],
                'gloss':            glosses[i],
                'speaker':          speakers[i],
                'poses_3d':         generated_pose[i][:f_pred],
                'gt_pose_sequence': batch['poses'][i][:f_gt],
            }

    return results


def main(args):
    global vocab  # used inside generate_predictions

    pl.seed_everything(42)
    torch.manual_seed(42)
    np.random.seed(42)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg       = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    vocab     = tokenizer.vocab

    dataset = SLPDataset(
        text_embeddings_path=args.text_embeddings,
        pt_file_path=args.poses,
        texts_max_length=MAX_TEXT_LEN,
        poses_max_length=MAX_FRAME_LEN,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False,
                            persistent_workers=(args.num_workers > 0))

    ae_model = torch.load(args.ae_ckpt, map_location=device, weights_only=False)

    ckpt_kwargs = dict(
        ae_model=ae_model,
        text_vocab=vocab,
        max_frame_len=MAX_FRAME_LEN,
        cfg=cfg,
        args=args,
        num_joints=176,
        num_feats=3,
        pose_dim=args.pose_dim,
    )
    if args.reference_pose:
        ckpt_kwargs["reference_pose_path"] = args.reference_pose
    model = DARSLPGenerator.load_from_checkpoint(args.ckpt, **ckpt_kwargs)
    model.to(device).eval()

    results = generate_predictions(model, dataloader, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(results, args.output)
    print(f"Saved {len(results)} samples -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default=r"configs\train_CSL.yaml")
    parser.add_argument("--ckpt",            required=True,
                        help="Path to DARSLPGenerator checkpoint (.ckpt)")
    parser.add_argument("--ae_ckpt",         required=True,
                        help="Path to DisentangledAE model (.pth)")
    parser.add_argument("--text_embeddings", required=True,
                        help="Dir with BERT embedding .npy files")
    parser.add_argument("--poses",           required=True,
                        help="Path to poses .pt file (train / dev / test)")
    parser.add_argument("--output",          default=r"predictions\predictions_csl_test.pt")
    parser.add_argument("--pose_dim",        type=int, default=80)
    parser.add_argument("--reference_pose",  default=None,
                        help="Path to reference_pose.pt (default: data/reference_pose.pt in repo root)")
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--num_workers",     type=int, default=1)
    main(parser.parse_args())
