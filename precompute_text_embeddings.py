"""
Precompute BERT Text Embeddings — DARSLP / A²V-SLP
===================================================
Extracts per-token BERT embeddings for every sample in a dataset split and
saves them as individual .npy files (one per sample, shape: (n_tokens, 768)).

These files are loaded by SLPDataset and padded to texts_max_length during
training. Precomputing them once avoids running BERT on every training epoch.

Output per sample:
    <output_dir>/<sample_key>.npy   shape: (n_tokens, 768), float32
    Includes [CLS] and [SEP] tokens — consistent with the original pipeline.

Supported datasets / tokenizers:
    PHOENIX-2014T  →  dbmdz/bert-base-german-uncased   (German)
    CSL-Daily      →  hfl/chinese-bert-wwm              (Chinese)
    TSL            →  dbmdz/bert-base-turkish-uncased   (Turkish)

NOTE on PHOENIX-2014T reproducibility:
    The original pre-computed embeddings distributed with DARSLP were extracted
    from the raw PHOENIX-2014T text (.txt) files, not from the .pt file's text
    field. The .pt text field may contain encoding artifacts that cause slightly
    different tokenization (and therefore different token counts). If you need
    to exactly reproduce the published results, use the pre-computed embeddings
    from the HuggingFace release. This script is the correct approach for new
    datasets or when re-generating embeddings from scratch.

Usage — PHOENIX-2014T train split:
    python precompute_text_embeddings.py \\
        --poses    /data/phoenix/train.pt \\
        --tokenizer dbmdz/bert-base-german-uncased \\
        --output_dir /data/phoenix/text_embeddings/train

Usage — CSL-Daily dev split:
    python precompute_text_embeddings.py \\
        --poses    /data/csl/dev.pt \\
        --tokenizer hfl/chinese-bert-wwm \\
        --output_dir /data/csl/text_embeddings/dev
"""

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading tokenizer and model: {args.tokenizer}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    bert      = AutoModel.from_pretrained(args.tokenizer).to(device).eval()

    data = torch.load(args.poses, map_location="cpu", weights_only=False)
    keys = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    existing  = set(os.listdir(args.output_dir))
    processed = 0
    skipped   = 0

    for i, key in enumerate(keys):
        out_file = os.path.join(args.output_dir, f"{key}.npy")

        if not args.overwrite and f"{key}.npy" in existing:
            skipped += 1
            continue

        text = data[key]["text"]

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = bert(**tokens)

        # Last hidden state: (1, seq_len, 768) → strip batch dim only.
        # [CLS] and [SEP] are kept — consistent with the original pipeline.
        hidden = outputs.last_hidden_state.squeeze(0)   # (seq_len, 768)

        np.save(out_file, hidden.cpu().float().numpy())
        processed += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(keys):
            print(f"  [{i+1}/{len(keys)}]  saved: {processed}  skipped: {skipped}")

    print(f"\nDone. {processed} embeddings saved to {args.output_dir}")
    if skipped:
        print(f"  ({skipped} already existed — use --overwrite to recompute)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses",      required=True,
                        help="Path to dataset .pt file (train / dev / test)")
    parser.add_argument("--tokenizer",  required=True,
                        help="HuggingFace tokenizer/model name (e.g. dbmdz/bert-base-german-uncased)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where per-sample .npy embeddings will be saved")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Recompute embeddings even if the .npy file already exists")
    main(parser.parse_args())
