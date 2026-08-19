"""
Visualize DARSLP Predictions — CSL-Daily
==========================================
Renders predicted pose sequences (optionally overlaid with ground truth) from
a predict_csl.py output .pt file as PNG frames or MP4 videos. Poses are
normalized coordinates and are back-translated to pixel space before plotting
(see src/visualization/plot_pose_mixed_our_norm.py).

Input: the .pt file produced by src/prediction/predict_csl.py, i.e.
    { "<sample_id>": {"poses_3d": Tensor[F_pred,178,3],
                       "gt_pose_sequence": Tensor[F_gt,178,3], ...}, ... }

Usage — a few evenly-spaced PNG frames for 5 random samples:
    python src/visualization/visualize_predictions_mixed_CSL.py \\
        --predictions predictions/predictions_csl_test.pt \\
        --output_dir  visualizations/csl \\
        --num_samples 5 --num_frames 6

Usage — full MP4 video for a specific sample:
    python src/visualization/visualize_predictions_mixed_CSL.py \\
        --predictions predictions/predictions_csl_test.pt \\
        --output_dir  visualizations/csl \\
        --keys S000504_P0008_T00 \\
        --format mp4 --fps 5
"""

import argparse
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.visualization.plot_pose_mixed_our_norm import plot_pose


def _fig_to_bgr(fig):
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def render_sample(key, pred, gt, output_dir, fmt, fps, num_frames, fixed_shoulder_distance, image_shape):
    os.makedirs(output_dir, exist_ok=True)
    num_avail = pred.shape[0] if gt is None else min(pred.shape[0], gt.shape[0])

    def make_fig(i):
        return plot_pose(
            pred[i],
            gt_pose=gt[i] if gt is not None else None,
            fixed_shoulder_distance=fixed_shoulder_distance,
            image_shape=image_shape,
        )

    if fmt == "png":
        indices = np.linspace(0, num_avail - 1, min(num_frames, num_avail), dtype=int)
        for i in indices:
            fig = make_fig(i)
            fig.savefig(os.path.join(output_dir, f"{key}_frame{i:04d}.png"), facecolor="white")
            plt.close(fig)
        print(f"[{key}] saved {len(indices)} PNG frame(s) -> {output_dir}")
    else:
        writer = None
        video_path = os.path.join(output_dir, f"{key}.mp4")
        for i in range(num_avail):
            fig = make_fig(i)
            frame = _fig_to_bgr(fig)
            plt.close(fig)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if writer is not None:
            writer.release()
            print(f"[{key}] saved video ({num_avail} frames) -> {video_path}")


def main(args):
    predictions = torch.load(args.predictions, weights_only=False)
    image_shape = {"width": args.image_width, "height": args.image_height}

    if args.keys:
        keys = args.keys
    else:
        random.seed(args.seed)
        keys = random.sample(list(predictions.keys()), min(args.num_samples, len(predictions)))

    for key in keys:
        if key not in predictions:
            print(f"Skipping '{key}': not found in {args.predictions}")
            continue
        sample = predictions[key]
        pred = sample["poses_3d"]
        pred = pred.numpy() if torch.is_tensor(pred) else pred
        gt = None
        if not args.pred_only:
            gt = sample.get("gt_pose_sequence")
            gt = gt.numpy() if torch.is_tensor(gt) else gt
        render_sample(key, pred, gt, args.output_dir, args.format, args.fps,
                      args.num_frames, args.fixed_shoulder_distance, image_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to a predict_csl.py output .pt file")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--keys",        nargs="+", default=None,
                        help="Specific sample IDs to render (default: random sample)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of random samples to render if --keys is not given")
    parser.add_argument("--num_frames",  type=int, default=6,
                        help="Frames per sample for --format png (evenly spaced across the sequence)")
    parser.add_argument("--format",      choices=["png", "mp4"], default="png")
    parser.add_argument("--fps",         type=int, default=5)
    parser.add_argument("--pred_only",   action="store_true",
                        help="Only plot the predicted pose, skip the ground-truth overlay")
    parser.add_argument("--fixed_shoulder_distance", type=int, default=158,
                        help="Assumed shoulder width in pixels for back-translation to pixel space")
    parser.add_argument("--image_width",  type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--seed",        type=int, default=42)
    main(parser.parse_args())
