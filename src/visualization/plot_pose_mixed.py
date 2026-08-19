import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from colorsys import hls_to_rgb
from mpl_toolkits.mplot3d import Axes3D

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.visualization.keypoint_def import JOINTS, EDGES


def get_palette(
    n: int, hue: float = 0.01, luminance: float = 0.6, saturation: float = 0.65
) -> np.array:
    hues = np.linspace(0, 1, n + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(int)
    palette = [hls_to_rgb(float(hue), luminance, saturation) for hue in hues]
    # palette = torch.tensor(palette)
    palette = np.array(palette)
    return palette


def getDimBox(points):
    """
    points: ... N x DIM
    output:	[[min_d1, max_d1], ..., [min_dN, max_dN]]
    """

    is_torch = torch.is_tensor(points[0])
    num_dim = points[0].shape[-1]
    if isinstance(points, list):
        assert is_torch or isinstance(points[0], np.ndarray)
        if is_torch:
            return np.array(
                [
                    [
                        np.median([pts[..., k].min().detach().cpu() for pts in points]),
                        np.median([pts[..., k].max().detach().cpu() for pts in points]),
                    ]
                    for k in range(num_dim)
                ]
            )
        else:
            return np.array(
                [
                    [
                        np.median([pts[..., k].min() for pts in points]),
                        np.median([pts[..., k].max() for pts in points]),
                    ]
                    for k in range(num_dim)
                ]
            )
    else:
        if is_torch:
            points = points.cpu().detach()
        if isinstance(points, np.ndarray):
            return np.array(
                [
                    [
                        np.median(points[..., k].min(-1)[0]),
                        np.median(points[..., k].max(-1)[0]),
                    ]
                    for k in range(num_dim)
                ]
            )
        else:
            # points[..., 0].shape -> [1, 178]
            return np.array(
                [
                    [
                        points[..., k].min(-1)[0].median(),
                        points[..., k].max(-1)[0].median(),
                    ]
                    for k in range(num_dim)
                ]
            )
            # array([[-0.19723222,  0.21148895],
            #    [-0.0667696 ,  0.58930105],
            #    [-0.15119821,  0.11362214]], dtype=float32)


def plot_pose(
    gen_pose: torch.Tensor,
    gt_pose: torch.Tensor = None,
    connections: list = None,
    s: int = 50,
    lw: int = 5,
):
    """
    Plot 2D (X-Y) generated pose, optionally overlaid with ground truth.

    Parameters
    ----------
    gen_pose : torch.Tensor
        Generated pose (NUM_PTS x 3), where only X and Y are used.
    gt_pose : torch.Tensor, optional
        Ground truth pose (NUM_PTS x 3), where only X and Y are used.
        If None, only the generated pose is drawn.
    connections : list, optional
        List of edges connecting keypoints.
    s : int, default: 50
        Size of the scatter points.
    lw : int, default: 5
        Line width for the connections.
    """

    fig = plt.figure(figsize=(13.0, 20.0))
    fig.patch.set_facecolor('white')  # ensure white figure background
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')         # ensure white axes background

    if gt_pose is not None:
        ax.scatter(gt_pose[:, 0], gt_pose[:, 1], color="red", marker="o", s=s, label="GT")
    ax.scatter(gen_pose[:, 0], gen_pose[:, 1], color="blue", marker="o", s=s, label="Generated")

    if connections is not None:
        try:
            for (x, y) in connections:
                if gt_pose is not None:
                    ax.plot([gt_pose[x, 0], gt_pose[y, 0]], [gt_pose[x, 1], gt_pose[y, 1]], color="red", linewidth=lw)
                ax.plot([gen_pose[x, 0], gen_pose[y, 0]], [gen_pose[x, 1], gen_pose[y, 1]], color="blue", linewidth=lw)
        except Exception as e:
            print(e)

    ax.axis("off")
    ax.grid(False)
    ax.invert_yaxis()

    ax.set_adjustable("datalim")
    if gt_pose is not None:
        ax.legend(fontsize=22)

    return fig