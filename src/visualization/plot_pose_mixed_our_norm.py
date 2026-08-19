import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.utils.draw_utils import translate_back_to_pixel, POSE_CONNECTIONS, FACE_CONNECTIONS, HAND_CONNECTIONS

def plot_pose(
    gen_pose: torch.Tensor,
    gt_pose: torch.Tensor = None,
    s: int = 50,
    lw: int = 5,
    fixed_shoulder_distance: int = 86,
    image_shape: dict = {"width": 210, "height": 260},
    include_face: bool = True,
):
    """
    Plot 2D (X-Y) generated pose, optionally overlaid with ground truth,
    converting from normalized to pixel space.

    Parameters
    ----------
    gen_pose : torch.Tensor
        Normalized generated pose (NUM_PTS x 3), where only X and Y are used.
    gt_pose : torch.Tensor, optional
        Normalized ground truth pose (NUM_PTS x 3), where only X and Y are used.
        If None, only the generated pose is drawn.
    s : int, default: 50
        Size of the scatter points.
    lw : int, default: 5
        Line width for the connections.
    fixed_shoulder_distance : int, default: 86
        The assumed fixed shoulder width in pixels for back transformation.
    image_shape : dict, default: {"width": 210, "height": 260}
        The image dimensions used for back translation.
    include_face : bool, default: True
        Whether to include face connections in the visualization.
    """

    # Convert normalized keypoints to pixel space
    gen_pose_px = translate_back_to_pixel(gen_pose, fixed_shoulder_distance, image_shape)
    gt_pose_px = translate_back_to_pixel(gt_pose, fixed_shoulder_distance, image_shape) if gt_pose is not None else None

    fig = plt.figure(figsize=(13.0, 20.0))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')

    if gt_pose_px is not None:
        ax.scatter(gt_pose_px[:, 0], gt_pose_px[:, 1], color="red", marker="o", s=s, label="GT")
    ax.scatter(gen_pose_px[:, 0], gen_pose_px[:, 1], color="blue", marker="o", s=s, label="Generated")

    connection_groups = [(POSE_CONNECTIONS, 0), (HAND_CONNECTIONS, 8), (HAND_CONNECTIONS, 29)]
    if include_face:
        connection_groups.append((FACE_CONNECTIONS, 50))

    try:
        for connections, offset in connection_groups:
            for (x, y) in connections:
                x, y = x + offset, y + offset
                if gt_pose_px is not None:
                    ax.plot([gt_pose_px[x, 0], gt_pose_px[y, 0]], [gt_pose_px[x, 1], gt_pose_px[y, 1]], color="red", linewidth=lw)
                ax.plot([gen_pose_px[x, 0], gen_pose_px[y, 0]], [gen_pose_px[x, 1], gen_pose_px[y, 1]], color="blue", linewidth=lw)
    except Exception as e:
        print(f"Error in plotting connections: {e}")

    ax.axis("off")
    ax.grid(False)
    ax.invert_yaxis()

    ax.set_adjustable("datalim")
    if gt_pose_px is not None:
        ax.legend()

    return fig
