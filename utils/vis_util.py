from typing import *

import cv2
import torch
import torchvision
import numpy as np
from torch import Tensor
from PIL import Image
import open3d as o3d
from numpy import typing as npt
import matplotlib.pyplot as plt



def apply_depth_to_colormap(depth_map: np.ndarray, cmap: str = "viridis", near_plane: float = 0.00, far_plane: float = 15.0):
    """
    depth_map: (H, W) is the depth map.
    """
    near_plane = max(near_plane, depth_map.min())
    far_plane = min(far_plane, depth_map.max())
    depth_map = (depth_map - near_plane) / (far_plane - near_plane + 1e-6)
    depth_map = np.clip(depth_map, 0.0, 1.0)
    depth_map = plt.get_cmap(cmap)(depth_map)
    return depth_map[..., :3]


def convert_depth_to_colormap(depths: Tensor) -> Tensor:
    """
    depths: (B, N, H, W) is the depth map. [0, 1]
    Returns: (B, N, 3, H, W) is the depth map in color.
    """
    depths = depths.permute(0, 2, 3, 1).float().cpu()
    depths = depths.clip(0, 1)
    depths = depths.mean(dim=-1)
    color_depths = apply_depth_to_colormap(depths, cmap="viridis")
    return torch.from_numpy(color_depths).permute(0, 3, 1, 2)




def colorize_depth(
    depth: Union[npt.NDArray, Image.Image], depth_min: int = -1, depth_max: int = -1, output_type: str = "np"
) -> npt.NDArray:
    """
    colorize depth map using JET colormap.
    Args:
        depth: (H, W) depth map.
        depth_min: minimum depth value for normalization. If <0, use 5th percentile.
        depth_max: maximum depth value for normalization. If <0, use 95th percentile.
        output_type: "np" or "pil".
    Returns:
        depth_color: (H, W, 3) colorized depth map.
    """
    if isinstance(depth, Image.Image):
        depth = np.array(depth)

    depth = 1.0 / (depth + 1e-6)
    invalid_mask = (depth <= 0) | (depth >= 1e6) | np.isnan(depth) | np.isinf(depth)
    if depth_min < 0 or depth_max < 0:
        depth_min = np.percentile(depth[~invalid_mask], 5)
        depth_max = np.percentile(depth[~invalid_mask], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min + 1e-10)
    depth_scaled_uint8 = np.uint8(np.clip(depth_scaled * 255, 0, 255))
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_JET)
    depth_color[invalid_mask, :] = 0
    depth_color = depth_color[..., ::-1]

    if output_type == "pil":
        depth_color = Image.fromarray(depth_color.astype(np.uint8)).convert("RGB")
    return depth_color


def save_color_depth_image(depth_imgs: Tensor, cmap: str = "viridis", output_path: str = "depths.png"):
    """
    Args:
        depth_imgs: (B, H, W) is the depth map.
        cmap: colormap to use.
        output_path: path to save the image.
    """
    depth_imgs = apply_depth_to_colormap(depth_imgs, cmap=cmap)  # BxT_out,H,W,3
    depth_imgs_torch = torch.from_numpy(depth_imgs).permute(0, 3, 1, 2)
    depth_imgs_torch = torchvision.utils.make_grid(depth_imgs_torch)
    torchvision.utils.save_image(depth_imgs_torch, output_path)
    return depth_imgs

def vis_camera_keyframe(fx: float, fy: float, width: int, height: int, c2w: np.ndarray) -> o3d.geometry.LineSet:
    """
    Visualize camera keyframe as a frustum.
    Args:
        fx, fy: camera intrinsic parameters.
        width, height: image width and height.
        c2w: (4, 4) camera-to-world pose.
    Returns:
        frustum: o3d.geometry.LineSet representing the camera frustum.
    """
    cam_intrinisc = np.array([
        [fx, 0, width / 2],
        [0, fy, height / 2],
        [0, 0, 1]
    ])
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(width, height,
        cam_intrinisc[0, 0], cam_intrinisc[1, 1], cam_intrinisc[0, 2],  cam_intrinisc[1, 2],
    )
    cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(c2w), scale=0.05)
    cam_lines.paint_uniform_color([1, 0, 0])
    return cam_lines