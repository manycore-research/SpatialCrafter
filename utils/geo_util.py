import torch
import numpy as np
import open3d as o3d
from torch import Tensor
from einops import rearrange
import torch.nn.functional as tF

from utils.typing import *

def normalize_normals(normals: Tensor, C2W: Tensor, i: int = 0) -> Tensor:
    """Normalize a batch of multi-view `normals` by the `i`-th view.

    Inputs:
        - `normals`: (B, V, 3, H, W)
        - `C2W`: (B, V, 4, 4)
        - `i`: the index of the view to normalize by

    Outputs:
        - `normalized_normals`: (B, V, 3, H, W)
    """
    _, _, R, C = C2W.shape  # (B, V, 4, 4)
    assert R == C == 4
    _, _, CC, _, _ = normals.shape  # (B, V, 3, H, W)
    assert CC == 3

    dtype = normals.dtype
    normals = normals.clone().float()
    transform = torch.inverse(C2W[:, i, :3, :3])  # (B, 3, 3)

    return torch.einsum("brc,bvchw->bvrhw", transform, normals).to(dtype)  # (B, V, 3, H, W)


def normalize_C2W(C2W: Tensor, i: int = 0, norm_radius: float = 0.0) -> Tensor:
    """Normalize a batch of multi-view `C2W` by the `i`-th view.

    Inputs:
        - `C2W`: (B, V, 4, 4)
        - `i`: the index of the view to normalize by
        - `norm_radius`: the normalization radius

    Outputs:
        - `normalized_C2W`: (B, V, 4, 4)
    """
    _, _, R, C = C2W.shape  # (B, V, 4, 4)
    assert R == C == 4

    device, dtype = C2W.device, C2W.dtype
    C2W = C2W.clone().float()

    if abs(norm_radius) > 0.0:
        radius = torch.norm(C2W[:, i, :3, 3], dim=1)  # (B,)
        C2W[:, :, :3, 3] *= norm_radius / radius.unsqueeze(1).unsqueeze(2)

    # The `i`-th view is normalized to a canonical matrix as the reference view
    transform = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, norm_radius], [0, 0, 0, 1]], dtype=torch.float32, device=device  # canonical c2w in OpenGL world convention
    ) @ torch.inverse(
        C2W[:, i, ...]
    )  # (B, 4, 4)

    return (transform.unsqueeze(1) @ C2W).to(dtype)  # (B, V, 4, 4)


def unproject_depth(depth_map: Tensor, C2W: Tensor, fxfycxcy: Tensor) -> Tensor:
    """Unproject depth map to 3D world coordinate.

    Inputs:
        - `depth_map`: (B, V, H, W)
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `xyz_world`: (B, V, 3, H, W)
    """
    device, dtype = depth_map.device, depth_map.dtype
    if len(depth_map.shape) == 2:
        depth_maps = depth_map[None, None, :, :]
        C2Ws = C2W[None, None, :, :]
        fxfycxcys = fxfycxcy[None, None, :]
    elif len(depth_map.shape) == 3:
        depth_maps = depth_map[None, :, :, :]
        C2Ws = C2W[None, :, :, :]
        fxfycxcys = fxfycxcy[None, :, :]
           
    B, V, H, W = depth_maps.shape

    depth_maps = depth_maps.reshape(B * V, H, W).float()
    C2Ws = C2Ws.reshape(B * V, 4, 4).float()
    fxfycxcys = fxfycxcys.reshape(B * V, 4).float()
    K = torch.zeros(B * V, 3, 3, dtype=torch.float32, device=device)
    K[:, 0, 0] = fxfycxcy[:, 0]
    K[:, 1, 1] = fxfycxcy[:, 1]
    K[:, 0, 2] = fxfycxcy[:, 2]
    K[:, 1, 2] = fxfycxcy[:, 3]
    K[:, 2, 2] = 1

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    y = y.to(device).unsqueeze(0).repeat(B * V, 1, 1) / (H - 1)
    x = x.to(device).unsqueeze(0).repeat(B * V, 1, 1) / (W - 1)
    # NOTE: To align with `plucker_ray(bug=False)`, should be:
    # y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
    # x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
    xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_maps[..., None]
    xyz = xyz_map.view(B * V, -1, 3)

    # Get point positions in camera coordinate
    xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 1, 2))
    xyz_map = xyz.view(B * V, H, W, 3)

    # Transform pts from camera to world coordinate
    xyz_homo = torch.ones((B * V, H, W, 4), device=device)
    xyz_homo[..., :3] = xyz_map
    xyz_world = torch.bmm(C2Ws, xyz_homo.reshape(B * V, -1, 4).permute(0, 2, 1))[:, :3, ...].to(dtype)  # (B*V, 3, H*W)
    xyz_world = xyz_world.reshape(B, V, 3, H, W)
    # elif len(depth_map.shape) == 2:
    #     H, W = depth_map.shape

    #     K = torch.zeros(3, 3, dtype=torch.float32, device=device)
    #     K[0, 0] = fxfycxcy[0]
    #     K[1, 1] = fxfycxcy[1]
    #     K[0, 2] = fxfycxcy[2]
    #     K[1, 2] = fxfycxcy[3]
    #     K[2, 2] = 1
    #     y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    #     y = y.to(device).unsqueeze(0) / (H - 1)
    #     x = x.to(device).unsqueeze(0) / (W - 1)
    #     xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map[..., None]
    #     xyz = xyz_map.view(-1, 3)

    #     # Get point positions in camera coordinate
    #     xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 0, 1))
    #     xyz_map = xyz.view(H, W, 3)

    #     # Transform pts from camera to world coordinate
    #     xyz_homo = torch.ones((H, W, 4), device=device)
    #     xyz_homo[..., :3] = xyz_map
    #     xyz_world = (C2W @ xyz_homo.reshape(-1, 4).permute(1, 0))[:3, ...].to(dtype)  # (3, H*W)
    #     xyz_world = xyz_world.reshape(3, H, W)
    
    if len(depth_map.shape) == 2:
        xyz_world = xyz_world.squeeze()
    elif len(depth_map.shape) == 3:
        xyz_world = xyz_world.squeeze(0)
    return xyz_world


def unproject_depth_np(depth_map: np.ndarray, c2w: np.ndarray, intrinsic_mat: np.ndarray) -> np.ndarray:
    """
    Unproject a depth map to 3D world coordinates.

    Args:
        depth_map (np.ndarray): The depth map of shape (H, W).
        c2w (np.ndarray): The camera-to-world transformation matrix of shape (4, 4).
        intrinsic_mat (np.ndarray): The camera intrinsic matrix of shape (3, 3).

    Returns:
        np.ndarray: The 3D points in world coordinates of shape (3, H, W).
    """
    
    H, W = depth_map.shape
    n_pts = H * W
    depth_map = depth_map.reshape(1, H, W)

    pts_x = np.linspace(0, W, W)
    pts_y = np.linspace(0, H, H)

    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)
    pts = np.stack((pts_xx, pts_yy, np.ones_like(pts_xx)), axis=0)
    pts = np.linalg.inv(intrinsic_mat) @ (pts * depth_map).reshape(3, n_pts)
    
    pts_world = c2w[:3, :3] @ pts + c2w[:3, 3:4]
    return pts_world.reshape(3, H, W)

def plucker_ray(h: int, w: int, C2W: Tensor, fxfycxcy: Tensor, bug: bool = True) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Get Plucker ray embeddings.

    Inputs:
        - `h`: image height
        - `w`: image width
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `plucker`: (B, V, 6, `h`, `w`)
        - `ray_o`: (B, V, 3, `h`, `w`)
        - `ray_d`: (B, V, 3, `h`, `w`)
    """
    device, dtype = C2W.device, C2W.dtype
    B, V = C2W.shape[:2]

    C2W = C2W.reshape(B * V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B * V, 4).float()

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")  # OpenCV/COLMAP camera convention
    y, x = y.to(device), x.to(device)
    if bug:  # BUG !!! same here: https://github.com/camenduru/GRM/blob/master/model/visual_encoder/vit_gs.py#L85
        y = y[None, :, :].expand(B * V, -1, -1).reshape(B * V, -1) / (h - 1)
        x = x[None, :, :].expand(B * V, -1, -1).reshape(B * V, -1) / (w - 1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    else:
        y = (y[None, :, :].expand(B * V, -1, -1).reshape(B * V, -1) + 0.5) / h
        x = (x[None, :, :].expand(B * V, -1, -1).reshape(B * V, -1) + 0.5) / w
        x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # (B*V, h*w, 3)
    ray_d = torch.bmm(ray_d, C2W[:, :3, :3].transpose(1, 2))  # (B*V, h*w, 3)
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # (B*V, h*w, 3)
    ray_o = C2W[:, :3, 3][:, None, :].expand_as(ray_d)  # (B*V, h*w, 3)

    ray_o = ray_o.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    ray_d = ray_d.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    plucker = torch.cat([torch.cross(ray_o, ray_d, dim=2).to(dtype), ray_d.to(dtype)], dim=2)

    return plucker, (ray_o, ray_d)


def orbit_camera(
    elevs: Tensor,
    azims: Tensor,
    radius: Optional[Tensor] = None,
    is_degree: bool = True,
    target: Optional[Tensor] = None,
    opengl: bool = True,
) -> Tensor:
    """Construct a camera pose matrix orbiting a target with elevation & azimuth angle.

    Inputs:
        - `elevs`: (B,); elevation in (-90, 90), from +y to -y is (-90, 90)
        - `azims`: (B,); azimuth in (-180, 180), from +z to +x is (0, 90)
        - `radius`: (B,); camera radius; if None, default to 1.
        - `is_degree`: bool; whether the input angles are in degree
        - `target`: (B, 3); look-at target position
        - `opengl`: bool; whether to use OpenGL convention

    Outputs:
        - `C2W`: (B, 4, 4); camera pose matrix
    """
    device, dtype = elevs.device, elevs.dtype

    if radius is None:
        radius = torch.ones_like(elevs)
    assert elevs.shape == azims.shape == radius.shape
    if target is None:
        target = torch.zeros(elevs.shape[0], 3, device=device, dtype=dtype)

    if is_degree:
        elevs = torch.deg2rad(elevs)
        azims = torch.deg2rad(azims)

    x = radius * torch.cos(elevs) * torch.sin(azims)
    y = -radius * torch.sin(elevs)
    z = radius * torch.cos(elevs) * torch.cos(azims)

    camposes = torch.stack([x, y, z], dim=1) + target  # (B, 3)
    R = look_at(camposes, target, opengl=opengl)  # (B, 3, 3)
    C2W = torch.cat([R, camposes[:, :, None]], dim=2)  # (B, 3, 4)
    C2W = torch.cat([C2W, torch.zeros_like(C2W[:, :1, :])], dim=1)  # (B, 4, 4)
    C2W[:, 3, 3] = 1.0
    return C2W


def look_at(camposes: Tensor, targets: Tensor, opengl: bool = True) -> Tensor:
    """Construct batched pose rotation matrices by look-at.

    Inputs:
        - `camposes`: (B, 3); camera positions
        - `targets`: (B, 3); look-at targets
        - `opengl`: whether to use OpenGL convention

    Outputs:
        - `R`: (B, 3, 3); normalized camera pose rotation matrices
    """
    device, dtype = camposes.device, camposes.dtype

    if not opengl:  # OpenCV convention
        # forward is camera -> target
        forward_vectors = tF.normalize(targets - camposes, dim=-1)
        up_vectors = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)[None, :].expand_as(forward_vectors)
        right_vectors = tF.normalize(torch.cross(forward_vectors, up_vectors), dim=-1)
        up_vectors = tF.normalize(torch.cross(right_vectors, forward_vectors), dim=-1)
    else:
        # forward is target -> camera
        forward_vectors = tF.normalize(camposes - targets, dim=-1)
        up_vectors = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)[None, :].expand_as(forward_vectors)
        right_vectors = tF.normalize(torch.cross(up_vectors, forward_vectors), dim=-1)
        up_vectors = tF.normalize(torch.cross(forward_vectors, right_vectors), dim=-1)

    R = torch.stack([right_vectors, up_vectors, forward_vectors], dim=-1)
    return R

def convert_distance_to_z(distance_map: torch.tensor, 
                          image_height: int = 256,
                          image_width: int = 256,
                          focal_length: float = 1.0) -> torch.tensor:
    """
    Convert distance map to depth map
    params:
        distance_map: [1, H, W]
        focal_length: noormlized focal length
    """
    vs, us = torch.meshgrid(torch.linspace(-1, 1, image_height), torch.linspace(-1, 1, image_width), indexing="ij")
    us = us.reshape(1, image_height, image_width)
    vs = vs.reshape(1, image_height, image_width)
    depth_cos = torch.cos(torch.atan2(torch.sqrt(us * us + vs * vs), torch.tensor(focal_length)))
    depth = distance_map * depth_cos.to(distance_map.device)

    return depth

def convert_distance_to_z_np(distance_map: np.array, 
                          image_height: int = 256,
                          image_width: int = 256,
                          focal_length: float = 1.0) -> np.array:
    """
    Convert distance map to depth map
    params:
        distance_map: [H, W]
        focal_length: noormlized focal length
    """
    vs, us = np.meshgrid(np.linspace(-1, 1, image_height), np.linspace(-1, 1, image_width), indexing="ij")
    us = us.reshape(image_height, image_width)
    vs = vs.reshape(image_height, image_width)
    depth_cos = np.cos(np.arctan2(np.sqrt(us * us + vs * vs), focal_length))
    depth = distance_map * depth_cos

    return depth

def rgbd_to_pointcloud(
    depth_image: np.array,
    c2w_pose: np.array,
    depth_scale: float = 1.0,
    intrinsic_mat: np.array = None,
    rgb_image: np.array = None,
    normal_image: np.array = None,
) -> o3d.geometry.PointCloud:
    """
    Convert RGBD image to point cloud
    params:
        depth_image: np.array, depth image
        c2w_pose: np.array, camera to world pose
        depth_scale: float, depth scale
        fl_x: float, focal length in x axis
        fl_y: float, focal length in y axis
        rgb_image: np.array, rgb image
    """

    depth_image = (depth_image * depth_scale).astype(np.float32)
    if len(depth_image.shape) == 2:
        depth_image = np.expand_dims(depth_image, axis=2)
    H, W, C = depth_image.shape
    n_pts = H * W
    # Get camera intrinsic
    if intrinsic_mat is not None:
        K = intrinsic_mat
    else:
        hfov = 90.0 * np.pi / 180.0
        fl_x = W / 2.0 / np.tan((hfov / 2.0))
        K = np.array([
                    [fl_x, 0.0, W / 2.0],
                    [0.0, fl_x, H / 2.0],
                    [0.0, 0.0, 1]]
        )

    depth_map = depth_image.reshape(1, H, W)

    pts_x = np.linspace(0, W, W)
    pts_y = np.linspace(0, H, H)

    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)
    pts = np.stack((pts_xx, pts_yy, np.ones_like(pts_xx)), axis=0)
    pts = np.linalg.inv(K) @ (pts * depth_map).reshape(3, n_pts)

    invalid_mask = depth_map <= 0
    if rgb_image is not None:
        color = (rgb_image[:, :, :3] / 255.0).clip(0.0, 1.0)
    else:
        color = np.zeros_like(pts[:3].T)

    if normal_image is not None:
        # convert normal to [-1, 1]
        normal = np.clip((normal_image + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
        normal = -normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)

        points, colors, normals = (
            np.transpose(pts)[:, :3],
            color.reshape(n_pts, 3),
            normal.reshape(n_pts, 3),
        )
    else:
        points, colors = np.transpose(pts)[:, :3], color.reshape(n_pts, 3)
        normals = np.zeros_like(points)

    o3d_ply = o3d.geometry.PointCloud()
    # valid_mask = ~invalid_mask.reshape(-1)
    # points = points[valid_mask]
    # colors = colors[valid_mask]
    # normals = normals[valid_mask]
    o3d_ply.points = o3d.utility.Vector3dVector(points)
    o3d_ply.colors = o3d.utility.Vector3dVector(colors)
    o3d_ply.normals = o3d.utility.Vector3dVector(normals)
    return o3d_ply.transform(c2w_pose)

def batch_unproject_depth_to_pointcloud(
    depth_maps: Float[Tensor, "B N H W 1"],
    c2w_poses: Float[Tensor, "B N 4 4"],
    intrinsic_mat: Float[Tensor, "B 3 3"],
) -> Float[Tensor, "B Nv Np 3"]:

    bsz, n_views, H, W = depth_maps.shape[:-1]
    batch_intrinsic_mat = intrinsic_mat[:, None, ...]

    pts_x = torch.linspace(0, W - 1, W, device=depth_maps.device)
    pts_y = torch.linspace(0, H - 1, H, device=depth_maps.device)
    pts_xx, pts_yy = torch.meshgrid(pts_x, pts_y, indexing="xy")
    pts = torch.stack((pts_xx, pts_yy, torch.ones_like(pts_xx)), dim=-1)
    pts = pts[None, None, ...].repeat(bsz, n_views, 1, 1, 1)
    pts: Float[Tensor, "B N 3 Np"] = torch.inverse(batch_intrinsic_mat) @ (pts * depth_maps).permute(0, 1, 4, 2, 3).reshape(bsz, n_views, 3, H * W)
    trans_pts = torch.einsum("BNdc, BNcp -> BNdp", c2w_poses[:, :, :3, :3], pts)
    trans_pts = trans_pts + c2w_poses[:, :, :3, 3][:, :, :, None]
    trans_pts = rearrange(trans_pts, "B N c Np -> B N Np c")

    return trans_pts
