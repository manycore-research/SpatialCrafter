import os
import sys

from regex import T
sys.path.append('.')
sys.path.append('..')
import gc
import time
import argparse
import logging
from tqdm import tqdm
import os.path as osp
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

import torch
import trimesh
import numpy as np
import pandas as pd
from einops import rearrange
from torchvision import transforms
from transformers.trainer_utils import set_seed

from data_engine.dinov2.model import DINOv2
from data_engine.re10k.re10k import sample_perspective_data
from utils.typing import *
from utils.filter_util import edge_filter
from utils.file_util import load_from_csv
from utils.pose_util import convert_pose_cv_to_blender
from utils.geo_util import rgbd_to_pointcloud, batch_unproject_depth_to_pointcloud

# camera sampling
MIN_SAMPLE_VIEWS = 16
MIN_CONDITION_VIEWS = 8
IMAGE_SIZE = 512
# DINO preprocess
DINO_IMG_SIZE = 518
DINO_DOWNSAMPLE_RATE = 14
DINO_PATCH_SIZE = DINO_IMG_SIZE // DINO_DOWNSAMPLE_RATE
transform_totensor = transforms.Compose(
    [
        transforms.Resize((DINO_IMG_SIZE, DINO_IMG_SIZE), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transform_topil = transforms.ToPILImage()
# voxel grid resolution
VOXEL_RES = 64


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SpatialGen data for Spatial3D")
    parser.add_argument(
        "--dataset_rootpath",
        type=str,
        default="/path/to/processed_re10k_train_data",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="/path/to/processed_re10k_train_data/dataset_stat.csv",
    )
    parser.add_argument(
        "--dinov2_ckpt_path",
        type=str,
        default="/path/to/models/dinov2/dinov2_vitl14_reg4_pretrain.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/debug_process_re10k_trellis_data_0123",
    )
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--text_cond", action="store_true", help="whether to prepare text conditional data")
    parser.add_argument("--img_cond", action="store_true", help="whether to prepare image conditional data")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def voxelize_features(
    points: Float[np.ndarray, "Npp 3"],
    features: Float[np.ndarray, "Nv H W C"],
    x_indices: List[Float[np.ndarray, "Np"]],
    y_indices: List[Float[np.ndarray, "Np"]],
    scene_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scene_scale: float = 1.0,
    batch_point_size: int = 10_000,
    voxel_res: int = 64,
    image_size: int = 512,
    debug_cond_view: bool = False,
):
    """
    Voxelizes the point cloud and DINO features.

    Args:
        points (np.ndarray): The point cloud, shape (NvxHxW, 3).
        features (np.ndarray): The DINO features, shape (Nv, H//14, W//14, C).
        x_indices (List[np.ndarray]): The x indices of the valid image pixels, shape (Nv, HxW).
        y_indices (List[np.ndarray]): The y indices of the valid image pixels, shape (Nv, HxW).
        batch_point_size (int): The maximum number of points per batch.
        voxel_res (int): The resolution of the voxel grid.
    """
    voxel_feats = np.zeros((voxel_res, voxel_res, voxel_res, features.shape[-1]), dtype=np.float32)
    voxel_visits = np.zeros((voxel_res, voxel_res, voxel_res), dtype=np.int32)

    # Create arrays for indexing
    n_views = features.shape[0]

    image_indices = np.repeat(np.arange(n_views), [len(x) for x in x_indices]).astype(np.uint16)
    flat_x_indices = np.concatenate(x_indices).astype(np.uint16)
    flat_y_indices = np.concatenate(y_indices).astype(np.uint16)
    # print(f"image_indices shape: {image_indices.shape}, flat_x_indices shape: {flat_x_indices.shape}, flat_y_indices shape: {flat_y_indices.shape}")

    # normalize the point cloud to [-0.5, 0.5]
    points = (points - scene_center) / scene_scale
    # convert cv coordinate to our coordinate, swap y and z, and flip the new z to make +z upward
    our2cv_transform=np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    cv2our_transform=np.linalg.inv(our2cv_transform)
    points = points @ cv2our_transform
    # if not debug_cond_view:
    #     import open3d as o3d
    #     o3d_points = o3d.geometry.PointCloud()
    #     o3d_points.points = o3d.utility.Vector3dVector(points)
    #     # downsample for visualization
    #     o3d_points = o3d_points.voxel_down_sample(voxel_size=0.005)
    #     o3d.io.write_point_cloud("debug_global_normalized_points.ply", o3d_points)
    # else:
    #     import open3d as o3d
    #     o3d_points = o3d.geometry.PointCloud()
    #     o3d_points.points = o3d.utility.Vector3dVector(points)
    #     # downsample for visualization
    #     o3d_points = o3d_points.voxel_down_sample(voxel_size=0.005)
    #     o3d.io.write_point_cloud("debug_condview_normalized_points.ply", o3d_points)
    
    points = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    voxels = ((points + 0.5) * voxel_res).astype(np.int32)
    voxels = np.clip(voxels, 0, voxel_res - 1)

    masks = np.zeros((n_views, image_size, image_size), dtype=bool)

    num_points = len(voxels)
    for i in range(0, num_points, batch_point_size):
        end = min(i + batch_point_size, num_points)

        # Get the current batch of voxels, and features
        batch_voxels = voxels[i:end]
        batch_image_indices = image_indices[i:end]
        batch_x_indices = flat_x_indices[i:end]
        batch_y_indices = flat_y_indices[i:end]

        masks[batch_image_indices, batch_y_indices, batch_x_indices] = True

        batch_features = features[
            batch_image_indices,
            batch_y_indices // DINO_DOWNSAMPLE_RATE,
            batch_x_indices // DINO_DOWNSAMPLE_RATE,
        ]

        # Aggregate features at the corresponding coordinates
        voxel_feats[batch_voxels[:, 0], batch_voxels[:, 1], batch_voxels[:, 2]] += batch_features
        voxel_visits[batch_voxels[:, 0], batch_voxels[:, 1], batch_voxels[:, 2]] += 1

    # Normalize the voxelized features by the number of visits
    voxel_feats /= voxel_visits[..., np.newaxis].clip(min=1)

    voxels = np.argwhere(voxel_visits > 0)
    voxel_feats = voxel_feats[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    if not debug_cond_view:
        print(f"Global voxels shape: {voxels.shape}, voxel_feats shape: {voxel_feats.shape}, masks shape: {masks.shape}")
    else:
        print(f"Condition view voxels shape: {voxels.shape}, voxel_feats shape: {voxel_feats.shape}, masks shape: {masks.shape}")

    return voxels, voxel_feats, masks



def process_single_room(
    room_idx: int,
    split_dataframe: pd.DataFrame,
    data_root_path: str = None,
    dino_model: torch.nn.Module = None,
    device: torch.device = torch.device("cuda:0"),
    min_sample_views: int = 24,
    min_cond_views: int = 6,
    batch_size: int = 16,
    output_dir: str = None,
    b_use_cond_image_voxels: bool = True,
):
    row = split_dataframe.iloc[room_idx]
    room_uid = row["scene_id"]

    set_seed(room_idx)

    DEPTH_SCALE = 1.0
    result_name = room_uid

    os.makedirs(output_dir, exist_ok=True)
    output_features_dir = os.path.join(output_dir, "features")
    output_voxels_dir = os.path.join(output_dir, "voxels")
    output_rgbs_dir = os.path.join(output_dir, "rgbs")
    output_depths_dir = os.path.join(output_dir, "depths")
    output_cams_dir = os.path.join(output_dir, "world_cameras")
    output_conds_dir = os.path.join(output_dir, "renders_cond")
    output_cond_voxels_dir = os.path.join(output_dir, "cond_features")
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_voxels_dir, exist_ok=True)
    os.makedirs(output_rgbs_dir, exist_ok=True)
    os.makedirs(output_depths_dir, exist_ok=True)
    os.makedirs(output_cams_dir, exist_ok=True)
    os.makedirs(output_conds_dir, exist_ok=True)
    os.makedirs(output_cond_voxels_dir, exist_ok=True)
    
    feat_npz_filepath = os.path.join(output_features_dir, f"{result_name}.npz")
    voxels_npz_filepath = os.path.join(output_voxels_dir, f"{result_name}.npz")
    rgbs_npz_filepath = os.path.join(output_rgbs_dir, f"{result_name}.npz")
    depths_npz_filepath = os.path.join(output_depths_dir, f"{result_name}.npz")
    world_cams_npz_filepath = os.path.join(output_cams_dir, f"{result_name}.npz")
    cond_npz_filepath = os.path.join(output_conds_dir, f"{result_name}.npz")
    cond_feat_npz_filepath = os.path.join(output_cond_voxels_dir, f"{result_name}.npz")
    if (
        osp.exists(feat_npz_filepath)
        and osp.exists(voxels_npz_filepath)
        and osp.exists(rgbs_npz_filepath)
        and osp.exists(depths_npz_filepath)
        and osp.exists(cond_npz_filepath)
        and osp.exists(cond_feat_npz_filepath)
    ):
        print(f"Room {room_uid} has been processed, skip.")
        return {"voxel_num": 0, "img_num": 0}

    tic = time.time()
    
    try:
        target_width, target_height = (IMAGE_SIZE, IMAGE_SIZE)
        # Step 1. sample views and load data
        data_dict = sample_perspective_data(data_root_path=data_root_path,
                                sample_room_id=room_uid,
                                image_width=target_width,
                                image_height=target_height,
                                min_sample_views=min_sample_views,
                                min_cond_views=min_cond_views,
                                depth_scale=DEPTH_SCALE,
                                device=device)
        original_cam_c2ws: Float[Tensor, "Nv 4 4"] = data_dict["c2ws"].clone()  # camera poses in world coordinate ,for debug
        rgb_images: Float[Tensor, "Nv C H W"] = data_dict["images"]
        depth_images: Float[Tensor, "Nv C H W"] = data_dict["depths"]
        mask_x_indices: List[np.array] = data_dict["mask_x_indices"]
        mask_y_indices: List[np.array] = data_dict["mask_y_indices"]
        intrinsic_mat: Float[Tensor, "3 3"] = data_dict["intrinsic_mat"]
        x_fovs = data_dict["fovs"]
        y_fovs = data_dict["fovs_y"]
        scene_center = data_dict["scene_center"]
        scene_scale = data_dict["scene_scale"]
        original_cond_poses: Float[Tensor, "N_cond 4 4"] = data_dict["cond_c2ws"]
        cond_depth_images: Float[Tensor, "N_cond C H W"] = data_dict["cond_depths"]
        cond_poses = original_cond_poses.clone()
        cond_rgbs: Float[Tensor, "N_cond C H W"] = data_dict["cond_images"]
        cond_x_fovs = data_dict["cond_fovs"]
        cond_y_fovs = data_dict["cond_fovs_y"]
        
        # filter out invalid depth maps
        min_depth, max_depth = depth_images.min(), depth_images.max()
        if min_depth <= 0 or (max_depth - min_depth) < 1.0:
            print(f"Room {room_uid} has invalid depth maps, skip.")
            return {"voxel_num": 0, "img_num": 0}

        # Step 2. normalize camera poses
        if not isinstance(scene_center, torch.Tensor):
            scene_center = torch.from_numpy(scene_center)
            scene_scale = torch.tensor([scene_scale])
        num_views = original_cam_c2ws.shape[0]
        cam_c2ws = original_cam_c2ws.clone()
        pose_scale_mat = torch.eye(4).to(cam_c2ws)
        pose_scale_mat[:3, 3] = -scene_center
        pose_scale_mat[:3] *= 1.0 / scene_scale
        for id in range(num_views):
            # scale pose_c2w
            subview_pose = pose_scale_mat @ cam_c2ws[id]
            R_c2w = (subview_pose[:3, :3]).detach().cpu().numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            subview_pose[:3, :3] = torch.from_numpy(R_c2w).to(cam_c2ws)
            # convert to our coordinate system (x-right, y-forward, z-up) to align with TRELLIS
            our2cv_transform = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32).to(cam_c2ws)
            cv2ours_transform = torch.inverse(our2cv_transform)
            subview_pose = cv2ours_transform @ subview_pose
            cam_c2ws[id] = subview_pose
        # # visualize the camera poses
        # from utils.vis_util import vis_camera_keyframe
        # import open3d as o3d
        # for i, cam_c2w in enumerate(cam_c2ws):
        #     cam_pose = cam_c2w.detach().cpu().numpy()
        #     # create a line set for visualization
        #     cam_lines = vis_camera_keyframe(fx=128, fy=128, width=256, height=256, c2w=cam_pose)
        #     o3d.io.write_line_set(f"{output_dir}/debug_cam_{i}.ply", cam_lines)
        for id in range(cond_poses.shape[0]):
            subview_pose = pose_scale_mat @ cond_poses[id]
            R_c2w = (subview_pose[:3, :3]).detach().cpu().numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            subview_pose[:3, :3] = torch.from_numpy(R_c2w).to(cond_poses)
            # convert cv-coordinate system to our coordinate system (x-right, y-forward, z-up) to align with TRELLIS
            our2cv_transform = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32).to(cam_c2ws)
            cv2ours_transform = torch.inverse(our2cv_transform)
            subview_pose = cv2ours_transform @ subview_pose
            cond_poses[id] = subview_pose


        # Step 3: extract DINO feature
        dino_feat_lst = []
        for i in range(0, num_views, batch_size):
            batch_rgbs: Float[Tensor, "B C H W"] = rgb_images[i : i + batch_size].cuda()
            batch_rgbs: Float[Tensor, "B C H W"] = transform_totensor(batch_rgbs)
            bs = batch_rgbs.shape[0]

            features: Float[Tensor, "B N C"] = dino_model(batch_rgbs)
            features = features.reshape(bs, DINO_PATCH_SIZE, DINO_PATCH_SIZE, features.shape[-1])
            # print(f"    Extracted DINO features for batch {i // batch_size}, features shape: {features.shape}")
            dino_feat_lst.append(features.detach().cpu().numpy())
            del features

        # Step 4: vooxelize the pointcloud and DINO feature
        dino_feats = np.concatenate(dino_feat_lst, axis=0)
        occu_voxel_coords, occu_voxel_feats, masks = voxelize_features(
            points=data_dict["pointclouds"].cpu().numpy(),
            features=dino_feats,
            x_indices=mask_x_indices,
            y_indices=mask_y_indices,
            scene_center=scene_center.cpu().numpy(),
            scene_scale=scene_scale.cpu().numpy(),
            voxel_res=VOXEL_RES,
            image_size=IMAGE_SIZE,
            debug_cond_view=False,
        )
        # # save the voxels for debugging
        # import open3d as o3d
        # voxel_pcd = o3d.geometry.PointCloud()
        # voxel_pcd.points = o3d.utility.Vector3dVector(occu_voxel_coords.astype(np.float32))
        # o3d.io.write_point_cloud(osp.join(output_voxels_dir, f"{result_name}_voxels.ply"), voxel_pcd)

        # return np image
        rgb_array = (rgb_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        depth_array = (depth_images.detach().cpu().permute(0, 2, 3, 1).numpy()).astype(np.float32) / scene_scale.cpu().numpy()
        cond_rgb_array = (cond_rgbs.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        del data_dict, dino_feats
        # # save the images for debugging
        # for i in range(num_views):
        #     pil_rgb = Image.fromarray(rgb_array[i])
        #     pil_rgb.save(osp.join(output_rgbs_dir, f"frame_{i:03d}.png"))
        #     depth_image = depth_array[i, :, :, 0] * scene_scale.cpu().numpy() * 1000.0
        #     Image.fromarray(depth_image.astype(np.uint16)).save(osp.join(output_depths_dir, f"frame_{i:03d}.png"))
        # convert cam_poses to Blender format
        new_cam_c2ws = convert_pose_cv_to_blender(cam_c2ws)

        # convert cond_poses to Blender format
        new_cond_c2ws = convert_pose_cv_to_blender(cond_poses)
        np.savez_compressed(
            file=cond_npz_filepath,
            renders_cond=cond_rgb_array,
            transforms=new_cond_c2ws,
            x_angles=cond_x_fovs,
        )
        
        # save the results, align with TRELLIS
        np.savez_compressed(
            file=feat_npz_filepath,
            indices=occu_voxel_coords.astype(np.int32),
            patchtokens=occu_voxel_feats,
            masks=masks,
            transforms=new_cam_c2ws,
            x_angles=x_fovs,
            y_angles=y_fovs,
        )
        np.savez_compressed(
            file=voxels_npz_filepath,
            indices=occu_voxel_coords.astype(np.int32),
        )
        np.savez_compressed(
            file=rgbs_npz_filepath,
            rgbs=rgb_array,
        )
        np.savez_compressed(
            file=depths_npz_filepath,
            depths=depth_array,
        )
        np.savez_compressed(
            file=world_cams_npz_filepath,
            transforms=original_cam_c2ws.detach().cpu().numpy(),
            scene_center=scene_center.cpu().numpy(),
            scene_scale=scene_scale.cpu().numpy(),
        )
        # prepare condition voxel_coords, features, and image masks
        if b_use_cond_image_voxels:
            cond_voxels, cond_voxel_feats, cond_masks = [], [], []
            
            scene_scale = scene_scale.cpu().numpy()
            scene_center = scene_center.cpu().numpy()
            
            for cond_idx, (cond_rgb, cond_c2w_pose, cond_depth) in enumerate(zip(cond_rgb_array, original_cond_poses, cond_depth_images)):
                cond_rgb: np.ndarray
                cond_c2w_pose: Float[Tensor, "4 4"]
                cond_depth: Float[Tensor, "C H W"]
                # Step 3: Generate world points from depth map
                cond_c2w_poses = cond_c2w_pose.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
                cond_intrinsics = intrinsic_mat.unsqueeze(0)              # (1, 3, 3)
                depth_maps: Float[Tensor, "Nv 1 H W"] = cond_depth.unsqueeze(0)  # (1, 1, H, W)
                if depth_maps.shape[1] != target_height or depth_maps.shape[2] != target_width:
                    depth_maps = torch.nn.functional.interpolate(depth_maps, 
                                                                size=(target_height, target_width), 
                                                                mode='nearest')  # (1, 1, H, W)
                pointcloud: Float[Tensor, "1 Nv Np 3"] = batch_unproject_depth_to_pointcloud(
                    depth_maps=depth_maps[..., None].to(device),
                    c2w_poses=cond_c2w_poses.to(device),
                    intrinsic_mat=cond_intrinsics.to(device),
                )
                # filter out invalid depth and edge
                subview_masks = [torch.logical_and(depth_map > 0, depth_map < 100.0).squeeze() for depth_map in depth_maps]
                subview_edge_masks = [ edge_filter(depth_map.squeeze().cpu().numpy(), times=0.3) for depth_map in depth_maps]
                subview_masks = [torch.logical_and(subview_masks[i], torch.from_numpy(1 - subview_edge_masks[i])) for i in range(len(subview_masks))]
                subview_mask_indices = [torch.where(mask) for mask in subview_masks]
                subview_mask_xs = [m[1].detach().cpu().numpy() for m in subview_mask_indices]  # List[Nv, (HxW)]
                subview_mask_ys = [m[0].detach().cpu().numpy() for m in subview_mask_indices]  # List[Nv, (HxW)]
                subview_masks: Float[Tensor, "Nv Np"] = rearrange(torch.stack(subview_masks), "Nv H W -> Nv (H W)")
                valid_pointcloud: Float[Tensor, "Npp 3"] = pointcloud[0, subview_masks]

                images: Float[Tensor, "B C H W"] = torch.from_numpy(cond_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                valid_points: Float[Tensor, "Np 3"] = valid_pointcloud
                
                # Step 5: extract DINO features from the input images
                dino_feat_lst = []
                for i in range(images.shape[0]):
                    batch_rgbs: Float[Tensor, "B C H W"] = images[i:i+1].cuda()
                    batch_rgbs: Float[Tensor, "B C H W"] = transform_totensor(batch_rgbs)
                    bs = batch_rgbs.shape[0]

                    features: Float[Tensor, "B N C"] = dino_model(batch_rgbs)
                    features = features.reshape(bs, DINO_PATCH_SIZE, DINO_PATCH_SIZE, features.shape[-1])
                    # print(f"    Extracted DINO features for batch {i }, features shape: {features.shape}")
                    dino_feat_lst.append(features.detach().cpu().numpy())
                    del features
                    
                # Step 6: voxelize the pointcloud and DINO feature
                dino_feats = np.concatenate(dino_feat_lst, axis=0)
                occu_voxel_coords, occu_voxel_feats, masks = voxelize_features(
                    points=valid_points.cpu().numpy(),
                    features=dino_feats,
                    x_indices=subview_mask_xs,
                    y_indices=subview_mask_ys,
                    scene_center=scene_center,
                    scene_scale=scene_scale,
                    voxel_res=VOXEL_RES,
                    image_size=IMAGE_SIZE,
                    debug_cond_view=True,
                )
                cond_voxels.append(occu_voxel_coords.astype(np.int32))
                cond_voxel_feats.append(occu_voxel_feats)
                cond_masks.append(masks)
                # # save the voxels for debugging
                # import open3d as o3d
                # voxel_pcd = o3d.geometry.PointCloud()
                # voxel_pcd.points = o3d.utility.Vector3dVector(occu_voxel_coords.astype(np.float32))
                # o3d.io.write_point_cloud(osp.join(output_voxels_dir, f"{result_name}_cond_voxels.ply"), voxel_pcd)
                
        
            del dino_feats
            
            cond_voxels_obj = cond_voxels
            cond_voxel_feats_obj = cond_voxel_feats
            cond_masks = np.stack(cond_masks, axis=0)
            # Note: cond_voxels and cond_voxel_feats might vary in shape, so using np.array(..., dtype=object) or saving via pickle payload is necessary.
            cond_voxels = np.array(cond_voxels_obj, dtype=object)
            cond_voxel_feats = np.array(cond_voxel_feats_obj, dtype=object)
            
            np.savez_compressed(
                file=cond_feat_npz_filepath,
                indices=cond_voxels,
                patchtokens=cond_voxel_feats,
                masks=cond_masks,
            )
        toc = time.time()
        return {"voxel_num": occu_voxel_coords.shape[0], "img_num": num_views}
        
    except Exception as e:
        print(f"Error in loading room {room_uid}: {e}")
        return {"voxel_num": 0, "img_num": 0}



if __name__ == "__main__":
    # 设置多进程启动方法（推荐用于解决CUDA相关问题）
    mp.set_start_method("spawn", force=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger_trimesh = logging.getLogger("trimesh")
    logger_trimesh.setLevel(logging.ERROR)
    logger_pil = logging.getLogger("PIL")
    logger_pil.setLevel(logging.ERROR)

    args = parse_args()
    data_root_path = args.dataset_rootpath
    split_file = args.split_file
    output_dir = args.output_dir
    dinov2_ckpt_path = args.dinov2_ckpt_path
    start_index = args.start_index
    end_index = args.end_index
    batch_size = args.batch_size
    cpu_num_workers = args.num_workers
    do_text_condition = args.text_cond
    do_image_condition = args.img_cond
    VOXEL_RES = args.voxel_resolution

    if not os.path.exists(split_file):
        logger.info(f"file {split_file} not exist!")
        exit(-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
    # load split
    scenes_df = load_from_csv(split_file, start_index, end_index)
    if scenes_df is None or len(scenes_df) == 0:
        print("No scenes to process")
        exit(0)
    logger.info(f"Task {args.task_id} starts from {args.start_index} to {args.end_index}. Voxel resolution: {VOXEL_RES}, Text cond: {do_text_condition}, Image cond: {do_image_condition}")

    # filter out the room_type is balcony
    logger.info(f"After filtering, {len(scenes_df)} rooms left.")

    # load dinov2 model
    dinov2_model = DINOv2(dinov2_ckpt_path)
    dinov2_model.to(device)
    
    # process_single_room(
    #     room_idx=0,
    #     split_dataframe=scenes_df,
    #     data_root_path=data_root_path,
    #     min_sample_views=MIN_SAMPLE_VIEWS,
    #     min_cond_views=MIN_CONDITION_VIEWS,
    #     dino_model=dinov2_model,  # dinov2 or dinov3
    #     batch_size=batch_size,
    #     device=device,
    #     output_dir=output_dir,
    #     b_use_cond_image_voxels=True,
    # )

    per_room_voxel_num, per_room_img_num = [], []
    with Pool(processes=cpu_num_workers) as pool:
        # Create a partial function with fixed arguments
        process_room_partial = partial(
            process_single_room,
            split_dataframe=scenes_df,
            data_root_path=data_root_path,
            min_sample_views=MIN_SAMPLE_VIEWS,
            min_cond_views=MIN_CONDITION_VIEWS,
            dino_model=dinov2_model,  # dinov2 or dinov3
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            b_use_cond_image_voxels=True,
        )

        # Use tqdm to show progress
        with tqdm(total=len(scenes_df)) as pbar:
            for ret_dict in pool.imap_unordered(process_room_partial, range(len(scenes_df))):
                voxel_num, img_num = ret_dict["voxel_num"], ret_dict["img_num"]
                per_room_voxel_num.append(voxel_num)
                per_room_img_num.append(img_num)
                pbar.update(1)
    # voxel statistics
    logger.info(f"Voxel statistics: {np.mean(per_room_voxel_num):.2f}, max {np.max(per_room_voxel_num)}, min {np.min(per_room_voxel_num)}")
    logger.info(f"Image statistics: {np.mean(per_room_img_num):.2f}, max {np.max(per_room_img_num)}, min {np.min(per_room_img_num)}")
    # 程序结束时的额外清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()