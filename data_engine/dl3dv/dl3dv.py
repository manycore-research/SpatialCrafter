import os
import glob
import sys
sys.path.append('.')
sys.path.append('..')
import os.path as osp

import torch
import pyexr
import imageio
import zipfile
import tempfile
import open3d as o3d
from PIL import Image
from einops import rearrange

from utils.typing import *
from utils.img_util import resize_pil_image
from utils.geo_util import batch_unproject_depth_to_pointcloud
from utils.filter_util import edge_filter
from data_engine.spatialgen.datasets.spatialgen import sample_camera_positions

DL3DV_DEPTH_MAX = 30.0 # in meters

def load_dl3dv_images(images_dir: str, image_type: str = "rgb", target_width: int = 512, target_height: int = 512) -> Float[Tensor, "N H W C"]:
    """
    Load RGB or depth images from a directory.
    
    Args:
        images_dir (str): Directory containing the images.
        image_type (str): Type of images to load ("rgb" or "depth").
    Returns:
        List of loaded images as numpy arrays.
    """
    file_suffix = ".exr" if image_type == "depth" else ".png"
    image_files = [f for f in os.listdir(images_dir) if f.endswith(file_suffix)]
    
    if image_type == "rgb":
        image_files.sort(key=lambda x: int(x.split('_')[1].split(".")[0]))
    else:
        image_files.sort(key=lambda x: int(x.split(".")[0]))

    images = []
    for file_name in image_files:
        img_fp = os.path.join(images_dir, file_name)
        if image_type == "rgb":
            img = resize_pil_image(img=Image.open(img_fp).convert("RGB"), 
                                   target_height=target_height, 
                                   target_width=target_width,
                                   resampling_mode=Image.BILINEAR,
                                   target_aspect_ratio=target_width / target_height,
                                   )
            img = np.clip(np.array(img) / 255.0, 0.0, 1.0)
        elif image_type == "depth":
            img = np.array(pyexr.read(img_fp).astype(np.float32))[:,:,0] * 4000.0
            depth_pil = Image.fromarray(img.astype(np.int32))
            depth_pil = resize_pil_image(img=depth_pil,
                                        target_height=target_height, 
                                        target_width=target_width,
                                        resampling_mode=Image.NEAREST,
                                        target_aspect_ratio=target_width / target_height,
                                        )
            img = np.array(depth_pil).astype(np.float32) / 4000.0
            img = np.expand_dims(img, axis=-1)  # (H, W
        else:
            raise ValueError(f"Unknown image_type: {image_type}")
        images.append(img)
    
    images = np.stack(images, axis=0)  # (N, H, W, C)
    images = torch.from_numpy(images).float()
    return images

def load_dl3dv_images(images_dir: str, image_type: str = "rgb", target_width: int = 512, target_height: int = 512) -> Float[Tensor, "N H W C"]:
    """
    Load RGB or depth images from a directory.
    
    Args:
        images_dir (str): Directory containing the images.
        image_type (str): Type of images to load ("rgb" or "depth").
    Returns:
        List of loaded images as numpy arrays.
    """
    if osp.isdir(images_dir):
        file_suffix = ".exr"
        image_files = [f for f in os.listdir(images_dir) if f.endswith(file_suffix)]
    elif osp.isfile(images_dir):
        # rgb viideo file
        assert image_type == "rgb", "Only rgb video file is supported."
        rgb_video_reader = imageio.get_reader(images_dir)
    
    if image_type == "rgb":
        image_files = list(range(int(rgb_video_reader.count_frames())))
    else:
        image_files.sort(key=lambda x: int(x.split(".")[0]))

    images = []
    for file_name in image_files:
        img_fp = os.path.join(images_dir, str(file_name))
        if image_type == "rgb":
            frame = rgb_video_reader.get_data(file_name)
            img = resize_pil_image(img=Image.fromarray(frame), 
                                   target_height=target_height, 
                                   target_width=target_width,
                                   resampling_mode=Image.BILINEAR,
                                   target_aspect_ratio=target_width / target_height,
                                   )
            img = np.clip(np.array(img) / 255.0, 0.0, 1.0)
        elif image_type == "depth":
            img = np.array(pyexr.read(img_fp).astype(np.float32))[:,:,0] * 4000.0
            depth_pil = Image.fromarray(img.astype(np.int32))
            depth_pil = resize_pil_image(img=depth_pil,
                                        target_height=target_height, 
                                        target_width=target_width,
                                        resampling_mode=Image.NEAREST,
                                        target_aspect_ratio=target_width / target_height,
                                        )
            img = np.array(depth_pil).astype(np.float32) / 4000.0
            img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        else:
            raise ValueError(f"Unknown image_type: {image_type}")
        images.append(img)
    
    images = np.stack(images, axis=0)  # (N, H, W, C)
    images = torch.from_numpy(images).float()
    return images

def load_dl3dv_depths_from_zip(zip_file: str, target_width: int = 512, target_height: int = 512) -> Float[Tensor, "N H W C"]:
    """
    Load depth images from a zip file containing .exr depth images.
    
    Args:
        images_dir (str): Directory containing the images.
        image_type (str): Type of images to load ("depth").
    Returns:
        List of loaded images as numpy arrays.
    """
    assert zip_file.endswith(".zip"), "Input file must be a zip file."
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.endswith('.exr')]
        image_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))

        images = []
        for file_name in image_files:
            with zip_ref.open(file_name) as img_file:
                with tempfile.NamedTemporaryFile(suffix=".exr") as tmp_file:
                    tmp_file.write(img_file.read())
                    tmp_file.flush()
                    img = np.array(pyexr.read(tmp_file.name).astype(np.float32))[:,:,0]
                if img.shape[0] != target_height or img.shape[1] != target_width:
                    depth_pil = Image.fromarray((img * 4000.0).astype(np.int32))
                    depth_pil = resize_pil_image(img=depth_pil,
                                                target_height=target_height, 
                                                target_width=target_width,
                                                resampling_mode=Image.NEAREST,
                                                target_aspect_ratio=target_width / target_height,
                                                )
                    img = np.array(depth_pil).astype(np.float32) / 4000.0
                else:
                    img = img.astype(np.float32)
                img = np.expand_dims(img, axis=-1)  # (H, W, 1)
                images.append(img)
        images = np.stack(images, axis=0)  # (N, H, W, C)
        images = torch.from_numpy(images).float()
        return images

def load_dl3dv_cameras(pose_filepath: str, intrinsic_filepath: str, target_width: int = 512, target_height: int = 512):
    """
    Load camera poses and intrinsics from .npz files.
    Args:
        pose_filepath (str): File path to the camera poses .npz file.
        intrinsic_filepath (str): File path to the camera intrinsics .npz file.
        image_width (int): Width of the images.
        image_height (int): Height of the images.
    Returns:
        cam_meta_dict: Dictionary mapping camera indices to their metadata.
        cam_poses: Tensor of shape (N, 4, 4) containing camera-to-world poses.
    """
    cam_poses = np.load(pose_filepath)["data"]  # (N, 4, 4)
    cam_intrinsics = np.load(intrinsic_filepath)["data"]  # (N, 4) fx, fy, cx, cy
    cam_indices = np.load(intrinsic_filepath)["inds"]
    cam_meta_dict = {}
    cam_intrs = []
    for idx in cam_indices:
        original_height = cam_intrinsics[idx][3] * 2
        fl_scale_factor = target_height / original_height
        fx = cam_intrinsics[idx][0] * fl_scale_factor
        fy = cam_intrinsics[idx][1] * fl_scale_factor
        intrinsic_mat = np.array([
            [fx, 0.0, target_width / 2],
            [0.0, fy, target_height / 2],
            [0.0, 0.0, 1.0],
        ])
        c2w_pose = cam_poses[idx]
        cam_position = c2w_pose[:3, 3]
        cam_meta_dict[str(idx)] = {
            "camera_id": int(idx),
            "image_height": target_height,
            "image_width": target_width,
            "fx": float(fx),
            "fy": float(fy),
            "intrinsic": intrinsic_mat.tolist(),
            "camera_position": cam_position.tolist(),
            "camera_transform": np.linalg.inv(c2w_pose).tolist(),
            }
        cam_intrs.append(intrinsic_mat)
    cam_intrs = torch.from_numpy(np.stack(cam_intrs, axis=0)).float()  # (N, 3, 3)
    cam_poses = torch.from_numpy(cam_poses).float()  # (N, 4, 4)
    return cam_meta_dict, cam_poses, cam_intrs


def sample_perspective_data(    
                        data_root_path: str,
                        sample_room_id: str,
                        image_width: int = 512,
                        image_height: int = 512,
                        min_sample_views: int = 24,
                        min_cond_views: int = 8,
                        depth_scale: float = 1.0,
                        device: torch.device = torch.device("cuda:0"),
                    ) -> Dict[str, Any]:

    # cam_pose_filepath = glob.glob(osp.join(data_root_path, "vipe_output/pose", "*.npz"))[0]
    # cam_intrinsic_filepath = glob.glob(osp.join(data_root_path, "vipe_output/intrinsics", "*.npz"))[0]
    # rgbs_dir = osp.join(data_root_path, "images_4")
    # depths_dir = osp.join(data_root_path, "vipe_output/depth")
    cam_pose_filepath = osp.join(data_root_path, "pose", f"{sample_room_id}.npz")
    cam_intrinsic_filepath = osp.join(data_root_path, "intrinsics", f"{sample_room_id}.npz")
    rgb_video_filepath = osp.join(data_root_path, "rgb", f"{sample_room_id}.mp4")
    depths_filepath = osp.join(data_root_path, "depth", f"{sample_room_id}.zip")
    
    # Step 1: load all RGB-D images
    rgbs: Float[Tensor, "N H W C"] = load_dl3dv_images(rgb_video_filepath, image_type="rgb", target_width=image_width, target_height=image_height)
    depths: Float[Tensor, "N H W C"] = load_dl3dv_depths_from_zip(depths_filepath, target_width=image_width, target_height=image_height)
    print(f"rgbs shape: {rgbs.shape}, depths shape: {depths.shape}")
    
    
    # Step 2: select at least 24 frames in each room
    cam_meta_dict, c2w_poses, cam_intrinsics = load_dl3dv_cameras(pose_filepath=cam_pose_filepath, 
                                                                  intrinsic_filepath=cam_intrinsic_filepath, 
                                                                  target_width=image_width, 
                                                                  target_height=image_height)
    valid_cam_keys = list(cam_meta_dict.keys())
    max_sample_views = min_sample_views + 8
    sampled_cam_keys = sample_camera_positions(cameras_dict=cam_meta_dict, valid_pano_view_keys=valid_cam_keys, num_sample_views=min_sample_views, max_sample_views=max_sample_views)
    # print(f"Sampled camera keys: {sampled_cam_keys}")
    sampled_rgbs = rgbs[sampled_cam_keys]
    sampled_depths = depths[sampled_cam_keys]
    print(f"sampled_rgbs.shape: {sampled_rgbs.shape}, sampled_depths.shape: {sampled_depths.shape}")
    
    # Step 3: unproject depth to point cloud
    sampled_poses: Float[Tensor, "Nv 4 4"] = c2w_poses[sampled_cam_keys]
    intrinsic_mat: Float[Tensor, "3 3"] = cam_intrinsics[0]
    fov_x = 2.0 * np.arctan2(image_width / 2.0, intrinsic_mat[0,0])
    fov_y = 2.0 * np.arctan2(image_height / 2.0, intrinsic_mat[1,1])

    pointcloud: Float[Tensor, "1 Nv Np 3"] = batch_unproject_depth_to_pointcloud(
        depth_maps=sampled_depths[None, ...].to(device),
        c2w_poses=sampled_poses[None, ...].to(device),
        intrinsic_mat=intrinsic_mat[None, ...].to(device),
    )
    # import open3d as o3d
    # scene_ply = o3d.geometry.PointCloud()
    # scene_ply.points = o3d.utility.Vector3dVector(pointcloud[0].detach().cpu().numpy().reshape(-1, 3))
    # o3d.io.write_point_cloud("debug_pointcloud.ply", scene_ply)
    
    subview_masks = [torch.logical_and(depth_map > 0, depth_map < DL3DV_DEPTH_MAX).squeeze() for depth_map in sampled_depths]
    # filter out edge
    subview_edge_masks = [ edge_filter(depth_map.squeeze().cpu().numpy(), times=0.2) for depth_map in sampled_depths]
    subview_masks = [torch.logical_and(subview_masks[i], torch.from_numpy(1 - subview_edge_masks[i])) for i in range(len(subview_masks))]
    subview_mask_indices = [torch.where(mask) for mask in subview_masks]
    subview_mask_xs = [m[1].detach().cpu().numpy() for m in subview_mask_indices]
    subview_mask_ys = [m[0].detach().cpu().numpy() for m in subview_mask_indices]
    subview_masks: Float[Tensor, "Nv Np"] = rearrange(torch.stack(subview_masks), "Nv H W -> Nv (H W)")
    valid_pointcloud: Float[Tensor, "Npp 3"] = pointcloud[0, subview_masks]
    
    # Step 4: calculate layout of a single room
    scene_ply = o3d.geometry.PointCloud()
    scene_ply.points = o3d.utility.Vector3dVector(pointcloud[0].detach().cpu().numpy().reshape(-1, 3))
    scene_center = np.array(scene_ply.get_center())
    scene_scale = np.max(np.array(scene_ply.get_max_bound()) - np.array(scene_ply.get_min_bound()))
    print(f"scene_center: {scene_center}, scene_scale: {scene_scale}")
    
    # Step 5: sample conditional images
    cond_cam_keys = [ int(cam_key) for cam_key in valid_cam_keys if int(cam_key) not in sampled_cam_keys ]
    cond_cam_keys = np.random.choice(cond_cam_keys, size=min(len(cond_cam_keys), min_cond_views), replace=False).tolist()
    # print(f"Conditional camera keys: {cond_cam_keys}")
    cond_c2ws = c2w_poses[cond_cam_keys]
    cond_images = rgbs[cond_cam_keys]
    cond_depths = depths[cond_cam_keys]
    cond_fovs = np.array([fov_x] * cond_c2ws.shape[0])
    cond_fovs_y = np.array([fov_y] * cond_c2ws.shape[0])
        
    sampled_rgbs = rearrange(sampled_rgbs, "Nv H W C -> Nv C H W")
    sampled_depths = rearrange(sampled_depths, "Nv H W C -> Nv C H W")
    cond_rgbs = rearrange(cond_images, "Nv H W C -> Nv C H W")
    cond_depths = rearrange(cond_depths, "Nv H W C -> Nv C H W")
    data = {
        "c2ws": sampled_poses,  # c2w poses, (Nv, 4, 4)
        "images": sampled_rgbs,  # rgb images [0,1], (Nv, 3, H, W)
        "depths": sampled_depths,  # metric depth maps, (Nv, 1, H, W)
        "pointclouds": valid_pointcloud,  # pointcloud in world coordinate
        "mask_x_indices": subview_mask_xs,  # list: [Nv, (HxW)]
        "mask_y_indices": subview_mask_ys,  # list: [Nv, (HxW)]
        "fovs": np.array([fov_x] * sampled_poses.shape[0]),
        "fovs_y": np.array([fov_y] * sampled_poses.shape[0]),
        "cond_c2ws": cond_c2ws,  # (N_cond, 4, 4)
        "cond_images": cond_rgbs,  # (N_cond, C, H, W)
        "cond_depths": cond_depths,  # (N_cond, 1, H, W)
        "cond_fovs": cond_fovs,  # (N_cond,)
        "cond_fovs_y": cond_fovs_y,  # (N_cond,)
        "scene_center": scene_center,
        "scene_scale": scene_scale,
        "intrinsic_mat": intrinsic_mat,
    }
    return data