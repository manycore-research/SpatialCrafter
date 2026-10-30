import os
import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import os.path as osp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
import pyexr
import imageio
import numpy as np
from PIL import Image

from utils.typing import Float, Tensor
from utils.file_util import load_from_csv, save_to_csv
from utils.img_util import resize_pil_image
from data_engine.re10k.re10k import load_re10k_depths_from_zip


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Wan-VACE training data, produce GT RGB-D video, conditional RGB-D video, refefrence image.")
    parser.add_argument(
        "--re10k_dataset_rootpath",
        type=str,
        default="/path/to/real-estate-10k/process_re10k_traindata",
        help="path to the root dir of processed spatial video dataset",
    )
    parser.add_argument(
        "--trellis_results_rootpath",
        type=str,
        default="/path/to/real-estate-10k/process_re10k_trellis_gs_inference",
        help="path to the root dir of trellis generation results",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="the scene id to be processed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/real-estate-10k/process_re10k_wan_vace_data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="number of worker processes, default is cpu_count()",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--task_id", type=int, default=0)
    return parser.parse_args()


def load_images_as_video(video_path: str, subfix="png", image_type="rgb", target_width=256, target_height=256):
    """Load images from a directory as a video (list of frames).
    
    Args:
        video_path (str): Path to the directory containing image frames.
    Returns:
        video_frames (list of np.ndarray): List of image frames as numpy arrays.
    """
    video_frames = []
    if subfix == "mp4":
        video_reader = imageio.get_reader(video_path, 'ffmpeg')
        num_frames = video_reader.count_frames()
        for id in range(num_frames):
            frame = video_reader.get_data(id)
            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                frame_pil = resize_pil_image(img=Image.fromarray(frame), 
                                   target_height=target_height, 
                                   target_width=target_width,
                                   resampling_mode=Image.BILINEAR,
                                   target_aspect_ratio=target_width / target_height,
                                   )
                frame = np.array(frame_pil)
            video_frames.append(frame)
        video_reader.close()
        
    else:
        frame_files = [f for f in os.listdir(video_path) if f.endswith(f".{subfix}")]
        frame_files.sort(key=lambda x: int(x.split(f".{subfix}")[0]))
        
        for frame_file in frame_files:
            filepath = os.path.join(video_path, frame_file)
            if image_type == "rgb":
                image = np.array(Image.open(filepath))
                if len(image.shape) == 3:  # RGB image
                    if image.shape[0] != target_height or image.shape[1] != target_width:
                        image_pil = resize_pil_image(img=Image.fromarray(image), 
                                        target_height=target_height, 
                                        target_width=target_width,
                                        resampling_mode=Image.BILINEAR,
                                        target_aspect_ratio=target_width / target_height,
                                        )
                        image = np.array(image_pil)
                    video_frames.append(image)
                    
            elif image_type == "depth" and subfix == "png":
                depth_image = np.array(Image.open(filepath))
                if depth_image.shape[0] != target_height or depth_image.shape[1] != target_width:
                    image_pil = resize_pil_image(img=Image.fromarray(depth_image), 
                                    target_height=target_height, 
                                    target_width=target_width,
                                    resampling_mode=Image.NEAREST,
                                    target_aspect_ratio=target_width / target_height,
                                    )
                    depth_image = np.array(image_pil)
                video_frames.append(depth_image.astype(np.float32) / 1000.0)  # Convert mm to meters
                    
            elif image_type == "depth" and subfix == "exr":
                depth_image = pyexr.read(filepath)[:,:,0]  # read depth image
                if depth_image.shape[0] != target_height or depth_image.shape[1] != target_width:
                    image_pil = resize_pil_image(img=Image.fromarray((depth_image * 1000).astype(np.uint16)),  # Convert meters to mm for resizing
                                    target_height=target_height, 
                                    target_width=target_width,
                                    resampling_mode=Image.NEAREST,
                                    target_aspect_ratio=target_width / target_height,
                                    )
                    depth_image = np.array(image_pil).astype(np.float32) / 1000.0  # Convert back to meters after resizing
                video_frames.append(depth_image)
                
        # normalize depths by min-max normalization
        if image_type == "depth":
            depths = np.stack(video_frames, axis=0)
            depths_min = np.min(depths)
            depths_max = np.max(depths)
            depths = (depths - depths_min) / (depths_max - depths_min + 1e-8)
            depth_images = (depths * 255.0).clip(0, 255).astype(np.uint8)
            depth_images = depth_images[..., np.newaxis].repeat(3, axis=-1)  # convert to 3-channel depth image
            video_frames = [depth_images[i] for i in range(depth_images.shape[0])]
        else:
            video_frames = video_frames
    
    meta_info = {
        "width": video_frames[0].shape[1],
        "height": video_frames[0].shape[0],
        "depth_min": float(depths_min) if image_type == "depth" else None,
        "depth_max": float(depths_max) if image_type == "depth" else None,
    }
    
    return video_frames, meta_info

def load_depths_from_zip_as_video(zip_path: str, target_width=256, target_height=256):
    """Load depth images from a zip file as a video (list of frames).
    
    Args:
        zip_path (str): Path to the zip file containing depth images.
    Returns:
        video_frames (list of np.ndarray): List of depth frames as numpy arrays.
    """
    video_frames = []
    depths: Float[Tensor, "N H W C"] = load_re10k_depths_from_zip(zip_path, target_width=target_width, target_height=target_height)
    depths_np = depths.cpu().numpy()
    
    # normalize depths by min-max normalization
    depths_min = np.min(depths_np)
    depths_max = np.max(depths_np)
    for i in range(depths_np.shape[0]):
        depth_image = depths_np[i]
        depth_image = (depth_image - depths_min) / (depths_max - depths_min + 1e-8)
        depth_image = (depth_image * 255.0).clip(0, 255).astype(np.uint8)
        depth_image = np.repeat(depth_image, 3, axis=-1)  # convert to 3-channel depth image
        video_frames.append(depth_image)

    meta_info = {
        "width": video_frames[0].shape[1],
        "height": video_frames[0].shape[0],
        "depth_min": float(depths_min),
        "depth_max": float(depths_max),
    }
    return video_frames, meta_info

def unproject_depth_np(depth_map: np.ndarray, c2w: np.ndarray, intrinsic_mat: np.ndarray) -> np.ndarray:
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

def load_depth_from_zip_as_scm_video(video_path: str, 
                                   rgb_video_lst: list,
                                   cam_c2ws: np.ndarray, 
                                   cam_intrinsics: np.ndarray,
                                   target_width: int = 256,
                                   target_height: int=256,
                                   scene_scale: float = 1.0):
    """Load depth images from a directory and convert them to scene coordinate maps (SCM).
    
    Args:
        video_path (str): Path to the directory containing depth image frames.
        cam_c2ws (np.ndarray): Camera extrinsic matrices. (N, 4, 4), where N is the number of frames.
        cam_intrinsics (np.ndarray): Camera intrinsic parameters. (N, 3, 3), normalized by image width and height, i.e., fx/image_width, fy/image_height, cx/image_width, cy/image_height.
        scene_scale (float): Scale factor for the scene.
    Returns:
        scm_video_frames (list of np.ndarray): List of SCM frames as numpy arrays.
    """
    video_frames = []
    depths: Float[Tensor, "N H W C"] = load_re10k_depths_from_zip(video_path, target_width=target_width, target_height=target_height)
    depths_np = depths.cpu().numpy()
    
    
    for idx, depth_map in enumerate(depths_np):
        c2w = cam_c2ws[idx]  # (4, 4)
        from copy import deepcopy
        intrinsic = deepcopy(cam_intrinsics[idx])  # (3, 3)
        # fxfycxcy = np.array([intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]])  # (fx, fy, cx, cy)
        intrinsic[0, :] *= depth_map.shape[1]  # scale fx to image size
        intrinsic[1, :] *= depth_map.shape[0]  # scale fy to image size
        normalized_depth_map = depth_map[:,:,0] / scene_scale  # normalize depth by scene scale
        scene_coord_map = unproject_depth_np(depth_map=normalized_depth_map, 
                                             c2w=c2w, 
                                             intrinsic_mat=intrinsic)    # [-1, 1]
        # print(f"scene_coord_map shape: {scene_coord_map.shape}, min: {scene_coord_map.min()}, max: {scene_coord_map.max()}")
        # Image.fromarray(rgb_video_lst[idx]).save(f"debug_re10k_v2_scm/debug_rgb_frame_{idx}.png")
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # points = scene_coord_map.transpose(1, 2, 0).reshape(-1, 3)
        # colors = rgb_video_lst[idx].reshape(-1, 3) / 255.0
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.io.write_point_cloud(f"debug_re10k_v2_scm/debug_scm_frame_{idx}.ply", pcd)
        video_frames.append(scene_coord_map)
            
    # normalize depths by min-max normalization
    scms = np.stack(video_frames, axis=0)
    scene_coords_min = np.min(scms)
    scene_coords_max = np.max(scms)
    
    meta_info = {
        "width": video_frames[0].shape[1],
        "height": video_frames[0].shape[0],
        "scm_min": float(scene_coords_min),
        "scm_max": float(scene_coords_max),
    }
    
    return scms, meta_info

def load_depth_images_as_scm_video(video_path: str, 
                                   rgb_video_lst: list,
                                   cam_c2ws: np.ndarray, 
                                   cam_intrinsics: np.ndarray,
                                   scene_scale: float = 1.0):
    """Load depth images from a directory and convert them to scene coordinate maps (SCM).
    
    Args:
        video_path (str): Path to the directory containing depth image frames.
        cam_c2ws (np.ndarray): Camera extrinsic matrices. (N, 4, 4), where N is the number of frames.
        cam_intrinsics (np.ndarray): Camera intrinsic parameters. (N, 3, 3), normalized by image width and height, i.e., fx/image_width, fy/image_height, cx/image_width, cy/image_height.
        scene_scale (float): Scale factor for the scene.
    Returns:
        scm_video_frames (list of np.ndarray): List of SCM frames as numpy arrays.
    """
    video_frames = []
    frame_files = [f for f in os.listdir(video_path) if f.endswith(".png") and not f.endswith("_colormap.png")]
    frame_files.sort(key=lambda x: int(x.split(".png")[0]))
    
    for idx, frame_file in enumerate(frame_files):
        filepath = os.path.join(video_path, frame_file)
        
        assert idx == int(frame_file.split(".png")[0]), f"Frame file {frame_file} does not match expected index {idx}"
        
        image = np.array(Image.open(filepath))
        assert len(image.shape) == 2, f"Expected depth image with 2 dimensions, but got shape {image.shape} for file {filepath}"
        
        depth_map = image.astype(np.float32) / 1000.0  # Convert mm to meters

        c2w = cam_c2ws[idx]  # (4, 4)
        from copy import deepcopy
        intrinsic = deepcopy(cam_intrinsics[idx])  # (3, 3)
        # fxfycxcy = np.array([intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]])  # (fx, fy, cx, cy)
        intrinsic[0, :] *= image.shape[1]  # scale fx to image size
        intrinsic[1, :] *= image.shape[0]  # scale fy to image size
        normalized_depth_map = depth_map / scene_scale  # normalize depth by scene scale
        scene_coord_map = unproject_depth_np(depth_map=normalized_depth_map, 
                                             c2w=c2w, 
                                             intrinsic_mat=intrinsic)    # [-1, 1]
        # print(f"scene_coord_map shape: {scene_coord_map.shape}, min: {scene_coord_map.min()}, max: {scene_coord_map.max()}")
        # Image.fromarray(rgb_video_lst[idx]).save(f"debug_re10k_v2_scm/debug_rgb_frame_{idx}.png")
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # points = scene_coord_map.transpose(1, 2, 0).reshape(-1, 3)
        # colors = rgb_video_lst[idx].reshape(-1, 3) / 255.0
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.io.write_point_cloud(f"debug_re10k_v2_scm/debug_scm_frame_{idx}.ply", pcd)
        video_frames.append(scene_coord_map)
            
    # normalize depths by min-max normalization
    scms = np.stack(video_frames, axis=0)
    scene_coords_min = np.min(scms)
    scene_coords_max = np.max(scms)
    
    meta_info = {
        "width": video_frames[0].shape[1],
        "height": video_frames[0].shape[0],
        "scm_min": float(scene_coords_min),
        "scm_max": float(scene_coords_max),
    }
    
    return scms, meta_info

def process_single_video(args_tuple):
    """Process a single video.
    
    Args:
        args_tuple: Tuple of (video_dir, re10k_video_data_path, trellis_results_path, output_path)
    
    Returns:
        dict: Metadata dictionary for this video
    """
    video_dir, re10k_video_data_path, trellis_results_path, output_path = args_tuple
    
    output_video_path = os.path.join(output_path, video_dir)
    os.makedirs(output_video_path, exist_ok=True)
    ref_rgb_image_path = os.path.join(output_video_path, "reference_rgb.png")
    save_gt_rgb_video_path = os.path.join(output_video_path, "video_rgb.mp4")
    save_gt_depth_video_path = os.path.join(output_video_path, "video_depth.mp4")
    save_cond_rgb_video_path = os.path.join(output_video_path, "video_rgb_coarse.mp4")
    save_cond_depth_video_path = os.path.join(output_video_path, "video_depth_coarse.mp4")
    save_gt_scm_npz_path = os.path.join(output_video_path, "video_scm.npz")
    save_cond_scm_npz_path = os.path.join(output_video_path, "video_scm_coarse.npz")
    if osp.exists(save_gt_rgb_video_path) and \
       osp.exists(save_gt_scm_npz_path) and \
       osp.exists(save_cond_rgb_video_path) and \
       osp.exists(save_cond_scm_npz_path) and \
       osp.exists(ref_rgb_image_path):
        print(f"Video {video_dir} already processed, skipping...")
        num_frames = imageio.get_reader(save_gt_rgb_video_path).count_frames()
        return {
            "video": save_gt_rgb_video_path,
            "video_depth": save_gt_depth_video_path,
            "prompt": 'high quality indoor scene',
            "control_video": save_cond_rgb_video_path,
            "control_video_depth": save_cond_depth_video_path,
            "reference_image": ref_rgb_image_path,
            "num_frames": num_frames,
            "width": 256,
            "height": 256,
        }
    
    try:
        gt_video_path = re10k_video_data_path
        trellis_video_path = os.path.join(trellis_results_path, video_dir)
        camera_params_path = os.path.join(trellis_video_path, "cameras.npz")
        
        # load RGB-D images
        gt_rgb_video_path = os.path.join(gt_video_path, "rgb", f"{video_dir}.mp4")
        gt_depth_video_path = os.path.join(gt_video_path, "depth", f"{video_dir}.zip")
        cond_rgb_video_path = os.path.join(trellis_video_path, "coarse_rgbs")
        cond_depth_video_path = os.path.join(trellis_video_path, "coarse_depths")
        coarse_rgb_video, _ = load_images_as_video(cond_rgb_video_path, subfix="png", image_type="rgb")
        coarse_depth_video, _ = load_images_as_video(cond_depth_video_path, subfix="png", image_type="depth")
        
        target_image_width, target_image_height = coarse_rgb_video[0].shape[1], coarse_rgb_video[0].shape[0]
        gt_rgb_video, _ = load_images_as_video(gt_rgb_video_path, subfix="mp4", image_type="rgb", target_width=target_image_width, target_height=target_image_height)
        # print(f"gt_rgb_video shape: {len(gt_rgb_video)} x {gt_rgb_video[0].shape}")
        # TODO: check the depth scale between trellis and spatial video data
        # gt_depth_video, depth_meta_info = load_images_as_video(gt_depth_video_path, subfix="exr", image_type="depth", target_width=target_image_width, target_height=target_image_height)
        gt_depth_video, depth_meta_info = load_depths_from_zip_as_video(gt_depth_video_path, target_width=target_image_width, target_height=target_image_height)

        # save camera parameters and depth scale
        camera_params = np.load(camera_params_path, allow_pickle=True)
        depth_min = depth_meta_info["depth_min"]
        depth_max = depth_meta_info["depth_max"]
        # load gt depth and convert to scene coordinate maps
        c2w_poses = camera_params["c2w_poses"]
        intrinsics = camera_params["intrinsics"]
        gt_scm_video, scm_meta_info = load_depth_from_zip_as_scm_video(video_path=gt_depth_video_path,
                                                                    rgb_video_lst=gt_rgb_video,
                                                                    cam_c2ws=c2w_poses, 
                                                                    cam_intrinsics=intrinsics,
                                                                    scene_scale=camera_params["scene_scale"])
        coarse_scm_video, _ = load_depth_images_as_scm_video(video_path=cond_depth_video_path,
                                                        rgb_video_lst=coarse_rgb_video,
                                                        cam_c2ws=c2w_poses, 
                                                        cam_intrinsics=intrinsics,
                                                        scene_scale=camera_params["scene_scale"])
        scm_min = scm_meta_info["scm_min"]
        scm_max = scm_meta_info["scm_max"]
        np.savez(os.path.join(output_video_path, "cameras.npz"),
                    c2w_poses=camera_params["c2w_poses"],
                    intrinsics=camera_params["intrinsics"],
                    scene_scale=camera_params["scene_scale"],
                    depth_min=depth_min,
                    depth_max=depth_max,
                    scm_min=scm_min,
                    scm_max=scm_max
            )
        # save the first frame as reference image
        imageio.imsave(ref_rgb_image_path, gt_rgb_video[0])
        
        # save as mp4 videos
        fps = 16
        imageio.mimwrite(save_gt_rgb_video_path, gt_rgb_video, fps=fps, quality=10)
        imageio.mimwrite(save_gt_depth_video_path, gt_depth_video, fps=fps, quality=10)
        imageio.mimwrite(save_cond_rgb_video_path, coarse_rgb_video, fps=fps, quality=10)
        imageio.mimwrite(save_cond_depth_video_path, coarse_depth_video, fps=fps, quality=10)

        np.savez(save_gt_scm_npz_path, scm=gt_scm_video)    # (N, 3, H, W)
        np.savez(save_cond_scm_npz_path, scm=coarse_scm_video)  # (N, 3, H, W)
        
        return {
            "video": save_gt_rgb_video_path,
            "video_depth": save_gt_scm_npz_path,
            "prompt": 'high quality indoor scene',
            "control_video": save_cond_rgb_video_path,
            "control_video_depth": save_cond_scm_npz_path,
            "reference_image": ref_rgb_image_path,
            "num_frames": len(gt_rgb_video),
            "width": gt_rgb_video[0].shape[1],
            "height": gt_rgb_video[0].shape[0],
        }
    except Exception as e:
        print(f"Error processing video {video_dir}: {str(e)}")
        return None

if __name__ == "__main__":
    args = parse_args()
    
    re10k_dataset_rootpath = args.re10k_dataset_rootpath
    trellis_results_path = args.trellis_results_rootpath
    scene_id = args.scene_id
    start_index = args.start_index
    end_index = args.end_index
    task_id = args.task_id
    num_workers = args.num_workers if args.num_workers > 0 else cpu_count() -1
    output_dir = args.output_dir
    
    if not os.path.exists(re10k_dataset_rootpath):
        raise FileNotFoundError(f"Dataset directory {re10k_dataset_rootpath} does not exist.")
    if not os.path.exists(trellis_results_path):
        raise FileNotFoundError(f"Predictions directory {trellis_results_path} does not exist.")
    
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    video_split_filepath = osp.join(trellis_results_path, "metadata.csv")
    video_df = load_from_csv(video_split_filepath, start_index=start_index, end_index=end_index)
    if scene_id is None:
        video_df = video_df[video_df["feature_dinov2_vitl14_reg"] == True]
        video_df = video_df[video_df["gs_inference"] == True]
        video_dirs = video_df["sha256"].unique().tolist()
    else:
        video_dirs = [scene_id]

    # Prepare arguments for multiprocessing
    process_args = [
        (video_dir, re10k_dataset_rootpath, trellis_results_path, output_dir)
        for video_dir in video_dirs
    ]
    
    # Process videos using multiprocessing
    print(f"Processing {len(video_dirs)} videos using {num_workers} workers...")
    meta_info = []
    
    # process_single_video(
    #     (video_dirs[0], re10k_dataset_rootpath, trellis_results_path, output_dir)
    # )
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, process_args),
            total=len(process_args),
            desc="Processing videos"
        ))
    
    # Filter out None results (failed processing)
    meta_info = [result for result in results if result is not None]

    # save the metadata file
    meta_filepath = os.path.join(output_dir, f"metadata_wan_funcontrol_{task_id}.csv")
    fieldnames = ["video", "video_depth", "prompt", "control_video", "control_video_depth", "reference_image", "num_frames", "width", "height"]
    save_to_csv(meta_info, meta_filepath, fieldnames)
    print(f"Preprocessing completed. Processed {len(meta_info)}/{len(video_dirs)} videos successfully.")
