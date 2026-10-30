import os
import sys
sys.path.append(".")
sys.path.append("..")
import json
import copy
import argparse
import os.path as osp
from tqdm import tqdm
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
torch.set_num_threads(1)
import trimesh
import imageio
import numpy as np
import pandas as pd
from PIL import Image
from easydict import EasyDict as edict

import trellis
import trellis.modules.sparse as sp
from trellis.representations import Gaussian
from trellis.models import SLatEncoder, SLatGaussianDecoder, ElasticSLatEncoder, ElasticSLatGaussianDecoder
from trellis.utils.render_utils import render_frames
from utils.typing import *
from utils.file_util import load_from_json, load_from_csv
from utils.vis_util import apply_depth_to_colormap
from data_engine.dl3dv.dl3dv import load_dl3dv_cameras


def parse_args():
    parser = argparse.ArgumentParser(description="Inference TRELLIS-GS-VAE on Spatial Video Dataset, produce RGB-D renderings on the specific camera trajectory.")
    parser.add_argument(
        "--dl3dv_dataset_rootpath",
        type=str,
        default="/path/to/process_dl3dv_traindata",
        help="path to the root dir of processed DL3DV dataset",
    )
    parser.add_argument(
        "--trellis_dataset_rootpath",
        type=str,
        default="/path/to/process_dl3dv_trellis_data",
        help="path to the root dir of processed trellis dataset",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="the scene id to be processed",
    )
    parser.add_argument(
        "--gsvae_ckpt_config_path",
        type=str,
        default="/path/to/pretrained_ckpts/sft_slat_vae_64/config.json",
        help="path to the GS-VAE checkpoint config file",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=512,
        help="target width of the rendered video frames",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=512,
        help="target height of the rendered video frames",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/process_dl3dv_trellis_gs_inference",
    )
    return parser.parse_args()

def load_gsvae_checkpoint(ckpt_config_path: str):
    """
    Load checkpoint of GS-VAE encoder and decoder.
    
    Args:
        ckpt_config_path (str): Path to the GS-VAE checkpoint config file.
    
    Returns:
        encoder (SLatEncoder): Loaded GS-VAE encoder.
        decoder (SLatGaussianDecoder): Loaded GS-VAE decoder.
    """
    
    train_cfg = edict(json.load(open(ckpt_config_path, "r")))
    encoder: SLatEncoder = getattr(trellis.models, train_cfg.models.encoder.name)(**train_cfg.models.encoder.args).cuda()
    encoder_ckpt_filepath = ckpt_config_path.replace("config.json", "ckpts/encoder_step0004000.pt")
    encoder.load_state_dict(torch.load(encoder_ckpt_filepath), strict=False)
    encoder.eval()
    print(f"Loaded {train_cfg.models.encoder.name} from {encoder_ckpt_filepath}")

    decoder: SLatGaussianDecoder = getattr(trellis.models, train_cfg.models.decoder.name)(**train_cfg.models.decoder.args).cuda()
    decoder_ckpt_filepath = ckpt_config_path.replace("config.json", "ckpts/decoder_step0004000.pt")
    decoder.load_state_dict(torch.load(decoder_ckpt_filepath), strict=False)
    decoder.eval()
    print(f"Loaded {train_cfg.models.decoder.name} from {decoder_ckpt_filepath}")
    return encoder, decoder

def load_structured_feature(dataset_root_dir: str, scene_id: str):
    """
    Load voxelized structured features for a given scene.
    
    Args:
        dataset_root_dir (str): Root directory of the processed trellis dataset.
        scene_id (str): Scene identifier.
    Returns:
        feats (sp.SparseTensor): Loaded structured features as a sparse tensor.
    """
    voxelized_feat_filepath = os.path.join(dataset_root_dir, "features", f"{scene_id}.npz")
    voxelized_feats = np.load(voxelized_feat_filepath)
    print(f"[load_structured_feature] feats shape: {voxelized_feats['patchtokens'].shape}, indices shape: {voxelized_feats['indices'].shape}")
    feats = sp.SparseTensor(
        feats=torch.from_numpy(voxelized_feats["patchtokens"]).float(),
        coords=torch.cat(
            [
                torch.zeros(voxelized_feats["patchtokens"].shape[0], 1).int(),
                torch.from_numpy(voxelized_feats["indices"]).int(),
            ],
            dim=1,
        ),
    )
    return feats

def load_video_camera_trajectory_and_scaling(video_data_root_dir: str, 
                                             scene_id: str, 
                                             scene_center: np.ndarray, 
                                             scene_scale: float,
                                             target_width: int = 512,
                                             target_height: int = 512,
                                             debug_out_dir: str = None):
    """
    Load and scale camera trajectory of DL3DV data into the GaussianSplating coordinate system.
    
    Args:
        video_data_root_dir (str): Root directory of the processed DL3DV dataset.
        scene_id (str): Scene identifier.
        scene_center (np.ndarray): Scene center for scaling.
        scene_scale (float): Scene scale for scaling.
    Returns:
        video_w2c_extrinsics (torch.Tensor): Scaled video camera extrinsics in world-to-camera format. Shape: (N, 4, 4)
        intrinsics (torch.Tensor): Video camera intrinsics. Shape: (N, 3, 3)
    """
    # intrinsics_npz_filepath = glob.glob(osp.join(video_data_root_dir, scene_id, "vipe_output/intrinsics/*.npz"))[0]
    # cameras_npz_filepath = glob.glob(osp.join(video_data_root_dir, scene_id, "vipe_output/pose/*.npz"))[0]
    intrinsics_npz_filepath = osp.join(video_data_root_dir, f"intrinsics/{scene_id}.npz")
    cameras_npz_filepath = osp.join(video_data_root_dir, f"pose/{scene_id}.npz")
    cam_meta_dict, _, _ = load_dl3dv_cameras(pose_filepath=cameras_npz_filepath, 
                                        intrinsic_filepath=intrinsics_npz_filepath, 
                                        target_width=target_width, 
                                        target_height=target_height)
    video_trajectory_lst = []
    intrinsics_lst = []
    
    DEBUG_CAMERA_TRAJECTORY = False
    for cam_id_str, cam_info in cam_meta_dict.items():
        img_height, img_width = cam_info["image_height"], cam_info["image_width"]
        cam_pose_w2c = np.array(cam_info["camera_transform"]).reshape((4, 4)).astype(np.float32)
        cam_pose_c2w = np.linalg.inv(cam_pose_w2c)
        
        # scale pose w.r.t scene center and scale
        pose_scale_mat = np.eye(4)
        pose_scale_mat[:3, 3] = -scene_center
        pose_scale_mat[:3] *= 1.0 / scene_scale
        
        scaled_c2w = pose_scale_mat @ cam_pose_c2w
        q_c2w = trimesh.transformations.quaternion_from_matrix(scaled_c2w[:3, :3])
        q_c2w = trimesh.transformations.unit_vector(q_c2w)
        R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
        scaled_c2w[:3, :3] = R_c2w
        cam_pose_c2w = scaled_c2w
        # convert to our coordinate system (x-right, y-forward, z-up) to align with TRELLIS
        our2cv_transform = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        cv2ours_transform = np.linalg.inv(our2cv_transform)
        cam_pose_c2w = cv2ours_transform @ cam_pose_c2w
        
        video_trajectory_lst.append(cam_pose_c2w)
        inntrinsic = np.array(cam_info["intrinsic"]).reshape((3, 3))
        normalized_intrinsic = inntrinsic.copy()
        normalized_intrinsic[0, :] *= 1.0 / img_width
        normalized_intrinsic[1, :] *= 1.0 / img_height
        intrinsics_lst.append(normalized_intrinsic)
        # # scale depth map accordingly
        # depth_filepath = osp.join(video_depth_dir, f"{cam_id_str}.png")
        # depth_map = np.array(Image.open(depth_filepath)).astype(np.float32) / 1000.0  # in meters
        # depth_map *= 1.0 / gs_scale
        # scaled_depth_dir = osp.join(input_video_data_dir, "video_scaled_depths")
        # os.makedirs(scaled_depth_dir, exist_ok=True)
        # Image.fromarray((depth_map * 4000).astype(np.uint16)).save(osp.join(scaled_depth_dir, f"{cam_id_str}.png"))
        
        if DEBUG_CAMERA_TRAJECTORY:
            import open3d as o3d
            from utils.vis_util import vis_camera_keyframe
            cam_lines = vis_camera_keyframe(
                fx=cam_info["fx"],
                fy=cam_info["fy"],
                width=cam_info["image_width"],
                height=cam_info["image_height"],
                c2w=cam_pose_c2w
            )
            o3d.io.write_line_set(osp.join(debug_out_dir, f"video_cam_{cam_id_str}.ply"), cam_lines)
            
    video_w2c_extrinsics = np.stack(video_trajectory_lst, axis=0)  # (N, 4, 4)
    video_w2c_extrinsics = torch.from_numpy(np.linalg.inv(video_w2c_extrinsics)).float()  # c2w to w2c
    intrinsics = torch.from_numpy(np.stack(intrinsics_lst, axis=0)).float()  # (N, 3, 3)
    return video_w2c_extrinsics, intrinsics

  
if __name__ == "__main__":
    
    args = parse_args()
    video_data_root_dir = args.dl3dv_dataset_rootpath
    trellis_data_root_dir = args.trellis_dataset_rootpath
    scene_id = args.scene_id
    gsvae_ckpt_config_path = args.gsvae_ckpt_config_path
    start_index = args.start_index
    end_index = args.end_index
    target_width = args.target_width
    target_height = args.target_height
    cpu_num_workers = args.num_workers
    task_id = args.task_id
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load GS-VAE encoder and decoder
    gsvae_encoder, gsvae_decoder = load_gsvae_checkpoint(ckpt_config_path=gsvae_ckpt_config_path)
    gsvae_encoder: ElasticSLatEncoder
    gsvae_decoder: ElasticSLatGaussianDecoder
    
    # get file list
    if os.path.exists(os.path.join(trellis_data_root_dir, "metadata.csv")):
        metadata = pd.read_csv(os.path.join(trellis_data_root_dir, "metadata.csv"))
    else:
        raise ValueError("metadata.csv not found")
    
    
    feature_tag = "feature_dinov2_vitl14_reg"
    metadata = metadata[metadata[feature_tag] == True]

    start_index = max(0, start_index)
    end_index = len(metadata) if end_index <= 0 else min(end_index, len(metadata))
    metadata = metadata[start_index:end_index]
    
    # filter out scenes that are already processed
    latents_tag = "gs_inference"
    sha256s = list(metadata["sha256"].values)
    filtered_sha256s = []
    for sha256 in sha256s:
        output_scene_rgb_video_file = osp.join(output_dir, sha256, "sample_gs_rgb.mp4")
        output_scene_depth_video_file = osp.join(output_dir, sha256, "sample_gs_depth.mp4")
        if os.path.exists(output_scene_rgb_video_file) and os.path.exists(output_scene_depth_video_file):
            print(f"Scene {sha256} already processed, skipping.")
            continue
        filtered_sha256s.append(sha256)
    sha256s = filtered_sha256s
    print(f"Total {len(sha256s)} scenes to process after filtering.")
    
    records = []
    
    # encode latents
    load_queue = Queue(maxsize=8)
    # try:
    with ThreadPoolExecutor(max_workers=cpu_num_workers) as loader_executor, ThreadPoolExecutor(max_workers=cpu_num_workers) as saver_executor:

        def loader(sha256):
            try:
                feats = np.load(os.path.join(trellis_data_root_dir, "features", f"{sha256}.npz"))
                # load scene center and scale
                gs_meta_dict = np.load(osp.join(trellis_data_root_dir, "world_cameras", f"{sha256}.npz"))
                gs_center = gs_meta_dict["scene_center"]  # (3, )
                gs_scale = gs_meta_dict["scene_scale"]  # float
                # load and scale video camera trajectory
                video_extrinsics, intrinsics = load_video_camera_trajectory_and_scaling(video_data_root_dir, 
                                                                                        sha256, 
                                                                                        scene_center=gs_center, 
                                                                                        scene_scale=gs_scale,
                                                                                        target_width=target_width,
                                                                                        target_height=target_height,
                                                                                        debug_out_dir=None)
                load_queue.put((sha256, feats, (gs_center, gs_scale), (video_extrinsics, intrinsics)))
            except Exception as e:
                print(f"Error loading features for {sha256}: {e}")

        loader_executor.map(loader, sha256s)

        def saver(sha256, pack):
            # Save rendered RGB-D video
            scene_output_dir = osp.join(output_dir, sha256)
            os.makedirs(scene_output_dir, exist_ok=True)
            output_rgbs_dir = osp.join(scene_output_dir, "coarse_rgbs")
            output_depths_dir = osp.join(scene_output_dir, "coarse_depths")
            os.makedirs(output_rgbs_dir, exist_ok=True)
            os.makedirs(output_depths_dir, exist_ok=True)

            # save c2w extrinsics and intrinsics
            c2w_poses = pack["extrinsics"]
            intrinsics = pack["intrinsics"]
            np.savez(os.path.join(scene_output_dir, "cameras.npz"),
                     c2w_poses=c2w_poses,
                     intrinsics=intrinsics,
                     scene_scale=pack["scale"],)
            
            rgb_video = pack["rgbs"]
            depth_video = pack["depths"]
            scene_scale = pack["scale"]
            for i in range(len(rgb_video)):
                pil_img = Image.fromarray(rgb_video[i].astype(np.uint8))
                pil_img.save(osp.join(output_rgbs_dir, f"{i}.png"))
            imageio.mimsave(f"{scene_output_dir}/sample_gs_rgb.mp4", rgb_video, fps=60)
            
            for i in range(len(depth_video)):
                depth_frame = depth_video[i][0]
                depth_frame[np.isinf(depth_frame)] = 0.0
                depth_frame[depth_frame < 0] = 0.0
                pil_img = Image.fromarray((depth_frame * scene_scale * 1000).astype(np.uint16))
                pil_img.save(osp.join(output_depths_dir, f"{i}.png"))
                
            depth_video_colored = [apply_depth_to_colormap(depth_frame[0], cmap="jet") for depth_frame in depth_video]
            imageio.mimsave(f"{scene_output_dir}/sample_gs_depth.mp4", depth_video_colored, fps=60)
            
            records.append({"sha256": sha256, "gs_inference": True})

        for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
            sha256, feats, (gs_center, gs_scale), (video_extrinsics, intrinsics) = load_queue.get()
            scene_output_dir = osp.join(output_dir, sha256)
            rgb_video_file = osp.join(scene_output_dir, "sample_gs_rgb.mp4")
            depth_video_file = osp.join(scene_output_dir, "sample_gs_depth.mp4")
            if os.path.exists(rgb_video_file) and os.path.exists(depth_video_file):
                print(f"Scene {sha256} already processed, skipping.")
                records.append({"sha256": sha256, "gs_inference": True})
                continue
            feats = sp.SparseTensor(
                feats=torch.from_numpy(feats["patchtokens"]).float(),
                coords=torch.cat(
                    [
                        torch.zeros(feats["patchtokens"].shape[0], 1).int(),
                        torch.from_numpy(feats["indices"]).int(),
                    ],
                    dim=1,
                ),
            ).cuda()

            # inference GS-VAE
            structure_latent, mean, logvar = gsvae_encoder(feats, sample_posterior=True, return_raw=True)
            print(f"Encoded latent code: {structure_latent.shape}, mean: {mean.shape}, std: {torch.exp(0.5 * logvar).shape}")
            assert torch.isfinite(structure_latent.feats).all(), "Non-finite latent"
            decoded_gaussians: List[Gaussian] = gsvae_decoder(structure_latent)[0]
            print(f"Decoded gaussians: {decoded_gaussians[0].get_xyz.shape}")
                    
            # Render video frames
            video_extrinsics = video_extrinsics.cuda()
            intrinsics = intrinsics.cuda()
            image_resolution = target_width
            background_color = (0, 0, 0)
            rendering_results = render_frames(
                sample=decoded_gaussians[0],
                extrinsics=video_extrinsics,
                intrinsics=intrinsics,
                options={"resolution": image_resolution, "bg_color": background_color},
            )
            # Save the outputs
            pack = {
                "rgbs": np.array(rendering_results['color']).astype(np.uint8),  # (N, H, W, 3) explicitly copy to a new array
                "depths": np.array(rendering_results['depth']).astype(np.float32), # (N, 1, H, W) explicitly copy to a new array
                "scale": gs_scale,
                "center": gs_center,
                "extrinsics": copy.deepcopy(torch.inverse(video_extrinsics).cpu().numpy()),
                "intrinsics": copy.deepcopy(intrinsics.cpu().numpy()),
            }
            del rendering_results, decoded_gaussians
            print(f"rendering colors shape: {pack['rgbs'].shape}, depths shape: {pack['depths'].shape}")
            saver_executor.submit(saver, sha256, pack)

        saver_executor.shutdown(wait=True)

    records = pd.DataFrame.from_records(records)
    print(f"--------------- Encoded latents for {len(records)} samples ---------------")
    
    # update the metadata.csv
    full_metadata = pd.read_csv(os.path.join(trellis_data_root_dir, "metadata.csv"))
    # full_metadata[latents_tag] = [False] * len(full_metadata)
    for sha256 in records["sha256"].values:
        full_metadata.loc[full_metadata["sha256"] == sha256, latents_tag] = True
    full_metadata.to_csv(os.path.join(output_dir, f"metadata_{task_id}.csv"), index=False)

