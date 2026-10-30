import os
import sys

sys.path.append(".")
sys.path.append("..")
import json
import argparse
from PIL import Image
from typing import List, Union

import torch
import utils3d
import imageio
import trimesh
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from easydict import EasyDict as edict
from torchvision import transforms

import trellis
import trellis.modules.sparse as sp
from trellis.pipelines import samplers
from trellis.utils import render_utils
from trellis.models import (ElasticSLatGaussianDecoder, 
                            ElasticSLatFlowModel, 
                            SparseStructureEncoder,
                            SparseStructureDecoder, 
                            SparseStructureFlowModel
)
from utils.typing import *
from utils.filter_util import edge_filter
from utils.geo_util import rgbd_to_pointcloud, batch_unproject_depth_to_pointcloud, unproject_depth_np
from depth_anything_3.api import DepthAnything3


VOXEL_RESOLUTION = 64
# DINO preprocess
DINO_IMG_SIZE = 518
DINO_DOWNSAMPLE_RATE = 14
DINO_PATCH_SIZE = DINO_IMG_SIZE // DINO_DOWNSAMPLE_RATE

def parse_args():
    parser = argparse.ArgumentParser(description="Validate Trellis img2scene model")
    parser.add_argument("--dataset_rootpath", 
                        type=str, 
                        default="/path/to/datasets/Spatial3D/v8_64",
                        help="Root directory of the dataset")
    parser.add_argument("--ss_vae_config_file", 
                        type=str, 
                        default="/path/to/experiments/DiffSynth-Studio/pretrained_ckpts/sft_ss_vae_64/config.json",
                        help="Path to the configuration file")
    parser.add_argument("--ss_vae_checkpoint_step", type=int, default=4000, help="Which step of Checkpoint to load")
    parser.add_argument("--ss_dit_config_file", 
                        type=str, 
                        default="/path/to/experiments/DiffSynth-Studio/pretrained_ckpts/sft_ss_dit_64/config.json",
                        help="Path to the configuration file")
    parser.add_argument("--ss_dit_checkpoint_step", type=int, default=50000, help="Which step of Checkpoint to load")
    parser.add_argument("--slat_vae_config_file", 
                        type=str, 
                        default="/path/to/experiments/DiffSynth-Studio/pretrained_ckpts/sft_slat_vae_64/config.json",
                        help="Path to the configuration file")
    parser.add_argument("--slat_vae_checkpoint_step", type=int, default=4000, help="Which step of Checkpoint to load")
    parser.add_argument("--slat_dit_config_file", 
                        type=str, 
                        default="/path/to/experiments/DiffSynth-Studio/pretrained_ckpts/sft_slat_dit_64/config.json",
                        help="Path to the configuration file")
    parser.add_argument("--slat_dit_checkpoint_step", type=int, default=8000, help="Which step of Checkpoint to load")
    # parser.add_argument("--scene_id", type=str, default="20240506_3FO4K5FY7V0G_room_539", help="Which scene to validate on")
    parser.add_argument("--scene_id", type=str, default="1ba74c22670ad047981441581d00f26f4a148d1010bcac7468c615adf5fa4d5d", help="Which scene to validate on")
    parser.add_argument("--output_dir", type=str, default="./validation_trellis_output", help="Directory to save the validation outputs")
    return parser.parse_args()

def infer_dav3(ref_rgb_array: List[np.ndarray], 
               model: DepthAnything3, 
               scene_output_dir: str,
               ref_rgb_poses: Float[Tensor, "N_ref 4 4"] = None,
               ref_rgb_intrinsic: Float[Tensor, "N_ref 3 3"] = None,
               pre_scene_center: np.ndarray = None,
               pre_scene_scale: float = None,
               target_image_size: int = 512,
               device: torch.device = torch.device("cuda"),
               dtype: torch.dtype = torch.float16,
               debug_points: bool = False,
    ):
    """
    Infer Metric depth maps and camera poses from the input RGB images using DepthAnything3 (DAV3).
    
    Args:
        ref_rgb_array (List[np.ndarray]): List of input RGB images as numpy arrays, each with shape (H, W, 3) and dtype uint8.
        model (DepthAnything3): The DAV3 model for inference.
        scene_output_dir (str): The directory to save the intermediate outputs for debugging.
        ref_rgb_poses (List[np.ndarray], optional): List of reference camera poses corresponding to the input RGB images, each with shape (4, 4). If None, the poses predicted by DAV3 will be used. Default is None.
        ref_rgb_intrinsic (List[np.ndarray], optional): List of reference camera intrinsic matrices (normalized params, not related to the image size) corresponding to the input RGB images, each with shape (3, 3). If None, the intrinsics predicted by DAV3 will be used. Default is None.
        pre_scene_center (np.ndarray, optional): The center of the scene for normalizing the point cloud. If None, it will be calculated from the predicted point cloud. Default is None.
        pre_scene_scale (float, optional): The scale of the scene for normalizing the point cloud. If None, it will be calculated from the predicted point cloud. Default is None.
        device (torch.device, optional): The device to run the inference on. Default is torch.device("cuda").
        dtype (torch.dtype, optional): The data type for the model inference. Default is torch.float16.
        debug_points (bool, optional): Whether to save the predicted point cloud for debugging. Default is False.
    
    Returns:
        Dict: A dictionary containing the following keys:
            - "images": The input RGB images as a tensor of shape (Nv, 3, H, W) and dtype float32, normalized to [0, 1].
            - "depths": The predicted depth maps as a tensor of shape (Nv, H, W, 1) and dtype float32, in metric scale.
            - "pointclouds": The valid point cloud in world coordinates as a tensor of shape (Npp, 3) and dtype float32.
            - "mask_x_indices": A list of numpy arrays containing the x indices of the valid pixels for each view.
            - "mask_y_indices": A list of numpy arrays containing the y indices of the valid pixels for each view.
            - "scene_center": The center of the scene used for normalizing the point cloud, as a numpy array of shape (3,).
            - "scene_scale": The scale of the scene used for normalizing the point cloud, as a float.
    """
    assert len(ref_rgb_array) == 1, "Currently only support single view DAV3 prediction, ref_rgb_array should have length 1"
        
    with torch.no_grad():
        # Step 1: Predict poses, depth maps, and point maps from the input image using VGGT.
        prediction = model.inference(ref_rgb_array)
            
        images = prediction.processed_images     # [N, H, W, 3] uint8   array
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device) / 255.0  # (N, 3, H, W)
        # resize the input image to target_image_size
        if images.shape[2] != target_image_size or images.shape[3] != target_image_size:
            images = torch.nn.functional.interpolate(images, 
                                                     size=(target_image_size, target_image_size), 
                                                     mode='bilinear', 
                                                     align_corners=False)
        
        # Step 2: Convert pose encoding to extrinsic and intrinsic matrices
        extrinsics = torch.from_numpy(prediction.extrinsics).float().to(device)  # [N, 3, 4]    float32 array # opencv w2c or colmap format
        intrinsics = torch.from_numpy(prediction.intrinsics).float().to(device)  # [N, 3, 3]    float32 array
        intrinsics = intrinsics.unsqueeze(0)  # (1, N, 3, 3)
        
        if ref_rgb_poses is None:
            # expand extrinsic to BxNvx4x4 matrix
            extrinsic_pad = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(extrinsics.shape[0], -1, -1, -1)  # (B, 1, 4, 4)
            extrinsic_pad[:, :extrinsics.shape[0], :extrinsics.shape[1], :extrinsics.shape[2]] = extrinsics.unsqueeze(1)  # (B, N_v, 4, 4)
        else:
            extrinsic_pad = ref_rgb_poses.unsqueeze(0).float()  # (B, N_v, 4, 4)
            logger.info(f"Using reference RGB camera poses for point cloud unprojection")
        
        if ref_rgb_intrinsic is not None:
            ref_rgb_intrinsic[:, 0, :] *= target_image_size  # scale fx by target_image_size
            ref_rgb_intrinsic[:, 1, :] *= target_image_size  # scale fy by target_image_size
            intrinsics = ref_rgb_intrinsic.unsqueeze(0).float()   # (1, N, 3, 3)
        logger.info(f"Extrinsic: {extrinsic_pad}, Intrinsic: {intrinsics}")
        # Step 3: Generate world points from depth map
        depth_maps: Float[Tensor, "Nv 1 H W"] = torch.from_numpy(prediction.depth).unsqueeze(1).float()  # (Nv, 1, H, W)
        if depth_maps.shape[1] != target_image_size or depth_maps.shape[2] != target_image_size:
            depth_maps = torch.nn.functional.interpolate(depth_maps, 
                                                         size=(target_image_size, target_image_size), 
                                                         mode='nearest')  # (Nv, 1, H, W)
            
        depth_maps = depth_maps / pre_scene_scale if pre_scene_scale is not None else depth_maps
        logger.info(f"Depth map shape: {depth_maps.shape}, depth map range: [{depth_maps.min()}, {depth_maps.max()}], pre_scene_scale: {pre_scene_scale}")
        pointcloud: Float[Tensor, "1 Nv Np 3"] = batch_unproject_depth_to_pointcloud(
            depth_maps=depth_maps[..., None].to(device),
            c2w_poses=extrinsic_pad.to(device),
            intrinsic_mat=intrinsics.squeeze(1).to(device),
        )
        # filter out invalid depth and edge
        subview_masks = [torch.logical_and(depth_map > 0, depth_map < 100.0).squeeze() for depth_map in depth_maps]
        subview_edge_masks = [ edge_filter(depth_map.squeeze().cpu().numpy(), times=0.3) for depth_map in depth_maps]
        subview_masks = [torch.logical_and(subview_masks[i], torch.from_numpy(1 - subview_edge_masks[i])) for i in range(len(subview_masks))]
        subview_mask_indices = [torch.where(mask) for mask in subview_masks]
        subview_mask_xs = [m[1].detach().cpu().numpy() for m in subview_mask_indices]
        subview_mask_ys = [m[0].detach().cpu().numpy() for m in subview_mask_indices]
        subview_masks: Float[Tensor, "Nv Np"] = rearrange(torch.stack(subview_masks), "Nv H W -> Nv (H W)")
        valid_pointcloud: Float[Tensor, "Npp 3"] = pointcloud[0, subview_masks]

        # Step 4: calculate layout of a single room if predefined scene scale and center is unknown
        if pre_scene_center is None and pre_scene_scale is None:
            scene_ply = o3d.geometry.PointCloud()
            scene_ply.points = o3d.utility.Vector3dVector(valid_pointcloud.detach().cpu().numpy().reshape(-1, 3))
            calc_scene_center = np.array(scene_ply.get_center())
            calc_scene_scale = np.max(np.array(scene_ply.get_max_bound()) - np.array(scene_ply.get_min_bound()))
            logger.info(f"Compute scene_center: {calc_scene_center}, scene_scale: {calc_scene_scale} from point cloud!")
            # normalize the point cloud to the computed scene scale
            valid_pointcloud = valid_pointcloud / float(calc_scene_scale)
            logger.info(f"Normalized point cloud to scene scale: {calc_scene_scale}")
        
        # save the point cloud as .ply file for visualization
        if debug_points:
            rgb_img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)
            o3d_color_points = rgbd_to_pointcloud(depth_image=depth_maps[0].permute(1, 2, 0).cpu().numpy(),
                        rgb_image=rgb_img,
                        c2w_pose=extrinsic_pad[0, 0].cpu().numpy(),
                        depth_scale=1.0,
                        intrinsic_mat=intrinsics[0,0].cpu().numpy(),
            )
            o3d.io.write_point_cloud(os.path.join(scene_output_dir, "dav3_world_points.ply"), o3d_color_points)
        del prediction # free up GPU memory
        
    scaled_depths = depth_maps if pre_scene_scale is not None else depth_maps / calc_scene_scale
    return {
        "images": images,  # rgb images [0,1], (Nv, 3, H, W)
        "depths": scaled_depths.squeeze(1).unsqueeze(-1),  # scaled depth maps, (Nv, H, W, 1)
        "pointclouds": valid_pointcloud,  # pointcloud of ref_rgb_array, normalized by scene scale
        "mask_x_indices": subview_mask_xs,  # list: [Nv, (HxW)]
        "mask_y_indices": subview_mask_ys,  # list: [Nv, (HxW)]
        "scene_center": pre_scene_center if pre_scene_scale is not None else calc_scene_center,
        "scene_scale": pre_scene_scale if pre_scene_scale is not None else calc_scene_scale,
    }

def voxelize_features(
    points: Float[np.ndarray, "Npp 3"],
    features: Float[np.ndarray, "Nv H W C"],
    x_indices: List[Float[np.ndarray, "Np"]],
    y_indices: List[Float[np.ndarray, "Np"]],
    # scene_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    # scene_scale: float = 1.0,
    batch_point_size: int = 10_000,
    voxel_res: int = 64,
    image_size: int = 512,
    global_debug: bool = False,
    debug_output_dir: str = "./debug_output",
):
    """
    Voxelizes the point cloud and DINO features.

    Args:
        points (np.ndarray): The point cloud, shape (NvxHxW, 3).
        features (np.ndarray): The DINO features, shape (Nv, H//14, W//14, C).
        x_indices (List[np.ndarray]): The x indices of the valid image pixels, shape (Nv, HxW).
        y_indices (List[np.ndarray]): The y indices of the valid image pixels, shape (Nv, HxW).
        scene_center (np.ndarray): The center of the scene for normalizing the point cloud, shape (3,).
        scene_scale (float): The scale of the scene for normalizing the point cloud.
        batch_point_size (int): The maximum number of points per batch.
        voxel_res (int): The resolution of the voxel grid.
        image_size (int): The size of the input image.
        global_debug (bool): Whether to save the global normalized point cloud for debugging. If False, only save the condition view normalized point cloud. Default is False.
        debug_output_dir (str): The directory to save the debug output files.
    """
    voxel_feats = np.zeros((voxel_res, voxel_res, voxel_res, features.shape[-1]), dtype=np.float32)
    voxel_visits = np.zeros((voxel_res, voxel_res, voxel_res), dtype=np.int32)

    # Create arrays for indexing
    n_views = features.shape[0]

    image_indices = np.repeat(np.arange(n_views), [len(x) for x in x_indices]).astype(np.uint16)
    flat_x_indices = np.concatenate(x_indices).astype(np.uint16)
    flat_y_indices = np.concatenate(y_indices).astype(np.uint16)

    ## normalize the point cloud to [-0.5, 0.5]
    # points = (points - scene_center) / scene_scale
    # points = points / (2.0 * scene_scale)
    
    # assume the points already in [-0.5, 0.5] range
    if global_debug and debug_output_dir is not None:
        import open3d as o3d
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)
        # downsample for visualization
        o3d_points = o3d_points.voxel_down_sample(voxel_size=0.005)
        o3d.io.write_point_cloud(f"{debug_output_dir}/debug_global_normalized_points.ply", o3d_points)
    elif global_debug == False and debug_output_dir is not None:
        import open3d as o3d
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)
        # downsample for visualization
        o3d_points = o3d_points.voxel_down_sample(voxel_size=0.005)
        o3d.io.write_point_cloud(f"{debug_output_dir}/debug_condview_normalized_points.ply", o3d_points)
    
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
    if global_debug:
        print(f"Global voxels shape: {voxels.shape}, voxel_feats shape: {voxel_feats.shape}, masks shape: {masks.shape}")
    else:
        print(f"Condition view voxels shape: {voxels.shape}, voxel_feats shape: {voxel_feats.shape}, masks shape: {masks.shape}")

    return voxels, voxel_feats, masks

class TempImageTo3DPipeline:
    def __init__(
        self,
        ss_sampler: samplers.Sampler,
        ss_sampler_params: dict,
        ss_flow_model: torch.nn.Module,
        ss_encoder: torch.nn.Module,
        ss_decoder: torch.nn.Module,
        slat_decoder_gs_model: torch.nn.Module,
        slat_flow_model: torch.nn.Module,
        slat_sampler: samplers.Sampler,
        slat_sampler_params: dict,
        slat_normalization: dict = None,
        dav3_model: DepthAnything3 = None,
        localdino_repo_path: str = "facebookresearch/dinov2",
        device: torch.device = torch.device("cuda"),
    ):
        """ Single image to 3D pipeline for testing the pretrained Trellis models. 
        It first samples the sparse structure with the sparse structure flow model, 
        then samples the structured latent with the slat flow model, 
        and finally decodes the structured latent to gaussian splats with the slat decoder.
        
        Args:
            ss_sampler (samplers.Sampler): The sampler for sparse structure sampling.
            ss_sampler_params (dict): The parameters for the sparse structure sampler.
            ss_flow_model (torch.nn.Module): The flow model for sparse structure sampling.
            ss_decoder (torch.nn.Module): The decoder for sparse structure.
            slat_decoder_gs_model (torch.nn.Module): The gaussian splat decoder for structured latent.
            slat_flow_model (torch.nn.Module): The flow model for structured latent sampling.
            slat_sampler (samplers.Sampler): The sampler for structured latent sampling.
            slat_sampler_params (dict): The parameters for the structured latent sampler.
            slat_normalization (dict): The normalization parameters for the structured latent, with keys "mean" and "std".
            localdino_repo_path (str): The local path to the DINOv2 repository for image conditioning model.
            device (torch.device): The device to run the pipeline on.
        
        """
        self.ss_sampler = ss_sampler
        self.sparse_structure_sampler_params = ss_sampler_params
        self.ss_flow_model = ss_flow_model
        self.ss_encoder = ss_encoder
        self.ss_decoder = ss_decoder
        self.slat_decoder_gs_model = slat_decoder_gs_model
        self.slat_flow_model = slat_flow_model
        self.slat_sampler = slat_sampler
        self.slat_sampler_params = slat_sampler_params
        self.slat_normalization = slat_normalization
        self.device = device
        self.cond_models = {}
        self._init_image_cond_model(local_repo_path=localdino_repo_path)
        self.dav3_model = dav3_model

    def _init_image_cond_model(self, local_repo_path: str = "facebookresearch/dinov2"):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load(local_repo_path, "dinov2_vitl14_reg", source="local", pretrained=True)
        dinov2_model.to(self.device)
        dinov2_model.eval()
        self.cond_models["image_cond_model"] = dinov2_model
        transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.image_cond_model_transform = transform

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.cond_models["image_cond_model"](image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def get_cond_with_dav3(self, 
                           image: Union[torch.Tensor, list[Image.Image]],
                           input_rgb_poses: Float[Tensor, "N_ref 4 4"] = None,
                           input_rgb_intrinsic: Float[Tensor, "N_ref 3 3"] = None,
                           pre_scene_center: np.ndarray = None,
                           pre_scene_scale: float = None,
                           output_dir: str = None,
                           ) -> Tuple[dict, Float[Tensor, "Nv H W 1"]]:
        """
        Get the conditioning information for the model using DAV3.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.
            input_rgb_poses (Float[Tensor, "N_ref 4 4"], optional): The camera poses for the reference images.
            input_rgb_intrinsic (Float[Tensor, "N_ref 3 3"], optional): The camera intrinsics for the reference images.
            pre_scene_center (np.ndarray, optional): The pre-defined scene center.
            pre_scene_scale (float, optional): The pre-defined scene scale.
            output_dir (str, optional): The directory to save debug outputs.

        Returns:
            Tuple[dict, Float[Tensor, "Nv H W 1"]]: The conditioning information and the validity mask for the points.
        """
        
        cond = self.encode_image(image)    # [B H//14xW//14 + 5 1024]
        neg_cond = torch.zeros_like(cond)

        assert self.ss_flow_model.in_channels == 16, f"The SS-FLOW in_channels should be 16 for DAV3 conditioning, but got {self.ss_flow_model.in_channels}"
        result_dict = infer_dav3(
            ref_rgb_array=[np.array(input_img) for input_img in image],
            model=self.dav3_model,
            ref_rgb_poses=input_rgb_poses,
            ref_rgb_intrinsic=input_rgb_intrinsic,
            pre_scene_center = pre_scene_center,
            pre_scene_scale = pre_scene_scale,
            target_image_size = 512,
            scene_output_dir=output_dir,
            debug_points=True,
            device=self.device,
        )
        valid_points: Float[Tensor, "Npp 3"] = result_dict["pointclouds"]  # (Npp, 3)
        subview_mask_xs = result_dict["mask_x_indices"]  # list: [Nv, (HxW)]
        subview_mask_ys = result_dict["mask_y_indices"]  # list: [Nv, (HxW)]
        scene_center = result_dict["scene_center"] if pre_scene_center is None else pre_scene_center
        scene_scale = result_dict["scene_scale"] if pre_scene_scale is None else pre_scene_scale
        # Step 6: voxelize the pointcloud and DINO feature
        dino_feats = cond[:, self.cond_models["image_cond_model"].num_register_tokens + 1:, :]    # num_register_tokens == 5 for DINOv2
        dino_feats = dino_feats.reshape(1, DINO_PATCH_SIZE, DINO_PATCH_SIZE, cond.shape[-1]).detach().cpu().numpy()  # (1, Ph, Pw, C)
        cond_voxel_coords, cond_voxel_feats, masks = voxelize_features(
            points=valid_points.cpu().numpy(),
            features=dino_feats,
            x_indices=subview_mask_xs,
            y_indices=subview_mask_ys,
            # scene_center=np.array([0, 0, 0]),
            # scene_scale=scene_scale,
            voxel_res=VOXEL_RESOLUTION,
            image_size=512,
            global_debug=False,
            debug_output_dir=output_dir,
        )
        if output_dir is not None:
            # save the voxels for debugging
            import open3d as o3d
            voxel_pcd = o3d.geometry.PointCloud()
            voxel_pcd.points = o3d.utility.Vector3dVector(cond_voxel_coords.astype(np.float32))
            o3d.io.write_point_cloud(f"{output_dir}/cond_voxels.ply", voxel_pcd)
        
        # encode the cond_voxel_coords
        bs = len(image)
        cond_ss = torch.zeros(bs, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, dtype=torch.long)
        cond_ss[:, cond_voxel_coords[:, 0], cond_voxel_coords[:, 1], cond_voxel_coords[:, 2]] = 1
        cond_ss = cond_ss.unsqueeze(0).cuda().float()
        cond_ss_latent = self.ss_encoder(cond_ss, sample_posterior=False)
        
        return_dict = {
            "cond": cond,
            "neg_cond": neg_cond,
            "cond_x_0": cond_ss_latent,
        }

        scaled_depth_maps: Float[Tensor, "Nv H W 1"] = result_dict["depths"].clip_(0.0, 1.0)  # scaled depth maps, (Nv, H, W, 1)
        return return_dict, scaled_depth_maps, scene_scale, scene_center
        
    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == "RGBA":
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert("RGB")
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            output = input
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        return output

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.ss_flow_model
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        print(f"Sampling sparse structure with noise shape: {noise.shape}")
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.ss_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=True).samples

        # Decode occupancy latent
        decoder = self.ss_decoder
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        return coords

    def sample_sparse_structure_with_condition_voxel(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            input_image (Image.Image): The input image.
            cond (dict): The conditioning information of the input image.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.ss_flow_model
        reso = flow_model.resolution
        assert flow_model.in_channels == 16, f"The SS-FLOW in_channels should be 16 for DAV3 conditioning, but got {flow_model.in_channels}"
        target_x_0_chann = flow_model.in_channels // 2
        noise = torch.randn(num_samples, target_x_0_chann, reso, reso, reso).to(self.device)
        print(f"Sampling sparse structure with noise shape: {noise.shape}")
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.ss_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=True).samples

        # Decode occupancy latent
        decoder = self.ss_decoder
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        return coords
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.slat_flow_model
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=True).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}

        if "mesh" in formats:
            raise NotImplementedError("Mesh decoding is not implemented yet.")
        if "gaussian" in formats:
            ret["gaussian"] = self.slat_decoder_gs_model(slat)[0]
        if "radiance_field" in formats:
            raise NotImplementedError("Radiance field decoding is not implemented yet.")
        return ret

    def run_sample(
        self,
        gt_voxel_coords: torch.Tensor = None,
        image: Image.Image = None,
        seed: int = 42,
        formats: List[str] = ["gaussian"],
        output_dir: str = "./",
    ) -> dict:
        image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        voxel_coords = self.sample_sparse_structure(cond)
        if gt_voxel_coords is not None:
            print(f"gt_voxel_coords shape: {gt_voxel_coords.shape}, sampled voxel_coords shape: {voxel_coords.shape}")
        utils3d.io.write_ply(f"{output_dir}/gen_sparse_structure.ply", voxel_coords[:, 1:].cpu().numpy())

        slat = self.sample_slat(cond, voxel_coords, self.slat_sampler_params)
        return self.decode_slat(slat, formats)

    def run_sample_with_dav3(
        self,
        gt_voxel_coords: torch.Tensor = None,
        image: Image.Image = None,
        input_rgb_poses: Float[Tensor, "N_ref 4 4"] = None,
        input_rgb_intrinsic: Float[Tensor, "N_ref 3 3"] = None,
        scene_center: np.ndarray = None,
        scene_scale: float = None,
        seed: int = 42,
        formats: List[str] = ["gaussian"],
        output_dir: str = "./",
    ) -> dict:
        """
        Run the sampling process with DAV3 conditioning.
        
        Args:
            gt_voxel_coords (torch.Tensor, optional): The ground truth voxel coordinates for debugging.
            image (Image.Image, optional): The input image for conditioning.
            input_rgb_poses (Float[Tensor, "N_ref 4 4"], optional): The camera poses of the reference RGB images for DAV3 conditioning.
            input_rgb_intrinsic (Float[Tensor, "N_ref 3 3"], optional): The camera intrinsics of the reference RGB images for DAV3 conditioning.
            scene_center (np.ndarray, optional): Predefined center of the scene for normalizing the point cloud. If None, it will be calculated from the predicted point cloud.
            scene_scale (float, optional): Predefined scale of the scene for normalizing the point cloud. If None, it will be calculated from the predicted point cloud.
            seed (int, optional): The random seed for sampling. Default is 42.
            formats (List[str], optional): The formats to decode the structured latent to. Default is ["gaussian"].
            output_dir (str, optional): The directory to save the output files such as the predicted point cloud and voxel coordinates. Default is "./".
        """
        image = self.preprocess_image(image)    # (518, 518, 3)
        cond, depth_maps, scene_scale, scene_center = self.get_cond_with_dav3(
            image=[image],
            input_rgb_poses=input_rgb_poses,
            input_rgb_intrinsic=input_rgb_intrinsic,
            pre_scene_center=scene_center,
            pre_scene_scale=scene_scale,
            output_dir=output_dir,
        )
        torch.manual_seed(seed)
        voxel_coords = self.sample_sparse_structure_with_condition_voxel(cond)
        if gt_voxel_coords is not None:
            print(f"gt_voxel_coords shape: {gt_voxel_coords.shape}, sampled voxel_coords shape: {voxel_coords.shape}")
        utils3d.io.write_ply(f"{output_dir}/gen_sparse_structure.ply", voxel_coords[:, 1:].cpu().numpy())

        if "cond_x_0" in cond:
            del cond["cond_x_0"] # free up GPU memory
        slat = self.sample_slat(cond, voxel_coords, self.slat_sampler_params)
        return_dict = self.decode_slat(slat, formats)
        return_dict["depths_for_inputs"] = depth_maps  # scaled depth maps, (Nv, H, W, 1), from DAV3 prediction, [0, 1]
        return_dict["scene_scale"] = scene_scale
        return_dict["scene_center"] = scene_center
        return return_dict
    
def load_ss_vae_and_dit(ss_vae_config_filepath: str, ss_vae_decoder_ckpt_path: str, ss_dit_config_filepath: str, ss_dit_ckpt_path: str):
    ss_vae_cfg = edict(json.load(open(ss_vae_config_filepath, "r")))
    ss_vae_encoder_ckpt_path = ss_vae_decoder_ckpt_path.replace("decoder", "encoder")
    ss_encoder: SparseStructureEncoder = getattr(trellis.models, ss_vae_cfg.models.encoder.name)(**ss_vae_cfg.models.encoder.args).cuda()
    ss_encoder.load_state_dict(torch.load(ss_vae_encoder_ckpt_path), strict=False)
    ss_encoder.eval()
    logger.info(f"Loaded {ss_vae_cfg.models.encoder.name} from {ss_vae_encoder_ckpt_path}")
    
    ss_decoder: SparseStructureDecoder = getattr(trellis.models, ss_vae_cfg.models.decoder.name)(**ss_vae_cfg.models.decoder.args).cuda()
    ss_decoder.load_state_dict(torch.load(ss_vae_decoder_ckpt_path), strict=False)
    ss_decoder.eval()
    logger.info(f"Loaded {ss_vae_cfg.models.decoder.name} from {ss_vae_decoder_ckpt_path}")

    ss_flow_cfg = edict(json.load(open(ss_dit_config_filepath, "r")))
    ss_denoiser: SparseStructureFlowModel = getattr(trellis.models, ss_flow_cfg.models.denoiser.name)(**ss_flow_cfg.models.denoiser.args).cuda()
    ss_denoiser.load_state_dict(torch.load(ss_dit_ckpt_path), strict=False)
    ss_denoiser.eval()
    logger.info(f"Loaded {ss_flow_cfg.models.denoiser.name} from {ss_dit_ckpt_path}")

    return ss_encoder, ss_decoder, ss_denoiser

def load_slat_vae_and_dit(slat_vae_config_filepath: str, slat_vae_decoder_ckpt_path: str, slat_denoiser_config_filepath: str, slat_denoiser_ckpt_path: str):
    slat_vae_cfg = edict(json.load(open(slat_vae_config_filepath, "r")))
    slat_decoder: ElasticSLatGaussianDecoder = getattr(trellis.models, slat_vae_cfg.models.decoder.name)(**slat_vae_cfg.models.decoder.args).cuda()
    slat_decoder.load_state_dict(torch.load(slat_vae_decoder_ckpt_path), strict=False)
    slat_decoder.eval()
    logger.info(f"Loaded {slat_vae_cfg.models.decoder.name} from {slat_vae_decoder_ckpt_path}")

    slat_flow_cfg = edict(json.load(open(slat_denoiser_config_filepath, "r")))
    slat_denoiser: ElasticSLatFlowModel = getattr(trellis.models, slat_flow_cfg.models.denoiser.name)(**slat_flow_cfg.models.denoiser.args).cuda()
    slat_denoiser.load_state_dict(torch.load(slat_denoiser_ckpt_path), strict=False)
    slat_denoiser.eval()
    logger.info(f"Loaded {slat_flow_cfg.models.denoiser.name} from {slat_denoiser_ckpt_path}")
    
    return slat_decoder, slat_denoiser

def build_pipeline_from_ckpt(
    ss_vae_config_filepath: str,
    ss_dit_config_filepath: str,
    slat_vae_config_filepath: str,
    slat_denoiser_config_filepath: str,
    ss_vae_checkpoint_step: int = 4000,
    ss_dit_checkpoint_step: int = 50000,
    slat_vae_checkpoint_step: int = 4000,
    slat_dit_checkpoint_step: int = 8000,
    dav3_ckpt_filepth: str = None,
):
    """Build the image to 3D pipeline with the given checkpoint paths.
    
    Args:
    ss_vae_config_filepath (str): The file path to the sparse structure VAE configuration file.
    ss_dit_config_filepath (str): The file path to the sparse structure DIT configuration file.
    slat_vae_config_filepath (str): The file path to the SLat VAE configuration file.
    slat_denoiser_config_filepath (str): The file path to the SLat DIT configuration file.
    ss_vae_checkpoint_step (int): Which step of checkpoint to load for sparse structure VAE.
    ss_dit_checkpoint_step (int): Which step of checkpoint to load for sparse structure DIT.
    slat_vae_checkpoint_step (int): Which step of checkpoint to load for SLat VAE.
    slat_dit_checkpoint_step (int): Which step of checkpoint to load for SLat DIT.
    
    Returns:
    TempImageTo3DPipeline: The initialized pipeline with the loaded models.
    """
    # Looad the first stage trained models (sparse structure VAE and DIT)
    ss_vae_decoder_ckpt_path = os.path.join(os.path.dirname(ss_vae_config_filepath), 'ckpts', f"decoder_step{ss_vae_checkpoint_step:07d}.pt")
    ss_dit_ckpt_path = os.path.join(os.path.dirname(ss_dit_config_filepath), 'ckpts', f"denoiser_step{ss_dit_checkpoint_step:07d}.pt")
    ss_encoder, ss_decoder, ss_denoiser = load_ss_vae_and_dit(ss_vae_config_filepath, ss_vae_decoder_ckpt_path, ss_dit_config_filepath, ss_dit_ckpt_path)
    
    # load second stage trained models (SLat VAE and DIT)
    slat_vae_decoder_ckpt_path = os.path.join(os.path.dirname(slat_vae_config_filepath), 'ckpts', f"decoder_step{slat_vae_checkpoint_step:07d}.pt")
    slat_denoiser_ckpt_path = os.path.join(os.path.dirname(slat_denoiser_config_filepath), 'ckpts', f"denoiser_step{slat_dit_checkpoint_step:07d}.pt")
    slat_decoder, slat_denoiser = load_slat_vae_and_dit(slat_vae_config_filepath, slat_vae_decoder_ckpt_path, slat_denoiser_config_filepath, slat_denoiser_ckpt_path)
    
    if dav3_ckpt_filepth is not None:
        dav3_model = DepthAnything3.from_pretrained(dav3_ckpt_filepth).cuda()
    else:
        dav3_model = None
    # initialize sampler
    sampler_args_dict = {
        "ss_sampler": {
            "name": "FlowEulerGuidanceIntervalSampler",
            "args": {"sigma_min": 1e-5},
            "params": {"steps": 25, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0},
        },
        "slat_sampler": {
            "name": "FlowEulerGuidanceIntervalSampler",
            "args": {"sigma_min": 1e-5},
            "params": {"steps": 25, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0},
        },
        "slat_normalization": {
            "mean": [
                -1.955322504043579,
                -0.17805132269859314,
                0.07887908816337585,
                1.3940876722335815,
                -1.1487646102905273,
                -0.014329985715448856,
                -1.7741789817810059,
                1.7416635751724243
            ],
            "std": [
                2.8198282718658447,
                2.0864219665527344,
                2.232968330383301,
                2.486429452896118,
                3.008762836456299,
                2.6273839473724365,
                2.5930721759796143,
                2.6468937397003174
            ],
        },
    }
    ss_sampler = getattr(samplers, sampler_args_dict["ss_sampler"]["name"])(**sampler_args_dict["ss_sampler"]["args"])
    slat_sampler = getattr(samplers, sampler_args_dict["slat_sampler"]["name"])(**sampler_args_dict["slat_sampler"]["args"])
    pipeline = TempImageTo3DPipeline(
        ss_sampler=ss_sampler,
        ss_sampler_params=sampler_args_dict["ss_sampler"]["params"],
        ss_encoder=ss_encoder,
        ss_decoder=ss_decoder,
        ss_flow_model=ss_denoiser,
        slat_decoder_gs_model=slat_decoder,
        slat_flow_model=slat_denoiser,
        slat_sampler=slat_sampler,
        slat_sampler_params=sampler_args_dict["slat_sampler"]["params"],
        slat_normalization=sampler_args_dict["slat_normalization"],
        localdino_repo_path="/path/to/your/local/dinov2",
        device=torch.device("cuda"),
        dav3_model=dav3_model,
    )
    
    return pipeline


def run_single_image_to_coarse_gs(
    pipeline: TempImageTo3DPipeline,
    cond_img: Image.Image,
    cam_w2c_poses: Float[Tensor, "N 4 4"],
    intrinsics: Float[Tensor, "N 3 3"],
    seed: int = 43,
    save_coarse_gs: bool = True,
    output_dir: str = './trellis_img2scene_output',
):
    """ Run the image to 3D pipeline on a single image and save the coarse gaussian splats and rendered videos.
    
    Args:
    pipeline (TempImageTo3DPipeline): The initialized image to 3D pipeline.
    cond_img (Image.Image): The conditioning image.
    cam_w2c_poses (Float[Tensor, "N 4 4"]): The camera world to camera extrinsic poses for rendering.
    intrinsics (Float[Tensor, "N 3 3"]): The camera intrinsics for rendering. Note: normalized intrinsics;
    seed (int): The random seed for sampling.
    save_coarse_gs (bool): Whether to save the coarse gaussian splats as ply files.
    output_dir (str): The directory to save the outputs.
    
    Returns:
    Tuple[str, str]: The file paths to the saved coarse RGB video and coarse depth video.
    """
    
    output = pipeline.run_sample(
        image=cond_img,
        seed=seed,
        formats=["gaussian"],
        gt_voxel_coords=None,
        output_dir=output_dir,
    )
    decoded_gaussians = output["gaussian"]

    # Render video frames
    video_extrinsics = cam_w2c_poses.cuda()
    intrinsics = intrinsics.cuda()
    image_resolution = cond_img.size[0]  # (W, H)
    background_color = (0, 0, 0)
    rendering_results = render_utils.render_frames(
        sample=decoded_gaussians[0],
        extrinsics=video_extrinsics,
        intrinsics=intrinsics,
        options={"resolution": image_resolution, "bg_color": background_color},
    )
    print(f"Rendered video frames with color shape: {len(rendering_results['color'])} x {rendering_results['color'][0].shape}, depth shape: {len(rendering_results['depth'])} x {rendering_results['depth'][0].shape}")
    rgb_frames = np.array(rendering_results['color']).astype(np.uint8)  # (N, H, W, 3)
    depth_frames = np.array(rendering_results['depth']).astype(np.float32)  # (N, 1, H, W)
    
    coarse_rgb_video_filepath = os.path.join(output_dir, "video_rgb_coarse.mp4")
    imageio.mimsave(coarse_rgb_video_filepath, rgb_frames, fps=15, quality=10)
    
    filter_depth_frames = []
    for i in range(depth_frames.shape[0]):
        depth_frame = depth_frames[i][0]
        depth_frame[np.isinf(depth_frame)] = 0.0
        depth_frame[depth_frame < 0] = 0.0
        filter_depth_frames.append(depth_frame)
    depth_frames = np.stack(filter_depth_frames, axis=0)
    depths_min = np.min(depth_frames)
    depths_max = np.max(depth_frames)
    depths = (depth_frames - depths_min) / (depths_max - depths_min + 1e-8)
    depth_images = (depths * 255.0).clip(0, 255).astype(np.uint8)
    depth_images = depth_images[..., np.newaxis].repeat(3, axis=-1)  # convert to 3-channel depth image
    coarse_depth_video_filepath = os.path.join(output_dir, "video_depth_coarse.mp4")
    imageio.mimsave(coarse_depth_video_filepath, depth_images, fps=15, quality=10)
    
    # Save Gaussians as PLY files
    if save_coarse_gs:
        decoded_gaussians[0].save_ply(f"{output_dir}/sample.ply")
        
    return coarse_rgb_video_filepath, coarse_depth_video_filepath

def run_single_image_to_coarse_gs_with_dav3(
    pipeline: TempImageTo3DPipeline,
    cond_img: Image.Image,
    cam_w2c_poses: Float[Tensor, "N 4 4"],
    intrinsics: Float[Tensor, "N 3 3"],
    scene_scale: float = None,
    seed: int = 43,
    save_coarse_gs: bool = True,
    return_scm: bool = False,
    scale_cam_poses=False,  # whether to scale w2c poses to match the scene scale
    output_dir: str = './trellis_img2scene_output',
) -> Tuple[str, str, np.ndarray]:
    """ Run the image to 3D pipeline on a single image and save the coarse gaussian splats and rendered videos.
    
    Args:
    pipeline (TempImageTo3DPipeline): The initialized image to 3D pipeline.
    cond_img (Image.Image): The conditioning image.
    cam_w2c_poses (Float[Tensor, "N 4 4"]): The camera world to camera extrinsic poses for rendering.
    intrinsics (Float[Tensor, "N 3 3"]): The camera intrinsics for rendering. Note: normalized intrinsics;
    seed (int): The random seed for sampling.
    save_coarse_gs (bool): Whether to save the coarse gaussian splats as ply files.
    return_scm (bool): Whether to return the scene coordinate map instead of depth.
    output_dir (str): The directory to save the outputs.
    
    Returns:
    Tuple[str, str, np.ndarray]: The file paths to the saved coarse RGB video, coarse depth video, and the reference depth maps [Nv, H, W, 3].
    """
    logger.info("Running pipeline with DAV3 conditioning, image size: {}x{}".format(cond_img.size[0], cond_img.size[1]))
    cond_img_c2w_pose = (cam_w2c_poses[0].clone()).inverse()  # use the first camera pose as the conditioning image pose
    cond_img_intrinsic = intrinsics[0].clone()  # use the first camera intrinsic as the conditioning image intrinsic
    output = pipeline.run_sample_with_dav3(
        image=cond_img,
        input_rgb_poses=cond_img_c2w_pose.unsqueeze(0),
        input_rgb_intrinsic=cond_img_intrinsic.unsqueeze(0),
        scene_center=None,
        scene_scale=scene_scale,
        seed=seed,
        formats=["gaussian"],
        gt_voxel_coords=None,
        output_dir=output_dir,
    )
    decoded_gaussians = output["gaussian"]

    # scale the input camera poses to match the scene scale if needed
    if scale_cam_poses and output.get("scene_scale", None) is not None and output.get("scene_center", None) is not None:
        scene_scale = output["scene_scale"]
        scene_center = output["scene_center"]
        logger.info(f"Scaling camera poses by scene scale: {scene_scale}, scene_center: {scene_center}")
        cam_c2w_poses = cam_w2c_poses.clone().inverse()  # convert to c2w

        pose_scale_mat = torch.eye(4).to(cam_c2w_poses)
        # pose_scale_mat[:3, 3] = -torch.from_numpy(scene_center).to(cam_c2w_poses)
        pose_scale_mat[:3] *= 1.0 / (2.0 * scene_scale)
        for id in range(cam_c2w_poses.shape[0]):
            # scale pose_c2w
            subview_pose = pose_scale_mat @ cam_c2w_poses[id]
            R_c2w = (subview_pose[:3, :3]).detach().cpu().numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            subview_pose[:3, :3] = torch.from_numpy(R_c2w).to(cam_c2w_poses)
            cam_c2w_poses[id] = subview_pose
        cam_w2c_poses = cam_c2w_poses.inverse()  # convert back to w2c
        # # debug camera trajectory
        from utils.vis_util import vis_camera_keyframe
        import open3d as o3d
        for idx in range(0, cam_c2w_poses.shape[0], 10):
            cam_keyframe = vis_camera_keyframe(fx=256, fy=256, width=512, height=512, c2w=cam_c2w_poses[idx].cpu().numpy())
            o3d.io.write_line_set(os.path.join(output_dir, f"cam_keyframe_{idx:03d}.ply"), cam_keyframe)

    # Render video frames
    video_extrinsics = cam_w2c_poses.cuda()
    intrinsics = intrinsics.cuda()
    image_resolution = cond_img.size[0]  # (W, H)
    background_color = (0, 0, 0)
    rendering_results = render_utils.render_frames(
        sample=decoded_gaussians[0],
        extrinsics=video_extrinsics,
        intrinsics=intrinsics,
        options={"resolution": image_resolution, "bg_color": background_color},
    )
    print(f"Rendered video frames with color shape: {len(rendering_results['color'])} x {rendering_results['color'][0].shape}, depth shape: {len(rendering_results['depth'])} x {rendering_results['depth'][0].shape}")
    rgb_frames = np.array(rendering_results['color']).astype(np.uint8)  # (N, H, W, 3)
    depth_frames = np.array(rendering_results['depth']).astype(np.float32)  # (N, 1, H, W)
    
    coarse_rgb_video_filepath = os.path.join(output_dir, "video_rgb_coarse.mp4")
    imageio.mimsave(coarse_rgb_video_filepath, rgb_frames, fps=15, quality=10)
    
    filter_depth_frames = []
    for i in range(depth_frames.shape[0]):
        depth_frame = depth_frames[i][0]
        depth_frame[np.isinf(depth_frame)] = 0.0
        depth_frame[depth_frame < 0] = 0.0
        if not return_scm:
            filter_depth_frames.append(depth_frame)
        else:
            c2w = np.linalg.inv(video_extrinsics[i].cpu().numpy())
            intrinsic = intrinsics[i].cpu().numpy()
            intrinsic[0, :] *= cond_img.size[0]  # scale fx to image size
            intrinsic[1, :] *= cond_img.size[1]  # scale fy to image size
            scene_coord_map = unproject_depth_np(depth_map=depth_frame, 
                                        c2w=c2w, 
                                        intrinsic_mat=intrinsic)    # [-1, 1]
            # print(f"scene_coord_map shape: {scene_coord_map.shape}, min: {scene_coord_map.min()}, max: {scene_coord_map.max()}")
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # points = scene_coord_map.transpose(1, 2, 0).reshape(-1, 3)
            # colors = rgb_frames[i].reshape(-1, 3) / 255.0
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(f"{output_dir}/debug_scm_frame_{i}.ply", pcd)
            filter_depth_frames.append(scene_coord_map)
    depth_frames = np.stack(filter_depth_frames, axis=0)
    
    if not return_scm:
        depths_min = np.min(depth_frames)
        depths_max = np.max(depth_frames)
        depths = (depth_frames - depths_min) / (depths_max - depths_min + 1e-8)
        depth_images = (depths * 255.0).clip(0, 255).astype(np.uint8)
        depth_images = depth_images[..., np.newaxis].repeat(3, axis=-1)  # convert to 3-channel depth image
        coarse_depth_video_filepath = os.path.join(output_dir, "video_depth_coarse.mp4")
        imageio.mimsave(coarse_depth_video_filepath, depth_images, fps=15, quality=10)
    else:
        coarse_depth_video_filepath = os.path.join(output_dir, "video_scm_coarse.npz")
        np.savez(coarse_depth_video_filepath, scm=depth_frames)
    
    # Save Gaussians as PLY files
    if save_coarse_gs:
        decoded_gaussians[0].save_ply(f"{output_dir}/sample.ply")
        
    depth_maps_for_inputs: Float[Tensor, "Nv H W 1"] = output["depths_for_inputs"]
    if not return_scm:
        depth_maps_for_inputs = depth_maps_for_inputs.cpu().numpy()  # (Nv, H, W, 1)
        depth_maps_for_inputs = np.repeat(depth_maps_for_inputs, 3, axis=-1)  # (Nv, H, W, 3)
    else:
        scm_for_inputs = []
        for i in range(output["depths_for_inputs"].shape[0]):
            scaled_depth_map = depth_maps_for_inputs[i, :, :, 0].cpu().numpy()  # (H, W)
            c2w = np.linalg.inv(video_extrinsics[i].cpu().numpy())
            intrinsic = intrinsics[i].cpu().numpy()
            intrinsic[0, :] *= cond_img.size[0]  # scale fx to image size
            intrinsic[1, :] *= cond_img.size[1]  # scale fy to image size
            scene_coord_map = unproject_depth_np(depth_map=scaled_depth_map, 
                                        c2w=c2w, 
                                        intrinsic_mat=intrinsic)    # [-1, 1]
            scm_for_inputs.append(np.transpose(scene_coord_map, (1, 2, 0)))
        depth_maps_for_inputs = np.stack(scm_for_inputs, axis=0)  # (Nv, H, W, 3)
    logger.info(f"depth_maps_for_inputs shape: {depth_maps_for_inputs.shape}, min: {depth_maps_for_inputs.min()}, max: {depth_maps_for_inputs.max()}")
    return coarse_rgb_video_filepath, coarse_depth_video_filepath, depth_maps_for_inputs


if __name__ == "__main__":
    args = parse_args()
    
    selected_room_uid = args.scene_id
    dataset_rootpath = args.dataset_rootpath
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, selected_room_uid)
    os.makedirs(output_dir, exist_ok=True)
    
    # Looad the first stage trained models (sparse structure VAE and DIT)
    ss_vae_config_filepath = args.ss_vae_config_file
    ss_vae_ckpt_step = args.ss_vae_checkpoint_step
    ss_vae_decoder_ckpt_path = os.path.join(os.path.dirname(ss_vae_config_filepath), 'ckpts', f"decoder_step{ss_vae_ckpt_step:07d}.pt")
    ss_dit_config_filepath = args.ss_dit_config_file
    ss_dit_ckpt_step = args.ss_dit_checkpoint_step
    ss_dit_ckpt_path = os.path.join(os.path.dirname(ss_dit_config_filepath), 'ckpts', f"denoiser_step{ss_dit_ckpt_step:07d}.pt")
    
    ss_encoder, ss_decoder, ss_denoiser = load_ss_vae_and_dit(ss_vae_config_filepath, ss_vae_decoder_ckpt_path, ss_dit_config_filepath, ss_dit_ckpt_path)
    
    # load second stage trained models (SLat VAE and DIT)
    slat_vae_config_filepath = args.slat_vae_config_file
    slat_vae_ckpt_step = args.slat_vae_checkpoint_step
    slat_vae_decoder_ckpt_path = os.path.join(os.path.dirname(slat_vae_config_filepath), 'ckpts', f"decoder_step{slat_vae_ckpt_step:07d}.pt")
    slat_denoiser_config_filepath = args.slat_dit_config_file
    slat_denoiser_ckpt_step = args.slat_dit_checkpoint_step
    slat_denoiser_ckpt_path = os.path.join(os.path.dirname(slat_denoiser_config_filepath), 'ckpts', f"denoiser_step{slat_denoiser_ckpt_step:07d}.pt")
    
    slat_decoder, slat_denoiser = load_slat_vae_and_dit(slat_vae_config_filepath, slat_vae_decoder_ckpt_path, slat_denoiser_config_filepath, slat_denoiser_ckpt_path)

    
    # initialize sampler
    sampler_args_dict = {
        "ss_sampler": {
            "name": "FlowEulerGuidanceIntervalSampler",
            "args": {"sigma_min": 1e-5},
            "params": {"steps": 25, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0},
        },
        "slat_sampler": {
            "name": "FlowEulerGuidanceIntervalSampler",
            "args": {"sigma_min": 1e-5},
            "params": {"steps": 25, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0},
        },
        "slat_normalization": {
            "mean": [
                -1.955322504043579,
                -0.17805132269859314,
                0.07887908816337585,
                1.3940876722335815,
                -1.1487646102905273,
                -0.014329985715448856,
                -1.7741789817810059,
                1.7416635751724243
            ],
            "std": [
                2.8198282718658447,
                2.0864219665527344,
                2.232968330383301,
                2.486429452896118,
                3.008762836456299,
                2.6273839473724365,
                2.5930721759796143,
                2.6468937397003174
            ],
            # Slat feature stats on DL3DV dataset
            # "mean": [
            #     -2.0431807041168213,
            #     -0.12475071102380753,
            #     0.05865802243351936,
            #     1.3303190469741821,
            #     -1.11150324344635,
            #     0.058131489902734756,
            #     -1.7424697875976562,
            #     1.6439416408538818
            # ],
            # "std": [
            #     2.6373584270477295,
            #     2.104987382888794,
            #     2.259613275527954,
            #     2.465149402618408,
            #     2.882244825363159,
            #     2.604531764984131,
            #     2.4391591548919678,
            #     2.533987283706665
            # ],
        },
    }
    ss_sampler = getattr(samplers, sampler_args_dict["ss_sampler"]["name"])(**sampler_args_dict["ss_sampler"]["args"])
    slat_sampler = getattr(samplers, sampler_args_dict["slat_sampler"]["name"])(**sampler_args_dict["slat_sampler"]["args"])
    pipeline = TempImageTo3DPipeline(
        ss_sampler=ss_sampler,
        ss_sampler_params=sampler_args_dict["ss_sampler"]["params"],
        ss_encoder=ss_encoder,
        ss_decoder=ss_decoder,
        ss_flow_model=ss_denoiser,
        slat_decoder_gs_model=slat_decoder,
        slat_flow_model=slat_denoiser,
        slat_sampler=slat_sampler,
        slat_sampler_params=sampler_args_dict["slat_sampler"]["params"],
        slat_normalization=sampler_args_dict["slat_normalization"],
        localdino_repo_path="/path/to/experiments/dinov2",
        device=torch.device("cuda"),
    )

    # randomly sample a dataset item to get the voxel coords
    voxelized_64_feat_filepath = f"{dataset_rootpath}/features/{selected_room_uid}.npz"
    if os.path.exists(f"{dataset_rootpath}/renders_cond/{selected_room_uid}.npz"):
        cond_imgs_filepath = f"{dataset_rootpath}/renders_cond/{selected_room_uid}.npz"
        cond_data = np.load(cond_imgs_filepath)
        cond_imgs_array = cond_data["renders_cond"]
        cond_imgs_poses_c2w_blender = cond_data["transforms"]
        cond_imgs_x_angles = cond_data["x_angles"]
    # else:
    #     cond_imgs_filepath = f"{dataset_rootpath}/rgbs/{selected_room_uid}.npz"
    #     cond_imgs_array = np.load(cond_imgs_filepath)["rgbs"]
    voxelized_feats = np.load(voxelized_64_feat_filepath)
    gt_voxelized_coords = torch.cat(
        [
            torch.zeros(voxelized_feats["indices"].shape[0], 1).int(),
            torch.from_numpy(voxelized_feats["indices"]).int(),
        ],
        dim=1,
    ).cuda()
    print(f"Loaded voxelized coords from {voxelized_64_feat_filepath}, total num voxels: {gt_voxelized_coords.shape[0]}")
    # save gt sparse cooords
    utils3d.io.write_ply(f"{output_dir}/gt_sparse_structure.ply", voxelized_feats["indices"])

    cond_img_index = np.random.randint(cond_imgs_array.shape[0])
    cond_img_pil = Image.fromarray((cond_imgs_array[cond_img_index]).astype(np.uint8))
    cond_img_pil.save(f"{output_dir}/cond_img.png")
    
    # get intrinsic and extrinsic
    cond_imgs_poses_c2w = torch.from_numpy(cond_imgs_poses_c2w_blender).float()  # convert blender to enu
    cond_imgs_poses_c2w[:, :3, 1:3] *= -1
    cond_img_c2w_pose = cond_imgs_poses_c2w[cond_img_index]
    fov = cond_imgs_x_angles[cond_img_index].astype(np.float32)
    cond_img_intrinsic = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
    cond_img_intrinsic[0, :] *= cond_img_pil.size[0]  # adjust intrinsics to the image size
    cond_img_intrinsic[1, :] *= cond_img_pil.size[1]
    print(f"cond_img_intrinsic: {cond_img_intrinsic}, cond_img_c2w_pose: {cond_img_c2w_pose}")
    
    output = pipeline.run_sample(
        gt_voxel_coords=gt_voxelized_coords,
        image=cond_img_pil,
        seed=42,
        formats=["gaussian"],
        output_dir=output_dir,
    )
    ## get scene center and scale for normalization
    # scene_info_filepath = f"{dataset_rootpath}/world_cameras/{selected_room_uid}.npz"
    # scene_info_data = np.load(scene_info_filepath)
    # scene_center = scene_info_data["scene_center"]
    # scene_scale = scene_info_data["scene_scale"]
    # output = pipeline.run_sample_with_dav3(
    #     gt_voxel_coords=gt_voxelized_coords,
    #     image=cond_img_pil,
    #     input_rgb_poses=cond_img_c2w_pose.unsqueeze(0),
    #     input_rgb_intrinsic=cond_img_intrinsic.unsqueeze(0),
    #     scene_center=np.array(scene_center),
    #     scene_scale=float(scene_scale),
    #     seed=42,
    #     formats=["gaussian"],
    #     output_dir=output_dir,
    # )
    decoded_gaussians = output["gaussian"]
    # Render the outputs
    video = render_utils.render_video(decoded_gaussians[0])["color"]
    imageio.mimsave(f"{output_dir}/sample_gs.mp4", video, fps=30)

    # Save Gaussians as PLY files
    decoded_gaussians[0].save_ply(f"{output_dir}/sample.ply")