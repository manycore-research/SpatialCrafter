# this script use multi-process framework to process the SpatialGen dataset through the following steps:
# Step 1: parse room_layout.json per-room.
# Step 2: select at least 16 pano-frames per-room.
# Step 3: prepare 8x16 perspective images from the selected pano-frames.
# Step 4: voxelize the point cloud of each room, voxel_size = 128.
# Step 5: Extract DINO features for each perspective image.
# Step 6: Encode Sparse Structures (sparse voxels) for each room.
# Step 7: Encode Structural Latent for each room.
# by Chuan Fang, 2025/09/04

import os
import sys

sys.path.append(".")
sys.path.append("..")
import gc
import json
import time
import argparse
import logging
import os.path as osp
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool

import torch

torch.set_num_threads(1)
import shapely
import trimesh
import numpy as np
import pandas as pd
from transformers.trainer_utils import set_seed
from torchvision import transforms
from einops import rearrange
from shapely import LinearRing
from PIL import Image

from utils.typing import *
from utils.filter_util import edge_filter
from utils.file_util import load_from_json, load_from_csv
from utils.geo_util import rgbd_to_pointcloud, batch_unproject_depth_to_pointcloud
from data_engine.dinov2.model import DINOv2
from data_engine.spatialgen.datasets.constants import SEMANTIC_ID_TO_CLASS_MAP
from data_engine.spatialgen.datasets.entity import WallEntity, BboxEntity, BBoxUtils, DoorEntity, WindowEntity
from data_engine.spatialgen.datasets.spatiallm import load_layout, compute_layout_center_and_scale, langseq_to_entities
from data_engine.spatialgen.datasets.language_sequence import LanguageSequence
from data_engine.spatialgen.datasets.spatialgen import (SPATIALGEN_DEPTH_MAX, 
                                                     sample_camera_keys, 
                                                     sample_perspective_data, 
                                                     sample_conditional_data)


# camera sampling
MIN_SAMPLE_VIEWS = 8
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        return super(NumpyEncoder, self).default(o)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SpatialGen data for Spatial3D")
    parser.add_argument(
        "--dataset_rootpath",
        type=str,
        default="/path/to/data/dataset",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="/path/to/data/dataset/SpatialGen/spatialgen_release_ids.csv",
    )
    parser.add_argument(
        "--dinov2_ckpt_path",
        type=str,
        default="/path/to/dinov2/dinov2_vitl14_reg4_pretrain.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/data/dataset/SpatialGen/processed_data",
    )
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--text_cond", action="store_true", help="whether to prepare text conditional data")
    parser.add_argument("--img_cond", action="store_true", help="whether to prepare image conditional data")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def get_room_structure(room_meta_dict: Dict, unit_scale: float = 1000.0, tolerance: float = 0.01):
    """
        Encode floor and ceil lines to WallEntity

    Args:
        room_meta_dict (Dict): room meta dict
        unit_scale (float, optional): unit scale. Defaults to 1000.0.
        tolerance (float, optional): tolerance. Defaults to 0.01.

    Returns:
        List[WallEntity]: wall entities
    """
    floor = room_meta_dict["floor"]
    ceil = room_meta_dict["ceil"]
    assert len(floor) == len(ceil), "floor and ceil must have the same length"

    floor_lines = []
    ceil_lines = []
    for floor_line, ceil_line in zip(floor, ceil):
        floor_start = floor_line["start"]
        floor_start = np.array([floor_start["x"], floor_start["y"], floor_start["z"]])
        floor_end = floor_line["end"]
        floor_end = np.array([floor_end["x"], floor_end["y"], floor_end["z"]])
        ceil_start = ceil_line["start"]
        ceil_start = np.array([ceil_start["x"], ceil_start["y"], ceil_start["z"]])
        ceil_end = ceil_line["end"]
        ceil_end = np.array([ceil_end["x"], ceil_end["y"], ceil_end["z"]])

        floor_lines.append([floor_start, floor_end])
        ceil_lines.append([ceil_start, ceil_end])

    floor_lines = np.array(floor_lines) / unit_scale
    ceil_lines = np.array(ceil_lines) / unit_scale

    heights = ceil_lines[:, :, 2] - floor_lines[:, :, 2]
    height = np.median(heights)
    floor_z = np.median(floor_lines[:, :, 2])

    floor_ring = LinearRing(floor_lines[:, 0, :2])
    if tolerance:
        floor_ring = floor_ring.simplify(tolerance=tolerance)
        floor_coords = np.array(floor_ring.coords)
        rolled_coords = np.roll(floor_coords[:-1], len(floor_coords) // 2, axis=0)
        floor_ring = LinearRing(rolled_coords)
        floor_ring = floor_ring.simplify(tolerance=tolerance)
    floor_coords = np.array(floor_ring.coords)  # the last coord is the same as the first

    # check if the order is clockwise
    if not floor_ring.is_ccw:
        floor_coords = np.flip(floor_coords, axis=0)

    num_walls = len(floor_coords) - 1
    wall_entities = []
    for wi in range(num_walls):
        wall_entity = WallEntity(
            {
                "id": wi,
                "a_x": floor_coords[wi][0],
                "a_y": floor_coords[wi][1],
                "a_z": floor_z,
                "b_x": floor_coords[wi + 1][0],
                "b_y": floor_coords[wi + 1][1],
                "b_z": floor_z,
                "height": height,
                "thickness": 0,
            }
        )
        wall_entities.append(wall_entity)
    return wall_entities


def get_object_bboxes(
    room_meta_dict: Dict,
    inst2sem_labels: Dict,
):
    """
        Encode object bboxes to BboxEntity

    Args:
        room_meta_dict (Dict): room meta dict
        inst2sem_labels (Dict): instance_id to semantic_id dict for each scene

    Returns:
        List[BboxEntity]: bbox entities
    """

    entities = []

    cam0_dict = room_meta_dict["cameras"]["0"]
    if "bboxes" not in cam0_dict:
        return []
    bbox_info_in_cam0 = cam0_dict["bboxes"]
    T_enu_cv = np.array(room_meta_dict["T_enu_cv"]).reshape(4, 4)
    T_cv_enu = np.linalg.inv(T_enu_cv)
    T_w2c0 = np.array(cam0_dict["camera_transform"]).reshape(4, 4)
    T_c02w = np.linalg.inv(T_w2c0)

    for bbox_info in bbox_info_in_cam0:
        box_id = bbox_info["id"]

        transform_matrix = np.array(bbox_info["transform"]).reshape(4, 4)
        box_sizes = np.array(bbox_info["size"])

        T_box2cam = T_cv_enu @ transform_matrix
        T_box2w = T_c02w @ T_box2cam

        try:
            scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(T_box2w)
        except Exception as e:
            print(f"matrix is singular, ignore this bbox: {box_id}")
            continue

        z_angle = angles[-1]
        box_sizes = box_sizes * scale
        class_id = inst2sem_labels.get(box_id, 65535)
        class_name = SEMANTIC_ID_TO_CLASS_MAP.get(class_id, "unknown")

        bbox_entity = BboxEntity(
            {
                "id": box_id,
                "class": class_name,
                "position_x": trans[0],
                "position_y": trans[1],
                "position_z": trans[2],
                "angle_z": z_angle,
                "scale_x": box_sizes[0],
                "scale_y": box_sizes[1],
                "scale_z": box_sizes[2],
            }
        )
        entities.append(bbox_entity)

    return entities


def save_layout_in_single_room(room_path: str):
    apartment_id = room_path.split("/")[-3]
    room_meta_filepath = os.path.join(room_path, "room_meta.json")
    root_data_dir = "/".join(room_path.split("/")[:-4])
    instance2semlabel_filepath = os.path.join(root_data_dir, "inst2segm", f"{apartment_id}.json")
    # load camera poses
    room_metadata = load_from_json(room_meta_filepath)
    instid2semid_metadata = load_from_json(instance2semlabel_filepath)
    logging.info(f"processing room: {room_path}, room_meta_filepath: {room_meta_filepath}, instance2semlabel_filepath: {instance2semlabel_filepath}")
    wall_entities = get_room_structure(room_meta_dict=room_metadata)
    all_bbox_entities = get_object_bboxes(room_meta_dict=room_metadata, inst2sem_labels=instid2semid_metadata)

    bbox_utils = BBoxUtils(wall_entities + all_bbox_entities)
    door_entities, window_entities = bbox_utils.extract_dw()

    bbox_utils = BBoxUtils(wall_entities + all_bbox_entities)
    valid_bbox_entities = bbox_utils.inside_wall()
    bbox_utils = BBoxUtils(wall_entities + valid_bbox_entities)
    object_entities = bbox_utils.extract_objects()

    # refine ids
    for ei, entity in enumerate(door_entities):
        entity.params["id"] = ei + DoorEntity.CLASS_ENUM
    for ei, entity in enumerate(window_entities):
        entity.params["id"] = ei + WindowEntity.CLASS_ENUM
    for ei, entity in enumerate(object_entities):
        entity.params["id"] = ei + BboxEntity.CLASS_ENUM

    output_dict = {
        WallEntity.LANGUAGE_STRING: [entity.params for entity in wall_entities],
        DoorEntity.LANGUAGE_STRING: [entity.params for entity in door_entities],
        WindowEntity.LANGUAGE_STRING: [entity.params for entity in window_entities],
        BboxEntity.LANGUAGE_STRING: [entity.params for entity in object_entities],
    }

    return output_dict


def process_layout_in_single_room(room_layout: Union[str, dict], scene_center: np.ndarray = None, scene_scale: float = None):
    lang_seq: LanguageSequence = load_layout(room_layout)
    if scene_center is None or scene_scale is None:
        # compute the center and scale of layout
        scene_center, scene_scale = compute_layout_center_and_scale(lang_seq)
        # print(f"Room  center: {scene_center}, scale: {scene_scale}")
    # normalize the layout to [-0.5, 0.5]
    lang_seq.translate(-scene_center)
    lang_seq.scale(1.0 / scene_scale)
    # convert to entities
    entities = langseq_to_entities(lang_seq)

    return entities, scene_center, scene_scale


def filter_points_outside_layout(
    points: Float[np.ndarray, "Npp 3"], entities: Dict[str, List[Dict[str, float]]], eps: float = 0.02, batch_size=10_000
) -> Float[np.ndarray, "Npp"]:
    """
        Filter points that are outside the layout

    Args:
        points: [Np, 3]
        entities: Dict[str, List[Dict[str, float]]], layout entities
        eps: float, layout polygon expansion.
        batch_size: int

    Returns:
        filtering masks: [Np,]
    """
    wall_points = []
    for wall_entity in entities["wall"]:
        point = [wall_entity["a_x"], wall_entity["a_y"]]
        wall_points.append(point)
    wall_points.append([entities["wall"][-1]["b_x"], entities["wall"][-1]["b_y"]])
    floor_ring = shapely.LinearRing(wall_points)
    floor_polygon = shapely.Polygon(list(floor_ring.coords))
    # expand the floor polygon by eps
    floor_polygon = floor_polygon.buffer(eps, join_style="mitre")

    # Process in batches to avoid memory issues
    masks = np.zeros(len(points), dtype=bool)
    # for i in tqdm(range(0, len(points), batch_size), desc="    Filtering points"):
    for i in range(0, len(points), batch_size):
        end = min(i + batch_size, len(points))

        batch_points = shapely.points(points[i:end, :2])
        batch_tree = shapely.STRtree(batch_points)
        batch_indices = batch_tree.query(floor_polygon, predicate="covers")
        mask = np.zeros(len(batch_points), dtype=bool)
        mask[batch_indices] = True
        masks[i:end] = mask

    return masks


def voxelize_features(
    points: Float[np.ndarray, "Npp 3"],
    features: Float[np.ndarray, "Nv H W C"],
    x_indices: List[Float[np.ndarray, "Np"]],
    y_indices: List[Float[np.ndarray, "Np"]],
    entities_dict: Dict[str, List[Dict[str, float]]],
    scene_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scene_scale: float = 1.0,
    scene_name: str = "room",
    batch_point_size: int = 10_000,
    voxel_res: int = 64,
    image_size: int = 512,
    is_cond_view: bool = False,
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

    # filter points that are not in the layout
    inside_mask = filter_points_outside_layout(points, entities_dict, eps=2.0 / voxel_res)
    print(f"    Filter {np.sum(inside_mask)} of {len(points)} points.")
    points = points[inside_mask]
    # if not is_cond_view:
    #     import open3d as o3d
    #     o3d_points = o3d.geometry.PointCloud()
    #     o3d_points.points = o3d.utility.Vector3dVector(points)
    #     # downsample for visualization
    #     o3d_points = o3d_points.voxel_down_sample(voxel_size=0.001)
    #     o3d.io.write_point_cloud(f"{scene_name}_global_normalized_points.ply", o3d_points)
    # else:
    #     import open3d as o3d
    #     o3d_points = o3d.geometry.PointCloud()
    #     o3d_points.points = o3d.utility.Vector3dVector(points)
    #     # downsample for visualization
    #     o3d_points = o3d_points.voxel_down_sample(voxel_size=0.005)
    #     o3d.io.write_point_cloud(f"{scene_name}_condview_normalized_points.ply", o3d_points)
    
    image_indices = image_indices[inside_mask]
    flat_x_indices = flat_x_indices[inside_mask]
    flat_y_indices = flat_y_indices[inside_mask]

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
    if is_cond_view:
        print(f"Conditional voxels shape: {voxels.shape}, voxel_feats shape: {voxel_feats.shape}, masks shape: {masks.shape}")
    else:
        print(f"Global voxels shape: {voxels.shape}, voxel_feats shape: {voxel_feats.shape}, masks shape: {masks.shape}")

    return voxels, voxel_feats, masks


def convert_pose_cv_to_blender(c2ws_cv: Float[Tensor, "N 4 4"]) -> np.ndarray:
    """
    Convert camera-to-world pose from CV convention to Blender convention.

    Args:
        c2ws_cv (np.ndarray): Camera-to-world pose in CV convention, shape (N, 4, 4).

    Returns:
        np.ndarray: Camera-to-world pose in Blender convention, shape (N, 4, 4).
    """
    new_c2ws_blender = []
    for id in range(c2ws_cv.shape[0]):
        c2w_pose = c2ws_cv[id].detach().cpu().numpy()
        t_c2w = c2w_pose[:3, 3]
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = c2w_pose[:3, :3]
        new_c2w[:3, 1:3] *= -1
        new_c2w[:3, 3] = t_c2w
        new_c2ws_blender.append(new_c2w)
    new_c2ws_blender = np.stack(new_c2ws_blender, axis=0)
    return new_c2ws_blender


def process_single_room(
    room_idx: int,
    split_dataframe: pd.DataFrame,
    data_root_path: str = None,
    dino_model: torch.nn.Module = None,
    device: torch.device = torch.device("cuda:0"),
    min_sample_views: int = 8,
    batch_size: int = 16,
    output_dir: str = None,
    text_cond: bool = False,
    img_cond: bool = False,
    b_use_cond_image_voxels: bool = True,
):
    row = split_dataframe.iloc[room_idx]
    scene_id = row["scene_id"]
    room_type = row["room_type"]
    room_uid = scene_id

    # 20240506_3FO4K5FY7KYD_room_513 to 20240506/3FO4K5FY7KYD/perspective/room_513
    room_relative_path = room_uid.replace("_room_", "/perspective/room_")
    room_relative_path = room_relative_path[:8] + "/" + room_relative_path[9:]
    room_data_dir = osp.join(data_root_path, room_relative_path)
    print(f"room_data_dir: {room_data_dir}")
    if not osp.exists(room_data_dir):
        print(f"Room {room_uid} does not exist")
        return {"voxel_num": 0, "img_num": 0}

    set_seed(room_idx)

    result_name = scene_id

    os.makedirs(output_dir, exist_ok=True)
    output_layout_dir = os.path.join(output_dir, "layouts")
    output_features_dir = os.path.join(output_dir, "features")
    output_voxels_dir = os.path.join(output_dir, "voxels")
    output_rgbs_dir = os.path.join(output_dir, "rgbs")
    output_normals_dir = os.path.join(output_dir, "normals")
    output_depths_dir = os.path.join(output_dir, "depths")
    output_conds_dir = os.path.join(output_dir, "renders_cond")
    output_cond_voxels_dir = os.path.join(output_dir, "cond_features")
    os.makedirs(output_layout_dir, exist_ok=True)
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_voxels_dir, exist_ok=True)
    os.makedirs(output_rgbs_dir, exist_ok=True)
    os.makedirs(output_normals_dir, exist_ok=True)
    os.makedirs(output_depths_dir, exist_ok=True)
    os.makedirs(output_conds_dir, exist_ok=True)
    os.makedirs(output_cond_voxels_dir, exist_ok=True)

    feat_npz_filepath = os.path.join(output_features_dir, f"{result_name}.npz")
    voxels_npz_filepath = os.path.join(output_voxels_dir, f"{result_name}.npz")
    rgbs_npz_filepath = os.path.join(output_rgbs_dir, f"{result_name}.npz")
    depths_npz_filepath = os.path.join(output_depths_dir, f"{result_name}.npz")
    normals_npz_filepath = os.path.join(output_normals_dir, f"{result_name}.npz")
    cond_npz_filepath = os.path.join(output_conds_dir, f"{result_name}.npz")
    cond_feat_npz_filepath = os.path.join(output_cond_voxels_dir, f"{result_name}.npz")
    if (
        osp.exists(feat_npz_filepath)
        and osp.exists(voxels_npz_filepath)
        and osp.exists(rgbs_npz_filepath)
        and osp.exists(depths_npz_filepath)
        and osp.exists(normals_npz_filepath)
        and osp.exists(cond_npz_filepath)
        and osp.exists(cond_feat_npz_filepath)
    ):
        print(f"Room {room_uid} has been processed, skip.")
        return {"voxel_num": 0, "img_num": 0}

    tic = time.time()

    try:
        # Step 1: parse room layout
        layout_entity_filepath = osp.join(room_data_dir, "room_layout_spatiallm.json")
        if not osp.exists(layout_entity_filepath):
            # convert to SceneScriptLanguage
            scene_lang_seq = save_layout_in_single_room(room_path=room_data_dir)
            with open(layout_entity_filepath, "w") as f:
                json.dump(scene_lang_seq, f, indent=4, cls=NumpyEncoder)
        # TODO: this scale might conflict with the scale of pointcloud
        normalized_entities, scene_center, scene_scale = process_layout_in_single_room(room_layout=layout_entity_filepath)
        output_layout_filepath = osp.join(output_layout_dir, f"{result_name}.json")
        with open(output_layout_filepath, "w") as f:
            json.dump(normalized_entities, f, indent=4, cls=NumpyEncoder)

        # Step 2: select at least 16 pano-frames in each room
        sample_keys, sample_yaws, sample_rolls, sample_pitchs = sample_camera_keys(sample_room_path=room_data_dir, min_sample_views=min_sample_views)

        # Step 3: prepare 8x16 perspective images from the selected pano-frames
        data_dict = sample_perspective_data(
            sample_room_path=room_data_dir,
            depth_scale=4000.0,
            image_width=IMAGE_SIZE,
            image_height=IMAGE_SIZE,
            sample_keys=sample_keys,
            sample_yaws=sample_yaws,
            sample_rolls=sample_rolls,
            sample_pitchs=sample_pitchs,
            num_persp_views=16,
            device=device,
        )
        data_dict["c2ws"] = rearrange(data_dict["c2ws"], "b n c d -> (b n) c d")
        rgb_images = rearrange(data_dict["images"], "b n c h w -> (b n) c h w")
        depth_images = rearrange(data_dict["depths"], "b n c h w -> (b n) c h w")
        normal_images = rearrange(data_dict["normals"], "b n c h w -> (b n) c h w")
        mask_x_indices: List[np.array] = data_dict["mask_x_indices"]
        mask_y_indices: List[np.array] = data_dict["mask_y_indices"]
        x_fovs = data_dict["fovs"]

        
        # Step 4. normalize camera poses
        if not isinstance(scene_center, torch.Tensor):
            scene_center = torch.from_numpy(scene_center)
            scene_scale = torch.tensor([scene_scale])
        num_views = data_dict["c2ws"].shape[0]
        cam_c2ws = data_dict["c2ws"]
        pose_scale_mat = torch.eye(4)
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
            # # convert to cv coordinate for later processing, swap y and z, and flip the new z to make +z upward
            # our2cv_transform = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32).to(cam_c2ws)
            # subview_pose = our2cv_transform @ subview_pose
            cam_c2ws[id] = subview_pose
        # # visualize the camera poses
        # from utils.vis_util import vis_camera_keyframe
        # import open3d as o3d
        # for i, cam_c2w in enumerate(cam_c2ws):
        #     cam_pose = cam_c2w.detach().cpu().numpy()
        #     # create a line set for visualization
        #     cam_lines = vis_camera_keyframe(fx=128, fy=128, width=256, height=256, c2w=cam_pose)
        #     o3d.io.write_line_set(f"{output_dir}/debug_cam_{i}.ply", cam_lines)
            
        # Step 5: extract DINO feature
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

        # Step 6: vooxelize the pointcloud and DINO feature
        dino_feats = np.concatenate(dino_feat_lst, axis=0)
        occu_voxel_coords, occu_voxel_feats, masks = voxelize_features(
            points=data_dict["pointclouds"].cpu().numpy(),
            features=dino_feats,
            x_indices=mask_x_indices,
            y_indices=mask_y_indices,
            entities_dict=normalized_entities,
            scene_center=scene_center.cpu().numpy(),
            scene_scale=scene_scale.cpu().numpy(),
            scene_name=result_name,
            voxel_res=VOXEL_RES,
            image_size=IMAGE_SIZE,
            is_cond_view=False,
        )
        # save the voxels for debugging
        import open3d as o3d
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(occu_voxel_coords.astype(np.float32))
        o3d.io.write_point_cloud(osp.join(output_voxels_dir, f"{result_name}_voxels.ply"), voxel_pcd)

        # return np image
        rgb_array = (rgb_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        depth_array = (depth_images.detach().cpu().permute(0, 2, 3, 1).numpy()).astype(np.float32) / scene_scale
        normal_array = (normal_images.detach().cpu().permute(0, 2, 3, 1).numpy()).astype(np.float32)
        del data_dict, dino_feats

        # convert cam_poses to Blender format
        new_cam_c2ws = convert_pose_cv_to_blender(cam_c2ws)

        # align the name with TRELLIS
        np.savez_compressed(
            file=feat_npz_filepath,
            indices=occu_voxel_coords.astype(np.int32),
            patchtokens=occu_voxel_feats,
            masks=masks,
            transforms=new_cam_c2ws,
            x_angles=x_fovs,
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
            file=normals_npz_filepath,
            normals=normal_array,
        )

        if text_cond:
            # TODO: prepare text condition data
            pass
        if not img_cond:
            return {"voxel_num": occu_voxel_coords.shape[0], "img_num": rgb_array.shape[0]}
        
        # apply perturbation to the camera pose and save the perturbed images
        perturb_sample_yaws = sample_yaws + np.random.uniform(-30, 30, size=len(sample_yaws))
        perturb_sample_pitchs = [float(0.0)] * len(perturb_sample_yaws)
        cond_data_dict = sample_conditional_data(
            sample_room_path=room_data_dir,
            image_width=IMAGE_SIZE,
            image_height=IMAGE_SIZE,
            sample_keys=sample_keys,
            sample_yaws=perturb_sample_yaws,
            sample_rolls=sample_rolls,
            sample_pitchs=perturb_sample_pitchs,
            num_persp_views=1,  # only use 1 perspective view at each pano-frame
            device=device,
        )
        original_cond_poses = rearrange(cond_data_dict["c2ws"], "b n c d -> (b n) c d")
        cond_rgbs = rearrange(cond_data_dict["images"], "b n c h w -> (b n) c h w")
        cond_depths = rearrange(cond_data_dict["depths"], "b n c h w -> (b n) c h w")
        cond_fovs = cond_data_dict["fovs"]
        intrinsic_mat = cond_data_dict["intrinsic_mat"]

        # scale the cond_poses
        cond_poses = original_cond_poses.clone()
        for id in range(cond_poses.shape[0]):
            # scale pose_c2w
            subview_pose = pose_scale_mat @ cond_poses[id]
            R_c2w = (subview_pose[:3, :3]).detach().cpu().numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            subview_pose[:3, :3] = torch.from_numpy(R_c2w).to(cond_poses)
            cond_poses[id] = subview_pose
        # convert cond_poses to Blender format
        new_cond_poses = convert_pose_cv_to_blender(cond_poses)

        del cond_data_dict

        cond_rgb_array = (cond_rgbs.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        np.savez_compressed(
            file=cond_npz_filepath,
            renders_cond=cond_rgb_array,
            transforms=new_cond_poses,
            x_angles=cond_fovs,
        )
        # prepare condition voxel_coords, features, and image masks
        cond_voxels, cond_voxel_feats, cond_masks = [], [], []
        if b_use_cond_image_voxels:
            scene_scale = scene_scale.cpu().numpy()
            scene_center = scene_center.cpu().numpy()
            
            for cond_idx, (cond_rgb, cond_c2w_pose, cond_depth) in enumerate(zip(cond_rgb_array, original_cond_poses, cond_depths)):
                cond_rgb: np.ndarray
                cond_c2w_pose: Float[Tensor, "4 4"]
                cond_depth: Float[Tensor, "C H W"]
                # Step 3: Generate world points from depth map
                cond_c2w_poses = cond_c2w_pose.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
                cond_intrinsics = intrinsic_mat.unsqueeze(0)              # (1, 3, 3)
                depth_maps: Float[Tensor, "Nv 1 H W"] = cond_depth.unsqueeze(0)  # (1, 1, H, W)
                if depth_maps.shape[1] != IMAGE_SIZE or depth_maps.shape[2] != IMAGE_SIZE:
                    depth_maps = torch.nn.functional.interpolate(depth_maps, 
                                                                size=(IMAGE_SIZE, IMAGE_SIZE), 
                                                                mode='nearest')  # (1, 1, H, W)
                pointcloud: Float[Tensor, "1 Nv Np 3"] = batch_unproject_depth_to_pointcloud(
                    depth_maps=depth_maps[..., None].to(device),
                    c2w_poses=cond_c2w_poses.to(device),
                    intrinsic_mat=cond_intrinsics.to(device),
                )
                # filter out invalid depth and edge
                subview_masks = [torch.logical_and(depth_map > 0, depth_map < SPATIALGEN_DEPTH_MAX).squeeze() for depth_map in depth_maps]
                subview_edge_masks = [ edge_filter(depth_map.squeeze().cpu().numpy(), times=0.3) for depth_map in depth_maps]
                subview_masks = [torch.logical_and(subview_masks[i], torch.from_numpy(1 - subview_edge_masks[i])) for i in range(len(subview_masks))]
                subview_mask_indices = [torch.where(mask) for mask in subview_masks]
                subview_mask_xs = [m[1].detach().cpu().numpy() for m in subview_mask_indices]  # List[Nv, (HxW)]
                subview_mask_ys = [m[0].detach().cpu().numpy() for m in subview_mask_indices]  # List[Nv, (HxW)]
                subview_masks: Float[Tensor, "Nv Np"] = rearrange(torch.stack(subview_masks), "Nv H W -> Nv (H W)")
                valid_pointcloud: Float[Tensor, "Npp 3"] = pointcloud[0, subview_masks]

                # save the point cloud as .ply file for visualization
                if False:
                    # savve cond_rgb for debugging
                    pil_cond_rgb = Image.fromarray(cond_rgb)
                    pil_cond_rgb.save(osp.join(output_cond_voxels_dir, f"cond_rgb_{cond_idx}.png"))
                    rgb_img = cond_rgb.astype(np.uint8)  # (H, W, 3)
                    o3d_color_points = rgbd_to_pointcloud(depth_image=depth_maps[0].permute(1, 2, 0).cpu().numpy(),
                                rgb_image=rgb_img,
                                c2w_pose=cond_c2w_poses[0, 0].cpu().numpy(),
                                depth_scale=1.0,
                                intrinsic_mat=cond_intrinsics[0].cpu().numpy(),
                    )
                    o3d.io.write_point_cloud(os.path.join(output_cond_voxels_dir, f"{result_name}_points_{cond_idx}.ply"), o3d_color_points)
                    
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
                    entities_dict=normalized_entities,
                    scene_center=scene_center,
                    scene_scale=scene_scale,                    
                    voxel_res=VOXEL_RES,
                    image_size=IMAGE_SIZE,
                    is_cond_view=True,
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
            # Note: cond_voxels and cond_voxel_feats might vary in shape, so using np.array(..., dtype=object).
            cond_voxels = np.array(cond_voxels_obj, dtype=object)
            cond_voxel_feats = np.array(cond_voxel_feats_obj, dtype=object)
            
            np.savez_compressed(
                file=cond_feat_npz_filepath,
                indices=cond_voxels,
                patchtokens=cond_voxel_feats,
                masks=cond_masks,
            )

        toc = time.time()
        print(f"Room  {room_uid}  completed in {toc - tic:.2f} seconds.")
        return {"voxel_num": occu_voxel_coords.shape[0], "img_num": cond_rgb_array.shape[0]}

    except Exception as e:
        print(f"Error processing room {room_uid}: {e}")
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
    logger.info(f"Task {args.task_id} starts from {args.start_index} to {args.end_index}. Text cond: {do_text_condition}, Image cond: {do_image_condition}")

    # filter out the room_type is balcony
    scenes_df = scenes_df[scenes_df["room_type"] != "balcony"]
    scenes_df = scenes_df[scenes_df["scene_id"].isin(["20240506_3FO4K5FY82OI_room_419"])]
    scenes_df = scenes_df.reset_index(drop=True)
    logger.info(f"After filtering, {len(scenes_df)} rooms left.")

    # load dinov2 model
    dinov2_model = DINOv2(dinov2_ckpt_path)
    dinov2_model.to(device)

    # process_single_room(
    #     room_idx=0,
    #     split_dataframe=scenes_df,
    #     data_root_path=data_root_path,
    #     dino_model=dinov2_model,
    #     device=device,
    #     min_sample_views=MIN_SAMPLE_VIEWS,
    #     batch_size=batch_size,
    #     output_dir=output_dir,
    #     text_cond=do_text_condition,
    #     img_cond=do_image_condition,
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
            dino_model=dinov2_model,  # dinov2 or dinov3
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            text_cond=do_text_condition,
            img_cond=do_image_condition,
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
