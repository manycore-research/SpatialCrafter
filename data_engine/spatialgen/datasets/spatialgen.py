import os
import sys

sys.path.append("..")
import os.path as osp
import math
import collections

import cv2
import torch
import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from PIL import Image
from scipy.spatial.transform import Rotation
from einops import rearrange

from utils.typing import *
from utils.file_util import load_from_json, load_from_txt
from utils.equilib import cube2equi, equi2pers
from utils.layout_util import room_meta_to_polygon, shrink_or_swell_polygon
from utils.pose_util import matrix_to_euler_angles_np, make_rotation_by_up_and_eye
from utils.geo_util import convert_distance_to_z, batch_unproject_depth_to_pointcloud

SPATIALGEN_DEPTH_MAX = 12.593009948730469


@staticmethod
def parse_room_polygon(room_metadata: Dict[str, Any], scale_factor: float = 0.001, shrink_factor: float = 0.10) -> Tuple[Polygon, float, float]:
    # parse original room layout
    ori_room_polygon = room_meta_to_polygon(room_meta_dict=room_metadata, SCALE=scale_factor)
    if ori_room_polygon.geom_type == "MultiPolygon":
        # take the polygon with the largest area
        areas = [poly.area for poly in ori_room_polygon.geoms]
        max_indice = np.argmax(areas)
        ori_room_polygon = ori_room_polygon.geoms[max_indice]
    # # show room layout polygon
    # from matplotlib import pyplot as plt
    # plt.figure()
    # xs, ys = ori_room_polygon.exterior.xy
    # plt.plot(xs, ys)

    # shrink the largest rectangle
    shrinked_room_polygon = shrink_or_swell_polygon(ori_room_polygon, shrink_factor=shrink_factor, swell=False)
    if shrinked_room_polygon.geom_type == "MultiPolygon":
        # take the polygon with the largest area
        areas = [poly.area for poly in shrinked_room_polygon.geoms]
        max_indice = np.argmax(areas)
        shrinked_room_polygon = shrinked_room_polygon.geoms[max_indice]

    shrinked_xs, shrinked_ys = shrinked_room_polygon.exterior.xy
    # plt.plot(shrinked_xs, shrinked_ys)
    # plt.savefig("roomlayout_polygons.png")

    # min max corners
    min_xs, min_ys = min(shrinked_xs), min(shrinked_ys)
    max_xs, max_ys = max(shrinked_xs), max(shrinked_ys)
    width = max_xs - min_xs
    length = max_ys - min_ys
    return shrinked_room_polygon, width, length, ori_room_polygon


def _load_koolai_valid_frame_ids(room_folderpath: str) -> List[str]:
    """load valid frame ids from room folder

    Args:
        room_folderpath (str): room folder path

    Returns:
        List[str]: valid frame ids
    """
    valid_frames_filepath = osp.join(room_folderpath, "valid_frames.txt")
    valid_frame_ids = [frame_id for frame_id in load_from_txt(valid_frames_filepath) if frame_id.strip() != ""]
    valid_frame_ids.sort(key=lambda x: int(x))
    return valid_frame_ids


@staticmethod
def _load_koolai_camera_pose(camera_meta_dict: Dict, idx: int) -> torch.Tensor:
    """load c2w pose from camera meta

    Args:
        camera_meta_dict (Dict): camera meta json
        idx (int): camera idx

    Returns:
        torch.Tensor: pose
    """
    cam_meta = camera_meta_dict[str(idx)]

    # w2c pose
    pose = np.array(cam_meta["camera_transform"]).reshape(4, 4)
    c2w_pose = np.linalg.inv(pose)
    c2w_pose = torch.from_numpy(c2w_pose).float()
    return c2w_pose


@staticmethod
def _load_pil_image(file_path: str) -> Float[Tensor, "1 C H W"]:
    """Load image , Pillow version"""
    img = Image.open(file_path)
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis].astype(np.float32)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def filter_cameras_outside_layout(room_meta_dict: Dict[str, Any], all_cam_keys: List[str], cameras_dict: Dict[str, Any]):

    cam_height = 1.2  # camera height, TODO: get from mean of all cameras
    # 1. load room layout, get the boundary of the room
    room_polygon, room_width, room_length, _ = parse_room_polygon(room_meta_dict, scale_factor=0.001, shrink_factor=0.10)
    room_centroid = np.array([room_polygon.centroid.x, room_polygon.centroid.y, cam_height]).astype(np.float32)
    # dist_to_centroid_max_thresh = 0.5  # ratio of distance to the centroid of the room
    # dist_to_centroid_min_thresh = 0.2  # ratio of distance to the centroid of the room
    # distance_max = dist_to_centroid_max_thresh * min(room_width, room_length)
    # distance_min = dist_to_centroid_min_thresh * min(room_width, room_length)

    # # show room layout and object bbox polygons
    # plt.figure()
    # xs, ys = ori_room_polygon.exterior.xy
    # plt.plot(xs, ys)
    # for obj_bbox_polygon in obj_bbox_polygons:
    #     xs, ys = obj_bbox_polygon.exterior.xy
    #     plt.plot(xs, ys)

    # load all cameras
    valid_cam_keys = []
    for cam_id_str in all_cam_keys:
        c2w_pose = _load_koolai_camera_pose(cameras_dict, int(cam_id_str))
        cam_pos_x = c2w_pose[0, 3]
        cam_pos_y = c2w_pose[1, 3]
        point_2d = Point(cam_pos_x, cam_pos_y)
        # plt.plot(cam_pos_x, cam_pos_y, "b*")

        # if the camera is inside the room, and the camera is not too far from the centroid
        if room_polygon.contains(point_2d):
            valid_cam_keys.append(cam_id_str)

    # # save figure
    # plt.savefig("room_and_object_bbox_polygons.png")
    if len(valid_cam_keys) == 0:
        valid_cam_keys = all_cam_keys
    return valid_cam_keys, room_centroid


def calc_sample_views_yaws(cameras_dict: Dict[str, Any], sample_cam_keys: List[str], room_centroid: np.ndarray):

    valid_camera_poses = [_load_koolai_camera_pose(cameras_dict, int(cam_id_str)) for cam_id_str in sample_cam_keys]
    valid_camera_poses = np.stack(valid_camera_poses, axis=0)

    dis_to_centroids = valid_camera_poses[:, :3, 3] - room_centroid
    dirs_to_centroid = dis_to_centroids / (np.linalg.norm(dis_to_centroids, axis=-1, keepdims=True) + 1e-6)

    ret_yaws = []
    # calculate the orientation between camera direction and centroid
    for idx, c2w_pose in enumerate(valid_camera_poses):
        dir_to_centroid = dirs_to_centroid[idx]
        c2w_R = c2w_pose[:3, :3]

        # construct new rotation matrix to make the camera direct to the centroid
        new_c2w_R = make_rotation_by_up_and_eye(up=np.array([0.0, 0.0, 1.0]), eye=dir_to_centroid)
        # rotation from c2w_R to new_c2w_R
        delta_R = np.dot(new_c2w_R.T, c2w_R)
        delta_angle_deg = matrix_to_euler_angles_np(delta_R) * 180 / np.pi
        delta_yaw_deg = -delta_angle_deg[2]
        ret_yaws.append(float(delta_yaw_deg))
        # print(f"camera {ret_keys[idx]} rotate angle: {delta_yaw_deg}")

    return ret_yaws


def sample_camera_positions(cameras_dict: Dict[str, Any], valid_pano_view_keys: List[str], num_sample_views: int = 8, max_sample_views: int = 20):
    """
    Sample cameras with grid sampling

    Args:
        camera_positions (np.ndarray): camera positions
        valid_indices (np.array): valid indices of the input camera positions
        num_sample_views (int): number of sample views
    """
    camera_positions = [_load_koolai_camera_pose(cameras_dict, int(view_key))[:3, 3].numpy() for view_key in valid_pano_view_keys]

    valid_indices = np.arange(len(camera_positions))
    if len(valid_indices) < num_sample_views:
        return valid_indices

    points = shapely.points(camera_positions)
    bounds = shapely.bounds(shapely.MultiPoint(points))
    center = [
        bounds[0] + (bounds[2] - bounds[0]) / 2,
        bounds[1] + (bounds[3] - bounds[1]) / 2,
    ]

    sampled_indices = []
    for step in [2, 1.5, 1, 0.75, 0.5, 0.25, 0.1]:
        grid_x = np.concatenate(
            (
                np.arange(center[0], bounds[0] - step, -step)[::-1],
                np.arange(center[0], bounds[2] + step, step)[1:],
            )
        )
        grid_y = np.concatenate(
            (
                np.arange(center[1], bounds[1] - step, -step)[::-1],
                np.arange(center[1], bounds[3] + step, step)[1:],
            )
        )

        grid_points = np.array(np.meshgrid(grid_x, grid_y, indexing="ij")).reshape(2, -1).T

        # Convert each grid point to a shapely box (cell) of size `step` centered at the grid point
        grid_boxes = [shapely.box(x - step / 2, y - step / 2, x + step / 2, y + step / 2) for x, y in grid_points]

        tree = shapely.STRtree(grid_boxes)
        _, box_indices = tree.query(points, predicate="within")

        sampled_indices = []
        for box_index in np.unique(box_indices):
            indices = np.where(box_indices == box_index)[0]
            indices = np.intersect1d(indices, valid_indices)
            sampled_index = np.random.choice(indices)
            sampled_indices.append(sampled_index)
        sampled_indices = np.array(sampled_indices)

        if len(sampled_indices) >= num_sample_views:
            print(f"    Sampled {len(sampled_indices)} views from {len(valid_indices)} positions at {step}m.")
            break

    if len(sampled_indices) < num_sample_views:
        # sampled enough views from the other views
        remaining_indices = np.setdiff1d(np.arange(len(camera_positions)), sampled_indices)
        remaining_indices = np.intersect1d(remaining_indices, valid_indices)
        supp_indices = np.random.choice(remaining_indices, num_sample_views - len(sampled_indices), replace=False)
        sampled_indices = np.concatenate((sampled_indices, supp_indices))

        print(f"    Sampled {num_sample_views} views (with {len(supp_indices)} supplementary) from {len(valid_indices)} positions.")

    return sampled_indices[:max_sample_views]


SubviewImage = collections.namedtuple("SubviewImage", ["id", "image", "yaw_angle", "pitch_angle", "roll_angle"])


def cubemap_to_persepctive_images(
    equi_img: Float[Tensor, "1 C H W"],
    subview_yaws_lst: List[float],
    subview_pitchs_lst: List[float],
    subview_rolls_lst: List[float],
    subview_fov_x: float = 90.0,
    image_type: str = "rgb",
    subview_img_resolution: int = 256,
    depth_scale: float = 4000.0,
    normalized_focal_length: float = 1.0,
) -> List[Float[Tensor, "C H W"]]:
    """
    Convert equirectangular image to perspective images
    Args:
        equi_img: pre-cached equirectangular image
        subview_yaws_lst: yaw angles of subviews
        subview_pitchs_lst: pitch angles of subviews
        subview_rolls_lst: roll angles of subviews
        subview_fov_x: fov of subviews
        image_type: image type, rgb, depth, normal, semantic
        subview_img_resolution: resolution of subviews
        depth_scale: scale of depth
        normalized_focal_length: normalized focal length

    Returns:
        List[Float[Tensor, "1 C H W"]]: list of perspective images
    """
    subview_img_lst = []
    interpolation_mode = "nearest" if image_type in ["depth", "normal"] else "bilinear"
    rotations_dict_lst = [
        {"roll": roll * np.pi / 180.0, "pitch": -pitch * np.pi / 180.0, "yaw": -yaw * np.pi / 180.0}
        for yaw, pitch, roll in zip(subview_yaws_lst, subview_pitchs_lst, subview_rolls_lst)
    ]
    num_pers_imgs = len(rotations_dict_lst)
    if image_type == "rgb":
        pers_img = equi2pers(
            equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
            rots=rotations_dict_lst,
            height=512,
            width=512,
            fov_x=subview_fov_x,
            mode=interpolation_mode,
        )  # [8, 3, 256, 256]
        if pers_img.shape[-2:] != (subview_img_resolution, subview_img_resolution):
            pers_img = [
                torch.from_numpy(
                    cv2.resize(
                        img.permute(1, 2, 0).cpu().numpy(),
                        (subview_img_resolution, subview_img_resolution),
                        interpolation=cv2.INTER_LINEAR,
                    )
                ).permute(2, 0, 1)[None, :, :, :]
                for img in pers_img
            ]
            pers_img = torch.cat(pers_img, dim=0)
        pers_img = pers_img.float() / 255.0
        pers_img = pers_img.clip(0.0, 1.0)
        subview_img_lst = [
            SubviewImage(
                id=idx,
                image=pers_img[idx],
                yaw_angle=subview_yaws_lst[idx],
                pitch_angle=subview_pitchs_lst[idx],
                roll_angle=subview_rolls_lst[idx],
            )
            for idx in range(num_pers_imgs)
        ]

    elif image_type == "depth":
        pers_img = equi2pers(
            equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
            rots=rotations_dict_lst,
            height=512,
            width=512,
            fov_x=subview_fov_x,
            mode=interpolation_mode,
        )  # [1, 1, 256, 256]
        if pers_img.shape[-2:] != (subview_img_resolution, subview_img_resolution):
            pers_img = [
                torch.from_numpy(
                    cv2.resize(
                        img.permute(1, 2, 0).cpu().numpy().squeeze(),
                        (subview_img_resolution, subview_img_resolution),
                        interpolation=cv2.INTER_NEAREST,
                    )
                )[None, None, :, :]
                for img in pers_img
            ]
            pers_img = torch.cat(pers_img, dim=0)
        pers_img = pers_img.float() / depth_scale
        pers_img = convert_distance_to_z(
            distance_map=pers_img, focal_length=normalized_focal_length, image_height=subview_img_resolution, image_width=subview_img_resolution
        )
        subview_img_lst = [
            SubviewImage(
                id=idx,
                image=pers_img[idx],
                yaw_angle=subview_yaws_lst[idx],
                pitch_angle=subview_pitchs_lst[idx],
                roll_angle=subview_rolls_lst[idx],
            )
            for idx in range(num_pers_imgs)
        ]

    elif image_type == "normal":
        pers_imgs = equi2pers(
            equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
            rots=rotations_dict_lst,
            height=512,
            width=512,
            fov_x=subview_fov_x,
            mode=interpolation_mode,
        )  # [1, 3, 256, 256]
        if pers_imgs.shape[-2:] != (subview_img_resolution, subview_img_resolution):
            pers_imgs = [
                torch.from_numpy(
                    cv2.resize(
                        img.permute(1, 2, 0).cpu().numpy(),
                        (subview_img_resolution, subview_img_resolution),
                        interpolation=cv2.INTER_NEAREST,
                    )
                ).permute(2, 0, 1)[None, :, :, :]
                for img in pers_imgs
            ]
            pers_imgs = torch.cat(pers_imgs, dim=0)
        for idx in range(num_pers_imgs):
            # convert normal to caemra space
            pers_img = pers_imgs[idx].permute(1, 2, 0).cpu().numpy()
            # convert rgb normal to [-1, 1]
            normal = np.clip((pers_img + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
            normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)
            # rotate reference view normal to subview
            R_z = Rotation.from_euler("z", rotations_dict_lst[idx]["roll"], degrees=False).as_matrix()
            R_x = Rotation.from_euler("x", -rotations_dict_lst[idx]["pitch"], degrees=False).as_matrix()
            R_y = Rotation.from_euler("y", -rotations_dict_lst[idx]["yaw"], degrees=False).as_matrix()
            R_ref_subview = R_y @ R_x @ R_z
            rot_normal = (normal.reshape(-1, 3) @ R_ref_subview).reshape(normal.shape)
            # save normal in camera space, flip to make +z upward
            pers_img = torch.from_numpy(rot_normal).permute(2, 0, 1).float()  # [3, 256, 256]
            # pers_img = pers_img.unsqueeze(0)  # [1, 3, 256, 256]
            subview_img_lst.append(
                SubviewImage(
                    id=idx,
                    image=pers_img,
                    yaw_angle=subview_yaws_lst[idx],
                    pitch_angle=subview_pitchs_lst[idx],
                    roll_angle=subview_rolls_lst[idx],
                )
            )

    elif image_type == "semantic":
        pers_img = equi2pers(
            equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
            rots=rotations_dict_lst,
            height=512,
            width=512,
            fov_x=subview_fov_x,
            mode=interpolation_mode,
        )  # [1, 3, 256, 256]
        pers_img = pers_img.float() / 255.0
        if pers_img.shape[-2:] != (subview_img_resolution, subview_img_resolution):
            pers_img = [
                torch.from_numpy(
                    cv2.resize(
                        img.permute(1, 2, 0).cpu().numpy(),
                        (subview_img_resolution, subview_img_resolution),
                        interpolation=cv2.INTER_NEAREST,
                    )
                ).permute(2, 0, 1)[None, :, :, :]
                for img in pers_img
            ]
            pers_img = torch.cat(pers_img, dim=0)
        subview_img_lst = [
            SubviewImage(
                id=idx,
                image=pers_img[idx],
                yaw_angle=subview_yaws_lst[idx],
                pitch_angle=subview_pitchs_lst[idx],
                roll_angle=subview_rolls_lst[idx],
            )
            for idx in range(num_pers_imgs)
        ]
    return subview_img_lst


def calc_subview_pose(
    T_c2w: Float[Tensor, "4 4"],
    T_cv_enu: Float[Tensor, "4 4"],
    yaw_angle: Float[Tensor, "..."],
    pitch_angle: Float[Tensor, "..."],
    roll_angle: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    """Calculate camera poses from room metadata

    Args:
        T_cv_enu: transformation matrix from cv to enu coordinate
        T_c2w: camera pose in enu coordinate
        rot_angle: rotation angle in degree

    Returns:
        Float[Tensor, "camera entry"]: camera pose
    """
    subview_pose = torch.eye(4).float()
    roll_angle = torch.deg2rad(torch.tensor([roll_angle]))
    pitch_angle = torch.deg2rad(torch.tensor([pitch_angle]))
    yaw_angle = torch.deg2rad(torch.tensor([yaw_angle]))
    R_z = torch.tensor(
        [
            [torch.cos(roll_angle), -torch.sin(roll_angle), 0],
            [torch.sin(roll_angle), torch.cos(roll_angle), 0],
            [0, 0, 1],
        ]
    )
    R_x = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(pitch_angle), -torch.sin(pitch_angle)],
            [0, torch.sin(pitch_angle), torch.cos(pitch_angle)],
        ]
    )
    R_y = torch.tensor(
        [
            [torch.cos(yaw_angle), 0, -torch.sin(yaw_angle)],
            [0, 1, 0],
            [torch.sin(yaw_angle), 0, torch.cos(yaw_angle)],
        ]
    )
    subview_pose[:3, :3] = R_y @ R_x @ R_z
    subview_pose = T_c2w @ T_cv_enu @ subview_pose
    return subview_pose


def sample_camera_keys(
    sample_room_path: str | None = None,
    min_sample_views: int = 8,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    room_meta_filepath = os.path.join(sample_room_path, "room_meta.json")

    # load camera poses
    room_metadata = load_from_json(room_meta_filepath)
    cameras_meta = room_metadata["cameras"]

    # filter panoramic views with room layout polygon
    total_pano_view_keys = _load_koolai_valid_frame_ids(sample_room_path)
    valid_pano_view_keys, room_centroid = filter_cameras_outside_layout(
        room_meta_dict=room_metadata, all_cam_keys=total_pano_view_keys, cameras_dict=cameras_meta
    )

    # sample viewpoints using grid_sampler
    sample_cam_indices = sample_camera_positions(
        cameras_dict=cameras_meta,
        valid_pano_view_keys=valid_pano_view_keys,
        num_sample_views=min_sample_views,
    )

    sample_keys = [valid_pano_view_keys[idx] for idx in sample_cam_indices]
    sample_yaws = calc_sample_views_yaws(cameras_dict=cameras_meta, sample_cam_keys=sample_keys, room_centroid=room_centroid)  # calc yaws
    sample_rolls = [float(0.0)] * len(sample_keys)
    # sample_pitchs = [float(0.0)] * len(sample_keys)
    sample_pitchs = [float(np.random.uniform(-60.0, 60.0)) for _ in range(len(sample_keys))]
    # print(f"sample_keys: {sample_keys}, sample_yaws: {sample_yaws}")

    return sample_keys, sample_yaws, sample_rolls, sample_pitchs


def resize_image(image: Float[Tensor, "1 3 H W"], image_type: str = "rgb", target_width: int = 512, target_height: int = 512) -> Float[Tensor, "1 3 H W"]:
    interpolation = cv2.INTER_NEAREST if image_type in ["depth", "normal"] else cv2.INTER_LINEAR
    if image.shape[-2:] != (target_height, target_width):
        if image_type == "depth":
            pers_img = torch.from_numpy(
                cv2.resize(
                    image[0].permute(1, 2, 0).cpu().numpy(),
                    (target_width, target_height),
                    interpolation=interpolation,
                )
            )[None, :, :]
        else:
            pers_img = torch.from_numpy(
                cv2.resize(
                    image[0].permute(1, 2, 0).cpu().numpy(),
                    (target_width, target_height),
                    interpolation=interpolation,
                )
            ).permute(2, 0, 1)
    else:
        pers_img = image[0]

    if image_type in ["rgb", "semantic"]:
        # normalize rgb to [-1, 1]
        pers_img = pers_img.float() / 255.0
        pers_img = pers_img.clip(0.0, 1.0)
    elif image_type in ["depth"]:
        pers_img = pers_img.float()
    elif image_type in ["normal"]:
        # convert normal to [-1, 1]
        normal_img = pers_img.permute(1, 2, 0).cpu().numpy()
        normal = np.clip((normal_img + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
        normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)
        # save normal in camera space, flip to make +z upward
        pers_img = torch.from_numpy(normal).permute(2, 0, 1).float()
    return pers_img.unsqueeze(0)


def sample_perspective_data(
    sample_room_path: str | None = None,
    depth_scale: float = 4000.0,
    image_width: int = 512,
    image_height: int = 512,
    sample_keys: List[str] | None = None,
    sample_yaws: List[float] | None = None,
    sample_rolls: List[float] | None = None,
    sample_pitchs: List[float] | None = None,
    num_persp_views: int = 16,
    device: torch.device = torch.device("cpu"),
) -> Dict:

    room_uid = "/".join(sample_room_path.split("/")[-4:])
    rgb_dir = os.path.join(sample_room_path, "pano_rgb")
    depth_dir = os.path.join(sample_room_path, "pano_depth")
    normal_dir = os.path.join(sample_room_path, "pano_normal")
    semantic_dir = os.path.join(sample_room_path, "pano_semantic")
    room_meta_filepath = os.path.join(sample_room_path, "room_meta.json")

    # load camera poses
    room_metadata = load_from_json(room_meta_filepath)
    cameras_meta = room_metadata["cameras"]
    T_enu_cv = np.array(room_metadata["T_enu_cv"]).reshape(4, 4)
    T_cv_enu = torch.inverse(torch.from_numpy(T_enu_cv).float())

    # c2w poses, rgbs, background colors
    poses, rgbs = [], []
    depths, normals = [], []
    points = []
    mask_x_indices, mask_y_indices = [], []
    # TODO: make fov_x random
    random_fov_x = np.random.uniform(60.0, 120.0)  # random fov_x in degree
    # random_fov_x = 90.0
    focal_len = 0.5 * image_width / math.tan(random_fov_x * math.pi / 360.0)
    normalized_focal_len = 2.0 * focal_len / image_width
    # normalized OpenCV intrinsics matrix
    intrinsic_mat = (
        torch.tensor(
            [
                [focal_len, 0, image_width / 2.0],
                [0, focal_len, image_height / 2.0],
                [0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )

    for idx, view_key in enumerate(sample_keys):
        subview_yaw_degree = sample_yaws[idx]

        # load c2w pose
        c2w_pose = _load_koolai_camera_pose(cameras_meta, view_key)
        # load equirectangular images
        rgb_pano_image_filepath = os.path.join(rgb_dir, f"{view_key}.png")
        depth_pano_image_filepath = os.path.join(depth_dir, f"{view_key}.png")
        normal_pano_image_filepath = os.path.join(normal_dir, f"{view_key}.png")

        equi_rgb: Float[Tensor, "1 3 H W"] = _load_pil_image(rgb_pano_image_filepath).to(device)
        equi_depth: Float[Tensor, "1 1 H W"] = _load_pil_image(depth_pano_image_filepath).to(device)
        equi_normal: Float[Tensor, "1 3 H W"] = _load_pil_image(normal_pano_image_filepath).to(device)

        # subview degrees
        subview_yaw_degrees = (subview_yaw_degree + np.arange(0.0, 360.0, 360.0 / num_persp_views)).tolist()
        subview_yaw_degrees = [yaw % 360.0 for yaw in subview_yaw_degrees]  # [0, 360]
        subview_pitch_degrees = [sample_pitchs[idx]] * len(subview_yaw_degrees)
        subview_roll_degress = [sample_rolls[idx]] * len(subview_yaw_degrees)
        assert len(subview_yaw_degrees) == num_persp_views, f"len(subview_yaw_degrees) should be equal to {num_persp_views}!"

        rgb_subview_lst: List[Float[Tensor, "3 H W"]] = cubemap_to_persepctive_images(
            equi_img=equi_rgb,
            subview_fov_x=random_fov_x,
            subview_yaws_lst=subview_yaw_degrees,
            subview_pitchs_lst=subview_pitch_degrees,
            subview_rolls_lst=subview_roll_degress,
            subview_img_resolution=image_width,
            image_type="rgb",
        )

        depth_subview_lst: List[Float[Tensor, "1 H W"]] = cubemap_to_persepctive_images(
            equi_img=equi_depth,
            subview_fov_x=random_fov_x,
            subview_yaws_lst=subview_yaw_degrees,
            subview_pitchs_lst=subview_pitch_degrees,
            subview_rolls_lst=subview_roll_degress,
            subview_img_resolution=image_width,
            depth_scale=depth_scale,
            image_type="depth",
            normalized_focal_length=normalized_focal_len,
        )

        normal_subview_lst: List[Float[Tensor, "3 H W"]] = cubemap_to_persepctive_images(
            equi_img=equi_normal,
            subview_fov_x=random_fov_x,
            subview_yaws_lst=subview_yaw_degrees,
            subview_pitchs_lst=subview_pitch_degrees,
            subview_rolls_lst=subview_roll_degress,
            subview_img_resolution=image_width,
            image_type="normal",
            normalized_focal_length=normalized_focal_len,
        )

        # rgb [0, 1], metric depth, normal [-1, 1]
        subview_rgbs = [rgb.image for rgb in rgb_subview_lst]
        subview_depths = [depth.image for depth in depth_subview_lst]
        subview_normals = [normal.image for normal in normal_subview_lst]

        subview_masks = [torch.logical_and(depth_map > 0, depth_map < SPATIALGEN_DEPTH_MAX).squeeze() for depth_map in subview_depths]
        subview_mask_indices = [torch.where(mask) for mask in subview_masks]
        subview_mask_xs = [m[1].detach().cpu().numpy() for m in subview_mask_indices]
        subview_mask_ys = [m[0].detach().cpu().numpy() for m in subview_mask_indices]

        subview_poses = [calc_subview_pose(c2w_pose, T_cv_enu, -rgb.yaw_angle, rgb.pitch_angle, rgb.roll_angle) for rgb in rgb_subview_lst]

        subview_poses = torch.stack(subview_poses, dim=0)  # 16, 4, 4
        subview_depths = torch.stack(subview_depths, dim=0)  # 16, 1 512, 512
        subview_rgbs = torch.stack(subview_rgbs, dim=0)  # 16, 3, 512, 512
        subview_normals = torch.stack(subview_normals, dim=0).to(device)  # 16, 3, 512, 512
        pointcloud: Float[Tensor, "1 Nv Np 3"] = batch_unproject_depth_to_pointcloud(
            depth_maps=subview_depths.permute(0, 2, 3, 1)[None, ...].to(device),
            c2w_poses=subview_poses[None, ...].to(device),
            intrinsic_mat=intrinsic_mat[None, ...].to(device),
        )
        subview_masks: Float[Tensor, "Nv Np"] = rearrange(torch.stack(subview_masks), "Nv H W -> Nv (H W)")
        valid_pointcloud: Float[Tensor, "Npp 3"] = pointcloud[0, subview_masks]

        poses.append(subview_poses.cpu())
        rgbs.append(subview_rgbs.detach().cpu())
        depths.append(subview_depths.detach().cpu())
        normals.append(subview_normals.detach().cpu())
        points.append(valid_pointcloud.detach().cpu())
        mask_x_indices += subview_mask_xs
        mask_y_indices += subview_mask_ys

    poses: Float[Tensor, "Nv 16 4 4"] = torch.stack(poses, dim=0)
    rgbs: Float[Tensor, "Nv 16 3 H W"] = torch.stack(rgbs, dim=0)
    depths: Float[Tensor, "Nv 16 1 H W"] = torch.stack(depths, dim=0)
    normals: Float[Tensor, "Nv 16 3 H W"] = torch.stack(normals, dim=0)
    points: Float[Tensor, "Npp 3"] = torch.cat(points, dim=0)

    # trannsform the pointcloud from camera coordinate to world coordinate
    # points_in_world = project_points(points, pano_poses.to(device))

    data = {
        "c2ws": poses,  # c2w poses, (Nv, 16, 4, 4)
        "images": rgbs,  # rgb images [0,1], (Nv, 16, 3, H, W)
        "depths": depths,  # metric depth maps, (Nv, 16, 1, H, W)
        "normals": normals,  # normal images [-1,1], (Nv, 16, 3, H, W)
        "pointclouds": points,  # pointcloud in world coordinate
        "mask_x_indices": mask_x_indices,  # list: [Nvx16, (HxW)]
        "mask_y_indices": mask_y_indices,  # list: [Nvx16, (HxW)]
        "fovs": np.array([np.deg2rad(random_fov_x)] * poses.shape[0] * poses.shape[1]),
    }
    return data


def sample_conditional_data(
    sample_room_path: str | None = None,
    depth_scale: float = 4000.0,
    image_width: int = 512,
    image_height: int = 512,
    sample_keys: List[str] | None = None,
    sample_yaws: List[float] | None = None,
    sample_rolls: List[float] | None = None,
    sample_pitchs: List[float] | None = None,
    num_persp_views: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Sample perspective images and corresponding camera poses from SpatialGen dataset

    Args:
        sample_room_path: path to the room folder
        image_width: width of the perspective images
        image_height: height of the perspective images
        sample_keys: list of pano view keys to sample from
        sample_yaws: list of yaw angles for each pano view key
        sample_rolls: list of roll angles for each pano view key
        sample_pitchs: list of pitch angles for each pano view key
        num_persp_views: number of perspective views to sample from each pano view
        device: torch device
    Returns:
        data: dict containing the following keys:
            "c2ws": camera poses, (Nv, num_persp_views, 4, 4)
            "images": rgb images [0,1], (Nv, num_persp_views, 3, H, W)
            "fovs": fov angles in radian, (Nv*num_persp_views,)
    """

    room_uid = "/".join(sample_room_path.split("/")[-4:])
    rgb_dir = os.path.join(sample_room_path, "pano_rgb")
    depth_dir = os.path.join(sample_room_path, "pano_depth")
    room_meta_filepath = os.path.join(sample_room_path, "room_meta.json")

    # load camera poses
    room_metadata = load_from_json(room_meta_filepath)
    cameras_meta = room_metadata["cameras"]
    T_enu_cv = np.array(room_metadata["T_enu_cv"]).reshape(4, 4)
    T_cv_enu = torch.inverse(torch.from_numpy(T_enu_cv).float())

    # c2w poses, rgbs, depths
    poses, rgbs, depths = [], [], []
    points = []
    mask_x_indices, mask_y_indices = [], []
    # TODO: make fov_x random
    random_fov_x = 90.0
    focal_len = 0.5 * image_width / math.tan(random_fov_x * math.pi / 360.0)
    normalized_focal_len = 2.0 * focal_len / image_width
    # normalized OpenCV intrinsics matrix
    intrinsic_mat = (
        torch.tensor(
            [
                [focal_len, 0, image_width / 2.0],
                [0, focal_len, image_height / 2.0],
                [0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )

    for idx, view_key in enumerate(sample_keys):
        subview_yaw_degree = sample_yaws[idx]

        # load c2w pose
        c2w_pose = _load_koolai_camera_pose(cameras_meta, view_key)
        # load equirectangular images
        rgb_pano_image_filepath = os.path.join(rgb_dir, f"{view_key}.png")
        depth_pano_image_filepath = os.path.join(depth_dir, f"{view_key}.png")

        equi_rgb: Float[Tensor, "1 3 H W"] = _load_pil_image(rgb_pano_image_filepath).to(device)
        equi_depth: Float[Tensor, "1 1 H W"] = _load_pil_image(depth_pano_image_filepath).to(device)

        # subview degrees
        subview_yaw_degrees = (subview_yaw_degree + np.arange(0.0, 360.0, 360.0 / num_persp_views)).tolist()
        subview_yaw_degrees = [yaw % 360.0 for yaw in subview_yaw_degrees]  # [0, 360]
        subview_pitch_degrees = [sample_pitchs[idx]] * len(subview_yaw_degrees)
        subview_roll_degress = [sample_rolls[idx]] * len(subview_yaw_degrees)
        assert len(subview_yaw_degrees) == num_persp_views, f"len(subview_yaw_degrees) should be equal to {num_persp_views}!"

        rgb_subview_lst: List[Float[Tensor, "3 H W"]] = cubemap_to_persepctive_images(
            equi_img=equi_rgb,
            subview_fov_x=random_fov_x,
            subview_yaws_lst=subview_yaw_degrees,
            subview_pitchs_lst=subview_pitch_degrees,
            subview_rolls_lst=subview_roll_degress,
            subview_img_resolution=image_width,
            image_type="rgb",
        )
        depth_subview_lst: List[Float[Tensor, "1 H W"]] = cubemap_to_persepctive_images(
            equi_img=equi_depth,
            subview_fov_x=random_fov_x,
            subview_yaws_lst=subview_yaw_degrees,
            subview_pitchs_lst=subview_pitch_degrees,
            subview_rolls_lst=subview_roll_degress,
            subview_img_resolution=image_width,
            depth_scale=depth_scale,
            image_type="depth",
            normalized_focal_length=normalized_focal_len,
        )
        # rgb [0, 1],
        subview_rgbs = [rgb.image for rgb in rgb_subview_lst]
        subview_depths = [depth.image for depth in depth_subview_lst]

        subview_masks = [torch.logical_and(depth_map > 0, depth_map < SPATIALGEN_DEPTH_MAX).squeeze() for depth_map in subview_depths]
        subview_mask_indices = [torch.where(mask) for mask in subview_masks]
        subview_mask_xs = [m[1].detach().cpu().numpy() for m in subview_mask_indices]
        subview_mask_ys = [m[0].detach().cpu().numpy() for m in subview_mask_indices]
        
        subview_poses = [calc_subview_pose(c2w_pose, T_cv_enu, -rgb.yaw_angle, rgb.pitch_angle, rgb.roll_angle) for rgb in rgb_subview_lst]

        subview_rgbs = torch.stack(subview_rgbs, dim=0)  # num_persp_views, 3, 512, 512
        subview_poses = torch.stack(subview_poses, dim=0)  # num_persp_views, 4, 4
        subview_depths = torch.stack(subview_depths, dim=0)  # num_persp_views, 1 512, 512

        pointcloud: Float[Tensor, "1 Nv Np 3"] = batch_unproject_depth_to_pointcloud(
            depth_maps=subview_depths.permute(0, 2, 3, 1)[None, ...].to(device),
            c2w_poses=subview_poses[None, ...].to(device),
            intrinsic_mat=intrinsic_mat[None, ...].to(device),
        )
        subview_masks: Float[Tensor, "Nv Np"] = rearrange(torch.stack(subview_masks), "Nv H W -> Nv (H W)")
        valid_pointcloud: Float[Tensor, "Npp 3"] = pointcloud[0, subview_masks]

        poses.append(subview_poses.cpu())
        rgbs.append(subview_rgbs.detach().cpu())
        depths.append(subview_depths.detach().cpu())
        points.append(valid_pointcloud.detach().cpu())
        mask_x_indices += subview_mask_xs
        mask_y_indices += subview_mask_ys

    poses: Float[Tensor, "Nv 1 4 4"] = torch.stack(poses, dim=0)
    rgbs: Float[Tensor, "Nv 1 3 H W"] = torch.stack(rgbs, dim=0)
    depths: Float[Tensor, "Nv 16 1 H W"] = torch.stack(depths, dim=0)
    points: Float[Tensor, "Npp 3"] = torch.cat(points, dim=0)
    
    data = {
        "c2ws": poses,  # c2w poses, (Nv, num_pers, 4, 4)
        "images": rgbs,  # rgb images [0,1], (Nv, num_pers, 3, H, W)
        "depths": depths,  # metric depth maps, (Nv, num_pers, 1, H, W)
        "pointclouds": points,  # pointcloud in world coordinate
        "mask_x_indices": mask_x_indices,  # list: [Nvxnum_pers, (HxW)]
        "mask_y_indices": mask_y_indices,  # list: [Nvxnum_pers, (HxW)]
        "intrinsic_mat": intrinsic_mat.cpu(),  # (3, 3)
        "fovs": np.array([np.deg2rad(random_fov_x)] * poses.shape[0] * poses.shape[1]),
    }
    return data
