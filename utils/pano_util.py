import os

import cv2
import numpy as np
import open3d as o3d

def np_coorx2u(coorx: np.ndarray, coorW: int = 1024):
    # convert image coordinate to spherical coordinate
    # move coordinate to the center of pixel
    return (coorx - (coorW * 0.5) + 0.5) / coorW * 2 * np.pi


def np_coory2v(coory: np.ndarray, coorH: int = 512):
    # convert image coordinate to spherical coordinate
    # move coordinate to the center of pixel
    return (coory - (coorH * 0.5) + 0.5) / coorH * np.pi

def get_unit_spherical_map(h: int = 512, w: int = 1024) -> np.ndarray:
    coorx, coory = np.meshgrid(np.arange(w), np.arange(h))
    us = np_coorx2u(coorx, w)
    vs = np_coory2v(coory, h)

    X = np.expand_dims(np.cos(vs) * np.sin(us), 2)
    Y = np.expand_dims(np.cos(vs) * np.cos(us), 2)
    Z = np.expand_dims(np.sin(vs), 2)
    unit_map = np.concatenate([X, Y, Z], axis=2) * -1

    return unit_map

def pano_rgbd_pointcloud(rgb_img_filepath:str, depth_img_filepath:str, saved_color_pcl_filepath:str = None, depth_scale:float=1000.0, normaliz:bool=False)->o3d.geometry.PointCloud:
    """
    :param rgb_img_filepath: rgb panorama image filepath
    :param depth_img_filepath: depth panorama image filepath
    :param saved_color_pcl_filepath: saved color point cloud filepath
    :return: o3d.geometry.PointCloud
    """

    assert os.path.exists(rgb_img_filepath), 'rgb panorama doesnt exist!!!'
    assert os.path.exists(depth_img_filepath), 'depth panorama doesnt exist!!!'

    raw_depth_img = cv2.imread(depth_img_filepath, cv2.IMREAD_UNCHANGED)
    if len(raw_depth_img.shape) == 3:
        raw_depth_img = cv2.cvtColor(raw_depth_img, cv2.COLOR_BGR2GRAY)
    depth_img = np.asarray(raw_depth_img)
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        return None

    raw_rgb_img = cv2.imread(rgb_img_filepath, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(raw_rgb_img, cv2.COLOR_BGR2RGB)
    if rgb_img.shape[2] == 4:
        rgb_img = rgb_img[:, :, :3]
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        return None
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0
    # print(f'raw_rgb shape: {rgb_img.shape} color shape: {color.shape}, ')

    depth_img = np.expand_dims((depth_img / depth_scale), axis=2)
    # max_depth = np.max(depth_img)
    # if normaliz:
    #     depth_img = depth_img / max_depth
    #     print(f'max depth: {max_depth}')
    pointcloud = depth_img * get_unit_spherical_map(h=depth_img.shape[0], w=depth_img.shape[1])

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    if saved_color_pcl_filepath is not None:
        o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return o3d_pointcloud