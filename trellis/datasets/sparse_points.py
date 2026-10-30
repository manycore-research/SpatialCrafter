import os
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import utils3d
from torch.utils.data.dataloader import default_collate

from trellis.datasets.components import StandardDatasetBase
from trellis.renderers import OctreeRenderer
from trellis.representations.octree import DfsOctree as Octree
from trellis.datasets.point_transform import Compose
from utils.typing import *


class SparsePoints(StandardDatasetBase):
    """
    Sparse point cloud dataset

    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
        min_num_voxels (int): minimum number of voxels of the instances to be included in the dataset
        max_num_voxels (int): maximum number of voxels of the instances to be included in the dataset
    """

    def __init__(
        self,
        roots,
        resolution: int = 256,
        min_aesthetic_score: float = 5.0,
        min_num_voxels: int = 1024,
        max_num_voxels: int = 600000,
        transform_cfg: dict = None,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.min_num_voxels = min_num_voxels
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        if transform_cfg is not None:
            self.transform = Compose(transform_cfg)
        else:
            default_transform_config = [
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.00390625,  # 1/256, fixed for 256 voxel resolution
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                    return_inverse=True,
                    return_displacement=True,  # compute point displacement to voxel center
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "inverse"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ]
            self.transform = Compose(default_transform_config)

        super().__init__(roots)

    def filter_metadata(self, metadata):
        stats = {}
        stats["Voxelized"] = len(metadata)
        # metadata = metadata[(metadata["num_points"] >= self.min_num_voxels)]
        # stats[f"Num voxels >= {self.min_num_voxels}"] = len(metadata)
        # metadata = metadata[(metadata["num_points"] >= self.min_num_voxels) & (metadata["num_points"] <= self.max_num_voxels)]
        # stats[f"Num voxels <= {self.max_num_voxels}"] = len(metadata)
        
        # filter out instances with large grid_size
        # cause they stand for large room-scale scenes
        # metadata = metadata[(metadata["grid_size"] <= 0.05)]
        return metadata, stats

    def get_instance(self, root, instance):
        points_file = os.path.join(root, "points", f"{instance}.npz")
        colors_file = os.path.join(root, "colors", f"{instance}.npz")
        normals_file = os.path.join(root, "normals", f"{instance}.npz")
        
        data_dict = {}
        data_dict["color"] = np.load(colors_file, allow_pickle=True)["colors"]
        data_dict["normal"] = np.load(normals_file, allow_pickle=True)["normals"]
        with np.load(points_file, allow_pickle=True) as data:
            data_dict["coord"] = data["points"]
            data_dict["grid_size"] = data["grid_size"]
            
        return self.transform(data_dict)

    @staticmethod
    def collate_fn(batch: List):
        """
        collate function for point cloud which support dict and list,
        'coord' is necessary to determine 'offset'
        """
        if not isinstance(batch, Sequence):
            raise TypeError(f"{batch.dtype} is not supported.")

        if isinstance(batch[0], torch.Tensor):
            return torch.cat(list(batch))
        elif isinstance(batch[0], str):
            # str is also a kind of Sequence, judgement should before Sequence
            return list(batch)
        elif isinstance(batch[0], Sequence):
            for data in batch:
                data.append(torch.tensor([data[0].shape[0]]))
            batch = [SparsePoints.collate_fn(samples) for samples in zip(*batch)]
            batch[-1] = torch.cumsum(batch[-1], dim=0).int()
            return batch
        elif isinstance(batch[0], Mapping):
            batch = {
                key: (
                    SparsePoints.collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        SparsePoints.collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                        dim=0,
                    )
                )
                for key in batch[0]
            }
            return batch
        else:
            return default_collate(batch)
        
    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        ss = ss if isinstance(ss, torch.Tensor) else ss["ss"]

        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = "voxel"

        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        # yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        # yaws = [y + yaws_offset for y in yaws]
        # pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]
        pitch = [-np.pi / 4, -np.pi / 8, 0, np.pi / 4]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = (
                torch.tensor(
                    [
                        np.sin(yaw) * np.cos(pitch),
                        np.cos(yaw) * np.cos(pitch),
                        np.sin(pitch),
                    ]
                )
                .float()
                .cuda()
                * 2
            )
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []

        # Build each representation
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device="cuda",
                primitive="voxel",
                sh_degree=0,
                primitive_config={"solid": True},
            )
            coords = torch.nonzero(ss[i, 0], as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(self.resolution)),
                dtype=torch.uint8,
                device="cuda",
            )

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(
                    representation, ext, intr, colors_overwrite=representation.position
                )
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)

        return torch.stack(images)


if __name__ == "__main__":
    default_transform_config = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.00390625,  # 1/256, fixed for 256 voxel resolution
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
            return_min_coord=True,
            return_displacement=True,  # compute point displacement to voxel center
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse", "displacement", "min_coord"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]
    dataset = SparsePoints(
        roots="/path/to/experiments/trellis/debug_concerto_data_1128",
        resolution=256,
        transform_cfg=default_transform_config,
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Single sample:--------------------")
    print({k: v.shape for k, v in sample.items()})
    # visualize points and grid_coords
    import open3d as o3d
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(sample["coord"][:, :3].cpu().numpy())
    points_pcd.colors = o3d.utility.Vector3dVector(sample["color"][:, :3].cpu().numpy())
    o3d.io.write_point_cloud(f"points.ply", points_pcd)
    grid_coord_pcd = o3d.geometry.PointCloud()
    grid_coord_pcd.points = o3d.utility.Vector3dVector(sample["grid_coord"][:, :3].cpu().numpy())
    o3d.io.write_point_cloud(f"grid_coords.ply", grid_coord_pcd)
    # grid_coords + displacement should be identical to original points (after scaling and shifting)
    grid_coord_disp_pcd = o3d.geometry.PointCloud()
    disp = sample["displacement"] + 0.5  # shift xyz coord to [0, 1]
    grid_coord_disp = (sample["grid_coord"][:, :3] + disp[:, :3]).cpu().numpy()
    recovered_points = grid_coord_disp * 0.00390625  # [0, 1)
    recovered_points += sample["min_coord"].cpu().numpy()
    grid_coord_disp_pcd.points = o3d.utility.Vector3dVector(recovered_points)
    o3d.io.write_point_cloud(f"grid_coords_disp.ply", grid_coord_disp_pcd)
            

    batch = [dataset[i] for i in range(4)]
    collated = SparsePoints.collate_fn(batch)
    print(f"Collated batch:--------------------")
    print({k: v.shape if isinstance(v, torch.Tensor) else len(v) for k, v in collated.items()})