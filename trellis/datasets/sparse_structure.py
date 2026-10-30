import os
from typing import Union

import numpy as np
import torch
import utils3d

from trellis.datasets.components import StandardDatasetBase
from trellis.renderers import OctreeRenderer
from trellis.representations.octree import DfsOctree as Octree


class SparseStructure(StandardDatasetBase):
    """
    Sparse structure dataset

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
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        min_num_voxels: int = 1024,
        max_num_voxels: int = 200000,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.min_num_voxels = min_num_voxels
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)

        super().__init__(roots)

    def filter_metadata(self, metadata):
        stats = {}
        stats["Voxelized"] = len(metadata)
        # metadata = metadata[(metadata["num_voxels"] >= self.min_num_voxels)]
        # stats[f"Num voxels >= {self.min_num_voxels}"] = len(metadata)
        metadata = metadata[(metadata["num_voxels"] >= self.min_num_voxels) & (metadata["num_voxels"] <= self.max_num_voxels)]
        stats[f"Num voxels <= {self.max_num_voxels}"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        with np.load(
            os.path.join(root, "voxels", f"{instance}.npz"), allow_pickle=True
        ) as data:
            coords = data["indices"]
        ss = torch.zeros(
            1, self.resolution, self.resolution, self.resolution, dtype=torch.long
        )
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return {"ss": ss}

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
