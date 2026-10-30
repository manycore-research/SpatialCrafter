import json
import os
from typing import Optional, Union

import numpy as np
import torch
import utils3d

from trellis import models
from trellis.datasets.components import (
    MLLMDescriptionConditionedMixin,
    MLLMConditionedMixin,
    StandardDatasetBase,
    ImageConditionedMixin,
)
from trellis.renderers import OctreeRenderer
from trellis.representations.octree import DfsOctree as Octree


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt

        self.loads = [self.metadata.loc[sha256, "num_voxels"] for _, sha256 in self.instances]

    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, "config.json"), "r"))
            decoder = getattr(models, cfg["models"]["decoder"]["name"])(**cfg["models"]["decoder"]["args"])
            ckpt_path = os.path.join(self.ss_dec_path, "ckpts", f"decoder_{self.ss_dec_ckpt}.pt")
            decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i : i + batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        # if isinstance(x_0, dict):
        #     print(f"room_uids: {x_0['room_uids']}")
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0["x_0"]
        x_0 = self.decode_latent(x_0.cuda())

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
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device="cuda",
                primitive="voxel",
                sh_degree=0,
                primitive_config={"solid": True},
            )
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(resolution)),
                dtype=torch.uint8,
                device="cuda",
            )

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)

        return torch.stack(images)


class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """

    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        min_num_voxels: int = 1024,
        max_num_voxels: int = 32768,
        pretrained_ss_dec: str = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.min_num_voxels = min_num_voxels
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)

        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization["std"]).reshape(-1, 1, 1, 1)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f"ss_latent_{self.latent_model}"]]
        stats["With sparse structure latents"] = len(metadata)
        # metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
        # stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        metadata = metadata[(metadata["num_voxels"] >= self.min_num_voxels) & (metadata["num_voxels"] <= self.max_num_voxels)]
        stats[f"Num voxels between {self.min_num_voxels} and {self.max_num_voxels}"] = len(metadata)
        if "avg_lpips" in metadata.columns:
            metadata = metadata[metadata["avg_lpips"].notna()]
            # metadata = metadata[metadata["avg_lpips"] <= 0.76303]
            metadata = metadata[metadata["avg_lpips"] <= 0.7]
            stats[f"LPIPS <= 0.7"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root, "ss_latents", f"{instance}.npz"))
        z = torch.tensor(latent["mean"]).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            "room_uid": instance,
            "x_0": z,
        }
        return pack

class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """

    @staticmethod
    def collate_fn(batch):
        pack = {}

        # for k in batch[0].keys():
        #     if k == "x_0":
        #         pack["x_0"] = torch.stack([b["x_0"] for b in batch], dim=0)
        #     else:
        #         pack[k] = [b[k] for b in batch]

        pack["x_0"] = torch.stack([b["x_0"] for b in batch], dim=0)
        pack["room_uids"] = [b["room_uid"] for b in batch]
        pack["cond"] = torch.stack([b["cond"] for b in batch], dim=0)
        pack["cond_x_0"] = torch.stack([b["cond_x_0"] for b in batch], dim=0)
        # logger.info(f"Collated batch with x_0 shape: {pack['x_0'].shape}, cond shape: {pack['cond'].shape}, cond_x_0 shape: {pack['cond_x_0'].shape}")
        return pack


class MLLMConditionedSparseStructureLatent(MLLMConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """

    @staticmethod
    def collate_fn(batch):
        pack = {}

        for k in batch[0].keys():
            if k == "x_0":
                pack["x_0"] = torch.stack([b["x_0"] for b in batch], dim=0)
            else:
                pack[k] = [b[k] for b in batch]

        return pack


class MLLMDescriptionConditionedSparseStructureLatent(MLLMDescriptionConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """

    @staticmethod
    def collate_fn(batch):
        pack = {}

        for k in batch[0].keys():
            if k == "x_0":
                pack["x_0"] = torch.stack([b["x_0"] for b in batch], dim=0)
            else:
                pack[k] = [b[k] for b in batch]

        return pack
