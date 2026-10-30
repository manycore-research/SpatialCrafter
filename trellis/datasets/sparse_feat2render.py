import os

import numpy as np
import torch
import utils3d
from PIL import Image
from einops import rearrange

from trellis.datasets.components import StandardDatasetBase
from trellis.modules.sparse.basic import SparseTensor
from utils.geo_util import unproject_depth


class SparseFeat2Render(StandardDatasetBase):
    """
    SparseFeat2Render dataset.

    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        model (str): model name
        resolution (int): resolution of the data
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """

    def __init__(
        self,
        roots: str,
        image_size: int,
        model: str = "dinov2_vitl14_reg",
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        min_num_voxels: int = 1024,
        max_num_voxels: int = 32768,
        use_gt_occ: bool = False,
        num_views: int = 1,
        data_resolution: int = 128,
    ):
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        self.min_num_voxels = min_num_voxels
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        self.use_gt_occ = use_gt_occ
        self.num_views = num_views  # number of views to supervise per sample
        self.data_resolution = data_resolution
        super().__init__(roots)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f"feature_{self.model}"]]
        stats["With features"] = len(metadata)
        metadata = metadata[(metadata["num_voxels"] >= self.min_num_voxels) & (metadata["num_voxels"] <= self.max_num_voxels)]
        stats[f"Num voxels <= {self.max_num_voxels}"] = len(metadata)
        return metadata, stats

    def _upsample_coords(self, coords, factor):
        # upsample coords by factor
        n_cube = torch.ones([factor] * 3, device=coords.device, dtype=torch.int)
        n_coords = torch.nonzero(n_cube)
        upsampled_coords = coords.unsqueeze(1) * factor
        upsampled_coords = upsampled_coords + n_coords.unsqueeze(0).to(coords.dtype)
        upsampled_coords = upsampled_coords.flatten(0, 1)
        return upsampled_coords

    # 将坐标编码为唯一的整数
    @staticmethod
    def coords_to_hash(coords, base=10000):
        # 假设坐标范围不超过resolution
        return coords[:, 0] * base**2 + coords[:, 1] * base + coords[:, 2]

    def _calc_occupied_mask(self, upsampled_coords, fine_level_coords):
        # calculate the occupied mask for upsampled coords
        fine_hash = SparseFeat2Render.coords_to_hash(fine_level_coords)
        upsampled_hash = SparseFeat2Render.coords_to_hash(upsampled_coords)
        occ_mask = torch.isin(upsampled_hash, fine_hash)
        return occ_mask

    def load_batch_images(self, imgs: np.ndarray, img_type: str = "rgb"):
        """
        Load Nv images from np.array

        Returns:
            images: [Bv, C, H, W]
        """
        if img_type == "rgb":
            images = [Image.fromarray(img.astype(np.uint8)) for img in imgs]
            images = [np.array(img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)) for img in images]
            images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).float() / 255.0
        elif img_type == "mask":
            images = [Image.fromarray(img.astype(np.uint8)) for img in imgs]
            images = [np.array(img.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)) for img in images]
            images = torch.from_numpy(np.array(images)).float()
        elif img_type == "depth":
            images = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float()
        elif img_type == "normal":
            images = torch.from_numpy(-imgs).permute(0, 3, 1, 2).float() * 0.5 + 0.5
        else:
            raise NotImplementedError(f"image type {img_type} is not implemented yet.")
        return images

    def load_batch_poses(self, c2w_poses: np.ndarray):
        c2w = torch.from_numpy(c2w_poses).float()
        c2w[:, :3, 1:3] *= -1
        # w2c
        extrinsics = torch.inverse(c2w)
        return extrinsics

    def _get_data(self, root, instance):
        DATA_RESOLUTION = self.data_resolution
        feats_fileapth = os.path.join(root, "features", f"{instance}.npz")
        with np.load(feats_fileapth, allow_pickle=True) as data:
            masks = data["masks"]
            transforms = data["transforms"]
            angles = data["x_angles"]
            coords = data["indices"]
            feats = data["patchtokens"]
            # depth_maps = data["depths"]
            # normal_maps = data["normals"]
        # load rgb
        rgbs_filepath = os.path.join(root, "rgbs", f"{instance}.npz")
        with np.load(rgbs_filepath) as data:
            images = data["rgbs"]
        # load depth
        depths_filepath = os.path.join(root, "depths", f"{instance}.npz")
        with np.load(depths_filepath) as data:
            depth_maps = data["depths"]
        # load normal
        normals_filepath = os.path.join(root, "normals", f"{instance}.npz")
        with np.load(normals_filepath) as data:
            normal_maps = data["normals"]
        # sample view_idx
        valid_views = np.where(np.any(masks, axis=(1, 2)))[0]
        view_idxs = np.random.choice(valid_views, size=self.num_views, replace=False)
        view_idxs = view_idxs.tolist()
        # print(f"Sampled view idxs: {view_idxs}")

        # get intrinsic and extrinsic
        fov = angles[view_idxs[0]].astype(np.float32)
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

        extrinsics = self.load_batch_poses(transforms[view_idxs])

        # get images and alpha
        rgbs = self.load_batch_images(images[view_idxs], img_type="rgb")
        alphas = self.load_batch_images(masks[view_idxs], img_type="mask")
        depths = self.load_batch_images(depth_maps[view_idxs], img_type="depth")
        normals = self.load_batch_images(normal_maps[view_idxs], img_type="normal")
        # print(f"rgbs shape: {rgbs.shape}, alphas shape: {alphas.shape}, depths shape: {depths.shape}, normals shape: {normals.shape}")

        fxfycxcy = torch.tensor([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]).to(intrinsics)
        c2ws = torch.from_numpy(transforms[view_idxs]).float()
        c2ws[:, :3, 1:3] *= -1
        scene_coord_maps = unproject_depth(depths[:, 0, :, :], C2W=c2ws, fxfycxcy=fxfycxcy[None, :].repeat(depths.shape[0], 1)) * 0.5 + 0.5
        intrinsics = intrinsics[None, :, :].repeat(depths.shape[0], 1, 1)

        # get sparse voxel coords and sparse feats
        coords = torch.from_numpy(coords).int()
        feats = torch.from_numpy(feats).float()
        fine_level_coords = coords.clone()

        if self.resolution != DATA_RESOLUTION:
            assert self.resolution < DATA_RESOLUTION, "Only support downsampling"
            # downsample coords and feats
            factor = DATA_RESOLUTION // self.resolution
            coords = coords // factor
            coords, idx = coords.unique(return_inverse=True, dim=0)
            feats = torch.scatter_reduce(
                torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce="mean",
            )
            if self.use_gt_occ:
                # upsample coarse level coords to fine level coords for supervision
                upsampled_coords = self._upsample_coords(coords, factor)
                gt_occ_mask = self._calc_occupied_mask(upsampled_coords, fine_level_coords)
                # debug if we get the correct occ_mask
                # debug_upsampled_coords = upsampled_coords[gt_occ_mask]
                # debug_gt_coords = fine_level_coords
                # breakpoint()
                # utils3d.io.write_ply(os.path.join(f"{instance}_upsampled.ply"), debug_upsampled_coords.cpu().numpy())
                # utils3d.io.write_ply(os.path.join(f"{instance}_gt.ply"), debug_gt_coords.cpu().numpy())
                assert (
                    upsampled_coords[gt_occ_mask].shape[0] == fine_level_coords.shape[0]
                ), f"Upsampled occupied voxels {upsampled_coords[gt_occ_mask].shape[0]} does not match fine level voxels {fine_level_coords.shape[0]}"
                gt_occ_mask = gt_occ_mask.int().unsqueeze(-1)
        else:
            if self.use_gt_occ:
                gt_occ_mask = torch.ones((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
        # print(f"rgbs: {rgbs.shape}, alphas: {alphas.shape}, scene_coord_maps: {scene_coord_maps.shape}, normals: {normals.shape}, extrinsics: {extrinsics.shape}, intrinsics: {intrinsics.shape}")
        # print(f"scene_coord_maps min: {scene_coord_maps.min().item()}, max: {scene_coord_maps.max().item()}")
        img = {
            "image": rgbs,
            "alpha": alphas,
            "scene_coord": scene_coord_maps,
            "normal": normals,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }

        if self.use_gt_occ:
            feat = {
                "coords": coords,
                "feats": feats,
                "gt_occ_mask": gt_occ_mask,
            }
        else:
            feat = {
                "coords": coords,
                "feats": feats,
            }

        return img, feat

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        print(f"room_uids: {sample['room_uids']}")
        if len(sample["image"].shape) == 5:
            return rearrange(sample["image"], "B V C H W -> (B V) C H W")
        elif len(sample["image"].shape) == 4:
            return sample["image"]
        else:
            raise NotImplementedError(f"Not support sample[image] type {sample['image'].shape}")

    @staticmethod
    def collate_fn(batch):
        pack = {}
        coords = []
        for i, b in enumerate(batch):
            # construct sparseTensor data structure
            coords.append(
                torch.cat(
                    [
                        torch.full((b["coords"].shape[0], 1), i, dtype=torch.int32),
                        b["coords"],
                    ],
                    dim=-1,
                )
            )
        coords = torch.cat(coords)
        feats = torch.cat([b["feats"] for b in batch])
        pack["feats"] = SparseTensor(
            coords=coords,
            feats=feats,
        )

        pack["image"] = torch.stack([b["image"] for b in batch])
        pack["alpha"] = torch.stack([b["alpha"] for b in batch])
        pack["normal"] = torch.stack([b["normal"] for b in batch])
        pack["scene_coord"] = torch.stack([b["scene_coord"] for b in batch])
        pack["extrinsics"] = torch.stack([b["extrinsics"] for b in batch])
        pack["intrinsics"] = torch.stack([b["intrinsics"] for b in batch])
        pack["room_uids"] = [b["room_uid"] for b in batch]
        # print(f"rgbs: {pack['image'].shape}, alphas: {pack['alpha'].shape}, scene_coord_maps: {pack['scene_coord'].shape}, normals: {pack['normal'].shape}, extrinsics: {pack['extrinsics'].shape}, intrinsics: {pack['intrinsics'].shape}")

        if "gt_occ_mask" not in batch[0]:
            return pack

        # gt_occ_masks = []
        # for i, b in enumerate(batch):
        #     gt_occ_masks.append(
        #         torch.cat(
        #             [
        #                 torch.full((b["gt_occ_mask"].shape[0], 1), i, dtype=torch.int32),
        #                 b["gt_occ_mask"].unsqueeze(-1),
        #             ],
        #             dim=-1,
        #         )
        #     )
        # gt_occ_masks = torch.cat(gt_occ_masks)
        # pack["gt_occ_mask"] = SparseTensor(
        #     coords=gt_occ_masks,
        #     feats=gt_occ_masks[:, 1:].float(),
        # )
        gt_occ_masks = torch.cat([b["gt_occ_mask"] for b in batch], dim=0)
        # print(f"gt_occ_masks shape: {gt_occ_masks.shape}")
        pack["gt_occ_mask"] = gt_occ_masks
        return pack

    def get_instance(self, root, instance):
        image, feat = self._get_data(root, instance)
        return {
            "room_uid": instance,
            **image,
            **feat,
        }
