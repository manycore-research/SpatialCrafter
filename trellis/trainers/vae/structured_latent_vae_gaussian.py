import copy
from typing import Dict, Tuple, List

import numpy as np
import torch
import utils3d.torch
import torch.nn.functional as tF
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.utils import save_image

from trellis.modules.sparse import SparseTensor
from trellis.renderers import GaussianRenderer
from trellis.representations import Gaussian
from trellis.utils.loss_utils import l1_loss, l2_loss, lpips, ssim
from trellis.trainers.basic import BasicTrainer
from utils.typing import *
from utils.cam_util import is_point_inside_frustum
from trellis.modules.sparse import DEBUG


class SLatVaeGaussianTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
        lambda_kl (float): KL loss weight.
        regularizations (dict): Regularization config.
    """

    def __init__(
        self,
        *args,
        loss_type: str = "l1",
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_kl: float = 1e-6,
        regularizations: Dict = {},
        enbale_frustum_culling: bool = False,
        enable_self_pruning: bool = False,
        lambda_depth: float = 0.0,
        lambda_normal: float = 0.0,
        lambda_occ: float = 0.0,
        enbale_dn_loss: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_kl = lambda_kl
        self.regularizations = regularizations
        self.enbale_frustum_culling = enbale_frustum_culling
        self.enable_self_pruning = enable_self_pruning
        self.lambda_depth = lambda_depth
        self.lambda_normal = lambda_normal
        self.lambda_occ = lambda_occ
        self.enbale_dn_loss = enbale_dn_loss
        self.enable_output_upsampling = lambda_occ > 0  # enable upsampling when occupancy loss is used
        self._init_renderer()
        print(f"[SLatVaeGaussianTrainer] enbale_frustum_culling: {self.enbale_frustum_culling}, enable_self_pruning: {self.enable_self_pruning}")

    def _init_renderer(self):
        rendering_options = {"near": 0.01, "far": 10.0, "bg_color": "random", "depth_weight": self.lambda_depth, "normal_weight": self.lambda_normal}
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.pipe.kernel_size = self.models["decoder"].rep_config["2d_filter_kernel_size"]

    def _render_batch(self, reps: List[Gaussian], extrinsics: Float[Tensor, "B V 4 4"], intrinsics: Float[Tensor, "B V 3 3"]) -> torch.Tensor:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = None
        for i, representation in enumerate(reps):
            render_pack = self.renderer.render(gausssian=representation, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys()) + ["bg_color"]}
            for k, v in render_pack.items():
                ret[k].append(v)
            ret["bg_color"].append(self.renderer.bg_color)
        for k, v in ret.items():
            ret[k] = torch.stack(v, dim=0)
        return ret

    @torch.no_grad()
    def _get_status(self, decoded_z: SparseTensor, reps: List[Gaussian]) -> Dict:
        xyz = torch.cat([g.get_xyz for g in reps], dim=0)
        xyz_base = (decoded_z.coords[:, 1:].float() + 0.5) / self.models["decoder"].resolution - 0.5
        # print(f"[SLatVaeGaussianTrainer:_get_status] xyz: {xyz.shape}, xyz_base: {xyz_base.shape}")
        offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.models["decoder"].rep_config["num_gaussians"], -1).reshape(-1, 3)
        status = {
            "xyz": xyz,
            "offset": offset,
            "scale": torch.cat([g.get_scaling for g in reps], dim=0),
            "opacity": torch.cat([g.get_opacity for g in reps], dim=0),
        }

        for k in list(status.keys()):
            status[k] = {
                "mean": status[k].mean().item(),
                "max": status[k].max().item(),
                "min": status[k].min().item(),
            }

        return status

    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if "lambda_vol" in self.regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)  # [N x 3]
            volume = torch.prod(scales, dim=1)  # [N]
            terms[f"reg_vol"] = volume.mean()
            loss = loss + self.regularizations["lambda_vol"] * terms[f"reg_vol"]
        if "lambda_opacity" in self.regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f"reg_opacity"] = (opacity - 1).pow(2).mean()
            loss = loss + self.regularizations["lambda_opacity"] * terms[f"reg_opacity"]
        return loss, terms

    def _frustum_culling(self, sp: SparseTensor, extrinsics: Float[Tensor, "B 4 4"], intrinsics: Float[Tensor, "B 3 3"]) -> Float[Tensor, "Np"]:
        assert sp.shape[0] == extrinsics.shape[0] == intrinsics.shape[0]

        bsz = sp.shape[0]
        batch_inside_masks = []
        for i in range(bsz):
            xyz = sp.coords[sp.layout[i]][:, 1:].float() + 0.5
            normalized_xyz: Float[Tensor, "Np 3"] = xyz / self.models["decoder"].init_resolution - 0.5  # [-0.5, 0.5]
            # batch_points.append(normalized_xyz)
            # check if points are inside camera frustum
            # print(f"extrinsics.shape: {extrinsics.shape}, intrinsics.shape: {intrinsics.shape}")
            inside_mask: Float[Tensor, "B Nv Np"] = is_point_inside_frustum(
                input_points=normalized_xyz[None, :, :],
                # batch_w2c_poses=extrinsics[i : i + 1, None, :, :],
                # batch_intrinsics=intrinsics[i : i + 1, None, :, :],
                batch_w2c_poses=extrinsics[i : i + 1, :, :, :],
                batch_intrinsics=intrinsics[i : i + 1, :, :, :],
            )
            batch_inside_masks.append(inside_mask.squeeze())
        # batch_points: Float[Tensor, "B N 3"] = torch.stack(batch_points, dim=0)  # deprecated, different batch contains different number of points

        batch_inside_masks = torch.cat(batch_inside_masks, dim=-1)
        assert batch_inside_masks.shape[0] == sp.coords.shape[0]
        return batch_inside_masks

    @staticmethod
    def scm_gradient_loss(
        prediction: Float[Tensor, "B H W C"], target: Float[Tensor, "B H W C"], mask: Float[Tensor, "B H W"], conf=None, gamma=1.0, alpha=0.2
    ):
        # from torchvision.utils import save_image
        mask = mask[..., None]
        # mask = torch.ones(prediction.shape[:-1], dtype=torch.float32, device=prediction.device)[:, :, :, None]
        M = torch.sum(mask, (1, 2, 3))

        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)
        # print(f"grad_x shape: {grad_x.shape}, min: {grad_x.min()}, mean: {grad_x.mean()} , max: {grad_x.max()}")
        # save_image(grad_x.permute(0, 3, 1, 2), "grad_x.png")

        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)
        # print(f"grad_y shape: {grad_y.shape}, min: {grad_y.min()}, mean: {grad_y.mean()} , max: {grad_y.max()}")
        # save_image(grad_y.permute(0, 3, 1, 2), "grad_y.png")
        grad_x = grad_x.clamp(max=100)
        grad_y = grad_y.clamp(max=100)

        if conf is not None:
            conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
            conf_x = conf[:, :, 1:]
            conf_y = conf[:, 1:, :]
            gamma = 1.0
            alpha = 0.2

            grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
            grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

        image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

        divisor = torch.sum(M)

        if divisor == 0:
            return 0
        else:
            image_loss = torch.sum(image_loss) / divisor

        return image_loss

    @staticmethod
    def scm_gradient_loss_multi_scale(
        prediction: Float[Tensor, "B H W C"],
        target: Float[Tensor, "B H W C"],
        mask: Float[Tensor, "B H W"],
        scales=4,
        gradient_loss_fn=scm_gradient_loss,
        conf=None,
    ):
        """
        Compute gradient loss across multiple scales
        Args:
            prediction: B, H, W, C
            target: B, H, W, C
            scales: number of scales to compute gradient loss
            gradient_loss_fn: function to compute gradient loss
            conf: confidence map
        """

        total = 0
        for scale in range(scales):
            step = pow(2, scale)

            total += gradient_loss_fn(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                conf=conf[:, ::step, ::step] if conf is not None else None,
            )

        total = total / scales
        return total

    def training_losses(
        self,
        feats: SparseTensor,
        image: Float[Tensor, "B V C H W"],
        alpha: Float[Tensor, "B V H W"],
        extrinsics: Float[Tensor, "B V 4 4"],
        intrinsics: Float[Tensor, "B V 3 3"],
        return_aux: bool = False,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        z, mean, logvar = self.training_models["encoder"](feats, sample_posterior=True, return_raw=True)
        # frustum culling
        if self.enbale_frustum_culling:
            z_mask = self._frustum_culling(z, extrinsics, intrinsics)
        else:
            z_mask = None

        if DEBUG:
            import open3d as o3d

            original_voxels = z.coords[:, 1:].float().detach().cpu().numpy()
            ori_pointcloud = o3d.geometry.PointCloud()
            ori_pointcloud.points = o3d.utility.Vector3dVector(original_voxels.reshape(-1, 3).astype(np.float32))
            o3d.io.write_point_cloud("./original.ply", ori_pointcloud)

            z_mask_np = rearrange(z_mask, "B N -> (B N)").detach().cpu().numpy()
            culled_voxels = original_voxels[z_mask_np]
            culled_pointcloud = o3d.geometry.PointCloud()
            culled_pointcloud.points = o3d.utility.Vector3dVector(culled_voxels.reshape(-1, 3).astype(np.float32))
            o3d.io.write_point_cloud("./culling.ply", culled_pointcloud)

            from torchvision.utils import save_image

            save_image(image, "./image.png")

        if self.enable_output_upsampling:
            assert "gt_occ_mask" in kwargs, "gt_occ_mask not in min_batch_data"
            gt_occ_masks = kwargs["gt_occ_mask"]
            reps, decoded_z, z_occ_losses = self.training_models["decoder"](
                z,
                z_mask,
                gt_occupancies=gt_occ_masks,
            )
        else:
            reps, decoded_z, _ = self.training_models["decoder"](z, z_mask)

        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)

        terms = edict(loss=0.0, rec=0.0)

        rec_image: Float[Tensor, "B V 3 H W"] = render_results["color"]
        rec_normals: Float[Tensor, "B V 3 H W"] = render_results["normal"]
        rec_scenecoords: Float[Tensor, "B V 3 H W"] = render_results["coord"]
        gt_image = image * alpha[:, :, None] + (1 - alpha[:, :, None]) * render_results["bg_color"][:, None, :, None, None]
        gt_normals = kwargs["normal"] * alpha[:, :, None] + (1 - alpha[:, :, None]) * render_results["bg_color"][:, None, :, None, None]
        gt_scene_coords = kwargs["scene_coord"] * alpha[:, :, None] + (1 - alpha[:, :, None]) * render_results["bg_color"][:, None, :, None, None]

        # ################################ Compute reconstruction losses/metrics ################################
        # rgb loss
        rec_image = rearrange(rec_image, "b v c h w -> (b v) c h w")
        gt_image = rearrange(gt_image, "b v c h w -> (b v) c h w")
        if self.loss_type == "l1":
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == "l2":
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]

        if self.enbale_dn_loss:
            # Coord & Normal loss
            if self.lambda_depth > 0:
                gt_scene_coords = rearrange(gt_scene_coords, "b v c h w -> (b v) c h w")
                rec_scenecoords = rearrange(rec_scenecoords, "b v c h w -> (b v) c h w")
                # gt_masks = rearrange(alpha, "b v h w -> (b v) h w")
                # coord_grad_loss = self.scm_gradient_loss_multi_scale(
                #     prediction=rec_scenecoords.permute(0, 2, 3, 1),
                #     target=gt_scene_coords.permute(0, 2, 3, 1),
                #     mask=gt_masks,
                #     scales=4,
                # )
                coord_grad_loss = 0.0
                terms["coord_mse"] = coord_mse = tF.mse_loss(gt_scene_coords, rec_scenecoords) + coord_grad_loss
                terms["rec"] = terms["rec"] + self.lambda_depth * coord_mse
            if self.lambda_normal > 0:
                terms["normal_cosim"] = normal_cosim = tF.cosine_similarity(gt_normals, rec_normals, dim=2).mean()
                terms["rec"] = terms["rec"] + self.lambda_normal * (1.0 - normal_cosim)
            normal_cosim = 0.0
            print(f"coord_mse: {coord_mse}, coord_grad_loss: {coord_grad_loss}, normal_cosim: {normal_cosim}")
        terms["loss"] = terms["loss"] + terms["rec"]

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss

        # voxel occupancy loss in upsampling layers
        if self.lambda_occ > 0:
            occ_loss = z_occ_losses.mean()
            print(f"[SLatVaeGaussianTrainer:training_losses] occ_loss: {occ_loss.item():.6f}")
            terms["occ_loss"] = occ_loss * self.lambda_occ
            terms["loss"] = terms["loss"] + terms["occ_loss"]

        status = self._get_status(decoded_z, reps)

        if return_aux:
            return terms, status, {"rec_image": rec_image, "gt_image": gt_image}
        return terms, status

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=(self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None),
        )

        # inference
        ret_dict = {}
        gt_images = []
        gt_normals = []
        gt_coordmaps = []
        exts = []
        ints = []
        reps = []
        for i in range(0, num_samples, batch_size):
            print(f"[SLatVaeGaussianTrainer:run_snapshot] Inference {i}/{num_samples}")

            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() for k, v in data.items() if k != "room_uids"}
            print(f"snapshot rooms: {data['room_uids']}")
            rgbs = rearrange(args["image"], "B V C H W -> (B V) C H W")
            alphas = rearrange(args["alpha"], "B V H W -> (B V) H W")
            normals = rearrange(args["normal"], "B V C H W -> (B V) C H W")
            scenecoords = rearrange(args["scene_coord"], "B V C H W -> (B V) C H W")
            gt_rgb = rgbs * alphas[:, None]
            gt_normal = normals * alphas[:, None]
            gt_coords = scenecoords * alphas[:, None]
            extrinsics = args["extrinsics"]
            intrinsics = args["intrinsics"]
            gt_images.append(gt_rgb)
            gt_normals.append(gt_normal)
            gt_coordmaps.append(gt_coords)
            exts.append(extrinsics)
            ints.append(intrinsics)
            z = self.models["encoder"](args["feats"], sample_posterior=True, return_raw=False)
            reps.extend(self.models["decoder"](z)[0])
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f"gt_image": {"value": gt_images, "type": "image"}})
        gt_normals = torch.cat(gt_normals, dim=0)
        ret_dict.update({f"gt_normal": {"value": gt_normals, "type": "normal"}})
        gt_coordmaps = torch.cat(gt_coordmaps, dim=0)
        ret_dict.update({f"gt_coordmap": {"value": gt_coordmaps, "type": "coord"}})

        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps, exts, ints)
        rec_normal_img, rec_depth_img, rec_coord_img, rec_rgb_img = (
            render_results["normal"],
            render_results["depth"],
            render_results["coord"],
            render_results["color"],
        )
        rec_rgb_img = rearrange(rec_rgb_img, "B V C H W -> (B V) C H W")
        rec_normal_img = rearrange(rec_normal_img, "B V C H W -> (B V) C H W")
        rec_depth_img = rearrange(rec_depth_img, "B V C H W -> (B V) C H W")
        rec_coord_img = rearrange(rec_coord_img, "B V C H W -> (B V) C H W")
        ret_dict.update({f"rec_image": {"value": rec_rgb_img, "type": "image"}})
        ret_dict.update({f"rec_normal": {"value": rec_normal_img, "type": "normal"}})
        ret_dict.update({f"rec_depth": {"value": rec_depth_img, "type": "depth"}})
        ret_dict.update({f"rec_coord": {"value": rec_coord_img, "type": "coord"}})

        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        miltiview_images = []
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
            fov = torch.deg2rad(torch.tensor(90)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                eye=torch.tensor([0, 0, 0]).float().cuda(),
                look_at=orig,
                up=torch.tensor([0, 0, 1]).float().cuda(),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            # expand to batch size
            extrinsics = extrinsics.unsqueeze(0).expand(batch_size, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1, -1)
            # print(f"reps: {len(reps)}, extrinsics shape: {extrinsics.shape}, intrinsics shape: {intrinsics.shape}")
            render_results = self._render_batch(reps, extrinsics, intrinsics)
            miltiview_images.append(render_results["color"])

        ## Concatenate views
        miltiview_images = torch.cat(
            [
                torch.cat(miltiview_images[:2], dim=-2),
                torch.cat(miltiview_images[2:], dim=-2),
            ],
            dim=-1,
        )
        ret_dict.update({f"miltiview_image": {"value": miltiview_images, "type": "image"}})

        self.renderer.rendering_options.bg_color = "random"

        return ret_dict
