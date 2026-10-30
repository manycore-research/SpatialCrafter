from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis.models.sparse_elastic_mixin import SparseTransformerElasticMixin
from trellis.models.structured_latent_vae.base import SparseTransformerBase
from trellis.modules import sparse as sp
from trellis.representations import Gaussian
from trellis.utils.random_utils import hammersley_sequence
from trellis.modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from trellis.modules.sparse.linear import SparseLinear
from trellis.modules.sparse.nonlinearity import SparseGELU, SparseSiLU
from trellis.modules.norm import LayerNorm32


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        in_channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """

    def __init__(
        self,
        in_channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or in_channels
        self.use_checkpoint = use_checkpoint

        self.act_layers = nn.Sequential(sp.SparseGroupNorm32(num_groups, in_channels), sp.SparseSiLU())

        self.sub = sp.SparseSubdivide()

        self.out_layers = nn.Sequential(
            sp.SparseConv3d(in_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(
                sp.SparseConv3d(
                    self.out_channels,
                    self.out_channels,
                    3,
                    indice_key=f"res_{self.out_resolution}",
                )
            ),
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(in_channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")

        self.occupancy_pred = nn.Sequential(
            LayerNorm32(self.out_channels, elementwise_affine=True, eps=1e-6),
            nn.Linear(self.out_channels, self.out_channels // 2),
            LayerNorm32(self.out_channels // 2, elementwise_affine=False, eps=1e-6),
            nn.SiLU(),
            nn.Linear(self.out_channels // 2, 1),
        )

        self.initialize_occpred_weights()

    def initialize_occpred_weights(self) -> None:
        for m in self.occupancy_pred.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # 使用小的gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.occupancy_pred[-1].weight, 0)
        nn.init.constant_(self.occupancy_pred[-1].bias, 0)

    def _forward(self, x: sp.SparseTensor, gt_occupancy: Optional[torch.Tensor] = None) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        # print(
        #     f"[DEBUG] h.feats - min: {h.feats.min().item():.4f}, "
        #     f"max: {h.feats.max().item():.4f}, "
        #     f"mean: {h.feats.mean().item():.4f}, "
        #     f"std: {h.feats.std().item():.4f}"
        #     f"dtype: {h.feats.dtype} "
        # )

        occupancy_logits = self.occupancy_pred(h.feats)  # (N, 1)
        # print(
        #     f"[DEBUG] occupancy_logits - min: {occupancy_logits.min().item():.4f}, "
        #     f"max: {occupancy_logits.max().item():.4f}, "
        #     f"mean: {occupancy_logits.mean().item():.4f}, "
        # )

        if self.training and gt_occupancy is not None:
            # Prune non-occupied voxels
            # Training: use ground truth occupancy to calculate loss
            occupancy_logits_clamped = torch.clamp(occupancy_logits, -10, 10)
            occupancy_loss = F.binary_cross_entropy_with_logits(occupancy_logits_clamped, gt_occupancy.float(), reduction="none")
        else:
            occupancy_loss = None
        # Inference: use predicted occupancy (self-pruning)
        mask = torch.sigmoid(occupancy_logits) > 0.5
        # print(f"[Decoder] Using predicted occupancy for pruning, occupied voxels: {mask.sum().item()} of upsampled voxels: {h.feats.shape[0]}")

        pruned_sp = sp.SparseTensor(
            feats=h.feats[mask.squeeze(-1)],
            coords=h.coords[mask.squeeze(-1)],
        )
        pruned_sp._scale = h._scale
        pruned_sp._spatial_cache = h._spatial_cache
        return pruned_sp, occupancy_loss

    def forward(self, x: sp.SparseTensor, gt_occupancy: Optional[torch.Tensor] = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, gt_occupancy, use_reentrant=False)
        else:
            return self._forward(x, gt_occupancy)


class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        enable_upsample: bool = False,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        if enable_upsample:
            self.upsample = nn.ModuleList(
                [
                    SparseSubdivideBlock3d(
                        in_channels=model_channels,
                        out_channels=model_channels,
                        resolution=resolution,
                        use_checkpoint=use_checkpoint,
                    ),
                ]
            )
            self.init_resolution = resolution  # resolution before upsample
            self.resolution = resolution * (2 ** len(self.upsample))  # upsample 1 times
            print(f"[SLatGaussianDecoder] Upsample {len(self.upsample)}x voxels from {self.init_resolution} to {self.resolution}")
        else:
            self.init_resolution = resolution
            self.resolution = resolution
        
        self.enable_upsample = enable_upsample
        self.rep_config = representation_config
        self._calc_layout()  # calculate out_channels and gs premitives shape
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)
        self._build_perturbation()

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        if self.enable_upsample:
            self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        if self.enable_upsample:
            self.upsample.apply(convert_module_to_f32)

    def _build_perturbation(self) -> None:
        perturbation = [hammersley_sequence(3, i, self.rep_config["num_gaussians"]) for i in range(self.rep_config["num_gaussians"])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config["voxel_size"]
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer("offset_perturbation", perturbation)
        print(
            f"[SLatGaussianDecoder] Built offset perturbation with shape: {perturbation.shape}, min: {perturbation.min().item()}, max: {perturbation.max().item()}"
        )

    def _calc_layout(self) -> None:
        """
        Calculate out_channels and gs premitives shape.
        """
        self.layout = {
            "_xyz": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_features_dc": {
                "shape": (self.rep_config["num_gaussians"], 1, 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_scaling": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_rotation": {
                "shape": (self.rep_config["num_gaussians"], 4),
                "size": self.rep_config["num_gaussians"] * 4,
            },
            "_opacity": {
                "shape": (self.rep_config["num_gaussians"], 1),
                "size": self.rep_config["num_gaussians"],
            },
        }
        start = 0
        for k, v in self.layout.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.out_channels = start

    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.
            voxel_occs: The voxel occupancy tensor.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
                scaling_bias=self.rep_config["scaling_bias"],
                opacity_bias=self.rep_config["opacity_bias"],
                scaling_activation=self.rep_config["scaling_activation"],
                scaling_max=self.rep_config["scaling_max"],
            )
            # convert xyz to [0, 1]
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            for k, v in self.layout.items():
                if k == "_xyz":
                    offset = x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]].reshape(-1, *v["shape"])
                    offset = offset * self.rep_config["lr"][k]
                    if self.rep_config["perturb_offset"]:
                        offset = offset + self.offset_perturbation
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config["voxel_size"]
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))

                else:
                    feats = x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]].reshape(-1, *v["shape"]).flatten(0, 1)
                    feats = feats * self.rep_config["lr"][k]
                    # if k == "_scaling":
                    #     feats = feats.clamp(max=self.rep_config["scaling_max"])
                    setattr(representation, k, feats)

            ret.append(representation)
        return ret

    def forward(self, x: sp.SparseTensor, x_mask: Optional[torch.tensor] = None, gt_occupancies: Optional[torch.Tensor] = None) -> List[Gaussian]:
        h = super().forward(x, x_mask)

        if self.enable_upsample:  # enable upsampling blocks
            if self.training and gt_occupancies is not None:
                occ_losses = []
                for idx, block in enumerate(self.upsample):
                    if idx < len(self.upsample) - 1:
                        h, _ = block(h)
                    else:
                        h, occ_loss = block(h, gt_occupancies)
                        occ_losses.append(occ_loss)
            else:
                for block in self.upsample:
                    h, _ = block(h)

            h = h.type(x.dtype)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = self.out_layer(h)

            ret_occ_loss = torch.cat(occ_losses, dim=0) if self.training and gt_occupancies is not None else None
            return self.to_representation(h), h, ret_occ_loss
        else:
            # wo upsampling blocks
            h = h.type(x.dtype)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = self.out_layer(h)
            return self.to_representation(h), h, None

class ElasticSLatGaussianDecoder(SparseTransformerElasticMixin, SLatGaussianDecoder):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM.
    """

    pass
