from torch import Tensor

import torch
from einops import rearrange, repeat
import numpy as np
from easydict import EasyDict as edict

from .gs_util import render

from trellis.representations.gaussian import Gaussian
from utils.typing import *
from utils.geo_util import unproject_depth


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options for RaDe-GS.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict(
            {
                "kernel_size": 0.1,
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
                "scale_modifier": 1.0,
                "debug": False,
            }
        )
        self.rendering_options = edict(
            {
                "resolution": None,
                "near": None,
                "far": None,
                "bg_color": "random",
                "render_type": "default",
                "deferred_bp_patch_size": 64,
                "depth_weight": 0.1,
                "normal_weight": 0.1,
                "scale_act_factor": 0.002,
                "scale_act_max": 0.3,
                "vis_depths": True,
                "vis_normals": True,
            }
        )
        self.rendering_options.update(rendering_options)
        self.bg_color = None

    # def render(
    #     self,
    #     gausssian: Gaussian,
    #     extrinsics: Float[Tensor, "4 4"],
    #     intrinsics: Float[Tensor, "3 3"],
    #     colors_overwrite: torch.Tensor = None,
    # ) -> edict:
    #     resolution = self.rendering_options["resolution"]
    #     near = self.rendering_options["near"]
    #     far = self.rendering_options["far"]

    #     if self.rendering_options["bg_color"] == "random":
    #         self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    #         if np.random.rand() < 0.5:
    #             self.bg_color += 1
    #     else:
    #         self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

    #     C2W = torch.inverse(extrinsics)
    #     fxfycxcy = torch.tensor([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]).to(intrinsics)
    #     render_dn = (
    #         self.rendering_options["depth_weight"] > 0.0
    #         or self.rendering_options["normal_weight"] > 0.0
    #         or self.rendering_options["vis_depths"]
    #         or self.rendering_options["vis_normals"]
    #     )  # whether render depth & normal
    #     pc = gausssian
    #     H, W = resolution, resolution
    #     render_results = render(
    #         pc=pc,
    #         height=H,
    #         width=W,
    #         C2W=C2W,
    #         fxfycxcy=fxfycxcy,
    #         znear=near,
    #         zfar=far,
    #         bg_color=self.bg_color,
    #         scaling_modifier=self.pipe["scale_modifier"],
    #         render_dn=render_dn,
    #     )
    #     rgb: Float[Tensor, "3 H W"] = render_results["image"]
    #     alpha: Float[Tensor, "1 H W"] = render_results["alpha"]
    #     depth: Float[Tensor, "1 H W"] = render_results["depth"]
    #     normal: Float[Tensor, "3 H W"] = render_results["normal"]

    #     bg_color = repeat(self.bg_color, "c -> c h w", h=H, w=W)

    #     coords: Float[Tensor, "3 H W"] = (unproject_depth(depth.squeeze(), C2W, fxfycxcy) * 0.5 + 0.5) * alpha + (1.0 - alpha) * bg_color
    #     normal_ = (normal * 0.5 + 0.5) * alpha + (1.0 - alpha) * bg_color
    #     ret = edict({"color": rgb, "alpha": alpha, "coord": coords, "depth": depth, "normal": normal_, "raw_depth": depth, "raw_normal": normal})

    #     return ret

    def render(
        self,
        gausssian: Gaussian,
        extrinsics: Float[Tensor, "V 4 4"],
        intrinsics: Float[Tensor, "V 3 3"],
        colors_overwrite: torch.Tensor = None,
    ) -> edict:
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]

        if self.rendering_options["bg_color"] == "random":
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        render_dn = (
            self.rendering_options["depth_weight"] > 0.0
            or self.rendering_options["normal_weight"] > 0.0
            or self.rendering_options["vis_depths"]
            or self.rendering_options["vis_normals"]
        )  # whether render depth & normal
        pc = gausssian
        H, W = resolution, resolution
        
        assert extrinsics.shape[0] == intrinsics.shape[0]
        V, device = extrinsics.shape[0], extrinsics.device
        images = torch.zeros(V, 3, H, W, dtype=torch.float32, device=device)
        alphas = torch.zeros(V, 1, H, W, dtype=torch.float32, device=device)
        depths = torch.zeros(V, 1, H, W, dtype=torch.float32, device=device)
        normals = torch.zeros(V, 3, H, W, dtype=torch.float32, device=device)
        fxfycxcys = torch.zeros(V, 4, dtype=torch.float32, device=device)
        for i in range(V): 
            C2W = torch.inverse(extrinsics[i])
            fxfycxcy = torch.tensor([intrinsics[i, 0, 0], intrinsics[i, 1, 1], intrinsics[i, 0, 2], intrinsics[i, 1, 2]]).to(intrinsics)
            render_results = render(
                pc=pc,
                height=H,
                width=W,
                C2W=C2W,
                fxfycxcy=fxfycxcy,
                znear=near,
                zfar=far,
                bg_color=self.bg_color,
                scaling_modifier=self.pipe["scale_modifier"],
                render_dn=render_dn,
            )
            images[i] = render_results["image"]
            alphas[i] = render_results["alpha"]
            depths[i] = render_results["depth"]
            normals[i] = render_results["normal"]
            fxfycxcys[i] = fxfycxcy

        bg_color = repeat(self.bg_color, "c -> v c h w", v=V, h=H, w=W)
        C2Ws = torch.inverse(extrinsics)
        coords: Float[Tensor, "V 3 H W"] = (unproject_depth(depths[:, 0,:,:], C2Ws, fxfycxcys) * 0.5 + 0.5) * alphas + (1.0 - alphas) * bg_color
        # print(f"rendered_coords shape: {coords.shape}, min: {coords.min().item()}, max: {coords.max().item()}")
        normals_ = (normals * 0.5 + 0.5) * alphas + (1.0 - alphas) * bg_color
        ret = edict({"color": images, "alpha": alphas, "coord": coords, "depth": depths, "normal": normals_, "raw_depth": depths, "raw_normal": normals})

        return ret