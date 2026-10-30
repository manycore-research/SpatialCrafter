import os
import shutil

import torch
import numpy as np
from PIL import Image
from loguru import logger

from accelerate import Accelerator
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core.data.operators import SCMImageCropAndResize

class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, 
                 validation_data_dir: str = None,
                 video_width: int = 512,
                 video_height: int = 512,
                 depth_concat_method: str = None,):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.validation_data_dir = validation_data_dir
        self.video_width = video_width
        self.video_height = video_height
        self.depth_concat_method = depth_concat_method

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            self.eval_model(accelerator, model)


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
    
    def eval_model(self, accelerator: Accelerator, model: torch.nn.Module):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            
            validation_pipeline = unwrap_model.pipe
            val_scene_ids = [ f for f in os.listdir(self.validation_data_dir) if os.path.isdir(os.path.join(self.validation_data_dir, f)) ]
            logger.info(f"Validating on {len(val_scene_ids)} scenes.")

            for scene_id in val_scene_ids:
                
                # load coarse videos and reference image
                coarse_rgb_video_path = f"{self.validation_data_dir}/{scene_id}/video_rgb_coarse.mp4"
                coarse_depth_video_path = f"{self.validation_data_dir}/{scene_id}/video_depth_coarse.mp4"
                control_video_rgb = VideoData(coarse_rgb_video_path, height=self.video_height, width=self.video_width)
                # control_video_depth = VideoData(coarse_depth_video_path, height=self.video_height, width=self.video_width)

                coarse_depth_video_path = f"{self.validation_data_dir}/{scene_id}/video_scm_coarse.npz"
                control_video_depth = np.load(coarse_depth_video_path, allow_pickle=True)["scm"]    # (num_frames, 3, h, w)
                scm_frame_oprator = SCMImageCropAndResize(height=self.video_height, width=self.video_width)
                control_video_depth = [ scm_frame_oprator(np.transpose(frame, (1, 2, 0))) for frame in control_video_depth ]
                
                num_frames = min(len(control_video_rgb), 49)
                print(f"Evaluating scene: {scene_id}")
                scene_output_dir = os.path.join(self.output_path, f"val-{self.num_steps}", scene_id)
                os.makedirs(scene_output_dir, exist_ok=True)
                    
                control_video_rgb = [control_video_rgb[i] for i in range(num_frames)]
                control_video_depth = [control_video_depth[i] for i in range(num_frames)]
                ref_rgb_fileapth = f"{self.validation_data_dir}/{scene_id}/reference_rgb.png"
                reference_image = Image.open(ref_rgb_fileapth).convert("RGB").resize((self.video_width, self.video_height))

                # gt_depth_video_path = f"{self.validation_data_dir}/{scene_id}/video_depth.mp4"
                # video_gt_depth = VideoData(gt_depth_video_path, height=self.video_height, width=self.video_width)
                # video_gt_depth = [video_gt_depth[i] for i in range(num_frames)]
                gt_depth_video_path = f"{self.validation_data_dir}/{scene_id}/video_scm.npz"
                video_gt_depth = np.load(gt_depth_video_path, allow_pickle=True)["scm"]
                video_gt_depth = [ scm_frame_oprator(np.transpose(frame, (1, 2, 0))) for frame in video_gt_depth ]
                video_gt_depth = [video_gt_depth[i] for i in range(num_frames)]

                # load reference depth image
                reference_depth_image = None
                if self.depth_concat_method is not None and self.depth_concat_method  == "width_wise":
                    reference_depth_image = video_gt_depth[0]

                # Control video
                tile_size = (self.video_height // 16, self.video_width // 16)
                tile_stride = (tile_size[0] // 2, tile_size[1] // 2)
                with torch.no_grad():
                    video = validation_pipeline(
                        prompt="high quality indoor scene",
                        negative_prompt="",
                        control_video=control_video_rgb,
                        control_video_depth=control_video_depth if self.depth_concat_method is not None else None,
                        depth_concat=self.depth_concat_method,
                        input_video_depth=video_gt_depth if self.depth_concat_method is not None else None,
                        num_frames=num_frames, 
                        reference_image=reference_image,
                        reference_depth_image=reference_depth_image if self.depth_concat_method is not None else None,
                        seed=1,
                        height=self.video_height,
                        width=self.video_width,
                        tiled=True,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    )

                    reference_image.save(f"{scene_output_dir}/reference_image.png")
                    shutil.copy(coarse_rgb_video_path, f"{scene_output_dir}/cond_video.mp4")
                    save_video_path = os.path.join(scene_output_dir, f"video_wan_functrl_lora.mp4")
                    save_video(video, save_video_path, fps=15, quality=10)
                        
            validation_pipeline.scheduler.set_timesteps(1000, training=True)
            logger.info("Validation done.")
        
        accelerator.wait_for_everyone()