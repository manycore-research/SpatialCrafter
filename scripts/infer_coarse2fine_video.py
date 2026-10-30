import os
import sys
sys.path.append(".")
sys.path.append("..")
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = os.getenv("DIFFSYNTH_MODEL_BASE_PATH", "/path/to/models")
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse

import torch
import numpy as np
from PIL import Image
import open3d as o3d

from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from utils.geo_util import rgbd_to_pointcloud
from utils.file_util import load_from_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Wan2.1-Fun-V1.1-1.3B-Control Model and reconstruc 3DGS from the output RGB-D images")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the model checkpoint")
    parser.add_argument("--ft_mode", type=str, default="lora", help="Fine-tuning mode: lora or full")
    parser.add_argument("--depth_concat", type=str, default=None, help="Method for depth concatenation: chann_wise or width_wise")
    parser.add_argument("--video_size", type=int, default=512, help="Height and width of the video frames")
    parser.add_argument("--test_data_dir", type=str, default="./data/real-estate-10k/test_256_150/process_re10k_wan_vace_data", help="Directory containing test data")
    parser.add_argument("--test_split_file", type=str, default="./data/real-estate-10k/test_256_150/process_re10k_wan_vace_data/metadata_wan_funcontrol_captioned.csv", help="File containing the list of test scenes")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of the test scenes")
    parser.add_argument("--end_index", type=int, default=-1, help="End index of the test scenes")
    parser.add_argument("--wan_version", type=str, default="14B", help="Version of Wan model to use")
    parser.add_argument("--seed", type=int, default=43, help="random seed for network")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Path to save the output video")
    return parser.parse_args()


def load_wanvideo_pipeline(wan_version: str = "1.3B", finetune_mode: str = "lora", ckpt_path: str = None):
    if wan_version == "1.3B":
        wan_model_configs = [
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ]
    elif wan_version == "14B":
        wan_model_configs = [
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ]
    else:
        raise ValueError(f"Wan version {wan_version} not supported.")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=wan_model_configs,
    )
    if finetune_mode == "full":
        state_dict = load_state_dict(ckpt_path)
        pipe.dit.load_state_dict(state_dict)
    else:
        pipe.load_lora(pipe.dit, ckpt_path, alpha=1)
        
    return pipe

if __name__ == "__main__":
    args = parse_args()
    
    ckpt_path = args.checkpoint_path
    ft_mode = args.ft_mode
    depth_concat_method = args.depth_concat
    video_width, video_height = args.video_size, args.video_size
    test_dataset_dir = args.test_data_dir
    test_split_filepath = args.test_split_file
    wan_version = args.wan_version
    output_dir = args.output_path
    
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")
    if not os.path.exists(test_dataset_dir):
        raise FileNotFoundError(f"Test dataset directory {test_dataset_dir} does not exist.")
    
    wanvideo_pipe = load_wanvideo_pipeline(wan_version=wan_version, finetune_mode=ft_mode, ckpt_path=ckpt_path)

    test_scene_df = load_from_csv(test_split_filepath, start_index=args.start_index, end_index=args.end_index)
    print(f"Evaluating on {len(test_scene_df)} scenes.")

    for idx, row in test_scene_df.iterrows():
        
        scene_id = os.path.basename(os.path.dirname(row["video"]))
        # if scene_id not in [
        # # SpatialVideo_test_v3 scenes
        # # "L3D943S403B0ENDOT25P5AUWLNPCLUFX7YADDA8",  
        # # "L3D943S403B0ENDOT26AGAUWLNPCLUFX7YDFLI8",
        # # "L3D943S403B0ENDOT26AJQUWKGIULUFX7YDFYA8",
        # # "L3D943S403B0ENDOT26AYAUWIHD2LUFX7YD2RQ8",
        # "L3D943S403B0ENDOT26BSQUWJSJ2LUFX7YD2LY8",
        # # "L3D943S403B0ENDOT26C7IUWIHD2LUFX7YD33Q8",
        # # "L3D943S403B0ENDOT26CHIUWJSJ2LUFX7YD3QQ8",
        # # "L3D943S403B0ENDOT26CTIUWINYILUFX7YD3ZI8",
        # # "L3D943S403B0ENDOT26ENQUWLNPCLUFX7YD3NY8"
        # ]:
        #     continue
        scene_output_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_output_dir, exist_ok=True)
        save_video_path = os.path.join(scene_output_dir, f"video_wan_functrl_{ft_mode}.mp4")
        if os.path.exists(save_video_path):
            print(f"Output video for scene {scene_id} already exists, skipping.")
            continue
        
        # load coarse videos and reference image
        coarse_rgb_video_path = row["control_video"]
        coarse_depth_video_path = row["control_video_depth"]
        control_video_rgb = VideoData(coarse_rgb_video_path, height=video_height, width=video_width)
        control_video_depth = VideoData(coarse_depth_video_path, height=video_height, width=video_width)
        
        if len(control_video_rgb) < 81:
            print(f"Scene {scene_id} has only {len(control_video_rgb)} frames, skipping.")
            continue  # skip short videos for now
        num_frames = min(len(control_video_rgb), 81)
        print(f"Evaluating scene: {scene_id}")

            
        control_video_rgb = [control_video_rgb[i] for i in range(num_frames)]
        control_video_depth = [control_video_depth[i] for i in range(num_frames)]
        ref_rgb_fileapth = row["reference_image"]
        reference_image = Image.open(ref_rgb_fileapth).convert("RGB").resize((video_width, video_height))
        
        # load reference depth image
        gt_depth_video_path = row["video_depth"]
        video_gt_depth = VideoData(gt_depth_video_path, height=video_height, width=video_width)
        video_gt_depth = [video_gt_depth[i] for i in range(num_frames)]
        
        # text prompt
        scene_prompt = row["prompt"]
        
        reference_depth_image = None
        if depth_concat_method is not None and depth_concat_method  == "width_wise":
            reference_depth_image = video_gt_depth[0]

        # Control video
        tile_size = (video_height // 16, video_width // 16)
        tile_stride = (tile_size[0] // 2, tile_size[1] // 2)
        video = wanvideo_pipe(
            prompt=scene_prompt,
            # negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            negative_prompt="",
            control_video=control_video_rgb,
            control_video_depth=control_video_depth if depth_concat_method is not None else None,
            depth_concat=depth_concat_method,
            input_video_depth=video_gt_depth if depth_concat_method is not None else None,
            num_frames=num_frames, 
            reference_image=reference_image,
            reference_depth_image=reference_depth_image if depth_concat_method is not None else None,
            seed=args.seed,
            height=video_height,
            width=video_width,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        

        reference_image.save(f"{scene_output_dir}/reference_image.png")

        merged_cond_video = []
        for rgb_frame, depth_frame in zip(control_video_rgb, control_video_depth):
            merged_frame = np.concatenate([np.array(rgb_frame), np.array(depth_frame)], axis=1)  # width-wise concatenation
            merged_cond_video.append(merged_frame)
        save_video(merged_cond_video, f"{scene_output_dir}/cond_video.mp4", fps=15, quality=10)
        
        cameras_file = os.path.join(f"{test_dataset_dir}/{scene_id}/cameras.npz")
        camera_params = np.load(cameras_file, allow_pickle=True)
        c2w_poses = camera_params["c2w_poses"]
        intrinsics = camera_params["intrinsics"]
        depth_min = camera_params["depth_min"]
        depth_max = camera_params["depth_max"]
        scene_scale = camera_params["scene_scale"]

        # extract depth map and rgb images
        if depth_concat_method is not None and depth_concat_method == "width_wise":
            data_dict = {"rgbs": [], "depths": [], "poses": [], "points": [], "colors": [], "intrinsic": None}
            rgb_frames = [ np.array(f)[:, :video_width, 0:3] for f in video]
            depth_frames = [ np.array(f)[:, video_width:, 0:1] for f in video]
            for idx, (rgb, depth) in enumerate(zip(rgb_frames, depth_frames)):
                depth = depth.astype(np.float32) / 255.0
                unscaled_depth = (depth * (depth_max - depth_min) + depth_min) / scene_scale
                # print(f"Depth range for frame {idx}: min {depth_min}, max {depth_max}")
                
                intrinsic_mat = intrinsics[idx]
                intrinsic_mat[:2] *= video_width  # scale intrinsics to image size

                # filter depth map at edge
                from utils.filter_util import edge_filter
                edge_mask = edge_filter(unscaled_depth, times=0.2)
                # Image.fromarray((edge_mask * 255).astype(np.uint8)).save(os.path.join(scene_output_dir,f"frame_{idx:03d}_edge.png"))
                if len(edge_mask.shape) == 2:
                    edge_mask = edge_mask[:, :, np.newaxis]
                unscaled_depth[edge_mask] = 0.0
                c2w_pose = c2w_poses[idx]
                cam_ply = rgbd_to_pointcloud(depth_image=unscaled_depth,
                                    rgb_image=rgb,
                                    c2w_pose=c2w_pose,
                                    depth_scale=1.0,
                                    intrinsic_mat=intrinsic_mat,
                )
                # o3d.io.write_point_cloud(os.path.join(scene_output_dir, f"frame_{idx:03d}.ply"), cam_ply)  
                # save iinto npz file
                data_dict["rgbs"].append(rgb)
                data_dict["depths"].append(unscaled_depth)
                data_dict["poses"].append(c2w_pose)
                data_dict["points"].append(np.array(cam_ply.points))
                data_dict["colors"].append(np.array(cam_ply.colors))
                
            data_dict["rgbs"] = np.stack(data_dict["rgbs"], axis=0)
            data_dict["depths"] = np.stack(data_dict["depths"], axis=0)
            data_dict["poses"] = np.stack(data_dict["poses"], axis=0)
            data_dict["points"] = np.stack(data_dict["points"], axis=0)
            data_dict["colors"] = np.stack(data_dict["colors"], axis=0)
            data_dict["intrinsic"] = intrinsic_mat
            np.savez(os.path.join(scene_output_dir, "inference_results.npz"), **data_dict)
        
        save_video(video, save_video_path, fps=15, quality=10)
        