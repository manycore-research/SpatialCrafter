import os
import sys
from weakref import ref
sys.path.append(".")
sys.path.append("..")
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = os.getenv("DIFFSYNTH_MODEL_BASE_PATH", "/path/to/")
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse

import torch
import numpy as np
from PIL import Image

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.core.data.operators import SCMImageCropAndResize
from utils.file_util import load_from_csv
from scripts.infer_coarse2fine_video import load_wanvideo_pipeline
from scripts.infer_trellis_img2scene import build_pipeline_from_ckpt, run_single_image_to_coarse_gs_with_dav3

def parse_args():
    parser = argparse.ArgumentParser(description="Validate single image to RGB-D video generation pipeline")
    # TRELLIS related arguments
    parser.add_argument("--ss_vae_config_file", 
                        type=str, 
                        default="/path/to/pretrained_ckpts/sft_ss_vae_64/config.json", 
                        help="Path to the configuration file")
    parser.add_argument("--ss_vae_checkpoint_step", 
                        type=int, 
                        default=4000, 
                        help="Which step of Checkpoint to load")
    parser.add_argument("--ss_dit_config_file", 
                        type=str, 
                        default="/path/to/pretrained_ckpts/trellis_ss_dit_64_spatialgen/config.json", 
                        help="Path to the configuration file")
    parser.add_argument("--ss_dit_checkpoint_step", 
                        type=int, 
                        default=10000, 
                        help="Which step of Checkpoint to load")
    parser.add_argument("--slat_vae_config_file", 
                        type=str, 
                        default="/path/to/pretrained_ckpts/sft_slat_vae_64/config.json", 
                        help="Path to the configuration file")
    parser.add_argument("--slat_vae_checkpoint_step", 
                        type=int, 
                        default=4000, 
                        help="Which step of Checkpoint to load")
    parser.add_argument("--slat_dit_config_file", 
                        type=str, 
                        default="/path/to/pretrained_ckpts/sft_slat_dit_64/config.json", 
                        help="Path to the configuration file")
    parser.add_argument("--slat_dit_checkpoint_step", 
                        type=int, 
                        default=4000, 
                        help="Which step of Checkpoint to load")
    # Wan-Video related arguments
    parser.add_argument("--wanvideo_ckpt_path", 
                        type=str, 
                        default="", 
                        help="Path to the model checkpoint")
    parser.add_argument("--ft_mode", 
                        type=str, 
                        default="lora", 
                        help="Fine-tuning mode: lora or full")
    parser.add_argument("--depth_concat", 
                        type=str, 
                        default=None, 
                        help="Method for depth concatenation: chann_wise or width_wise")
    parser.add_argument("--video_size", 
                        type=int, 
                        default=512, 
                        help="Height and width of the video frames")
    parser.add_argument("--wan_version", 
                        type=str, 
                        default="14B", 
                        help="Version of Wan model to use")
    # DepthAnything-V3 related arguments
    parser.add_argument("--dav3_ckpt_path", 
                        type=str, 
                        default="/path/to/DA3NESTED-GIANT-LARGE-1.1", 
                        help="Path to the DepthAnything-V3 model checkpoint")
    # Evaluation dataset related arguments
    parser.add_argument("--test_split_file", 
                        type=str, 
                        default="./test_data/tests.csv",
                        help="File containing the list of test scenes")
    parser.add_argument("--start_index", 
                        type=int, 
                        default=0, 
                        help="Start index of the test scenes")
    parser.add_argument("--end_index", 
                        type=int, 
                        default=-1, 
                        help="End index of the test scenes")
    parser.add_argument("--seed",
                        type=int,
                        default=43,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_path", 
                        type=str, 
                        default="./outputs", 
                        help="Path to save the output video")
    return parser.parse_args()

def run_coarse_to_fine_video_generation(
    wanvideo_pipe: WanVideoPipeline,
    ref_image_path: str,
    coarse_rgb_video_path: str,
    coarse_depth_video_path: str,
    video_height: int = 256,
    video_width: int = 256,
    depth_concat_method: str = "width_wise",
    ref_depth_maps: np.ndarray = None,
    scene_prompt: str = "high quality indoor scene",
    seed: int = 43,
    use_scm: bool = False,
):
    # load coarse videos
    reference_image = Image.open(ref_image_path).convert("RGB").resize((video_width, video_height))
    control_video_rgb = VideoData(coarse_rgb_video_path, height=video_height, width=video_width)
    if not use_scm:
        control_video_depth = VideoData(coarse_depth_video_path, height=video_height, width=video_width)
    else:
        control_video_depth = np.load(coarse_depth_video_path, allow_pickle=True)["scm"]    # (num_frames, 3, h, w)
        scm_frame_oprator = SCMImageCropAndResize(height=video_height, width=video_width)
        control_video_depth = [ scm_frame_oprator(np.transpose(frame, (1, 2, 0))) for frame in control_video_depth ]
    
    scene_id = os.path.basename(os.path.dirname(ref_image_path))
    if len(control_video_rgb) < 81:
        print(f"Scene {scene_id} has only {len(control_video_rgb)} frames, skipping.")
        return None, None, None  # skip short videos for now
    num_frames = min(len(control_video_rgb), 81)
    print(f"Evaluating scene: {scene_id}")
        
    control_video_rgb = [control_video_rgb[i] for i in range(num_frames)]
    control_video_depth = [control_video_depth[i] for i in range(num_frames)]

    # load reference depth image
    reference_depth_image = None
    if depth_concat_method is not None and depth_concat_method  == "width_wise":
        if not use_scm:
            # convert [0,1] depth map to pil image
            reference_depth_image = Image.fromarray((ref_depth_maps[0] * 255).astype(np.uint8))
        else:
            reference_depth_image = scm_frame_oprator(ref_depth_maps[0])
                
    # generate RGB-D video
    tile_size = (video_height // 16, video_width // 16)
    tile_stride = (tile_size[0] // 2, tile_size[1] // 2)
    gen_rgbd_video = wanvideo_pipe(
        prompt=scene_prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        control_video=control_video_rgb,
        control_video_depth=control_video_depth if depth_concat_method is not None else None,
        depth_concat=depth_concat_method,
        input_video_depth=None,
        num_frames=num_frames, 
        reference_image=reference_image,
        reference_depth_image=reference_depth_image if depth_concat_method is not None else None,
        seed=seed,
        height=video_height,
        width=video_width,
        tiled=True,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    if use_scm:
        # convert control_video_depth into numpy and (H W 3) format for saving as video
        control_video_depth_np = [ frame.permute(1, 2, 0).cpu().numpy() for frame in control_video_depth ]  # (num_frames, 3, H, W) -> (num_frames, H, W, 3)
        control_video_depth_np = [ ((frame + 1.0) * 127.5).astype(np.uint8) for frame in control_video_depth_np ]  # scale to [0, 255] for visualization
        control_video_depth = control_video_depth_np
    return gen_rgbd_video, control_video_rgb, control_video_depth

if __name__ == "__main__":
    args = parse_args()
    # TRELLIS pipeline params
    ss_vae_config_filepath = args.ss_vae_config_file
    ss_vae_ckpt_step = args.ss_vae_checkpoint_step
    ss_dit_config_filepath = args.ss_dit_config_file
    ss_dit_ckpt_step = args.ss_dit_checkpoint_step
    slat_vae_config_filepath = args.slat_vae_config_file
    slat_vae_ckpt_step = args.slat_vae_checkpoint_step
    slat_denoiser_config_filepath = args.slat_dit_config_file
    slat_denoiser_ckpt_step = args.slat_dit_checkpoint_step
    # Wan-Video pipeline params
    wanvideo_ckpt_path = args.wanvideo_ckpt_path
    ft_mode = args.ft_mode
    depth_concat_method = args.depth_concat
    video_width, video_height = args.video_size, args.video_size
    test_split_filepath = args.test_split_file
    wan_version = args.wan_version
    # DAv3 params
    dav3_ckpt_path = args.dav3_ckpt_path or os.getenv("DA3NESTED_MODEL_BASE_PATH", "/path/to/DA3NESTED-GIANT-LARGE-1.1") 
    output_dir = args.output_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(ss_vae_config_filepath):
        raise FileNotFoundError(f"SS-VAE config file path {ss_vae_config_filepath} does not exist.")
    if not os.path.exists(ss_dit_config_filepath):
        raise FileNotFoundError(f"SS-DIT config file path {ss_dit_config_filepath} does not exist.")
    if not os.path.exists(slat_vae_config_filepath):
        raise FileNotFoundError(f"SLAT-VAE config file path {slat_vae_config_filepath} does not exist.")
    if not os.path.exists(slat_denoiser_config_filepath):
        raise FileNotFoundError(f"SLAT-Denoiser config file path {slat_denoiser_config_filepath} does not exist.")
    if not os.path.exists(wanvideo_ckpt_path):
        raise FileNotFoundError(f"Wan-Video Checkpoint path {wanvideo_ckpt_path} does not exist.")
    
    test_scene_df = load_from_csv(test_split_filepath, start_index=args.start_index, end_index=args.end_index)
    print(f"Evaluating on {len(test_scene_df)} scenes.")
    
    # TRELLIS pipeline setup
    trellis_pipe = build_pipeline_from_ckpt(
        ss_vae_config_filepath=ss_vae_config_filepath,
        ss_dit_config_filepath=ss_dit_config_filepath,
        ss_vae_checkpoint_step=ss_vae_ckpt_step,
        ss_dit_checkpoint_step=ss_dit_ckpt_step,
        slat_vae_config_filepath=slat_vae_config_filepath,
        slat_denoiser_config_filepath=slat_denoiser_config_filepath,
        slat_vae_checkpoint_step=slat_vae_ckpt_step,
        slat_dit_checkpoint_step=slat_denoiser_ckpt_step,
        dav3_ckpt_filepth=dav3_ckpt_path,
    )
    # Wan-Video pipeline setup
    wanvideo_pipe = load_wanvideo_pipeline(wan_version=wan_version, finetune_mode=ft_mode, ckpt_path=wanvideo_ckpt_path)

    import time
    for idx, row in test_scene_df.iterrows():
        begin_tms = time.time()
        scene_id = os.path.basename(os.path.dirname(row["reference_image"]))
        scene_output_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_output_dir, exist_ok=True)
        save_video_path = os.path.join(scene_output_dir, f"spatialcrafter_rgb_scm.mp4")
        if os.path.exists(save_video_path):
            print(f"Output video for scene {scene_id} already exists, skipping.")
            continue
        
        # load ref_image and do single image to coarse gaussian video generation
        ref_rgb_fileapth = row["reference_image"]
        reference_image = Image.open(ref_rgb_fileapth).convert("RGB")
        # camera trajectory and intrinsics
        cameras_file = os.path.join(os.path.dirname(ref_rgb_fileapth), "cameras.npz")
        camera_params = np.load(cameras_file, allow_pickle=True)
        cv_c2w_poses = camera_params["c2w_poses"]
        intrinsics = camera_params["intrinsics"]
        depth_min = camera_params["depth_min"]
        depth_max = camera_params["depth_max"]
        scene_scale = camera_params["scene_scale"]
        
        if len(cv_c2w_poses) < 81:
            print(f"Scene {scene_id} has only {len(cv_c2w_poses)} camera poses, skipping.")
            continue
        
        # # debug camera trajectory
        # from utils.vis_util import vis_camera_keyframe
        # for idx in range(0, cv_c2w_poses.shape[0], 10):
        #     fx = intrinsics[idx, 0, 0] * video_width  # scale fx to image size
        #     fy = intrinsics[idx, 1, 1] * video_height  # scale fy to image size
        #     cam_keyframe = vis_camera_keyframe(fx=fx, fy=fy, width=video_width, height=video_height, c2w=cv_c2w_poses[idx])
        #     o3d.io.write_line_set(os.path.join(scene_output_dir, f"cam_keyframe_{idx:03d}.ply"), cam_keyframe)

        w2c_poses_tensor = torch.from_numpy(np.linalg.inv(cv_c2w_poses)).float()  # convert from c2w to w2c and to tensor
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        coarse_rgb_video_path, coarse_depth_video_path, ref_depth_maps = run_single_image_to_coarse_gs_with_dav3(
                                                            pipeline=trellis_pipe,
                                                            cond_img=reference_image,
                                                            cam_w2c_poses=w2c_poses_tensor,
                                                            intrinsics=intrinsics_tensor,
                                                            scene_scale=float(scene_scale), 
                                                            seed=args.seed,
                                                            save_coarse_gs=True,    # for debugging
                                                            return_scm=True,    # return scm for Wan-Video conditioning
                                                            output_dir=scene_output_dir,
                                                        )
        stage1_end_tms = time.time()
        stage1_tms = stage1_end_tms - begin_tms
        print(f"Stage 1 took {stage1_tms:.2f} seconds. Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
        

        # run coarse-to-fine video generation with Wan-Video pipeline
        gen_rgbd_video, control_video_rgb, control_video_depth = run_coarse_to_fine_video_generation(
            wanvideo_pipe=wanvideo_pipe,
            ref_image_path=ref_rgb_fileapth,
            coarse_rgb_video_path=coarse_rgb_video_path,
            coarse_depth_video_path=coarse_depth_video_path,
            video_height=video_height,
            video_width=video_width,
            depth_concat_method=depth_concat_method,
            ref_depth_maps=ref_depth_maps,
            scene_prompt=row["prompt"],
            seed=args.seed,
            use_scm=True,
        )
        if gen_rgbd_video is None or control_video_rgb is None or control_video_depth is None:
            continue
        stage2_end_tms = time.time()
        stage2_tms = stage2_end_tms - stage1_end_tms
        print(f"Stage 2 took {stage2_tms:.2f} seconds. Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
        
        reference_image.save(f"{scene_output_dir}/reference_image.png")
        merged_cond_video = []
        for rgb_frame, depth_frame in zip(control_video_rgb, control_video_depth):
            merged_frame = np.concatenate([np.array(rgb_frame), np.array(depth_frame)], axis=1)  # width-wise concatenation
            merged_cond_video.append(merged_frame)
        save_video(merged_cond_video, f"{scene_output_dir}/cond_video.mp4", fps=15, quality=10)
        
        save_video(gen_rgbd_video, save_video_path, fps=15, quality=10)
