# This script evaluates the reconstruction quality of Re10k annotations by computing a masked LPIPS score 
# between GT reference videos and gs-coarse rendering videos. 
# It also generates visualizations of the spatial error distribution as heatmaps, highlighting areas of misalignment while ignoring invalid background regions. 
# The results are saved as images for qualitative analysis, and the average masked LPIPS scores are printed for quantitative comparison across different pairs.
import os
import sys

from librosa import ex
sys.path.append(".")
sys.path.append("..")
import random
import multiprocessing as mp
from functools import partial

import torch
import lpips
import imageio
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.file_util import load_from_csv, save_to_csv

# Global variables for worker processes
global_lpips_model = None
global_device = None

def init_worker():
    """
    初始化每个子进程的全局 LPIPS 模型（只在进程启动时加载一次）。
    """
    global global_lpips_model, global_device
    global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_lpips_model = lpips.LPIPS(net='alex', spatial=True).to(global_device)

def process_scene(scene_id, ann_dataset_dir, output_dir, num_samples, black_threshold):
    """
    独立进程调用的辅助函数。
    """
    global global_lpips_model, global_device
    
    clean_file = os.path.join(ann_dataset_dir, scene_id, "video_rgb.mp4")
    coarse_file = os.path.join(ann_dataset_dir, scene_id, "video_rgb_coarse.mp4")
    
    if os.path.exists(clean_file) and os.path.exists(coarse_file):
        lpips_val = evaluate_and_visualize_pair(
            clean_file, coarse_file, 
            output_dir=output_dir,
            scene_id=scene_id,
            num_samples=num_samples, 
            lpips_model=global_lpips_model,
            device=global_device,
            black_threshold=black_threshold
        )
        return scene_id, lpips_val
    else:
        return scene_id, None

def evaluate_and_visualize_pair(clean_path, coarse_path, output_dir, scene_id, num_samples=20, lpips_model=None, device='cpu', black_threshold=15):
    """
    计算 Masked LPIPS 并输出误差热力图可视化结果
    """
    if not os.path.exists(clean_path) or not os.path.exists(coarse_path):
        return None

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        reader_clean = imageio.get_reader(clean_path)
        reader_coarse = imageio.get_reader(coarse_path)

        total_frames = min(reader_clean.count_frames(), reader_coarse.count_frames())
        if total_frames == 0:
            return None
    except Exception as e:
        print(f"Error reading videos for scene {scene_id}: {e}")
        return None
        
    actual_samples = min(num_samples, total_frames)
    # sample_indices = sorted(random.sample(range(total_frames), actual_samples))
    # use uniform sampling to cover the whole video
    sample_indices = np.linspace(0, total_frames - 1, actual_samples, dtype=int).tolist()
    
    masked_lpips_scores = []
    
    for count, idx in enumerate(sample_indices):
        try:
            frame_clean = reader_clean.get_data(idx)
            frame_coarse = reader_coarse.get_data(idx)
        except IndexError:
            continue
            
        # 获取原始宽高
        H, W, _ = frame_clean.shape

        # 1. 生成有效区域 Mask
        intensity_coarse = np.mean(frame_coarse, axis=-1)
        mask_2d = (intensity_coarse > black_threshold).astype(np.float32)
        
        # 转换为 Tensor
        tensor_clean = torch.from_numpy(frame_clean).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
        tensor_coarse = torch.from_numpy(frame_coarse).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
        
        mask_3d = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float().to(device)
        tensor_clean_masked = tensor_clean * mask_3d + (-1.0) * (1 - mask_3d) 
        tensor_coarse_masked = tensor_coarse * mask_3d + (-1.0) * (1 - mask_3d)
        
        # 2. 计算空间特征误差图
        with torch.no_grad():
            dist_map = lpips_model(tensor_clean_masked, tensor_coarse_masked) # Shape: [1, 1, H', W']
        
        # 将 Mask 缩放到特征图大小以计算得分
        mask_resized_for_score = F.interpolate(mask_3d, size=(dist_map.shape[2], dist_map.shape[3]), mode='nearest')
        valid_pixels = mask_resized_for_score.sum()
        
        if valid_pixels > 0:
            masked_error = (dist_map * mask_resized_for_score).sum()
            current_masked_lpips = (masked_error / valid_pixels).item()
            masked_lpips_scores.append(current_masked_lpips)
        else:
            current_masked_lpips = 1000.0
        
        # ==========================================
        # 3. 可视化热力图生成逻辑
        # ==========================================
        # 将误差图上采样回原始图像分辨率 [1, 1, H, W]
        dist_map_upsampled = F.interpolate(dist_map, size=(H, W), mode='bilinear', align_corners=False)
        error_np = dist_map_upsampled.squeeze().cpu().numpy()
        
        # 仅保留 Mask 内的误差，去除背景干扰
        error_np_masked = error_np * mask_2d
        
        # 动态归一化：使用 99% 分位数作为最大值，防止极个别噪点破坏颜色对比度
        valid_errors = error_np_masked[mask_2d > 0]
        if len(valid_errors) > 0:
            vmax = np.percentile(valid_errors, 99) 
            vmax = max(vmax, 0.05) # 设置一个保底上限，防止完全一致的图因为极小除数导致全红
        else:
            vmax = 1.0

        error_normalized = np.clip(error_np_masked / vmax, 0, 1)
        
        # 使用 matplotlib 的 JET 颜色映射 (蓝=低误差, 红=高误差)
        heatmap_color = (plt.cm.jet(error_normalized)[:, :, :3] * 255).astype(np.uint8)
        
        # 将无效背景强制设为纯黑
        heatmap_color[mask_2d == 0] = [0, 0, 0]
        
        # 4. 拼接图像：[Clean 真实参照 | Coarse 渲染结果 | Heatmap 误差热力图]
        combined_img = np.concatenate((frame_clean, frame_coarse, heatmap_color), axis=1)
        
        # 写入硬盘
        output_scene_dir = os.path.join(output_dir, scene_id)
        os.makedirs(output_scene_dir, exist_ok=True)
        out_filename = os.path.join(output_scene_dir, f"frame_{idx:04d}_lpips_{current_masked_lpips:.3f}.jpg")
        imageio.imwrite(out_filename, combined_img)
            
    reader_clean.close()
    reader_coarse.close()
    
    return np.mean(masked_lpips_scores) if masked_lpips_scores else 0

if __name__ == "__main__":
    # Remove initializing global LPIPS model since we'll do it per-worker
    # loss_fn_alex = lpips.LPIPS(net='alex', spatial=True)
    
    ann_dataset_dir = "/path/to/RealEstate10K/DFoT_processed_data/datasets/real-estate-10k/process_re10k_wan_vace_data_0403/"
    ann_dataset_split_file = "/path/to/RealEstate10K/DFoT_processed_data/datasets/real-estate-10k/process_re10k_trellis_data_0403/metadata.csv"
    
    dataset_df = load_from_csv(ann_dataset_split_file)
    scene_ids = dataset_df["sha256"].tolist()
    
    # Initialize the new column 'avg_lpips' with pd.NA or zeros
    if 'avg_lpips' not in dataset_df.columns:
        dataset_df['avg_lpips'] = pd.NA
        
    results = {}
    NUM_SAMPLES = 16 
    OUTPUT_DIR = "/tmp/lpips_visualizations" # 可视化结果保存到/tmp/lpips_visualizations目录下
    NUM_WORKERS = max(1, mp.cpu_count() // 2)
    NUM_WORKERS = 32

    print(f"=== Masked LPIPS & Visualization using {NUM_WORKERS} workers ===")
    
    worker_func = partial(
        process_scene, 
        ann_dataset_dir=ann_dataset_dir, 
        output_dir=OUTPUT_DIR, 
        num_samples=NUM_SAMPLES, 
        black_threshold=15
    )

    # Use initializer to load the model once per worker process
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        for scene_id, lpips_val in tqdm(pool.imap_unordered(worker_func, scene_ids), total=len(scene_ids), desc="Processing Scenes"):
            if lpips_val is not None:
                results[scene_id] = lpips_val
                # Update the avg_lpips column in dataset_df
                dataset_df.loc[dataset_df["sha256"] == scene_id, "avg_lpips"] = lpips_val
            else:
                tqdm.write(f"Skipping Pair {scene_id} because files are incomplete or processing failed.")
            
    # Save the updated DataFrame back to the CSV file
    dataset_df.to_csv(ann_dataset_split_file, index=False)
    print(f"Updated dataset CSV saved to {ann_dataset_split_file}")
            
    if results:
        print("=== Final Alignment Evaluation ===")
        best_pair = min(results.items(), key=lambda x: x[1])[0]
        # Only print first few pairs to avoid cluttering the terminal
        for i, (scene_id, score) in enumerate(list(results.items())[:10]):
            print(f"Pair {scene_id}: Masked LPIPS = {score:.4f}")
        if len(results) > 10:
            print(f"... and {len(results) - 10} more.")
        print(f"\nConclusion: Pair {best_pair} has the best overall alignment.")