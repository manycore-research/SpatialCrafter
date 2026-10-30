# This script is used to annotate RE10K dataset with [ViPE](https://arxiv.org/abs/2307.09288) preprocessing
# It processes all mp4 videos in the input directory and saves the results to the output directory.
import os
import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from utils.file_util import load_from_csv



def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess RE10K dataset with ViPE")
    parser.add_argument("--input_dir", 
                        type=str, 
                        default="/path/to/real-estate-10k/training_256", 
                        help="Directory containing input mp4 videos")
    parser.add_argument("--split_file", 
                        type=str, 
                        default="/path/to/real-estate-10k/train_splits.csv", 
                        help="CSV file containing dataset splits")
    parser.add_argument("--output_dir", 
                        type=str, 
                        default="/path/to/real-estate-10k/processed_re10k_train_data", 
                        help="Directory to save processed results")
    parser.add_argument("--num_workers", 
                        type=int, 
                        default=2, 
                        help="Number of parallel processes to use")
    parser.add_argument("--start_index", 
                        type=int, 
                        default=0, 
                        help="Start index of videos to process")
    parser.add_argument("--end_index", 
                        type=int, 
                        default=100, 
                        help="End index of videos to process")
    parser.add_argument("--task_id", 
                        type=int, 
                        default=0, 
                        help="Task ID for logging or tracking purposes")
    return parser.parse_args()

def process_video(video_file: str, output_dir: str) -> tuple[str, bool, str]:
    """
    Process a single video file using ViPE and save the results to the output directory.
    
    Args:
        video_file: Path to the input video file.
        output_dir: Directory to save the processed results.
    
    Returns:
        tuple: (视频文件名, 是否成功, 错误信息)
    """
    video_name = os.path.basename(video_file)
    
    result_depth_anno_file = os.path.join(output_dir, 'depth', f"{video_name.split('.')[0]}.zip")
    result_intrinsic_anno_file = os.path.join(output_dir, 'intrinsics', f"{video_name.split('.')[0]}.npz")
    result_pose_anno_file = os.path.join(output_dir, 'pose', f"{video_name.split('.')[0]}.npz")
    result_rgb_anno_file = os.path.join(output_dir, 'rgb', f"{video_name.split('.')[0]}.mp4")
    if os.path.exists(result_depth_anno_file) \
        and os.path.exists(result_intrinsic_anno_file) \
        and os.path.exists(result_pose_anno_file) \
        and os.path.exists(result_rgb_anno_file):
        print(f"Skipping {video_name} as it has already been processed.")
        return (video_name, True, None)
    # 构建命令
    cmd = [
        "vipe", "infer",
        video_file,
        "--pipeline", "dav3",
        "--output", output_dir,
    ]
    
    try:
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # # extract all zip files in OUTPUT_DIR/depth/ folders
        # depth_zip_files = list(Path(OUTPUT_DIR).glob("depth/*.zip"))
        # for zip_file in depth_zip_files:
        #     subprocess.run(
        #         ["unzip", "-o", str(zip_file), "-d", str(zip_file.parent)],
        #         check=True
        #     )
        #     os.remove(zip_file)
        return (video_name, True, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Return code: {e.returncode}, Error: {e.stderr}"
        return (video_name, False, error_msg)
    except Exception as e:
        return (video_name, False, str(e))


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    split_filepath = args.split_file
    output_dir = args.output_dir
    num_workers = args.num_workers
    start_index = args.start_index
    end_index = args.end_index
    task_id = args.task_id
    
    scene_df = load_from_csv(split_filepath, start_index=start_index, end_index=end_index)
    scene_ids = scene_df["video_files"].tolist()
    video_files = [os.path.join(input_dir, scene_id) for scene_id in scene_ids]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = len(video_files)
    
    print(f"Found {total_files} video files to process")
    print(f"Using {num_workers} processes for processing")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    if total_files == 0:
        print("No mp4 files found!")
        exit(0)
    
    # Create a pool of processes and process
    successful = 0
    failed = 0
    
    partial_process_video = partial(process_video, output_dir=output_dir)
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered to get results and show progress bar
        results = list(tqdm(
            pool.imap_unordered(partial_process_video, video_files),
            total=total_files,
            desc="Progress"
        ))
    
    # Statistics
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    
    for video_name, success, error in results:
        if success:
            successful += 1
        else:
            failed += 1
            print(f"✗ 失败: {video_name}")
            if error:
                print(f"  错误信息: {error}")
    
    print(f"\n成功: {successful}/{total_files}")
    print(f"失败: {failed}/{total_files}")
    
    # 如果有失败的，保存到日志文件
    if failed > 0:
        log_file = f"task{task_id}_failed_videos.log"
        with open(log_file, 'w') as f:
            for video_name, success, error in results:
                if not success:
                    f.write(f"{video_name}: {error}\n")
        print(f"\n失败的文件列表已保存到: {log_file}")
