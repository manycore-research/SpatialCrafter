import os
import sys
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count


import imageio
import pandas as pd
from tqdm import tqdm


def extract_num_frames(video_path):
    """
    提取视频的帧数
    返回: (视频文件名, 帧数) 或 (视频文件名, None) 如果出错
    """
    try:
        video_reader = imageio.get_reader(video_path, 'ffmpeg')
        num_frames = video_reader.count_frames()
        video_reader.close()
        return (os.path.basename(video_path).split(".")[0], num_frames)
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return (os.path.basename(video_path).split(".")[0], None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Postprocess RE10K data")
    parser.add_argument("--vipe_result_path", 
                        type=str, 
                        default="/path/to/process_dl3dv_traindata", 
                        help="Path to the ViPE processed results")
    parser.add_argument("--num_processes", 
                        type=int, 
                        default=16, 
                        help="Number of processes for parallel processing")
    args = parser.parse_args()
    vipe_result_path = args.vipe_result_path
    num_processes = args.num_processes
    print(f"Using {num_processes} processes for parallel processing")
    
    # traverse mp4 files
    rgb_result_path = os.path.join(vipe_result_path, "rgb")
    rgb_video_files = [os.path.join(rgb_result_path, f) for f in os.listdir(rgb_result_path) if f.endswith(".mp4")]
    print(f"Found {len(rgb_video_files)} RGB video files in {rgb_result_path}")

    # 使用多进程池处理视频文件
    results = []
    with Pool(processes=num_processes) as pool:
        # 使用imap_unordered可以在结果准备好时立即返回，提高效率
        # tqdm显示进度条
        for result in tqdm(
            pool.imap_unordered(extract_num_frames, rgb_video_files),
            total=len(rgb_video_files),
            desc="Processing videos"
        ):
            results.append(result)
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(results, columns=['scene_id', 'num_cameras'])
    
    # 按视频文件名排序
    df = df.sort_values('scene_id').reset_index(drop=True)
    
    # 保存到CSV文件
    output_csv = os.path.join(vipe_result_path, 'dataset_stats.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # filter out videos with num_cameras less than 49
    filtered_df = df[df['num_cameras'] >= 49].reset_index(drop=True)
    print(f"Filtered dataset: {len(filtered_df)} videos with at least 49 frames")
    # save the filtered dataset stats
    filtered_csv_filepath = os.path.join(vipe_result_path, "dataset_stats_lg_49.csv")
    filtered_df.to_csv(filtered_csv_filepath, index=False)
    print(f"Filtered dataset stats saved to {filtered_csv_filepath}")
    
    # 打印统计信息
    print(f"\nStatistics:")
    print(f"Total videos processed: {len(df)}")
    print(f"Successful: {df['num_cameras'].notna().sum()}")
    print(f"Failed: {df['num_cameras'].isna().sum()}")
    if df['num_cameras'].notna().any():
        print(f"Average frames per video: {df['num_cameras'].mean():.2f}")
        print(f"Min frames: {df['num_cameras'].min()}")
        print(f"Max frames: {df['num_cameras'].max()}")