import os
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm


def process_single_file(name):
    """处理单个文件的函数"""
    feat_filepath = os.path.join(result_path, "features", f"{name}.npz")
    voxel_filepath = os.path.join(result_path, "voxels", f"{name}.npz")
    rgb_filepath = os.path.join(result_path, "rgbs", f"{name}.npz")
    render_cond_filepath = os.path.join(result_path, "renders_cond", f"{name}.npz")

    try:

        assert os.path.exists(feat_filepath), f"Feature file not found: {feat_filepath}"
        assert os.path.exists(voxel_filepath), f"Voxel file not found: {voxel_filepath}"
        assert os.path.exists(rgb_filepath), f"RGB file not found: {rgb_filepath}"
        with np.load(voxel_filepath, allow_pickle=True) as data:
            indices = data["indices"]

        cond_rendered = False
        if os.path.exists(render_cond_filepath):
            # with np.load(render_cond_filepath, allow_pickle=True) as data:
            #     rgbs = data["rgbs"]
            # num_conds = len(rgbs)
            cond_rendered = True

        return {
            "sha256": name,
            "num_voxels": len(indices),
            "feature_dinov2_vitl14_reg": True if len(indices) > 1000 else False,  # 1000 is a threshold for whether the scene has enough voxels, which is determined by manual inspection of the data
            "cond_rendered": cond_rendered,
        }

    except Exception as e:
        return {"sha256": name, "num_voxels": 0, "feature_dinov2_vitl14_reg": False, "cond_rendered": False}


if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--result_path", type=str, default="/path/to/real-estate-10k/process_re10k_trellis_data")
    args = parse.parse_args()
    result_path = args.result_path
    
    # 获取所有文件
    result_files = [f[:-4] for f in os.listdir(os.path.join(result_path, "features")) if f.endswith(".npz")]

    # 使用多进程处理
    num_processes = min(len(result_files), cpu_count())
    print(f"Using {num_processes} processes to process {len(result_files)} files.")

    with Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 配合 tqdm 显示进度
        records = list(tqdm(pool.imap_unordered(process_single_file, result_files), total=len(result_files), desc="Processing files"))

    # 转换为 DataFrame
    records = pd.DataFrame(records)
    print(f"length of records: {len(records)}")
    records.to_csv(os.path.join(result_path, "metadata.csv"), index=False)
