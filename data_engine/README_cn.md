# Dataset Processing Pipeline Documentation

本文档详细说明三种数据集（SpatialGen、DL3DV-10K、RealEstate10K）的原始数据目录结构要求以及数据处理管线各阶段的输出目录结构。

## Pipeline Overview
数据处理管线分为以下五个阶段：
1. **ViPE 标注 / 数据转换**: 将原始数据转换为统一的RGB视频、Depth视频和相机参数格式，并生成 `dataset_stats.csv` 进行质量控制。
2. **TRELLIS 数据预处理**: 从ViPE输出的数据生成体素网格、特征提取，并过滤掉无效数据。
3. **TRELLIS-SLAT-GS-VAE 推理**: 使用微调后的SLAT-VAE模型生成3D高斯点云和RGB-D渲染结果。
4. **WanVideo 数据整合**: 将ViPE和TRELLIS的数据整合为WanVideo训练格式，并生成最终的 `metadata_wan_funcontrol.csv`。
5. **Caption生成**: 使用Gemini 3 Pro为每个视频生成描述性Caption，并更新元数据。
---

## Table of Contents
- [1. SpatialGen Dataset](#1-spatialgen-dataset)
- [2. DL3DV-10K Dataset](#2-dl3dv-10k-dataset)
- [3. RealEstate10K Dataset](#3-realestate10k-dataset)
- [Common Output Formats](#common-output-formats)
- [Processing Pipeline Overview](#processing-pipeline-overview)

---

## Prerequisites
- Environment: [spatialcrafter](../pyproject.toml)
- 预训练模型: [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth)
- 预训练模型: [TRELLIS-SLAT-VAE](https://huggingface.co/Chuan99/TRELLIS-Scene), put under `./pretrained_ckpts`

---

## Processing Pipeline Overview

### 通用5阶段处理流程

```
原始数据 (Raw Dataset)
    ↓
[Stage 1] ViPE 标注 (Real-world Data) / 全景图数据转换 (Synthetic Data)
    ↓
process_[dataset_name]_traindata/
    ├── RGB视频
    ├── Depth视频
    ├── 相机参数
    └── dataset_stats.csv (过滤<49帧)
    ↓
[Stage 2] TRELLIS 数据预处理
    ↓
process_[dataset_name]_trellis_data/
    ├── 体素网格
    ├── 特征提取
    └── dataset_stats.csv (过滤无效体素)
    ↓
[Stage 3] TRELLIS-SLAT-VAE 推理
    ↓
process_[dataset_name]_trellis_gs_inference/
    ├── 3D高斯点云
    ├── RGB-D渲染
    └── 渲染视频
    ↓
[Stage 4] WanVideo 数据整合
    ↓
process_[dataset_name]_wan_vace_data/
    ├── 整合ViPE和TRELLIS数据
    └── metadata_wan_funcontrol.csv
    ↓
[Stage 5] Caption生成
    ↓
process_[dataset_name]_wan_vace_data/ (更新)
    ├── 添加captions/
    └── metadata_wan_funcontrol_captioned.csv (含caption)
```

### 数据集特点对比

| 数据集 | 原始格式 | 分辨率 | 投影类型 | 特殊处理 |
|--------|---------|--------|---------|---------|
| DL3DV-10K | 图像序列 | 960p | 透视投影 | 标准3D场景 |
| RealEstate10K | 图像序列+TXT相机参数 | 256×256 | 透视投影 | 需解析相机参数文件 |
| SpatialGen | 立方体贴图 | 可变 | 全景投影 | 立方体→等距柱状投影转换 |

---

## 1. SpatialGen Dataset
### 1.1 原始数据结构要求

```
/path/to/your/raw_data/  # RAW_DATASET_DIR
├── scene_0000_0000/
│   ├── pano_rgb/                  # 全景RGB图像序列
│   │   ├── 0.png                  
│   │   ├── 1.png                  
│   │   ├── 2.png                 
│   │   ├── ...                  
│   ├── pano_depth/                  # 全景深度图像序列
│   ├── pano_normal/                  # 全景法线图像序列
│   ├── pano_semantic/                  # 全景语义图像序列
│   ├── cubemap_rgb/                  # 立方体贴图渲染
│   │   ├── 0_skybox0_sami.png                  # 6个面的RGB图像
│   │   ├── 0_skybox1_sami.png                  # 6个面的RGB图像
│   │   ├── 0_skybox2_sami.png                  # 6个面的RGB图像
│   │   ├── 0_skybox3_sami.png                  # 6个面的RGB图像
│   │   ├── 0_skybox4_sami.png                  # 6个面的RGB图像
│   │   ├── 0_skybox5_sami.png                  # 6个面的RGB图像
│   │   ├──  ...
│   ├── cubemap_depth/                  # 立方体贴图渲染的深度图
│   ├── cubemap_normal/                  # 立方体贴图渲染的法线图
│   ├── cubemap_semantic/                  # 立方体贴图渲染的语义图
│   ├── cubemap_depth/                  # 立方体贴图渲染的深度图
│   └── room_meta.json                 # 房间元数据: layout信息,相机轨迹参数, 房间类型
│
├── scene_0000_0001/
│   └── ...
```

**关键要求:**
- 立方体贴图格式（6个面：front, back, left, right, top, bottom）
- 每个面包含时序帧
- 需要包含RGB和深度（depth）数据
- 相机参数描述全景相机的运动轨迹

**特殊性:** 
- SpatialGen是多视点全景数据，需要从全景投影为透视图以满足TRELLIS训练数据需求（equirectangular projection）

### 3.2 数据处理管线输出结构

#### Stage 1: 全景图像转换为TRELLIS训练数据格式 (`processed_trellis_data/`)

```
processed_trellis_data/           # TRELLIS_DATASET_DIR
├── features/                     # 从全景数据生成的voxelized multi-view DINO features
│   ├── scene_0000_0000.npz
│   ├── ...           
├── depths/                       # 从全景数据生成的multi-view深度图
│   ├── scene_0000_0000.npz           
│   ├── ...
├── rgbs/                         # 从全景数据生成的multi-view RGB图
│   ├── scene_0000_0000.npz   
│   ├── ...
├── voxels/                       # 从全景数据生成的场景体素数据
│   ├── scene_0000_0000.npz 
│   ├── ...
├── normals/                      # 从全景数据生成的multi-view 法线图
│   ├── scene_0000_0000.npz
│   ├── ...
├── renders_cond/                 # 从全景数据生成的multi-view RGB图和深度图（TRELLIS-GS渲染条件）
│   ├── scene_0000_0000.npz 
│   ├── ...
├── world_cameras/                # Metric 世界坐标系下的相机参数
│   ├── scene_0000_0000.npz 
│   ├── ...
├── metadata.csv                  # TRELLIS 数据集元数据
```

**处理脚本:**
```bash
python3 data_engine/spatialgen/preprocess_trellis_data.py --dataset_rootpath {dataset_root_dir} \
                        --split_file {cfg.train_split_file} \
                        --dinov2_ckpt_path {cfg.dinov2_ckpt_file} \
                        --voxel_resolution {cfg.voxel_resolution} \
                        --start_index {start} \
                        --end_index {end+1} \
                        --num_workers {workers} \
                        --img_cond \
                        --output_dir {output_path}
```

---


## 2. DL3DV-10K Dataset

完整的数据处理脚本:

```bash
bash data_engine/dl3dv/run.sh
``` 

### 2.1 原始数据结构要求(如果你已经ViPE标注好了数据，可以跳过这部分，直接看后续的处理管线输出结构)

```
/path/to/DL3DV-10K/960p/
├── 10K/                          # 原始数据目录 (RAW_DATASET_DIR)
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61/               # 场景文件夹
│   │   ├── images_4/               # RGB图像序列
│   │   │   ├── frame_00001.png
│   │   │   ├── frame_00002.png
│   │   │   └── ...
│   │   └── transforms.json         # 场景元数据（可选）
│   ├── scene_0002/
│   └── ...
│
└── train_splits.csv              # 训练集划分文件 (RAW_DATASET_SPLIT_FILE)
    # CSV格式示例:
    # scene_id
    # scene_0001
    # scene_0002
    # scene_0003
```

### 2.2 数据处理管线输出结构

#### Stage 1: ViPE 标注结果 (`process_dl3dv_traindata/`)

```
process_dl3dv_traindata/          # ANNOTATE_DATASET_DIR
├── rgb/                          # RGB视频输出
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.mp4
│   ├── scene_0002.mp4
│   └── ...
│
├── depth/                        # 深度视频输出
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.zip
│   ├── scene_0002.zip
│   └── ...
│
├── intrinsics/                   # 内参输出
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.npz
│   └── ...
│
├── pose/                # 相机位姿输出
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.npz          # C2W 位姿序列
│   └── ...
│
└── dataset_stats.csv             # 数据集统计信息
    # 包含字段: scene_id, num_cameras
    # 由 postprocess_dl3dv_data.py 生成
```

**处理脚本:**
```bash
# 标注处理
python data_engine/dl3dv/preprocess_dl3dv_data.py \
    --input_dir "$RAW_DATASET_DIR" \
    --split_file "$RAW_DATASET_SPLIT_FILE" \
    --output_dir "$ANNOTATE_DATASET_DIR" \
    --num_workers 1 \
    --start_index 0 \
    --end_index 1000 \
    --task_id 0

# 后处理和统计
python data_engine/dl3dv/postprocess_dl3dv_data.py \
    --vipe_result_path "$ANNOTATE_DATASET_DIR" \
    --num_processes 16
```

#### Stage 3: TRELLIS 格式数据 (`process_dl3dv_trellis_data/`)

```
process_dl3dv_trellis_data/       # TRELLIS_DATASET_DIR
├── depths/                       # 深度图数据
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.npz
│   ├── ...
├── features/                    # voxelized DINO特征数据
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.npz
│   └── ...
├── rgbs/                        # RGB图像数据
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.npz
│   └── ...
├── voxels/                      # 场景体素数据
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.npz
│   └── ...
├── world_cameras/               # 世界坐标系下的相机参数
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61.npz
│   └── ...
│
└── metadata.csv             # TRELLIS数据集统计
    # 由 postprocess_trellis_data.py 生成
    # 过滤掉无效的体素数据
```

**处理脚本:**
```bash
# 转换为TRELLIS格式
python data_engine/dl3dv/preprocess_trellis_data.py \
    --dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --split_file "$ANNOTATE_DATASET_DIR/dataset_stats.csv" \
    --output_dir "$TRELLIS_DATASET_DIR" \
    --dinov2_ckpt_path "$DINOV2_CKPT_PATH" \
    --img_cond \
    --start_index 0 \
    --end_index 100 \
    --num_workers 8 \
    --task_id 0

# 后处理和验证
python data_engine/dl3dv/postprocess_trellis_data.py --result_path "$TRELLIS_DATASET_DIR"
```

#### Stage 4: TRELLIS 推理结果 (`process_dl3dv_trellis_gs_inference/`)

```
process_dl3dv_trellis_gs_inference/  # TRELLIS_INFERENCE_DATA_DIR
├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61/
│   ├── coarse_rgbs/             # 渲染的RGB图像序列
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   │
│   ├── coarse_depths/           # 渲染的深度图序列
│   │   ├── 0.png
│   │   └── ...
│   │
│   ├── sample_gs_rgb.mp4    # RGB渲染视频
│   ├── sample_gs_depth.mp4  # 深度渲染视频
│   └── cameras.npz      # 相机轨迹、内参、场景尺度
│
└── scene_0002/
    └── ...
```

**处理脚本:**
```bash
python data_engine/dl3dv/inference_trellis_gsvae.py \
    --dl3dv_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_dataset_rootpath "$TRELLIS_DATASET_DIR" \
    --gsvae_ckpt_config_path "$TRELLIS_SLAT_VAE_CKPT_CONFIG_PATH" \
    --target_width 512 --target_height 512  \
    --output_dir "$TRELLIS_INFERENCE_DATA_DIR"
```

#### Stage 5: WanVideo 训练数据 (`process_dl3dv_wan_vace_data/`)

```
process_dl3dv_wan_vace_data/      # WANVIDEO_DATASET_DIR
├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61/                   
│   ├── video_rgb.mp4      # GT RGB
│   ├── video_depth.mp4   # GT Depth
│   ├── video_rgb_coarse.mp4   # TRELLIS output Coarse RGB
│   ├── video_depth_coarse.mp4   # TRELLIS output Coarse Depth
│   ├── reference_rgb.png   # 参考RGB图像
│   └── cameras.npz      # 相机轨迹、内参、场景尺度
│
├── ...
│
└── metadata_wan_funcontrol.csv   # WanVideo训练元数据
    # 包含字段: video, video_depth, prompt, control_video, 
    #          control_video_depth, reference_image等;
```

**处理脚本:**
```bash
python data_engine/dl3dv/preprocess_wan_data.py \
    --dl3dv_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_results_rootpath "$TRELLIS_INFERENCE_DATA_DIR" \
    --output_dir "$WANVIDEO_DATASET_DIR" \
    --num_workers 16
```

#### Stage 6: WanFunControl 最终数据（含Caption）

```
process_dl3dv_wan_vace_data/      # 同上，但添加了caption
├── ...                           # 所有之前的数据
└── metadata_wan_funcontrol.csv   # 更新prompt字段
```

**处理脚本:**
```bash
python data_engine/re10k/postprocess_re10k_caption.py \
    --dataset_rootpath "$WANVIDEO_DATASET_DIR" \
    --split_file "$WANVIDEO_DATASET_DIR/metadata_wan_funcontrol.csv" \
    --output_dir "$WANVIDEO_DATASET_DIR" \
    --sample_fps 1.0
```

---

## 3. RealEstate10K Dataset

### 3.1 原始数据结构要求

```
/path/to/real-estate-10k/
├── training_256/                 # 原始数据目录 (RAW_DATASET_DIR)
│   ├── 0000cc6d8b108390.mp4     # 视频文件
│   └── ...
│
└── train_splits.csv              # 训练集划分文件
    # CSV格式:
    # video_id
    # 0000cc6d8b108390
    # 0a0933c250c35356
```

### 3.2 数据处理管线输出结构

#### Stage 1: ViPE 标注结果 (`process_re10k_traindata/`)

```
process_re10k_traindata/          # ANNOTATE_DATASET_DIR
├── rgb/                          # RGB视频输出
│   ├── 0000cc6d8b108390.mp4
│   ├── 0a0933c250c35356.mp4
│   └── ...
│
├── depth/                        # 深度视频输出
│   ├── 0000cc6d8b108390.zip
│   ├── 0a0933c250c35356.zip
│   └── ...
│
├── intrinsics/                   # 内参输出
│   ├── 0000cc6d8b108390.npz
│   └── ...
│
├── pose/                # 相机位姿输出
│   ├── 0000cc6d8b108390.npz          # C2W 位姿序列
│   └── ...
│
│
└── dataset_stats.csv             # 统计信息（过滤<49帧的视频）
    # 包含字段: scene_id, num_cameras
    # 由 postprocess__re10k_data.py 生成
```

**处理脚本:**
```bash
python data_engine/re10k/preprocess_re10k_data.py \
    --input_dir "$RAW_DATASET_DIR" \
    --split_file "$RAW_DATASET_SPLIT_FILE" \
    --output_dir "$ANNOTATE_DATASET_DIR"

python data_engine/re10k/postprocess_re10k_data.py \
    --vipe_result_path "$ANNOTATE_DATASET_DIR" \
    --num_processes 16
```

#### Stage 2: TRELLIS 格式数据 (`process_re10k_trellis_data/`)

```
process_re10k_trellis_data/       # TRELLIS_DATASET_DIR
├── depths/                       # 深度图数据
│   ├── 0000cc6d8b108390.npz
│   ├── ...
├── features/                    # voxelized DINO特征数据
│   ├── 0000cc6d8b108390.npz
│   └── ...
├── rgbs/                        # RGB图像数据
│   ├── 0000cc6d8b108390.npz
│   └── ...
├── voxels/                      # 场景体素数据
│   ├── 0000cc6d8b108390.npz
│   └── ...
├── world_cameras/               # 世界坐标系下的相机参数
│   ├── 0000cc6d8b108390.npz
│   └── ...
│
└── metadata.csv             # TRELLIS数据集统计
    # 由 postprocess_trellis_data.py 生成
    # 过滤掉无效的体素数据
```

**处理脚本:**
```bash
python data_engine/re10k/preprocess_trellis_data.py \
    --dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --split_file "$ANNOTATE_DATASET_DIR/dataset_stats.csv" \
    --output_dir "$TRELLIS_DATASET_DIR" \
    --img_cond \
     --dinov2_ckpt_path "$DINOV2_CKPT_PATH" \
    --start_index 0 \
    --end_index 100 \
    --num_workers 8 \
    --task_id 0

python data_engine/re10k/postprocess_trellis_data.py \
    --result_path "$TRELLIS_DATASET_DIR"
```

#### Stage 3: TRELLIS 推理结果 (`process_re10k_trellis_gs_inference/`)

```
process_re10k_trellis_gs_inference/  # TRELLIS_INFERENCE_DATA_DIR
├── 0000cc6d8b108390/
│   ├── coarse_rgbs/             # 渲染的RGB图像序列
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   │
│   ├── coarse_depths/           # 渲染的深度图序列
│   │   ├── 0.png
│   │   └── ...
│   │
│   ├── sample_gs_rgb.mp4    # RGB渲染视频
│   ├── sample_gs_depth.mp4  # 深度渲染视频
│   └── cameras.npz      # 相机轨迹、内参、场景尺度
│
└── ...
```

**处理脚本:**
```bash
python data_engine/re10k/inference_trellis_gsvae.py \
    --re10k_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_dataset_rootpath "$TRELLIS_DATASET_DIR" \
    --gsvae_ckpt_config_path "$TRELLIS_SLAT_VAE_CKPT_CONFIG_PATH" \
    --target_width 256 --target_height 256  \
    --output_dir "$TRELLIS_INFERENCE_DATA_DIR"
```

#### Stage 4 & 5: WanVideo 训练数据 (`process_re10k_wan_vace_data/`)

```
process_re10k_wan_vace_data/      # WANVIDEO_DATASET_DIR

```

---

## Common Output Formats

### dataset_stats.csv 通用格式

所有数据集的 `dataset_stats.csv` 包含以下核心字段：

```csv
scene_id,num_frames,width,height,fps,valid,notes
scene_0001,120,512,512,30,True,
scene_0002,45,512,512,30,False,less than 49 frames
scene_0003,180,512,512,30,True,
```

**字段说明:**
- `scene_id`: 场景/视频唯一标识符
- `num_frames`: 视频帧数
- `width`, `height`: 视频分辨率
- `fps`: 帧率
- `valid`: 是否通过验证（例如: ≥49帧）
- `notes`: 额外说明或错误信息

### metadata_wan_funcontrol.csv 格式

WanVideo训练使用的最终元数据：

```csv
scene_id,rgb_vipe_path,rgb_trellis_path,depth_vipe_path,depth_trellis_path,num_frames,caption,caption_timestamp
scene_0001,rgb_videos/scene_0001_vipe.mp4,rgb_videos/scene_0001_trellis.mp4,depth_videos/scene_0001_vipe.mp4,depth_videos/scene_0001_trellis.mp4,120,"A modern living room with...",2024-01-15T10:30:00
```

**字段说明:**
- `scene_id`: 场景ID
- `rgb_vipe_path`: ViPE生成的RGB视频路径
- `rgb_trellis_path`: TRELLIS渲染的RGB视频路径
- `depth_vipe_path`: ViPE生成的深度视频路径
- `depth_trellis_path`: TRELLIS渲染的深度视频路径
- `num_frames`: 帧数
- `caption`: Gemini 3 Pro生成的视频描述
- `caption_timestamp`: Caption生成时间戳


---

## 常见问题


---

## 总结

本文档详细说明了三种数据集的完整处理流程。关键要点：

1. **原始数据格式各异**: 需要根据数据集类型准备相应的目录结构
2. **5阶段标准流程**: ViPE标注 → TRELLIS转换 → TRELLIS推理 → WanVideo训练数据整合 → Caption
3. **多进程加速**: 利用 `--num_workers` 和 `--num_processes` 参数
4. **质量控制**: 每个阶段生成 `dataset_stats.csv` 用于验证和过滤
5. **批处理支持**: 使用索引范围支持大规模数据集的分布式处理

如有任何疑问，请参考各阶段的处理脚本和参数说明。