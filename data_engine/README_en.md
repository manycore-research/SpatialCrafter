# Dataset Processing Pipeline

Raw data layout and per-stage outputs for three datasets: SpatialGen, DL3DV-10K, and RealEstate10K.

## Pipeline Overview
1. **ViPE annotation / data conversion**: convert raw data into RGB videos, depth videos, and camera parameters; generate `dataset_stats.csv` for quality control.
2. **TRELLIS preprocessing**: voxelize scenes, extract features, and filter invalid samples.
3. **TRELLIS-SLAT-GS-VAE inference**: generate 3D Gaussians and RGB-D renderings with the fine-tuned SLAT-VAE.
4. **WanVideo data assembly**: merge ViPE and TRELLIS outputs into WanVideo training format; generate `metadata_wan_funcontrol.csv`.
5. **Captioning**: caption each video with Gemini 3 Pro and update the metadata.

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
- Pretrained model: [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth)
- Pretrained model: [TRELLIS-SLAT-VAE](https://huggingface.co/Chuan99/TRELLIS-Scene), put under `./pretrained_ckpts`

---

## Processing Pipeline Overview

```
Raw Dataset
    ↓
[Stage 1] ViPE annotation (real-world data) / panorama conversion (synthetic data)
    ↓
process_[dataset_name]_traindata/
    ├── RGB videos
    ├── Depth videos
    ├── Camera parameters
    └── dataset_stats.csv (drops videos with <49 frames)
    ↓
[Stage 2] TRELLIS preprocessing
    ↓
process_[dataset_name]_trellis_data/
    ├── Voxel grids
    ├── Features
    └── dataset_stats.csv (drops invalid voxels)
    ↓
[Stage 3] TRELLIS-SLAT-VAE inference
    ↓
process_[dataset_name]_trellis_gs_inference/
    ├── 3D Gaussians
    ├── RGB-D renderings
    └── Rendered videos
    ↓
[Stage 4] WanVideo data assembly
    ↓
process_[dataset_name]_wan_vace_data/
    ├── Merged ViPE + TRELLIS data
    └── metadata_wan_funcontrol.csv
    ↓
[Stage 5] Captioning
    ↓
process_[dataset_name]_wan_vace_data/ (updated)
    ├── captions/
    └── metadata_wan_funcontrol_captioned.csv
```

### Dataset Comparison

| Dataset | Raw Format | Resolution | Projection | Notes |
|---------|-----------|------------|------------|-------|
| DL3DV-10K | Image sequences | 960p | Perspective | Standard 3D scenes |
| RealEstate10K | Image sequences + TXT cameras | 256×256 | Perspective | Requires parsing camera files |
| SpatialGen | Cubemaps | Variable | Panoramic | Cubemap → equirectangular conversion |

---

## 1. SpatialGen Dataset
### 1.1 Raw Data Layout

```
/path/to/your/raw_data/  # RAW_DATASET_DIR
├── scene_0000_0000/
│   ├── pano_rgb/                  # panoramic RGB sequence
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── ...
│   ├── pano_depth/                # panoramic depth sequence
│   ├── pano_normal/               # panoramic normal sequence
│   ├── pano_semantic/             # panoramic semantic sequence
│   ├── cubemap_rgb/               # cubemap renderings (6 faces per frame)
│   │   ├── 0_skybox0_sami.png
│   │   ├── 0_skybox1_sami.png
│   │   ├── 0_skybox2_sami.png
│   │   ├── 0_skybox3_sami.png
│   │   ├── 0_skybox4_sami.png
│   │   ├── 0_skybox5_sami.png
│   │   ├── ...
│   ├── cubemap_depth/             # cubemap depth
│   ├── cubemap_normal/            # cubemap normal
│   ├── cubemap_semantic/          # cubemap semantic
│   └── room_meta.json             # layout, camera trajectory, room type
│
├── scene_0000_0001/
│   └── ...
```

**Requirements:**
- Cubemap format (6 faces: front, back, left, right, top, bottom), each with temporal frames
- RGB and depth are required
- Camera parameters describe the panoramic camera trajectory

**Note:** SpatialGen is multi-view panoramic data; panoramas are projected to perspective views to match TRELLIS training data.

### 1.2 Pipeline Outputs

#### Stage 1: Panorama → TRELLIS format (`processed_trellis_data/`)

```
processed_trellis_data/           # TRELLIS_DATASET_DIR
├── features/                     # voxelized multi-view DINO features
├── depths/                       # multi-view depth maps
├── rgbs/                         # multi-view RGB images
├── voxels/                       # scene voxels
├── normals/                      # multi-view normal maps
├── renders_cond/                 # multi-view RGB + depth (TRELLIS-GS render conditions)
├── world_cameras/                # cameras in metric world coordinates
└── metadata.csv                  # TRELLIS dataset metadata
```
Each subfolder contains one `{scene_id}.npz` per scene.

**Script:**
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

Full pipeline:

```bash
bash data_engine/dl3dv/run.sh
```

### 2.1 Raw Data Layout (skip if data is already ViPE-annotated)

```
/path/to/DL3DV-10K/960p/
├── 10K/                          # RAW_DATASET_DIR
│   ├── 9a0cf0ed740f0bb01f721ff302de3b0b0e42c12cdf7fe4e62399bd88dd22ec61/   # scene
│   │   ├── images_4/             # RGB sequence
│   │   │   ├── frame_00001.png
│   │   │   ├── frame_00002.png
│   │   │   └── ...
│   │   └── transforms.json       # scene metadata (optional)
│   ├── scene_0002/
│   └── ...
│
└── train_splits.csv              # RAW_DATASET_SPLIT_FILE
    # scene_id
    # scene_0001
    # scene_0002
```

### 2.2 Pipeline Outputs

#### Stage 1: ViPE annotation (`process_dl3dv_traindata/`)

```
process_dl3dv_traindata/          # ANNOTATE_DATASET_DIR
├── rgb/                          # {scene_id}.mp4
├── depth/                        # {scene_id}.zip
├── intrinsics/                   # {scene_id}.npz
├── pose/                         # {scene_id}.npz, C2W poses
└── dataset_stats.csv             # fields: scene_id, num_cameras; from postprocess_dl3dv_data.py
```

**Scripts:**
```bash
# Annotation
python data_engine/dl3dv/preprocess_dl3dv_data.py \
    --input_dir "$RAW_DATASET_DIR" \
    --split_file "$RAW_DATASET_SPLIT_FILE" \
    --output_dir "$ANNOTATE_DATASET_DIR" \
    --num_workers 1 \
    --start_index 0 \
    --end_index 1000 \
    --task_id 0

# Post-processing and statistics
python data_engine/dl3dv/postprocess_dl3dv_data.py \
    --vipe_result_path "$ANNOTATE_DATASET_DIR" \
    --num_processes 16
```

#### Stage 2: TRELLIS format (`process_dl3dv_trellis_data/`)

```
process_dl3dv_trellis_data/       # TRELLIS_DATASET_DIR
├── depths/                       # depth maps
├── features/                     # voxelized DINO features
├── rgbs/                         # RGB images
├── voxels/                       # scene voxels
├── world_cameras/                # cameras in world coordinates
└── metadata.csv                  # from postprocess_trellis_data.py; invalid voxels filtered
```
Each subfolder contains one `{scene_id}.npz` per scene.

**Scripts:**
```bash
# Convert to TRELLIS format
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

# Post-processing and validation
python data_engine/dl3dv/postprocess_trellis_data.py --result_path "$TRELLIS_DATASET_DIR"
```

#### Stage 3: TRELLIS inference (`process_dl3dv_trellis_gs_inference/`)

```
process_dl3dv_trellis_gs_inference/  # TRELLIS_INFERENCE_DATA_DIR
├── {scene_id}/
│   ├── coarse_rgbs/              # rendered RGB frames (0.png, 1.png, ...)
│   ├── coarse_depths/            # rendered depth frames
│   ├── sample_gs_rgb.mp4         # rendered RGB video
│   ├── sample_gs_depth.mp4       # rendered depth video
│   └── cameras.npz               # trajectory, intrinsics, scene scale
└── ...
```

**Script:**
```bash
python data_engine/dl3dv/inference_trellis_gsvae.py \
    --dl3dv_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_dataset_rootpath "$TRELLIS_DATASET_DIR" \
    --gsvae_ckpt_config_path "$TRELLIS_SLAT_VAE_CKPT_CONFIG_PATH" \
    --target_width 512 --target_height 512  \
    --output_dir "$TRELLIS_INFERENCE_DATA_DIR"
```

#### Stage 4: WanVideo training data (`process_dl3dv_wan_vace_data/`)

```
process_dl3dv_wan_vace_data/      # WANVIDEO_DATASET_DIR
├── {scene_id}/
│   ├── video_rgb.mp4             # GT RGB
│   ├── video_depth.mp4           # GT depth
│   ├── video_rgb_coarse.mp4      # TRELLIS coarse RGB
│   ├── video_depth_coarse.mp4    # TRELLIS coarse depth
│   ├── reference_rgb.png         # reference RGB image
│   └── cameras.npz               # trajectory, intrinsics, scene scale
├── ...
└── metadata_wan_funcontrol.csv   # fields: video, video_depth, prompt, control_video,
                                  #         control_video_depth, reference_image, ...
```

**Script:**
```bash
python data_engine/dl3dv/preprocess_wan_data.py \
    --dl3dv_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_results_rootpath "$TRELLIS_INFERENCE_DATA_DIR" \
    --output_dir "$WANVIDEO_DATASET_DIR" \
    --num_workers 16
```

#### Stage 5: Captioning

```
process_dl3dv_wan_vace_data/      # same as above
├── ...
└── metadata_wan_funcontrol.csv   # prompt field updated
```

**Script:**
```bash
python data_engine/re10k/postprocess_re10k_caption.py \
    --dataset_rootpath "$WANVIDEO_DATASET_DIR" \
    --split_file "$WANVIDEO_DATASET_DIR/metadata_wan_funcontrol.csv" \
    --output_dir "$WANVIDEO_DATASET_DIR" \
    --sample_fps 1.0
```

---

## 3. RealEstate10K Dataset

### 3.1 Raw Data Layout

```
/path/to/real-estate-10k/
├── training_256/                 # RAW_DATASET_DIR
│   ├── 0000cc6d8b108390.mp4
│   └── ...
│
└── train_splits.csv
    # video_id
    # 0000cc6d8b108390
    # 0a0933c250c35356
```

### 3.2 Pipeline Outputs

#### Stage 1: ViPE annotation (`process_re10k_traindata/`)

```
process_re10k_traindata/          # ANNOTATE_DATASET_DIR
├── rgb/                          # {video_id}.mp4
├── depth/                        # {video_id}.zip
├── intrinsics/                   # {video_id}.npz
├── pose/                         # {video_id}.npz, C2W poses
└── dataset_stats.csv             # fields: scene_id, num_cameras; drops videos with <49 frames
```

**Scripts:**
```bash
python data_engine/re10k/preprocess_re10k_data.py \
    --input_dir "$RAW_DATASET_DIR" \
    --split_file "$RAW_DATASET_SPLIT_FILE" \
    --output_dir "$ANNOTATE_DATASET_DIR"

python data_engine/re10k/postprocess_re10k_data.py \
    --vipe_result_path "$ANNOTATE_DATASET_DIR" \
    --num_processes 16
```

#### Stage 2: TRELLIS format (`process_re10k_trellis_data/`)

```
process_re10k_trellis_data/       # TRELLIS_DATASET_DIR
├── depths/                       # depth maps
├── features/                     # voxelized DINO features
├── rgbs/                         # RGB images
├── voxels/                       # scene voxels
├── world_cameras/                # cameras in world coordinates
└── metadata.csv                  # from postprocess_trellis_data.py; invalid voxels filtered
```
Each subfolder contains one `{video_id}.npz` per video.

**Scripts:**
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

#### Stage 3: TRELLIS inference (`process_re10k_trellis_gs_inference/`)

```
process_re10k_trellis_gs_inference/  # TRELLIS_INFERENCE_DATA_DIR
├── {video_id}/
│   ├── coarse_rgbs/              # rendered RGB frames
│   ├── coarse_depths/            # rendered depth frames
│   ├── sample_gs_rgb.mp4         # rendered RGB video
│   ├── sample_gs_depth.mp4       # rendered depth video
│   └── cameras.npz               # trajectory, intrinsics, scene scale
└── ...
```

**Script:**
```bash
python data_engine/re10k/inference_trellis_gsvae.py \
    --re10k_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_dataset_rootpath "$TRELLIS_DATASET_DIR" \
    --gsvae_ckpt_config_path "$TRELLIS_SLAT_VAE_CKPT_CONFIG_PATH" \
    --target_width 256 --target_height 256  \
    --output_dir "$TRELLIS_INFERENCE_DATA_DIR"
```

#### Stage 4 & 5: WanVideo training data (`process_re10k_wan_vace_data/`)

Same layout as DL3DV-10K (Stage 4 & 5).

---

## Common Output Formats

### dataset_stats.csv

```csv
scene_id,num_frames,width,height,fps,valid,notes
scene_0001,120,512,512,30,True,
scene_0002,45,512,512,30,False,less than 49 frames
scene_0003,180,512,512,30,True,
```

- `scene_id`: scene/video ID
- `num_frames`: number of frames
- `width`, `height`: resolution
- `fps`: frame rate
- `valid`: passed validation (e.g. ≥49 frames)
- `notes`: extra info or error message

### metadata_wan_funcontrol.csv

```csv
scene_id,rgb_vipe_path,rgb_trellis_path,depth_vipe_path,depth_trellis_path,num_frames,caption,caption_timestamp
scene_0001,rgb_videos/scene_0001_vipe.mp4,rgb_videos/scene_0001_trellis.mp4,depth_videos/scene_0001_vipe.mp4,depth_videos/scene_0001_trellis.mp4,120,"A modern living room with...",2024-01-15T10:30:00
```

- `scene_id`: scene ID
- `rgb_vipe_path` / `depth_vipe_path`: ViPE RGB / depth video
- `rgb_trellis_path` / `depth_trellis_path`: TRELLIS-rendered RGB / depth video
- `num_frames`: number of frames
- `caption`: Gemini 3 Pro caption
- `caption_timestamp`: caption generation time

---

## Summary

- Pipeline: ViPE annotation → TRELLIS preprocessing → TRELLIS inference → WanVideo data assembly → Captioning
- Parallelism: `--num_workers` / `--num_processes`
- Quality control: `dataset_stats.csv` / `metadata.csv` at each stage
- Sharding: `--start_index` / `--end_index` / `--task_id` for distributed processing
