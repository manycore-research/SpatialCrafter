#!/bin/bash

RAW_DATASET_ROOT=/path/to/real-estate-10k
RAW_DATASET_DIR=$RAW_DATASET_ROOT/training_256
RAW_DATASET_SPLIT_FILE=$RAW_DATASET_ROOT/train_splits.csv
ANNOTATE_DATASET_DIR=$RAW_DATASET_ROOT/process_re10k_traindata
TRELLIS_DATASET_DIR=$RAW_DATASET_ROOT/process_re10k_trellis_data
TRELLIS_SLAT_VAE_CKPT_CONFIG_PATH=./pretrained_ckpts/sft_slat_vae_64/config.json
DINOV2_CKPT_PATH=/path/to/models/dinov2/dinov2_vitl14_reg4_pretrain.pth
TRELLIS_INFERENCE_DATA_DIR=$RAW_DATASET_ROOT/process_re10k_trellis_gs_inference
WANVIDEO_DATASET_DIR=$RAW_DATASET_ROOT/process_re10k_wan_vace_data

# 1. run ViPE to annotate the input RGB video and get the depth video and camera parameters
python data_engine/re10k/preprocess_re10k_data.py \
    --input_dir "$RAW_DATASET_DIR" \
    --split_file "$RAW_DATASET_SPLIT_FILE" \
    --output_dir "$ANNOTATE_DATASET_DIR" \
    --num_workers 1 \
    --start_index 0 \
    --end_index 1000 \
    --task_id 0
# postprocess the results of ViPE to get the dataset statistics and filter out videos with less than 49 frames
python data_engine/re10k/postprocess_re10k_data.py \
    --vipe_result_path "$ANNOTATE_DATASET_DIR" \
    --num_processes 16

# 2. convert the annotation results into the input format of TRELLIS-SLAT-VAE
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
# postprocess the results to get the dataset statistics and filter out invalid voxels
python data_engine/re10k/postprocess_trellis_data.py \
    --result_path "$TRELLIS_DATASET_DIR" 

# 3. inference TRELLIS-SLAT-VAE to get the coarse 3D gaussian and RGB-D renderings
python data_engine/re10k/inference_trellis_gsvae.py \
    --re10k_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_dataset_rootpath "$TRELLIS_DATASET_DIR" \
    --gsvae_ckpt_config_path "$TRELLIS_SLAT_VAE_CKPT_CONFIG_PATH" \
    --target_width 256 --target_height 256  \
    --output_dir "$TRELLIS_INFERENCE_DATA_DIR"

# 4. rearrange the fake RGB-D videos from ViPE and the coarse RGB-D renderings from TRELLIS-SLAT-VAE, 
# get the final dataset for training WanVideo
python data_engine/re10k/preprocess_wan_data.py \
    --re10k_dataset_rootpath "$ANNOTATE_DATASET_DIR" \
    --trellis_results_rootpath "$TRELLIS_INFERENCE_DATA_DIR" \
    --output_dir "$WANVIDEO_DATASET_DIR" \
    --num_workers 16

# 5. caption the RGB videos with Gemini 3 Pro and get the final dataset for training WanFunControl
python data_engine/re10k/postprocess_re10k_caption.py \
    --dataset_rootpath "$WANVIDEO_DATASET_DIR" \
    --split_file "$WANVIDEO_DATASET_DIR/metadata_wan_funcontrol.csv" \
    --output_dir "$WANVIDEO_DATASET_DIR" \
    --start_index 0 \
    --end_index 100 \
    --sample_fps 1.0 \
    --num_workers 8 \
    --task_id 0
