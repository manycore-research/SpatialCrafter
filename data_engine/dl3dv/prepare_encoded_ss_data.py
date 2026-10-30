import os
import sys

sys.path.append(".")
sys.path.append("..")
import copy
import json
import argparse
import logging

import numpy as np
import pandas as pd
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from easydict import EasyDict as edict

import torch
import trellis.models as models
from trellis.models import SparseStructureEncoder
import trellis.modules.sparse as sp

logger = logging.getLogger(__name__)


def get_voxels_from_np(voxel_file: str, resolution: int, data_dir: str):
    with np.load(os.path.join(data_dir, 'voxels', f"{voxel_file}.npz"), allow_pickle=True) as data:
        coords = data["indices"]
    ss = torch.zeros(
        1, resolution, resolution, resolution, dtype=torch.long
    )
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    
    # load cond_voxels
    cond_voxel_coord_file = os.path.join(data_dir, 'cond_features', f"{voxel_file}.npz")
    cond_ss = []
    with np.load(cond_voxel_coord_file, allow_pickle=True) as data:
        cond_coords = data["indices"]  # (8, )
    for cond_coord in cond_coords:
        cond_ss_i = torch.zeros(
            1, resolution, resolution, resolution, dtype=torch.long
        )
        cond_ss_i[:, cond_coord[:, 0], cond_coord[:, 1], cond_coord[:, 2]] = 1
        cond_ss.append(cond_ss_i)
    cond_ss = torch.stack(cond_ss, dim=0)
    return ss, cond_ss

def parse_args():
    parser = argparse.ArgumentParser(description="Encode Sparse Strcuture for first stage genrator")
    parser.add_argument(
        "--dataset_output_dir",
        type=str,
        default="/path/to/process_dl3dv_trellis_data",
    )
    parser.add_argument(
        "--enc_model",
        type=str,
        default="ss_vae_conv3d_16l8_fp16",
    )
    parser.add_argument(
        "--encoder_ckpt_path",
        type=str,
        default="/path/to/pretrained_ckpts/sft_ss_vae_64/ckpts/encoder_step0004000.pt",
        help="Checkpoint to load",
    )
    parser.add_argument('--resolution', type=int, default=64, help='voxel resolution')
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--task_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger_trimesh = logging.getLogger("trimesh")
    logger_trimesh.setLevel(logging.ERROR)
    logger_pil = logging.getLogger("PIL")
    logger_pil.setLevel(logging.ERROR)

    args = parse_args()

    output_dir = args.dataset_output_dir
    enc_model = args.enc_model
    encoder_ckpt_filepath = args.encoder_ckpt_path
    start_index = args.start_index
    end_index = args.end_index
    num_workers = args.num_workers

    cfg_filepath = os.path.dirname(encoder_ckpt_filepath) + "/../config.json"
    config = edict(json.load(open(cfg_filepath, "r")))
    encoder: SparseStructureEncoder = getattr(models, config.models.encoder.name)(**config.models.encoder.args).cuda()
    encoder.load_state_dict(torch.load(encoder_ckpt_filepath), strict=False)
    encoder.eval()
    logger.info(f"Loaded {config.models.encoder.name} from {encoder_ckpt_filepath}")

    ckpt = os.path.basename(encoder_ckpt_filepath).replace("encoder_", "").replace(".pt", "")
    latent_name = f"{enc_model}_{ckpt}"
    
    output_latent_dir = os.path.join(output_dir, "ss_latents")
    os.makedirs(output_latent_dir, exist_ok=True)
    output_cond_latent_dir = os.path.join(output_dir, "cond_ss_latents")
    os.makedirs(output_cond_latent_dir, exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(output_dir, "metadata.csv")):
        metadata = pd.read_csv(os.path.join(output_dir, "metadata.csv"))
    else:
        raise ValueError("metadata.csv not found")

    metadata = metadata[metadata['num_voxels'] > 0]
    logger.info(f"Total {len(metadata)} instances with voxels")

    if f'ss_latent_{latent_name}' in metadata.columns:
        metadata = metadata[metadata[f'ss_latent_{latent_name}'] == False]
    print(f"key : ss_latent_{latent_name}")
    start_index = max(0, start_index)
    end_index = len(metadata) if end_index <= 0 else min(end_index, len(metadata))
    metadata = metadata[start_index:end_index]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata["sha256"].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(output_dir, "ss_latents", f"{sha256}.npz")):
            records.append({"sha256": sha256, f"ss_latent_{latent_name}": True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    # try:
    with ThreadPoolExecutor(max_workers=num_workers) as loader_executor, ThreadPoolExecutor(max_workers=num_workers) as saver_executor:

        def loader(sha256):
            try:
                ss, cond_ss = get_voxels_from_np(sha256, args.resolution, output_dir)
                load_queue.put((sha256, ss[None,].float(), cond_ss.float()))
            except Exception as e:
                logger.warning(f"Error loading features for {sha256}: {e}")

        loader_executor.map(loader, sha256s)

        def saver(sha256, pack):
            save_path = os.path.join(output_dir, "ss_latents", f"{sha256}.npz")
            # np.savez_compressed(save_path, **pack)
            np.savez_compressed(
                save_path,
                mean=pack['mean'],
            )
            cond_save_path = os.path.join(output_dir, "cond_ss_latents", f"{sha256}.npz")
            np.savez_compressed(
                cond_save_path,
                mean=pack['cond_ss_mean'],
            )
            records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})

        for _ in tqdm(range(len(sha256s)), desc="Extracting sparse structure latents"):
            sha256, ss, cond_ss = load_queue.get()
            ss = ss.cuda().float()
            # logger.info(f"sparse_structure shape: {ss.shape}")
            
            latent = encoder(ss, sample_posterior=False)
            assert torch.isfinite(latent).all(), "Non-finite latent"
            
            cond_ss = cond_ss.cuda().float()
            cond_latent = encoder(cond_ss, sample_posterior=False)
            logger.info(f"Encoded cond sparse voxel {cond_ss.shape} to latent code: {cond_latent.shape}")
            pack = {
                'mean': latent[0].detach().cpu().numpy(),
                'cond_ss_mean': cond_latent.detach().cpu().numpy(),
            }
            saver_executor.submit(saver, sha256, pack)

        saver_executor.shutdown(wait=True)

    records = pd.DataFrame.from_records(records)
    # records.to_csv(os.path.join(output_dir, f"latent_{latent_name}.csv"), index=False)

    # update the metadata.csv
    full_metadata = pd.read_csv(os.path.join(output_dir, "metadata.csv"))
    key_in_csv = f"ss_latent_{latent_name}"
    if key_in_csv not in full_metadata.columns:
        full_metadata[key_in_csv] = [False] * len(full_metadata)
    for sha256 in records["sha256"].values:
        full_metadata.loc[full_metadata["sha256"] == sha256, key_in_csv] = True
    full_metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
