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
from trellis.models import ElasticSLatEncoder
import trellis.modules.sparse as sp

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Encode SLat for second stage genrator")
    parser.add_argument(
        "--dataset_rootpath",
        type=str,
        default="/path/to/spatialgen_data",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="/path/to/spatialgen_data/spatialgen_release_ids.csv",
    )
    parser.add_argument("--feat_model", type=str, default="dinov2_vitl14_reg", help="feature extraction model")
    parser.add_argument(
        "--enc_model",
        type=str,
        default="slat_vae_enc_dec_gs_swin8_B_64l8_fp16",
    )
    parser.add_argument(
        "--encoder_ckpt_path",
        type=str,
        default="/path/to/pretrained_ckpts/sft_slat_vae_64/ckpts/encoder_step0004000.pt",
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/process_dl3dv_trellis_data",
    )
    parser.add_argument("--text_cond", action="store_true", help="whether to prepare text conditional data")
    parser.add_argument("--img_cond", action="store_true", help="whether to prepare image conditional data")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
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
    dataset_rootpath = args.dataset_rootpath
    split_file = args.split_file
    output_dir = args.output_dir
    feat_model = args.feat_model
    enc_model = args.enc_model
    encoder_ckpt_filepath = args.encoder_ckpt_path
    text_cond = args.text_cond
    img_cond = args.img_cond
    start_index = args.start_index
    end_index = args.end_index
    num_workers = args.num_workers

    cfg_filepath = os.path.dirname(encoder_ckpt_filepath) + "/../config.json"
    config = edict(json.load(open(cfg_filepath, "r")))
    encoder: ElasticSLatEncoder = getattr(models, config.models.encoder.name)(**config.models.encoder.args).cuda()
    encoder.load_state_dict(torch.load(encoder_ckpt_filepath), strict=False)
    encoder.eval()
    print(f"Loaded {config.models.encoder.name} from {encoder_ckpt_filepath}")

    ckpt = os.path.basename(encoder_ckpt_filepath).replace("encoder_", "").replace(".pt", "")
    latent_name = f"{enc_model}_{ckpt}"
    output_latent_dir = os.path.join(output_dir, "latents")
    os.makedirs(output_latent_dir, exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(output_dir, "metadata.csv")):
        metadata = pd.read_csv(os.path.join(output_dir, "metadata.csv"))
    else:
        raise ValueError("metadata.csv not found")

    metadata = metadata[metadata[f"feature_{feat_model}"] == True]
    # if f"latent_{latent_name}" in metadata.columns:
    #     metadata = metadata[metadata[f"latent_{latent_name}"] == False]

    start_index = max(0, start_index)
    end_index = len(metadata) if end_index <= 0 else min(end_index, len(metadata))
    metadata = metadata[start_index:end_index]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata["sha256"].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(output_dir, "latents", f"{sha256}.npz")):
            records.append({"sha256": sha256, f"latent_{latent_name}": True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    # try:
    with ThreadPoolExecutor(max_workers=num_workers) as loader_executor, ThreadPoolExecutor(max_workers=num_workers) as saver_executor:

        def loader(sha256):
            try:
                feats = np.load(os.path.join(output_dir, "features", f"{sha256}.npz"))
                load_queue.put((sha256, feats))
            except Exception as e:
                print(f"Error loading features for {sha256}: {e}")

        loader_executor.map(loader, sha256s)

        def saver(sha256, pack):
            save_path = os.path.join(output_dir, "latents", f"{sha256}.npz")
            np.savez_compressed(save_path, **pack)
            records.append({"sha256": sha256, f"latent_{latent_name}": True})

        for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
            sha256, feats = load_queue.get()
            print(f"feats shape: {feats['patchtokens'].shape}, indices shape: {feats['indices'].shape}")
            feats = sp.SparseTensor(
                feats=torch.from_numpy(feats["patchtokens"]).float(),
                coords=torch.cat(
                    [
                        torch.zeros(feats["patchtokens"].shape[0], 1).int(),
                        torch.from_numpy(feats["indices"]).int(),
                    ],
                    dim=1,
                ),
            ).cuda()

            latent = encoder(feats, sample_posterior=False)
            assert torch.isfinite(latent.feats).all(), "Non-finite latent"
            pack = {
                "feats": latent.feats.detach().cpu().numpy().astype(np.float32),
                "coords": latent.coords[:, 1:].detach().cpu().numpy().astype(np.uint8),
            }
            print(f"latent feats shape: {pack['feats'].shape}, latent coords shape: {pack['coords'].shape}")
            saver_executor.submit(saver, sha256, pack)

        saver_executor.shutdown(wait=True)

    records = pd.DataFrame.from_records(records)
    print(f"--------------- Encoded latents for {len(records)} samples ---------------")
    
    # calculate statistics of the latents
    latent_means = []
    latent_mean2s = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor, \
        tqdm(total=len(records["sha256"].values), desc="Calculate features") as pbar:
        def worker(sha256):
            try:
                feats = np.load(os.path.join(output_dir, 'latents', f'{sha256}.npz'))
                feats = feats['feats']
                latent_means.append(feats.mean(axis=0))
                latent_mean2s.append((feats ** 2).mean(axis=0))
                pbar.update()
            except Exception as e:
                print(f"Error Calculating features for {sha256}: {e}")
                pbar.update()

        executor.map(worker, records["sha256"].values)
        executor.shutdown(wait=True)

    mean = np.array(latent_means).mean(axis=0)
    mean2 = np.array(latent_mean2s).mean(axis=0)
    std = np.sqrt(mean2 - mean ** 2)

    print('SLat mean:', mean)
    print('SLat std:', std)

    with open(os.path.join(output_dir, 'latents', 'stats.json'), 'w') as f:
        json.dump({
            'mean': mean.tolist(),
            'std': std.tolist(),
        }, f, indent=4)

    # update the metadata.csv
    full_metadata = pd.read_csv(os.path.join(output_dir, "metadata.csv"))
    key_in_csv = f"latent_{latent_name}"
    full_metadata[key_in_csv] = [False] * len(full_metadata)
    for sha256 in records["sha256"].values:
        full_metadata.loc[full_metadata["sha256"] == sha256, key_in_csv] = True
    full_metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
