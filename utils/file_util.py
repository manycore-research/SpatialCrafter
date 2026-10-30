import os
import json
import csv
from pathlib import Path
from typing import Union, List, Any

import torch
import torchvision
import imageio
import pandas as pd
from einops import rearrange

def load_from_json(filepath: Union[str, Path]) -> Any:
    """Loads a JSON file and returns the content."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        return json.load(f)
def save_to_json(data: Any, filepath: Union[str, Path]) -> None:
    """Saves data to a JSON file."""
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_from_txt(filepath: Union[str, Path]) -> List[str]:
    """Reads in a text file and returns lines in a list."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    # Remove empty lines
    lines = [line for line in lines if line.strip()]
    return lines


def load_from_jsonl(filepath: Union[str, Path]) -> List[dict]:
    """Loads a JSON Lines file and returns a list of dictionaries."""
    filepath = Path(filepath)
    assert filepath.suffix == ".jsonl"
    if not filepath.exists():
        return None

    data = []
    with open(filepath, encoding="utf-8") as f:
        for row in f:
            data.append(json.loads(row))
    return data

def load_from_csv(split_file: str, start_index: int = 0, end_index: int = -1) -> pd.DataFrame:

    # load split
    df = pd.read_csv(split_file)
    num_scenes = len(df)
    if start_index >= num_scenes:
        print(f"Start index {start_index} is out of range for {num_scenes} scenes")
        return None

    end_index = end_index
    if end_index > num_scenes:
        end_index = num_scenes
    if end_index < 0 or end_index > num_scenes:
        end_index = num_scenes
    df = df.iloc[start_index:end_index]
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1

    return df

def save_to_csv(data: List[dict], output_file: Union[str, Path], fieldnames: List[str]) -> None:
    """Saves a list of dictionaries to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
    copy from: 
    https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)