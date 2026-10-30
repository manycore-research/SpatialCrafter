import sys
sys.path.append(".")
sys.path.append("..")

from PIL import Image
import numpy as np
from tqdm import tqdm
import imageio
import re
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# ---------------------------------------------------------------------------
# 日志配置（每个子进程独立写，主进程汇总）
# ---------------------------------------------------------------------------
def setup_logger(log_file: str = "preprocess.log") -> logging.Logger:
    logger = logging.getLogger("preprocess")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# 纯函数工具
# ---------------------------------------------------------------------------
def natural_sort_key(s: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def center_crop(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """等比放大（若需要）后中心裁剪。"""
    img = img.convert("RGB")
    w, h = img.size
    if w < target_width or h < target_height:
        scale = max(target_width / w, target_height / h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.LANCZOS)
    left = (w - target_width) // 2
    top  = (h - target_height) // 2
    return img.crop((left, top, left + target_width, top + target_height))


def process_image(
    img_path: Path,
    output_path: Path,
    target_width: int,
    target_height: int,
) -> np.ndarray | None:
    """读取 → 裁剪 → 保存 → 返回 RGB 帧。出错返回 None。"""
    try:
        img = Image.open(img_path)
        cropped = center_crop(img, target_width, target_height)
        # cropped.save(output_path, quality=95)
        return np.array(cropped)
    except Exception as e:
        # 子进程内只能用 print/简单 logging，避免多进程 handler 竞争
        print(f"[WARN] process_image failed: {img_path}: {e}")
        return None


def save_video(frames: list[np.ndarray], video_path: Path, fps: int) -> None:
    """帧列表 → MP4（libx264, yuv420p）。"""
    if not frames:
        print(f"[WARN] No frames for {video_path}, skipping.")
        return
    imageio.mimwrite(
        str(video_path),
        frames,
        fps=fps,
        # codec="libx264",
        quality=10,
        # pixelformat="yuv420p",
    )


# ---------------------------------------------------------------------------
# 场景处理（在子进程中运行）
# ---------------------------------------------------------------------------
def process_scene_worker(args: tuple) -> tuple[str, int, str | None]:
    """
    子进程入口，返回 (scene_name, frame_count, error_message)。
    使用 tuple 传参以兼容 ProcessPoolExecutor.map。
    """
    scene_path, output_scene_path, target_width, target_height, fps = args

    scene_path       = Path(scene_path)
    output_scene_path = Path(output_scene_path)
    scene_name        = scene_path.name

    try:
        output_scene_path.mkdir(parents=True, exist_ok=True)
        images_dir = scene_path / "images_4"
        img_paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() == ".png"],
            key=lambda p: natural_sort_key(p.name),
        )
        video_path = output_scene_path / f"{scene_name}.mp4"
        
        if not img_paths:
            return scene_name, 0, "No images found"
        if video_path.exists():
            return scene_name, 360, None  # 已存在视频，跳过处理（假设每个视频固定360帧）
        
        frames = []
        for img_path in img_paths:
            frame = process_image(
                img_path,
                output_scene_path / img_path.name,
                target_width,
                target_height,
            )
            if frame is not None:
                frames.append(frame)

        save_video(frames, video_path, fps)
        return scene_name, len(frames), None

    except Exception as e:
        import traceback
        return scene_name, 0, traceback.format_exc()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def preprocess_dataset(
    data_root: str | Path,
    output_root: str | Path,
    target_width: int  = 512,
    target_height: int = 512,
    fps: int           = 16,
    num_workers: int   = 0,          # 0 = 自动（CPU 核心数 - 1）
    max_tasks_per_child: int = 4,    # 子进程处理 N 个任务后重启，防止内存泄漏
) -> None:
    logger = setup_logger()

    data_root   = Path(data_root)
    output_root = Path(output_root)

    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        sys.exit(1)

    scenes = sorted(
        [p for p in data_root.iterdir() if p.is_dir()],
        key=lambda p: natural_sort_key(p.name),
    )
    if not scenes:
        logger.warning("No scene directories found.")
        return

    # 自动决定进程数
    cpu_count = multiprocessing.cpu_count()
    workers   = num_workers if num_workers > 0 else max(1, cpu_count - 1)
    workers   = min(workers, len(scenes))    # 进程数不超过场景数
    logger.info(f"Found {len(scenes)} scenes | workers={workers} | cpu={cpu_count}")

    # 构建任务参数列表（纯 Python 原生类型，跨进程序列化友好）
    task_args = [
        (
            str(scene),
            # str(output_root / scene.name),
            str(output_root),
            target_width,
            target_height,
            fps,
        )
        for scene in scenes
    ]

    success_count = 0
    fail_count    = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        # max_tasks_per_child=max_tasks_per_child,
    ) as executor:
        future_to_scene = {
            executor.submit(process_scene_worker, args): args[0]
            for args in task_args
        }

        with tqdm(total=len(scenes), desc="Scenes", unit="scene") as pbar:
            for future in as_completed(future_to_scene):
                scene_dir = future_to_scene[future]
                try:
                    scene_name, frame_count, error = future.result()
                    if error:
                        logger.error(f"[FAIL] {scene_name}: {error}")
                        fail_count += 1
                    else:
                        logger.info(f"[OK]   {scene_name}: {frame_count} frames")
                        success_count += 1
                except Exception as exc:
                    logger.error(f"[FAIL] {scene_dir}: unexpected exception: {exc}")
                    fail_count += 1
                finally:
                    pbar.update(1)

    logger.info(
        f"Finished. success={success_count}, failed={fail_count}, total={len(scenes)}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess DL3DV dataset: center crop and save as MP4.")
    parser.add_argument("--dataset_path", 
                        type=str, 
                        default="/path/to/DL3DV-10K/1K", 
                        help="Path to the root directory of the raw DL3DV dataset.")
    parser.add_argument("--output_path", 
                        type=str, 
                        default="/path/to/DL3DV-10K/960p/10K_preprocessed", 
                        help="Path to the directory where processed videos will be saved.")
    parser.add_argument("--target_width", type=int, default=512, help="Target width for center cropping.")
    parser.add_argument("--target_height", type=int, default=512, help="Target height for center cropping.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output videos.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel worker processes to use. 0 means auto (CPU cores - 1).")
    args = parser.parse_args()
    
    # Windows 多进程必须在 __main__ 保护块内启动
    preprocess_dataset(
        data_root     = args.dataset_path,
        output_root   = args.output_path,
        target_width  = args.target_width,
        target_height = args.target_height,
        fps           = args.fps,
        num_workers   = args.num_workers,    # 0 = 自动
    )