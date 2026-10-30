import os
import sys
sys.path.append(".")
sys.path.append("..")
import asyncio
import base64
import tempfile
import shutil
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

import cv2
import imageio
import numpy as np
import PIL
from PIL import Image
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg, ImageBlock, Base64Source, TextBlock
from loguru import logger

from utils.file_util import load_from_csv

os.environ["OPENAI_API_KEY"] = "put-your-key-here"  # REPLACE YOUR API Key HERE !!!


@dataclass
class VideoAnnotationConfig:
    """视频标注配置"""
    fps: float = 1.0  # 采样帧率
    max_frames: Optional[int] = None  # 最大帧数限制
    image_quality: int = 95  # JPEG质量(1-100)
    max_image_size: tuple = (1920, 1080)  # 最大图像尺寸
    batch_size: int = 10  # 批处理大小，避免一次发送过多图片
    temperature: float = 1.0
    max_tokens: int = 8000


class SystemPrompts:
    """系统提示词管理"""
    
    ROOM_CAPTION = """You are an expert room description assistant. Analyze the provided video frames of a 3D room rendering and synthesize them into a natural, flowing description.
    CRITICAL OUTPUT REQUIREMENTS:
    - Write ONLY in narrative paragraph form - absolutely NO bullet points, numbered lists, asterisks (*), or section headers
    - Do NOT use formatting like "**bold**", "Room Function:", "Key Features:", etc.
    - Do NOT structure your response with labels or categories
    - Write as if you're naturally describing what you see to someone
    
    Your description should flow naturally and include: the room's apparent function, notable furniture and fixtures with their positions, flooring and wall treatments, windows and lighting, and overall spatial layout. Mention materials, colors, and style where relevant.

    STRICT LENGTH REQUIREMENT: Maximum 100 words. Write 3-5 sentences in a single paragraph. Be concise and selective about details. Write in clear, conversational English as if verbally describing the room to a colleague.
    
    Example of CORRECT format:
    "This modern living room features floor-to-ceiling windows along one wall offering views of surrounding greenery, with grey curtains framing the glass. The space has dark grey marble flooring with white veining, and one wall combines white marble paneling with vertical dark wood slats accented by a circular dual-ring sconce. A contemporary chandelier with gold accents hangs centrally above the glossy dark upper cabinetry and beige lower cabinets that line one side of the room."
    
    Example of INCORRECT format (never do this):
    "**Room Type:** Living room
    **Key Features:**
    - Floor-to-ceiling windows
    - Dark marble flooring"""

    ROOM_TYPE = """You are an expert room classification assistant. Analyze the video frames showing a 3D room from multiple angles and classify it into a specific room type.
    
    CRITICAL OUTPUT REQUIREMENTS:
    - Respond with ONLY the room type label - a single short phrase
    - NO explanations, NO additional text, NO formatting, NO markdown
    - NO phrases like "Based on the frames, this is..." or "Room type:" or "This appears to be..."
    - Just output the label directly
    
    Consider the primary function based on furniture, fixtures, and layout. For multi-functional spaces, use compound labels.
    
    Example CORRECT outputs:
    living room
    master bedroom
    kitchen with dining area
    home office
    bathroom
    
    Example INCORRECT outputs (never do this):
    Based on the video frames, this appears to be a living room
    **Room Type:** Bedroom
    This is a kitchen
    The room type is: bedroom"""

    GENERAL_DESCRIPTION = """You are an expert visual analysis assistant. Analyze the provided video frames showing an indoor scene from multiple perspectives.
    CRITICAL OUTPUT REQUIREMENTS:
    - Write in natural narrative paragraphs ONLY
    - Absolutely NO bullet points, numbered lists, asterisks (*), or section headers
    - Do NOT use formatting like "**bold**", "Objects:", "Layout:", "Style:", etc.
    - Do NOT structure with categories or labels
    - Write as if verbally describing the scene to someone
    
    Your description should flow naturally and comprehensively cover:
    - Objects present and their spatial relationships
    - Room layout and functional areas
    - Visual style: lighting, colors, materials, composition
    - Any notable patterns or transitions across frames
    
    Write 2-4 flowing paragraphs. Use clear, descriptive English.
    
    Example of CORRECT format:
    "The space is a contemporary open-plan living area dominated by floor-to-ceiling windows that flood the room with natural light and offer expansive views of the surrounding landscape. Dark grey marble flooring with dramatic white veining creates a luxurious foundation, while the walls feature an interesting interplay of textures combining smooth white marble panels with vertical dark wood slats.
    The lighting is layered and sophisticated, with a statement chandelier featuring multiple arms and gold accents serving as the central fixture, complemented by track lighting for ambient illumination. A unique circular dual-ring wall sconce mounted on the wood paneling adds an artistic touch. The built-in cabinetry along one wall presents a two-tone design with glossy dark upper cabinets and lighter beige lower units."
    
    Example of INCORRECT format (never do this):
    "**Objects and Layout:**
    - Floor-to-ceiling windows
    - Dark marble flooring
    **Lighting:**
    - Chandelier with gold accents
    - Track lighting"""

    @classmethod
    def get_prompt(cls, prompt_type: str) -> str:
        """获取指定类型的提示词"""
        prompts = {
            "roomcaption": cls.ROOM_CAPTION,
            "roomtype": cls.ROOM_TYPE,
            "general": cls.GENERAL_DESCRIPTION,
        }
        return prompts.get(prompt_type, cls.ROOM_CAPTION)


class VideoFrameExtractor:
    """视频帧提取器"""
    
    def __init__(self, config: VideoAnnotationConfig):
        self.config = config
    
    def extract_frames(self, video_path: str) -> List[str]:
        """
        从视频中提取帧并保存为临时图片
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[str]: 提取的图片路径列表
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="video_frames_")
        frame_paths = []
        
        try:
            reader = imageio.get_reader(str(video_path))
            meta_data = reader.get_meta_data()
            video_fps = meta_data.get('fps', 30.0)
            total_frames = int(reader.count_frames())
            
            if video_fps <= 0:
                logger.warning(f"Invalid FPS ({video_fps}), using default 30")
                video_fps = 30.0
            
            # 计算采样间隔
            frame_interval = int(video_fps / self.config.fps)
            if frame_interval < 1:
                frame_interval = 1
            
            frame_count = 0
            saved_count = 0
            
            for i, frame in enumerate(reader):
                # 按间隔采样
                if i % frame_interval == 0:
                    # 检查最大帧数限制
                    if self.config.max_frames and saved_count >= self.config.max_frames:
                        logger.info(f"Reached max frames limit: {self.config.max_frames}")
                        break
                    
                    frame = self._resize_frame(frame)
                    
                    # 保存帧
                    frame_path = Path(temp_dir) / f"frame_{saved_count:06d}.jpg"
                    frame.save(frame_path, format="JPEG", quality=self.config.image_quality)
                    frame_paths.append(str(frame_path))
                    saved_count += 1
            
            reader.close()
            # logger.info(f"Extracted {saved_count} frames to {temp_dir}")
            
            if not frame_paths:
                raise ValueError("No frames extracted from video")
            
            return frame_paths
            
        except Exception as e:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Frame extraction failed: {e}") from e
    
    def _resize_frame(self, frame: np.ndarray) -> PIL.Image.Image:
        """调整帧尺寸以符合限制"""
        h, w = frame.shape[:2]
        max_w, max_h = self.config.max_image_size
        
        if w <= max_w and h <= max_h:
            return Image.fromarray(frame)
        
        # 计算缩放比例
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return Image.fromarray(frame).resize((new_w, new_h), resample=Image.LANCZOS)
    
    @staticmethod
    def cleanup_frames(frame_paths: List[str]):
        """清理临时帧文件"""
        if not frame_paths:
            return
        
        temp_dir = Path(frame_paths[0]).parent
        try:
            shutil.rmtree(temp_dir)
            # logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")


class GeminiVideoAnnotator:
    """Gemini视频标注器"""
    
    def __init__(self, config: VideoAnnotationConfig):
        self.config = config
        self.model = self._init_model()
        self.frame_extractor = VideoFrameExtractor(config)
    
    def _init_model(self) -> OpenAIChatModel:
        """初始化模型"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        logger.info("Initializing Gemini model")
        
        return OpenAIChatModel(
            api_key=api_key,
            model_name="gemini-3.1-pro-preview",
            stream=False,  # 视频标注场景建议关闭stream以提高稳定性
            client_args={"base_url": "https://api.openai.com/v1"},
            generate_kwargs={
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                # 添加这些参数可能有助于减少格式化输出
                # "top_p": 0.95,
                # "top_k": 40,
            }
        )
    
    async def annotate_video(
        self,
        video_path: str,
        prompt_type: str = "general",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        标注视频
        
        Args:
            video_path: 视频文件路径
            prompt_type: 提示词类型 ('roomcaption', 'roomtype', 'general')
            custom_prompt: 自定义提示词（优先于prompt_type）
            
        Returns:
            str: 标注结果
        """
        frame_paths = []
        
        try:
            # 提取帧
            # logger.info(f"Extracting frames from video: {video_path}")
            frame_paths = self.frame_extractor.extract_frames(video_path)
            
            # 获取提示词
            if custom_prompt:
                user_prompt = custom_prompt
                system_prompt = SystemPrompts.GENERAL_DESCRIPTION
            else:
                system_prompt = SystemPrompts.get_prompt(prompt_type)
                user_prompt = self._build_user_prompt(prompt_type, len(frame_paths))
            
            # 分批处理帧（如果帧数过多）
            if len(frame_paths) > self.config.batch_size:
                logger.info(f"Processing {len(frame_paths)} frames in batches of {self.config.batch_size}")
                results = []
                for i in range(0, len(frame_paths), self.config.batch_size):
                    batch = frame_paths[i:i + self.config.batch_size]
                    result = await self._process_frames(batch, system_prompt, user_prompt, prompt_type)
                    results.append(result)
                
                # 合并结果（可根据需要调整合并策略）
                return self._merge_results(results, prompt_type)
            else:
                # 一次性处理所有帧
                return await self._process_frames(frame_paths, system_prompt, user_prompt, prompt_type)
            
        finally:
            # 清理临时文件
            if frame_paths:
                VideoFrameExtractor.cleanup_frames(frame_paths)
    
    def _build_user_prompt(self, prompt_type: str, frame_count: int) -> str:
        """构建用户提示词"""
        base_prompts = {
            "roomcaption": f"Analyze these {frame_count} video frames and provide a concise room description. IMPORTANT: Maximum 100 words, use plain paragraph form with no formatting.",
            "roomtype": f"Based on these {frame_count} video frames, classify the room type.",
            "general": f"Analyze these {frame_count} video frames and provide a comprehensive description of the indoor scene.",
        }
        return base_prompts.get(prompt_type, base_prompts["roomcaption"])
    
    async def _process_frames(
        self,
        frame_paths: List[str],
        system_prompt: str,
        user_prompt: str,
        prompt_type: str = "roomcaption"
    ) -> str:
        """处理一批帧"""
        try:
            # 创建agent
            agent = ReActAgent(
                name="VideoAnnotator",
                sys_prompt=system_prompt,
                model=self.model,
                formatter=OpenAIChatFormatter(),
            )
            
            # 构建消息内容
            contents = []
            for path in frame_paths:
                try:
                    contents.append(
                        ImageBlock(
                            type="image",
                            source=Base64Source(
                                type="base64",
                                media_type="image/jpeg",
                                data=self._get_base64_data(path)
                            ),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
            
            if not contents:
                raise ValueError("No valid images loaded")
            
            contents.append(TextBlock(type="text", text=user_prompt))
            
            # 发送请求
            msg = Msg(name="user", role="user", content=contents)
            logger.info(f"Sending {len(frame_paths)} frames to model...")
            
            resp = await agent(msg)
            
            if not resp or not resp.content:
                raise ValueError("Empty response from model")
            
            # 提取响应文本 (处理返回为字符串或字典的情况)
            if isinstance(resp.content, str):
                raw_text = resp.content
            elif isinstance(resp.content, list):
                for content in resp.content:
                    # skip thinking logs or meta-cognitive self-evaluation paragraphs (common in Gemini outputs)
                    if content.get("type") == "thinking":
                        logger.info(f"Skipping thinking log: {content}")
                        continue
                    if content.get("type") == "text":
                        raw_text = str(content.get("text", ""))
                        break
            elif isinstance(resp.content, dict):
                assert content.get("type") == "text", f"Unexpected content type: {content.get('type')}"
                raw_text = resp.content.get("text", str(resp.content))
            else:
                raw_text = str(resp.content)
            # logger.info(f"Received response content: {resp.content}")
            # logger.info(f"Raw response received: {raw_text}...") 
            # 清理响应文本
            cleaned_content = self._clean_response(raw_text, prompt_type)
            
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            raise
    
    @staticmethod
    def _get_base64_data(image_path: str) -> str:
        """读取图片并转换为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    @staticmethod
    def _clean_response(text: str, prompt_type: str) -> str:
        """
        清理响应文本，移除不需要的格式化标记
        
        Args:
            text: 原始响应文本
            prompt_type: 提示词类型
            
        Returns:
            str: 清理后的文本
        """
        import re
        
        if not text:
            return text
        
        # 移除常见的前缀
            # r"^This (appears to be|is)\s*",
        prefixes_to_remove = [
            r"^Based on the (video )?frames?,?\s*",
            r"^Room [Tt]ype:\s*",
            r"^Room [Ff]unction:\s*",
            r"^\*\*Room Type:\*\*\s*",
            r"^Here is (a |the )?description:?\s*",
            r"^The room type is:?\s*",
        ]
        
        cleaned = text.strip()
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)
        
        # 对于roomtype，只保留第一行并移除所有格式化
        if prompt_type == "roomtype":
            # 取第一个非空行
            lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
            if lines:
                cleaned = lines[0]
            
            # 移除markdown格式
            cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned)  # **text** -> text
            cleaned = re.sub(r'\*(.+?)\*', r'\1', cleaned)      # *text* -> text
            
            # 移除多余标点
            cleaned = cleaned.strip('*-•:：')
        
        # 对于描述类任务，移除markdown格式但保留段落结构
        elif prompt_type in ("roomcaption", "general"):
            # 移除markdown标题
            cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
            
            # 移除markdown粗体/斜体
            cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned)
            cleaned = re.sub(r'\*(.+?)\*', r'\1', cleaned)
            
            # 移除项目符号（如果出现在段落开头）
            cleaned = re.sub(r'^\s*[-•*]\s+', '', cleaned, flags=re.MULTILINE)
            
            # 移除可能的标签行（如 "Key Features:" 单独成行）
            lines = cleaned.split('\n')
            filtered_lines = []
            for line in lines:
                # 跳过只有标签的行
                if re.match(r'^[\*\-\s]*[A-Z][^:]*:\s*$', line.strip()):
                    continue
                # 跳过纯符号行
                if re.match(r'^[\*\-•\s]+$', line.strip()):
                    continue
                filtered_lines.append(line)
            
            cleaned = '\n'.join(filtered_lines)
            
            # 压缩多个换行符
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def _merge_results(self, results: List[str], prompt_type: str) -> str:
        """合并多批次的结果"""
        if prompt_type == "roomtype":
            # 对于房间类型，返回最常见的分类
            # 简单实现：返回第一个结果
            return results[0]
        else:
            # # 对于描述类任务，合并所有结果
            # return "\n\n".join(f"Batch {i+1}:\n{result}" for i, result in enumerate(results))
            return results[0] # 目前先返回第一批结果，后续可根据需要改进合并策略


async def annotate_single_video(
    video_path: str,
    prompt_type: str = "general",
    config: Optional[VideoAnnotationConfig] = None
) -> str:
    """
    便捷函数：标注单个视频
    
    Args:
        video_path: 视频路径
        prompt_type: 提示词类型
        config: 配置（可选）
        
    Returns:
        str: 标注结果
    """
    if config is None:
        config = VideoAnnotationConfig()
    
    annotator = GeminiVideoAnnotator(config)
    return await annotator.annotate_video(video_path, prompt_type)


async def batch_annotate_videos(
    video_paths: List[str],
    prompt_type: str = "general",
    config: Optional[VideoAnnotationConfig] = None,
    max_concurrent: int = 4
) -> Dict[str, str]:
    """
    批量标注多个视频（并发处理）
    
    Args:
        video_paths: 视频路径列表
        prompt_type: 提示词类型
        config: 配置（可选）
        max_concurrent: 最大并发数
        
    Returns:
        Dict[str, str]: 视频路径到标注结果的映射
    """
    if config is None:
        config = VideoAnnotationConfig()
    
    annotator = GeminiVideoAnnotator(config)
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(video_paths), desc="Annotating videos", unit="video")

    async def process_with_limit(video_path: str):
        async with semaphore:
            try:
                result = await annotator.annotate_video(video_path, prompt_type)
                pbar.update(1)
                return video_path, result
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                pbar.update(1)
                return video_path, f"ERROR: {str(e)}"    
            
    tasks = [process_with_limit(vp) for vp in video_paths]
    results = await asyncio.gather(*tasks)
    pbar.close()
    
    return dict(results)


# ============ 示例用法 ============

def parse_args():
    parser = argparse.ArgumentParser(description="Annotate SpatialVideo Dataset with Gemini")
    parser.add_argument(
        "--dataset_rootpath",
        type=str,
        default="/path/to/real-estate-10k/process_re10k_wan_vace_data",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="/path/to/real-estate-10k/process_re10k_wan_vace_data/metadata_wan_funcontrol.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/real-estate-10k/process_re10k_wan_vace_data",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--task_id", type=int, default=0)
    return parser.parse_args()

async def main(args):
    """示例主函数"""
    
    dataset_path = args.dataset_rootpath
    split_filepath = args.split_file
    start_index = args.start_index
    end_index = args.end_index
    num_workers = args.num_workers
    task_id = args.task_id
    
    # annotation config
    config = VideoAnnotationConfig(
        fps=args.sample_fps,  # 1帧/秒
        max_frames=30,  # 最多30帧
        batch_size=20,  # 每批10帧
        image_quality=90,
        max_image_size=(2048, 1024),
    )
    
    scene_df = load_from_csv(split_filepath, start_index=start_index, end_index=end_index)
    logger.info(f"Loaded {len(scene_df)} scenes from split file")
    
    video_list = scene_df["video"].tolist()
    results = await batch_annotate_videos(video_paths=video_list, prompt_type="roomcaption", config=config, max_concurrent=num_workers)
    
    processed_video_paths, processed_video_descs = [], []
    for video_path, result in results.items():
        # scene_desc_txt = result[0]['text']
        scene_desc_txt = result
        processed_video_paths.append(video_path)
        processed_video_descs.append(scene_desc_txt)

    task_i_caption_result_filepath = os.path.join(dataset_path, f"task{task_id}_video_caption_results.csv")
    df = pd.DataFrame({
        "video_path": processed_video_paths,
        "video_caption": processed_video_descs,
    })
    df.to_csv(task_i_caption_result_filepath, index=False)
    logger.info(f"Saved caption results to {task_i_caption_result_filepath}")
    
    # # update the original split file with the video_path to video_caption mapping
    # caption_results = dict(zip(processed_video_paths, processed_video_descs))
    # if "prompt" not in scene_df.columns:
    #     scene_df["prompt"] = ""
    # scene_df["prompt"] = scene_df["video"].map(caption_results).fillna(scene_df["prompt"])
    # updated_split_filepath = os.path.join(os.path.dirname(split_filepath), "metadata_wan_funcontrol_captioned.csv")
    # # updated_split_filepath = os.path.join("metadata_wan_funcontrol_captioned.csv")
    # scene_df.to_csv(updated_split_filepath, index=False)
    # logger.info(f"Updated split file with captions saved to {updated_split_filepath}")

if __name__ == "__main__":
    
    args = parse_args()
    asyncio.run(main(args))
    