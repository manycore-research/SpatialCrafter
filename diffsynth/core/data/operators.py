import os
from pickle import TRUE
import torch, imageio, torchvision
import numpy as np
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data, *args, **kwargs):
        for operator in self.operators:
            data = operator(data, *args, **kwargs)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data, *args, **kwargs):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data, *args, **kwargs):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data, *args, **kwargs):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data, *args, **kwargs):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data, *args, **kwargs):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str, *args, **kwargs):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image, *args, **kwargs):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data, *args, **kwargs):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, start_frame=0, end_frame=-1, *args, **kwargs):
        # print(f"[LoadVideo] Loading video from: {data}")
        reader = imageio.get_reader(data)
        # desired number of frames to load
        num_frames = self.get_num_frames(reader)
        # print(f"[LoadVideo] Total frames to load: {num_frames}")
        
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame + frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            # frame.save(f"debug_video_frame_{frame_id}.png")
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        reader.close()
        # print(f"[LoadVideo] Loaded {len(frames)} frames from {start_frame} to {end_frame} on Mp4: {data}")
        return frames

class SCMImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image: np.ndarray, target_height, target_width):
        width, height = image.shape[1], image.shape[0]
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            torch.from_numpy(image).permute(2, 0, 1).float(),
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image: np.ndarray):
        if self.height is None or self.width is None:
            width, height = image.shape[1], image.shape[0]
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: np.ndarray, *args, **kwargs):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image
    
class LoadSCM(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def get_num_frames(self, actual_num_frames: int):
        num_frames = self.num_frames
        if actual_num_frames < num_frames:
            num_frames = actual_num_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def __call__(self, data: str, start_frame=0, end_frame=-1, *args, **kwargs):
        scene_coord_maps_dict = np.load(data, allow_pickle=True)
        scene_coord_maps = scene_coord_maps_dict["scm"]      # (num_frames, 3, h, w), range [-1, 1]

        actual_num_frames = scene_coord_maps.shape[0]
        num_frames = self.get_num_frames(actual_num_frames)

        frames = []
        for frame_id in range(num_frames):
            frame = scene_coord_maps[start_frame + frame_id]    # (3, h, w)
            frame = np.transpose(frame, (1, 2, 0))              # (h, w, 3)
            # print(f"[LoadSCM] frame shape {frame.shape} from SCM: {data}")
            frame = self.frame_processor(frame)                 # (3, h, w)
            # print(f"[LoadSCM] Processed frame shape {frame.shape}")
            # if True:
            #     torchvision.utils.save_image(frame, f"debug_scm_frame_{frame_id}.png")
            #     import open3d as o3d
            #     points = o3d.geometry.PointCloud()
            #     points.points = o3d.utility.Vector3dVector(frame.permute(1, 2, 0).reshape(-1, 3).numpy())
            #     o3d.io.write_point_cloud(f"debug_scm_frame_{frame_id}.ply", points)
            frames.append(frame)
            if len(frames) >= num_frames:
                break

        # print(f"[LoadSCM] Loaded {len(frames)} frames from {start_frame} to {end_frame} on SCM: {data}")
        return frames
    
class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data, *args, **kwargs):
        return [self.operator(i, *args, **kwargs) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, start_frame=0, end_frame=-1, *args, **kwargs):
        # desired number of frames to load
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
            
        for i, img in enumerate(images):
            if i < start_frame:
                continue
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        # print(f"[LoadGIF] Loaded {len(frames)} frames from GIF: {data}")
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str, *args, **kwargs):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data, *args, **kwargs)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data, *args, **kwargs):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data, *args, **kwargs)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data, *args, **kwargs):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data, *args, **kwargs):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str, *args, **kwargs):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio
