import imageio, os, torch, warnings, torchvision, argparse, json, random
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator



class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            # Ensure prompt column is string type to avoid float conversion for NaN values
            if 'prompt' in metadata.columns:
                metadata['prompt'] = metadata['prompt'].astype(str)
                # Replace 'nan' string (from NaN) with empty string
                metadata['prompt'] = metadata['prompt'].replace('nan', '')
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
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
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        action_base_path=None,
        enable_icl=False,
        icl_num_examples=2,
        icl_context_frames=8,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            # In-context learning parameters
            if hasattr(args, 'enable_icl'):
                enable_icl = args.enable_icl
            if hasattr(args, 'icl_num_examples'):
                icl_num_examples = args.icl_num_examples
            if hasattr(args, 'icl_context_frames'):
                icl_context_frames = args.icl_context_frames
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        # In-context learning parameters
        self.enable_icl = enable_icl
        self.icl_num_examples = icl_num_examples
        self.icl_context_frames = icl_context_frames
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            # Ensure prompt column is string type to avoid float conversion for NaN values
            if 'prompt' in metadata.columns:
                metadata['prompt'] = metadata['prompt'].astype(str)
                # Replace 'nan' string (from NaN) with empty string
                metadata['prompt'] = metadata['prompt'].replace('nan', '')
                
                # CRITICAL FIX: Clean prompt - remove video path prefix if present
                # Some CSV prompts start with "video_name.mp4 " prefix, which should be removed
                def clean_prompt(prompt_str):
                    if not isinstance(prompt_str, str) or not prompt_str:
                        return prompt_str
                    # Check if prompt starts with a video path (contains .mp4 or /)
                    # Pattern: "VideoName/1234_5678.mp4 " or "VideoName.mp4 "
                    import re
                    # Match pattern: word/word.mp4 or word.mp4 at the start, followed by space
                    pattern = r'^[A-Za-z0-9_]+(/[A-Za-z0-9_]+)?\.mp4\s+'
                    cleaned = re.sub(pattern, '', prompt_str)
                    # Also handle truncated prompts ending with "..."
                    if cleaned.endswith('...'):
                        cleaned = cleaned[:-3].rstrip()
                    return cleaned.strip()
                
                metadata['prompt'] = metadata['prompt'].apply(clean_prompt)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        self.action_base_path = action_base_path
        
        if self.enable_icl:
            print(f"In-context learning enabled: {icl_num_examples} examples, {icl_context_frames} context frames each")
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
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
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        # Handle multiple frame paths separated by '|' (for frame sequences)
        if '|' in str(file_path):
            # Split the path by '|' to get individual frame paths
            frame_paths = str(file_path).split('|')
            frames = []
            
            # Get base_path (dataset root)
            if not hasattr(self, 'base_path') or not self.base_path:
                warnings.warn(f"Cannot determine base directory for frame sequence: {file_path}")
                return None
            
            base_dir = self.base_path  # This is the dataset root
            
            # Check the first path to determine the format
            first_frame = frame_paths[0].strip() if frame_paths else ""
            
            # If first frame is already an absolute path (from __getitem__ joining),
            # extract the base directory from it
            if os.path.isabs(first_frame):
                # Extract base directory from first frame path
                # First frame format: /path/to/dataset/frames/video_name/frame.png
                # We need to get /path/to/dataset
                parts = first_frame.split(os.sep)
                # Find 'frames' in the path and get everything before it
                if 'frames' in parts:
                    frames_idx = parts.index('frames')
                    base_dir = os.sep.join(parts[:frames_idx])
                else:
                    # Fallback: use self.base_path
                    base_dir = self.base_path
            
            for frame_path in frame_paths:
                frame_path = frame_path.strip()
                if not frame_path:
                    continue
                
                # Construct full path
                if os.path.isabs(frame_path):
                    # Already absolute path (from __getitem__)
                    full_frame_path = frame_path
                else:
                    # Relative path - need to construct full path
                    # Remove 'frames/' prefix if present (we'll add it consistently)
                    if frame_path.startswith('frames/'):
                        frame_path = frame_path[7:]  # Remove 'frames/' prefix
                    
                    # Always join with base_dir + 'frames/' since base_dir is dataset root
                    full_frame_path = os.path.join(base_dir, 'frames', frame_path)
                
                # Load individual frame
                if os.path.exists(full_frame_path):
                    if self.is_image(full_frame_path):
                        frame_data = self.load_image(full_frame_path)
                        if frame_data:
                            frames.extend(frame_data)
                    else:
                        warnings.warn(f"Frame is not an image: {full_frame_path}")
                else:
                    warnings.warn(f"Frame not found: {full_frame_path}")
            
            if frames:
                return frames
            else:
                warnings.warn(f"No frames loaded from sequence: {file_path}")
                return None
        
        # Handle single file (image or video)
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in ["video_name", "start_frame", "end_frame"]:
                if "actions" in data:
                    continue
                try:
                    video_name = data.get("video_name")
                    if video_name is None:
                        warnings.warn(f"video_name is missing in metadata for data_id {data_id}. Skipping action loading.")
                        continue
                    
                    if video_name.endswith(".mp4"):
                        video_name = ".".join(video_name.split(".")[:-1])
                    if "_" in video_name:
                        video_name = "_".join(video_name.split("_")[:4])
                    
                    import json
                    json_path = os.path.join(self.action_base_path, video_name + ".json")
                    
                    # Check if action file exists
                    if not os.path.exists(json_path):
                        warnings.warn(f"Action file does not exist: {json_path}. Skipping action loading for data_id {data_id}.")
                        continue
                    
                    start_frame = data.get("start_frame")
                    end_frame = data.get("end_frame")
                    if start_frame is None or end_frame is None:
                        warnings.warn(f"start_frame or end_frame is missing in metadata for data_id {data_id}. Skipping action loading.")
                        continue
                    
                    json_data = json.load(open(json_path, "r"))['actions']
                    actions = []
                    current_yaw = 0.0
                    for frame_id in range(start_frame+1, end_frame+1):
                        frame_str = str(frame_id)
                        if frame_str not in json_data:
                            warnings.warn(f"Frame {frame_id} not found in action file {json_path}. Skipping this frame.")
                            continue
                        
                        action = json_data[frame_str]
                        new_action = [0.0] * (2 + 2 + 3 + 1 + 2)
                        if action['ws'] == 1:
                            new_action[0] = 1
                        elif action['ws'] == 2:
                            new_action[1] = 1

                        if action['ad'] == 1:
                            new_action[2] = 1
                        elif action['ad'] == 2:
                            new_action[3] = 1
                        
                        if action['scs'] == 1 and action.get("jump_invalid", 0) == 0:
                            new_action[4] = 1
                        elif action['scs'] == 2:
                            new_action[5] = 1
                        elif action['scs'] == 3:
                            new_action[6] = 1
                        
                        if action.get('collision', 0) == 1:
                            new_action[7] = 1
                            new_action[0] = 0
                            new_action[1] = 0
                            new_action[2] = 0
                            new_action[3] = 0

                        pre_pitch = action.get('pre_pitch', 0.0)
                        current_pitch = pre_pitch + action.get('pitch_delta', 0.0) * 15.0
                        current_yaw += action.get('yaw_delta', 0.0) * 15.0
                        new_action[8] = current_pitch
                        new_action[9] = current_yaw

                        actions.append(new_action)
                    data["actions"] = actions
                except Exception as e:
                    warnings.warn(f"Exception while loading actions for data_id {data_id}: {e}. Continuing without actions.")
                    # Don't return None, just continue without actions
                    continue
            elif key == "video":
                # Check if data[key] exists and is not None
                if key not in data or data[key] is None:
                    warnings.warn(f"Video key '{key}' is missing or None in metadata for data_id {data_id}. Skipping this sample.")
                    return None
                
                # Handle frame sequences (paths with '|' separator)
                video_path_str = str(data[key])
                if '|' in video_path_str:
                    # For frame sequences, pass the full path string to load_data
                    # load_data will handle splitting and loading individual frames
                    path = os.path.join(self.base_path, video_path_str)
                    # Don't check path existence here for frame sequences
                    # load_data will handle individual frame loading
                else:
                    path = os.path.join(self.base_path, data[key])
                    # Check if path exists (only for single files)
                    if not os.path.exists(path):
                        warnings.warn(f"Video file does not exist: {path}. Skipping this sample.")
                        return None
                try:
                    data[key] = self.load_data(path)
                    if data[key] is None:
                        warnings.warn(f"Failed to load video file: {path}. load_data returned None.")
                        return None
                except Exception as e:
                    warnings.warn(f"Exception while loading video file {path}: {e}. Skipping this sample.")
                    return None
        
        # In-context learning: sample context examples from dataset
        if self.enable_icl and len(self.data) > 1:
            context_frames_list = []
            context_actions_list = []
            
            # Sample random examples from dataset (excluding current one)
            current_idx = data_id % len(self.data)
            candidate_indices = [i for i in range(len(self.data)) if i != current_idx]
            if len(candidate_indices) > 0:
                num_samples = min(self.icl_num_examples, len(candidate_indices))
                sampled_indices = random.sample(candidate_indices, num_samples)
                
                for sample_idx in sampled_indices:
                    sample_data = self.data[sample_idx].copy()
                    # Load video for context
                    if "video" in self.data_file_keys and "video" in sample_data:
                        video_path = os.path.join(self.base_path, sample_data["video"])
                        sample_video = self.load_data(video_path)
                        if sample_video is not None and len(sample_video) >= self.icl_context_frames:
                            # Sample context_frames from the video
                            start_idx = random.randint(0, max(0, len(sample_video) - self.icl_context_frames))
                            context_frames = sample_video[start_idx:start_idx + self.icl_context_frames]
                            context_frames_list.extend(context_frames)
                            
                            # Load corresponding actions if available
                            if self.action_base_path is not None and "video_name" in sample_data:
                                try:
                                    sample_video_name = sample_data["video_name"]
                                    if sample_video_name.endswith(".mp4"):
                                        sample_video_name = ".".join(sample_video_name.split(".")[:-1])
                                    if "_" in sample_video_name:
                                        sample_video_name = "_".join(sample_video_name.split("_")[:4])
                                    sample_json_path = os.path.join(self.action_base_path, sample_video_name + ".json")
                                    if os.path.exists(sample_json_path):
                                        sample_json_data = json.load(open(sample_json_path, "r"))['actions']
                                        sample_start_frame = sample_data.get("start_frame", 0)
                                        sample_end_frame = sample_data.get("end_frame", len(sample_video))
                                        
                                        # Get actions for the context frames
                                        context_actions = []
                                        context_yaw = 0.0
                                        for frame_idx in range(sample_start_frame + start_idx + 1, 
                                                             min(sample_start_frame + start_idx + self.icl_context_frames + 1, sample_end_frame + 1)):
                                            if str(frame_idx) in sample_json_data:
                                                action = sample_json_data[str(frame_idx)]
                                                new_action = [0.0] * (2 + 2 + 3 + 1 + 2)
                                                if action['ws'] == 1:
                                                    new_action[0] = 1
                                                elif action['ws'] == 2:
                                                    new_action[1] = 1
                                                if action['ad'] == 1:
                                                    new_action[2] = 1
                                                elif action['ad'] == 2:
                                                    new_action[3] = 1
                                                if action['scs'] == 1 and action.get("jump_invalid", 0) == 0:
                                                    new_action[4] = 1
                                                elif action['scs'] == 2:
                                                    new_action[5] = 1
                                                elif action['scs'] == 3:
                                                    new_action[6] = 1
                                                if action.get('collision', 0) == 1:
                                                    new_action[7] = 1
                                                    new_action[0] = 0
                                                    new_action[1] = 0
                                                    new_action[2] = 0
                                                    new_action[3] = 0
                                                pre_pitch = action.get('pre_pitch', 0.0)
                                                current_pitch = pre_pitch + action.get('pitch_delta', 0.0) * 15.0
                                                context_yaw += action.get('yaw_delta', 0.0) * 15.0
                                                new_action[8] = current_pitch
                                                new_action[9] = context_yaw
                                                context_actions.append(new_action)
                                        context_actions_list.extend(context_actions[:len(context_frames)])
                                except Exception as e:
                                    # If loading actions fails, just skip
                                    pass
            
            if context_frames_list:
                data["context_frames"] = context_frames_list
                if context_actions_list and len(context_actions_list) == len(context_frames_list):
                    data["context_actions"] = context_actions_list
        
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat

    @staticmethod
    def get_one_hot(action, range=2):
        one_hot = [0] * (range + 1)
        one_hot[action] = 1
        return one_hot



import numpy as np


class CamVideoDataset(torch.utils.data.Dataset):
    """Dataset for Context-as-Memory camera pose conditioned training (ported from VWM).

    Loads 81 PNG frames from UE scenes with random temporal cropping and extracts
    corresponding camera poses as 12-dim relative RT vectors subsampled to match
    the 21 latent frames.
    """
    def __init__(
        self,
        base_path=None,
        num_frames=81,
        height=None, width=None,
        max_pixels=1920*1080,
        height_division_factor=16, width_division_factor=16,
        repeat=1,
        metadata_path=None,
        args=None,
        cam_position_scale=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = getattr(args, "dataset_metadata_path", metadata_path)
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            repeat = args.dataset_repeat
            cam_position_scale = getattr(args, "cam_position_scale", 0.01)
            self.use_condition_context_frames = getattr(args, "use_condition_context_frames", False)
            self.condition_first_frame = getattr(args, "condition_first_frame", False)
            self.condition_history_keyframes = getattr(args, "condition_history_keyframes", False)
            self.condition_use_camera_pose = getattr(args, "condition_use_camera_pose", True)
            self.num_condition_frames = getattr(args, "num_condition_frames", 1)
            self.condition_frame_mode = getattr(args, "condition_frame_mode", "first_frame_only")
            self.overlap_labels_root = getattr(args, "overlap_labels_root", None)
            self.condition_t2v_ratio = getattr(args, "condition_t2v_ratio", 0.10)
            self.condition_i2v_ratio = getattr(args, "condition_i2v_ratio", 0.10)
        else:
            self.use_condition_context_frames = False
            self.condition_first_frame = False
            self.condition_history_keyframes = False
            self.condition_use_camera_pose = True
            self.num_condition_frames = 1
            self.condition_frame_mode = "first_frame_only"
            self.overlap_labels_root = None
            self.condition_t2v_ratio = 0.10
            self.condition_i2v_ratio = 0.10

        if cam_position_scale is None:
            cam_position_scale = 0.01
        self.cam_position_scale = float(cam_position_scale)

        self.base_path = base_path
        self.frames_dir = os.path.join(base_path, "frames")
        self.jsons_dir = os.path.join(base_path, "jsons")
        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.repeat = repeat

        if height is not None and width is not None:
            self.dynamic_resolution = False
        else:
            self.dynamic_resolution = True

        captions_path = os.path.join(base_path, "captions.txt")
        self.scene_captions = {}
        with open(captions_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) < 2:
                    continue
                clip_path, caption = parts
                scene_name = "/".join(clip_path.split("/")[:-1])
                fname = clip_path.split("/")[-1].replace(".mp4", "")
                clip_start = int(fname.split("_")[0])
                if scene_name not in self.scene_captions:
                    self.scene_captions[scene_name] = []
                self.scene_captions[scene_name].append((clip_start, caption))

        for scene_name in self.scene_captions:
            self.scene_captions[scene_name].sort(key=lambda x: x[0])

        self.scene_names = sorted(self.scene_captions.keys())
        self.metadata_rows = []
        if metadata_path and os.path.isfile(metadata_path):
            metadata = pd.read_csv(metadata_path)
            if "prompt" in metadata.columns:
                metadata["prompt"] = metadata["prompt"].astype(str)
            self.metadata_rows = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        self.pose_cache = {}
        self.overlap_cache = {}
        self.invalid_scenes = set()
        self.invalid_metadata_indices = set()
        self.overlap_labels_root = self._resolve_overlap_labels_root(base_path, self.overlap_labels_root)
        self._validate_condition_config()

        total_scenes = len(self.scene_names)
        total_captions = sum(len(v) for v in self.scene_captions.values())
        metadata_msg = f", metadata_rows={len(self.metadata_rows)}" if self.metadata_rows else ""
        effective_len = (len(self.metadata_rows) if self.metadata_rows else total_scenes) * repeat
        print(f"CamVideoDataset: {total_scenes} scenes, {total_captions} captions{metadata_msg}, "
              f"repeat={repeat}, cam_position_scale={self.cam_position_scale}, "
              f"effective length={effective_len}")

    def _resolve_overlap_labels_root(self, base_path, overlap_labels_root):
        candidate_roots = []
        if overlap_labels_root is not None:
            candidate_roots.append(overlap_labels_root)
        if base_path is not None:
            candidate_roots.append(os.path.join(base_path, "overlap_labels"))
        for root in candidate_roots:
            if root is not None and os.path.isdir(root):
                return root
        return overlap_labels_root

    def _validate_condition_config(self):
        if self.condition_t2v_ratio < 0 or self.condition_i2v_ratio < 0:
            raise ValueError("Condition sampling ratios must be non-negative.")
        if self.condition_t2v_ratio + self.condition_i2v_ratio >= 1.0:
            raise ValueError("condition_t2v_ratio + condition_i2v_ratio must be < 1.0.")
        needs_overlap = (
            self.use_condition_context_frames
            and self.condition_frame_mode == "first_plus_overlap"
            and self.condition_history_keyframes
            and self.num_condition_frames > 1
        )
        if needs_overlap and (self.overlap_labels_root is None or not os.path.isdir(self.overlap_labels_root)):
            raise FileNotFoundError(
                "K-frame condition mode requires overlap_labels_root. "
                "Pass --overlap_labels_root or keep overlap_labels under dataset_base_path/overlap_labels."
            )

    def _load_scene_poses(self, scene_name):
        if scene_name not in self.pose_cache:
            json_path = os.path.join(self.jsons_dir, scene_name + ".json")
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Pose JSON for scene '{scene_name}' is missing or corrupt: {e}")
            if not isinstance(data, dict) or "CineCameraActor" not in data:
                raise ValueError(
                    f"Pose JSON for scene '{scene_name}' lacks 'CineCameraActor' key "
                    f"(found keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__})."
                )
            cine = data["CineCameraActor"]
            if not isinstance(cine, dict) or len(cine) == 0:
                raise ValueError(f"Pose JSON for scene '{scene_name}' has empty 'CineCameraActor' entries.")
            self.pose_cache[scene_name] = cine
        return self.pose_cache[scene_name]

    def _find_nearest_caption(self, scene_name, start_frame):
        captions = self.scene_captions[scene_name]
        best_idx = 0
        best_dist = abs(captions[0][0] - start_frame)
        for i, (clip_start, _) in enumerate(captions):
            dist = abs(clip_start - start_frame)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return captions[best_idx][1]

    @staticmethod
    def _compute_rt(position, rotation):
        x, y, z = position
        yaw_rad = np.radians(rotation[2])
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
        return [x, y, z] + R.flatten().tolist()

    @staticmethod
    def _to_relative_rt(rt_list, ref_rt):
        R_ref = np.array(ref_rt[3:]).reshape(3, 3)
        T_ref = np.array(ref_rt[:3]).reshape(3, 1)
        R_ref_inv = R_ref.T
        T_ref_inv = -R_ref_inv @ T_ref
        result = []
        for rt in rt_list:
            R_i = np.array(rt[3:]).reshape(3, 3)
            T_i = np.array(rt[:3]).reshape(3, 1)
            R_new = R_ref_inv @ R_i
            T_new = R_ref_inv @ T_i + T_ref_inv
            result.append(T_new.flatten().tolist() + R_new.flatten().tolist())
        return result

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def _load_resized_frame(self, scene_name, frame_index, target_height, target_width):
        frame_path = os.path.join(self.frames_dir, scene_name, f"{frame_index:04d}.png")
        img = Image.open(frame_path).convert("RGB")
        return self.crop_and_resize(img, target_height, target_width)

    @staticmethod
    def _parse_frame_token(token):
        token = str(token).strip()
        if not token:
            return None, None
        parts = token.split("/")
        if parts and parts[0] == "frames":
            parts = parts[1:]
        if len(parts) < 2:
            return None, None
        scene_name = "/".join(parts[:-1])
        stem = os.path.splitext(parts[-1])[0]
        try:
            frame_index = int(stem)
        except ValueError:
            return None, None
        return scene_name, frame_index

    def _metadata_scene_and_indices(self, row):
        video_field = str(row.get("video", "") or "")
        tokens = [t for t in video_field.split("|") if t]
        parsed = [self._parse_frame_token(t) for t in tokens]
        parsed = [(s, i) for s, i in parsed if s is not None and i is not None]
        if parsed:
            scene_name = str(row.get("video_name", "") or parsed[0][0])
            frame_indices = [i for _, i in parsed[: self.num_frames]]
        else:
            scene_name = str(row.get("video_name", "") or "").strip()
            if not scene_name:
                raise ValueError("metadata row lacks video_name and parseable video paths")
            start_frame = int(row.get("start_frame", 0) or 0)
            frame_indices = list(range(start_frame, start_frame + self.num_frames))
        if len(frame_indices) < self.num_frames:
            raise ValueError(f"metadata row has {len(frame_indices)} frames, expected {self.num_frames}")
        return scene_name, frame_indices[: self.num_frames]

    def _load_overlap_frames(self, scene_name, frame_index):
        if self.overlap_labels_root is None:
            return []
        cache_key = (scene_name, int(frame_index))
        if cache_key not in self.overlap_cache:
            overlap_path = os.path.join(self.overlap_labels_root, scene_name, f"{int(frame_index)}.json")
            if not os.path.exists(overlap_path):
                self.overlap_cache[cache_key] = []
            else:
                with open(overlap_path, "r") as f:
                    overlap_data = json.load(f)
                overlaps = overlap_data.get("overlapping_frames", [])
                self.overlap_cache[cache_key] = [int(idx) for idx in overlaps]
        return self.overlap_cache[cache_key]

    def _compute_scene_rt(self, scene_name, frame_index):
        frame_data = self._load_scene_poses(scene_name)[str(int(frame_index))]
        raw_pos = frame_data["position"]
        pos = [float(p) * self.cam_position_scale for p in raw_pos]
        return self._compute_rt(pos, frame_data["rotation"])

    def _sample_condition_mode(self):
        if not self.use_condition_context_frames:
            return "disabled"
        if (
            self.condition_frame_mode != "first_plus_overlap"
            or not self.condition_history_keyframes
            or self.num_condition_frames <= 1
        ):
            return "first_frame_only"
        sample = random.random()
        if sample < self.condition_t2v_ratio:
            return "text_only"
        if sample < self.condition_t2v_ratio + self.condition_i2v_ratio:
            return "first_frame_only"
        return "first_plus_overlap"

    def _sample_overlap_conditions(self, scene_name, start_frame, ref_rt, target_height, target_width, num_extra_conditions):
        if num_extra_conditions <= 0:
            return [], [], []
        window_indices = set(range(start_frame, start_frame + self.num_frames))
        target_candidates = list(range(start_frame + 1, start_frame + self.num_frames))
        sampled_target_frames = random.sample(target_candidates, k=min(num_extra_conditions, len(target_candidates)))
        overlap_frames = []
        overlap_indices = []
        overlap_actions = []
        used_condition_indices = set()
        for target_frame_idx in sampled_target_frames:
            candidate_indices = [
                idx for idx in self._load_overlap_frames(scene_name, target_frame_idx)
                if idx not in window_indices and idx != target_frame_idx and idx not in used_condition_indices
            ]
            if len(candidate_indices) == 0:
                return None
            chosen_idx = random.choice(candidate_indices)
            used_condition_indices.add(chosen_idx)
            overlap_indices.append(chosen_idx)
            overlap_frames.append(self._load_resized_frame(scene_name, chosen_idx, target_height, target_width))
            if self.condition_use_camera_pose:
                overlap_rt = self._compute_scene_rt(scene_name, chosen_idx)
                overlap_actions.append(self._to_relative_rt([overlap_rt], ref_rt)[0])
        if len(overlap_frames) != num_extra_conditions:
            return None
        return overlap_frames, overlap_indices, overlap_actions

    def _try_get_sample(self, scene_name):
        cam_data = self._load_scene_poses(scene_name)
        max_start = len(cam_data) - self.num_frames
        if max_start < 0:
            raise ValueError(f"Scene {scene_name} has fewer than {self.num_frames} frames.")
        start_frame = random.randint(0, max_start)
        end_frame = start_frame + self.num_frames - 1

        frames = []
        for i in range(start_frame, end_frame + 1):
            frame_path = os.path.join(self.frames_dir, scene_name, f"{i:04d}.png")
            img = Image.open(frame_path).convert("RGB")
            img = self.crop_and_resize(img, *self.get_height_width(img))
            frames.append(img)

        prompt = self._find_nearest_caption(scene_name, start_frame)

        rt_list_abs = []
        for i in range(start_frame, end_frame + 1):
            key = str(i)
            if key not in cam_data:
                raise ValueError(f"Scene {scene_name} missing pose for frame {i}.")
            frame_data = cam_data[key]
            raw_pos = frame_data["position"]
            pos = [float(p) * self.cam_position_scale for p in raw_pos]
            rt = self._compute_rt(pos, frame_data["rotation"])
            rt_list_abs.append(rt)

        rt_list = self._to_relative_rt(rt_list_abs, rt_list_abs[0])
        pose_indices = list(range(0, self.num_frames, 4))
        actions = [rt_list[i] for i in pose_indices]

        return {
            "video": frames,
            "prompt": prompt,
            "actions": actions,
            "video_name": scene_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            **self._build_condition_context_payload(
                frames=frames,
                scene_name=scene_name,
                start_frame=start_frame,
                ref_rt=rt_list_abs[0],
                actions=actions,
            ),
        }

    def _try_get_metadata_sample(self, row):
        scene_name, frame_indices = self._metadata_scene_and_indices(row)
        start_frame = int(frame_indices[0])
        end_frame = int(frame_indices[-1])
        cam_data = self._load_scene_poses(scene_name)

        frames = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(self.frames_dir, scene_name, f"{int(frame_idx):04d}.png")
            img = Image.open(frame_path).convert("RGB")
            img = self.crop_and_resize(img, *self.get_height_width(img))
            frames.append(img)

        prompt = row.get("prompt", None)
        if prompt is None or str(prompt).strip() == "" or str(prompt).lower() == "nan":
            prompt = self._find_nearest_caption(scene_name, start_frame)
        else:
            prompt = str(prompt)

        rt_list_abs = []
        for frame_idx in frame_indices:
            key = str(int(frame_idx))
            if key not in cam_data:
                raise ValueError(f"Scene {scene_name} missing pose for frame {frame_idx}.")
            frame_data = cam_data[key]
            raw_pos = frame_data["position"]
            pos = [float(p) * self.cam_position_scale for p in raw_pos]
            rt = self._compute_rt(pos, frame_data["rotation"])
            rt_list_abs.append(rt)

        rt_list = self._to_relative_rt(rt_list_abs, rt_list_abs[0])
        pose_indices = list(range(0, len(frame_indices), 4))
        actions = [rt_list[i] for i in pose_indices]

        return {
            "video": frames,
            "prompt": prompt,
            "actions": actions,
            "video_name": scene_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            **self._build_condition_context_payload(
                frames=frames,
                scene_name=scene_name,
                start_frame=start_frame,
                ref_rt=rt_list_abs[0],
                actions=actions,
            ),
        }

    def __getitem__(self, data_id):
        if self.metadata_rows:
            n = len(self.metadata_rows)
            max_attempts = min(64, n)
            last_error = None
            for attempt in range(max_attempts):
                idx = (data_id + attempt) % n
                if idx in self.invalid_metadata_indices:
                    continue
                try:
                    return self._try_get_metadata_sample(self.metadata_rows[idx])
                except (ValueError, FileNotFoundError, KeyError, OSError) as e:
                    self.invalid_metadata_indices.add(idx)
                    last_error = e
                    if attempt < 3 or attempt % 8 == 0:
                        print(
                            f"[CamVideoDataset] Skipping invalid metadata row {idx} "
                            f"({type(e).__name__}: {e}); attempt {attempt + 1}/{max_attempts}"
                        )
                    continue
            raise RuntimeError(
                f"CamVideoDataset: exhausted {max_attempts} metadata attempts starting from index {data_id}; "
                f"last error: {type(last_error).__name__}: {last_error}"
            )

        n = len(self.scene_names)
        if n == 0:
            raise RuntimeError("CamVideoDataset has no scenes.")
        max_attempts = min(64, n)
        last_error = None
        for attempt in range(max_attempts):
            idx = (data_id + attempt) % n
            scene_name = self.scene_names[idx]
            if scene_name in self.invalid_scenes:
                continue
            try:
                return self._try_get_sample(scene_name)
            except (ValueError, FileNotFoundError, KeyError, OSError) as e:
                self.invalid_scenes.add(scene_name)
                last_error = e
                if attempt < 3 or attempt % 8 == 0:
                    print(
                        f"[CamVideoDataset] Skipping invalid scene '{scene_name}' "
                        f"({type(e).__name__}: {e}); attempt {attempt + 1}/{max_attempts}"
                    )
                continue
        raise RuntimeError(
            f"CamVideoDataset: exhausted {max_attempts} attempts starting from index {data_id}; "
            f"last error: {type(last_error).__name__}: {last_error}"
        )

    def _build_condition_context_payload(self, frames, scene_name, start_frame, ref_rt, actions):
        if not self.use_condition_context_frames:
            return {}
        payload = {
            "use_condition_context_frames": False,
            "condition_frames": [],
            "condition_frame_indices": [],
            "condition_source": None,
            "condition_actions": [],
        }
        condition_mode = self._sample_condition_mode()
        payload["condition_source"] = condition_mode
        if condition_mode == "text_only":
            return payload
        payload["use_condition_context_frames"] = True
        if self.condition_first_frame:
            payload["condition_frames"].append(frames[0])
            payload["condition_frame_indices"].append(start_frame)
            payload["condition_source"] = "first_frame_only"
            if self.condition_use_camera_pose and actions:
                payload["condition_actions"].append(list(actions[0]))
        if (
            condition_mode == "first_plus_overlap"
            and self.condition_history_keyframes
            and self.num_condition_frames > len(payload["condition_frames"])
        ):
            num_extra_conditions = self.num_condition_frames - len(payload["condition_frames"])
            overlap_payload = self._sample_overlap_conditions(
                scene_name=scene_name,
                start_frame=start_frame,
                ref_rt=ref_rt,
                target_height=frames[0].size[1],
                target_width=frames[0].size[0],
                num_extra_conditions=num_extra_conditions,
            )
            if overlap_payload is None:
                return payload
            overlap_frames, overlap_indices, overlap_actions = overlap_payload
            payload["condition_frames"].extend(overlap_frames)
            payload["condition_frame_indices"].extend(overlap_indices)
            if self.condition_use_camera_pose:
                payload["condition_actions"].extend(overlap_actions)
            payload["condition_source"] = "first_plus_overlap"
        return payload

    def __len__(self):
        if self.metadata_rows:
            return len(self.metadata_rows) * self.repeat
        return len(self.scene_names) * self.repeat


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)



def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], drop_last=True)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(loss)
                scheduler.step()
        model_logger.on_epoch_end(accelerator, model, epoch_id)

def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], drop_last=True)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--use_condition_context_frames", default=False, action="store_true", help="Enable appended clean condition latents.")
    parser.add_argument("--condition_first_frame", default=False, action="store_true", help="Use the current clip first frame as a clean condition frame.")
    parser.add_argument("--condition_history_keyframes", default=False, action="store_true", help="Use overlap-based keyframes as conditions.")
    parser.add_argument("--condition_use_camera_pose", default=True, action="store_true", help="Inject camera pose for condition frames.")
    parser.add_argument("--num_condition_frames", type=int, default=1, help="Number of condition frames.")
    parser.add_argument("--condition_frame_mode", type=str, default="first_frame_only", help="Condition frame selection mode.")
    parser.add_argument("--overlap_labels_root", type=str, default=None, help="Root dir for overlap label JSONs.")
    parser.add_argument("--condition_t2v_ratio", type=float, default=0.10, help="Ratio of text-only condition samples.")
    parser.add_argument("--condition_i2v_ratio", type=float, default=0.10, help="Ratio of first-frame-only condition samples.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser
