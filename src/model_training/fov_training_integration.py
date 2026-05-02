"""
FOV-based context frame retrieval for Context-as-Memory (CAM) training integration.

Implements CAM's FOV overlap-based context frame selection during training
"""

import os
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from PIL import Image


def _load_frame_png(frame_file: str) -> Optional[Image.Image]:
    """Load a single frame from PNG file."""
    if os.path.exists(frame_file):
        try:
            return Image.open(frame_file).convert('RGB')
        except Exception:
            pass
    return None

try:
    from .fov_retrieval import load_overlap_frames, load_camera_pose, load_poses_dict, compute_fov_overlap_3d
    from .rt_utils import pose_to_rt, rt_to_pose, convert_rt_to_relative
except Exception:
    try:
        from fov_retrieval import load_overlap_frames, load_camera_pose, load_poses_dict, compute_fov_overlap_3d
        from rt_utils import pose_to_rt, rt_to_pose, convert_rt_to_relative
    except Exception:
        # Minimal fallback to keep training running even if import path is odd.
        def load_overlap_frames(overlap_labels_dir, video_name, frame_idx):
            import json
            overlap_file = os.path.join(overlap_labels_dir, video_name, f"{frame_idx}.json")
            if not os.path.exists(overlap_file):
                return []
            try:
                with open(overlap_file, "r") as f:
                    data = json.load(f)
                frames = data.get("overlapping_frames", [])
                out = []
                for x in frames:
                    try:
                        out.append(int(x))
                    except:
                        pass
                return out
            except:
                return []
        
        def load_camera_pose(json_file, frame_idx):
            return None

        def load_poses_dict(json_file):
            return {}
        
        def compute_fov_overlap_3d(pose1, pose2, fov_degrees=52.67, max_distance=500.0):
            return 0.0


def retrieve_simple_context_frames(
    data: Dict,
    dataset_base_path: str,
    top_k: int = 4,  # Number of overlap frames to retrieve. First Frame will be added automatically.
    drop_overlap_probability: float = 0.1,  # 10% probability to drop overlap frames (paper strategy)
    use_rt_relative: bool = False,  # Experiment 1_4_2: use RT relative conversion (aligned with CAM paper)
) -> Tuple[List[Image.Image], List, List[int], int, str, str]:
    """
    Retrieve context frames according to CAM paper [2506.03141] Context as Memory.
    
    Data Structure (Precomputed Retrieval Results):
    - Each JSON file: overlap_labels/{video_name}/{frame_index}.json
    - JSON structure: {
        "frame_index": "0",  # First Frame (short-term memory)
        "overlapping_frames": ["2796", "2797", ..., "3839", ...]  # Long-term memory (sample 4)
      }
    - frame_index and the next 80 frames constitute GT (target frames to generate)
    - Iterating through all JSON files = one epoch
    
    Design principles:
    1. First Frame (frame_index from JSON) as Immediate Condition:
       - The frame_index in JSON is the First Frame (short-term memory)
       - Provides immediate visual and temporal starting point (Image-to-Video mode)
       - Always included as context
       - GT: frame_index and the next 80 frames (81 frames total)
    
    2. Overlap Frames (from overlapping_frames in JSON) as Long-term Memory:
       - Retrieved from precomputed overlap_labels JSON files
       - Precomputed lists may be very long (e.g., [2796, 3839, 4183, ..., 6339])
       - Random uniform sampling: sample top_k (4) frames from overlapping_frames list
       - Provides long-term consistency information
       - Memory frames are unordered snapshots (no temporal sequence)
    
    3. Context Composition:
       - Order: [First Frame, Overlap Frame 1, Overlap Frame 2, Overlap Frame 3, Overlap Frame 4]
       - Total: 1 First Frame + top_k Overlap Frames = top_k + 1 frames (e.g., 5 frames)
       - Context frames are concatenated with target frames in temporal dimension
    
    4. 10% Probability Drop Strategy:
       - With 10% probability, drop all Overlap Frames, only use First Frame
       - Simulates video generation starting stage (no historical memory)
       - Forces model to generate reasonable videos without long-term memory assistance
    
    5. Epoch Definition:
       - One epoch = iterate through all JSON files in overlap_labels/{video_name}/
       - Each JSON file = one training sample
    
    6. Positional Encoding Note:
       - Memory frames should NOT use original absolute time positions
       - They should be treated as unordered image collection or use memory ID encoding only
       - First Frame should use explicit "Ref Frame" encoding
    
    Args:
        data: Training data dict containing video frames and metadata
        dataset_base_path: Base path to Context-as-Memory dataset
        top_k: Number of overlap frames to retrieve (default: 4). First Frame will be added automatically.
        drop_overlap_probability: Probability to drop overlap frames (default: 0.1 = 10%)

    Returns:
        Tuple of:
          (context_frames, context_actions, context_indices, current_frame_idx, video_name, source)
    """
    import random

    video_frames = data.get("video", [])
    # Get segment boundaries
    start_frame = data.get("start_frame", 0)
    end_frame = data.get("end_frame", None)
    
    # Use frame_idx if available, otherwise calculate from segment (middle of segment)
    # This ensures we can find previous frames for context retrieval
    if "frame_idx" in data:
        current_frame_idx = data.get("frame_idx")
    else:
        # Calculate middle of segment as reference frame (same as training convention)
        if end_frame is not None:
            current_frame_idx = (start_frame + end_frame) // 2
        else:
            # Fallback: use middle of video_frames if available
            if len(video_frames) > 0:
                current_frame_idx = len(video_frames) // 2
            else:
                current_frame_idx = 0
    
    # First Frame: current segment's first frame (start_frame) - ALWAYS included
    first_frame_idx = start_frame
    
    video_name = data.get("video_name", "")
    context_frames: List[Image.Image] = []
    context_actions: List = []
    context_indices: List[int] = []
    source = "none"

    # Get video frames from data
    if not isinstance(video_frames, list):
        video_frames = []

    if not video_name:
        # Try to infer from data
        if "video_path" in data:
            video_name = os.path.basename(data["video_path"]).replace(".mp4", "").replace(".avi", "")
        elif "file_path" in data:
            video_name = os.path.basename(data["file_path"]).replace(".mp4", "").replace(".avi", "")

    # Step 1: Load First Frame (current segment's first frame) - ALWAYS included
    # According to JSON structure: frame_index in JSON is the First Frame (short-term memory)
    # frame_index and the next 80 frames constitute GT (target frames to generate)
    # First Frame provides immediate visual and temporal starting point (Image-to-Video mode)
    frames_dir = os.path.join(dataset_base_path, 'frames', video_name)
    first_frame_loaded = False
    
    # Experiment 1_4_2: Load camera poses once for all context frames (one JSON read)
    json_file = os.path.join(dataset_base_path, "jsons", f"{video_name}.json")
    poses_dict = load_poses_dict(json_file)
    first_frame_pose_rt = None
    first_frame_pose = poses_dict.get(str(first_frame_idx))
    if first_frame_pose is not None:
        first_frame_pose_rt = pose_to_rt(first_frame_pose)
        if first_frame_pose_rt is not None:
            # Experiment 1_4_2: When use_rt_relative, first frame = reference frame = identity RT
            # Target actions use ref=first_frame, so context first frame must also be identity
            # to align with target's coordinate system (same frame = same RT representation)
            if use_rt_relative:
                # Identity RT: [t=0,0,0, R=eye(3)] = [0,0,0,1,0,0,0,1,0,0,0,1]
                context_actions.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            else:
                context_actions.append(first_frame_pose_rt)
        else:
            context_actions.append([0.0] * 12)
    else:
        context_actions.append([0.0] * 12)
    
    if os.path.exists(frames_dir):
        first_frame_file = os.path.join(frames_dir, f"{first_frame_idx:04d}.png")
        if os.path.exists(first_frame_file):
            try:
                first_frame = Image.open(first_frame_file).convert('RGB')
                context_frames.append(first_frame)
                context_indices.append(first_frame_idx)
                first_frame_loaded = True
            except Exception as e:
                pass  # Frame loading failed, skip
    
    # Fallback: use first frame from video_frames if file not found
    if not first_frame_loaded and len(video_frames) > 0:
        if isinstance(video_frames[0], Image.Image):
            context_frames.append(video_frames[0])
            context_indices.append(start_frame)
            first_frame_loaded = True
        elif isinstance(video_frames[0], str):
            try:
                first_frame = Image.open(video_frames[0]).convert('RGB')
                context_frames.append(first_frame)
                context_indices.append(start_frame)
                first_frame_loaded = True
            except Exception as e:
                pass  # Fallback frame loading failed, skip

    # Step 2: Retrieve Overlap Frames (long-term memory) - with 10% probability drop
    # According to the data structure:
    # - Each JSON file in overlap_labels/{video_name}/{frame_index}.json contains:
    #   - frame_index: First Frame (short-term memory) - this is the current segment's start
    #   - overlapping_frames: List of historical frames (long-term memory) - sample 4 from this list
    # - frame_index and the next 80 frames constitute GT (target frames to generate)
    # - Iterating through all JSON files = one epoch
    
    drop_overlap = random.random() < drop_overlap_probability
    
    if not drop_overlap:
        # Use precomputed overlap_labels for FOV overlap-based selection
        overlap_labels_dir = os.path.join(dataset_base_path, 'overlap_labels')
        if os.path.exists(overlap_labels_dir):
            # Load overlapping_frames from JSON file: overlap_labels/{video_name}/{current_frame_idx}.json
            # The JSON structure: {"frame_index": "...", "overlapping_frames": ["2796", "2797", ...]}
            overlapping_indices = load_overlap_frames(
                overlap_labels_dir,
                video_name,
                current_frame_idx
            )
            # Filter: exclude first_frame_idx, but allow frames with FOV overlap (may include future frames)
            # FOV overlap indicates visual similarity, which is valuable for context memory
            # even if the overlapping frame is from the future
            overlapping_indices = [idx for idx in overlapping_indices
                                  if idx != first_frame_idx]

            # If we have overlap frames, randomly sample top_k from them
            # Note: overlap_labels are precomputed FOV overlap results, containing potentially
            # very long lists of non-contiguous frame indices (e.g., [2796, 3839, 4183, ...]).
            # We cannot use all frames due to memory constraints, so we use random uniform sampling.
            # Sampling all JSON files once = one epoch
            if overlapping_indices:
                # Experiment 1_4_2: Use RT relative conversion for better geometric consistency
                # Aligned with CAM paper [2506.03141] - use relative camera poses for FOV overlap
                if use_rt_relative:
                    ref_pose = poses_dict.get(str(current_frame_idx))
                    
                    if ref_pose is not None:
                        # Convert reference pose to RT format
                        ref_rt = pose_to_rt(ref_pose)
                        
                        if ref_rt is not None:
                            # Score candidates using relative poses
                            candidate_scores = []
                            for candidate_idx in overlapping_indices:
                                candidate_pose = poses_dict.get(str(candidate_idx))
                                if candidate_pose is None:
                                    continue
                                
                                # Convert to RT and compute relative pose
                                candidate_rt = pose_to_rt(candidate_pose)
                                if candidate_rt is not None:
                                    relative_rt_list = convert_rt_to_relative([candidate_rt], ref_rt)
                                    if relative_rt_list:
                                        relative_pose = rt_to_pose(relative_rt_list[0])
                                        if relative_pose is not None:
                                            # Compute FOV overlap using relative poses
                                            ref_relative_pose = {'position': [0, 0, 0], 'rotation': [0, 0, 0]}
                                            score = compute_fov_overlap_3d(
                                                ref_relative_pose,
                                                relative_pose,
                                                fov_degrees=52.67,
                                                max_distance=500.0
                                            )
                                            candidate_scores.append((candidate_idx, score))
                            
                            # Sort by score and select top_k
                            if candidate_scores:
                                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                                sampled_overlap_indices = [idx for idx, _ in candidate_scores[:top_k]]
                            else:
                                # Fallback to random sampling if RT conversion fails
                                num_overlap_frames = max(1, top_k)
                                sampled_overlap_indices = random.sample(
                                    overlapping_indices, 
                                    min(len(overlapping_indices), num_overlap_frames)
                                )
                        else:
                            # Fallback to random sampling if RT conversion fails
                            num_overlap_frames = max(1, top_k)
                            sampled_overlap_indices = random.sample(
                                overlapping_indices, 
                                min(len(overlapping_indices), num_overlap_frames)
                            )
                    else:
                        # Fallback to random sampling if reference pose not found
                        num_overlap_frames = max(1, top_k)
                        sampled_overlap_indices = random.sample(
                            overlapping_indices, 
                            min(len(overlapping_indices), num_overlap_frames)
                        )
                else:
                    # Original strategy: Random Uniform Sampling (recommended for robustness)
                    # This allows the model to learn from diverse temporal spans of memory.
                    # The precomputed overlapping_frames list may contain hundreds or thousands of indices,
                    # but we only sample top_k (e.g., 4) frames due to memory constraints.
                    num_overlap_frames = max(1, top_k)
                    
                    # Random uniform sampling from the precomputed overlapping_frames list
                    # This treats memory frames as an unordered image collection
                    # IMPORTANT: Do NOT sort sampled_indices - they are unordered snapshots, not a temporal sequence
                    # Positional encoding should NOT use original absolute time positions for memory frames
                    sampled_overlap_indices = random.sample(
                        overlapping_indices, 
                        min(len(overlapping_indices), num_overlap_frames)
                    )
                
                # Load overlap frames (memory frames) - long-term memory
                # Experiment 1_4_2: Use first_frame (start_frame) as reference for ALL context RTs
                # Target actions use ref=first_frame; context must use same ref for trajectory alignment
                ref_pose_for_rt = first_frame_pose_rt if use_rt_relative else None
                
                if os.path.exists(frames_dir):
                    to_load = [(idx, os.path.join(frames_dir, f"{idx:04d}.png")) for idx in sampled_overlap_indices[:top_k]]
                    frames_loaded = {}
                    with ThreadPoolExecutor(max_workers=max(1, min(5, len(to_load)))) as ex:
                        futures = {ex.submit(_load_frame_png, fp): idx for idx, fp in to_load}
                        for fut in as_completed(futures):
                            frame_idx = futures[fut]
                            frame = fut.result()
                            if frame is not None:
                                frames_loaded[frame_idx] = frame
                    for frame_idx in sampled_overlap_indices[:top_k]:
                        if frame_idx not in frames_loaded:
                            continue
                        frame = frames_loaded[frame_idx]
                        context_frames.append(frame)
                        context_indices.append(frame_idx)
                        pose = poses_dict.get(str(frame_idx))
                        if pose is not None:
                            rt_pose = pose_to_rt(pose)
                            if rt_pose is not None:
                                if use_rt_relative and ref_pose_for_rt is not None:
                                    relative_rt_list = convert_rt_to_relative([rt_pose], ref_pose_for_rt)
                                    context_actions.append(relative_rt_list[0] if relative_rt_list else rt_pose)
                                else:
                                    context_actions.append(rt_pose)
                            else:
                                context_actions.append([0.0] * 12)
                        else:
                            context_actions.append([0.0] * 12)
                
                source = "overlap_labels_random"
            else:
                # No overlap frames found, will use fallback below
                source = "first_frame_only"
        else:
            # No overlap_labels directory, will use fallback below
            source = "first_frame_only"
    else:
        # 10% probability: drop overlap frames, only use First Frame
        # This simulates video generation starting stage (no historical memory)
        source = "first_frame_only_dropped"
    # Step 3: Fallback if we don't have enough overlap frames (and not dropped)
    # Only fill if we haven't dropped overlap frames and need more frames
    if source not in ["first_frame_only_dropped"] and len(context_frames) < top_k + 1:
        # Try random fallback: use random previous frames before current_frame_idx
        max_prev_frame = max(1, current_frame_idx - 1)
        if max_prev_frame > 1:
            # Exclude first_frame_idx from random sampling
            candidate_indices = [idx for idx in range(max_prev_frame) 
                                if idx != first_frame_idx and idx < current_frame_idx]
            if candidate_indices:
                num_needed = top_k + 1 - len(context_frames)
                num_random_frames = min(len(candidate_indices), num_needed)
                if num_random_frames > 0:
                    sampled_indices = random.sample(candidate_indices, num_random_frames)
                    
                    # Load random frames
                    if os.path.exists(frames_dir):
                        for frame_idx in sampled_indices:
                            frame_file = os.path.join(frames_dir, f"{frame_idx:04d}.png")
                            if os.path.exists(frame_file):
                                try:
                                    frame = Image.open(frame_file).convert('RGB')
                                    context_frames.append(frame)
                                    context_indices.append(frame_idx)
                                    pose = poses_dict.get(str(frame_idx))
                                    if pose is not None:
                                        rt_pose = pose_to_rt(pose)
                                        if rt_pose is not None:
                                            # Convert to relative RT if enabled
                                            if use_rt_relative and first_frame_pose_rt is not None:
                                                relative_rt_list = convert_rt_to_relative([rt_pose], first_frame_pose_rt)
                                                context_actions.append(relative_rt_list[0] if relative_rt_list else rt_pose)
                                            else:
                                                context_actions.append(rt_pose)
                                        else:
                                            context_actions.append([0.0] * 12)
                                    else:
                                        context_actions.append([0.0] * 12)
                                except Exception as e:
                                    pass  # Frame loading failed, skip
                    
                    if source == "first_frame_only":
                        source = "random_fallback"

    # Step 4: Final fallback - use additional frames from current segment if needed
    # Target is top_k + 1 frames (1 First Frame + top_k Overlap/Random Frames)
    target_total_frames = top_k + 1
    if len(context_frames) < target_total_frames and len(video_frames) > 0:
        num_needed = target_total_frames - len(context_frames)
        # Use additional frames from current segment (after first frame)
        segment_start_idx = 1 if len(context_frames) > 0 else 0
        for i in range(segment_start_idx, min(segment_start_idx + num_needed, len(video_frames))):
            frame_idx_seg = start_frame + i
            if isinstance(video_frames[i], Image.Image):
                context_frames.append(video_frames[i])
                context_indices.append(frame_idx_seg)
            elif isinstance(video_frames[i], str):
                # If it's a path, load it
                try:
                    frame = Image.open(video_frames[i]).convert('RGB')
                    context_frames.append(frame)
                    context_indices.append(frame_idx_seg)
                except:
                    pass
            
            pose = poses_dict.get(str(frame_idx_seg))
            if pose is not None:
                rt_pose = pose_to_rt(pose)
                if rt_pose is not None:
                    # Convert to relative RT if enabled
                    if use_rt_relative and first_frame_pose_rt is not None:
                        relative_rt_list = convert_rt_to_relative([rt_pose], first_frame_pose_rt)
                        context_actions.append(relative_rt_list[0] if relative_rt_list else rt_pose)
                    else:
                        context_actions.append(rt_pose)
                else:
                    context_actions.append([0.0] * 12)
            else:
                context_actions.append([0.0] * 12)
        
        # Update source if we used segment frames
        if len(context_frames) >= num_needed and source in ["first_frame_only", "none"]:
            source = "segment_fallback"

    # Step 5: Ensure we return exactly top_k + 1 frames
    # If we have some frames but not enough, pad by repeating the last frame
    if len(context_frames) < target_total_frames:
        if context_frames:
            last_frame = context_frames[-1]
            last_idx = context_indices[-1] if context_indices else first_frame_idx
            last_action = context_actions[-1] if context_actions else [0.0] * 12
            while len(context_frames) < target_total_frames:
                context_frames.append(last_frame)
                context_indices.append(last_idx)
                context_actions.append(last_action)  # Always pad with pose data

    # Limit to top_k + 1 (in case we have more)
    context_frames = context_frames[:target_total_frames]
    if context_indices:
        context_indices = context_indices[:target_total_frames]
    if context_actions:
        context_actions = context_actions[:target_total_frames]
    
    # Ensure context_actions length matches context_frames (pad with zeros if needed)
    # Always ensure context_actions are provided (not just when use_rt_relative=True)
    while len(context_actions) < len(context_frames):
        context_actions.append([0.0] * 12)

    return context_frames, context_actions, context_indices, current_frame_idx, video_name, source


def retrieve_fov_context_frames(
    data: Dict,
    dataset_base_path: str,
    fov_retriever=None,
    top_k: int = 4,  # Number of overlap frames to retrieve. First Frame will be added automatically.
    use_precomputed_overlaps: bool = True,
    use_rt_relative: bool = False,  # Experiment 1_4_2: Use RT relative conversion (aligned with CAM paper)
    strict_overlap_labels: bool = False,
    allow_realtime_fallback: bool = True,
    allow_segment_fallback: bool = True,
    drop_overlap_probability: float = 0.1,  # 10% probability to drop overlap frames (paper strategy)
):
    """
    Backward-compatible wrapper.
    We use FOV overlap scoring to select top-k overlap frames (as per CAM paper).
    
    According to CAM paper [2506.03141]:
    - First Frame (current segment's first frame) is always included as immediate condition
    - Overlap Frames are retrieved as long-term memory
    - With 10% probability, drop overlap frames to simulate starting stage
    
    Experiment 1_4_2: Uses RT relative conversion for better geometric consistency.
    """
    context_frames, context_actions, context_indices, cur_idx, video_name, source = retrieve_simple_context_frames(
        data=data,
        dataset_base_path=dataset_base_path,
        top_k=top_k,  # top_k is number of overlap frames (4), First Frame will be added automatically (total: 5)
        use_rt_relative=use_rt_relative,  # Experiment 1_4_2: RT relative conversion
        drop_overlap_probability=drop_overlap_probability,  # 10% probability to drop overlap frames
    )
    # Check if we have top_k + 1 frames (1 First Frame + top_k Overlap Frames)
    target_total_frames = top_k + 1
    if strict_overlap_labels and len(context_frames) < target_total_frames:
        return [], [], [], cur_idx, video_name, "overlap_labels_insufficient"
    return context_frames, context_actions, context_indices, cur_idx, video_name, source


def save_sampling_jsonl(
    output_path: str,
    video_name: str,
    frame_index: int,
    context_indices: List[int],
    prompt: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    source: Optional[str] = None,
    append: bool = True,
) -> None:
    """
    Save context sampling result to JSONL file for eval consistency.
    
    Format:
    {
        "video_name": "AncientTempleEnv_0",
        "frame_index": 0,  # First Frame (short-term memory)
        "context_indices": [0, 2796, 3839, 4183, 6339],  # First Frame + 4 Overlap Frames
        "prompt": "...",  # Optional
        "start_frame": 0,  # Optional: GT segment start
        "end_frame": 80,  # Optional: GT segment end
        "source": "overlap_labels_random"  # Optional: sampling source
    }
    
    Args:
        output_path: Path to JSONL file
        video_name: Video name
        frame_index: First Frame index (from JSON file)
        context_indices: List of context frame indices [first_frame, overlap1, overlap2, ...]
        prompt: Optional prompt text
        start_frame: Optional GT segment start frame
        end_frame: Optional GT segment end frame
        source: Optional sampling source
        append: Whether to append to existing file (default: True)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    item = {
        "video_name": video_name,
        "frame_index": frame_index,
        "context_indices": context_indices,
    }
    
    if prompt is not None:
        item["prompt"] = prompt
    if start_frame is not None:
        item["start_frame"] = start_frame
    if end_frame is not None:
        item["end_frame"] = end_frame
    if source is not None:
        item["source"] = source
    
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_sampling_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load context sampling results from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of sampling items, each containing:
        {
            "video_name": str,
            "frame_index": int,
            "context_indices": List[int],
            "prompt": Optional[str],
            "start_frame": Optional[int],
            "end_frame": Optional[int],
            "source": Optional[str]
        }
    """
    if not os.path.exists(jsonl_path):
        return []
    
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                items.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSONL line: {e}")
                continue
    
    return items


def setup_fov_retriever_for_training(
    dataset_base_path: str,
    enable_fov_retrieval: bool = True,
) -> Optional[object]:
    """
    Setup FOV retriever for training (simplified version).
    Context frames are selected by FOV overlap scoring from precomputed `overlap_labels`.
    """
    return None
