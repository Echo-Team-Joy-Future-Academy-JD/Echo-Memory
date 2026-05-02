"""
Simplified Context Retrieval - Direct Sampling from Overlap Labels
Removes FOV computation complexity, directly samples from precomputed overlaps
"""

import os
import random
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


def simple_retrieve_context_frames(
    data: Dict,
    dataset_base_path: str,
    top_k: int = 5,
    use_precomputed_overlaps: bool = True,
    randomize_context: bool = False,
    context_window_size: int = 100  # Look at frames within this window
) -> Tuple[List[Image.Image], List, List[int], int, str, str]:
    """
    Simplified context frame retrieval - directly sample from overlap labels.
    
    Args:
        data: Training data dict
        dataset_base_path: Base path to dataset
        top_k: Number of context frames to retrieve
        use_precomputed_overlaps: Use overlap_labels if available
        randomize_context: Randomly sample instead of taking top-k
        context_window_size: Maximum frame window to consider
        
    Returns:
        (context_frames, context_actions, context_indices, current_frame_idx, video_name, source)
    """
    context_frames = []
    context_actions = []
    context_indices = []
    source = "none"
    
    # Extract video information from data
    video_path = data.get('video_path', '')  # e.g., "frames/FeudalJapan_9"
    video_name = None
    start_frame = data.get('start_frame')
    end_frame = data.get('end_frame')
    video_frames = data.get('video_frames', [])
    
    # Extract video name
    if video_path:
        parts = video_path.split("/")
        if len(parts) >= 2 and parts[-1]:  # "frames/FeudalJapan_9" -> "FeudalJapan_9"
            video_name = parts[-1]
        elif parts and parts[0]:
            video_name = parts[0]
    
    if video_name is None:
        return [], [], [], 0, "", "no_video_name"
    
    # Determine current frame (middle of segment)
    if start_frame is not None and end_frame is not None:
        current_frame_idx = (start_frame + end_frame) // 2
    else:
        current_frame_idx = 0
    
    # Use overlap_labels if available (simplest approach)
    if use_precomputed_overlaps:
        overlap_labels_dir = os.path.join(dataset_base_path, 'overlap_labels')
        if os.path.exists(overlap_labels_dir):
            overlapping_indices = load_overlap_frames_safe(
                overlap_labels_dir,
                video_name,
                current_frame_idx
            )
            
            if overlapping_indices:
                # Filter for temporal coherence: only frames before current
                overlapping_indices = [idx for idx in overlapping_indices if idx < current_frame_idx]
                
                if randomize_context and len(overlapping_indices) > top_k:
                    # Randomly sample top_k frames
                    overlapping_indices = random.sample(overlapping_indices, top_k)
                else:
                    # Take top-k frames (already sorted by overlap score)
                    overlapping_indices = overlapping_indices[:top_k]
                
                context_indices = overlapping_indices
                
                # Load frames
                frames_dir = os.path.join(dataset_base_path, 'frames', video_name)
                if os.path.exists(frames_dir):
                    for frame_idx in overlapping_indices:
                        frame_file = os.path.join(frames_dir, f"{frame_idx:04d}.png")
                        if os.path.exists(frame_file):
                            try:
                                frame = Image.open(frame_file).convert('RGB')
                                context_frames.append(frame)
                            except Exception as e:
                                logger.debug(f"Failed to load frame {frame_file}: {e}")
                
                if context_frames:
                    source = "overlap_labels_simple"
                    logger.debug(f"Retrieved {len(context_frames)} context frames from overlap_labels for {video_name}")
                
                return context_frames, context_actions, context_indices, current_frame_idx, video_name, source
    
    # Fallback: Simple temporal sampling without FOV computation
    # Just sample from previous frames in temporal order
    if start_frame is not None:
        # Sample from recent frames (temporal locality)
        start_context = max(0, start_frame - context_window_size)
        candidate_frames = list(range(start_context, start_frame))
        
        if candidate_frames:
            if randomize_context:
                # Random sampling
                if len(candidate_frames) > top_k:
                    selected_indices = sorted(random.sample(candidate_frames, top_k))
                else:
                    selected_indices = candidate_frames
            else:
                # Take most recent frames
                selected_indices = candidate_frames[-top_k:]
            
            context_indices = selected_indices
            
            # Load frames
            frames_dir = os.path.join(dataset_base_path, 'frames', video_name)
            if os.path.exists(frames_dir):
                for frame_idx in selected_indices:
                    frame_file = os.path.join(frames_dir, f"{frame_idx:04d}.png")
                    if os.path.exists(frame_file):
                        try:
                            frame = Image.open(frame_file).convert('RGB')
                            context_frames.append(frame)
                        except Exception as e:
                            logger.debug(f"Failed to load frame {frame_file}: {e}")
            
            if context_frames:
                source = "temporal_sampling"
                logger.debug(f"Retrieved {len(context_frames)} context frames via temporal sampling for {video_name}")
    
    # Final fallback: use first frames of segment
    if not context_frames and len(video_frames) >= top_k:
        for i in range(top_k):
            if i < len(video_frames):
                if isinstance(video_frames[i], Image.Image):
                    context_frames.append(video_frames[i])
                elif isinstance(video_frames[i], str):
                    try:
                        frame = Image.open(video_frames[i]).convert('RGB')
                        context_frames.append(frame)
                    except:
                        pass
        if context_frames:
            source = "segment_fallback"
            context_indices = list(range(len(context_frames)))
    
    return context_frames, context_actions, context_indices, current_frame_idx, video_name, source


def load_overlap_frames_safe(overlap_labels_dir: str, video_name: str, current_frame_idx: int) -> List[int]:
    """
    Safely load overlap frames with error handling for distributed training.
    """
    overlap_file = os.path.join(overlap_labels_dir, video_name, f"{current_frame_idx}.json")
    
    if not os.path.exists(overlap_file):
        return []
    
    try:
        with open(overlap_file, 'r') as f:
            data = json.load(f)
            overlapping_frames = data.get('overlapping_frames', [])
            # Convert string indices to integers
            return [int(f) for f in overlapping_frames if f.isdigit() or isinstance(f, int)]
    except Exception as e:
        logger.debug(f"Error loading overlap labels from {overlap_file}: {e}")
        return []