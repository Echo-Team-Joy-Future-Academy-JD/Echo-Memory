"""
FOV (Field of View) Overlap-based Memory Retrieval Module
Implements geometric retrieval based on camera pose overlap for Context-as-Memory

Aligned with the Context-as-Memory paper [2506.03141].
"""

import os
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import torch
from PIL import Image

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * np.pi / 180


def compute_rotation_list_z_only(x: float, y: float, z: float, yaw_degrees: float) -> List[float]:
    """
    Paper-aligned: rotation only around Z-axis (yaw). 4 parameters.
    R = R_z(yaw). No roll/pitch.

    Returns:
        [t_x, t_y, t_z, R_11, R_12, ..., R_33] (12 elements)
    """
    yaw_rad = degrees_to_radians(yaw_degrees)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R_z = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    return [float(x), float(y), float(z)] + R_z.flatten().tolist()


def compute_rotation_list_yaw_pitch(
    x: float, y: float, z: float, yaw_degrees: float, pitch_degrees: float
) -> List[float]:
    """
    Five-parameter rotation: R = R_z(pitch) @ R_y(yaw). Same convention as external pipelines.
    - R_y: rotation around Y (yaw in degrees).
    - R_z: rotation around Z (pitch in degrees).
    Returns [t_x, t_y, t_z, R_11, ..., R_33] (12 elements), encoder-compatible.
    """
    yaw_rad = degrees_to_radians(yaw_degrees)
    pitch_rad = degrees_to_radians(pitch_degrees)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    R_y = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float64)
    R_z = np.array([
        [cp, -sp, 0],
        [sp, cp, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    R = R_z @ R_y
    return [float(x), float(y), float(z)] + R.flatten().tolist()


def compute_rotation_list(params: List[float]) -> List[float]:
    """
    Compute rotation matrix from camera parameters and flatten to list.
    - 4 params [x, y, z, yaw]: Z-only, R = R_z(yaw) (paper default).
    - 5 params [x, y, z, yaw, pitch]: R = R_z(pitch) @ R_y(yaw).
    Returns [t_x, t_y, t_z, R_11, ..., R_33] (12 elements).
    """
    if len(params) >= 5:
        x, y, z = float(params[0]), float(params[1]), float(params[2])
        yaw = float(params[3])
        pitch = float(params[4])
        return compute_rotation_list_yaw_pitch(x, y, z, yaw, pitch)
    if len(params) >= 4:
        x, y, z = float(params[0]), float(params[1]), float(params[2])
        yaw = float(params[3])
        return compute_rotation_list_z_only(x, y, z, yaw)
    x = y = z = yaw = 0.0
    return compute_rotation_list_z_only(x, y, z, yaw)


def yaw_deg_from_rt(rt: List[float]) -> float:
    """Extract yaw (degrees) from 12-dim RT. Z-only: yaw = atan2(R_21, R_11)."""
    if rt is None or len(rt) < 12:
        return 0.0
    R = np.array(rt[3:12]).reshape(3, 3)
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def flip_yaw_rt(rt: List[float]) -> List[float]:
    """Return RT with yaw negated (CW<->CCW). rt is 12 elements [t_x,t_y,t_z, R_11..R_33]."""
    if rt is None or len(rt) < 12:
        return list(rt) if rt else []
    tx, ty, tz = rt[0], rt[1], rt[2]
    yaw_deg = yaw_deg_from_rt(rt)
    return compute_rotation_list_z_only(tx, ty, tz, -yaw_deg)


def flip_yaw_rt_list(rt_list: List[List[float]]) -> List[List[float]]:
    """Flip yaw for each RT in list (data aug for direction sensitivity)."""
    return [flip_yaw_rt(rt) for rt in rt_list]


def convert_rt_to_relative(rt_list_all: List[List[float]], ref_rt: List[float]) -> List[List[float]]:
    """
    Convert RT (rotation-translation) poses to relative coordinates.

    This aligns with the Context-as-Memory paper's use of relative camera poses
    for better geometric consistency in context frame retrieval.

    Args:
        rt_list_all: List of RT poses, each is [t_x, t_y, t_z, R_11, R_12, ..., R_33] (12 elements)
        ref_rt: Reference RT pose [t_x, t_y, t_z, R_11, R_12, ..., R_33] (12 elements)

    Returns:
        new_rt_list: List of relative RT poses in the same format
    """
    def parse_rt(rt: List[float]) -> tuple:
        """Parse RT list into rotation matrix R and translation vector t."""
        t = np.array(rt[:3]).reshape((3, 1))
        R = np.array(rt[3:]).reshape((3, 3))
        return R, t

    R_ref, T_ref = parse_rt(ref_rt)
    R_ref_inv = R_ref.T
    T_ref_inv = -R_ref_inv @ T_ref

    new_rt_list = []

    for rt in rt_list_all:
        R_i, T_i = parse_rt(rt)

        # Convert to relative coordinates
        R_new = R_ref_inv @ R_i
        T_new = R_ref_inv @ T_i + T_ref_inv

        # Flatten back to list format: [t_x, t_y, t_z, R_11, R_12, ..., R_33]
        rt_new = T_new.flatten().tolist() + R_new.flatten().tolist()
        new_rt_list.append(rt_new)

    return new_rt_list


def pose_to_rt(pose: Dict, constrain_to_xy: bool = True) -> Optional[List[float]]:
    """
    Convert camera pose dict to RT format. Paper-aligned by default.

    - 2D plane (constrain_to_xy=True): use position x, y only; z=0.
    - Rotation: Z-axis only; use rotation[2] as yaw (degrees). Roll/pitch ignored.

    Args:
        pose: Dict with 'position' [x, y, z] and 'rotation' [roll, pitch, yaw] in degrees
        constrain_to_xy: If True (default), set z=0 for strict XY-plane displacement (paper).

    Returns:
        RT list: [t_x, t_y, t_z, R_11, R_12, ..., R_33] (12 elements) or None if invalid
    """
    if pose is None:
        return None

    pos = pose.get('position', [0, 0, 0])
    rot = pose.get('rotation', [0, 0, 0])

    if len(pos) < 2:
        return None

    x = float(pos[0])
    y = float(pos[1])
    z = 0.0 if constrain_to_xy else (float(pos[2]) if len(pos) >= 3 else 0.0)
    # Paper: rotation only around Z-axis -> use yaw (index 2) only
    yaw = float(rot[2]) if len(rot) > 2 else 0.0

    return compute_rotation_list([x, y, z, yaw])


def rt_to_pose(rt: List[float]) -> Optional[Dict]:
    """
    Convert RT format back to pose dict.

    Args:
        rt: RT list [t_x, t_y, t_z, R_11, R_12, ..., R_33] (12 elements)

    Returns:
        pose dict with 'position' and 'rotation' or None if invalid
    """
    if rt is None or len(rt) < 12:
        return None

    t = np.array(rt[:3])
    R = np.array(rt[3:]).reshape((3, 3))

    # Extract Euler angles from rotation matrix
    # Using ZYX convention (yaw-pitch-roll)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return {
        'position': t.tolist(),
        'rotation': [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]
    }



def _parse_poses_dict(data: dict) -> dict:
    """Extract poses dict from JSON data (CineCameraActor or flat dict)."""
    if 'CineCameraActor' in data:
        return data['CineCameraActor']
    return data if isinstance(data, dict) else {}


def load_poses_dict(json_file: str) -> dict:
    """
    Load full camera poses dict from JSON file (one read for all frames).

    Args:
        json_file: Path to camera pose JSON file

    Returns:
        Dict mapping frame_idx (str) -> pose dict, or {} if failed
    """
    if not os.path.exists(json_file):
        return {}
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return _parse_poses_dict(data)
    except Exception as e:
        print(f"Error loading poses from {json_file}: {e}")
        return {}


def load_camera_pose(json_file: str, frame_idx: int) -> Optional[Dict]:
    """
    Load camera pose for a specific frame from JSON file.

    Args:
        json_file: Path to camera pose JSON file
        frame_idx: Frame index

    Returns:
        Dict with 'position' and 'rotation' keys, or None if not found
    """
    poses = load_poses_dict(json_file)
    frame_key = str(frame_idx)
    return poses.get(frame_key)


def load_camera_poses_batch(json_file: str, frame_indices: List[int]) -> List[Optional[Dict]]:
    """
    Load camera poses for multiple frames in one JSON read.

    Args:
        json_file: Path to camera pose JSON file
        frame_indices: List of frame indices

    Returns:
        List of pose dicts (or None) in same order as frame_indices
    """
    poses = load_poses_dict(json_file)
    return [poses.get(str(fi)) for fi in frame_indices]


def load_overlap_frames(overlap_labels_dir: str, video_name: str, frame_idx: int) -> List[int]:
    """
    Load overlapping frame indices for a given frame from overlap_labels.

    Args:
        overlap_labels_dir: Base directory for overlap labels
        video_name: Name of the video
        frame_idx: Current frame index

    Returns:
        List of overlapping frame indices
    """
    overlap_file = os.path.join(overlap_labels_dir, video_name, f"{frame_idx}.json")
    if not os.path.exists(overlap_file):
        return []

    try:
        # Add distributed training safety - timeout protection
        import signal
        import torch.distributed as dist

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Timeout loading overlap file: {overlap_file}")

        # Set 10-second timeout for file operations in distributed training
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)

        try:
            with open(overlap_file, 'r') as f:
                data = json.load(f)
                overlapping_frames = data.get('overlapping_frames', [])
                # Convert string indices to integers
                result = [int(f) for f in overlapping_frames if f.isdigit() or isinstance(f, int)]
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    signal.alarm(0)  # Cancel alarm
                return result
        finally:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                signal.alarm(0)  # Always cancel alarm

    except TimeoutError as e:
        print(f"Warning: Timeout loading overlap file {overlap_file} (distributed training): {e}")
        return []
    except Exception as e:
        print(f"Error loading overlap labels from {overlap_file}: {e}")
        return []


def compute_fov_overlap_3d(
    pose1: Dict,
    pose2: Dict,
    fov_degrees: float = 52.67,
    max_distance: float = 500.0  # Increased default from 50.0 to 500.0 for large scenes
) -> float:
    """
    Compute FOV overlap score between two camera poses using 3D geometry.

    This implementation uses full 6-DoF camera poses (position + rotation)
    to compute more accurate FOV overlap, as described in Context-as-Memory.

    Args:
        pose1: First camera pose with 'position' [x, y, z] and 'rotation' [roll, pitch, yaw] in degrees
        pose2: Second camera pose with 'position' [x, y, z] and 'rotation' [roll, pitch, yaw] in degrees
        fov_degrees: Field of view in degrees (default: 52.67 from paper)
        max_distance: Maximum distance to consider (meters)

    Returns:
        Overlap score between 0 and 1
    """
    if pose1 is None or pose2 is None:
        return 0.0

    pos1 = np.array(pose1.get('position', [0, 0, 0]), dtype=np.float32)
    pos2 = np.array(pose2.get('position', [0, 0, 0]), dtype=np.float32)

    # Distance between cameras
    distance = np.linalg.norm(pos2 - pos1)
    # Instead of returning 0.0 for distances > max_distance, we use a soft threshold
    # that still gives some score based on direction similarity even for far cameras
    distance_exceeds_max = distance > max_distance

    if distance < 1e-6:
        # Same position - high overlap
        return 1.0

    # Extract rotation angles (assuming [roll, pitch, yaw] or [x, y, z] rotation in degrees)
    rot1 = pose1.get('rotation', [0, 0, 0])
    rot2 = pose2.get('rotation', [0, 0, 0])

    # Convert to numpy array
    rot1 = np.array(rot1, dtype=np.float32)
    rot2 = np.array(rot2, dtype=np.float32)

    # Compute rotation matrices from Euler angles
    # Note: The rotation order may vary by dataset. Common conventions:
    # - ZYX (yaw-pitch-roll): R = R_z(yaw) * R_y(pitch) * R_x(roll)
    # - XYZ (roll-pitch-yaw): R = R_x(roll) * R_y(pitch) * R_z(yaw)
    # Based on Context-as-Memory dataset, rotation[2] is yaw (rotation around Z-axis)
    # We'll use ZYX convention: yaw (Z), pitch (Y), roll (X)

    def euler_to_rotation_matrix(euler_angles):
        """Convert Euler angles [roll, pitch, yaw] in degrees to rotation matrix (ZYX order)"""
        roll, pitch, yaw = np.radians(euler_angles)

        # Rotation around X-axis (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Rotation around Y-axis (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Rotation around Z-axis (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # ZYX order: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R

    # Handle different rotation formats
    if len(rot1) >= 3:
        # Euler angles [roll, pitch, yaw] or [x, y, z]
        R1 = euler_to_rotation_matrix([rot1[0], rot1[1], rot1[2]])
    elif len(rot1) == 9:
        # Rotation matrix flattened (3x3 = 9 elements)
        R1 = rot1.reshape(3, 3)
    else:
        # Fallback: only yaw
        R1 = euler_to_rotation_matrix([0, 0, rot1[2] if len(rot1) > 2 else 0])

    if len(rot2) >= 3:
        R2 = euler_to_rotation_matrix([rot2[0], rot2[1], rot2[2]])
    elif len(rot2) == 9:
        R2 = rot2.reshape(3, 3)
    else:
        R2 = euler_to_rotation_matrix([0, 0, rot2[2] if len(rot2) > 2 else 0])

    # Camera forward vector (typically Z-axis in camera coordinate system)
    # In OpenCV/OpenGL convention, forward is usually -Z or +Z
    # Based on Context-as-Memory dataset, we assume forward is +Z (third column)
    forward1 = R1[:, 2]  # Third column of rotation matrix
    forward2 = R2[:, 2]

    # Vector from camera1 to camera2
    vec_1_to_2 = pos2 - pos1
    vec_1_to_2_norm = np.linalg.norm(vec_1_to_2)
    vec_1_to_2_unit = vec_1_to_2 / (vec_1_to_2_norm + 1e-6)

    # FOV half-angle threshold (cosine of half FOV)
    fov_rad = np.radians(fov_degrees)
    fov_half_cos = np.cos(fov_rad / 2)

    # Check if camera1 can see camera2's position (within FOV)
    # cos(angle) = dot(forward, vec_to_target)
    # angle < fov/2  =>  cos(angle) > cos(fov/2)
    dot1 = np.dot(forward1, vec_1_to_2_unit)
    can_1_see_2 = dot1 > fov_half_cos

    # Check if camera2 can see camera1's position (within FOV)
    dot2 = np.dot(forward2, -vec_1_to_2_unit)  # Negative because looking back
    can_2_see_1 = dot2 > fov_half_cos

    # Compute overlap score based on mutual visibility and distance
    # Normalize distance for scoring (use max_distance as reference, but don't hard-cut)
    normalized_distance = min(1.0, distance / max_distance) if max_distance > 0 else 1.0

    if can_1_see_2 and can_2_see_1:
        # Both cameras can see each other - high overlap
        # Score decreases with distance, but never goes to 0
        distance_factor = 1.0 - normalized_distance * 0.5
        overlap = 0.8 + 0.2 * distance_factor
    elif can_1_see_2 or can_2_see_1:
        # One camera can see the other - medium overlap
        distance_factor = 1.0 - normalized_distance * 0.6
        overlap = 0.4 + 0.3 * distance_factor
    else:
        # Check if cameras are looking in similar directions (even if not directly at each other)
        # This handles the case where both cameras see the same scene from different angles
        forward_similarity = np.dot(forward1, forward2)
        if forward_similarity > 0.7:  # Cameras looking in similar directions
            # Even for far cameras, if they're looking in similar directions, there's some overlap
            distance_factor = 1.0 - normalized_distance * 0.7
            overlap = 0.2 + 0.3 * distance_factor
        elif forward_similarity > 0.0:
            # Cameras looking in somewhat similar directions
            distance_factor = 1.0 - normalized_distance * 0.8
            overlap = 0.05 + 0.15 * distance_factor * forward_similarity
        else:
            # Cameras looking away from each other - very low overlap
            # But still give some score based on distance (closer = slightly better)
            overlap = max(0.0, 0.01 - normalized_distance * 0.01)

    # Apply distance penalty for cameras exceeding max_distance (soft penalty)
    if distance_exceeds_max:
        # Reduce score by distance penalty, but don't make it zero
        distance_penalty = min(0.5, (distance - max_distance) / max_distance * 0.3)
        overlap = overlap * (1.0 - distance_penalty)

    return np.clip(overlap, 0.0, 1.0)


# Keep the simple version as fallback
def compute_fov_overlap_simple(
    pose1: Dict,
    pose2: Dict,
    fov_degrees: float = 52.67,
    max_distance: float = 50.0
) -> float:
    """
    Simplified FOV overlap computation (fallback).
    Use compute_fov_overlap_3d for more accurate results.
    """
    return compute_fov_overlap_3d(pose1, pose2, fov_degrees, max_distance)


class FOVMemoryRetriever:
    """
    FOV-based Memory Retriever for Context-as-Memory.

    Retrieves relevant historical frames based on FOV overlap with current frame.
    """

    def __init__(
        self,
        dataset_base_path: str,
        fov_degrees: float = 52.67,
        max_distance: float = 50.0,
        use_precomputed_overlaps: bool = True
    ):
        """
        Initialize FOV Memory Retriever.

        Args:
            dataset_base_path: Base path to Context-as-Memory dataset
            fov_degrees: Field of view in degrees (default from paper: 52.67)
            max_distance: Maximum distance to consider for overlap (meters)
            use_precomputed_overlaps: Whether to use precomputed overlap_labels if available
        """
        self.dataset_base_path = dataset_base_path
        self.fov_degrees = fov_degrees
        self.max_distance = max_distance
        self.use_precomputed_overlaps = use_precomputed_overlaps

        self.jsons_dir = os.path.join(dataset_base_path, 'jsons')
        self.overlap_labels_dir = os.path.join(dataset_base_path, 'overlap_labels')

        # Cache for loaded poses
        self._pose_cache: Dict[str, Dict] = {}

    def retrieve_frames(
        self,
        video_name: str,
        current_frame_idx: int,
        candidate_frame_indices: List[int],
        top_k: int = 5,
        include_last_frame: bool = True,
        use_relative_poses: bool = False  # Experiment 1_4_2: use RT relative conversion
    ) -> List[int]:
        """
        Retrieve top-k most relevant frames based on FOV overlap.

        According to Context-as-Memory, for temporal coherence, we should:
        1. Always include the last frame (current_frame_idx - 1) as short-term memory
        2. Retrieve top-(k-1) frames from history as long-term memory

        Args:
            video_name: Name of the video
            current_frame_idx: Index of current frame to generate
            candidate_frame_indices: List of candidate frame indices to consider
            top_k: Number of frames to retrieve
            include_last_frame: Whether to force include the last frame (default: True, per Context-as-Memory)
            use_relative_poses: Whether to use RT relative conversion (experiment 1_4_2, aligned with paper)

        Returns:
            List of top-k frame indices sorted by relevance (last frame first if included)
        """
        if not candidate_frame_indices:
            return []

        retrieved_frames = []

        # Step 1: Force include last frame for short-term memory (Context-as-Memory requirement)
        if include_last_frame and current_frame_idx > 0:
            last_frame_idx = current_frame_idx - 1
            if last_frame_idx in candidate_frame_indices:
                retrieved_frames.append(last_frame_idx)
                # Remove from candidates to avoid duplication
                candidate_frame_indices = [idx for idx in candidate_frame_indices if idx != last_frame_idx]

        # Calculate how many more frames we need
        remaining_k = top_k - len(retrieved_frames)
        if remaining_k <= 0:
            return retrieved_frames[:top_k]

        # Step 2: Retrieve long-term memory frames using FOV overlap
        # If precomputed overlaps are available, use them
        if self.use_precomputed_overlaps:
            overlap_frames = load_overlap_frames(
                self.overlap_labels_dir,
                video_name,
                current_frame_idx
            )

            # Filter to only include candidate frames (and exclude already included last frame)
            overlap_frames = [f for f in overlap_frames
                            if f in candidate_frame_indices and f not in retrieved_frames]

            if overlap_frames:
                # Take top remaining_k frames
                retrieved_frames.extend(overlap_frames[:remaining_k])
                return retrieved_frames[:top_k]

        # Step 3: Compute FOV overlap using camera poses (if precomputed not available)
        current_pose = self._load_pose(video_name, current_frame_idx)
        if current_pose is None:
            # Fallback: return first k candidates
            retrieved_frames.extend(candidate_frame_indices[:remaining_k])
            return retrieved_frames[:top_k]

        # Experiment 1_4_2: Convert to relative poses if enabled (aligned with Context-as-Memory)
        if use_relative_poses:
            # Convert current pose to RT format
            ref_rt = pose_to_rt(current_pose)
            if ref_rt is None:
                # Fallback to absolute poses if conversion fails
                use_relative_poses = False

        # Compute overlap scores for all candidates
        overlap_scores = []
        for candidate_idx in candidate_frame_indices:
            if candidate_idx in retrieved_frames:
                continue  # Skip already included frames

            candidate_pose = self._load_pose(video_name, candidate_idx)
            if candidate_pose is None:
                continue

            # Experiment 1_4_2: Use relative poses for FOV overlap computation
            if use_relative_poses and ref_rt is not None:
                # Convert candidate pose to RT format
                candidate_rt = pose_to_rt(candidate_pose)
                if candidate_rt is not None:
                    # Convert to relative coordinates
                    relative_rt_list = convert_rt_to_relative([candidate_rt], ref_rt)
                    if relative_rt_list:
                        # Convert back to pose format for FOV overlap computation
                        relative_pose = rt_to_pose(relative_rt_list[0])
                        if relative_pose is not None:
                            # Use relative pose for overlap computation
                            # Reference pose in relative coordinates is identity (origin)
                            ref_relative_pose = {'position': [0, 0, 0], 'rotation': [0, 0, 0]}
                            score = compute_fov_overlap_3d(
                                ref_relative_pose,
                                relative_pose,
                                self.fov_degrees,
                                self.max_distance
                            )
                            overlap_scores.append((candidate_idx, score))
                            continue

            # Fallback: Use absolute poses (original method)
            score = compute_fov_overlap_3d(
                current_pose,
                candidate_pose,
                self.fov_degrees,
                self.max_distance
            )
            overlap_scores.append((candidate_idx, score))

        # Sort by score (descending) and take top remaining_k
        overlap_scores.sort(key=lambda x: x[1], reverse=True)
        retrieved_frames.extend([idx for idx, _ in overlap_scores[:remaining_k]])

        return retrieved_frames[:top_k]

    def _load_pose(self, video_name: str, frame_idx: int) -> Optional[Dict]:
        """Load and cache camera pose."""
        cache_key = f"{video_name}_{frame_idx}"
        if cache_key in self._pose_cache:
            return self._pose_cache[cache_key]

        json_file = os.path.join(self.jsons_dir, f"{video_name}.json")
        pose = load_camera_pose(json_file, frame_idx)

        if pose is not None:
            self._pose_cache[cache_key] = pose

        return pose

    def clear_cache(self):
        """Clear pose cache."""
        self._pose_cache.clear()


def create_fov_retriever(dataset_base_path: str) -> Optional[FOVMemoryRetriever]:
    """
    Create FOV retriever if dataset has camera pose information.

    Args:
        dataset_base_path: Base path to dataset

    Returns:
        FOVMemoryRetriever instance or None if dataset doesn't support it
    """
    jsons_dir = os.path.join(dataset_base_path, 'jsons')
    if not os.path.exists(jsons_dir):
        return None

    return FOVMemoryRetriever(dataset_base_path)


# Context/FOV retrieval integration helpers.
def _load_frame_png(frame_file: str) -> Optional[Image.Image]:
    """Load a single frame from PNG file."""
    if os.path.exists(frame_file):
        try:
            return Image.open(frame_file).convert('RGB')
        except Exception:
            pass
    return None



def retrieve_simple_context_frames(
    data: Dict,
    dataset_base_path: str,
    top_k: int = 4,  # Number of overlap frames to retrieve. First Frame will be added automatically.
    drop_overlap_probability: float = 0.1,  # 10% probability to drop overlap frames (paper strategy)
    use_rt_relative: bool = False,  # Experiment 1_4_2: use RT relative conversion (aligned with Context-as-Memory)
) -> Tuple[List[Image.Image], List, List[int], int, str, str]:
    """
    Retrieve context frames according to the Context-as-Memory paper [2506.03141].

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
                # Aligned with Context-as-Memory [2506.03141] - use relative camera poses for FOV overlap
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
    use_rt_relative: bool = False,  # Experiment 1_4_2: Use RT relative conversion (aligned with Context-as-Memory)
    strict_overlap_labels: bool = False,
    allow_realtime_fallback: bool = True,
    allow_segment_fallback: bool = True,
    drop_overlap_probability: float = 0.1,  # 10% probability to drop overlap frames (paper strategy)
):
    """
    Backward-compatible wrapper.
    We use FOV overlap scoring to select top-k overlap frames (as in Context-as-Memory).

    According to Context-as-Memory [2506.03141]:
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


_WARN_ONCE_KEYS = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(key)
    print(f"[context_retrieval] WARN: {msg}", file=sys.stderr, flush=True)


def _load_latent(latent_dir: str, video_name: str, frame_idx: int):
    """Load a single-frame latent. Expects latent_dir/video_name/{frame_idx:04d}.pt or .pt with key 'latent' or raw tensor."""
    base = os.path.join(latent_dir, video_name)
    for fmt in (f"{frame_idx:04d}.pt", f"{frame_idx}.pt"):
        path = os.path.join(base, fmt)
        if os.path.isfile(path):
            try:
                try:
                    x = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    x = torch.load(path, map_location="cpu")
                if isinstance(x, dict) and "latent" in x:
                    z = x["latent"]
                else:
                    z = x
                if hasattr(z, "shape"):
                    # (C, 1, H, W) or (C, H, W) -> flatten for similarity
                    return z.flatten()
                return None
            except Exception:
                pass
    return None


def latent_sim_rank(
    video_name: str,
    first_frame_idx: int,
    overlapping_indices: List[int],
    dataset_base_path: str,
    top_k: int,
    latent_dir: Optional[str] = None,
    use_cosine: bool = True,
) -> List[int]:
    """
    Rank overlapping frame indices by latent similarity to the first (reference) frame.
    If latent_dir is None or latents are missing, falls back to random sample.
    Expects per-frame latents under latent_dir/video_name/{frame_idx:04d}.pt (or {frame_idx}.pt).
    """
    if not overlapping_indices or top_k <= 0:
        return []
    if latent_dir is None or not os.path.isdir(latent_dir):
        return random.sample(overlapping_indices, min(top_k, len(overlapping_indices)))

    ref = _load_latent(latent_dir, video_name, first_frame_idx)
    if ref is None:
        return random.sample(overlapping_indices, min(top_k, len(overlapping_indices)))

    ref = ref.float().unsqueeze(0)
    scores = []
    for idx in overlapping_indices:
        cand = _load_latent(latent_dir, video_name, idx)
        if cand is None:
            continue
        cand = cand.float().unsqueeze(0)
        if use_cosine:
            sim = torch.nn.functional.cosine_similarity(ref, cand, dim=1).item()
        else:
            sim = -((ref - cand) ** 2).sum().item()
        scores.append((idx, sim))
    if not scores:
        return random.sample(overlapping_indices, min(top_k, len(overlapping_indices)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:top_k]]


def retrieve_context_frames_advanced(
    data: Dict,
    dataset_base_path: str,
    top_k: int = 4,
    drop_overlap_probability: float = 0.1,
    use_rt_relative: bool = False,
    retrieval_method: str = "fov",
    latent_retrieval_dir: Optional[str] = None,
    strict_overlap_labels: bool = False,
) -> Tuple[List, List, List[int], int, str, str]:
    """
    Retrieve context frames with pluggable retrieval method.
    Interface matches retrieve_fov_context_frames return:
      (context_frames, context_actions, context_indices, cur_idx, video_name, source).

    retrieval_method:
      - "fov": use existing FOV/overlap_labels logic (random or RT-scored from overlap).
      - "latent_sim": rank overlap candidates by latent similarity to first frame (requires latent_retrieval_dir).
    When latent_sim is used but latent_retrieval_dir is missing or latents absent, falls back to FOV behavior.
    """
    if retrieval_method == "latent_sim" and not latent_retrieval_dir:
        _warn_once(
            "latent_sim_missing_dir",
            "retrieval_method=latent_sim but latent_retrieval_dir is not set; fallback to FOV retrieval.",
        )
    if retrieval_method == "fov" or (retrieval_method == "latent_sim" and not latent_retrieval_dir):
        return retrieve_simple_context_frames(
            data=data,
            dataset_base_path=dataset_base_path,
            top_k=top_k,
            drop_overlap_probability=drop_overlap_probability,
            use_rt_relative=use_rt_relative,
        )
    if retrieval_method == "latent_sim" and latent_retrieval_dir and not os.path.isdir(latent_retrieval_dir):
        _warn_once(
            "latent_sim_bad_dir",
            f"latent_retrieval_dir not found: {latent_retrieval_dir}; latent_sim will degrade to random-overlap selection.",
        )

    # latent_sim: we need to inject a custom ranking into the flow. We do a minimal duplicate of the
    # overlap selection step then reuse the rest via a wrapper around retrieve_simple_context_frames
    # by passing a custom rank function. Since retrieve_simple_context_frames doesn't support that yet,
    # we implement a full path here that mirrors it but uses latent_sim_rank for overlap selection.

    video_frames = data.get("video", [])
    start_frame = data.get("start_frame", 0)
    end_frame = data.get("end_frame", None)
    if "frame_idx" in data:
        current_frame_idx = data.get("frame_idx")
    else:
        current_frame_idx = (start_frame + end_frame) // 2 if end_frame is not None else (len(video_frames) // 2 if video_frames else 0)
    first_frame_idx = start_frame
    video_name = data.get("video_name", "")
    if not video_name and "video_path" in data:
        video_name = os.path.basename(data["video_path"]).replace(".mp4", "").replace(".avi", "")
    elif not video_name and "file_path" in data:
        video_name = os.path.basename(data["file_path"]).replace(".mp4", "").replace(".avi", "")

    frames_dir = os.path.join(dataset_base_path, "frames", video_name)
    json_file = os.path.join(dataset_base_path, "jsons", f"{video_name}.json")
    poses_dict = load_poses_dict(json_file) if os.path.isfile(json_file) else {}
    first_frame_pose = poses_dict.get(str(first_frame_idx))
    first_frame_pose_rt = pose_to_rt(first_frame_pose) if first_frame_pose is not None and pose_to_rt else None

    context_frames: List[Image.Image] = []
    context_actions: List = []
    context_indices: List[int] = []
    source = "none"

    def _append_pose(frame_idx: int):
        pose = poses_dict.get(str(frame_idx))
        if pose is not None and pose_to_rt and use_rt_relative and first_frame_pose_rt is not None:
            rt = pose_to_rt(pose)
            if rt is not None and convert_rt_to_relative:
                rel = convert_rt_to_relative([rt], first_frame_pose_rt)
                context_actions.append(rel[0] if rel else [0.0] * 12)
            else:
                context_actions.append([0.0] * 12)
        elif pose is not None and pose_to_rt:
            rt = pose_to_rt(pose)
            context_actions.append(rt if rt is not None else [0.0] * 12)
        else:
            context_actions.append([0.0] * 12)

    if first_frame_pose_rt is not None and use_rt_relative:
        context_actions.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    else:
        context_actions.append(first_frame_pose_rt if first_frame_pose_rt is not None else [0.0] * 12)

    if os.path.isdir(frames_dir):
        first_path = os.path.join(frames_dir, f"{first_frame_idx:04d}.png")
        if os.path.isfile(first_path):
            try:
                context_frames.append(Image.open(first_path).convert("RGB"))
                context_indices.append(first_frame_idx)
            except Exception:
                pass
    if not context_frames and video_frames:
        if isinstance(video_frames[0], Image.Image):
            context_frames.append(video_frames[0])
            context_indices.append(start_frame)
        elif isinstance(video_frames[0], str) and os.path.isfile(video_frames[0]):
            try:
                context_frames.append(Image.open(video_frames[0]).convert("RGB"))
                context_indices.append(start_frame)
            except Exception:
                pass

    drop_overlap = random.random() < drop_overlap_probability
    if drop_overlap:
        target_total = top_k + 1
        if strict_overlap_labels and len(context_frames) < target_total:
            return [], [], [], current_frame_idx, video_name, "overlap_labels_insufficient"
        return context_frames, context_actions, context_indices, current_frame_idx, video_name, "first_frame_only_dropped"

    overlap_labels_dir = os.path.join(dataset_base_path, "overlap_labels")
    overlapping_indices = []
    if os.path.isdir(overlap_labels_dir):
        overlapping_indices = load_overlap_frames(overlap_labels_dir, video_name, current_frame_idx)
    overlapping_indices = [i for i in overlapping_indices if i != first_frame_idx]

    if not overlapping_indices:
        source = "first_frame_only"
        if strict_overlap_labels and len(context_frames) < top_k + 1:
            return [], [], [], current_frame_idx, video_name, "overlap_labels_insufficient"
        return context_frames, context_actions, context_indices, current_frame_idx, video_name, source

    sampled_overlap_indices = latent_sim_rank(
        video_name, first_frame_idx, overlapping_indices, dataset_base_path, top_k,
        latent_dir=latent_retrieval_dir, use_cosine=True,
    )

    def _load_frame(path: str):
        if os.path.isfile(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
        return None

    to_load = [(idx, os.path.join(frames_dir, f"{idx:04d}.png")) for idx in sampled_overlap_indices[:top_k]]
    with ThreadPoolExecutor(max_workers=max(1, min(5, len(to_load)))) as ex:
        futures = {ex.submit(_load_frame, path): idx for idx, path in to_load}
        for fut in futures:
            idx = futures[fut]
            frame = fut.result()
            if frame is not None:
                context_frames.append(frame)
                context_indices.append(idx)
                _append_pose(idx)

    source = "latent_sim"
    if strict_overlap_labels and len(context_frames) < top_k + 1:
        return [], [], [], current_frame_idx, video_name, "overlap_labels_insufficient"
    return context_frames, context_actions, context_indices, current_frame_idx, video_name, source
