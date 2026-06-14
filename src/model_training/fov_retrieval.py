"""
FOV (Field of View) Overlap-based Memory Retrieval Module
Implements geometric retrieval based on camera pose overlap for Context-as-Memory

Aligned with the Context-as-Memory paper [2506.03141].
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

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

