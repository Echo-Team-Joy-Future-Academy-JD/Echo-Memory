"""
RT (Rotation-Translation) utility functions for camera pose processing.
Aligned with the Context-as-Memory paper [2506.03141].

Strict paper alignment (default for 4-param):
- 2D plane: displacement in XY only (Z forced to 0 when constrain_to_xy=True).
- Rotation: Z-axis only (yaw; roll/pitch ignored). Params = [x, y, z, yaw] (4 parameters).
- 为何不是五参数：Context-as-Memory 论文与 2D 平面设定下只使用 Z 轴旋转 (yaw)，故默认 4 参数；若数据或管线使用 yaw+pitch，可用 compute_rotation_list_yaw_pitch 或传入 5 参数。

Alignment with camera encoder and external RT pipelines:
- 12-dim layout: [t_x, t_y, t_z, R_11, R_12, R_13, R_21, R_22, R_23, R_31, R_32, R_33] (R row-major).
- convert_rt_to_relative: R_new = R_ref_inv @ R_i, T_new = R_ref_inv @ T_i + T_ref_inv; output same 12-dim order.
- Encoder input: same 12-dim; rotation may be Z-only (here) or R_z(pitch)*R_y(yaw) from other scripts.
"""

import numpy as np
from typing import List, Dict, Optional


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
