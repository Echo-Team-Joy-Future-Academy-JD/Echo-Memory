#!/usr/bin/env python3
"""Generate the action JSON files used by training and inference recipes."""

import json
import math
import os

def degrees_to_radians(degrees):
    return degrees * math.pi / 180

def generate_rotation_matrix(yaw_degrees):
    """Z-axis (yaw) rotation matrix, paper-aligned; row-major flatten: R_11..R_33."""
    yaw_rad = degrees_to_radians(yaw_degrees)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # R_z = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]

def generate_action_json(total_frames=81, total_rotation_degrees=180, output_path=None):
    actions = {}
    for frame_idx in range(total_frames):
        rotation_degrees = (frame_idx / (total_frames - 1)) * total_rotation_degrees
        R = generate_rotation_matrix(rotation_degrees)
        action = [0.0, 0.0, 0.0] + R
        actions[str(frame_idx)] = action
    if output_path:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(actions, f, indent=2)
        print(f"Generated {output_path} with {total_rotation_degrees}° rotation over {total_frames} frames")
    return actions

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_left_45 = os.path.join(script_dir, "action_rotation_left_45.json")
    out_right_45 = os.path.join(script_dir, "action_rotation_right_45.json")
    generate_action_json(total_frames=81, total_rotation_degrees=45, output_path=out_left_45)
    generate_action_json(total_frames=81, total_rotation_degrees=-45, output_path=out_right_45)
    print("\nGenerated action JSON files: left_45, right_45")
