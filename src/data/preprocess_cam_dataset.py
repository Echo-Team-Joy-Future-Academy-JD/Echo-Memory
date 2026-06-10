#!/usr/bin/env python3
"""
Preprocess Context-as-Memory dataset folders into Echo-Memory metadata CSV.

Expected dataset layout:
- frames/: frame images organized by video
- jsons/: camera pose information for each video
- overlap_labels/: FOV overlap information for memory retrieval
- captions.txt: video segment captions
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def parse_caption_line(line: str) -> Tuple[str, str]:
    """
    Parse a line from captions.txt.

    Format: "video_name/start_end.mp4\tcaption text..."
    Returns: (video_path, caption)
    """
    parts = line.strip().split("\t", 1)
    if len(parts) != 2:
        return None, None
    video_path = parts[0]
    caption = parts[1]
    return video_path, caption


def load_captions(captions_file: str) -> Dict[str, str]:
    """Load captions.txt as video_name -> caption."""
    captions = {}
    if not os.path.exists(captions_file):
        print(f"Warning: Captions file not found: {captions_file}")
        return captions

    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            video_path, caption = parse_caption_line(line)
            if video_path and caption:
                video_name = video_path.split("/")[0]
                if video_name not in captions:
                    captions[video_name] = []
                captions[video_name].append(caption)

    for video_name in captions:
        captions[video_name] = captions[video_name][0] if captions[video_name] else ""

    return captions


def get_frame_files(frames_dir: str, video_name: str) -> List[str]:
    """Get sorted frame paths for one video, relative to frames_dir."""
    video_frames_dir = os.path.join(frames_dir, video_name)
    if not os.path.exists(video_frames_dir):
        return []

    frame_files = []
    for frame_file in sorted(os.listdir(video_frames_dir)):
        if frame_file.endswith(".png"):
            frame_files.append(os.path.join(video_name, frame_file))

    return frame_files


def load_camera_poses(json_file: str) -> Dict:
    """Load camera poses from a JSON file."""
    if not os.path.exists(json_file):
        return {}

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "CineCameraActor" in data:
        return data["CineCameraActor"]
    if isinstance(data, dict):
        return data
    return {}


def load_overlap_labels(overlap_dir: str, video_name: str, frame_idx: int) -> List[int]:
    """Load overlapping frame indices for a given frame."""
    overlap_file = os.path.join(overlap_dir, video_name, f"{frame_idx}.json")
    if not os.path.exists(overlap_file):
        return []

    try:
        with open(overlap_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            overlapping_frames = data.get("overlapping_frames", [])
            return [int(frame) for frame in overlapping_frames if str(frame).isdigit()]
    except Exception:
        return []


def create_metadata_csv(
    dataset_base_path: str,
    output_csv: str,
    segment_length: int = 81,
    context_frames: int = 5,
):
    """
    Create metadata CSV for the Context-as-Memory dataset.

    Args:
        dataset_base_path: root of the dataset.
        output_csv: output CSV path.
        segment_length: frames per training segment.
        context_frames: context frames reserved by downstream workflows.
    """
    frames_dir = os.path.join(dataset_base_path, "frames")
    captions_file = os.path.join(dataset_base_path, "captions.txt")

    captions = load_captions(captions_file)

    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return

    video_names = [
        d for d in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, d))
    ]

    print(f"Found {len(video_names)} videos")
    print(f"Context frames: {context_frames}")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "video",
            "prompt",
            "video_name",
            "start_frame",
            "end_frame",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_segments = 0

        for video_name in sorted(video_names):
            print(f"Processing video: {video_name}")

            frame_files = get_frame_files(frames_dir, video_name)
            if len(frame_files) < segment_length:
                print(
                    f"  Skipping {video_name}: only {len(frame_files)} frames "
                    f"(need at least {segment_length})"
                )
                continue

            prompt = captions.get(video_name, f"A scene from {video_name}")
            step = max(1, segment_length // 2)
            video_segments = 0

            for start_idx in range(0, len(frame_files) - segment_length + 1, step):
                end_idx = start_idx + segment_length - 1
                segment_frames = frame_files[start_idx:end_idx + 1]

                if len(segment_frames) < segment_length:
                    continue

                frame_paths = "|".join(segment_frames)
                video_path = os.path.join("frames", frame_paths)

                writer.writerow({
                    "video": video_path,
                    "prompt": prompt,
                    "video_name": video_name,
                    "start_frame": start_idx,
                    "end_frame": end_idx,
                })

                total_segments += 1
                video_segments += 1

            print(f"  Created {video_segments} segments for {video_name}")

        print(f"\nTotal segments created: {total_segments}")
        print(f"Metadata CSV saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Context-as-Memory Dataset")
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        required=True,
        help="Base path to Context-as-Memory dataset",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="metadata.csv",
        help="Output CSV file path (default: metadata.csv)",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=81,
        help="Length of video segments (default: 81 frames)",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=5,
        help="Number of context frames (default: 5)",
    )

    args = parser.parse_args()

    if not os.path.isabs(args.output_csv):
        args.output_csv = os.path.join(args.dataset_base_path, args.output_csv)

    create_metadata_csv(
        dataset_base_path=args.dataset_base_path,
        output_csv=args.output_csv,
        segment_length=args.segment_length,
        context_frames=args.context_frames,
    )


if __name__ == "__main__":
    main()
