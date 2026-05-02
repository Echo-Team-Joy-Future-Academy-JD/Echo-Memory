#!/usr/bin/env python3
"""
Precompute VAE latents for Context-as-Memory dataset with separate ctx and target storage:
- ctx: 1 latent per frame (for context/memory frames)
- target: 1 latent per 4 frames (time_division_factor=4)

8-GPU distributed: each rank processes a subset of segments.

Two modes:
1. With metadata: --metadata_path metadata_full.csv (uses VideoDataset)
2. No metadata: --no_metadata - auto-discovers segments from frames/ + captions.txt (or captions.jsonl)

Usage (8 GPUs, no metadata):
  accelerate launch --num_processes 8 scripts/precompute_ctx_target_latents.py \\
    --dataset_base_path /path/to/Context-as-Memory-Dataset \\
    --output_dir /path/to/latents \\
    --model_paths '["dit.safetensors","t5.pth","VAE.pth"]' \\
    --no_metadata
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import VideoDataset


def load_captions_txt(captions_path):
    """Load captions.txt: video_name/start_end.mp4\\tcaption -> video_name -> caption."""
    captions = {}
    if not os.path.isfile(captions_path):
        return captions
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            video_path, caption = parts[0], parts[1]
            video_name = video_path.split("/")[0]
            if video_name not in captions:
                captions[video_name] = caption
    return captions


def load_captions_jsonl(captions_path):
    """Load captions.jsonl: each line {"video_name": "...", "prompt": "..."} or similar."""
    captions = {}
    if not os.path.isfile(captions_path):
        return captions
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                vn = obj.get("video_name") or obj.get("video") or obj.get("id", "")
                prompt = obj.get("prompt") or obj.get("caption") or obj.get("text", "")
                if vn:
                    captions[vn] = prompt
            except json.JSONDecodeError:
                continue
    return captions


def build_segments_from_frames(
    frames_dir,
    captions_path=None,
    captions_jsonl_path=None,
    num_frames=81,
    segment_stride=None,
    overlap_labels_dir=None,
    overlap_labels_dense=False,
):
    """
    Build segment list without metadata CSV.
    Modes:
    1. overlap_labels_dir + dense=False: 1 segment per overlap_labels JSON (~240k)
    2. overlap_labels_dir + dense=True: stride=1 within each video (~10x-40x more)
    3. else: stride-based from frames/ - stride 40 ~19k, stride 1 ~760k

    Returns: [(video_name, start_frame, end_frame, frame_paths, prompt), ...]
    """
    captions = {}
    if captions_path:
        captions = load_captions_txt(captions_path)
    if captions_jsonl_path:
        captions.update(load_captions_jsonl(captions_jsonl_path))
    default_prompt = "A video scene."

    segments = []
    stride = segment_stride if segment_stride is not None else max(1, num_frames // 2)

    if overlap_labels_dir and os.path.isdir(overlap_labels_dir):
        video_dirs = sorted(
            [d for d in os.listdir(overlap_labels_dir) if os.path.isdir(os.path.join(overlap_labels_dir, d))]
        )
        for video_name in video_dirs:
            video_frames_dir = os.path.join(frames_dir, video_name)
            if not os.path.isdir(video_frames_dir):
                continue
            frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith(".png")])
            if len(frame_files) < num_frames:
                continue
            prompt = captions.get(video_name, default_prompt)

            if overlap_labels_dense:
                # Dense: stride=1 for max segments (~10x-40x more). Overridable via segment_stride.
                seg_stride = segment_stride if segment_stride is not None else 1
            else:
                # 1 segment per overlap_labels JSON
                video_overlap_dir = os.path.join(overlap_labels_dir, video_name)
                json_files = sorted([f for f in os.listdir(video_overlap_dir) if f.endswith(".json")])
                for jf in json_files:
                    try:
                        start_frame = int(jf.replace(".json", ""))
                    except ValueError:
                        continue
                    end_frame = start_frame + num_frames - 1
                    frame_paths = [
                        os.path.join(video_name, f"{start_frame + i:04d}.png") for i in range(num_frames)
                    ]
                    first_path = os.path.join(frames_dir, frame_paths[0])
                    last_path = os.path.join(frames_dir, frame_paths[-1])
                    if os.path.isfile(first_path) and os.path.isfile(last_path):
                        segments.append((video_name, start_frame, end_frame, frame_paths, prompt))
                continue

            for start in range(0, len(frame_files) - num_frames + 1, seg_stride):
                end = start + num_frames - 1
                frame_paths = [os.path.join(video_name, frame_files[i]) for i in range(start, end + 1)]
                segments.append((video_name, start, end, frame_paths, prompt))
    else:
        # Stride-based from frames/
        video_dirs = sorted([d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))])
        for video_name in video_dirs:
            video_dir = os.path.join(frames_dir, video_name)
            frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".png")])
            if len(frame_files) < num_frames:
                continue
            prompt = captions.get(video_name, default_prompt)
            for start in range(0, len(frame_files) - num_frames + 1, stride):
                end = start + num_frames - 1
                frame_paths = [os.path.join(video_name, frame_files[i]) for i in range(start, end + 1)]
                segments.append((video_name, start, end, frame_paths, prompt))
    return segments


class FrameSegmentDataset(torch.utils.data.Dataset):
    """Dataset that loads frames from segment list (no metadata CSV)."""

    def __init__(self, base_path, segments, height, width):
        self.base_path = base_path
        self.frames_dir = os.path.join(base_path, "frames")
        self.segments = segments
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.segments)

    def _load_image(self, rel_path):
        path = os.path.join(self.frames_dir, rel_path)
        img = Image.open(path).convert("RGB")
        import torchvision.transforms.functional as TF
        w, h = img.size
        scale = max(self.width / w, self.height / h)
        img = TF.resize(img, (round(h * scale), round(w * scale)), interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, (self.height, self.width))
        return img

    def __getitem__(self, idx):
        video_name, start_frame, end_frame, frame_paths, prompt = self.segments[idx]
        frames = []
        for fp in frame_paths:
            try:
                frames.append(self._load_image(fp))
            except Exception:
                return None
        if len(frames) != len(frame_paths):
            return None
        return {
            "video": frames,
            "prompt": prompt,
            "video_name": video_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute ctx (1 latent/frame) and target (1 latent/4 frames) latents."
    )
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        required=True,
        help="Dataset root (contains frames/, metadata).",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Metadata CSV path. Omit when using --no_metadata.",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Skip metadata CSV; auto-discover segments from frames/ + captions.",
    )
    parser.add_argument(
        "--captions_path",
        type=str,
        default=None,
        help="captions.txt path (for --no_metadata). Default: {dataset_base_path}/captions.txt",
    )
    parser.add_argument(
        "--captions_jsonl_path",
        type=str,
        default=None,
        help="Optional captions.jsonl path (for --no_metadata).",
    )
    parser.add_argument(
        "--segment_stride",
        type=int,
        default=None,
        help="Stride between segments. Default: 40 (stride mode), 1 (--overlap_labels_dense). Use 1 for max.",
    )
    parser.add_argument(
        "--use_overlap_labels",
        action="store_true",
        help="When --no_metadata: use overlap_labels/ to discover segments (matches metadata_full ~240k).",
    )
    parser.add_argument(
        "--overlap_labels_dense",
        action="store_true",
        help="With --use_overlap_labels: stride=1 per video (~10x-40x more segments).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory. Will create ctx_latents/ and target_latents/ subdirs.",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help='JSON array of model paths, e.g. \'["dit.safetensors","t5.pth","Wan2.1_VAE.pth"]\'.',
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument(
        "--context_frames",
        type=int,
        default=5,
        help="Number of context frames (each gets 1 latent).",
    )
    parser.add_argument(
        "--target_frames_per_latent",
        type=int,
        default=4,
        help="Target: 1 latent per N frames (default 4).",
    )
    parser.add_argument("--action_base_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=8,
        help="Batch size for VAE encode (frames per call). Higher = better GPU util, more VRAM.",
    )
    parser.add_argument(
        "--segment_batch_size",
        type=int,
        default=8,
        help="Process N segments per iteration (DataLoader batch). Speeds up I/O + encode.",
    )
    return parser.parse_args()


def make_dataset_args(args):
    return argparse.Namespace(
        dataset_base_path=args.dataset_base_path,
        dataset_metadata_path=args.metadata_path,
        height=args.height,
        width=args.width,
        max_pixels=1920 * 1080,
        num_frames=args.num_frames,
        dataset_repeat=1,
        data_file_keys="video,video_name,start_frame,end_frame",
        action_base_path=args.action_base_path or args.dataset_base_path,
    )


def crop_and_resize(image, target_height, target_width):
    import torchvision.transforms.functional as TF
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = TF.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    image = TF.center_crop(image, (target_height, target_width))
    return image


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.no_metadata and not args.metadata_path:
        raise ValueError("Either --metadata_path or --no_metadata is required.")

    accelerator = Accelerator()
    if accelerator.num_processes != 8 and accelerator.is_main_process:
        print(f"Info: using {accelerator.num_processes} processes (expected 8).")

    ctx_dir = os.path.join(args.output_dir, "ctx_latents")
    target_dir = os.path.join(args.output_dir, "target_latents")
    os.makedirs(ctx_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    # Redirect rank 0 stdout to log file in output_dir
    _rank0_log_file = None
    if accelerator.is_main_process:
        log_path = os.path.join(args.output_dir, "precompute_log.txt")
        _rank0_log_file = open(log_path, "w", encoding="utf-8")
        _rank0_log_file.write(f"[{datetime.now().isoformat()}] precompute_ctx_target_latents started\n")
        _rank0_log_file.write(f"output_dir={args.output_dir}\n")
        _rank0_log_file.flush()

        class _Tee:
            def __init__(self, *files):
                self.files = files

            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()

            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = _Tee(sys.__stdout__, _rank0_log_file)

    if args.no_metadata:
        frames_dir = os.path.join(args.dataset_base_path, "frames")
        captions_path = args.captions_path or os.path.join(args.dataset_base_path, "captions.txt")
        overlap_labels_dir = None
        if args.use_overlap_labels:
            overlap_labels_dir = os.path.join(args.dataset_base_path, "overlap_labels")
        segments = build_segments_from_frames(
            frames_dir,
            captions_path=captions_path,
            captions_jsonl_path=args.captions_jsonl_path,
            num_frames=args.num_frames,
            segment_stride=args.segment_stride,
            overlap_labels_dir=overlap_labels_dir,
            overlap_labels_dense=args.overlap_labels_dense,
        )
        dataset = FrameSegmentDataset(
            args.dataset_base_path, segments, args.height, args.width
        )
        if accelerator.is_main_process:
            if overlap_labels_dir:
                src = "overlap_labels (dense)" if args.overlap_labels_dense else "overlap_labels"
            else:
                src = "frames (stride)"
            print(f"No-metadata mode: discovered {len(segments)} segments from {src}")
    else:
        dataset_args = make_dataset_args(args)
        dataset = VideoDataset(args=dataset_args)

    total = len(dataset)
    if total == 0:
        if accelerator.is_main_process:
            print("Dataset is empty. Exit.")
        return

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )
    indices = list(sampler)
    n_local = len(indices)

    if accelerator.is_main_process:
        print(f"Dataset size: {total}. Rank 0 processing {n_local} indices.")
        meta = {
            "dataset_base_path": args.dataset_base_path,
            "metadata_path": args.metadata_path,
            "no_metadata": args.no_metadata,
            "total_samples": total,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "context_frames": args.context_frames,
            "target_frames_per_latent": args.target_frames_per_latent,
        }
        with open(os.path.join(args.output_dir, "metadata_precompute.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Load pipeline (VAE only)
    model_paths = json.loads(args.model_paths)
    model_configs = [ModelConfig(path=p) for p in model_paths]
    from_pretrained_kw = {
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
        "model_configs": model_configs,
    }
    if args.tokenizer_path:
        from_pretrained_kw["tokenizer_config"] = ModelConfig(path=args.tokenizer_path)

    if accelerator.is_main_process:
        print("Loading pipeline (VAE)...")
    pipe = WanVideoPipeline.from_pretrained(**from_pretrained_kw)
    pipe.vae.to(accelerator.device)
    pipe.vae.eval()

    K = args.context_frames
    step = args.target_frames_per_latent

    def preprocess_frames(frames):
        return pipe.preprocess_video(frames)

    @torch.no_grad()
    def encode_frame(frame_pil):
        """Encode single frame -> (C, 1, H//8, W//8)"""
        vid = preprocess_frames([frame_pil])
        if vid.dim() == 5:
            vid = vid.squeeze(0)
        lat = pipe.vae.encode([vid], device=accelerator.device, tiled=False, tile_size=None, tile_stride=None)
        return lat[0].cpu()

    failed = 0
    skipped = 0
    for idx in tqdm(
        indices,
        desc=f"Rank {accelerator.process_index}",
        disable=not accelerator.is_local_main_process,
    ):
        sample = dataset[idx]
        ctx_path = os.path.join(ctx_dir, f"{idx:08d}.pt")
        target_path = os.path.join(target_dir, f"{idx:08d}.pt")
        if args.skip_existing and os.path.isfile(ctx_path) and os.path.isfile(target_path):
            skipped += 1
            continue
        try:
            if sample is None:
                failed += 1
                continue
            video_frames = sample.get("video")
            if not video_frames or len(video_frames) == 0:
                failed += 1
                continue
            if len(video_frames) != args.num_frames:
                if len(video_frames) > args.num_frames:
                    video_frames = video_frames[: args.num_frames]
                else:
                    last = video_frames[-1] if video_frames else None
                    while len(video_frames) < args.num_frames and last is not None:
                        video_frames = video_frames + [last]
                    if len(video_frames) < args.num_frames:
                        failed += 1
                        continue

            # Context: 1 latent per frame (frames 0..K-1)
            ctx_latents_list = []
            for i in range(min(K, len(video_frames))):
                lat = encode_frame(video_frames[i])
                if isinstance(lat, (list, tuple)):
                    lat = lat[0]
                ctx_latents_list.append(lat)
            ctx_latent = torch.cat(ctx_latents_list, dim=1)
            if ctx_latent.dim() == 5:
                ctx_latent = ctx_latent.squeeze(0)

            # Target: 1 latent per `step` frames (frames K, K+step, ...)
            target_indices = list(range(K, len(video_frames), step))
            target_latents_list = []
            for i in target_indices:
                lat = encode_frame(video_frames[i])
                if isinstance(lat, (list, tuple)):
                    lat = lat[0]
                target_latents_list.append(lat)
            if not target_latents_list:
                failed += 1
                continue
            target_latent = torch.cat(target_latents_list, dim=1)
            if target_latent.dim() == 5:
                target_latent = target_latent.squeeze(0)

            save_meta = {
                "prompt": sample.get("prompt", ""),
                "video_name": sample.get("video_name"),
                "start_frame": sample.get("start_frame"),
                "end_frame": sample.get("end_frame"),
            }
            if "actions" in sample and sample["actions"] is not None:
                a = sample["actions"]
                save_meta["actions"] = torch.tensor(a) if not isinstance(a, torch.Tensor) else a.cpu()

            torch.save({"latent": ctx_latent, **save_meta}, ctx_path)
            torch.save({"latent": target_latent, **save_meta}, target_path)
        except Exception as e:
            if accelerator.is_local_main_process:
                tqdm.write(f"Rank {accelerator.process_index} idx {idx}: {e}")
            failed += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"Precompute done. ctx_latents/ and target_latents/ under {args.output_dir}. "
            f"Failed: {failed}, Skipped: {skipped}."
        )
        if _rank0_log_file is not None:
            sys.stdout = sys.__stdout__
            _rank0_log_file.close()


if __name__ == "__main__":
    main()
