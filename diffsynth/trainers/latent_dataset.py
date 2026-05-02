"""
LatentDataset: Load precomputed ctx and target latents for Context-as-Memory dataset.

Use after running precompute_ctx_target_latents.py. Returns samples with:
- context_latents: (1, C, K, H//8, W//8) - 1 latent per context frame
- target_latents: (1, C, T, H//8, W//8) - 1 latent per 4 target frames
- prompt, video_name, start_frame, end_frame, actions

Compatible with training that uses precomputed latents instead of encoding on the fly.
"""

import json
import os
import warnings

import torch


class LatentDataset(torch.utils.data.Dataset):
    """
    Dataset that loads precomputed ctx and target latents.
    """

    def __init__(
        self,
        latent_dir,
        metadata_path=None,
        action_base_path=None,
        repeat=1,
        num_frames=81,
        context_frames=5,
        target_frames_per_latent=4,
    ):
        """
        Args:
            latent_dir: Directory containing ctx_latents/ and target_latents/ subdirs.
            metadata_path: Optional. If provided, used to get total_samples and validate.
            action_base_path: Base path for action JSON files (for loading actions if not in .pt).
            repeat: Dataset repeat factor.
            num_frames: Expected num_frames per segment.
            context_frames: Number of context frames (K).
            target_frames_per_latent: Target: 1 latent per N frames.
        """
        self.latent_dir = latent_dir
        self.ctx_dir = os.path.join(latent_dir, "ctx_latents")
        self.target_dir = os.path.join(latent_dir, "target_latents")
        self.action_base_path = action_base_path or latent_dir
        self.repeat = repeat
        self.num_frames = num_frames
        self.context_frames = context_frames
        self.target_frames_per_latent = target_frames_per_latent

        # Infer valid indices from existing files (both ctx and target must exist)
        self._indices = []
        if os.path.isdir(self.ctx_dir) and os.path.isdir(self.target_dir):
            ctx_files = {f.replace(".pt", "") for f in os.listdir(self.ctx_dir) if f.endswith(".pt")}
            target_files = {f.replace(".pt", "") for f in os.listdir(self.target_dir) if f.endswith(".pt")}
            common = sorted([int(x) for x in ctx_files & target_files])
            self._indices = common
        if not self._indices:
            meta_path = os.path.join(latent_dir, "metadata_precompute.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self._total = meta.get("total_samples", 0)
                self._indices = list(range(self._total))
            else:
                self._total = 0
        else:
            self._total = len(self._indices)

    def __len__(self):
        return self._total * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % self._total
        if self._indices is not None:
            real_idx = self._indices[real_idx]

        ctx_path = os.path.join(self.ctx_dir, f"{real_idx:08d}.pt")
        target_path = os.path.join(self.target_dir, f"{real_idx:08d}.pt")

        if not os.path.isfile(ctx_path) or not os.path.isfile(target_path):
            warnings.warn(f"Latent files not found for idx {real_idx}. Returning None.")
            return None

        ctx_data = torch.load(ctx_path, map_location="cpu", weights_only=True)
        target_data = torch.load(target_path, map_location="cpu", weights_only=True)

        ctx_latent = ctx_data["latent"]
        target_latent = target_data["latent"]

        # Ensure batch dimension: (C, K, H, W) -> (1, C, K, H, W)
        if ctx_latent.dim() == 4:
            ctx_latent = ctx_latent.unsqueeze(0)
        if target_latent.dim() == 4:
            target_latent = target_latent.unsqueeze(0)

        out = {
            "context_latents": ctx_latent,
            "target_latents": target_latent,
            "prompt": ctx_data.get("prompt", ""),
            "video_name": ctx_data.get("video_name"),
            "start_frame": ctx_data.get("start_frame"),
            "end_frame": ctx_data.get("end_frame"),
        }
        if "actions" in ctx_data and ctx_data["actions"] is not None:
            out["actions"] = ctx_data["actions"]
        elif "actions" in target_data and target_data["actions"] is not None:
            out["actions"] = target_data["actions"]

        return out


def get_latent_dataset_args(latent_dir, action_base_path=None, **kwargs):
    """Build argparse.Namespace for LatentDataset from precompute metadata."""
    meta_path = os.path.join(latent_dir, "metadata_precompute.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    from argparse import Namespace
    return Namespace(
        latent_dir=latent_dir,
        action_base_path=action_base_path or meta.get("dataset_base_path", latent_dir),
        num_frames=meta.get("num_frames", 81),
        context_frames=meta.get("context_frames", 5),
        target_frames_per_latent=meta.get("target_frames_per_latent", 4),
        **kwargs,
    )
