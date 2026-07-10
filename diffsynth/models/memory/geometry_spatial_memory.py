import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometrySpatialMemory(nn.Module):
    """Encode a rendered static 3D scene into compact conditioning tokens.

    The input must be VAE latents of a geometry-rendered video produced from a
    persistent 3D representation (for example a TSDF-fused point cloud). This
    module intentionally does not infer geometry from ordinary context tokens.
    """

    def __init__(
        self,
        dim: int,
        latent_channels: int = 16,
        patch_size=(1, 2, 2),
        grid_size: int = 8,
        temporal_bins: int = 4,
        num_tokens: int = 64,
    ):
        super().__init__()
        self.dim = int(dim)
        self.latent_channels = int(latent_channels)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.grid_size = int(grid_size)
        self.temporal_bins = int(temporal_bins)
        self.num_tokens = int(num_tokens)
        self.geometry_patch_embedding = nn.Conv3d(
            self.latent_channels,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        source_tokens = self.temporal_bins * self.grid_size * self.grid_size
        self.register_buffer(
            "geometry_config",
            torch.tensor(
                [1, self.grid_size, self.temporal_bins],
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.geometry_to_tokens = nn.Parameter(torch.empty(source_tokens, self.num_tokens))
        self.norm = nn.LayerNorm(self.dim)
        nn.init.normal_(self.geometry_to_tokens, std=0.02)

    def initialize_from_dit_patch_embedding(self, patch_embedding: nn.Conv3d) -> None:
        """Initialize the geometry encoder from the pretrained video tokenizer."""
        if not isinstance(patch_embedding, nn.Conv3d):
            return
        if patch_embedding.weight.shape != self.geometry_patch_embedding.weight.shape:
            return
        with torch.no_grad():
            self.geometry_patch_embedding.weight.copy_(patch_embedding.weight)
            if (
                self.geometry_patch_embedding.bias is not None
                and patch_embedding.bias is not None
                and self.geometry_patch_embedding.bias.shape == patch_embedding.bias.shape
            ):
                self.geometry_patch_embedding.bias.copy_(patch_embedding.bias)

    def forward(self, geometry_latents: torch.Tensor) -> torch.Tensor:
        if geometry_latents is None or geometry_latents.ndim != 5:
            raise ValueError(
                "GeometrySpatialMemory expects geometry VAE latents shaped (B, C, F, H, W)."
            )
        if int(geometry_latents.shape[1]) != self.latent_channels:
            raise ValueError(
                f"GeometrySpatialMemory channel mismatch: input={geometry_latents.shape[1]} "
                f"expected={self.latent_channels}"
            )
        features = self.geometry_patch_embedding(geometry_latents)
        features = F.adaptive_avg_pool3d(
            features,
            (self.temporal_bins, self.grid_size, self.grid_size),
        )
        features = features.flatten(2).transpose(1, 2)
        mix = torch.softmax(self.geometry_to_tokens, dim=0)
        memory = torch.einsum("bsd,sm->bmd", features, mix)
        return self.norm(memory / math.sqrt(max(self.temporal_bins, 1)))

