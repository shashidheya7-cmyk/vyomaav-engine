"""
VYOMAAV Base Model Engine
Module: base_model.rope3d

Implements 3D Rotary Position Embeddings (3D-RoPE). Applies rotational frequency
encodings across (x, y, z) 3D spatial coordinates directly to Transformer Query and Key matrices.
"""

from typing import Tuple
import torch
import torch.nn as nn


class RotaryEmbedding3D(nn.Module):
    """3D Rotary Position Embeddings (3D-RoPE) for spatial-temporal transformer tokens."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        # Allocate dimension evenly across 3 spatial axes (x, y, z)
        assert dim % 6 == 0, "3D-RoPE dimension must be divisible by 6 (2 dims per axis * 3 axes)."
        self.dim = dim
        self.axis_dim = dim // 3
        self.base = base

        # Frequency bands for each axis
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_axis_cos_sin(
        self, pos: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pos: (B, N) -> 1D positions along one coordinate axis
        freqs = torch.einsum("bn,f->bnf", pos.to(device), self.inv_freq.to(device))  # (B, N, axis_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, N, axis_dim)
        return emb.cos(), emb.sin()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, coords_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies 3D rotary transforms to Query and Key tensors.

        Args:
            q, k: (B, H, N, D) -> Attention Query and Key tensors
            coords_3d: (B, N, 3) -> Real-world or camera-space 3D coordinates (x, y, z)
        """
        B, H, N, D = q.shape
        device = q.device

        # Split D into x, y, z coordinate channels
        q_x, q_y, q_z = torch.chunk(q, 3, dim=-1)
        k_x, k_y, k_z = torch.chunk(k, 3, dim=-1)

        cos_x, sin_x = self._compute_axis_cos_sin(coords_3d[..., 0], device)
        cos_y, sin_y = self._compute_axis_cos_sin(coords_3d[..., 1], device)
        cos_z, sin_z = self._compute_axis_cos_sin(coords_3d[..., 2], device)

        # Reshape for multi-head broadcasting: (B, 1, N, axis_dim)
        cos_x, sin_x = cos_x.unsqueeze(1), sin_x.unsqueeze(1)
        cos_y, sin_y = cos_y.unsqueeze(1), sin_y.unsqueeze(1)
        cos_z, sin_z = cos_z.unsqueeze(1), sin_z.unsqueeze(1)

        # Rotate each axis channel
        q_x_rot = (q_x * cos_x) + (self._rotate_half(q_x) * sin_x)
        k_x_rot = (k_x * cos_x) + (self._rotate_half(k_x) * sin_x)

        q_y_rot = (q_y * cos_y) + (self._rotate_half(q_y) * sin_y)
        k_y_rot = (k_y * cos_y) + (self._rotate_half(k_y) * sin_y)

        q_z_rot = (q_z * cos_z) + (self._rotate_half(q_z) * sin_z)
        k_z_rot = (k_z * cos_z) + (self._rotate_half(k_z) * sin_z)

        q_out = torch.cat([q_x_rot, q_y_rot, q_z_rot], dim=-1)
        k_out = torch.cat([k_x_rot, k_y_rot, k_z_rot], dim=-1)

        return q_out, k_out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)