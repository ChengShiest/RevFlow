"""LightningDiT building blocks for a grid of image tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        values = x.float()
        normed = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        return normed.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        inner = int(4 * width * 2 / 3)
        self.w12 = nn.Linear(width, 2 * inner)
        self.w3 = nn.Linear(inner, width)

    def forward(self, x):
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


def grid_coordinates(side):
    coordinates = torch.arange(side, dtype=torch.float32)
    return torch.meshgrid(coordinates, coordinates, indexing="ij")


def fixed_positions(side, width):
    row, col = grid_coordinates(side)
    frequencies = 10000 ** (-torch.arange(width // 4, dtype=torch.float64) / (width // 4))
    angles_x = col.flatten()[:, None] * frequencies
    angles_y = row.flatten()[:, None] * frequencies
    return torch.cat((angles_x.sin(), angles_x.cos(), angles_y.sin(), angles_y.cos()), dim=-1).float()[None]


class Rotary2D(nn.Module):
    def __init__(self, side, head_dim):
        super().__init__()
        row, col = grid_coordinates(side)
        frequencies = 1 / (10000 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
        angles = torch.cat([axis.flatten()[:, None] * frequencies for axis in (row, col)], dim=-1)
        angles = angles.repeat_interleave(2, dim=-1)
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, x):
        even, odd = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * self.freqs_cos + rotated * self.freqs_sin


class Attention(nn.Module):
    def __init__(self, width, heads, qk_norm=False, qkv_bias=True):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=qkv_bias)
        self.proj = nn.Linear(width, width)
        self.q_norm = RMSNorm(width // heads) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(width // heads) if qk_norm else nn.Identity()

    def forward(self, x, rope):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=t.device).float() / 128)
        angles = t.float()[:, None] * frequencies
        return self.mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))
