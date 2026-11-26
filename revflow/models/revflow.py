"""Shared LightningDiT with global hi and spatial lo conditioning."""

import torch
from torch import nn

from .layers import Attention, RMSNorm, Rotary2D, SwiGLU, TimeEmbedding, fixed_positions
from .ddt import HiDDTHead


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = Attention(width, heads)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, condition, rope):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(condition).chunk(6, dim=-1)
        x = x + gate_a * self.attn(self.norm1(x) * (1 + scale_a) + shift_a, rope)
        return x + gate_m * self.mlp(self.norm2(x) * (1 + scale_m) + shift_m)


class Readout(nn.Module):
    def __init__(self, width, channels):
        super().__init__()
        self.norm_final = RMSNorm(width)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.linear = nn.Linear(width, channels)

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class RevFlow(nn.Module):
    def __init__(self, input_size=16, hi_channels=1024, lo_channels=128, hidden_size=1152,
                 depth=28, num_heads=16, base_model_depth=8, num_classes=1000,
                 ddt_width=None, ddt_heads=16):
        super().__init__()
        if not 1 <= base_model_depth <= depth or hidden_size % (4 * num_heads):
            raise ValueError("Check the base readout depth and the rotary head dimension")
        self.side, self.base_depth = input_size, base_model_depth
        self.hi_channels, self.lo_channels = hi_channels, lo_channels
        self.num_classes = num_classes
        self.pos_embed = nn.Parameter(fixed_positions(input_size, hidden_size), requires_grad=False)
        self.rope = Rotary2D(input_size, hidden_size // num_heads)
        self.blocks = nn.ModuleList(Block(hidden_size, num_heads) for _ in range(depth))
        self.input_embedders = nn.ModuleDict({task: nn.Linear(channels, hidden_size)
                                              for task, channels in (("hi", hi_channels), ("lo", lo_channels))})
        self.condition_projection = nn.Linear(hi_channels, hidden_size)
        self.t_embedder = TimeEmbedding(hidden_size)
        self.sigma_embedder = TimeEmbedding(hidden_size)
        self.task_embedder = nn.Embedding(2, hidden_size)
        self.y_embedder = nn.Module()
        self.y_embedder.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.final_hi, self.base_hi = Readout(hidden_size, hi_channels), Readout(hidden_size, hi_channels)
        self.final_lo, self.base_lo = Readout(hidden_size, lo_channels), Readout(hidden_size, lo_channels)
        self.initialize_weights()
        self.ddt_hi = None
        if ddt_width is not None:
            if ddt_width % (4 * ddt_heads):
                raise ValueError("DDT head dimension must be divisible by four")
            del self.final_hi
            self.ddt_hi = HiDDTHead(hidden_size, hi_channels, ddt_width, ddt_heads, input_size)

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight, std=0.02)
        for embedding in (self.t_embedder, self.sigma_embedder):
            for layer in (embedding.mlp[0], embedding.mlp[2]):
                nn.init.normal_(layer.weight, std=0.02)
        readouts = (self.final_hi, self.final_lo, self.base_hi, self.base_lo)
        for module in (*self.blocks, *readouts):
            nn.init.zeros_(module.adaLN_modulation[-1].weight)
            nn.init.zeros_(module.adaLN_modulation[-1].bias)
        for module in readouts:
            nn.init.zeros_(module.linear.weight)
            nn.init.zeros_(module.linear.bias)

    def hi_condition(self, labels):
        return self.y_embedder.embedding_table(labels)[:, None]

    def lo_condition(self, hi, sigma):
        return self.condition_projection(hi), self.sigma_embedder(sigma)

    def forward_stage(self, x, t, task, condition):
        if task == "hi":
            time = self.t_embedder(t)
            c = (time + self.task_embedder.weight[0])[:, None] + condition
            base = self.base_hi
        elif task == "lo":
            spatial, sigma = condition
            time = self.t_embedder(t + 1) + self.task_embedder.weight[1]
            c = spatial + (time + sigma)[:, None]
            base = self.base_lo
        else:
            raise ValueError(f"Unknown stage {task}")
        noisy = x
        x = self.input_embedders[task](x) + self.pos_embed
        for depth, block in enumerate(self.blocks, start=1):
            x = block(x, c, self.rope)
            if depth == self.base_depth:
                mid = x
        if task == "hi" and self.ddt_hi is not None:
            final = self.ddt_hi(noisy, x, time)
        else:
            final = (self.final_hi if task == "hi" else self.final_lo)(x, c)
        return final, base(mid, c)

    def forward(self, hi, t_hi, lo, t_lo, labels, teacher_hi, sigma):
        final_hi, base_hi = self.forward_stage(hi, t_hi, "hi", self.hi_condition(labels))
        final_lo, base_lo = self.forward_stage(lo, t_lo, "lo", self.lo_condition(teacher_hi, sigma))
        return final_hi, final_lo, base_hi, base_lo
