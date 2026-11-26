"""Deep-stage DDT prediction head used by the native-matching parent."""

from torch import nn
from torch.nn import functional as F

from .layers import Attention, RMSNorm, Rotary2D, SwiGLU


class DecoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.attn = Attention(width, heads, qk_norm=True, qkv_bias=False)
        self.mlp = SwiGLU(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, condition, rope):
        sa, ca, ga, sf, cf, gf = self.modulation(condition).chunk(6, dim=-1)
        x = x + ga * self.attn(self.norm1(x) * (1 + ca) + sa, rope)
        return x + gf * self.mlp(self.norm2(x) * (1 + cf) + sf)


class HiDDTHead(nn.Module):
    def __init__(self, trunk_width, channels, width, heads, grid):
        super().__init__()
        self.x_embedder = nn.Linear(channels, width)
        self.semantic_projector = nn.Linear(trunk_width, width)
        self.blocks = nn.ModuleList(DecoderBlock(width, heads) for _ in range(2))
        self.rope = Rotary2D(grid, width // heads)
        self.norm = RMSNorm(width)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.linear = nn.Linear(width, channels)
        for layer in (self.x_embedder, self.semantic_projector):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.linear, self.final_modulation[-1], *(b.modulation[-1] for b in self.blocks)]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, noisy_hi, semantic, time):
        condition = self.semantic_projector(F.silu(semantic + time[:, None]))
        x = self.x_embedder(noisy_hi)
        for block in self.blocks:
            x = block(x, condition, self.rope)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale) + shift)
