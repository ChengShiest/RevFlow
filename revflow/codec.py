"""Frozen DINOv3-L/16 feature encoder and ViT-B pixel decoder for ImageNet 256."""

import torch
from torch import nn
from torch.nn import functional as F

from revflow.models.layers import Attention, RMSNorm, SwiGLU, grid_coordinates


class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = self.convs(1024, 128)
        self.decoder = self.convs(128, 1024)
        self.enc_norm = nn.LayerNorm(128, elementwise_affine=False)
        self.dec_norm = nn.LayerNorm(1024)

    @staticmethod
    def convs(source, target):
        return nn.Sequential(nn.Conv2d(source, 256, 3, padding=1), nn.GELU(),
                             nn.Conv2d(256, target, 3, padding=1))

    @staticmethod
    def project(tokens, conv, norm):
        grid = tokens.transpose(1, 2).reshape(len(tokens), -1, 16, 16)
        return norm(conv(grid).flatten(2).transpose(1, 2))

    def encode(self, x):
        return self.project(x, self.encoder, self.enc_norm)

    def decode(self, x):
        return self.project(x, self.decoder, self.dec_norm)


class DecoderRotary(nn.Module):
    """The codec uses split-half rotary pairs and prepends a class token."""
    def __init__(self):
        super().__init__()
        row, col = grid_coordinates(16)
        frequencies = 1 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
        angles = torch.cat([axis.flatten()[:, None] * frequencies for axis in (row, col)], dim=-1)
        angles = F.pad(angles, (0, 0, 1, 0))
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        left, right = x.chunk(2, dim=-1)
        return x * self.cos + torch.cat((-right, left), dim=-1) * self.sin


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(768), RMSNorm(768)
        self.attn = Attention(768, 12, qk_norm=True)
        self.mlp = SwiGLU(768)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class PixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_embed = nn.Linear(1024, 768)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.rope = DecoderRotary()
        self.decoder_layers = nn.ModuleList(DecoderBlock() for _ in range(12))
        self.final_layer = nn.Module()
        self.final_layer.norm_final = RMSNorm(768)
        self.final_layer.linear = nn.Linear(768, 16 * 16 * 3)

    def forward(self, tokens):
        x = torch.cat((self.trainable_cls_token.expand(len(tokens), -1, -1), self.decoder_embed(tokens)), dim=1)
        for layer in self.decoder_layers:
            x = layer(x, self.rope)
        patches = self.final_layer.linear(self.final_layer.norm_final(x[:, 1:]))
        grid = patches.reshape(len(x), 16, 16, 16, 16, 3)
        return grid.permute(0, 5, 1, 3, 2, 4).reshape(len(x), 3, 256, 256)


def load_bottleneck(weights):
    bottleneck = Bottleneck()
    bottleneck.load_state_dict(weights["bottleneck"], strict=True)
    return bottleneck.requires_grad_(False).eval()


class FeatureEncoder(nn.Module):
    def __init__(self, dino_repo, dino_weights, codec_weights):
        super().__init__()
        self.dino = torch.hub.load(str(dino_repo), "dinov3_vitl16", source="local", pretrained=False)
        self.dino.load_state_dict(torch.load(dino_weights, map_location="cpu", weights_only=True, mmap=True))
        self.bottleneck = load_bottleneck(codec_weights)
        self.requires_grad_(False).eval()

    def forward(self, images):
        lo, hi = self.dino.get_intermediate_layers(images, n=[7, 23], norm=False)
        return F.layer_norm(hi, (1024,)), self.bottleneck.encode(lo)


class ImageDecoder(nn.Module):
    def __init__(self, codec_weights):
        super().__init__()
        self.bottleneck = load_bottleneck(codec_weights)
        self.decoder = PixelDecoder()
        self.decoder.load_state_dict(codec_weights["decoder"], strict=True)
        self.requires_grad_(False).eval()

    def forward(self, lo):
        pixels = self.decoder(self.bottleneck.decode(lo.float()))
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (pixels * std + mean).clamp(0, 1)


class FlowStats(nn.Module):
    def __init__(self, state):
        super().__init__()
        for stage, channels in (("source", 1024), ("target", 128)):
            for moment in ("mean", "std"):
                key = f"{stage}_{moment}"
                value = state[key].float().reshape(-1, channels)
                if value.shape[0] not in (1, 256) or not value.isfinite().all():
                    raise ValueError(f"Invalid flow statistics: {key}")
                if moment == "std" and not (value > 0).all():
                    raise ValueError(f"Expected positive standard deviations: {key}")
                self.register_buffer(key, value)

    def normalize(self, hi, lo):
        return ((hi.float() - self.source_mean) / self.source_std,
                (lo.float() - self.target_mean) / self.target_std)

    def decode_lo(self, lo):
        return lo.float() * self.target_std + self.target_mean
