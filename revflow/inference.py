"""Generate class-conditional PNG images with the RevFlow EMA."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from revflow.checkpoints import DEFAULT_CONFIG, load_weights, read_config
from revflow.models import RevFlow
from revflow.codec import FlowStats, ImageDecoder
from revflow.flow import sample
from revflow.native_matching.model import guided_pair


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("checkpoint", "codec", "stats", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--classes", nargs="+", type=int, required=True, help="One image per class ID, in order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--hi-scale", type=float, help="override sampling.hi_scale of the config")
    parser.add_argument("--lo-scale", type=float, help="override sampling.lo_scale of the config")
    parser.add_argument("--hi-min-time", type=float, help="override sampling.hi_min_time (guidance only for shifted t above it)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = read_config(args.config)
    for key in ("hi_scale", "lo_scale", "hi_min_time"):
        if getattr(args, key) is not None:
            if key == "hi_min_time" and config.get("native_matching"):
                parser.error("--hi-min-time does not apply to native-matching checkpoints")
            config["sampling"][key] = getattr(args, key)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    model = RevFlow(**config["model"])
    if any(label < 0 or label >= model.num_classes for label in args.classes):
        parser.error(f"Class IDs must be in [0, {model.num_classes - 1}]")
    payload = load_weights(args.checkpoint)
    model.load_state_dict(payload.get("ema", payload), strict=True)
    del payload
    model.to(device).eval()
    decoder = ImageDecoder(load_weights(args.codec)).to(device)
    stats = FlowStats(load_weights(args.stats)).to(device)
    nm = config.get("native_matching")
    generator = torch.Generator(device="cpu" if nm else device).manual_seed(args.seed)
    args.output.mkdir(exist_ok=True, parents=True)
    for offset in range(0, len(args.classes), args.batch_size):
        labels = torch.tensor(args.classes[offset:offset + args.batch_size], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16" and device.type == "cuda"):
            if nm:
                noise = torch.randn(len(labels), model.side ** 2, model.hi_channels + model.lo_channels,
                                    generator=generator).to(device)
                time = torch.ones(len(labels), device=device)
                raw = guided_pair(model, noise, time, labels, **config["sampling"])
                lo = raw[..., model.hi_channels:]
            else:
                _, lo = sample(model, labels, config["flow"], config["sampling"], generator)
        pixels = decoder(stats.decode_lo(lo)).mul(255)
        if not nm:
            pixels.add_(.5)
        images = pixels.clamp_(0, 255).byte().permute(0, 2, 3, 1).cpu()
        for index, pixels in enumerate(images):
            path = args.output / f"{offset + index:05d}-class{args.classes[offset + index]:04d}.png"
            Image.fromarray(pixels.numpy()).save(path)


if __name__ == "__main__":
    main()
