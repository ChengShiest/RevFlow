"""Export tensor-only checkpoints or compute DINOv3 feature statistics."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from revflow.checkpoints import atomic_save, load_weights
from revflow.data import ImageNet
from revflow.codec import FeatureEncoder, FlowStats


def export(args):
    source = torch.load(args.input, map_location="cpu", weights_only=not args.trusted_legacy, mmap=True)
    if args.kind == "codec":
        state = source.get("ema", source.get("model", source))
        result = {}
        for module, prefixes in (("bottleneck", ("encoder.bottleneck.", "bottleneck.")), ("decoder", ("decoder.",))):
            for prefix in prefixes:
                weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                if weights:
                    break
            if not weights:
                raise ValueError(f"Missing {module} in Stage 1 checkpoint")
            result[module] = weights
    elif args.kind == "generator":
        if args.ema_decay is not None:
            result = source["ema_gens"][args.ema_decay]
        elif "ema_model" in source:
            result = source["ema_model"]
        elif "ema_gen" in source:
            result = source["ema_gen"]
        else:
            result = source.get("ema", source)
        if result and all(key.startswith("net.") for key in result):
            result = {key[4:]: value for key, value in result.items()}
        if "condition_projection.weight" not in result or not all(torch.is_tensor(v) for v in result.values()):
            raise ValueError("Expected RevFlow generator weights or a checkpoint containing its EMA")
    else:
        result = {key: value.cpu() for key, value in FlowStats(source).state_dict().items()}
    atomic_save(result, args.output)


@torch.inference_mode()
def statistics(args):
    device = torch.device(args.device)
    encoder = FeatureEncoder(args.dino_repo, args.dino_weights, load_weights(args.codec)).to(device)
    dataset = ImageNet(args.data)
    if args.indices is not None:
        if args.limit is not None:
            raise ValueError("Use either --indices or --limit")
        indices = np.load(args.indices, allow_pickle=False)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not 0 <= indices.min() <= indices.max() < len(dataset):
            raise ValueError("Expected ImageFolder indices within the dataset")
        dataset = Subset(dataset, indices.tolist())
    elif args.limit is not None:
        if not 1 <= args.limit <= len(dataset):
            raise ValueError("The sample count must fit the dataset")
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[:args.limit]
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    sums, squares = {}, {}
    count = 0
    for images, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features = encoder(images.to(device))
        for name, feature in zip(("source", "target"), features):
            values = feature.double()
            if name not in sums:
                sums[name] = torch.zeros_like(values[0])
                squares[name] = torch.zeros_like(values[0])
            sums[name] += values.sum(0)
            squares[name] += values.square().sum(0)
        count += len(images)
    result = {}
    for name, total in sums.items():
        mean = total / count
        variance = (squares[name] / count - mean.square()).clamp_min(0)
        result[name + "_mean"] = mean.float().cpu()
        result[name + "_std"] = (variance + 1e-5).sqrt().float().cpu()
    atomic_save(result, args.output)
    print(f"Computed per-position feature statistics over {count} images.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("export")
    convert.add_argument("kind", choices=["codec", "generator", "stats"])
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--ema-decay", help="Select an exact key from a multi-EMA generator checkpoint")
    convert.add_argument("--trusted-legacy", action="store_true",
                         help="Allow Python pickle in a trusted original training checkpoint")
    stats = commands.add_parser("stats")
    for name in ("data", "dino-repo", "dino-weights", "codec", "output"):
        stats.add_argument("--" + name, type=Path, required=True)
    stats.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    stats.add_argument("--batch-size", type=int, default=16)
    stats.add_argument("--workers", type=int, default=4)
    stats.add_argument("--limit", type=int)
    stats.add_argument("--indices", type=Path, help="ImageFolder indices of the images to use (.npy)")
    args = parser.parse_args()
    if args.command == "export":
        if args.input.resolve() == args.output.resolve():
            parser.error("Choose a separate output file for export")
        export(args)
    else:
        statistics(args)


if __name__ == "__main__":
    main()
