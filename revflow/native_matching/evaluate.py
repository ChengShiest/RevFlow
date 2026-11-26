"""Sample the NM EMA at 2 NFE and score uint8 images against JiT ImageNet moments."""

import argparse
import json
import os
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from revflow.checkpoints import load_weights, read_config
from revflow.models import RevFlow
from revflow.codec import FlowStats, ImageDecoder
from .model import guided_pair


@torch.inference_mode()
def generate(args, device, rank, world):
    config = read_config(args.config)
    model = RevFlow(**config["model"])
    weights = load_weights(args.checkpoint)
    model.load_state_dict(weights.get("ema", weights))
    del weights
    model.to(device).eval()
    decoder = ImageDecoder(load_weights(args.codec)).to(device)
    stats = FlowStats(load_weights(args.stats)).to(device)
    count = args.count // world
    rng = torch.Generator().manual_seed(args.seed + rank)
    temporary = args.output / f"rank-{rank}.partial.npy"
    images = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(count, 256, 256, 3))
    for start in range(0, count, args.batch_size):
        n = min(args.batch_size, count - start)
        noise = torch.randn(n, model.side ** 2, model.hi_channels + model.lo_channels, generator=rng).to(device)
        labels = torch.randint(model.num_classes, (n,), generator=rng).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = guided_pair(model, noise, torch.ones(n, device=device), labels, **config["sampling"])
        pixels = decoder(stats.decode_lo(raw[..., model.hi_channels:]))
        images[start:start + n] = pixels.mul(255).clamp_(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    images.flush()
    del images
    temporary.replace(args.output / f"rank-{rank}.npy")


@torch.inference_mode()
def score(directory, reference, inception, device):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    run = json.loads((directory / "sampling.json").read_text())
    extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"],
                  feature_extractor_weights_path=str(inception), feature_extractor_internal_dtype="float32")
    extractor.to(device).eval()
    torch.backends.cudnn.allow_tf32 = False
    features = torch.empty(run["count"], 2048)
    offset = 0
    for rank in range(run["world_size"]):
        images = np.load(directory / f"rank-{rank}.npy", mmap_mode="r", allow_pickle=False)
        if images.shape != (run["count"] // run["world_size"], 256, 256, 3) or images.dtype != np.uint8:
            raise ValueError("Incomplete evaluation shard")
        for i in range(0, len(images), 128):
            batch = torch.from_numpy(np.array(images[i:i + 128])).permute(0, 3, 1, 2).contiguous().to(device)
            features[offset:offset + len(batch)] = extractor(batch)[0].cpu()
            offset += len(batch)
    generated = fid_features_to_statistics(features)
    with np.load(reference, allow_pickle=False) as asset:
        real = {k: asset[k] for k in ("mu", "sigma")}
    for key in real:
        dtype = np.result_type(real[key].dtype, generated[key].dtype)
        real[key], generated[key] = real[key].astype(dtype), generated[key].astype(dtype)
    fid = fid_statistics_to_metric(generated, real, False)["frechet_inception_distance"]
    result = dict(run, fid=float(fid))
    (directory / "fid.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    for key in ("config", "checkpoint", "codec", "stats"):
        gen.add_argument("--" + key, required=True, type=Path)
    gen.add_argument("--count", type=int, default=50000)
    gen.add_argument("--batch-size", type=int, default=25)
    gen.add_argument("--seed", type=int, default=4321)
    metric = commands.add_parser("score")
    metric.add_argument("--reference", type=Path, required=True)
    metric.add_argument("--inception", type=Path, required=True)
    for p in (gen, metric):
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    if args.command == "score":
        if world != 1:
            parser.error("Run the scoring command on one process")
        return score(args.output, args.reference, args.inception, device)
    if args.count < world or args.count % world or args.batch_size < 1:
        parser.error("Image count must divide evenly across processes and batch size must be positive")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()
        generate(args, device, rank, world)
        if world > 1:
            dist.barrier()
        if rank == 0:
            metadata = dict(count=args.count, world_size=world, batch_size=args.batch_size, seed=args.seed,
                            total_nfe=2, device=device.type, config=read_config(args.config),
                            checkpoint=args.checkpoint.name, codec=args.codec.name, stats=args.stats.name)
            (args.output / "sampling.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
