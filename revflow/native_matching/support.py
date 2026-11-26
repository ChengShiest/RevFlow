"""Prepare normalized real feature bags and the fixed parent output coordinates."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from revflow.checkpoints import atomic_save, load_weights, read_config
from revflow.data import ImageNet
from revflow.models import RevFlow
from revflow.codec import FeatureEncoder, FlowStats
from .model import readouts


class Support:
    def __init__(self, directory, model, parent_stats=None, source_stats=None):
        directory = Path(directory)
        self.data = np.load(directory / "clean.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(directory / "labels.npy", allow_pickle=False)
        shape = (len(self.labels), model.side ** 2, model.hi_channels + model.lo_channels)
        if self.data.shape != shape or self.labels.ndim != 1 or self.labels.dtype.kind not in "iu":
            raise ValueError("Invalid real support shapes or class IDs")
        self.rows = [np.flatnonzero(self.labels == c) for c in range(model.num_classes)]
        if any(len(rows) == 0 for rows in self.rows) or sum(map(len, self.rows)) != len(self.labels):
            raise ValueError("The support must contain every configured class and no unknown labels")
        self.conversion = []
        if source_stats is not None:
            if parent_stats is None:
                raise ValueError("Specify the parent stats when converting a cache")
            old, new = load_weights(source_stats), load_weights(parent_stats)
            for stage in ("source", "target"):
                mean, std = stage + "_mean", stage + "_std"
                self.conversion.append((old[std] / new[std], (old[mean] - new[mean]) / new[std]))
        self.hi_channels = model.hi_channels

    def get(self, rows, device):
        values = torch.from_numpy(np.array(self.data[rows], copy=True)).to(device=device, dtype=torch.float32)
        if self.conversion:
            hi, lo = values[..., :self.hi_channels], values[..., self.hi_channels:]
            values = torch.cat([v * scale.to(device) + shift.to(device)
                                for v, (scale, shift) in zip((hi, lo), self.conversion)], dim=-1)
        return values, torch.as_tensor(self.labels[rows], device=device, dtype=torch.long)


@torch.inference_mode()
def build_cache(args):
    dataset = ImageNet(args.data)
    labels = np.array([label for _, label in dataset.samples])
    if args.indices:
        indices = np.load(args.indices, allow_pickle=False)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0
                or np.any(indices < 0) or np.any(indices >= len(labels)) or len(np.unique(indices)) != len(indices)):
            raise ValueError("Expected unique ImageFolder row indices within the dataset")
        if args.labels and not np.array_equal(labels[indices], np.load(args.labels, allow_pickle=False)):
            raise ValueError("The supplied support does not match ImageFolder class ordering")
    else:
        if args.labels:
            raise ValueError("--labels is used to validate an explicit --indices file")
        rng = np.random.default_rng(42)
        base, extra = divmod(args.count, len(dataset.classes))
        if base < 1:
            raise ValueError("Select at least one image per class")
        indices = np.concatenate([rng.choice(np.flatnonzero(labels == c), base + int(c < extra), replace=False)
                                  for c in range(len(dataset.classes))])
    device = torch.device(args.device)
    encoder = FeatureEncoder(args.dino_repo, args.dino_weights, load_weights(args.codec)).to(device)
    stats = FlowStats(load_weights(args.stats)).to(device)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, num_workers=args.workers)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        directory = Path(temp) / "cache"
        directory.mkdir()
        output = np.lib.format.open_memmap(directory / "clean.npy", mode="w+", dtype=np.float32,
                                           shape=(len(indices), 256, 1152))
        offset = 0
        for images, _ in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hi, lo = stats.normalize(*encoder(images.to(device)))
            output[offset:offset + len(images)] = torch.cat((hi, lo), dim=-1).cpu().numpy()
            offset += len(images)
        output.flush()
        del output
        np.save(directory / "labels.npy", labels[indices])
        np.save(directory / "indices.npy", indices)
        directory.rename(args.output)


@torch.inference_mode()
def build_chart(args):
    device = torch.device(args.device)
    model = RevFlow(**read_config(args.config)["model"])
    model.load_state_dict(load_weights(args.parent))
    support = Support(args.cache, model, args.stats, args.cache_stats)
    chart, moments = {}, {}
    for name, layer in readouts(model).items():
        weight = layer.weight.detach().to(device, torch.float64)
        _, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        if not (singular > singular.max() * 1e-8).all():
            raise ValueError(f"Rank-deficient {name} final readout")
        chart[name] = {"inverse": torch.linalg.inv(weight @ vh.T).T.float(), "bias": layer.bias.to(device)}
        moments[name] = torch.zeros(2, len(singular), dtype=torch.float64, device=device)
    for start in range(0, len(support.labels), args.batch_size):
        values, _ = support.get(slice(start, start + args.batch_size), device)
        for name, raw in zip(("hi", "lo"), values.split((model.hi_channels, model.lo_channels), -1)):
            coords = ((raw - chart[name]["bias"]) @ chart[name]["inverse"]).double()
            moments[name][0] += coords.sum((0, 1))
            moments[name][1] += coords.square().sum((0, 1))
    count = len(support.labels) * model.side ** 2
    for name, (total, squares) in moments.items():
        mean = total / count
        chart[name].update(mean=mean.float(), std=(squares / count - mean.square()).clamp_min(1e-8).sqrt().float())
    atomic_save({stage: {k: v.cpu() for k, v in value.items()} for stage, value in chart.items()}, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache", help="Encode a fixed ImageFolder support set")
    for key in ("data", "codec", "stats", "dino-repo", "dino-weights"):
        cache.add_argument("--" + key, required=True, type=Path)
    cache.add_argument("--indices", type=Path)
    cache.add_argument("--labels", type=Path)
    cache.add_argument("--count", type=int, default=51200)
    cache.add_argument("--workers", type=int, default=4)
    chart = sub.add_parser("chart", help="Measure parent readout coordinates on the support")
    for key in ("config", "parent", "cache"):
        chart.add_argument("--" + key, required=True, type=Path)
    chart.add_argument("--cache-stats", type=Path)
    chart.add_argument("--stats", type=Path)
    for p in (chart, cache):
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    convert = sub.add_parser("import-chart", help="Pack the original geometry and normalization tensors")
    convert.add_argument("--geometry", type=Path, required=True)
    convert.add_argument("--normalization", type=Path, required=True)
    for p in (chart, cache, convert):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache":
        build_cache(args)
    elif args.command == "chart":
        build_chart(args)
    else:
        geom, norm = load_weights(args.geometry), load_weights(args.normalization)
        state = {s: {"inverse": geom[s]["inverse"], "bias": geom[s]["bias"],
                     "mean": norm[s]["mean"], "std": norm[s]["std"]} for s in ("hi", "lo")}
        atomic_save(state, args.output)


if __name__ == "__main__":
    main()
