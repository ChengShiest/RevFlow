"""Rebuild the ImageNet-1k training split exactly as the original RevFlow runs read it.

The original runs read the Hugging Face parquet release of ImageNet-1k (ILSVRC/imagenet-1k) in
parquet row order: shards sorted by name, rows in file order. The native-matching support indices and the
flow-statistics subset refer to that row order. RevFlow reads an ImageFolder tree, so this script
writes every encoded image unchanged to <out>/train/<label:04d>/<row:07d>.JPEG. Sorted class
directories then give the original label IDs, and remap() converts parquet row numbers into the
ImageFolder indices used by revflow.data.ImageNet.

  python data_prep/imagenet.py download --root raw/imagenet-1k
  python data_prep/imagenet.py extract --parquet raw/imagenet-1k/data --out data/imagenet
  python data_prep/imagenet.py verify --out data/imagenet
  python data_prep/imagenet.py remap --out data/imagenet --rows data_prep/nm_support/rows.npy \
      --output data/imagenet/nm-support-indices.npy
  python data_prep/imagenet.py remap --out data/imagenet --range 0 50000 --output data/imagenet/flowstats-indices.npy
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO = "ILSVRC/imagenet-1k"
REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
EXPECTED = json.loads((Path(__file__).parent / "expected" / "imagenet_train.json").read_text())


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<i8").tobytes()).hexdigest()


def download(args):
    from huggingface_hub import snapshot_download  # pip install 'revflow[data]'
    # The dataset is gated: accept the ImageNet terms on its Hugging Face page and run
    # `huggingface-cli login` (or set HF_TOKEN) first. HF_ENDPOINT selects a mirror.
    path = snapshot_download(REPO, repo_type="dataset", revision=args.revision, local_dir=args.root,
                             allow_patterns=["data/train-*.parquet", "classes.py", "README.md"],
                             max_workers=args.workers)
    shards = sorted(Path(path, "data").glob("train-*.parquet"))
    print(f"{len(shards)} train shards in {path}/data (expected {EXPECTED['shards']})")


def extract(args):
    import pyarrow.parquet as pq  # pip install 'revflow[data]'
    shards = sorted(Path(args.parquet).glob("train-*.parquet"))
    if len(shards) != EXPECTED["shards"]:
        raise SystemExit(f"Expected {EXPECTED['shards']} train shards, found {len(shards)} in {args.parquet}")
    root = Path(args.out) / "train"
    for label in range(1000):
        (root / f"{label:04d}").mkdir(parents=True, exist_ok=True)
    labels, lengths = [], []
    row = 0
    for number, shard in enumerate(shards):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1024, columns=["image", "label"]):
            for record, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                payload = record["bytes"]
                path = root / f"{int(label):04d}" / f"{row:07d}.JPEG"
                if not (args.resume and path.exists() and path.stat().st_size == len(payload)):
                    temporary = path.with_suffix(".partial")
                    temporary.write_bytes(payload)
                    os.replace(temporary, path)
                labels.append(int(label))
                lengths.append(len(payload))
                row += 1
        print(f"{number + 1}/{len(shards)} shards, {row} images", flush=True)
    np.savez(Path(args.out) / "train_rows.npz", labels=np.array(labels, dtype=np.int64),
             lengths=np.array(lengths, dtype=np.int64))
    check(Path(args.out), scan=False)


def check(out, scan):
    rows = np.load(out / "train_rows.npz")
    labels, lengths = rows["labels"], rows["lengths"]
    found = dict(images=int(len(labels)), labels_sha256=digest(labels), lengths_sha256=digest(lengths),
                 total_bytes=int(lengths.sum()))
    failures = [key for key, value in found.items() if EXPECTED[key] != value]
    if scan:
        root = out / "train"
        for label in range(1000):
            names = sorted(p.name for p in (root / f"{label:04d}").iterdir() if p.suffix == ".JPEG")
            want = [f"{r:07d}.JPEG" for r in np.flatnonzero(labels == label)]
            if names != want:
                failures.append(f"files in class {label:04d}")
        first = (root / f"{labels[0]:04d}" / "0000000.JPEG").read_bytes()
        if hashlib.sha256(first).hexdigest() != EXPECTED["row0_sha256"]:
            failures.append("row 0 bytes")
    print(json.dumps(found, indent=2))
    if failures:
        raise SystemExit("Differs from the original training split: " + ", ".join(failures))
    print("Matches the original training split (row order, labels, byte lengths).")


def remap(args):
    labels = np.load(Path(args.out) / "train_rows.npz")["labels"]
    if args.rows is not None:
        rows = np.load(args.rows, allow_pickle=False).astype(np.int64)
    else:
        rows = np.arange(*args.range, dtype=np.int64)
    if rows.min() < 0 or rows.max() >= len(labels):
        raise SystemExit("Row numbers fall outside the training split")
    # revflow.data.ImageNet orders by class directory, then file name, i.e. by (label, row).
    position = np.empty(len(labels), dtype=np.int64)
    position[np.argsort(labels, kind="stable")] = np.arange(len(labels))
    np.save(args.output, position[rows])
    print(f"Wrote {len(rows)} ImageFolder indices to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download", help="Fetch the 294 train parquet shards (about 146 GB)")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--workers", type=int, default=8)
    p = commands.add_parser("extract", help="Write the ImageFolder tree and train_rows.npz")
    p.add_argument("--parquet", type=Path, required=True, help="Directory holding train-*.parquet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Keep files that already have the right size")
    p = commands.add_parser("verify", help="Compare with the original row order, labels and byte lengths")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quick", action="store_true", help="Check train_rows.npz only, skip the file scan")
    p = commands.add_parser("remap", help="Parquet row numbers -> revflow ImageFolder indices")
    p.add_argument("--out", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--rows", type=Path, help=".npy of parquet row numbers")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "STOP"))
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(args)
    elif args.command == "extract":
        extract(args)
    elif args.command == "verify":
        check(args.out, scan=not args.quick)
    else:
        remap(args)


if __name__ == "__main__":
    main()
