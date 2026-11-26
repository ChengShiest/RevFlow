# Preparing Data and Assets

Step-by-step setup for training, native matching, and FID evaluation. Nothing is downloaded automatically at runtime.

Download released checkpoints, codec, and statistics from ModelScope first:

```bash
modelscope download --model chengshi0612/RevFlow --include "pretrained_models/*" --local_dir .
```

| # | Asset | Source | Size | Used for |
|---|---|---|---:|---|
| 1 | ImageNet-1k train | Hugging Face `ILSVRC/imagenet-1k` (gated) | 146.9 GB | Training, flow stats, NM support |
| 2 | DINOv3 source | `facebookresearch/dinov3` (`94a96ac`) | 10 MB | Training, flow stats, NM support |
| 3 | DINOv3 ViT-L/16 | Meta (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`) | 1.2 GB | Training, flow stats, NM support |
| 4 | Stage 1 codec | `pretrained_models/stage1/codec.pt` | 367 MB | All workflows |
| 5 | Flow statistics | `pretrained_models/stage1/stats.pt` | 2.4 MB | All workflows |
| 6 | NM parent + chart | `pretrained_models/native_matching/nm-{parent,chart}.pt` | 3.4 GB | Native matching |
| 7 | NM support cache | Built from 1–5 ([rows.npy](nm_support/rows.npy)) | 60.4 GB | Native matching |
| 8 | FID reference + Inception | JiT + torch-fidelity (`fetch_assets.sh`) | 125 MB | 2-NFE FID |

Install the optional extras before running these scripts:

```bash
python -m pip install -e '.[data,fid]'
```

## 1. ImageNet-1k train split

`imagenet.py` converts the Hugging Face `ILSVRC/imagenet-1k` parquet shards into an ImageFolder tree (`train/<label:04d>/<row:07d>.JPEG`), preserving parquet row numbers so fixed subsets can be remapped cleanly.

```bash
huggingface-cli login                                    # accept ImageNet terms on HF first
python data_prep/imagenet.py download --root raw/imagenet-1k
python data_prep/imagenet.py extract --parquet raw/imagenet-1k/data --out data/imagenet
python data_prep/imagenet.py verify --out data/imagenet
```

`verify` checks row count, labels, per-image byte lengths, and row-0 SHA-256 against [expected/imagenet_train.json](expected/imagenet_train.json). Standard ImageFolder directories extracted from `ILSVRC2012_img_train.tar` also work directly for training.

## 2–3. DINOv3 and FID assets

Request the signed URL for `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` from <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>, then run:

```bash
DINOV3_URL='<signed URL from Meta>' bash data_prep/fetch_assets.sh
```

This clones DINOv3 at commit `94a96ac` into `third_party/dinov3`, downloads the ViT-L/16 weights, downloads the JiT FID reference and Inception weights into `weights/fid/`, and verifies SHA-256 checksums. Set `SKIP_DINOV3=1` if you only need the FID assets.

## 4. Flow statistics (optional rebuild)

Use the released `pretrained_models/stage1/stats.pt` with released generators. To recompute flow statistics from ImageNet rows 0–49,999:

```bash
python data_prep/imagenet.py remap --out data/imagenet --range 0 50000 --output data/imagenet/flowstats-indices.npy
revflow-assets stats --data data/imagenet/train --indices data/imagenet/flowstats-indices.npy \
    --dino-repo third_party/dinov3 --dino-weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --codec pretrained_models/stage1/codec.pt --output weights/stats.pt
```

## 5. Native-matching support cache

Build the 51,200-image support cache (`data/nm-support/{clean,labels,indices}.npy`, ~60.4 GB) from the shipped fixed row indices:

```bash
python data_prep/imagenet.py remap --out data/imagenet --rows data_prep/nm_support/rows.npy \
    --output data/imagenet/nm-support-indices.npy

revflow-nm-support cache --data data/imagenet/train \
    --dino-repo third_party/dinov3 --dino-weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --codec pretrained_models/stage1/codec.pt --stats pretrained_models/native_matching/nm-stats.pt \
    --indices data/imagenet/nm-support-indices.npy --labels data_prep/nm_support/labels.npy \
    --output data/nm-support
```

Omit `--indices/--labels` to draw a new class-stratified subset (then compute a matching chart with `revflow-nm-support chart`, see [docs/native_matching.md](../docs/native_matching.md)).

## Verify checksums

```bash
sha256sum -c data_prep/expected/SHA256SUMS --ignore-missing
python data_prep/imagenet.py verify --out data/imagenet
```
