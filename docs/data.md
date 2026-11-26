# Data and Assets

Overview of the datasets, pretrained assets, and file formats used by RevFlow. For step-by-step setup commands, see [data_prep/README.md](../data_prep/README.md).

| Asset | Location / Command | Notes |
|---|---|---|
| ImageNet-1k train | `data_prep/imagenet.py` | Gated Hugging Face dataset (`ILSVRC/imagenet-1k`) or standard ImageFolder |
| DINOv3 source + ViT-L/16 | `data_prep/fetch_assets.sh` | Pinned commit `94a96ac`; weights from Meta under DINOv3 License |
| Stage 1 codec | `pretrained_models/stage1/codec.pt` | Released on ModelScope |
| Flow statistics | `pretrained_models/stage1/stats.pt` | Released on ModelScope; rebuildable via `revflow-assets stats` |
| Generators (XL / H / NM) | `pretrained_models/stage2/`, `native_matching/` | Released on ModelScope |
| Native-matching parent & chart | `pretrained_models/native_matching/nm-{parent,chart}.pt` | Released on ModelScope (`nm-chart.pt` rebuildable via `revflow-nm-support chart`) |
| Native-matching support cache | `revflow-nm-support cache` | 51,200 encoded images (~60.4 GB) built from fixed indices |
| FID reference & Inception | `data_prep/fetch_assets.sh` | JiT reference moments + torch-fidelity Inception weights |

## ImageNet

Training, flow statistics, and support caching read ImageNet-1k `train/` in standard ImageFolder layout (1,000 sorted class folders). Images are center-cropped to 256×256 with the ADM box-halving + bicubic resize procedure and normalized with ImageNet mean/std.

Using `data_prep/imagenet.py` with Hugging Face `ILSVRC/imagenet-1k` parquet shards writes `train/<label:04d>/<row:07d>.JPEG`, preserving parquet row order so fixed subsets can be remapped directly to ImageFolder indices:

| Subset | Parquet rows | Used by |
|---|---|---|
| Flow statistics | Rows `0..49,999` | `stats.pt` |
| Native-matching support | 51,200 rows in [data_prep/nm_support/rows.npy](../data_prep/nm_support/rows.npy) (51–52 per class) | `data/nm-support`, `nm-chart.pt` |

Any standard ImageNet-1k `train/` ImageFolder (e.g., from `ILSVRC2012_img_train.tar`) works for training; only the fixed subsets above rely on parquet row numbering.

## DINOv3

RevFlow uses `facebookresearch/dinov3` at commit `94a96ac83c2446f15f9bdcfae23cad3c6a9d4988` with `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (SHA-256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`). The frozen encoder runs in BF16 and extracts `get_intermediate_layers(n=[7, 23], norm=False)`: block 8 feeds the `lo` codec bottleneck, and block 24 followed by affine-free LayerNorm forms the `hi` target.

## Stage 1 codec

`pretrained_models/stage1/codec.pt` stores two tensor dictionaries:
- `bottleneck`: 3×3 convolutional encoder (`1024 → 256 → 128`) and decoder (`128 → 256 → 1024`) with LayerNorms.
- `decoder`: 12-block ViT-B pixel decoder (width 768, patch size 16) mapping 16×16×1024 features to 256×256 RGB.

## Flow statistics

`stats.pt` stores per-position (256 tokens) mean and standard deviation (`std = sqrt(var + 1e-5)`) for `hi` (1024 ch) and `lo` (128 ch), computed over the first 50,000 training images. All released generators and native-matching assets share the same statistics tensors (`stage1/stats.pt`, `native_matching/nm-stats.pt`, and `native_matching/nm-cache-stats.pt`).

To recompute statistics for a custom dataset or codec:

```bash
python data_prep/imagenet.py remap --out data/imagenet --range 0 50000 --output data/imagenet/flowstats-indices.npy
revflow-assets stats --data data/imagenet/train --indices data/imagenet/flowstats-indices.npy \
    --dino-repo third_party/dinov3 --dino-weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --codec pretrained_models/stage1/codec.pt --output weights/stats.pt
```

## Native-matching assets

| File | Contents |
|---|---|
| `nm-parent.pt` | 852M XL + DDT epoch-80 EMA weights |
| `nm-chart.pt` | Per-stage `inverse`, `bias`, `mean`, `std` of the frozen final readout coordinates |
| `nm-stats.pt` / `nm-cache-stats.pt` | Flow normalization statistics |
| `data/nm-support/` | `clean.npy` (`float32 [51200, 256, 1152]`), `labels.npy`, `indices.npy` |

## FID reference

| File | Source | SHA-256 |
|---|---|---|
| `jit_in256_stats.npz` | JiT `cbc743a2ada5e9762697da2c83f8c4f8379e8c17` | `412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c` |
| `weights-inception-2015-12-05-6726825d.pth` | torch-fidelity v0.2.0 | `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2` |
| `torch-fidelity` | `LTH14/torch-fidelity@bfe72b995473e4fee9351705d7b005f17442a4ca` (`pip install -e '.[fid]'`) | — |
