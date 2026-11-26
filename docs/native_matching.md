# Native Matching (2 NFE)

Native matching adapts the 80-epoch EMA of the 852M-parameter **RevFlow-XL/1 + DDT** generator (`nm-parent.pt`) for 1,000 updates so that one `hi` evaluation and one `lo` evaluation from pure noise produce an image (**2 NFE**, FID-50k **2.90** for the released `pretrained_models/native_matching/revflow-xl-nm.pt`).

## Required assets

| File | Contents |
|---|---|
| `pretrained_models/native_matching/nm-parent.pt` | Epoch-80 XL + DDT EMA weights |
| `pretrained_models/stage1/codec.pt` | Frozen Stage 1 bottleneck and pixel decoder |
| `pretrained_models/native_matching/nm-stats.pt` | Flow statistics |
| `pretrained_models/native_matching/nm-chart.pt` | Frozen inverse readout matrices, biases, and support moments |
| `pretrained_models/native_matching/nm-cache-stats.pt` | Cache normalization statistics (identical tensors to `nm-stats.pt`) |
| `data/nm-support/` | 51,200 encoded real images ([build instructions](../data_prep/README.md#5-native-matching-support-cache)) |
| `weights/fid/*` | JiT reference moments and Inception weights (`data_prep/fetch_assets.sh`) |

Install optional dependencies for training and FID evaluation:

```bash
python -m pip install -e '.[nm,fid]'
```

## Recipe

Configuration: [revflow/configs/revflow_xl_nm_256.yaml](../revflow/configs/revflow_xl_nm_256.yaml)

| Setting | Value |
|---|---|
| Updates | 1,000, constant learning rate |
| Global batch | 4,096 per update (128 classes × 32 support images), microbatch 16 |
| Optimizers | GMuon (Polar Express, Nesterov, momentum 0.95, `lr` 5e-5) on 2D matrices; AdamW (`lr` 2.5e-5, β=(0.9, 0.95), wd 0) on remaining parameters; grad clip 1.0 |
| EMA | 0.9985 |
| Guidance | Training: `hi` 2.0 / `lo` 2.0; Sampling: `hi` 3.0 / `lo` 2.0 (no CFG) |
| Distribution gradient | Multi-scale window drifting (2×2, 4×4, 16×16 mean/std) scaled to full-image energy-distance gradient norm, plus `image_marginals` IMQ MMD (hi + lo) |
| Reference | Same-class EMA prediction from real features noised to `t = 0.3` |
| FM regularization | Weight `0.1351080981029292`, shared `t ~ U[0.05, 0.5]`, condition noise `σ ~ U[0.1, 0.3]`, base heads × 0.25 |

The final readout layers stay frozen; matching happens in the normalized inverse-readout chart coordinates. The IMQ coefficient is calibrated on update 1 to half the norm of the drift gradient and held fixed thereafter. See [field.py](../revflow/native_matching/field.py) and [train.py](../revflow/native_matching/train.py).

## Run adaptation

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=node0 WEIGHTS=pretrained_models bash examples/native_matching.sh
```

Or launch training directly:

```bash
CFG=$(python -c 'from revflow.checkpoints import DEFAULT_CONFIG; print(DEFAULT_CONFIG.with_name("revflow_xl_nm_256.yaml"))')
torchrun --standalone --nproc-per-node 8 -m revflow.native_matching.train \
    --config $CFG --parent pretrained_models/native_matching/nm-parent.pt \
    --cache data/nm-support --chart pretrained_models/native_matching/nm-chart.pt \
    --cache-stats pretrained_models/native_matching/nm-cache-stats.pt \
    --stats pretrained_models/native_matching/nm-stats.pt \
    --output outputs/nm/train
```

The process count must divide 128. `latest.pt` is written every 250 updates and at the end; resume by replacing `--parent` with `--resume outputs/nm/train/latest.pt` (same config and process count). Use `--stop-after N` to stop early at update `N`.

### Custom support set or chart

If you build a custom support subset (omitting `--indices/--labels` in `revflow-nm-support cache`), compute a matching chart before training:

```bash
revflow-nm-support chart --config $CFG \
    --parent pretrained_models/native_matching/nm-parent.pt \
    --cache data/nm-support --output weights/new-chart.pt
```

## Sampling

```bash
revflow-sample --config revflow/configs/revflow_xl_nm_256.yaml \
    --checkpoint pretrained_models/native_matching/revflow-xl-nm.pt \
    --codec pretrained_models/stage1/codec.pt \
    --stats pretrained_models/native_matching/nm-stats.pt \
    --classes 207 360 817 --seed 42 --output outputs/nm-samples
```

## FID-50k evaluation

Protocol: 4 sampling processes, batch size 25 per process, CPU seed `4321 + rank`, uniformly random classes, BF16 generator, FP32 decoder, uint8 output, evaluated against JiT ImageNet-256 reference moments.

```bash
torchrun --standalone --nproc-per-node 4 -m revflow.native_matching.evaluate generate \
    --config revflow/configs/revflow_xl_nm_256.yaml \
    --checkpoint pretrained_models/native_matching/revflow-xl-nm.pt \
    --codec pretrained_models/stage1/codec.pt \
    --stats pretrained_models/native_matching/nm-stats.pt \
    --count 50000 --batch-size 25 --seed 4321 --output outputs/nm/fid50k

revflow-nm-evaluate score --output outputs/nm/fid50k \
    --reference weights/fid/jit_in256_stats.npz \
    --inception weights/fid/weights-inception-2015-12-05-6726825d.pth
```

Use a fresh output directory for each run (50,000 uint8 images occupy 9.2 GiB). Results are saved to `fid.json`.
