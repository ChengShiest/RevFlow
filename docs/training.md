# Training

## Standard 5-epoch training

```bash
NNODES=1 GPUS_PER_NODE=8 bash examples/train.sh --data data/imagenet/train \
    --dino-repo third_party/dinov3 --dino-weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --codec pretrained_models/stage1/codec.pt --stats pretrained_models/stage1/stats.pt --output outputs/xl
```

`examples/train.sh` launches `torchrun -m revflow.train`. For multi-node runs, also set `NODE_RANK`, `MASTER_ADDR`, and optionally `MASTER_PORT`. Pass `--config revflow/configs/revflow_h_256.yaml` to train RevFlow-H.

Recipe ([revflow_xl_256.yaml](../revflow/configs/revflow_xl_256.yaml)):

| Setting | Value |
|---|---|
| Data | ImageNet-1k train, ADM center crop to 256, random horizontal flip |
| Targets | Frozen DINOv3-L/16 + Stage 1 codec bottleneck, normalized with `stats.pt` |
| Batch size | Global 256 (5,004 updates/epoch, 5 epochs = 25,020 updates) |
| Optimizers | Muon (`lr` 0.02 → 0.001) on trunk matrices; AdamW (`lr` 3e-4, β=(0.9, 0.95), wd 0.01) on embeddings, norms, modulation, and conditioning |
| Schedule | 1-epoch linear warmup, cosine decay to final LR |
| Precision | BF16 autocast, BF16 gradient all-reduce, grad-norm clip 1.0 |
| EMA | 0.9985, updated every step |
| Time sampling | Independent logit-normal `t`, shifted by 8 (`hi`) and √8 (`lo`) |
| `lo` conditioning | Variance-preserving noise on ground-truth `hi`, `σ ~ U[0.1, 0.3]` |
| Loss | `x`-prediction error divided by `max(t, 0.05)`: `final_hi + 0.5 * final_lo + 0.5 * (2 * base_hi + base_lo)` |

### Checkpointing and export

`outputs/xl/latest.pt` is saved every 500 updates and at each epoch boundary, storing live weights, EMA weights, optimizer states, step counter, and per-rank RNG states. Resume with `--resume outputs/xl/latest.pt` using the same config and process count.

Export a standalone tensor-only EMA checkpoint:

```bash
revflow-assets export generator outputs/xl/latest.pt outputs/xl/generator.pt
```

`revflow-sample` can also read `latest.pt` directly.

## Native matching (2 NFE)

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=<0|1> MASTER_ADDR=<node0> WEIGHTS=pretrained_models bash examples/native_matching.sh
```

Adapts `nm-parent.pt` for 1,000 updates, then runs 50k sampling and FID scoring on node 0. See [docs/native_matching.md](native_matching.md) for all flags and options.
