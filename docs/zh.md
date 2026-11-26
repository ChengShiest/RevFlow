# RevFlow 中文说明

论文：[Generation as Reverse Encoding (arXiv:2611.34782)](https://arxiv.org/abs/2611.34782)

用**一个共享的** LightningDiT 主干（HyC-DiT）分两阶段生成 ImageNet 256×256 类条件图像：

- **hi 阶段（语义）**：先生成 16×16 的 DINOv3-L 深层特征（第 24 层，1024 维），类别、时间与阶段嵌入合成全局条件做 AdaLN 调制。
- **lo 阶段（细节）**：再生成 128 维的 codec bottleneck（DINOv3 第 8 层卷积压缩），每个位置的 hi 特征经 `Linear(1024, 1152)` 投影后逐位置做空间 AdaLN 调制。
- **像素解码**：冻结的 89M ViT-B 解码器将 lo 特征解码为 256×256 RGB 图像。
- **内部引导**：两个阶段都在第 8 层读出 base、第 28 层读出 final，采样时在单次前向内完成外推引导（REPA / internal guidance），无需额外前向次数或 CFG。

## 模型库

| 模型 | 参数量 | 训练 | NFE | FID-50k ↓ | 发布文件 |
|---|---:|---|---:|---:|---|
| **RevFlow-XL** | 690M | 5 epoch | 25 + 25 | 1.96 | `stage2/revflow-xl-5ep.pt` |
| **RevFlow-H** | 1.55B | 5 epoch | 25 + 25 | 1.74 | `stage2/revflow-h-5ep.pt` |
| **RevFlow-XL** | 690M | 80 epoch | 25 + 25 | 1.36 | `stage2/revflow-xl-80ep.pt` |
| **RevFlow-H** | 1.55B | 80 epoch | 25 + 25 | **1.17** | `stage2/revflow-h-80ep.pt` |
| **RevFlow-XL + DDT (NM)** | 852M | 80ep + 1k NM | **1 + 1** | **2.90** | `native_matching/revflow-xl-nm.pt` |

NFE 仅统计生成器前向次数，不含冻结解码器；FID-50k 基于 50,000 张样本与 JiT ImageNet-256 参考统计计算（无 CFG）。

## 下载权重

模型权重、codec 和统计文件托管在 [ModelScope](https://www.modelscope.cn/models/chengshi0612/RevFlow)：

```bash
modelscope download --model chengshi0612/RevFlow --include "pretrained_models/*" --local_dir .          # 全部（约 25 GB）
modelscope download --model chengshi0612/RevFlow --include "pretrained_models/stage1/*" --local_dir .   # 仅 codec + stats
```

全部文件均为纯张量格式，支持 `torch.load(..., weights_only=True)` 加载，校验和见 [manifest.json](../pretrained_models/manifest.json)。

## 快速使用

```bash
python -m pip install -e '.[nm,fid,data]'
P=pretrained_models

# 采样（XL-5ep）
revflow-sample --checkpoint $P/stage2/revflow-xl-5ep.pt --codec $P/stage1/codec.pt --stats $P/stage1/stats.pt \
    --classes 207 360 817 --seed 42 --output outputs/samples

# 2 NFE 采样（Native Matching）
revflow-sample --config revflow/configs/revflow_xl_nm_256.yaml \
    --checkpoint $P/native_matching/revflow-xl-nm.pt --codec $P/stage1/codec.pt --stats $P/native_matching/nm-stats.pt \
    --classes 207 360 817 --seed 42 --output outputs/nm

# 5-epoch 标准训练
NNODES=1 GPUS_PER_NODE=8 bash examples/train.sh --data data/imagenet/train --dino-repo third_party/dinov3 \
    --dino-weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --codec $P/stage1/codec.pt --stats $P/stage1/stats.pt --output outputs/xl

# Native Matching 适配训练与 50k FID 评测
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=node0 WEIGHTS=$P bash examples/native_matching.sh
```

H 模型采样/训练加 `--config revflow/configs/revflow_h_256.yaml`；80 epoch 模型采样加 `--precision fp32`，H-80 再加 `--hi-scale 2.5 --lo-scale 2.0 --hi-min-time 0.6`。

## 数据准备（训练与 Native Matching 适配需要）

仅采样无需准备 ImageNet 或 DINOv3。训练和 Native Matching 适配的完整准备步骤见 [data_prep/README.md](../data_prep/README.md)：

1. **ImageNet-1k 训练集**：使用 `data_prep/imagenet.py` 下载并解包 `ILSVRC/imagenet-1k`（或使用标准 ImageFolder）。
2. **DINOv3 与 FID 参考文件**：运行 `DINOV3_URL=... bash data_prep/fetch_assets.sh` 拉取 DINOv3 源码（`94a96ac`）、ViT-L/16 权重及 FID 参考统计。
3. **Native Matching support cache**：基于仓库内置的 51,200 张固定索引（`data_prep/nm_support/`），使用 `revflow-nm-support cache` 构建缓存（约 60.4 GB）。

## 训练配方速览

- **标准 5-epoch**：全局 batch 256（25,020 步），BF16，Muon（0.02→0.001）+ AdamW（3e-4），1 epoch warmup + cosine，EMA 0.9985；hi/lo 时间 shift 为 8 / √8；lo 条件噪声 σ∈[0.1, 0.3]。
- **Native Matching（2 NFE）**：1,000 步，全局 batch 4096（128 类 × 32 张），microbatch 16，GMuon / AdamW 学习率 5e-5 / 2.5e-5，EMA 0.9985；训练引导 hi 2 / lo 2，采样引导 hi 3 / lo 2。

更多详情见 [training.md](training.md)、[native_matching.md](native_matching.md) 和 [data.md](data.md)。

## 引用

```bibtex
@article{shi2026revflow,
  title={Generation as Reverse Encoding},
  author={Shi, Cheng and Yang, Sibei and Guo, Hanzhong and Qin, Yipeng and Lio, Pietro and Yu, Yizhou},
  journal={arXiv preprint arXiv:2611.34782},
  year={2026}
}
```
