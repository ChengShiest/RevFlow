# Model Weights and Shared Assets

Pretrained generators, the frozen Stage 1 codec, native-matching assets, and flow statistics are hosted on [ModelScope](https://www.modelscope.cn/models/chengshi0612/RevFlow) (~25 GB total).

To download RevFlow-XL (5 epochs) and its shared assets from the repository root:

```bash
python -m pip install modelscope
modelscope download --model chengshi0612/RevFlow \
  --include "pretrained_models/stage1/*" "pretrained_models/stage2/revflow-xl-5ep.pt" \
  --local_dir .
```

See the [Model Zoo](../README.md#model-zoo) and [manifest.json](manifest.json) for all checkpoints, file sizes, and SHA-256 digests.
