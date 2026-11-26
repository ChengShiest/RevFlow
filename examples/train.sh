#!/usr/bin/env bash
# Standard RevFlow-XL/1, 5 epochs. Run once on every node; torchrun starts one worker per GPU.
#   NNODES=1 GPUS_PER_NODE=8 bash examples/train.sh --data data/imagenet/train \
#     --dino-repo third_party/dinov3 --dino-weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
#     --codec pretrained_models/stage1/codec.pt --stats pretrained_models/stage1/stats.pt --output outputs/xl
# Global batch is 256 for any process count that divides it. Resume with the same command plus
#   --resume outputs/xl/latest.pt (same data, assets, config and process count).
set -euo pipefail
: "${NNODES:?set the total number of nodes}" "${GPUS_PER_NODE:?set the local GPU count}"
if [[ $NNODES == 1 ]]; then
  rendezvous=(--standalone)
else
  rendezvous=(--node-rank "${NODE_RANK:?}" --master-addr "${MASTER_ADDR:?}" --master-port "${MASTER_PORT:-29500}")
fi
exec torchrun --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE" "${rendezvous[@]}" -m revflow.train "$@"
