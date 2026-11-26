#!/usr/bin/env bash
# Native matching (2 NFE): adapt the epoch-80 DDT parent for 1,000 updates, then sample 50k images and score FID.
# Run once on every node (NNODES, GPUS_PER_NODE, NODE_RANK, MASTER_ADDR as in examples/train.sh).
# Re-running the script resumes training from $OUT/train/latest.pt if it exists.
#   NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=node0 bash examples/native_matching.sh
# Paths (override with environment variables):
#   WEIGHTS  pretrained_models/ as downloaded from ModelScope (stage1/codec.pt, native_matching/nm-*.pt),
#            or a flat directory with codec.pt, nm-parent.pt, nm-chart.pt, nm-stats.pt, nm-cache-stats.pt
#   CACHE    support cache directory (clean.npy, labels.npy), see data_prep/README.md
#   FID_REF  JiT ImageNet-256 reference moments, INCEPTION torch-fidelity Inception weights
set -euo pipefail
: "${NNODES:?}" "${GPUS_PER_NODE:?}"
WEIGHTS=${WEIGHTS:-pretrained_models}
if [[ -d $WEIGHTS/native_matching ]]; then CODEC=$WEIGHTS/stage1/codec.pt; NM=$WEIGHTS/native_matching; else CODEC=$WEIGHTS/codec.pt; NM=$WEIGHTS; fi
CACHE=${CACHE:-data/nm-support}
OUT=${OUT:-outputs/nm}
FID_REF=${FID_REF:-weights/fid/jit_in256_stats.npz}
INCEPTION=${INCEPTION:-weights/fid/weights-inception-2015-12-05-6726825d.pth}
config=$(python -c 'from revflow.checkpoints import DEFAULT_CONFIG; print(DEFAULT_CONFIG.with_name("revflow_xl_nm_256.yaml"))')
if [[ $NNODES == 1 ]]; then
  rendezvous=(--standalone)
else
  rendezvous=(--node-rank "${NODE_RANK:?}" --master-addr "${MASTER_ADDR:?}" --master-port "${MASTER_PORT:-29500}")
fi
source=(--parent "$NM/nm-parent.pt")
[[ -f $OUT/train/latest.pt ]] && source=(--resume "$OUT/train/latest.pt")
torchrun --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE" "${rendezvous[@]}" -m revflow.native_matching.train \
  --config "$config" "${source[@]}" --cache "$CACHE" --chart "$NM/nm-chart.pt" \
  --cache-stats "$NM/nm-cache-stats.pt" --stats "$NM/nm-stats.pt" --output "$OUT/train"
[[ ${NODE_RANK:-0} == 0 ]] || exit 0
# Four sampling processes (seed 4321 + rank).
torchrun --standalone --nproc-per-node 4 -m revflow.native_matching.evaluate generate \
  --config "$config" --checkpoint "$OUT/train/latest.pt" --codec "$CODEC" \
  --stats "$NM/nm-stats.pt" --count 50000 --batch-size 25 --seed 4321 --output "$OUT/fid50k"
python -m revflow.native_matching.evaluate score --output "$OUT/fid50k" --reference "$FID_REF" --inception "$INCEPTION"
