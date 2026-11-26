#!/usr/bin/env bash
# Fetch the public third-party assets RevFlow needs, at the versions used by the original runs,
# and check them against expected/SHA256SUMS. Run from the repository root:
#   DINOV3_URL='<signed URL from Meta>' bash data_prep/fetch_assets.sh
# Layout produced (all paths relative to the repository root):
#   third_party/dinov3/                                   DINOv3 source at 94a96ac
#   weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth  DINOv3 ViT-L/16 LVD-1689M backbone
#   weights/fid/jit_in256_stats.npz                       JiT ImageNet-256 FID reference moments
#   weights/fid/weights-inception-2015-12-05-6726825d.pth torch-fidelity Inception weights
# Set SKIP_DINOV3=1 if you only sample or run NM from an existing support cache.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p third_party weights/fid

DINOV3_COMMIT=94a96ac83c2446f15f9bdcfae23cad3c6a9d4988
JIT_COMMIT=cbc743a2ada5e9762697da2c83f8c4f8379e8c17

if [[ ${SKIP_DINOV3:-0} != 1 ]]; then
  if [[ ! -d third_party/dinov3/.git ]]; then
    git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
  fi
  git -C third_party/dinov3 checkout --quiet "$DINOV3_COMMIT"
  # The backbone weights are distributed by Meta after accepting the DINOv3 license:
  # request access at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
  # and pass the emailed URL for dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth as DINOV3_URL.
  if [[ ! -f weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth ]]; then
    : "${DINOV3_URL:?set DINOV3_URL to the download URL of dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
    curl -L --fail -o weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth "$DINOV3_URL"
  fi
fi

# FID reference: the ImageNet 256x256 training-set moments shipped with JiT.
[[ -f weights/fid/jit_in256_stats.npz ]] || curl -L --fail -o weights/fid/jit_in256_stats.npz \
  "https://github.com/LTH14/JiT/raw/$JIT_COMMIT/fid_stats/jit_in256_stats.npz"
# Inception weights used by torch-fidelity (the URL hard-coded in torch-fidelity itself).
[[ -f weights/fid/weights-inception-2015-12-05-6726825d.pth ]] || curl -L --fail \
  -o weights/fid/weights-inception-2015-12-05-6726825d.pth \
  "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"

# Check everything that is present.
grep -E 'dinov3_vitl16|weights/fid/' "$here/expected/SHA256SUMS" | while read -r sum path; do
  [[ -f $path ]] && echo "$sum  $path"
done | sha256sum -c -
