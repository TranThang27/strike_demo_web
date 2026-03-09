#!/usr/bin/env bash
# Usage:
#   ./play.sh                  → B_DadDance (default)
#   ./play.sh B_LongDance      → chọn clip khác
#   ./play.sh B_DadDance 4     → 4 envs

CLIP=${1:-B_DadDance}
NUM_ENVS=${2:-1}

DATA_DIR="$(cd "$(dirname "$0")/data_motion/$CLIP" && pwd)"
CKPT="$DATA_DIR/${CLIP}_policy.pt"
NPZ="$DATA_DIR/${CLIP}.npz"

if [ ! -f "$CKPT" ]; then
  echo " Không tìm thấy checkpoint: $CKPT"
  echo "   Các clip có sẵn:"
  ls "$(dirname "$0")/data_motion/"
  exit 1
fi

echo "▶  Clip     : $CLIP"
echo "   Checkpoint: $CKPT"
echo "   Motion    : $NPZ"
echo "   Num envs  : $NUM_ENVS"
echo "   Viser UI  : http://localhost:8080"
echo ""

cd "$(dirname "$0")/mjlab"
MUJOCO_GL=egl .venv/bin/python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file "$CKPT" \
  --motion-file    "$NPZ" \
  --num-envs       "$NUM_ENVS" \
  --viewer viser
