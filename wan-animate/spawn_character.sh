#!/usr/bin/env bash
# Put a character (from one image) into a video with Wan2.2-Animate-14B.
#   --mode replace : swap the person in the video for your character (keeps the scene)
#   --mode animate : your character performs the video's motion (character image's background)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODE=replace VIDEO="" IMAGE="" OUT="" RES_W=1280 RES_H=720 EXTRA=()

usage() {
  echo "Usage: $0 --mode replace|animate --video in.mp4 --image character.png [--out out.mp4] [--res 1280x720] [--multi-gpu N]"
  exit 1
}

NGPU=1
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --video) VIDEO="$(realpath "$2")"; shift 2 ;;
    --image) IMAGE="$(realpath "$2")"; shift 2 ;;
    --out) OUT="$(realpath -m "$2")"; shift 2 ;;
    --res) RES_W="${2%x*}"; RES_H="${2#*x}"; shift 2 ;;
    --multi-gpu) NGPU="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$VIDEO" ] && [ -n "$IMAGE" ] || usage
[ "$MODE" = replace ] || [ "$MODE" = animate ] || usage

CKPT="$HERE/Wan2.2-Animate-14B"
[ -d "$CKPT" ] && [ -d "$HERE/Wan2.2" ] || { echo "Run ./setup.sh first." >&2; exit 1; }

NAME="$(basename "${VIDEO%.*}")_$(basename "${IMAGE%.*}")_$MODE"
WORK="$HERE/work/$NAME"
OUT="${OUT:-$HERE/outputs/$NAME.mp4}"
mkdir -p "$WORK" "$(dirname "$OUT")"

cd "$HERE/Wan2.2"

echo "==> Preprocessing ($MODE)"
PRE=(python ./wan/modules/animate/preprocess/preprocess_data.py
  --ckpt_path "$CKPT/process_checkpoint"
  --video_path "$VIDEO"
  --refer_path "$IMAGE"
  --save_path "$WORK"
  --resolution_area "$RES_W" "$RES_H")
if [ "$MODE" = replace ]; then
  PRE+=(--iterations 3 --k 7 --w_len 1 --h_len 1 --replace_flag)
else
  PRE+=(--retarget_flag)
fi
"${PRE[@]}"

echo "==> Generating"
GEN=(generate.py --task animate-14B --ckpt_dir "$CKPT" --src_root_path "$WORK"
  --refert_num 1 --save_file "$OUT")
[ "$MODE" = replace ] && GEN+=(--replace_flag --use_relighting_lora)

if [ "$NGPU" -gt 1 ]; then
  python -m torch.distributed.run --nnodes 1 --nproc_per_node "$NGPU" "${GEN[@]}" \
    --dit_fsdp --t5_fsdp --ulysses_size "$NGPU"
else
  # Offload + dtype conversion keeps single-GPU memory down.
  python "${GEN[@]}" --offload_model True --convert_model_dtype
fi

echo "==> Done: $OUT"
