#!/usr/bin/env bash
# Must produce a raw video file at `<output_dir>/<base_name>.raw`.
# A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)`
# where N is the number of frames, H and W match the original video dimensions, no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="$(ls "${DATA_DIR}/${BASE}".* 2>/dev/null | head -n1)"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ -z "$SRC" ] && echo "ERROR: ${DATA_DIR}/${BASE}.* not found" >&2 && exit 1

  printf "Decoding + resizing %s ... " "$line"
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
