#!/usr/bin/env bash
# Encodes each video with SVT-AV1 at reduced resolution, then zips the result.
# Settings can be overridden with env vars, e.g. HEIGHT=336 CRF=44 bash compress.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
OUT_DIR="${OUT_DIR:-${HERE}}"  # where archive/ and archive.zip go
ARCHIVE_DIR="${OUT_DIR}/archive"

HEIGHT="${HEIGHT:-384}"
CRF="${CRF:-36}"
PRESET="${PRESET:-0}"
KEYINT="${KEYINT:-1200}"
FILM_GRAIN="${FILM_GRAIN:-22}"
TUNE="${TUNE:-0}"
SCALE_FLAGS="${SCALE_FLAGS:-lanczos}"
DOWNSCALE="${DOWNSCALE:-ffmpeg}"  # ffmpeg: scale to HEIGHT with SCALE_FLAGS; net: 512x384 network-matched bilinear

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR" "${OUT_DIR}/archive.zip"
mkdir -p "$ARCHIVE_DIR"

while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  IN="${IN_DIR}/${rel}"
  OUT="${ARCHIVE_DIR}/${rel%.*}.obu"
  echo "→ ${IN}  →  ${OUT}"
  ENC=(-c:v libsvtav1 -preset "$PRESET" -crf "$CRF" -g "$KEYINT"
       -svtav1-params "tune=${TUNE}:film-grain=${FILM_GRAIN}:scd=0" -f obu "$OUT")  # raw OBU: no per-frame container headers
  if [[ "$DOWNSCALE" == "net" ]]; then
    # 512x384 frames downsampled exactly like the networks' own preprocessing
    (cd "$PD" && python -m submissions.${SUB_NAME}.downscale "$IN") | ffmpeg -nostdin -y -hide_banner -loglevel error \
      -f rawvideo -pix_fmt rgb24 -s 512x384 -r 20 -i - \
      -vf "scale=out_color_matrix=bt601:out_range=tv" -pix_fmt yuv420p "${ENC[@]}"
  else
    ffmpeg -nostdin -y -hide_banner -loglevel error \
      -r 20 -fflags +genpts -i "$IN" -map 0:v:0 -an -sn -dn -map_metadata -1 \
      -vf "scale=-2:${HEIGHT}:flags=${SCALE_FLAGS}" -pix_fmt yuv420p "${ENC[@]}"
  fi
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -q -0 -X -r "${OUT_DIR}/archive.zip" .
echo "Compressed to ${OUT_DIR}/archive.zip ($(stat -f%z "${OUT_DIR}/archive.zip" 2>/dev/null || stat -c%s "${OUT_DIR}/archive.zip") bytes)"
