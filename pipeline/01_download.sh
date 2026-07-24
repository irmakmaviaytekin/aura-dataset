#!/usr/bin/env bash
# Stage 1 (curl fallback): download WildChat-1M parquet shards directly.
#
# huggingface_hub's Python downloader hangs in this environment (its xet
# fast-path never receives bytes), but the dataset is NOT gated and plain
# HTTPS GETs against the resolve/main URLs work fine. This script fetches
# all 14 shards in parallel with curl, with resume + retry, straight into
# data_raw/data/ (same layout 02_prefilter.py expects).
#
# Usage:  bash pipeline/01_download.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/data_raw/data"
mkdir -p "$OUT_DIR"

REPO="allenai/WildChat-1M"
N_SHARDS=14
PARALLEL="${PARALLEL:-2}"

urls=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  fname=$(printf "train-%05d-of-%05d.parquet" "$i" "$N_SHARDS")
  urls+=("$fname")
done

download_one() {
  fname="$1"
  url="https://huggingface.co/datasets/${REPO}/resolve/main/data/${fname}"
  out="${OUT_DIR}/${fname}"
  # -C - resumes partial downloads; retries handle transient CDN hiccups.
  curl -sSL -C - --retry 15 --retry-delay 5 --retry-all-errors \
    -o "$out" "$url"
  status=$?
  size=$(stat -f%z "$out" 2>/dev/null || echo 0)
  echo "DONE $fname status=$status size=$size"
}
export -f download_one
export OUT_DIR REPO

printf "%s\n" "${urls[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'download_one "$@"' _ {}

echo "ALL_SHARDS_ATTEMPTED"
