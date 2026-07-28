"""Stage 1: download the WildChat-1M parquet shards (gated dataset).

Test for github

Requires:
  - An HF account that has ACCEPTED the license on the dataset page.
  - An HF token in env var HF_TOKEN (or run `huggingface-cli login` first).

Usage:
  HF_TOKEN=hf_xxx python pipeline/01_download.py
"""
import os
import sys
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(__file__))
import config as C  # noqa: E402


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("NOTE: no HF_TOKEN env var found; relying on cached `huggingface-cli login`.")
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=C.HF_REPO,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=str(C.RAW_DIR),
        token=token,
    )
    shards = sorted(C.RAW_DIR.rglob("*.parquet"))
    print(f"Downloaded to {path}")
    print(f"{len(shards)} parquet shard(s):")
    for s in shards[:10]:
        print("  ", s.relative_to(C.RAW_DIR))
    if len(shards) > 10:
        print(f"   ... and {len(shards) - 10} more")
    if not shards:
        sys.exit("ERROR: no parquet shards downloaded. Did you accept the license?")


if __name__ == "__main__":
    main()
