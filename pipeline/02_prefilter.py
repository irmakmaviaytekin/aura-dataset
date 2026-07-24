"""Stage 2: language filter + heuristic personal-signal prefilter.

Reads the raw parquet shards, keeps only the columns we need, applies the
cheap high-recall filter from lib.py, and writes candidates.parquet.

Usage:  python pipeline/02_prefilter.py
"""
import os
import sys
import pyarrow.parquet as pq
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C          # noqa: E402
from lib import flatten_conversation, heuristic_prefilter  # noqa: E402

# Columns we read from each shard (keep memory small).
READ_COLS = ["conversation_hash", "hashed_ip", "timestamp", "language",
             "country", "state", "turn", "toxic", "redacted", "model", "conversation"]


def process_shard(path):
    pf = pq.ParquetFile(path)
    rows = []
    for batch in pf.iter_batches(batch_size=2000, columns=READ_COLS):
        df = batch.to_pandas()
        if C.KEEP_LANGUAGES is not None:
            df = df[df["language"].isin(C.KEEP_LANGUAGES)]
        if C.DROP_TOXIC and "toxic" in df.columns:
            df = df[~df["toxic"].fillna(False)]
        if df.empty:
            continue
        for _, r in df.iterrows():
            flat = flatten_conversation(r["conversation"])
            if flat["n_user_turns"] == 0:
                continue
            keep, signals, cf = heuristic_prefilter(
                flat["user_text"], C.MIN_PERSONAL_SIGNALS, C.MAX_CODE_FRACTION)
            if not keep:
                continue
            rows.append({
                "conversation_hash": r["conversation_hash"],
                "hashed_ip": r["hashed_ip"],
                "timestamp": r["timestamp"],
                "language": r["language"],
                "country": r.get("country"),
                "state": r.get("state"),
                "turn": r.get("turn"),
                "toxic": r.get("toxic"),
                "model": r.get("model"),
                "n_user_turns": flat["n_user_turns"],
                "personal_signals": signals,
                "code_fraction": round(cf, 3),
                "user_text": flat["user_text"],
                "full_text": flat["full_text"],
            })
    return rows


def main():
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(C.RAW_DIR.rglob("*.parquet"))
    if not shards:
        sys.exit("No parquet shards in data_raw/. Run 01_download.py first.")
    all_rows, total_seen = [], 0
    for i, sh in enumerate(shards, 1):
        rows = process_shard(sh)
        all_rows.extend(rows)
        total_seen += 1
        print(f"[{i}/{len(shards)}] {sh.name}: kept {len(rows):,} (running total {len(all_rows):,})")
    out = pd.DataFrame(all_rows)
    before = len(out)
    # WildChat's raw export contains exact duplicate rows for some
    # conversations (same conversation_hash/content, logged 2-3x a few
    # seconds/minutes apart). Collapse them so downstream joins stay 1:1.
    out = out.drop_duplicates(subset="conversation_hash", keep="first")
    if before != len(out):
        print(f"Dropped {before - len(out):,} exact-duplicate rows (same conversation_hash).")
    out.to_parquet(C.CANDIDATES_PARQUET, index=False)
    print(f"\nWrote {len(out):,} candidate conversations -> {C.CANDIDATES_PARQUET}")
    if len(out):
        print(f"Distinct IPs among candidates: {out['hashed_ip'].nunique():,}")


if __name__ == "__main__":
    main()
