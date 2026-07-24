"""Stage 4: assemble personal 'sets'.

Joins LLM labels onto candidates, keeps personal conversations, groups by
hashed_ip, and slices each IP's timeline into sets that span 1-7 days with at
least MIN_CONVOS_PER_SET conversations. Writes personal_sets.jsonl + summary.json.

Usage:  python pipeline/04_build_sets.py
"""
import os
import sys
import json
import hashlib

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C  # noqa: E402


def load_labels():
    rows = []
    with open(C.LABELS_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Defensive: keep the last label per conversation_hash in case the cache
    # file ever picks up a repeated key (e.g. a resumed/interrupted run).
    return df.drop_duplicates(subset="conversation_hash", keep="last")


def windows_for_ip(group, max_span, min_span, min_convos):
    """Greedy left-to-right slicing of one IP's sorted convos into <=max_span sets."""
    group = group.sort_values("timestamp").reset_index(drop=True)
    sets, i, n = [], 0, len(group)
    while i < n:
        start_ts = group.loc[i, "timestamp"]
        j = i
        while j + 1 < n and (group.loc[j + 1, "timestamp"] - start_ts).total_seconds() <= max_span * 86400:
            j += 1
        window = group.loc[i:j]
        span_days = (window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]).total_seconds() / 86400
        if len(window) >= min_convos and span_days >= min_span:
            sets.append((window, span_days))
        i = j + 1
    return sets


def main():
    cand = pd.read_parquet(C.CANDIDATES_PARQUET)
    labels = load_labels()
    df = cand.merge(labels, on="conversation_hash", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    personal = df[(df["is_personal"]) & (df["confidence"] >= C.PERSONAL_CONFIDENCE_MIN)].copy()
    print(f"{len(df):,} labeled; {len(personal):,} personal (conf>={C.PERSONAL_CONFIDENCE_MIN}).")

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_sets = n_convos = 0
    ips_in_sets = set()
    with open(C.SETS_JSONL, "w") as out:
        for ip, group in personal.groupby("hashed_ip"):
            for window, span_days in windows_for_ip(
                    group, C.MAX_SPAN_DAYS, C.MIN_SPAN_DAYS, C.MIN_CONVOS_PER_SET):
                sid = hashlib.sha1(
                    f"{ip}-{window['timestamp'].iloc[0]}".encode()).hexdigest()[:12]
                convos = []
                for _, r in window.iterrows():
                    convos.append({
                        "conversation_hash": r["conversation_hash"],
                        "timestamp": r["timestamp"].isoformat(),
                        "category": r["category"],
                        "confidence": round(float(r["confidence"]), 2),
                        "topics": r["topics"],
                        "n_user_turns": int(r["n_user_turns"]),
                        "full_text": r["full_text"],
                    })
                rec = {
                    "set_id": sid,
                    "hashed_ip": ip,
                    "country": window["country"].iloc[0],
                    "state": window["state"].iloc[0],
                    "n_conversations": len(convos),
                    "span_days": round(span_days, 2),
                    "start": window["timestamp"].iloc[0].isoformat(),
                    "end": window["timestamp"].iloc[-1].isoformat(),
                    "categories": sorted(set(window["category"])),
                    "conversations": convos,
                }
                out.write(json.dumps(rec, default=str) + "\n")
                n_sets += 1
                n_convos += len(convos)
                ips_in_sets.add(ip)

    summary = {
        "labeled_conversations": int(len(df)),
        "personal_conversations": int(len(personal)),
        "sets": n_sets,
        "conversations_in_sets": n_convos,
        "distinct_ips_in_sets": len(ips_in_sets),
        "config": {
            "min_convos_per_set": C.MIN_CONVOS_PER_SET,
            "min_span_days": C.MIN_SPAN_DAYS,
            "max_span_days": C.MAX_SPAN_DAYS,
            "personal_confidence_min": C.PERSONAL_CONFIDENCE_MIN,
            "keep_languages": sorted(C.KEEP_LANGUAGES) if C.KEEP_LANGUAGES else None,
        },
    }
    with open(C.SETS_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {n_sets:,} sets ({n_convos:,} conversations) -> {C.SETS_JSONL}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
