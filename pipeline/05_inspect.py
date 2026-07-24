"""Stage 5: eyeball a few random sets to gauge quality.

Usage:  python pipeline/05_inspect.py [n]
"""
import os
import sys
import json
import random
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
import config as C  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sets = [json.loads(l) for l in open(C.SETS_JSONL)]
    if not sets:
        print("No sets in output.")
        return
    print(f"{len(sets):,} sets total. Showing {min(n, len(sets))} at random.\n")
    for s in random.sample(sets, min(n, len(sets))):
        print("=" * 70)
        print(f"set {s['set_id']} | ip {s['hashed_ip'][:10]}… | {s['country']} "
              f"| {s['n_conversations']} convos over {s['span_days']}d | {s['categories']}")
        for c in s["conversations"]:
            print(f"\n  — {c['timestamp'][:10]} [{c['category']} conf={c['confidence']}] "
                  f"topics={c.get('topics')}")
            # show the user's opening lines only
            user_lines = [ln for ln in c["full_text"].splitlines() if ln.startswith("user:")]
            preview = " ".join(user_lines)[:400]
            print(textwrap.fill(preview, width=68, initial_indent="    ", subsequent_indent="    "))
        print()


if __name__ == "__main__":
    main()
