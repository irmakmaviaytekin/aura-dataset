"""Stage 3 (fallback): heuristic-only labels when no LLM key is available.

Produces the same labels.jsonl format as 03_classify.py, but assigns
is_personal from the heuristic personal-signal score instead of an LLM. This
is noisier than the LLM pass (that's the whole point of the hybrid design), so
it's tightened: it requires more personal signals and low code content, and it
subtracts obvious utilitarian markers.

When you later get an ANTHROPIC_API_KEY, delete work/labels.jsonl and run
03_classify.py instead for higher precision.

Usage:  python pipeline/03b_heuristic_labels.py
"""
import os
import sys
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C  # noqa: E402

# Tighter than the recall-oriented prefilter: standalone heuristic precision.
MIN_SIGNALS = 2          # need >=2 distinct personal cues
MAX_CODE = 0.15          # almost no code

# Utilitarian phrases that, if present, argue against 'personal' even with cues.
UTILITARIAN = [
    "write a", "generate a", "essay", "translate", "paraphrase", "rewrite",
    "code", "python", "javascript", "sql", "function", "html", "as an ai",
    "act as", "you are a", "story about", "poem about", "marketing", "resume",
]


def categorize(text_low, signals):
    if any(k in text_low for k in ["suicid", "self harm", "self-harm", "want to die", "kill myself"]):
        return "mental_health"
    if any(k in text_low for k in ["girlfriend", "boyfriend", "wife", "husband", "partner",
                                   "breakup", "broke up", "divorce", "my ex", "relationship"]):
        return "relationships"
    if any(k in text_low for k in ["anxious", "anxiety", "depress", "lonely", "panic",
                                   "overwhelmed", "hopeless", "therapy", "therapist"]):
        return "mental_health"
    if any(k in text_low for k in ["should i", "what should i do", "i need advice", "my future"]):
        return "life_decisions"
    return "self_reflection"


def main():
    df = pd.read_parquet(C.CANDIDATES_PARQUET)
    if C.CLASSIFY_LIMIT:
        df = df.head(C.CLASSIFY_LIMIT)
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    n_pers = 0
    with open(C.LABELS_JSONL, "w") as out:
        for r in df.itertuples():
            low = r.user_text.lower()
            util_hits = sum(1 for k in UTILITARIAN if k in low)
            score = int(r.personal_signals) - util_hits
            is_personal = (
                int(r.personal_signals) >= MIN_SIGNALS
                and float(r.code_fraction) <= MAX_CODE
                and score >= 1
            )
            if is_personal:
                n_pers += 1
            # Confidence: scaled signal strength, capped.
            conf = min(0.5 + 0.1 * max(score - MIN_SIGNALS, 0), 0.9) if is_personal else 0.2
            out.write(json.dumps({
                "conversation_hash": r.conversation_hash,
                "is_personal": bool(is_personal),
                "confidence": round(conf, 2),
                "category": categorize(low, r.personal_signals) if is_personal else "utilitarian",
                "topics": [],
                "labeler": "heuristic",
            }) + "\n")
    print(f"Heuristic-labeled {len(df):,} candidates; {n_pers:,} marked personal.")
    print(f"Wrote {C.LABELS_JSONL}")
    print("NOTE: heuristic labels are noisier than the LLM pass. Provide "
          "ANTHROPIC_API_KEY and run 03_classify.py to upgrade precision.")


if __name__ == "__main__":
    main()
