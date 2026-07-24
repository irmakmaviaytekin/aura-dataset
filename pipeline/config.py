"""Central configuration for the WildChat 'personal sets' pipeline.

Every tunable lives here so you can change behavior without touching logic.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"      # downloaded parquet shards
WORK_DIR = ROOT / "work"         # intermediate artifacts
OUT_DIR = ROOT / "out"           # final deliverables

CANDIDATES_PARQUET = WORK_DIR / "candidates.parquet"   # after heuristic prefilter
LABELS_JSONL = WORK_DIR / "labels.jsonl"               # LLM classification cache (resumable)
SETS_JSONL = OUT_DIR / "personal_sets.jsonl"           # final sets
SETS_SUMMARY = OUT_DIR / "summary.json"                # counts / stats

# ---- Dataset ---------------------------------------------------------------
HF_REPO = "allenai/WildChat-1M"

# Languages to keep. WildChat spans 74 languages; default to English only for
# classifier quality and cost. Set to None to keep every language.
KEEP_LANGUAGES = {"English"}

# Drop conversations flagged toxic by the dataset? For a clinical tool you may
# actually WANT distress signals that trip toxicity filters, so default False.
DROP_TOXIC = False

# ---- 'Set' definition (one IP's activity over a window) --------------------
MIN_CONVOS_PER_SET = 2       # at least this many personal convos
MIN_SPAN_DAYS = 1.0          # first->last personal convo spans at least this
MAX_SPAN_DAYS = 7.0          # ...and at most this
# If one IP has personal convos spread over >MAX_SPAN_DAYS, we slice them into
# multiple sets with a greedy 7-day rolling window (see 04_build_sets.py).

# ---- Heuristic prefilter (cheap, high-recall) ------------------------------
# A conversation survives to the LLM if it shows >=1 personal signal and is not
# dominated by code. Tune MIN_PERSONAL_SIGNALS up to shrink the LLM workload.
MIN_PERSONAL_SIGNALS = 1
MAX_CODE_FRACTION = 0.5      # drop if >50% of user text looks like code

# ---- LLM classifier --------------------------------------------------------
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_CHARS = 4000         # truncate user turns fed to the classifier
LLM_CONCURRENCY = 8          # parallel requests
LLM_MAX_RETRIES = 4
# A conversation counts as 'personal' if the model returns is_personal=true
# with confidence >= this threshold.
PERSONAL_CONFIDENCE_MIN = 0.5

# Optional cap for a cheap trial run (None = classify all candidates).
CLASSIFY_LIMIT = None
