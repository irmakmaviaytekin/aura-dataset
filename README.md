# WildChat → Personal "Sets" pipeline

Filters [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
into **sets**: one person's *personal* conversations with an LLM over a 1–7 day
window, for a mental-health entry-summary tool.

## What a "set" is (and an important caveat)

A set = **≥2 personal conversations from the same `hashed_ip`, spanning 1–7 days.**

WildChat has **no persistent user ID**. `hashed_ip` is the closest proxy for
"one person," but it is noisy: shared/NAT/mobile-carrier IPs merge several
people, and dynamic IPs split one person across hashes. Treat a set as
"one IP's activity over a window," not a guaranteed single individual. This
matters for a clinical use case — validate before relying on it.

## Ethics / license

WildChat is **gated** under AI2's terms and licensed **ODC-BY** (attribution
required). It contains real, sometimes highly sensitive conversations. Using it
for a mental-health product carries real privacy/consent obligations beyond this
code — review the license and your own IRB/compliance requirements.

## Setup

```bash
pip install -r requirements.txt
```

**1. Get HuggingFace access (dataset is gated):**
   - Log in / create an account at huggingface.co
   - Go to https://huggingface.co/datasets/allenai/WildChat-1M and click
     **"Agree and access repository"** to accept the license.
   - Create a token: https://huggingface.co/settings/tokens (Read scope).

**2. Get an Anthropic API key** for the classifier: https://console.anthropic.com

## Run

```bash
export HF_TOKEN=hf_xxx
export ANTHROPIC_API_KEY=sk-ant-xxx

python pipeline/01_download.py      # ~3.4 GB parquet -> data_raw/
python pipeline/02_prefilter.py     # language + heuristic filter -> work/candidates.parquet
python pipeline/03_classify.py      # LLM personal/utilitarian labels -> work/labels.jsonl (resumable)
python pipeline/04_build_sets.py    # assemble sets -> out/personal_sets.jsonl + out/summary.json
```

Tune everything (languages, set thresholds, classifier model, confidence) in
`pipeline/config.py`. Stage 3 is resumable — safe to re-run after interruption.
Do a cheap trial first by setting `CLASSIFY_LIMIT` in config.py.

## Output

`out/personal_sets.jsonl` — one JSON object per set:
```
{ set_id, hashed_ip, country, state, n_conversations, span_days,
  start, end, categories, conversations: [ {conversation_hash, timestamp,
  category, confidence, topics, n_user_turns, full_text}, ... ] }
```
