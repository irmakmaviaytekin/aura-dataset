"""Stage 3: LLM classification of candidate conversations.

Sends each candidate (truncated user turns) to Claude and asks whether it is a
genuinely PERSONAL conversation suitable for a mental-health entry-summary
context. Results are cached to labels.jsonl keyed by conversation_hash, so the
run is fully resumable — re-running skips already-labeled conversations.

Requires:  ANTHROPIC_API_KEY in env.
Usage:     ANTHROPIC_API_KEY=sk-ant-... python pipeline/03_classify.py
"""
import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C  # noqa: E402

SYSTEM = (
    "You label chat transcripts for a mental-health intake tool. "
    "A conversation is PERSONAL if the human is disclosing or working through "
    "their own feelings, relationships, mental health, life situation, or "
    "seeking emotional/personal support or advice about their own life. "
    "It is NOT personal if it is primarily a utilitarian task: coding, math, "
    "translation, general knowledge Q&A, content generation (essays, stories, "
    "marketing), roleplay/fiction not about the user, or homework. "
    "Respond with ONLY a compact JSON object and nothing else."
)

PROMPT_TMPL = (
    "Classify this conversation.\n\n"
    "Return JSON: {{\"is_personal\": bool, \"confidence\": 0.0-1.0, "
    "\"category\": one of [\"emotional_support\",\"relationships\",\"mental_health\","
    "\"life_decisions\",\"self_reflection\",\"utilitarian\",\"roleplay_fiction\",\"other\"], "
    "\"topics\": [short strings]}}\n\n"
    "USER TURNS:\n\"\"\"\n{body}\n\"\"\""
)


def load_cache():
    done = {}
    if C.LABELS_JSONL.exists():
        with open(C.LABELS_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done[obj["conversation_hash"]] = obj
                except Exception:
                    pass
    return done


def classify_one(client, ch, user_text):
    body = user_text[: C.LLM_MAX_CHARS]
    last_err = None
    for attempt in range(C.LLM_MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=C.LLM_MODEL,
                max_tokens=300,
                system=SYSTEM,
                messages=[{"role": "user", "content": PROMPT_TMPL.format(body=body)}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            text = text.strip()
            if text.startswith("```"):
                text = text.strip("`").split("\n", 1)[-1]
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return {
                "conversation_hash": ch,
                "is_personal": bool(data.get("is_personal", False)),
                "confidence": float(data.get("confidence", 0.0)),
                "category": data.get("category", "other"),
                "topics": data.get("topics", []),
            }
        except Exception as e:  # rate limit, transient, or parse error
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    return {"conversation_hash": ch, "is_personal": False, "confidence": 0.0,
            "category": "error", "topics": [], "error": str(last_err)}


def main():
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("Install the SDK:  pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment.")

    df = pd.read_parquet(C.CANDIDATES_PARQUET)
    if C.CLASSIFY_LIMIT:
        df = df.head(C.CLASSIFY_LIMIT)
    done = load_cache()
    todo = df[~df["conversation_hash"].isin(done.keys())]
    print(f"{len(df):,} candidates; {len(done):,} already labeled; {len(todo):,} to do.")
    if todo.empty:
        print("Nothing to classify.")
        return

    client = Anthropic()
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(C.LABELS_JSONL, "a") as out, \
            ThreadPoolExecutor(max_workers=C.LLM_CONCURRENCY) as ex:
        futs = {ex.submit(classify_one, client, r.conversation_hash, r.user_text): r.conversation_hash
                for r in todo.itertuples()}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            written += 1
            if written % 50 == 0:
                print(f"  labeled {written:,}/{len(todo):,}")
    print(f"Done. Wrote {written:,} new labels -> {C.LABELS_JSONL}")


if __name__ == "__main__":
    main()
