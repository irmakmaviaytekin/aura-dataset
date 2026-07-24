"""Shared helpers: conversation flattening + the heuristic personal-signal filter."""
from __future__ import annotations
import re

# --- Signal lexicons --------------------------------------------------------
# High-recall, deliberately broad. The LLM stage handles precision.

PERSONAL_KW = [
    # emotion / affect
    "i feel", "i'm feeling", "im feeling", "i am feeling", "feeling so", "feeling really",
    "anxious", "anxiety", "depress", "lonely", "loneliness", "so alone", "feel alone",
    "hopeless", "worthless", "empty inside", "numb", "overwhelmed", "burnt out", "burnout",
    "i can't cope", "i cant cope", "breaking down", "falling apart", "panic", "panicking",
    "i cried", "can't stop crying", "cant stop crying", "i hate myself", "hate my life",
    # crisis
    "want to die", "kill myself", "suicid", "self harm", "self-harm", "end it all",
    "no reason to live", "better off without me",
    # relationships / people
    "my girlfriend", "my boyfriend", "my wife", "my husband", "my partner", "my ex",
    "my mom", "my mother", "my dad", "my father", "my parents", "my family",
    "my sister", "my brother", "my son", "my daughter", "my kid", "my friend",
    "my best friend", "my boss", "my therapist", "my crush",
    "breakup", "broke up", "broken up", "divorce", "cheated on me", "ghosted me",
    # help-seeking / reflection
    "should i", "what should i do", "i need advice", "i don't know what to do",
    "i dont know what to do", "help me understand", "am i wrong", "was i wrong",
    "how do i deal with", "how do i cope", "i keep thinking", "i can't stop thinking",
    "i cant stop thinking", "my life", "my future", "afraid of", "scared of",
    "i regret", "i'm struggling", "im struggling", "struggling with",
    "therapy", "therapist", "counselor", "counsellor", "mental health",
]

# Utilitarian markers used only to compute a code fraction / obvious-task hint.
CODE_MARKERS = [
    "```", "def ", "class ", "import ", "function ", "public static", "select ",
    "#include", "console.log", "print(", "</", "/>", "{\n", "=>", "for (", "while (",
    "return ", "npm ", "pip install", "sql", "regex",
]


def flatten_conversation(conv) -> dict:
    """Turn WildChat's list-of-turns into plain text.

    Returns dict with user_text, assistant_text, full_text, n_user_turns.
    Robust to missing 'role' (alternating fallback) and non-string content.
    """
    user_parts, asst_parts, all_parts = [], [], []
    # pyarrow's to_pandas() yields list columns as numpy.ndarray (of dicts),
    # not native list/tuple — accept anything sized and iterable instead of
    # gating on exact type.
    if conv is None or not hasattr(conv, "__len__") or len(conv) == 0:
        return {"user_text": "", "assistant_text": "", "full_text": "", "n_user_turns": 0}
    for i, turn in enumerate(conv):
        if not isinstance(turn, dict):
            continue
        content = turn.get("content")
        if content is None:
            continue
        content = str(content)
        role = turn.get("role")
        if role is None:
            role = "user" if i % 2 == 0 else "assistant"
        role = str(role).lower()
        all_parts.append(f"{role}: {content}")
        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            asst_parts.append(content)
    return {
        "user_text": "\n".join(user_parts),
        "assistant_text": "\n".join(asst_parts),
        "full_text": "\n".join(all_parts),
        "n_user_turns": len(user_parts),
    }


def code_fraction(text: str) -> float:
    """Rough share of text inside fenced code blocks or dense code lines."""
    if not text:
        return 0.0
    # Characters inside ``` fences.
    fenced = sum(len(m) for m in re.findall(r"```.*?```", text, flags=re.DOTALL))
    frac = fenced / max(len(text), 1)
    # Line-level heuristic: lines that look like code.
    lines = text.splitlines() or [text]
    code_lines = sum(
        1 for ln in lines
        if any(mk in ln for mk in CODE_MARKERS) or (ln.strip().endswith((";", "{", "}")))
    )
    frac = max(frac, code_lines / max(len(lines), 1))
    return min(frac, 1.0)


def personal_signal_count(user_text: str) -> int:
    low = user_text.lower()
    return sum(1 for kw in PERSONAL_KW if kw in low)


def heuristic_prefilter(user_text: str, min_signals: int, max_code_fraction: float):
    """Return (keep: bool, signals: int, code_frac: float)."""
    signals = personal_signal_count(user_text)
    cf = code_fraction(user_text)
    keep = signals >= min_signals and cf <= max_code_fraction
    return keep, signals, cf
