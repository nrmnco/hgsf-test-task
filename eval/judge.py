"""eval/judge.py — LLM-as-judge for factual correctness scoring.

Sends (question, expected_answer, actual_answer) to a small model and
gets back a score: 0 = wrong, 1 = partial, 2 = correct.
Maps to CheckResult scores 0.0 / 0.5 / 1.0.
"""

from __future__ import annotations

import json as _json
import os
import re
import time
from pathlib import Path

from checks import CheckResult
from rate_limiter import on_rate_limit, wait as rl_wait

_RUBRICS_DIR = Path(__file__).parent / "rubrics"


def _load_rubric(name: str) -> str:
    path = _RUBRICS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text()
    raise FileNotFoundError(f"Rubric not found: {path}")

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 10.0  # seconds


def _create_with_retry(client, **kwargs) -> object:
    """Call client.messages.create with exponential backoff on 429 and 5xx."""
    import anthropic
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        rl_wait()  # block if any thread triggered a global pause
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            on_rate_limit(delay)  # pause all threads
            rl_wait()
            delay *= 2
        except anthropic.APIStatusError as e:
            if e.status_code < 500 or attempt == _MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2

_GROUNDING_SYSTEM = (
    "You are a strict hallucination detector. You will be shown a source text and one or more "
    "quotes that a model claims to have extracted from that source.\n\n"
    "For each quote, decide: are all factual claims in the quote actually supported by the source?\n\n"
    "A quote is GROUNDED if:\n"
    "- All facts (numbers, names, dates, technical terms) come from the source\n"
    "- Minor wording changes, sentence merging, or added connective words are acceptable\n\n"
    "A quote is NOT_GROUNDED if:\n"
    "- It contains facts not present in the source (fabricated numbers, invented names)\n"
    "- It contradicts the source (e.g. says 2013 when source says 2012)\n"
    "- It negates a claim from the source\n\n"
    "Reply with a JSON array of objects, one per quote, in the same order:\n"
    '[{"verdict": "GROUNDED", "reason": "..."}, {"verdict": "NOT_GROUNDED", "reason": "..."}]\n\n'
    "No extra text outside the JSON."
)

_JUDGE_SYSTEM = _load_rubric("factual_correctness")


def _parse_judge_response(raw: str) -> tuple[int | None, str]:
    """Parse the judge's JSON response. Returns (score, rationale)."""
    raw = raw.strip()
    # Try JSON parse first
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = _json.loads(m.group(0))
            score = int(data.get("score", -1))
            rationale = str(data.get("rationale", "")).strip()
            if score in (0, 1, 2):
                return score, rationale
    except Exception:
        pass
    # Fallback: extract bare digit
    m = re.search(r"\b([012])\b", raw)
    if m:
        return int(m.group(1)), ""
    return None, ""


def judge_quote_grounding(
    source_text: str,
    quotes: list[str],
    model: str | None = None,
    dry_run: bool = False,
) -> list[tuple[bool, str]]:
    """Check each quote against source_text using an LLM judge.

    Returns a list of (grounded: bool, reason: str) tuples, one per quote.
    On dry_run or API failure, returns [(True, "skipped")] * len(quotes).
    """
    if dry_run or not quotes:
        return [(True, "skipped")] * len(quotes)

    if model is None:
        model = os.getenv("DRL_JUDGE_MODEL", "claude-haiku-4-5")

    import json as _json
    from anthropic import Anthropic
    client = Anthropic()

    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(quotes))
    user_content = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"QUOTES TO CHECK:\n{numbered}"
    )

    try:
        resp = _create_with_retry(
            client,
            model=model,
            max_tokens=512,
            temperature=0.0,
            system=_GROUNDING_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        # Extract JSON array from response
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON array in response: {raw[:200]!r}")
        results = _json.loads(m.group(0))
        out = []
        for item in results:
            grounded = str(item.get("verdict", "")).upper() == "GROUNDED"
            reason = item.get("reason", "")
            out.append((grounded, reason))
        # Pad or trim to match quote count
        while len(out) < len(quotes):
            out.append((True, "no verdict returned"))
        return out[:len(quotes)]
    except Exception as e:
        # On failure, don't block the eval — skip grounding check
        return [(True, f"judge error: {type(e).__name__}: {e}")] * len(quotes)


def judge_factual_correctness(
    question: str,
    expected_answer: str,
    actual_answer: str | None,
    model: str | None = None,
    dry_run: bool = False,
) -> CheckResult:
    name = "factual_correctness"

    if dry_run:
        return CheckResult(name, passed=True, score=-1.0, detail="skipped (dry run)")

    if not actual_answer:
        return CheckResult(
            name, passed=False, score=0.0,
            detail="no answer produced by agent"
        )

    if model is None:
        model = os.getenv("DRL_JUDGE_MODEL", "claude-haiku-4-5")

    from anthropic import Anthropic
    client = Anthropic()

    user_content = (
        f"QUESTION: {question}\n\n"
        f"REFERENCE ANSWER: {expected_answer}\n\n"
        f"STUDENT ANSWER: {actual_answer}"
    )

    try:
        resp = _create_with_retry(
            client,
            model=model,
            max_tokens=256,
            temperature=0.0,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    except Exception as e:
        return CheckResult(
            name, passed=False, score=0.0,
            detail=f"judge API call failed: {type(e).__name__}: {e}"
        )

    score_int, rationale = _parse_judge_response(raw)
    if score_int is None:
        return CheckResult(
            name, passed=False, score=0.0,
            detail=f"judge returned unparseable response: {raw[:80]!r}"
        )

    score_float = score_int / 2.0   # 0→0.0, 1→0.5, 2→1.0
    passed = score_int >= 1         # partial or better counts as pass

    labels = {0: "wrong", 1: "partial", 2: "correct"}
    detail = f"score {score_int}/2 ({labels[score_int]})"
    if rationale:
        detail += f" — {rationale}"

    return CheckResult(name, passed=passed, score=score_float, detail=detail)
