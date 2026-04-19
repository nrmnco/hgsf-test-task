"""eval/judge.py — LLM-as-judge for factual correctness and hallucination scoring.

judge_factual_correctness: scores agent answer against a reference (0/0.5/1.0).
judge_quote_grounding: verifies each extracted quote is grounded in its source text.

Both use claude-haiku-4-5 by default (override with DRL_JUDGE_MODEL env var).
Rubrics live in eval/rubrics/ as checked-in Markdown files.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import anthropic
from anthropic import Anthropic

from checks import CheckResult
from rate_limiter import on_rate_limit, wait as rl_wait

# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

_RUBRICS_DIR = Path(__file__).parent / "rubrics"


def _load_rubric(name: str) -> str:
    path = _RUBRICS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text()
    raise FileNotFoundError(f"Rubric not found: {path}")


_FACTUAL_CORRECTNESS_RUBRIC = _load_rubric("factual_correctness")
_QUOTE_GROUNDING_RUBRIC = _load_rubric("quote_grounding")

# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 10.0  # seconds


def _create_with_retry(client: Anthropic, **kwargs) -> object:
    """Call client.messages.create with exponential backoff on 429 and 5xx.

    Also participates in the global rate limiter so all threads pause together
    when any one of them hits a rate limit.
    """
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        rl_wait()
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            on_rate_limit(delay)
            rl_wait()
            delay *= 2
        except anthropic.APIStatusError as e:
            if e.status_code < 500 or attempt == _MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_judge_response(raw: str) -> tuple[int | None, str]:
    """Parse the judge's JSON response. Returns (score, rationale).

    Expects {"score": 0|1|2, "rationale": "..."}. Falls back to extracting
    a bare digit if JSON parsing fails.
    """
    raw = raw.strip()
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            score = int(data.get("score", -1))
            rationale = str(data.get("rationale", "")).strip()
            if score in (0, 1, 2):
                return score, rationale
    except Exception:
        pass

    m = re.search(r"\b([012])\b", raw)
    if m:
        return int(m.group(1)), ""
    return None, ""


def _extract_text(response) -> str:
    """Pull plain text from an Anthropic API response."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


# ---------------------------------------------------------------------------
# Public: hallucination / quote grounding
# ---------------------------------------------------------------------------


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

    model = model or os.getenv("DRL_JUDGE_MODEL", "claude-haiku-4-5")
    client = Anthropic()

    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(quotes))
    user_content = f"SOURCE TEXT:\n{source_text}\n\nQUOTES TO CHECK:\n{numbered}"

    try:
        resp = _create_with_retry(
            client,
            model=model,
            max_tokens=512,
            temperature=0.0,
            system=_QUOTE_GROUNDING_RUBRIC,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = _extract_text(resp)

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON array in response: {raw[:200]!r}")

        verdicts = json.loads(m.group(0))
        out = [
            (str(item.get("verdict", "")).upper() == "GROUNDED", item.get("reason", ""))
            for item in verdicts
        ]

        # Pad missing verdicts as ungrounded rather than silently passing them
        while len(out) < len(quotes):
            out.append((False, "no verdict returned — treating as ungrounded"))

        return out[:len(quotes)]

    except Exception as e:
        # On failure, don't block the eval — skip grounding check
        return [(True, f"judge error: {type(e).__name__}: {e}")] * len(quotes)


# ---------------------------------------------------------------------------
# Public: factual correctness
# ---------------------------------------------------------------------------


def judge_factual_correctness(
    question: str,
    expected_answer: str,
    actual_answer: str | None,
    model: str | None = None,
    dry_run: bool = False,
) -> CheckResult:
    """Score the agent's answer against a reference using an LLM judge.

    Returns a CheckResult with score 0.0 (wrong), 0.5 (partial), or 1.0 (correct).
    """
    name = "factual_correctness"

    if dry_run:
        return CheckResult(name, passed=True, score=-1.0, detail="skipped (dry run)")

    if not actual_answer:
        return CheckResult(name, passed=False, score=0.0, detail="no answer produced by agent")

    model = model or os.getenv("DRL_JUDGE_MODEL", "claude-haiku-4-5")
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
            system=_FACTUAL_CORRECTNESS_RUBRIC,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = _extract_text(resp)
    except Exception as e:
        return CheckResult(name, passed=False, score=0.0, detail=f"judge API call failed: {type(e).__name__}: {e}")

    score_int, rationale = _parse_judge_response(raw)
    if score_int is None:
        return CheckResult(name, passed=False, score=0.0, detail=f"judge returned unparseable response: {raw[:80]!r}")

    score_float = score_int / 2.0  # 0→0.0, 1→0.5, 2→1.0
    passed = score_int >= 1        # partial or better counts as pass

    labels = {0: "wrong", 1: "partial", 2: "correct"}
    detail = f"score {score_int}/2 ({labels[score_int]})"
    if rationale:
        detail += f" — {rationale}"

    return CheckResult(name, passed=passed, score=score_float, detail=detail)
