"""Deep Research Lite — tool implementations.

All four tools are deterministic given the same inputs (modulo the LLM call
inside `extract_quotes`, which uses temperature=0 and a fixed small model).

Tools:
  * web_search(query, k) -> list[SearchResult]   # BM25 over local corpus
  * fetch_url(url) -> str                        # raises PageNotFound on miss
  * extract_quotes(text, topic, max_quotes)      # small-model extraction
  * finish(answer, citations) -> None            # sentinel; agent loop catches
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
INDEX_PATH = CORPUS_DIR / "index.json"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


@dataclass
class Page:
    url: str
    title: str
    file: str
    text: str


def _load_corpus() -> dict[str, Page]:
    with INDEX_PATH.open() as f:
        index = json.load(f)
    pages: dict[str, Page] = {}
    for entry in index["pages"]:
        path = CORPUS_DIR / entry["file"]
        text = path.read_text()
        pages[entry["url"]] = Page(
            url=entry["url"], title=entry["title"], file=entry["file"], text=text
        )
    return pages


_PAGES: dict[str, Page] = _load_corpus()
_URLS: list[str] = list(_PAGES.keys())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


_BM25_CORPUS = [_tokenize(p.title + "\n" + p.text) for p in _PAGES.values()]
_BM25 = BM25Okapi(_BM25_CORPUS)


# ---------------------------------------------------------------------------
# Public tool: web_search
# ---------------------------------------------------------------------------


def _snippet(text: str, query_tokens: list[str], max_len: int = 200) -> str:
    """Build a query-centered snippet from a page."""
    lower = text.lower()
    best_pos = -1
    for tok in query_tokens:
        pos = lower.find(tok)
        if pos != -1:
            best_pos = pos if best_pos == -1 else min(best_pos, pos)
    if best_pos == -1:
        return text[:max_len].strip().replace("\n", " ")
    start = max(0, best_pos - max_len // 3)
    end = min(len(text), start + max_len)
    s = text[start:end].strip().replace("\n", " ")
    if start > 0:
        s = "…" + s
    if end < len(text):
        s = s + "…"
    return s


def web_search(query: str, k: int = 5) -> list[dict[str, str]]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _BM25.get_scores(tokens)
    ranked = sorted(
        zip(scores, _URLS), key=lambda pair: pair[0], reverse=True
    )
    results: list[dict[str, str]] = []
    for score, url in ranked[:k]:
        # Only include positive matches; a zero-score "hit" is pure noise.
        if score <= 0:
            continue
        page = _PAGES[url]
        results.append(
            {
                "url": page.url,
                "title": page.title,
                "snippet": _snippet(page.text, tokens),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Public tool: fetch_url
# ---------------------------------------------------------------------------


class PageNotFound(Exception):
    """Raised when the agent asks to fetch a URL not in the corpus."""


def fetch_url(url: str) -> str:
    page = _PAGES.get(url)
    if page is None:
        raise PageNotFound(
            f"No page at {url!r}. The corpus only contains URLs returned by "
            f"web_search."
        )
    return page.text


# ---------------------------------------------------------------------------
# Public tool: extract_quotes  (LLM-backed, small model)
# ---------------------------------------------------------------------------

# This intentionally uses a small model. It is known to occasionally return
# paraphrased or mildly hallucinated quotes — that is a planted defect the
# agent does not verify.

_EXTRACT_SYSTEM = (
    "You are a quote extractor. Given a TEXT and a TOPIC, return the "
    "{max_quotes} sentences from TEXT that are most relevant to TOPIC. "
    "Return strictly a JSON array of strings. Do not summarize. Do not add "
    "commentary. No markdown."
)


def _call_small_model(messages: list[dict[str, Any]]) -> str:
    """Send a non-tool-using request to a small Anthropic model and return text."""
    from anthropic import Anthropic

    client = Anthropic()
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] != "system"
    ]
    resp = client.messages.create(
        model=os.getenv("DRL_SMALL_MODEL", "claude-haiku-4-5"),
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=user_msgs,
    )
    return "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )


def extract_quotes(text: str, topic: str, max_quotes: int = 3) -> list[str]:
    system = _EXTRACT_SYSTEM.format(max_quotes=max_quotes)
    user = f"TOPIC: {topic}\n\nTEXT:\n{text}"
    raw = _call_small_model(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    # Best-effort JSON parse; fall back to line-splitting.
    raw = raw.strip()
    # Strip fenced code blocks if the model ignored the instruction.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(q) for q in parsed[:max_quotes]]
    except json.JSONDecodeError:
        pass
    # Fallback: naive split.
    lines = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]
    return lines[:max_quotes]


# ---------------------------------------------------------------------------
# Public tool: finish  (sentinel; the agent loop never actually calls this)
# ---------------------------------------------------------------------------


def finish(answer: str, citations: list[str]) -> None:
    """Sentinel. The agent loop detects this tool call and terminates."""
    return None


# ---------------------------------------------------------------------------
# Tool schema — exposed to the main agent loop.
# ---------------------------------------------------------------------------


TOOL_SCHEMA = [
    {
        "name": "web_search",
        "description": (
            "Search the local corpus. Returns up to k results, each with "
            "url, title, and a short snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the full text of a page by its URL. The URL must have "
            "been returned by web_search; otherwise the call raises "
            "PageNotFound."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "extract_quotes",
        "description": (
            "Extract up to max_quotes sentences from `text` that are "
            "relevant to `topic`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "topic": {"type": "string"},
                "max_quotes": {"type": "integer", "default": 3},
            },
            "required": ["text", "topic"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Terminate the run with a final answer and a list of citation "
            "URLs. `citations` must contain the URLs of pages you actually "
            "fetched."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["answer", "citations"],
        },
    },
]


TOOL_IMPLS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "extract_quotes": extract_quotes,
    # finish is handled specially by the agent loop.
}
