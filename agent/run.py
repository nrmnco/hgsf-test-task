"""Deep Research Lite — one-shot CLI.

Usage:
    python run.py "your question here"

Writes the full trace to ./traces/<run_id>.json and prints the final answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.agent import run_agent


def main() -> int:
    load_dotenv()

    if len(sys.argv) < 2:
        print('usage: python run.py "your question"', file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("empty question", file=sys.stderr)
        return 2

    result = run_agent(question)

    traces_dir = Path(__file__).parent.parent / "traces"
    traces_dir.mkdir(exist_ok=True)
    trace_path = traces_dir / f"{result.run_id}.json"
    with trace_path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print()
    print("=" * 60)
    print(f"ANSWER ({result.stopped_reason}):")
    print("=" * 60)
    print(result.final_answer or "")
    if result.citations:
        print()
        print("Citations:")
        for i, c in enumerate(result.citations, 1):
            print(f"  [{i}] {c}")
    print()
    print(
        f"model={result.model} "
        f"tokens=in:{result.total_tokens['input']}/out:{result.total_tokens['output']} "
        f"cost=${result.cost_usd:.4f} "
        f"wall={result.wall_time_ms}ms"
    )
    print(f"trace -> {trace_path}")
    return 0 if result.stopped_reason != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
