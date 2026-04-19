"""eval/runner.py — Main entry point for the Deep Research Lite eval framework.

Usage:
    python eval/runner.py                              # run all cases (parallel)
    python eval/runner.py --case-id space-01           # single case
    python eval/runner.py --case-id space-01 space-02
    python eval/runner.py --dry-run                    # load existing traces, skip LLM judge
    python eval/runner.py --concurrency 8              # increase parallelism (default: 4)
    python eval/runner.py --repeats 3                  # run each case 3 times (flakiness)
    python eval/runner.py --output my_report.json

Traces are written to traces/eval-{case_id}.json (single run) or
traces/eval-{case_id}-rep{N}.json (repeats). Transient API errors (429, 5xx)
are retried with exponential backoff. Assertion failures are never retried.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Make the repo root importable
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from checks import CheckResult, run_deterministic_checks
from judge import judge_factual_correctness
from report import load_baseline, print_diff, print_report, save_report
from viewer import generate_index, generate_viewer

# Lock for thread-safe console output
_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    case_id: str
    category: str
    question: str
    run_id: str | None
    stopped_reason: str
    total_tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    cost_usd: float = 0.0
    wall_time_ms: int = 0
    final_answer: str | None = None
    citations: list = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    overall_passed: bool = False
    error: str | None = None
    # Repeat / flakiness support (populated when --repeats > 1)
    sub_runs: list["EvalResult"] = field(default_factory=list)
    passed_count: int = 1   # how many of total_runs passed
    total_runs: int = 1     # always 1 unless --repeats N


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    return data["cases"]


def _result_from_dict(d: dict):
    """Reconstruct a RunResult-compatible object from a saved trace dict."""
    from agent.agent import RunResult
    return RunResult(
        run_id=d.get("run_id", ""),
        question=d.get("question", ""),
        messages=d.get("messages", []),
        final_answer=d.get("final_answer"),
        citations=d.get("citations", []),
        stopped_reason=d.get("stopped_reason", "error"),
        total_tokens=d.get("total_tokens", {"input": 0, "output": 0}),
        cost_usd=d.get("cost_usd", 0.0),
        wall_time_ms=d.get("wall_time_ms", 0),
        model=d.get("model", ""),
        error=d.get("error"),
    )


def _trace_name(case_id: str, repeat_index: int) -> str:
    """Return trace filename stem for a given case and repeat index.

    repeat_index=0 → eval-{case_id}  (single-run, backward-compatible)
    repeat_index=N → eval-{case_id}-rep{N}
    """
    if repeat_index == 0:
        return f"eval-{case_id}"
    return f"eval-{case_id}-rep{repeat_index}"


def _load_trace(traces_dir: Path, case_id: str, repeat_index: int = 0):
    """Load an existing eval trace for this case, or return None."""
    trace_path = traces_dir / f"{_trace_name(case_id, repeat_index)}.json"
    if trace_path.exists():
        with trace_path.open() as f:
            return _result_from_dict(json.load(f))
    return None


def _save_trace(traces_dir: Path, case_id: str, result, repeat_index: int = 0) -> None:
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{_trace_name(case_id, repeat_index)}.json"
    with trace_path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def _make_failed_result(case_id: str, question: str, error: str):
    """Synthetic RunResult for when run_agent itself raised an exception."""
    from agent.agent import RunResult
    return RunResult(
        run_id=None,
        question=question,
        messages=[],
        final_answer=None,
        citations=[],
        stopped_reason="error",
        total_tokens={"input": 0, "output": 0},
        cost_usd=0.0,
        wall_time_ms=0,
        model="",
        error=error,
    )


def _overall_passed(checks: list[CheckResult]) -> bool:
    """True iff all applicable (non-skipped) checks passed."""
    applicable = [c for c in checks if c.score != -1.0]
    return all(c.passed for c in applicable)


# ---------------------------------------------------------------------------
# Retry logic — wraps run_agent for transient API errors
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 5.0  # seconds


def _run_agent_with_retry(question: str, case_id: str) -> object:
    """Call run_agent with exponential backoff on rate-limit and server errors.

    Only retries on:
      - 429 RateLimitError
      - 5xx APIStatusError
    Never retries on assertion-level failures.
    """
    import anthropic
    from agent.agent import run_agent

    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return run_agent(question)
        except anthropic.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            _log(f"  [RATE] {case_id}: rate limited, retrying in {delay:.0f}s "
                 f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
        except anthropic.APIStatusError as e:
            if e.status_code < 500 or attempt == _MAX_RETRIES:
                raise
            _log(f"  [ERR ] {case_id}: server error {e.status_code}, retrying in {delay:.0f}s "
                 f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------------------
# Core: evaluate one case (one repeat)
# ---------------------------------------------------------------------------


def evaluate_case(
    case: dict,
    traces_dir: Path,
    dry_run: bool,
    repeat_index: int = 0,
) -> EvalResult:
    """Evaluate one case, one repeat.

    repeat_index=0 means single run (backward-compatible trace name).
    repeat_index>=1 means this is the Nth repeat of a --repeats run.
    """
    case_id = case["id"]
    question = case["question"]
    category = case["category"]

    label = case_id if repeat_index == 0 else f"{case_id}[rep{repeat_index}]"

    # Step 1: get a RunResult (from disk or live)
    agent_result = None
    loaded_from_disk = False

    if dry_run:
        agent_result = _load_trace(traces_dir, case_id, repeat_index)
        if agent_result is None:
            _log(f"  [SKIP] {label}: no trace found on disk (dry-run mode)")
            return EvalResult(
                case_id=case_id,
                category=category,
                question=question,
                run_id=None,
                stopped_reason="error",
                error="no trace found (dry-run mode)",
                checks=[],
                overall_passed=False,
            )
        loaded_from_disk = True
    else:
        agent_result = _load_trace(traces_dir, case_id, repeat_index)
        if agent_result is not None:
            loaded_from_disk = True
        else:
            try:
                _log(f"  [RUN ] {label}: running agent…")
                agent_result = _run_agent_with_retry(question, label)
                _save_trace(traces_dir, case_id, agent_result, repeat_index)
            except Exception as e:
                err_str = f"{type(e).__name__}: {e}"
                _log(f"  [ERR ] {label}: {err_str}")
                agent_result = _make_failed_result(case_id, question, err_str)

    source = "disk" if loaded_from_disk else "live"
    _log(f"  [CHEK] {label} ({source}): stopped_reason={agent_result.stopped_reason}")

    # Step 2: deterministic checks
    checks = run_deterministic_checks(case, agent_result, dry_run=dry_run)

    # Step 3: LLM judge
    judge_result = judge_factual_correctness(
        question=question,
        expected_answer=case.get("expected_answer", ""),
        actual_answer=agent_result.final_answer,
        dry_run=dry_run,
    )
    checks.append(judge_result)

    eval_result = EvalResult(
        case_id=case_id,
        category=category,
        question=question,
        run_id=agent_result.run_id,
        stopped_reason=agent_result.stopped_reason,
        total_tokens=agent_result.total_tokens,
        cost_usd=agent_result.cost_usd,
        wall_time_ms=agent_result.wall_time_ms,
        final_answer=agent_result.final_answer,
        citations=agent_result.citations,
        checks=checks,
        overall_passed=_overall_passed(checks),
        error=agent_result.error,
    )

    # Generate HTML trace viewer
    trace_path = traces_dir / f"{_trace_name(case_id, repeat_index)}.json"
    viewer_path = traces_dir / f"{_trace_name(case_id, repeat_index)}.html"
    generate_viewer(eval_result, trace_path, viewer_path)

    return eval_result


# ---------------------------------------------------------------------------
# Repeat aggregation
# ---------------------------------------------------------------------------


def _aggregate_repeats(case: dict, runs: list[EvalResult]) -> EvalResult:
    """Combine N repeat EvalResults into one summary EvalResult.

    Aggregated checks show "X/N runs passed" as detail.
    overall_passed = True only if ALL runs passed.
    """
    passed_count = sum(1 for r in runs if r.overall_passed)
    total_runs = len(runs)

    # Build aggregated checks: one CheckResult per check name
    check_names = [c.check_name for c in runs[0].checks] if runs else []
    agg_checks: list[CheckResult] = []
    for check_name in check_names:
        per_run = [
            c for r in runs
            for c in r.checks
            if c.check_name == check_name
        ]
        applicable = [c for c in per_run if c.score != -1.0]
        if not applicable:
            agg_checks.append(CheckResult(check_name, passed=True, score=-1.0, detail="not applicable"))
        else:
            n_passed = sum(1 for c in applicable if c.passed)
            n_total = len(applicable)
            avg_score = sum(c.score for c in applicable) / n_total
            agg_checks.append(CheckResult(
                check_name,
                passed=(n_passed == n_total),
                score=round(avg_score, 4),
                detail=f"{n_passed}/{n_total} runs passed",
            ))

    return EvalResult(
        case_id=case["id"],
        category=case["category"],
        question=case["question"],
        run_id=runs[0].run_id if runs else None,
        stopped_reason=runs[0].stopped_reason if runs else "error",
        total_tokens={
            "input": sum(r.total_tokens.get("input", 0) for r in runs),
            "output": sum(r.total_tokens.get("output", 0) for r in runs),
        },
        cost_usd=sum(r.cost_usd for r in runs),
        wall_time_ms=max((r.wall_time_ms for r in runs), default=0),
        final_answer=runs[0].final_answer if runs else None,
        citations=runs[0].citations if runs else [],
        checks=agg_checks,
        overall_passed=(passed_count == total_runs),
        error=None,
        sub_runs=runs,
        passed_count=passed_count,
        total_runs=total_runs,
    )


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def run_all_cases(
    cases: list[dict],
    traces_dir: Path,
    dry_run: bool,
    concurrency: int,
    repeats: int,
) -> list[EvalResult]:
    """Run all cases (with optional repeats) in parallel.

    Returns one EvalResult per case, in the same order as `cases`.
    When repeats > 1, each EvalResult is an aggregate with sub_runs populated.
    """
    # Build the full job list: (case, repeat_index)
    # repeat_index=0 for single run, 1..N for repeats
    if repeats == 1:
        jobs = [(case, 0) for case in cases]
    else:
        jobs = [(case, i) for case in cases for i in range(1, repeats + 1)]

    # {case_id: {repeat_index: EvalResult}}
    raw: dict[str, dict[int, EvalResult]] = {case["id"]: {} for case in cases}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_job = {
            pool.submit(evaluate_case, case, traces_dir, dry_run, repeat_index): (case, repeat_index)
            for case, repeat_index in jobs
        }
        for future in as_completed(future_to_job):
            case, repeat_index = future_to_job[future]
            case_id = case["id"]
            try:
                raw[case_id][repeat_index] = future.result()
            except Exception as e:
                _log(f"  [FAIL] {case_id}: unhandled exception: {e}")
                raw[case_id][repeat_index] = EvalResult(
                    case_id=case_id,
                    category=case["category"],
                    question=case["question"],
                    run_id=None,
                    stopped_reason="error",
                    error=f"unhandled: {type(e).__name__}: {e}",
                    overall_passed=False,
                )

    # Assemble results in original case order
    results: list[EvalResult] = []
    for case in cases:
        case_id = case["id"]
        run_map = raw[case_id]

        if repeats == 1:
            results.append(run_map.get(0) or EvalResult(
                case_id=case_id, category=case["category"], question=case["question"],
                run_id=None, stopped_reason="error", overall_passed=False,
            ))
        else:
            runs = [run_map[i] for i in range(1, repeats + 1) if i in run_map]
            results.append(_aggregate_repeats(case, runs))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Deep Research Lite against the eval dataset."
    )
    p.add_argument(
        "--case-id",
        nargs="+",
        metavar="ID",
        help="Run only specific case IDs (e.g. --case-id space-01 conflict-01)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load existing traces from disk; skip live agent calls and LLM judge.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=2,
        metavar="N",
        help="Maximum number of cases to run in parallel (default: 2). "
             "Keep low to avoid hitting the 50 RPM rate limit.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help="Run each case N times to measure flakiness (default: 1).",
    )
    p.add_argument(
        "--dataset",
        default=str(_REPO_ROOT / "eval_dataset.json"),
        metavar="PATH",
        help="Path to eval_dataset.json (default: repo root)",
    )
    p.add_argument(
        "--traces-dir",
        default=str(_REPO_ROOT / "traces"),
        metavar="PATH",
        help="Directory for eval trace files (default: traces/)",
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "eval_report.json"),
        metavar="PATH",
        help="Path to write the JSON report (default: eval_report.json)",
    )
    p.add_argument(
        "--baseline",
        default=str(_REPO_ROOT / "eval_report_baseline.json"),
        metavar="PATH",
        help="Path to a previous report to diff against (default: eval_report_baseline.json)",
    )
    p.add_argument(
        "--save-baseline",
        action="store_true",
        help="After running, save the current report as the new baseline.",
    )
    return p.parse_args()


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")
    args = parse_args()

    if not args.dry_run:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print(
                "ERROR: ANTHROPIC_API_KEY is not set.\n"
                "Set it in .env or the environment, or use --dry-run to load existing traces.",
                file=sys.stderr,
            )
            return 1

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        return 1
    cases = _load_dataset(dataset_path)

    if args.case_id:
        requested = set(args.case_id)
        filtered = [c for c in cases if c["id"] in requested]
        unknown = requested - {c["id"] for c in filtered}
        if unknown:
            print(f"ERROR: unknown case ID(s): {sorted(unknown)}", file=sys.stderr)
            return 1
        cases = filtered

    traces_dir = Path(args.traces_dir)
    repeat_str = f", repeats={args.repeats}" if args.repeats > 1 else ""
    mode = "[dry-run]" if args.dry_run else f"[live, concurrency={args.concurrency}{repeat_str}]"
    print(f"\nEvaluating {len(cases)} case(s) {mode}\n")

    if not args.dry_run and args.concurrency * args.repeats > 4:
        print(
            f"  WARNING: concurrency={args.concurrency} × repeats={args.repeats} may hit the "
            f"50 RPM rate limit. Consider --concurrency 1 or --concurrency 2 for large runs.\n"
        )

    results = run_all_cases(
        cases=cases,
        traces_dir=traces_dir,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        repeats=args.repeats,
    )

    print_report(results)
    save_report(results, args.output)

    # Generate HTML index in repo root for easy access
    index_path = _REPO_ROOT / "report.html"
    generate_index(results, index_path)
    print(f"Viewer index: {index_path}\n")

    # Diff vs baseline
    baseline = load_baseline(args.baseline)
    if baseline:
        print_diff(results, baseline)
    else:
        print(f"(No baseline found at {args.baseline} — run with --save-baseline to create one.)\n")

    # Optionally promote current report to baseline
    if args.save_baseline:
        import shutil
        shutil.copy(args.output, args.baseline)
        print(f"Baseline saved to {args.baseline}\n")

    all_passed = all(r.overall_passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
