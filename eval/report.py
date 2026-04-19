"""eval/report.py — Console table and JSON report for eval results."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runner import EvalResult


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

_COL = {
    "id":       22,
    "category": 22,
    "result":   7,
    "checks":   14,
    "judge":    10,
    "cost":     9,
}


def _hdr(label: str, width: int) -> str:
    return label.ljust(width)


def _row(values: list[tuple[str, int]]) -> str:
    return "  ".join(v.ljust(w) for v, w in values)


def _judge_cell(checks: list) -> str:
    for c in checks:
        if c.check_name == "factual_correctness":
            if c.score == -1.0:
                return "skipped"
            pct = int(c.score * 100)
            return f"{pct}%"
    return "n/a"


def _checks_cell(checks: list) -> str:
    applicable = [c for c in checks if c.score != -1.0 and c.check_name != "factual_correctness"]
    passed = sum(1 for c in applicable if c.passed)
    total = len(applicable)
    failed_names = [c.check_name for c in applicable if not c.passed]
    cell = f"{passed}/{total}"
    if failed_names:
        cell += f" ({','.join(failed_names[:2])}{'…' if len(failed_names) > 2 else ''})"
    return cell


def print_report(results: list["EvalResult"]) -> None:
    header = _row([
        ("Case ID", _COL["id"]),
        ("Category", _COL["category"]),
        ("Result", _COL["result"]),
        ("Checks", _COL["checks"] + 14),
        ("Judge", _COL["judge"]),
        ("Cost", _COL["cost"]),
    ])
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)

    for r in results:
        if r.total_runs > 1:
            result_label = f"{r.passed_count}/{r.total_runs}"
        else:
            result_label = "PASS" if r.overall_passed else "FAIL"

        checks_cell = _checks_cell(r.checks)
        judge_cell = _judge_cell(r.checks)
        cost_cell = f"${r.cost_usd:.4f}"

        print(_row([
            (r.case_id, _COL["id"]),
            (r.category, _COL["category"]),
            (result_label, _COL["result"]),
            (checks_cell, _COL["checks"] + 14),
            (judge_cell, _COL["judge"]),
            (cost_cell, _COL["cost"]),
        ]))

        # Print failed check details indented
        for c in r.checks:
            if not c.passed and c.score != -1.0 and c.detail:
                print(f"    ✗ {c.check_name}: {c.detail}")

        # For repeats: show per-check variance for checks that weren't perfectly consistent
        if r.total_runs > 1:
            for c in r.checks:
                if c.score not in (-1.0, 0.0, 1.0) and c.detail:
                    print(f"    ~ {c.check_name}: {c.detail} (flaky)")

    # Per-category summary
    by_cat: dict[str, list] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    print()
    print("Category Summary")
    print(sep)
    cat_header = _row([
        ("Category", _COL["category"]),
        ("Total", 8),
        ("Passed", 8),
        ("Pass Rate", 12),
        ("Avg Judge", 12),
    ])
    print(cat_header)
    print(sep)

    for cat in sorted(by_cat):
        cases = by_cat[cat]
        total = len(cases)
        passed = sum(1 for r in cases if r.overall_passed)
        rate = passed / total if total else 0.0

        judge_scores = [
            c.score for r in cases for c in r.checks
            if c.check_name == "factual_correctness" and c.score != -1.0
        ]
        avg_judge = (
            f"{sum(judge_scores)/len(judge_scores)*100:.0f}%"
            if judge_scores else "n/a"
        )

        print(_row([
            (cat, _COL["category"]),
            (str(total), 8),
            (str(passed), 8),
            (f"{rate:.0%}", 12),
            (avg_judge, 12),
        ]))

    # Overall
    total_all = len(results)
    passed_all = sum(1 for r in results if r.overall_passed)
    total_cost = sum(r.cost_usd for r in results)
    total_in = sum(r.total_tokens.get("input", 0) for r in results)
    total_out = sum(r.total_tokens.get("output", 0) for r in results)

    print()
    print(sep)
    print(
        f"Overall: {passed_all}/{total_all} passed "
        f"({passed_all/total_all:.0%})  "
        f"Cost: ${total_cost:.4f}  "
        f"Tokens: in={total_in:,} out={total_out:,}"
    )
    print()


# ---------------------------------------------------------------------------
# Diff vs baseline
# ---------------------------------------------------------------------------


def load_baseline(path: str | Path) -> dict | None:
    """Load a previous report JSON as a baseline. Returns None if not found."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def print_diff(results: list["EvalResult"], baseline: dict) -> None:
    """Compare current results against a baseline report and print regressions/fixes."""
    # Build lookup: case_id -> {check_name -> passed}
    def _index(cases: list[dict]) -> dict[str, dict]:
        out = {}
        for c in cases:
            out[c["case_id"]] = {
                "overall": c["overall_passed"],
                "checks": {ch["check_name"]: ch["passed"] for ch in c.get("checks", [])},
            }
        return out

    baseline_idx = _index(baseline.get("cases", []))
    current_idx = _index([
        {
            "case_id": r.case_id,
            "overall_passed": r.overall_passed,
            "checks": [{"check_name": c.check_name, "passed": c.passed} for c in r.checks],
        }
        for r in results
    ])

    regressions: list[tuple[str, list[str]]] = []  # (case_id, [check_names that broke])
    fixes: list[str] = []
    new_cases: list[str] = []

    for case_id, cur in current_idx.items():
        if case_id not in baseline_idx:
            new_cases.append(case_id)
            continue
        base = baseline_idx[case_id]

        if base["overall"] and not cur["overall"]:
            # Find which specific checks newly failed
            broke = [
                name for name, passed in cur["checks"].items()
                if not passed and base["checks"].get(name, True)
            ]
            regressions.append((case_id, broke))
        elif not base["overall"] and cur["overall"]:
            fixes.append(case_id)

    if not regressions and not fixes and not new_cases:
        print("Diff vs baseline: no changes detected.\n")
        return

    sep = "-" * 60
    print()
    print("Diff vs baseline")
    print(sep)

    if regressions:
        print(f"REGRESSIONS ({len(regressions)}) — were passing, now failing:")
        for case_id, broke in regressions:
            checks_str = ", ".join(broke) if broke else "overall"
            print(f"  ✗ {case_id}: {checks_str}")

    if fixes:
        print(f"FIXES ({len(fixes)}) — were failing, now passing:")
        for case_id in fixes:
            print(f"  ✓ {case_id}")

    if new_cases:
        print(f"NEW ({len(new_cases)}) — not in baseline:")
        for case_id in new_cases:
            print(f"  + {case_id}")

    print()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def _check_to_dict(c) -> dict:
    return {
        "check_name": c.check_name,
        "passed": c.passed,
        "score": c.score,
        "detail": c.detail,
    }


def save_report(results: list["EvalResult"], path: str | Path) -> None:
    by_cat: dict[str, list] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    category_summary = {}
    for cat, cases in by_cat.items():
        total = len(cases)
        passed = sum(1 for r in cases if r.overall_passed)
        judge_scores = [
            c.score for r in cases for c in r.checks
            if c.check_name == "factual_correctness" and c.score != -1.0
        ]
        category_summary[cat] = {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_judge_score": (
                round(sum(judge_scores) / len(judge_scores), 4)
                if judge_scores else None
            ),
        }

    total_all = len(results)
    passed_all = sum(1 for r in results if r.overall_passed)
    total_cost = sum(r.cost_usd for r in results)
    total_in = sum(r.total_tokens.get("input", 0) for r in results)
    total_out = sum(r.total_tokens.get("output", 0) for r in results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cases": total_all,
            "passed": passed_all,
            "failed": total_all - passed_all,
            "pass_rate": round(passed_all / total_all, 4) if total_all else 0.0,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": {"input": total_in, "output": total_out},
            "by_category": category_summary,
        },
        "cases": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "question": r.question,
                "run_id": r.run_id,
                "stopped_reason": r.stopped_reason,
                "cost_usd": round(r.cost_usd, 6),
                "total_tokens": r.total_tokens,
                "final_answer": r.final_answer,
                "citations": r.citations,
                "overall_passed": r.overall_passed,
                "passed_count": r.passed_count,
                "total_runs": r.total_runs,
                "error": r.error,
                "checks": [_check_to_dict(c) for c in r.checks],
            }
            for r in results
        ],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {path}")
