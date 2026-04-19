"""Plugin check: citation_count

Verifies the agent cited at least one source in its answer.
Demonstrates the plugin pattern — drop any *.py file in eval/plugins/
and decorate with @register_check. No edits to core files required.
"""

from checks import CheckResult, _FAIL, _PASS, _SKIP, register_check


@register_check
def check_citation_count(case, result) -> CheckResult:
    """Agent must cite at least one URL unless the case is unanswerable."""
    name = "citation_count"

    if case.get("category") in ("unanswerable",):
        return _SKIP(name)

    if not result.citations:
        return _FAIL(name, "Agent produced no citations")

    return CheckResult(
        name, passed=True, score=1.0,
        detail=f"{len(result.citations)} citation(s)"
    )
