"""eval/viewer.py — Generate a self-contained HTML trace viewer for one eval run.

Produces one HTML file per case at traces/eval-{case_id}.html.
The file has no external dependencies — all CSS and JS are inlined.
A human should be able to find the failing step in under 30 seconds.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runner import EvalResult

# ---------------------------------------------------------------------------
# CSS + JS (inlined, no external deps)
# ---------------------------------------------------------------------------

_STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; font-size: 14px; line-height: 1.6; }
a { color: #63b3ed; }

/* Header */
.header { background: #1a1d2e; border-bottom: 1px solid #2d3748;
          padding: 16px 24px; display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }
.header h1 { font-size: 16px; font-weight: 600; color: #f7fafc; flex: 1; min-width: 200px; }
.badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.pass  { background: #1a3a2a; color: #68d391; border: 1px solid #2f855a; }
.fail  { background: #3a1a1a; color: #fc8181; border: 1px solid #c53030; }
.skip  { background: #2a2a1a; color: #f6e05e; border: 1px solid #744210; }
.meta  { font-size: 12px; color: #718096; }

/* Layout */
.body { display: grid; grid-template-columns: 340px 1fr; height: calc(100vh - 65px); }

/* Checks panel */
.checks { background: #1a1d2e; border-right: 1px solid #2d3748;
          overflow-y: auto; padding: 16px; }
.checks h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
             color: #718096; margin-bottom: 12px; }
.check-row { display: flex; align-items: flex-start; gap: 8px;
             padding: 8px 10px; border-radius: 6px; margin-bottom: 4px; }
.check-row:hover { background: #2d3748; }
.check-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
.check-name { font-size: 13px; font-weight: 500; }
.check-detail { font-size: 11px; color: #a0aec0; margin-top: 2px; word-break: break-word; }

/* Timeline */
.timeline { overflow-y: auto; padding: 16px 24px; }
.timeline h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
               color: #718096; margin-bottom: 16px; }

.msg { margin-bottom: 10px; border-radius: 8px; border: 1px solid #2d3748;
       overflow: hidden; }
.msg-header { display: flex; align-items: center; gap: 10px;
              padding: 8px 14px; cursor: pointer; user-select: none; }
.msg-header:hover { background: rgba(255,255,255,.04); }
.role-tag { font-size: 11px; font-weight: 700; letter-spacing: .06em;
            padding: 2px 8px; border-radius: 4px; flex-shrink: 0; }
.role-system    { background: #2d3748; color: #a0aec0; }
.role-user      { background: #1a3a4a; color: #63b3ed; }
.role-assistant { background: #2a1a3a; color: #b794f4; }
.role-tool      { background: #1a2a1a; color: #68d391; }
.role-tool.err  { background: #3a1a1a; color: #fc8181; }
.tool-name { font-size: 12px; color: #a0aec0; }
.latency { font-size: 11px; color: #4a5568; margin-left: auto; }
.chevron { color: #4a5568; font-size: 12px; transition: transform .15s; flex-shrink: 0; }
.chevron.open { transform: rotate(90deg); }

.msg-body { display: none; border-top: 1px solid #2d3748; }
.msg-body.open { display: block; }
.section { padding: 10px 14px; }
.section + .section { border-top: 1px solid #1a2030; }
.section-label { font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
                 color: #4a5568; margin-bottom: 6px; }
pre { background: #0d1117; border: 1px solid #2d3748; border-radius: 6px;
      padding: 10px 12px; overflow-x: auto; font-size: 12px; color: #a0aec0;
      white-space: pre-wrap; word-break: break-word; }

/* Final answer box */
.answer-box { margin-bottom: 20px; background: #1a2030; border: 1px solid #2d3748;
              border-radius: 8px; padding: 14px 16px; }
.answer-box h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
                 color: #718096; margin-bottom: 8px; }
.answer-text { font-size: 13px; color: #e2e8f0; white-space: pre-wrap; }
.citations { margin-top: 10px; }
.citations a { display: block; font-size: 12px; margin-top: 3px; }
"""

_SCRIPT = """
function toggle(el) {
    const body = el.nextElementSibling;
    const chev = el.querySelector('.chevron');
    body.classList.toggle('open');
    chev.classList.toggle('open');
}
document.addEventListener('DOMContentLoaded', () => {
    // Auto-expand first non-system message
    const msgs = document.querySelectorAll('.msg-header');
    let first = true;
    msgs.forEach(h => {
        const role = h.dataset.role;
        if (first && role !== 'system') { toggle(h); first = false; }
    });
});
"""

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _e(s: Any) -> str:
    """HTML-escape a value for safe embedding."""
    return html.escape(str(s) if s is not None else "")


def _json_block(obj: Any) -> str:
    try:
        pretty = json.dumps(obj, indent=2, default=str)
    except Exception:
        pretty = str(obj)
    return f"<pre>{_e(pretty)}</pre>"


def _check_icon(score: float, passed: bool) -> str:
    if score == -1.0:
        return "–"
    return "✓" if passed else "✗"


def _check_row(c) -> str:
    if c.score == -1.0:
        cls = "skip"
        icon = "–"
    elif c.passed:
        cls = "pass"
        icon = "✓"
    else:
        cls = "fail"
        icon = "✗"

    detail_html = f'<div class="check-detail">{_e(c.detail)}</div>' if c.detail else ""
    return (
        f'<div class="check-row">'
        f'<span class="check-icon badge {cls}">{icon}</span>'
        f'<div><div class="check-name">{_e(c.check_name)}</div>{detail_html}</div>'
        f'</div>'
    )


def _role_tag(role: str, is_error: bool = False) -> str:
    err_cls = " err" if is_error else ""
    return f'<span class="role-tag role-{_e(role)}{err_cls}">{_e(role.upper())}</span>'


def _message_html(msg: dict, index: int) -> str:
    role = msg.get("role", "unknown")
    tool_calls = msg.get("tool_calls", [])
    tool_name = msg.get("name", "")
    latency = msg.get("latency_ms")
    content = msg.get("content")

    is_error = isinstance(content, dict) and "error" in content

    # Header summary
    summary_parts = []
    if tool_calls:
        names = ", ".join(tc.get("name", "?") for tc in tool_calls)
        summary_parts.append(f'<span class="tool-name">→ {_e(names)}</span>')
    if tool_name:
        summary_parts.append(f'<span class="tool-name">← {_e(tool_name)}</span>')

    latency_html = ""
    if latency is not None:
        latency_html = f'<span class="latency">{latency}ms</span>'

    summary_html = " ".join(summary_parts)

    header = (
        f'<div class="msg-header" onclick="toggle(this)" data-role="{_e(role)}">'
        f'{_role_tag(role, is_error)}'
        f'{summary_html}'
        f'{latency_html}'
        f'<span class="chevron">▶</span>'
        f'</div>'
    )

    # Body sections
    sections = []

    # Tool calls (from assistant)
    for tc in tool_calls:
        tc_name = tc.get("name", "?")
        args = tc.get("args", {})
        sections.append(
            f'<div class="section">'
            f'<div class="section-label">Tool call: {_e(tc_name)}</div>'
            f'{_json_block(args)}'
            f'</div>'
        )

    # Content
    if content is not None:
        label = "Result" if role == "tool" else "Content"
        if isinstance(content, str) and len(content) > 0:
            sections.append(
                f'<div class="section">'
                f'<div class="section-label">{label}</div>'
                f'<pre>{_e(content[:4000])}{"…" if len(content) > 4000 else ""}</pre>'
                f'</div>'
            )
        elif isinstance(content, (list, dict)):
            sections.append(
                f'<div class="section">'
                f'<div class="section-label">{label}</div>'
                f'{_json_block(content)}'
                f'</div>'
            )

    body_html = "".join(sections) if sections else '<div class="section"><span style="color:#4a5568">No content</span></div>'

    return (
        f'<div class="msg">'
        f'{header}'
        f'<div class="msg-body">{body_html}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Page builder
# ---------------------------------------------------------------------------


def _build_html(trace: dict, checks: list, overall_passed: bool) -> str:
    question = trace.get("question", "")
    final_answer = trace.get("final_answer") or ""
    citations = trace.get("citations", [])
    stopped_reason = trace.get("stopped_reason", "")
    model = trace.get("model", "")
    cost = trace.get("cost_usd", 0.0)
    tokens = trace.get("total_tokens", {})
    wall = trace.get("wall_time_ms", 0)
    messages = trace.get("messages", [])

    result_cls = "pass" if overall_passed else "fail"
    result_label = "PASS" if overall_passed else "FAIL"

    # Meta line
    meta = (
        f"model={_e(model)}  "
        f"tokens=in:{tokens.get('input',0):,}/out:{tokens.get('output',0):,}  "
        f"cost=${cost:.4f}  "
        f"wall={wall}ms  "
        f"stopped={_e(stopped_reason)}"
    )

    # Checks panel
    checks_html = "".join(_check_row(c) for c in checks)

    # Citations
    cite_html = ""
    if citations:
        links = "".join(f'<a href="{_e(u)}">{_e(u)}</a>' for u in citations)
        cite_html = f'<div class="citations"><b style="font-size:11px;color:#718096">CITATIONS</b>{links}</div>'

    # Final answer box
    answer_html = (
        f'<div class="answer-box">'
        f'<h3>Final Answer</h3>'
        f'<div class="answer-text">{_e(final_answer) if final_answer else "<em>none</em>"}</div>'
        f'{cite_html}'
        f'</div>'
    ) if final_answer else ""

    # Timeline messages
    timeline_html = "".join(_message_html(m, i) for i, m in enumerate(messages))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(question[:60])}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="header">
  <h1>{_e(question)}</h1>
  <span class="badge {result_cls}">{result_label}</span>
  <span class="meta">{_e(meta)}</span>
</div>
<div class="body">
  <div class="checks">
    <h2>Checks</h2>
    {checks_html}
  </div>
  <div class="timeline">
    <h2>Trace</h2>
    {answer_html}
    {timeline_html}
  </div>
</div>
<script>{_SCRIPT}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_viewer(
    result: "EvalResult",
    trace_path: Path,
    output_path: Path,
) -> None:
    """Generate an HTML viewer for one EvalResult.

    Reads the raw trace JSON from trace_path for full message content,
    combines it with the scored checks from result, and writes output_path.
    """
    if not trace_path.exists():
        return  # no trace to render

    with trace_path.open() as f:
        trace = json.load(f)

    html_content = _build_html(
        trace=trace,
        checks=result.checks,
        overall_passed=result.overall_passed,
    )

    output_path.write_text(html_content, encoding="utf-8")


def generate_index(results: list["EvalResult"], output_path: Path) -> None:
    """Generate an index HTML page listing all cases with links to their viewers."""
    rows = []
    for r in results:
        cls = "pass" if r.overall_passed else "fail"
        failed = [c.check_name for c in r.checks if not c.passed and c.score != -1.0]
        failed_str = ", ".join(failed) if failed else ""

        # Build label and links depending on whether this is a repeats run
        if r.sub_runs:
            n_passed = r.passed_count
            n_total = r.total_runs
            label = f"{n_passed}/{n_total}"
            badge_cls = "pass" if n_passed == n_total else "fail"
            links_html = " ".join(
                f'<a href="traces/eval-{_e(r.case_id)}-rep{i}.html">rep{i}</a>'
                for i in range(1, n_total + 1)
            )
            case_cell = f'{_e(r.case_id)}<br><span style="font-size:11px">{links_html}</span>'
        else:
            label = "PASS" if r.overall_passed else "FAIL"
            badge_cls = cls
            viewer_file = f"traces/eval-{r.case_id}.html"
            case_cell = f'<a href="{_e(viewer_file)}">{_e(r.case_id)}</a>'

        rows.append(
            f'<tr class="{cls}">'
            f'<td>{case_cell}</td>'
            f'<td>{_e(r.category)}</td>'
            f'<td><span class="badge {badge_cls}">{label}</span></td>'
            f'<td style="color:#fc8181;font-size:12px">{_e(failed_str)}</td>'
            f'<td>${r.cost_usd:.4f}</td>'
            f'</tr>'
        )

    rows_html = "\n".join(rows)
    total = len(results)
    passed = sum(1 for r in results if r.overall_passed)

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Report</title>
<style>
{_STYLE}
body {{ padding: 24px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
th {{ text-align: left; padding: 8px 12px; font-size: 11px; text-transform: uppercase;
     letter-spacing: .08em; color: #718096; border-bottom: 1px solid #2d3748; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #1a2030; font-size: 13px; }}
tr.pass:hover td {{ background: #0d1f0d; }}
tr.fail:hover td {{ background: #1f0d0d; }}
h1 {{ font-size: 18px; margin-bottom: 4px; }}
.summary {{ color: #a0aec0; font-size: 13px; margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>Eval Report</h1>
<div class="summary">{passed}/{total} passed</div>
<table>
<thead><tr>
  <th>Case ID</th><th>Category</th><th>Result</th><th>Failed checks</th><th>Cost</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>"""

    output_path.write_text(index_html, encoding="utf-8")
