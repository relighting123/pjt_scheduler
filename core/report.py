"""HTML + Markdown report generation from benchmark evaluation results."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from .evaluation import BenchmarkEvalResult

_HTML_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Scheduler Benchmark Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;color:#222;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.5rem .75rem;text-align:right}}
th{{background:#f5f5f5;text-align:center}}
td:first-child,th:first-child{{text-align:left}}
h2{{margin-top:2.5rem}}
.delta-pos{{color:#0a7}}.delta-neg{{color:#c33}}
.summary{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:.5rem;padding:1rem 1.25rem}}
</style></head><body>
<h1>Equipment Switching Scheduler — Benchmark Report</h1>
<p class="summary">Generated: {generated_at}<br>
Datasets: {n_datasets} &nbsp;|&nbsp; Avg Optimal: {avg_optimal:.3f} &nbsp;|&nbsp;
Avg RL: {avg_rl:.3f} &nbsp;|&nbsp; Avg Heuristic: {avg_heuristic:.3f}</p>

<h2>Summary per Benchmark</h2>
<table><thead><tr>
<th>Dataset</th>
<th>Optimal Achv.</th><th>RL Achv.</th><th>Heuristic Achv.</th>
<th>Optimal Switches</th><th>RL Switches</th>
<th>RL vs Optimal</th>
</tr></thead><tbody>
{summary_rows}
</tbody></table>

{detail_sections}

</body></html>
"""


def _fmt_delta(a: float, b: float) -> str:
    d = a - b
    cls = "delta-pos" if d >= 0 else "delta-neg"
    return f'<span class="{cls}">{d:+.3f}</span>'


def render_html(results: List[BenchmarkEvalResult], output_path: str) -> str:
    """벤치마크 평가 결과를 HTML 리포트로 렌더링.

    Args:
        results: evaluate_all_benchmark_datasets[_dynamic]의 출력.
        output_path: 출력 HTML 경로.

    Returns:
        출력 경로 (str).

    Example:
        render_html(results, "artifacts/reports/benchmark_wip_static.html")
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not results:
        Path(output_path).write_text("<html><body><p>No benchmark results.</p></body></html>")
        return output_path

    n = len(results)
    avg_opt = sum(r.optimal.avg_achievement for r in results) / n
    avg_rl = sum(r.rl.avg_achievement for r in results) / n
    avg_heu = sum(r.heuristic.avg_achievement for r in results) / n

    summary_rows = []
    for r in results:
        summary_rows.append(
            "<tr>"
            f"<td>{r.dataset}</td>"
            f"<td>{r.optimal.avg_achievement:.3f}</td>"
            f"<td>{r.rl.avg_achievement:.3f}</td>"
            f"<td>{r.heuristic.avg_achievement:.3f}</td>"
            f"<td>{r.optimal.switches}</td>"
            f"<td>{r.rl.switches}</td>"
            f"<td>{_fmt_delta(r.rl.avg_achievement, r.optimal.avg_achievement)}</td>"
            "</tr>"
        )

    detail_sections = []
    for r in results:
        rows = []
        keys = sorted(set(r.optimal.per_target) | set(r.rl.per_target))
        for pk, op in keys:
            opt_v = r.optimal.per_target.get((pk, op), 0.0)
            rl_v = r.rl.per_target.get((pk, op), 0.0)
            heu_v = r.heuristic.per_target.get((pk, op), 0.0)
            rows.append(
                "<tr>"
                f"<td>{pk} / {op}</td>"
                f"<td>{opt_v:.3f}</td>"
                f"<td>{rl_v:.3f}</td>"
                f"<td>{heu_v:.3f}</td>"
                "</tr>"
            )
        alloc_rows = []
        if r.rl.allocations:
            for a in r.rl.allocations.allocations:
                alloc_rows.append(
                    "<tr>"
                    f"<td>{a.batch_id}</td><td>{a.plan_prod_key}</td>"
                    f"<td>{a.oper_id}</td><td>{a.eqp_model_cd}</td>"
                    f"<td>{a.eqp_qty}</td></tr>"
                )
        detail_sections.append(
            f"<h2>{r.dataset}</h2>"
            "<h3>Per-target achievement</h3>"
            "<table><thead><tr><th>plan_prod_key / oper</th>"
            "<th>Optimal</th><th>RL</th><th>Heuristic</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<h3>RL Allocation</h3>"
            "<table><thead><tr><th>batch</th><th>plan_prod_key</th>"
            "<th>oper</th><th>model</th><th>qty</th></tr></thead>"
            f"<tbody>{''.join(alloc_rows)}</tbody></table>"
        )

    html = _HTML_TEMPLATE.format(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        n_datasets=n,
        avg_optimal=avg_opt,
        avg_rl=avg_rl,
        avg_heuristic=avg_heu,
        summary_rows="\n".join(summary_rows),
        detail_sections="\n".join(detail_sections),
    )
    Path(output_path).write_text(html)
    return output_path


def render_markdown(results: List[BenchmarkEvalResult], output_path: str) -> str:
    """평가 결과를 MODEL_BENCHMARK_<mode>.md 형식으로 출력.

    Example:
        render_markdown(results, "MODEL_BENCHMARK_wip_static.md")
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not results:
        Path(output_path).write_text("# Model Benchmark\n\n_No benchmark results yet._\n")
        return output_path
    n = len(results)
    avg_opt = sum(r.optimal.avg_achievement for r in results) / n
    avg_rl = sum(r.rl.avg_achievement for r in results) / n
    avg_heu = sum(r.heuristic.avg_achievement for r in results) / n

    lines = [
        "# Model Benchmark",
        "",
        f"_Updated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"- Datasets: **{n}**",
        f"- Avg Optimal achievement: **{avg_opt:.3f}**",
        f"- Avg RL achievement: **{avg_rl:.3f}**",
        f"- Avg Heuristic achievement: **{avg_heu:.3f}**",
        "",
        "| Dataset | Optimal | RL | Heuristic | Opt Switches | RL Switches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.dataset} | {r.optimal.avg_achievement:.3f} | {r.rl.avg_achievement:.3f} | "
            f"{r.heuristic.avg_achievement:.3f} | {r.optimal.switches} | {r.rl.switches} |"
        )
    lines.append("")
    Path(output_path).write_text("\n".join(lines))
    return output_path
