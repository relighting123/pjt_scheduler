"""Post-infer summary: per-target achievement, eqp counts, daily capacity."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.domain import AllocationSet, SchedulingProblem
from core.sim.simulator import Simulator


def _infer_horizon_hours(settings: dict, mode: str) -> float:
    if mode == "dynamic":
        dyn = settings.get("dynamic", {})
        return float(dyn.get("slot_hours", 1.0))
    return float(settings.get("infer", {}).get("horizon_hours", 1.0))


def _hours_per_day(settings: dict) -> float:
    return float(settings.get("infer", {}).get("hours_per_day", 24.0))


def build_infer_report(
    problem: SchedulingProblem,
    allocation: AllocationSet,
    settings: dict,
    mode: str,
) -> Dict[str, Any]:
    """(batch, pk, op)별 달성률·모델별 댓수·일 capa 합산."""
    horizon = _infer_horizon_hours(settings, mode)
    hours_day = _hours_per_day(settings)
    ignore_wip = mode == "plan-only"
    sim = Simulator(problem, horizon_hours=horizon, ignore_wip=ignore_wip)
    sim_result = sim.simulate(allocation)

    targets: List[Dict[str, Any]] = []
    total_daily_capacity = 0.0
    total_produced = 0.0
    capa_scale = horizon / hours_day if hours_day > 0 else 0.0

    for pk, op, plan_qty in sorted(problem.plan_targets()):
        batch_id = problem.batch_of(pk, op)
        eqp_by_model: Dict[str, int] = {}
        daily_capacity = 0.0
        for a in allocation.allocations:
            if a.plan_prod_key != pk or a.oper_id != op:
                continue
            qty = max(0, int(a.eqp_qty))
            if qty == 0:
                continue
            eqp_by_model[a.eqp_model_cd] = eqp_by_model.get(a.eqp_model_cd, 0) + qty
            uph = problem.uph_of(pk, op, a.eqp_model_cd)
            daily_capacity += uph * qty * hours_day

        pk_op = (pk, op)
        produced_qty = float(sim_result.produced_by_pko.get(pk_op, 0.0))
        horizon_capacity = daily_capacity * capa_scale
        utilization_rate = (
            min(1.0, produced_qty / horizon_capacity)
            if horizon_capacity > 0
            else 0.0
        )
        targets.append({
            "batch_id": batch_id,
            "plan_prod_key": pk,
            "oper_id": op,
            "plan_qty": float(plan_qty),
            "produced_qty": produced_qty,
            "achievement_rate": float(sim_result.achievement_by_pko.get(pk_op, 0.0)),
            "utilization_rate": utilization_rate,
            "eqp_qty_by_model": eqp_by_model,
            "daily_capacity": daily_capacity,
            "horizon_capacity": horizon_capacity,
        })
        total_daily_capacity += daily_capacity
        total_produced += produced_qty

    total_horizon_capacity = total_daily_capacity * capa_scale
    avg_utilization = (
        min(1.0, total_produced / total_horizon_capacity)
        if total_horizon_capacity > 0
        else 0.0
    )

    return {
        "horizon_hours": horizon,
        "hours_per_day": hours_day,
        "avg_achievement": float(sim_result.avg_achievement),
        "avg_utilization": avg_utilization,
        "total_daily_capacity": total_daily_capacity,
        "total_horizon_capacity": total_horizon_capacity,
        "targets": targets,
    }


def format_infer_report_log(report: Dict[str, Any]) -> str:
    """운영 로그용 텍스트."""
    lines = [
        "--- infer report ---",
        f"avg_achievement: {report['avg_achievement']:.4f}",
        f"avg_utilization: {report.get('avg_utilization', 0.0):.4f}",
        f"total_daily_capacity: {report['total_daily_capacity']:.1f}",
        "batch_id | plan_prod_key | oper_id | achv | util | eqp_by_model | daily_capa",
    ]
    for t in report.get("targets", []):
        models = ", ".join(
            f"{m}:{q}" for m, q in sorted(t.get("eqp_qty_by_model", {}).items())
        ) or "-"
        lines.append(
            f"{t.get('batch_id', '')} | {t['plan_prod_key']} | {t['oper_id']} | "
            f"{t['achievement_rate']:.4f} | {t.get('utilization_rate', 0.0):.4f} | "
            f"[{models}] | {t['daily_capacity']:.1f}"
        )
    return "\n".join(lines)


def resolve_infer_report_path(
    settings: dict,
    rule_timekey: str,
    mode: str,
    override: Optional[str] = None,
) -> Path:
    """infer KPI HTML 출력 경로."""
    if override:
        return Path(override)
    infer_cfg = settings.get("infer", {})
    template = infer_cfg.get("report_path")
    if template:
        return Path(
            str(template).format(rule_timekey=rule_timekey, mode=mode),
        )
    report_dir = Path(infer_cfg.get("report_dir", "artifacts/reports"))
    return report_dir / f"infer_{rule_timekey}_{mode}.html"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(rate: float) -> str:
    return f"{100.0 * rate:.1f}%"


def render_infer_report_html(
    report: Dict[str, Any],
    output_path: str | Path,
    *,
    rule_timekey: str,
    mode: str,
    fac_id: str = "",
    source: str = "oracle",
    rows: int = 0,
    allocation_count: int = 0,
    input_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """infer KPI를 HTML 리포트로 저장."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    avg = float(report.get("avg_achievement", 0.0))
    avg_util = float(report.get("avg_utilization", 0.0))
    total_capa = float(report.get("total_daily_capacity", 0.0))
    total_horizon_capa = float(report.get("total_horizon_capacity", 0.0))
    horizon = float(report.get("horizon_hours", 1.0))
    hours_day = float(report.get("hours_per_day", 24.0))

    target_rows = []
    for t in report.get("targets", []):
        models = ", ".join(
            f"{_esc(m)}:{q}" for m, q in sorted(t.get("eqp_qty_by_model", {}).items())
        ) or "-"
        achv = float(t.get("achievement_rate", 0.0))
        util = float(t.get("utilization_rate", 0.0))
        achv_cls = "kpi-good" if achv >= 0.99 else ("kpi-warn" if achv >= 0.5 else "kpi-bad")
        util_cls = "kpi-good" if util >= 0.99 else ("kpi-warn" if util >= 0.5 else "kpi-bad")
        target_rows.append(
            "<tr>"
            f"<td>{_esc(t.get('batch_id', ''))}</td>"
            f"<td>{_esc(t['plan_prod_key'])}</td>"
            f"<td>{_esc(t['oper_id'])}</td>"
            f"<td class='num'>{t.get('plan_qty', 0):.1f}</td>"
            f"<td class='num'>{t.get('produced_qty', 0):.1f}</td>"
            f"<td class='num {achv_cls}'>{_pct(achv)}</td>"
            f"<td class='num {util_cls}'>{_pct(util)}</td>"
            f"<td>{models}</td>"
            f"<td class='num'>{t.get('daily_capacity', 0):.1f}</td>"
            "</tr>"
        )

    summary = input_summary or {}
    summary_items = "".join(
        f"<li><strong>{_esc(k)}</strong>: {_esc(v)}</li>"
        for k, v in sorted(summary.items())
    )

    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Infer KPI — {_esc(rule_timekey)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;color:#222;padding:0 1rem}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:1.5rem 0}}
.kpi{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:.5rem;padding:1rem 1.1rem}}
.kpi label{{display:block;font-size:.8rem;color:#666;margin-bottom:.35rem}}
.kpi .value{{font-size:1.45rem;font-weight:600}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.5rem .75rem}}
th{{background:#f5f5f5;text-align:center}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
td:first-child,th:first-child{{text-align:left}}
.kpi-good{{color:#0a7;font-weight:600}}
.kpi-warn{{color:#b80;font-weight:600}}
.kpi-bad{{color:#c33;font-weight:600}}
.meta{{color:#555;font-size:.95rem}}
h2{{margin-top:2rem}}
ul.summary{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:.5rem;padding:.75rem 1.25rem}}
</style></head><body>
<h1>Inference KPI Report</h1>
<p class="meta">Generated: {_esc(datetime.now().isoformat(timespec='seconds'))}<br>
RULE_TIMEKEY: <strong>{_esc(rule_timekey)}</strong> &nbsp;|&nbsp;
FAC_ID: <strong>{_esc(fac_id)}</strong> &nbsp;|&nbsp;
Mode: <strong>{_esc(mode)}</strong> &nbsp;|&nbsp;
Source: {_esc(source)}</p>

<div class="kpi-grid">
  <div class="kpi"><label>평균 달성률</label><div class="value">{_pct(avg)}</div></div>
  <div class="kpi"><label>평균 가동률</label><div class="value">{_pct(avg_util)}</div></div>
  <div class="kpi"><label>합산 일 Capa</label><div class="value">{total_capa:,.0f}</div></div>
  <div class="kpi"><label>합산 horizon Capa</label><div class="value">{total_horizon_capa:,.0f}</div></div>
  <div class="kpi"><label>전환 출력 행</label><div class="value">{rows}</div></div>
  <div class="kpi"><label>할당 건수</label><div class="value">{allocation_count}</div></div>
  <div class="kpi"><label>시뮬 horizon (h)</label><div class="value">{horizon:g}</div></div>
  <div class="kpi"><label>일 Capa 기준 (h)</label><div class="value">{hours_day:g}</div></div>
</div>

<h2>입력 요약</h2>
<ul class="summary">{summary_items or '<li>—</li>'}</ul>

<h2>공정별 KPI</h2>
<table>
<thead><tr>
<th>batch_id</th><th>plan_prod_key</th><th>oper_id</th>
<th>plan_qty</th><th>produced</th><th>달성률</th><th>가동률</th><th>eqp (model:qty)</th><th>일 capa</th>
</tr></thead>
<tbody>{''.join(target_rows) or '<tr><td colspan="9">No plan targets</td></tr>'}</tbody>
</table>
</body></html>
"""
    path.write_text(body, encoding="utf-8")
    return str(path)
