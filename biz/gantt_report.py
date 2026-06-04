"""가상 호기 간트 HTML (SEQ × START/END 축)."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List

from core.domain import AllocationSet, SchedulingProblem

from .schedule_output import build_schedule_rows
from .virtual_eqp import gantt_config


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def render_gantt_html(
    schedule_rows: List,
    output_path: str | Path,
    *,
    rule_timekey: str,
    num_slots: int,
    slot_hours: float,
    fac_id: str = "",
    mode: str = "",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body_rows = []
    for r in schedule_rows:
        body_rows.append(
            "<tr>"
            f"<td>{_esc(r.eqp_id)}</td>"
            f"<td class='num'>{r.seq_no}</td>"
            f"<td>{_esc(r.plan_prod_key)}</td>"
            f"<td>{_esc(r.start_time)}</td>"
            f"<td>{_esc(r.end_time)}</td>"
            f"<td class='num'>{r.produce_qty:g}</td>"
            "</tr>"
        )

    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Gantt — {_esc(rule_timekey)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1rem;color:#222}}
.meta{{color:#555;font-size:.9rem;margin-bottom:1rem}}
table.schedule{{border-collapse:collapse;width:100%;max-width:1100px}}
table.schedule th,table.schedule td{{border:1px solid #ddd;padding:.45rem .6rem;font-size:.85rem}}
table.schedule th{{background:#f5f5f5;text-align:left}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body>
<h1>호기 스케줄 (SEQ)</h1>
<p class="meta">RULE_TIMEKEY: <strong>{_esc(rule_timekey)}</strong> |
FAC_ID: {_esc(fac_id)} | mode: {_esc(mode)} |
호라이즌: {num_slots} × {slot_hours:g}h</p>
<p class="meta">동일 EQP_ID·연속 동일 제품 → 동일 SEQ_NO. 제품 변경 시 SEQ 증가.</p>
<table class="schedule">
<thead><tr>
<th>EQP_ID</th><th>SEQ_NO</th><th>PLAN_PROD_KEY</th>
<th>START_TIME</th><th>END_TIME</th><th>PRODUCE_QTY</th>
</tr></thead>
<tbody>{''.join(body_rows) or '<tr><td colspan="6">No schedule</td></tr>'}</tbody>
</table>
</body></html>
"""
    path.write_text(body, encoding="utf-8")
    return str(path)


def build_and_render_gantt(
    problem: SchedulingProblem,
    allocation: AllocationSet,
    settings: dict,
    output_path: str | Path,
    *,
    rule_timekey: str,
    mode: str,
    fac_id: str = "",
) -> Dict[str, Any]:
    num_slots, slot_hours = gantt_config(settings)
    schedule_rows = build_schedule_rows(problem, allocation, settings, mode=mode)
    path = render_gantt_html(
        schedule_rows, output_path,
        rule_timekey=rule_timekey,
        num_slots=num_slots,
        slot_hours=slot_hours,
        fac_id=fac_id,
        mode=mode,
    )
    from .virtual_eqp import expand_allocation_to_virtual
    units = expand_allocation_to_virtual(allocation)
    return {
        "gantt_html": path,
        "virtual_eqp_count": len(units),
        "gantt_slots": num_slots,
        "slot_hours": slot_hours,
        "schedule_row_count": len(schedule_rows),
    }


def resolve_gantt_path(settings: dict, rule_timekey: str, mode: str) -> Path:
    infer = settings.get("infer", {})
    report_dir = Path(infer.get("report_dir", "artifacts/reports"))
    return report_dir / f"gantt_{rule_timekey}_{mode}.html"
