"""가상 호기 간트 HTML."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List

from core.domain import AllocationSet, SchedulingProblem

from .virtual_eqp import (
    build_gantt_segments,
    expand_allocation_to_virtual,
    gantt_config,
)


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _color_for_pk(pk: str) -> str:
    palette = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ]
    return palette[hash(pk) % len(palette)]


def render_gantt_html(
    segments: List,
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

    row_ids = sorted({s.virtual_eqp_id for s in segments})
    slot_headers = "".join(
        f"<th class='slot-h'>{i * slot_hours:.0f}-{(i + 1) * slot_hours:.0f}h</th>"
        for i in range(num_slots)
    )

    rows_html = []
    for rid in row_ids:
        segs = [s for s in segments if s.virtual_eqp_id == rid]
        s0 = segs[0]
        label = f"{rid}<br><small>{_esc(s0.eqp_model_cd)} / {_esc(s0.batch_id)}</small>"
        cells = [f"<td class='eqp-label'>{label}</td>"]
        for t in range(num_slots):
            s = segs[0] if segs else None
            if s is None or not s.plan_prod_key:
                cells.append("<td class='slot empty'></td>")
                continue
            title = f"{s.plan_prod_key} / {s.oper_id}"
            bg = _color_for_pk(s.plan_prod_key)
            cells.append(
                f"<td class='slot' title='{_esc(title)}'>"
                f"<div class='bar' style='background:{bg}'></div></td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Gantt — {_esc(rule_timekey)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1rem;color:#222}}
.meta{{color:#555;font-size:.9rem;margin-bottom:1rem}}
.gantt-wrap{{overflow-x:auto}}
table.gantt{{border-collapse:collapse;min-width:900px}}
.gantt th,.gantt td{{border:1px solid #ddd;padding:0}}
.gantt th{{background:#f5f5f5;font-size:.7rem;padding:.25rem}}
.eqp-label{{min-width:140px;padding:.4rem .5rem;font-size:.8rem;vertical-align:middle}}
.slot{{width:28px;height:28px;padding:0;vertical-align:middle}}
.slot .bar{{height:22px;margin:2px;border-radius:2px}}
.slot.empty{{background:#fafafa}}
.slot-h{{min-width:28px}}
</style></head><body>
<h1>가상 호기 간트 (24h)</h1>
<p class="meta">RULE_TIMEKEY: <strong>{_esc(rule_timekey)}</strong> |
FAC_ID: {_esc(fac_id)} | mode: {_esc(mode)} |
슬롯: {num_slots} × {slot_hours:g}h</p>
<p class="meta">가상호기 ID = V-모델@배치#순번 (eqp_qty 확장)</p>
<div class="gantt-wrap">
<table class="gantt">
<thead><tr><th>가상호기</th>{slot_headers}</tr></thead>
<tbody>{''.join(rows_html) or '<tr><td colspan="25">No allocation</td></tr>'}</tbody>
</table>
</div>
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
    del problem  # reserved for future slot-varying schedule
    num_slots, slot_hours = gantt_config(settings)
    units = expand_allocation_to_virtual(allocation)
    segments = build_gantt_segments(units, num_slots=num_slots)
    path = render_gantt_html(
        segments, output_path,
        rule_timekey=rule_timekey,
        num_slots=num_slots,
        slot_hours=slot_hours,
        fac_id=fac_id,
        mode=mode,
    )
    return {
        "gantt_html": path,
        "virtual_eqp_count": len(units),
        "gantt_slots": num_slots,
        "slot_hours": slot_hours,
    }


def resolve_gantt_path(settings: dict, rule_timekey: str, mode: str) -> Path:
    infer = settings.get("infer", {})
    report_dir = Path(infer.get("report_dir", "artifacts/reports"))
    return report_dir / f"gantt_{rule_timekey}_{mode}.html"
