"""가상 호기 간트 HTML."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.domain import AllocationSet, SchedulingProblem

from .virtual_eqp import (
    GanttSegment,
    assign_virtual_units,
    build_gantt_segments,
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
    segments: List[GanttSegment],
    output_path: str | Path,
    *,
    rule_timekey: str,
    num_slots: int,
    slot_hours: float,
    fac_id: str = "",
    mode: str = "",
    constraint_notes: Optional[List[str]] = None,
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
        label = rid
        if segs:
            s0 = segs[0]
            label = f"{rid}<br><small>{_esc(s0.eqp_model_cd)} / {_esc(s0.batch_id)}</small>"
        cells = ["<td class='eqp-label'>" + label + "</td>"]
        occupied = [None] * num_slots
        for s in segs:
            for t in range(max(0, s.slot_start), min(num_slots, s.slot_end)):
                occupied[t] = s
        for t in range(num_slots):
            s = occupied[t]
            if s is None or not s.plan_prod_key:
                cells.append("<td class='slot empty'></td>")
                continue
            title = f"{s.plan_prod_key} / {s.oper_id}"
            if not s.allowed:
                title += f" [BLOCKED: {s.block_reason}]"
            bg = _color_for_pk(s.plan_prod_key) if s.allowed else "#ccc"
            border = "" if s.allowed else "border:2px dashed #c33;"
            cells.append(
                f"<td class='slot' title='{_esc(title)}'>"
                f"<div class='bar' style='background:{bg};{border}'></div></td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    notes = constraint_notes or []
    notes_html = "".join(f"<li>{_esc(n)}</li>" for n in notes) or "<li>가상 호기 = 모델×배치×순번. 호기별 제한은 settings.virtual_eqp.unit_rules</li>"

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
.slot-h{{writing-mode:horizontal-tb;min-width:28px}}
.legend{{margin:1rem 0;font-size:.85rem}}
h2{{margin-top:1.5rem;font-size:1.1rem}}
</style></head><body>
<h1>가상 호기 간트 (24h)</h1>
<p class="meta">RULE_TIMEKEY: <strong>{_esc(rule_timekey)}</strong> |
FAC_ID: {_esc(fac_id)} | mode: {_esc(mode)} |
슬롯: {num_slots} × {slot_hours:g}h</p>
<div class="gantt-wrap">
<table class="gantt">
<thead><tr><th>가상호기</th>{slot_headers}</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
</div>
<h2>호기별 제약 / 재공</h2>
<ul class="legend">{notes_html}</ul>
<p class="legend">빨간 점선 = 모델·재공·unit_rules 중 하나에 걸려 해당 (제품,공정) 투입 불가로 표시.</p>
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
    virtual_assignments = assign_virtual_units(problem, allocation, settings)
    segments = build_gantt_segments(
        virtual_assignments,
        num_slots=num_slots,
        slot_hours=slot_hours,
        rule_timekey=rule_timekey,
    )
    blocked = [va for va in virtual_assignments if not va.allowed]
    notes = [
        f"blocked virtual units: {len(blocked)} / {len(virtual_assignments)}",
    ]
    for va in blocked[:10]:
        notes.append(
            f"{va.virtual_eqp.virtual_eqp_id}: {va.block_reason or 'not allowed'}",
        )
    path = render_gantt_html(
        segments, output_path,
        rule_timekey=rule_timekey,
        num_slots=num_slots,
        slot_hours=slot_hours,
        fac_id=fac_id,
        mode=mode,
        constraint_notes=notes,
    )
    return {
        "gantt_html": path,
        "virtual_eqp_count": len(virtual_assignments),
        "blocked_count": len(blocked),
        "gantt_slots": num_slots,
        "slot_hours": slot_hours,
        "segments": [
            {
                "virtual_eqp_id": s.virtual_eqp_id,
                "plan_prod_key": s.plan_prod_key,
                "oper_id": s.oper_id,
                "slot_start": s.slot_start,
                "slot_end": s.slot_end,
                "allowed": s.allowed,
                "block_reason": s.block_reason,
            }
            for s in segments
        ],
    }


def resolve_gantt_path(settings: dict, rule_timekey: str, mode: str) -> Path:
    infer = settings.get("infer", {})
    report_dir = Path(infer.get("report_dir", "artifacts/reports"))
    return report_dir / f"gantt_{rule_timekey}_{mode}.html"
