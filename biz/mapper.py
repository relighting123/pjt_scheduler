"""Map RTS_LINEDSDB_INF rows into SchedulingDataset."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from core.domain import (
    BatchOper,
    EqpCount,
    ModelAvail,
    ModelUph,
    OperWip,
    PlanSlot,
    SchedulingDataset,
    ToolQty,
)


def _parse_hour_key(rule_timekey: str) -> datetime:
    s = rule_timekey[:14].ljust(14, "0")
    return datetime.strptime(s, "%Y%m%d%H%M%S")


def _hour_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H")


def _next_day_07(rule_timekey: str) -> datetime:
    base = _parse_hour_key(rule_timekey)
    next_day = (base + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
    if base.hour >= 7:
        next_day = (base + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
    else:
        next_day = base.replace(hour=7, minute=0, second=0, microsecond=0)
        if next_day <= base:
            next_day += timedelta(days=1)
    return next_day


def rows_to_dataset(rows: list[dict[str, Any]], rule_timekey: str) -> SchedulingDataset:
    """Convert flat GBN_CD attribute rows into structured dataset."""
    ds = SchedulingDataset(rule_timekey=rule_timekey)
    by_key: dict[tuple, dict[str, str]] = defaultdict(dict)

    for row in rows:
        rtk = str(row.get("RULE_TIMEKEY", rule_timekey))
        fac = str(row.get("FAC_ID", ""))
        batch_id = str(row.get("BATCH_ID", ""))
        plan = str(row.get("PLAN_PROD_KEY", ""))
        oper = str(row.get("OPER_ID", ""))
        model = str(row.get("EQP_MODEL_CD", ""))
        gbn = str(row.get("GBN_CD", ""))
        val = str(row.get("ATTR_VAL", ""))
        oper_seq = row.get("OPER_SEQ")
        key = (rtk, fac, batch_id, plan, oper, model)
        by_key[key][gbn] = val
        if oper_seq is not None:
            by_key[key]["OPER_SEQ"] = str(oper_seq)

    batches_seen: set[tuple[str, str, str]] = set()
    for (rtk, _fac, batch_id, plan, oper, model), attrs in by_key.items():
        if plan and oper and batch_id:
            bk = (batch_id, plan, oper)
            if bk not in batches_seen:
                batches_seen.add(bk)
                ds.batch_opers.append(
                    BatchOper(rule_timekey=rtk, batch_id=batch_id, plan_prod_key=plan, oper_id=oper)
                )

        if attrs.get("WIP_QTY"):
            seq = int(float(attrs.get("OPER_SEQ", 1)))
            ds.oper_wip.append(
                OperWip(rule_timekey=rtk, plan_prod_key=plan, oper_id=oper, oper_seq=seq, wip_qty=float(attrs["WIP_QTY"]))
            )

        if attrs.get("UPH"):
            ds.model_uph.append(
                ModelUph(rule_timekey=rtk, plan_prod_key=plan, oper_id=oper, eqp_model_cd=model, uph=float(attrs["UPH"]))
            )
            ds.model_avail.append(
                ModelAvail(rule_timekey=rtk, plan_prod_key=plan, oper_id=oper, eqp_model_cd=model, avail_yn="Y")
            )

        if attrs.get("ASSIGN_EQUIP_CNT"):
            ds.eqp_counts.append(
                EqpCount(
                    rule_timekey=rtk,
                    batch_id=batch_id,
                    eqp_model_cd=model,
                    eqp_qty=int(float(attrs["ASSIGN_EQUIP_CNT"])),
                )
            )

        if attrs.get("TOOL_QTY"):
            ds.tool_qty.append(
                ToolQty(
                    rule_timekey=rtk,
                    batch_id=batch_id,
                    eqp_model_cd=model,
                    tool_qty=int(float(attrs["TOOL_QTY"])),
                )
            )

    base_dt = _parse_hour_key(rule_timekey)
    d0_end = _next_day_07(rule_timekey)
    d1_end = d0_end + timedelta(days=1)

    for (rtk, _fac, batch_id, plan, oper, model), attrs in by_key.items():
        if not plan or not oper:
            continue
        for gbn, end_dt, label in (
            ("D0_TARGET_QTY", d0_end, "D0"),
            ("D1_TARGET_QTY", d1_end, "D1"),
        ):
            if attrs.get(gbn):
                qty = float(attrs[gbn])
                start = _hour_str(base_dt if gbn == "D0_TARGET_QTY" else d0_end)
                end = _hour_str(end_dt)
                ds.plan_slots.append(
                    PlanSlot(
                        rule_timekey=rtk,
                        plan_prod_key=plan,
                        oper_id=oper,
                        start_time=start,
                        end_time=end,
                        plan_qty=qty,
                    )
                )

    return ds
