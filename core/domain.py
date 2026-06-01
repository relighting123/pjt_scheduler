"""Domain models for equipment transition scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class OperWip:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    oper_seq: int
    wip_qty: float


@dataclass
class ModelUph:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    uph: float


@dataclass
class ModelAvail:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    avail_yn: str


@dataclass
class EqpCount:
    rule_timekey: str
    batch_id: str
    eqp_model_cd: str
    eqp_qty: int


@dataclass
class ToolQty:
    rule_timekey: str
    batch_id: str
    eqp_model_cd: str
    tool_qty: int


@dataclass
class BatchOper:
    rule_timekey: str
    batch_id: str
    plan_prod_key: str
    oper_id: str


@dataclass
class PlanSlot:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    start_time: str
    end_time: str
    plan_qty: float


@dataclass
class ConversionRecord:
    rule_timekey: str
    from_batch: str
    from_plan_prod_key: str
    from_oper_id: str
    eqp_model_cd: str
    to_batch_id: str
    to_plan_prod_key: str
    to_oper_id: str
    start_conv_time: str
    eqp_qty: int

    def to_row(self) -> dict[str, Any]:
        return {
            "RULE_TIMEKEY": self.rule_timekey,
            "FROM_BATCH": self.from_batch,
            "FROM_PLAN_PROD_KEY": self.from_plan_prod_key,
            "FROM_OPER_ID": self.from_oper_id,
            "EQP_MODEL_CD": self.eqp_model_cd,
            "TO_BATCH_ID": self.to_batch_id,
            "TO_PLAN_PROD_KEY": self.to_plan_prod_key,
            "TO_OPER_ID": self.to_oper_id,
            "START_CONV_TIME": self.start_conv_time,
            "EQP_QTY": self.eqp_qty,
        }


@dataclass
class SchedulingDataset:
    """In-memory scheduling inputs for one RULE_TIMEKEY snapshot."""

    rule_timekey: str
    oper_wip: list[OperWip] = field(default_factory=list)
    model_uph: list[ModelUph] = field(default_factory=list)
    eqp_counts: list[EqpCount] = field(default_factory=list)
    model_avail: list[ModelAvail] = field(default_factory=list)
    batch_opers: list[BatchOper] = field(default_factory=list)
    tool_qty: list[ToolQty] = field(default_factory=list)
    plan_slots: list[PlanSlot] = field(default_factory=list)

    def batch_for(self, plan_prod_key: str, oper_id: str) -> str | None:
        for bo in self.batch_opers:
            if bo.plan_prod_key == plan_prod_key and bo.oper_id == oper_id:
                return bo.batch_id
        return None

    def uph(self, plan_prod_key: str, oper_id: str, eqp_model_cd: str) -> float | None:
        for row in self.model_uph:
            if (
                row.plan_prod_key == plan_prod_key
                and row.oper_id == oper_id
                and row.eqp_model_cd == eqp_model_cd
            ):
                return row.uph
        return None

    def is_available(self, plan_prod_key: str, oper_id: str, eqp_model_cd: str) -> bool:
        for row in self.model_avail:
            if (
                row.plan_prod_key == plan_prod_key
                and row.oper_id == oper_id
                and row.eqp_model_cd == eqp_model_cd
            ):
                return str(row.avail_yn).upper() in ("Y", "YES", "1", "TRUE")
        return False

    def total_eqp_qty(self, eqp_model_cd: str) -> int:
        return sum(e.eqp_qty for e in self.eqp_counts if e.eqp_model_cd == eqp_model_cd)

    def oper_keys(self) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for slot in self.plan_slots:
            key = (slot.plan_prod_key, slot.oper_id)
            if key not in seen:
                seen.add(key)
                keys.append(key)
        for bo in self.batch_opers:
            key = (bo.plan_prod_key, bo.oper_id)
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    def plan_qty_by_oper(self) -> dict[tuple[str, str], float]:
        totals: dict[tuple[str, str], float] = {}
        for slot in self.plan_slots:
            key = (slot.plan_prod_key, slot.oper_id)
            totals[key] = totals.get(key, 0.0) + slot.plan_qty
        return totals

    def validate(self) -> list[str]:
        errors: list[str] = []
        for key in self.oper_keys():
            plan_prod_key, oper_id = key
            models = {m.eqp_model_cd for m in self.model_uph if m.plan_prod_key == plan_prod_key and m.oper_id == oper_id}
            if not models:
                errors.append(f"Missing UPH for {plan_prod_key}/{oper_id}")
            for model in models:
                if self.uph(plan_prod_key, oper_id, model) is None:
                    errors.append(f"UPH missing: {plan_prod_key}/{oper_id}/{model}")
        return errors

    @classmethod
    def from_csv_dir(cls, directory: str | Path, rule_timekey: str | None = None) -> SchedulingDataset:
        base = Path(directory)
        meta_path = base / "meta.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            rtk = str(meta.iloc[0]["RULE_TIMEKEY"]) if rule_timekey is None else rule_timekey
        else:
            rtk = rule_timekey or "2026051707000000"

        def _read(name: str) -> pd.DataFrame:
            path = base / name
            if not path.exists():
                return pd.DataFrame()
            return pd.read_csv(path)

        ds = cls(rule_timekey=rtk)

        wip_df = _read("oper_wip.csv")
        for _, r in wip_df.iterrows():
            ds.oper_wip.append(
                OperWip(
                    rule_timekey=str(r.get("RULE_TIMEKEY", rtk)),
                    plan_prod_key=str(r["PLAN_PROD_KEY"]),
                    oper_id=str(r["OPER_ID"]),
                    oper_seq=int(r["OPER_SEQ"]),
                    wip_qty=float(r["WIP_QTY"]),
                )
            )

        uph_df = _read("model_uph.csv")
        for _, r in uph_df.iterrows():
            ds.model_uph.append(
                ModelUph(
                    rule_timekey=str(r.get("RULE_TIMEKEY", rtk)),
                    plan_prod_key=str(r["PLAN_PROD_KEY"]),
                    oper_id=str(r["OPER_ID"]),
                    eqp_model_cd=str(r["EQP_MODEL_CD"]),
                    uph=float(r["UPH"]),
                )
            )

        eqp_df = _read("eqp_count.csv")
        for _, r in eqp_df.iterrows():
            ds.eqp_counts.append(
                EqpCount(
                    rule_timekey=str(r.get("RULE_TIMEKEY", rtk)),
                    batch_id=str(r["BATCH_ID"]),
                    eqp_model_cd=str(r["EQP_MODEL_CD"]),
                    eqp_qty=int(r["EQP_QTY"]),
                )
            )

        avail_df = _read("model_avail.csv")
        for _, r in avail_df.iterrows():
            ds.model_avail.append(
                ModelAvail(
                    rule_timekey=str(r.get("RULE_TIMEKEY", rtk)),
                    plan_prod_key=str(r["PLAN_PROD_KEY"]),
                    oper_id=str(r["OPER_ID"]),
                    eqp_model_cd=str(r["EQP_MODEL_CD"]),
                    avail_yn=str(r["AVAIL_YN"]),
                )
            )

        batch_df = _read("batch_oper.csv")
        for _, r in batch_df.iterrows():
            ds.batch_opers.append(
                BatchOper(
                    rule_timekey=str(r.get("RULE_TIMEKEY", rtk)),
                    batch_id=str(r["BATCH_ID"]),
                    plan_prod_key=str(r["PLAN_PROD_KEY"]),
                    oper_id=str(r["OPER_ID"]),
                )
            )

        tool_df = _read("tool_qty.csv")
        for _, r in tool_df.iterrows():
            ds.tool_qty.append(
                ToolQty(
                    rule_timekey=str(r.get("RULE_TIMEKEY", rtk)),
                    batch_id=str(r["BATCH_ID"]),
                    eqp_model_cd=str(r["EQP_MODEL_CD"]),
                    tool_qty=int(r["TOOL_QTY"]),
                )
            )

        plan_df = _read("plan_slots.csv")
        for _, r in plan_df.iterrows():
            ds.plan_slots.append(
                PlanSlot(
                    rule_timekey=str(r.get("RULE_TIMEKEY", rtk)),
                    plan_prod_key=str(r["PLAN_PROD_KEY"]),
                    oper_id=str(r["OPER_ID"]),
                    start_time=str(r["START_TIME"]),
                    end_time=str(r["END_TIME"]),
                    plan_qty=float(r["PLAN_QTY"]),
                )
            )

        return ds

    def to_conversions_df(self, records: list[ConversionRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=[
                    "RULE_TIMEKEY",
                    "FROM_BATCH",
                    "FROM_PLAN_PROD_KEY",
                    "FROM_OPER_ID",
                    "EQP_MODEL_CD",
                    "TO_BATCH_ID",
                    "TO_PLAN_PROD_KEY",
                    "TO_OPER_ID",
                    "START_CONV_TIME",
                    "EQP_QTY",
                ]
            )
        return pd.DataFrame([r.to_row() for r in records])
