"""Domain data classes shared by simulator, optimizer, and RL engine.

The biz layer materializes these from the physical Oracle table and feeds the
core simulator. Keep this module dependency-free so it can be imported in any
environment (including the no-DB benchmark harness).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class WipRecord:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    oper_seq: int
    wip_qty: float


@dataclass(frozen=True)
class UphRecord:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    uph: float


@dataclass(frozen=True)
class EquipmentRecord:
    rule_timekey: str
    batch_id: str
    eqp_model_cd: str
    eqp_qty: int


@dataclass(frozen=True)
class AvailabilityRecord:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    avail_yn: bool


@dataclass(frozen=True)
class ToolGroupRecord:
    """batch_id is determined by (plan_prod_key, oper_id). N:1 mapping."""
    rule_timekey: str
    batch_id: str
    plan_prod_key: str
    oper_id: str


@dataclass(frozen=True)
class ToolQtyRecord:
    rule_timekey: str
    batch_id: str
    eqp_model_cd: str
    tool_qty: int


@dataclass(frozen=True)
class PlanRecord:
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    start_time: str
    end_time: str
    plan_qty: float


@dataclass
class Allocation:
    """One equipment-model allocation to a (batch, plan_prod_key, oper)."""
    batch_id: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    eqp_qty: int


@dataclass
class AllocationSet:
    """Allocation decision plus the prior state, used to derive conversions."""
    rule_timekey: str
    allocations: List[Allocation] = field(default_factory=list)


@dataclass
class SchedulingProblem:
    """Materialized snapshot for one rule_timekey.

    Lookups are pre-built so the simulator/optimizer/RL env can stay hot-path
    cheap (dict/array ops only).
    """
    rule_timekey: str
    wip: List[WipRecord]
    uph: List[UphRecord]
    equipment: List[EquipmentRecord]
    availability: List[AvailabilityRecord]
    tool_groups: List[ToolGroupRecord]
    tool_qty: List[ToolQtyRecord]
    plans: List[PlanRecord]
    # batch group: conversion is restricted to model codes within the same group
    eqp_model_groups: Dict[str, List[str]] = field(default_factory=dict)

    # cached lookups
    _uph_lookup: Dict[Tuple[str, str, str], float] = field(default_factory=dict, init=False, repr=False)
    _avail_lookup: Dict[Tuple[str, str, str], bool] = field(default_factory=dict, init=False, repr=False)
    _batch_of_pko: Dict[Tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)
    _eqp_qty_by_batch_model: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)
    _plan_qty_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict, init=False, repr=False)
    _wip_by_pko: Dict[Tuple[str, str], float] = field(default_factory=dict, init=False, repr=False)
    _oper_seq_by_pko: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for u in self.uph:
            self._uph_lookup[(u.plan_prod_key, u.oper_id, u.eqp_model_cd)] = float(u.uph)
        for a in self.availability:
            self._avail_lookup[(a.plan_prod_key, a.oper_id, a.eqp_model_cd)] = bool(a.avail_yn)
        for g in self.tool_groups:
            self._batch_of_pko[(g.plan_prod_key, g.oper_id)] = g.batch_id
        for e in self.equipment:
            key = (e.batch_id, e.eqp_model_cd)
            self._eqp_qty_by_batch_model[key] = self._eqp_qty_by_batch_model.get(key, 0) + int(e.eqp_qty)
        for p in self.plans:
            key = (p.plan_prod_key, p.oper_id)
            self._plan_qty_by_pko[key] = self._plan_qty_by_pko.get(key, 0.0) + float(p.plan_qty)
        for w in self.wip:
            key = (w.plan_prod_key, w.oper_id)
            self._wip_by_pko[key] = self._wip_by_pko.get(key, 0.0) + float(w.wip_qty)
            self._oper_seq_by_pko[key] = int(w.oper_seq)

    # --- accessors ----------------------------------------------------------
    def plan_targets(self) -> List[Tuple[str, str, float]]:
        return [(pk, op, qty) for (pk, op), qty in self._plan_qty_by_pko.items()]

    def batch_of(self, plan_prod_key: str, oper_id: str) -> str:
        return self._batch_of_pko.get((plan_prod_key, oper_id), "")

    def uph_of(self, plan_prod_key: str, oper_id: str, eqp_model_cd: str) -> float:
        return self._uph_lookup.get((plan_prod_key, oper_id, eqp_model_cd), 0.0)

    def is_available(self, plan_prod_key: str, oper_id: str, eqp_model_cd: str) -> bool:
        if not self._avail_lookup.get((plan_prod_key, oper_id, eqp_model_cd), False):
            return False
        # availability without UPH means cannot proceed
        return self._uph_lookup.get((plan_prod_key, oper_id, eqp_model_cd), 0.0) > 0.0

    def equipment_pool(self) -> Dict[Tuple[str, str], int]:
        return dict(self._eqp_qty_by_batch_model)

    def wip_of(self, plan_prod_key: str, oper_id: str) -> float:
        return self._wip_by_pko.get((plan_prod_key, oper_id), 0.0)

    def plan_qty_of(self, plan_prod_key: str, oper_id: str) -> float:
        return self._plan_qty_by_pko.get((plan_prod_key, oper_id), 0.0)

    def model_group_of(self, eqp_model_cd: str) -> List[str]:
        for members in self.eqp_model_groups.values():
            if eqp_model_cd in members:
                return list(members)
        return [eqp_model_cd]
