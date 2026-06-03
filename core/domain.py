"""Domain data classes shared by simulator, optimizer, and RL engine.

The biz layer materializes these from the physical Oracle table and feeds the
core simulator. Keep this module dependency-free so it can be imported in any
environment (including the no-DB benchmark harness).

Vocabulary recap:
  plan_prod_key  계획 제품 키, e.g. "M15/59C/H5UDGSTED/E1S/NA"
  oper_id        공정 ID, e.g. "Z1020000A"
  batch_id       동일 tool 그룹 ID, e.g. "9C/92"
  eqp_model_cd   장비 모델 코드, e.g. "T5833"
  rule_timekey   스냅샷 시각, e.g. "2026051707000000"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WipRecord:
    """공정 큐에 쌓인 재공 한 줄.

    Example:
        WipRecord(rule_timekey="2026051707000000",
                  plan_prod_key="P1", oper_id="OP10",
                  oper_seq=1, wip_qty=120.0)
    """
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    oper_seq: int
    wip_qty: float


@dataclass(frozen=True)
class UphRecord:
    """(제품, 공정, 장비 모델)별 시간당 생산량.

    Example:
        UphRecord("2026051707000000", "P1", "OP10", "T5833", uph=200.0)
        → T5833 1대가 P1/OP10을 시간당 200개 처리
    """
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    uph: float


@dataclass(frozen=True)
class EquipmentRecord:
    """(배치, 모델) 풀에 있는 장비 수.

    Example:
        EquipmentRecord("2026051707000000", "9C/92", "T5833", eqp_qty=4)
        → 배치 9C/92 안에 T5833 4대
    """
    rule_timekey: str
    batch_id: str
    eqp_model_cd: str
    eqp_qty: int


@dataclass(frozen=True)
class AvailabilityRecord:
    """(제품, 공정, 장비 모델) 처리 가능 여부.

    Example:
        AvailabilityRecord("...", "P1", "OP10", "T5833", avail_yn=True)
    """
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    avail_yn: bool


@dataclass(frozen=True)
class ToolGroupRecord:
    """(plan_prod_key, oper_id) → batch_id 매핑. N:1.

    Example:
        ToolGroupRecord("...", batch_id="9C/92",
                        plan_prod_key="P1", oper_id="OP10")
        → P1/OP10 작업은 9C/92 배치의 tool을 사용
    """
    rule_timekey: str
    batch_id: str
    plan_prod_key: str
    oper_id: str


@dataclass(frozen=True)
class ToolQtyRecord:
    """배치별 tool 보유 수.

    Example:
        ToolQtyRecord("...", "9C/92", "T5833", tool_qty=4)
    """
    rule_timekey: str
    batch_id: str
    eqp_model_cd: str
    tool_qty: int


@dataclass(frozen=True)
class PlanRecord:
    """(제품, 공정)별 생산 계획.

    Example:
        PlanRecord("2026051707000000", "P1", "OP10",
                   start_time="2026051707", end_time="2026051708",
                   plan_qty=600.0)
        → 2026-05-17 07-08시 사이 P1/OP10을 600개 생산
    """
    rule_timekey: str
    plan_prod_key: str
    oper_id: str
    start_time: str
    end_time: str
    plan_qty: float


@dataclass
class Allocation:
    """장비 N대를 (batch, plan_prod_key, oper, model)에 배정한 결정 한 줄.

    Example:
        Allocation(batch_id="9C/92", plan_prod_key="P1", oper_id="OP10",
                   eqp_model_cd="T5833", eqp_qty=2)
        → 9C/92 배치의 T5833 2대를 P1/OP10에 배정
    """
    batch_id: str
    plan_prod_key: str
    oper_id: str
    eqp_model_cd: str
    eqp_qty: int


@dataclass
class AllocationSet:
    """한 시각(rule_timekey)에 대한 전체 할당 결정 — Allocation의 모음.

    Example:
        AllocationSet(rule_timekey="2026051707000000",
                      allocations=[Allocation("9C/92","P1","OP10","T5833",2),
                                   Allocation("9C/92","P2","OP10","T5833",1)])
    """
    rule_timekey: str
    allocations: List[Allocation] = field(default_factory=list)


@dataclass
class SchedulingProblem:
    """한 rule_timekey 스냅샷을 모은 입력 컨테이너.

    7종 record 리스트를 받으면 __post_init__에서 (pk,op,...) 키 기반 dict
    인덱스를 구축한다. 시뮬레이터/옵티마이저/RL env가 매 step마다 dict O(1)
    조회로 동작하도록 하기 위함.

    Example:
        problem = SchedulingProblem(
            rule_timekey="2026051707000000",
            wip=[WipRecord(...), ...],
            uph=[UphRecord(...), ...],
            equipment=[EquipmentRecord(...), ...],
            availability=[AvailabilityRecord(...), ...],
            tool_groups=[ToolGroupRecord(...), ...],
            tool_qty=[ToolQtyRecord(...), ...],
            plans=[PlanRecord(...), ...],
            eqp_model_groups={"G001": ["9C/92", "9C/102"]},
        )
        problem.plan_targets()    # [(pk, op, qty), ...]
        problem.uph_of("P1", "OP10", "T5833")  # 200.0
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
        """모든 계획 항목 리스트.

        Returns:
            [(plan_prod_key, oper_id, plan_qty), ...]
            예: [("P1", "OP10", 600.0), ("P1", "OP20", 400.0)]
        """
        return [(pk, op, qty) for (pk, op), qty in self._plan_qty_by_pko.items()]

    def batch_of(self, plan_prod_key: str, oper_id: str) -> str:
        """(pk, op)가 속한 batch_id. 없으면 빈 문자열.

        Example: batch_of("P1", "OP10") → "9C/92"
        """
        return self._batch_of_pko.get((plan_prod_key, oper_id), "")

    def uph_of(self, plan_prod_key: str, oper_id: str, eqp_model_cd: str) -> float:
        """(pk, op, model) 조합의 시간당 생산량. 없으면 0.

        Example: uph_of("P1", "OP10", "T5833") → 200.0
        """
        return self._uph_lookup.get((plan_prod_key, oper_id, eqp_model_cd), 0.0)

    def is_available(self, plan_prod_key: str, oper_id: str, eqp_model_cd: str) -> bool:
        """모델이 (pk, op)를 처리 가능한가? UPH 기록이 없으면 False.

        Example: is_available("P1", "OP10", "T5833") → True
        """
        if not self._avail_lookup.get((plan_prod_key, oper_id, eqp_model_cd), False):
            return False
        # availability without UPH means cannot proceed
        return self._uph_lookup.get((plan_prod_key, oper_id, eqp_model_cd), 0.0) > 0.0

    def equipment_pool(self) -> Dict[Tuple[str, str], int]:
        """(batch_id, eqp_model_cd) → 보유 수량.

        Returns:
            {("9C/92","T5833"): 4, ("9C/102","T5833"): 2, ...}
        """
        return dict(self._eqp_qty_by_batch_model)

    def wip_of(self, plan_prod_key: str, oper_id: str) -> float:
        """(pk, op)의 현재 재공 수량.

        Example: wip_of("P1", "OP20") → 50.0
        """
        return self._wip_by_pko.get((plan_prod_key, oper_id), 0.0)

    def plan_qty_of(self, plan_prod_key: str, oper_id: str) -> float:
        """(pk, op)의 계획 수량.

        Example: plan_qty_of("P1", "OP10") → 600.0
        """
        return self._plan_qty_by_pko.get((plan_prod_key, oper_id), 0.0)

    def model_group_of(self, eqp_model_cd: str) -> List[str]:
        """장비 모델이 속한 batch 그룹의 멤버 리스트. 그룹 없으면 자기 자신만.

        Example: model_group_of("9C/92") → ["9C/92", "9C/102"]
        """
        for members in self.eqp_model_groups.values():
            if eqp_model_cd in members:
                return list(members)
        return [eqp_model_cd]

    def oper_order_of(self, plan_prod_key: str) -> List[str]:
        """제품의 공정을 OPER_SEQ 순으로 정렬해 반환.

        Example: oper_order_of("P1") → ["OP10", "OP20", "OP30"]
        """
        ops = sorted(
            (seq, op)
            for (pk, op), seq in self._oper_seq_by_pko.items()
            if pk == plan_prod_key
        )
        return [op for _, op in ops]

    def next_oper_of(self, plan_prod_key: str, oper_id: str) -> Optional[str]:
        """현재 공정의 다음 공정 (WIP 흐름 방향). 마지막이면 None.

        Example: next_oper_of("P1", "OP10") → "OP20"
                 next_oper_of("P1", "OP30") → None
        """
        order = self.oper_order_of(plan_prod_key)
        if oper_id in order:
            idx = order.index(oper_id)
            if idx + 1 < len(order):
                return order[idx + 1]
        return None
