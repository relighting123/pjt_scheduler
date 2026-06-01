"""Scheduling simulator: conversions, production, plan achievement."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.domain import ConversionRecord, SchedulingDataset


@dataclass
class AllocationState:
    """Current equipment allocation: model -> batch_id -> qty."""

    allocation: dict[str, dict[str, int]] = field(default_factory=dict)

    def qty_on_batch(self, model: str, batch_id: str) -> int:
        return self.allocation.get(model, {}).get(batch_id, 0)

    def total_for_model(self, model: str) -> int:
        return sum(self.allocation.get(model, {}).values())

    def move(self, model: str, from_batch: str, to_batch: str, qty: int) -> None:
        if qty <= 0:
            return
        self.allocation.setdefault(model, {})
        self.allocation[model][from_batch] = max(0, self.qty_on_batch(model, from_batch) - qty)
        self.allocation[model][to_batch] = self.qty_on_batch(model, to_batch) + qty
        if self.allocation[model].get(from_batch, 0) == 0:
            self.allocation[model].pop(from_batch, None)


@dataclass
class SimulationResult:
    avg_achievement_rate: float
    achievement_by_oper: dict[tuple[str, str], float]
    achievement_by_model: dict[str, float]
    conversion_count: int
    produced_by_oper: dict[tuple[str, str], float]
    plan_by_oper: dict[tuple[str, str], float]


class SchedulingSimulator:
    """Simulate hourly production and tool-change rules from conversion schedule."""

    HOURS_PER_STEP = 1

    def __init__(self, dataset: SchedulingDataset):
        self.dataset = dataset
        self._batch_to_opers: dict[str, list[tuple[str, str]]] = {}
        for bo in dataset.batch_opers:
            self._batch_to_opers.setdefault(bo.batch_id, []).append((bo.plan_prod_key, bo.oper_id))

    def _hours_in_horizon(self) -> int:
        if not self.dataset.plan_slots:
            return 24
        times: list[int] = []
        for slot in self.dataset.plan_slots:
            times.append(int(slot.start_time[:10]))
            times.append(int(slot.end_time[:10]))
        if not times:
            return 24
        span = max(times) - min(times)
        return max(1, min(168, span + 1))

    def _plan_qty_in_hour(
        self, plan_prod_key: str, oper_id: str, hour_idx: int, horizon_start: int
    ) -> float:
        hour_key = horizon_start + hour_idx
        total = 0.0
        for slot in self.dataset.plan_slots:
            if slot.plan_prod_key != plan_prod_key or slot.oper_id != oper_id:
                continue
            start = int(slot.start_time[:10])
            end = int(slot.end_time[:10])
            if start <= hour_key < end:
                slot_hours = max(1, end - start)
                total += slot.plan_qty / slot_hours
        return total

    def _validate_fleet(self, state: AllocationState) -> None:
        for model in self.dataset.fleet_models():
            on_line = state.total_for_model(model)
            cap = self.dataset.fleet_qty(model)
            if on_line > cap:
                raise ValueError(f"Fleet exceeded for {model}: {on_line} > {cap}")

    def apply_conversions(
        self,
        conversions: list[ConversionRecord],
        initial_state: AllocationState | None = None,
    ) -> AllocationState:
        state = initial_state or self._initial_state()
        for conv in conversions:
            from_batch = conv.from_batch or "_POOL_"
            qty = conv.eqp_qty
            if qty <= 0:
                continue
            available = state.qty_on_batch(conv.eqp_model_cd, from_batch)
            if from_batch != "_POOL_" and available < qty:
                qty = available
            state.move(conv.eqp_model_cd, from_batch, conv.to_batch_id, qty)
            self._validate_fleet(state)
        return state

    def _initial_state(self) -> AllocationState:
        state = AllocationState()
        for eqp in self.dataset.eqp_counts:
            state.allocation.setdefault(eqp.eqp_model_cd, {})
            state.allocation[eqp.eqp_model_cd][eqp.batch_id] = (
                state.qty_on_batch(eqp.eqp_model_cd, eqp.batch_id) + eqp.eqp_qty
            )
        return state

    def simulate(self, conversions: list[ConversionRecord]) -> SimulationResult:
        state = self.apply_conversions(conversions)
        horizon_start = int(self.dataset.rule_timekey[:10]) if len(self.dataset.rule_timekey) >= 10 else 0
        hours = self._hours_in_horizon()
        plan_totals = self.dataset.plan_qty_by_oper()
        produced: dict[tuple[str, str], float] = {k: 0.0 for k in plan_totals}
        model_produced: dict[str, float] = {}

        for hour_idx in range(hours):
            for model, batches in state.allocation.items():
                for batch_id, qty in batches.items():
                    if qty <= 0:
                        continue
                    opers = self._batch_to_opers.get(batch_id, [])
                    if not opers:
                        continue
                    share = qty / len(opers)
                    for plan_prod_key, oper_id in opers:
                        if not self.dataset.is_available(plan_prod_key, oper_id, model):
                            continue
                        uph = self.dataset.uph(plan_prod_key, oper_id, model)
                        if uph is None:
                            continue
                        output = uph * share * self.HOURS_PER_STEP
                        key = (plan_prod_key, oper_id)
                        produced[key] = produced.get(key, 0.0) + output
                        model_produced[model] = model_produced.get(model, 0.0) + output

        achievement_by_oper: dict[tuple[str, str], float] = {}
        for key, plan_qty in plan_totals.items():
            if plan_qty <= 0:
                achievement_by_oper[key] = 1.0
            else:
                achievement_by_oper[key] = min(1.0, produced.get(key, 0.0) / plan_qty)

        rates = list(achievement_by_oper.values()) or [0.0]
        avg_rate = sum(rates) / len(rates)

        achievement_by_model: dict[str, float] = {}
        total_plan = sum(plan_totals.values()) or 1.0
        for model, out in model_produced.items():
            achievement_by_model[model] = min(1.0, out / total_plan)

        tool_changes = self._count_tool_changes(conversions)

        return SimulationResult(
            avg_achievement_rate=avg_rate,
            achievement_by_oper=achievement_by_oper,
            achievement_by_model=achievement_by_model,
            conversion_count=tool_changes,
            produced_by_oper=produced,
            plan_by_oper=plan_totals,
        )

    def _count_tool_changes(self, conversions: list[ConversionRecord]) -> int:
        changes = 0
        for conv in conversions:
            from_batch = conv.from_batch or ""
            to_batch = conv.to_batch_id or ""
            if from_batch and to_batch and from_batch != to_batch:
                changes += 1
        return changes
