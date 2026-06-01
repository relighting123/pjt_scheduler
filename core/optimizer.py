"""Reference solvers: optimal (benchmark), heuristic, and demonstration export."""

from __future__ import annotations

import json
from pathlib import Path

from core.domain import ConversionRecord, SchedulingDataset
from core.simulator import SchedulingSimulator


class HeuristicSolver:
    """Greedy: assign equipment to highest UPH-efficiency oper per batch."""

    def solve(self, dataset: SchedulingDataset) -> list[ConversionRecord]:
        simulator = SchedulingSimulator(dataset)
        state = simulator._initial_state()
        conversions: list[ConversionRecord] = []
        plan_by_oper = dataset.plan_qty_by_oper()
        if not plan_by_oper:
            return conversions

        ranked = sorted(
            plan_by_oper.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for (plan_prod_key, oper_id), _ in ranked:
            batch_id = dataset.batch_for(plan_prod_key, oper_id)
            if not batch_id:
                continue
            candidates = [
                m
                for m in self._models_for(dataset, plan_prod_key, oper_id)
                if dataset.is_available(plan_prod_key, oper_id, m)
            ]
            if not candidates:
                continue
            best_model = max(
                candidates,
                key=lambda m: dataset.uph(plan_prod_key, oper_id, m) or 0.0,
            )
            uph = dataset.uph(plan_prod_key, oper_id, best_model) or 1.0
            hours = max(1, simulator._hours_in_horizon())
            need = max(1, int(plan_by_oper[(plan_prod_key, oper_id)] / (uph * hours) + 0.999))
            need = min(need, dataset.fleet_qty(best_model))
            current = state.qty_on_batch(best_model, batch_id)
            if current >= need:
                continue
            pool_batch = self._find_donor_batch(state, best_model, batch_id)
            if pool_batch is None:
                continue
            deficit = min(need - current, state.qty_on_batch(best_model, pool_batch))
            if deficit <= 0:
                continue
            from_bo = next(
                (b for b in dataset.batch_opers if b.batch_id == (pool_batch if pool_batch != "_POOL_" else batch_id)),
                None,
            )
            from_plan = from_bo.plan_prod_key if from_bo else plan_prod_key
            from_oper = from_bo.oper_id if from_bo else oper_id
            from_batch = pool_batch if pool_batch != "_POOL_" else ""
            conv = ConversionRecord(
                rule_timekey=dataset.rule_timekey,
                from_batch=from_batch,
                from_plan_prod_key=from_plan,
                from_oper_id=from_oper,
                eqp_model_cd=best_model,
                to_batch_id=batch_id,
                to_plan_prod_key=plan_prod_key,
                to_oper_id=oper_id,
                start_conv_time=dataset.rule_timekey,
                eqp_qty=deficit,
            )
            conversions.append(conv)
            state.move(best_model, pool_batch, batch_id, deficit)

        return conversions

    def _models_for(self, dataset: SchedulingDataset, plan_prod_key: str, oper_id: str) -> list[str]:
        return list(
            {
                m.eqp_model_cd
                for m in dataset.model_uph
                if m.plan_prod_key == plan_prod_key and m.oper_id == oper_id
            }
        )

    def _find_donor_batch(self, state, model: str, exclude: str) -> str | None:
        batches = state.allocation.get(model, {})
        donors = [(b, q) for b, q in batches.items() if b != exclude and q > 0]
        if not donors:
            return None
        return max(donors, key=lambda x: x[1])[0]


class ReferenceAllocationSolver:
    """Fleet-constrained reference: maximize plan coverage, emit conversions from initial layout."""

    def solve(self, dataset: SchedulingDataset) -> list[ConversionRecord]:
        simulator = SchedulingSimulator(dataset)
        initial = simulator._initial_state()
        hours = max(1, simulator._hours_in_horizon())
        plan_by_oper = dataset.plan_qty_by_oper()

        fleet_left = {m: dataset.fleet_qty(m) for m in dataset.fleet_models()}
        target: dict[str, dict[str, int]] = {m: {} for m in fleet_left}

        for (plan_prod_key, oper_id), plan_qty in sorted(
            plan_by_oper.items(), key=lambda x: -x[1]
        ):
            batch_id = dataset.batch_for(plan_prod_key, oper_id)
            if not batch_id:
                continue
            models = [
                m
                for m in self._models_for(dataset, plan_prod_key, oper_id)
                if dataset.is_available(plan_prod_key, oper_id, m)
            ]
            if not models:
                continue
            best_model = max(
                models, key=lambda m: dataset.uph(plan_prod_key, oper_id, m) or 0.0
            )
            uph = dataset.uph(plan_prod_key, oper_id, best_model) or 1.0
            need_eqp = max(1, int(plan_qty / (uph * hours) + 0.999))
            assign = min(need_eqp, fleet_left.get(best_model, 0))
            if assign <= 0:
                continue
            target[best_model][batch_id] = target[best_model].get(batch_id, 0) + assign
            fleet_left[best_model] -= assign

        return self._diff_to_conversions(dataset, initial, target)

    def _models_for(self, dataset: SchedulingDataset, plan_prod_key: str, oper_id: str) -> list[str]:
        return list(
            {
                m.eqp_model_cd
                for m in dataset.model_uph
                if m.plan_prod_key == plan_prod_key and m.oper_id == oper_id
            }
        )

    def _diff_to_conversions(
        self,
        dataset: SchedulingDataset,
        initial,
        target: dict[str, dict[str, int]],
    ) -> list[ConversionRecord]:
        conversions: list[ConversionRecord] = []
        for model in dataset.fleet_models():
            batches = set(initial.allocation.get(model, {})) | set(target.get(model, {}))
            surplus: list[list] = []
            deficit: list[list] = []
            for batch in batches:
                cur = initial.qty_on_batch(model, batch)
                tgt = target.get(model, {}).get(batch, 0)
                if cur > tgt:
                    surplus.append([batch, cur - tgt])
                elif tgt > cur:
                    deficit.append([batch, tgt - cur])

            for from_batch, qty_left in surplus:
                from_bo = next(
                    (b for b in dataset.batch_opers if b.batch_id == from_batch), None
                )
                while qty_left > 0 and deficit:
                    to_batch, need = deficit[0]
                    move = min(qty_left, need)
                    if move <= 0:
                        deficit.pop(0)
                        continue
                    to_bo = next(
                        (b for b in dataset.batch_opers if b.batch_id == to_batch), None
                    )
                    conversions.append(
                        ConversionRecord(
                            rule_timekey=dataset.rule_timekey,
                            from_batch=from_batch,
                            from_plan_prod_key=from_bo.plan_prod_key if from_bo else "",
                            from_oper_id=from_bo.oper_id if from_bo else "",
                            eqp_model_cd=model,
                            to_batch_id=to_batch,
                            to_plan_prod_key=to_bo.plan_prod_key if to_bo else "",
                            to_oper_id=to_bo.oper_id if to_bo else "",
                            start_conv_time=dataset.rule_timekey,
                            eqp_qty=int(move),
                        )
                    )
                    qty_left -= move
                    need -= move
                    if need <= 0:
                        deficit.pop(0)
                    else:
                        deficit[0][1] = need
        return conversions


class OptimalSolver(ReferenceAllocationSolver):
    """Benchmark reference optimum under fleet constraints."""


class ImprovedGreedySolver(HeuristicSolver):
    """Weight plan gap and UPH when moving equipment (suboptimal vs reference)."""

    def solve(self, dataset: SchedulingDataset) -> list[ConversionRecord]:
        simulator = SchedulingSimulator(dataset)
        conversions: list[ConversionRecord] = []
        state = simulator._initial_state()
        plan_by_oper = dataset.plan_qty_by_oper()
        hours = max(1, simulator._hours_in_horizon())

        for (plan_prod_key, oper_id), plan_qty in sorted(plan_by_oper.items(), key=lambda x: -x[1]):
            batch_id = dataset.batch_for(plan_prod_key, oper_id)
            if not batch_id:
                continue
            models = [
                m
                for m in self._models_for(dataset, plan_prod_key, oper_id)
                if dataset.is_available(plan_prod_key, oper_id, m)
            ]
            if not models:
                continue
            best_model = max(
                models,
                key=lambda m: (dataset.uph(plan_prod_key, oper_id, m) or 0) * plan_qty,
            )
            uph = dataset.uph(plan_prod_key, oper_id, best_model) or 1.0
            target = min(
                dataset.fleet_qty(best_model),
                max(1, int(plan_qty / (uph * hours) + 0.999)),
            )
            on_batch = state.qty_on_batch(best_model, batch_id)
            donor = self._find_donor_batch(state, best_model, batch_id)
            if not donor:
                continue
            move_qty = min(target - on_batch, state.qty_on_batch(best_model, donor))
            if move_qty <= 0:
                continue
            from_batch = donor
            from_bo = next((b for b in dataset.batch_opers if b.batch_id == donor), None) if donor else None
            conversions.append(
                ConversionRecord(
                    rule_timekey=dataset.rule_timekey,
                    from_batch=from_batch,
                    from_plan_prod_key=from_bo.plan_prod_key if from_bo else plan_prod_key,
                    from_oper_id=from_bo.oper_id if from_bo else oper_id,
                    eqp_model_cd=best_model,
                    to_batch_id=batch_id,
                    to_plan_prod_key=plan_prod_key,
                    to_oper_id=oper_id,
                    start_conv_time=dataset.rule_timekey,
                    eqp_qty=int(move_qty),
                )
            )
            state.move(best_model, donor, batch_id, int(move_qty))

        return conversions


def load_ground_truth_conversions(path: Path) -> list[ConversionRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in data.get("conversions", []):
        records.append(
            ConversionRecord(
                rule_timekey=str(row["RULE_TIMEKEY"]),
                from_batch=str(row.get("FROM_BATCH", "")),
                from_plan_prod_key=str(row["FROM_PLAN_PROD_KEY"]),
                from_oper_id=str(row["FROM_OPER_ID"]),
                eqp_model_cd=str(row["EQP_MODEL_CD"]),
                to_batch_id=str(row["TO_BATCH_ID"]),
                to_plan_prod_key=str(row["TO_PLAN_PROD_KEY"]),
                to_oper_id=str(row["TO_OPER_ID"]),
                start_conv_time=str(row.get("START_CONV_TIME", row["RULE_TIMEKEY"])),
                eqp_qty=int(row["EQP_QTY"]),
            )
        )
    return records
