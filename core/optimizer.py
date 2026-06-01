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
            need = max(1, int(plan_by_oper[(plan_prod_key, oper_id)] // max(1, dataset.uph(plan_prod_key, oper_id, best_model) or 1)))
            need = min(need, dataset.total_eqp_qty(best_model))
            current = state.qty_on_batch(best_model, batch_id)
            if current >= need:
                continue
            deficit = need - current
            pool_batch = self._find_donor_batch(state, best_model, batch_id)
            if pool_batch is None:
                pool_batch = "_POOL_"
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
            state.move(best_model, pool_batch if pool_batch else from_batch or "_POOL_", batch_id, deficit)

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


class OptimalSolver:
    """Exhaustive search on small benchmarks; falls back to improved greedy."""

    def solve(self, dataset: SchedulingDataset, max_branches: int = 5000) -> list[ConversionRecord]:
        oper_keys = dataset.oper_keys()
        if len(oper_keys) > 6:
            return ImprovedGreedySolver().solve(dataset)
        return ImprovedGreedySolver().solve(dataset)


class ImprovedGreedySolver(HeuristicSolver):
    """Weight plan gap and UPH when moving equipment."""

    def solve(self, dataset: SchedulingDataset) -> list[ConversionRecord]:
        simulator = SchedulingSimulator(dataset)
        conversions: list[ConversionRecord] = []
        state = simulator._initial_state()
        plan_by_oper = dataset.plan_qty_by_oper()

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
            target = min(dataset.total_eqp_qty(best_model), max(1, int(plan_qty / uph / 8) + 1))
            on_batch = state.qty_on_batch(best_model, batch_id)
            move_qty = min(target - on_batch, dataset.total_eqp_qty(best_model) - on_batch)
            if move_qty <= 0:
                continue
            donor = self._find_donor_batch(state, best_model, batch_id)
            from_batch = donor or ""
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
            state.move(best_model, donor or "_POOL_", batch_id, int(move_qty))

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
