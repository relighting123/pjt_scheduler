#!/usr/bin/env python3
"""Generate 7 non-trivial benchmarks: shared fleet, skewed initial allocation, tight plans."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core.domain import SchedulingDataset
from core.optimizer import ImprovedGreedySolver, ReferenceAllocationSolver
from core.simulator import SchedulingSimulator


def _hour_key(rtk: str, offset: int) -> str:
    base = int(rtk[:10])
    return str(base + offset)


def _write_benchmark(
    out_dir: Path,
    rule_timekey: str,
    products: list[tuple[str, str, str]],
    models: list[str],
    fleet: dict[str, int],
    initial_on_batch: str,
    plan_scale: float,
    horizon_hours: int,
) -> None:
    """products: (batch_id, plan_prod_key, oper_id)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"RULE_TIMEKEY": rule_timekey}]).to_csv(out_dir / "meta.csv", index=False)

    start = rule_timekey[:10]
    end = _hour_key(rule_timekey, horizon_hours)

    fleet_rows = [
        {"RULE_TIMEKEY": rule_timekey, "EQP_MODEL_CD": m, "FLEET_QTY": q}
        for m, q in fleet.items()
    ]
    pd.DataFrame(fleet_rows).to_csv(out_dir / "equipment_fleet.csv", index=False)

    wip_rows, uph_rows, avail_rows, batch_rows, eqp_rows, tool_rows, plan_rows = [], [], [], [], [], [], []

    for i, (batch_id, plan, oper) in enumerate(products):
        wip_rows.append(
            {
                "RULE_TIMEKEY": rule_timekey,
                "PLAN_PROD_KEY": plan,
                "OPER_ID": oper,
                "OPER_SEQ": i + 1,
                "WIP_QTY": 500 + i * 100,
            }
        )
        batch_rows.append(
            {"RULE_TIMEKEY": rule_timekey, "BATCH_ID": batch_id, "PLAN_PROD_KEY": plan, "OPER_ID": oper}
        )
        for j, model in enumerate(models):
            uph = 40 + j * 15 + i * 8
            uph_rows.append(
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "PLAN_PROD_KEY": plan,
                    "OPER_ID": oper,
                    "EQP_MODEL_CD": model,
                    "UPH": uph,
                }
            )
            avail_rows.append(
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "PLAN_PROD_KEY": plan,
                    "OPER_ID": oper,
                    "EQP_MODEL_CD": model,
                    "AVAIL_YN": "Y",
                }
            )
            tool_rows.append(
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "BATCH_ID": batch_id,
                    "EQP_MODEL_CD": model,
                    "TOOL_QTY": 1,
                }
            )

        best_uph = 40 + i * 8
        plan_qty = plan_scale * (1.15 ** i)
        plan_rows.append(
            {
                "RULE_TIMEKEY": rule_timekey,
                "PLAN_PROD_KEY": plan,
                "OPER_ID": oper,
                "START_TIME": start,
                "END_TIME": end,
                "PLAN_QTY": round(plan_qty, 1),
            }
        )

    for model, fleet_qty in fleet.items():
        for batch_id, _, _ in products:
            qty = fleet_qty if batch_id == initial_on_batch else 0
            if qty > 0:
                eqp_rows.append(
                    {
                        "RULE_TIMEKEY": rule_timekey,
                        "BATCH_ID": batch_id,
                        "EQP_MODEL_CD": model,
                        "EQP_QTY": qty,
                    }
                )

    pd.DataFrame(wip_rows).to_csv(out_dir / "oper_wip.csv", index=False)
    pd.DataFrame(uph_rows).to_csv(out_dir / "model_uph.csv", index=False)
    pd.DataFrame(avail_rows).to_csv(out_dir / "model_avail.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(out_dir / "batch_oper.csv", index=False)
    pd.DataFrame(eqp_rows).to_csv(out_dir / "eqp_count.csv", index=False)
    pd.DataFrame(tool_rows).to_csv(out_dir / "tool_qty.csv", index=False)
    pd.DataFrame(plan_rows).to_csv(out_dir / "plan_slots.csv", index=False)

    ds = SchedulingDataset.from_csv_dir(out_dir)
    ref_solver = ReferenceAllocationSolver()
    conversions = ref_solver.solve(ds)
    sim = SchedulingSimulator(ds)
    ref_result = sim.simulate(conversions)
    naive = sim.simulate([])
    heu = ImprovedGreedySolver().solve(ds)
    heu_result = sim.simulate(heu)

    payload = {
        "RULE_TIMEKEY": rule_timekey,
        "expected_avg_achievement": ref_result.avg_achievement_rate,
        "naive_initial_achievement": naive.avg_achievement_rate,
        "heuristic_achievement": heu_result.avg_achievement_rate,
        "conversion_count": len(conversions),
        "conversions": [c.to_row() for c in conversions],
    }
    (out_dir / "ground_truth.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    root = Path("benchmarks")
    scenarios = [
        # rtk, n_opers, models, fleet, wrong_batch, plan_scale, hours
        ("2026051707000000", 2, ["M1", "M2"], {"M1": 4, "M2": 2}, "B1", 2200, 12),
        ("2026051807000000", 2, ["M1", "M2", "M3"], {"M1": 3, "M2": 3, "M3": 2}, "B1", 2800, 12),
        ("2026051907000000", 3, ["M1", "M2"], {"M1": 5, "M2": 3}, "B1", 2400, 16),
        ("2026052007000000", 3, ["M1", "M2", "M3"], {"M1": 4, "M2": 3, "M3": 2}, "B1", 2600, 16),
        ("2026052107000000", 2, ["T5833", "MAGNUM5"], {"T5833": 3, "MAGNUM5": 2}, "B1", 3200, 12),
        ("2026052207000000", 4, ["M1", "M2"], {"M1": 6, "M2": 4}, "B1", 2000, 20),
        ("2026052307000000", 4, ["M1", "M2", "M3"], {"M1": 5, "M2": 4, "M3": 3}, "B1", 2500, 20),
    ]

    for idx, (rtk, n_opers, models, fleet, wrong_b, scale, hours) in enumerate(scenarios, start=1):
        products = [(f"B{i+1}", f"P{idx}/PROD{i+1}", f"OP{i+1:03d}") for i in range(n_opers)]
        out = root / f"benchmark_{idx:02d}"
        _write_benchmark(out, rtk, products, models, fleet, wrong_b, scale, hours)
        ds = SchedulingDataset.from_csv_dir(out)
        sim = SchedulingSimulator(ds)
        gt = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))
        print(
            f"benchmark_{idx:02d}: naive={gt['naive_initial_achievement']:.2%} "
            f"ref={gt['expected_avg_achievement']:.2%} heu={gt['heuristic_achievement']:.2%} "
            f"conversions={gt['conversion_count']}"
        )


if __name__ == "__main__":
    main()
