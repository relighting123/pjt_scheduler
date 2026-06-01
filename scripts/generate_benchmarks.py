#!/usr/bin/env python3
"""Generate 7 benchmark datasets with known ground-truth conversions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core.domain import SchedulingDataset
from core.optimizer import ImprovedGreedySolver
from core.simulator import SchedulingSimulator


def _write_benchmark(
    out_dir: Path,
    rule_timekey: str,
    products: list[tuple[str, str, str]],
    models: list[str],
    plan_qty: float,
    eqp_per_model: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"RULE_TIMEKEY": rule_timekey}]).to_csv(out_dir / "meta.csv", index=False)

    wip_rows = []
    uph_rows = []
    avail_rows = []
    batch_rows = []
    eqp_rows = []
    tool_rows = []
    plan_rows = []

    start = rule_timekey[:10]
    end = str(int(start) + 8)

    for i, (batch_id, plan, oper) in enumerate(products):
        wip_rows.append(
            {
                "RULE_TIMEKEY": rule_timekey,
                "PLAN_PROD_KEY": plan,
                "OPER_ID": oper,
                "OPER_SEQ": i + 1,
                "WIP_QTY": 100 * (i + 1),
            }
        )
        batch_rows.append(
            {"RULE_TIMEKEY": rule_timekey, "BATCH_ID": batch_id, "PLAN_PROD_KEY": plan, "OPER_ID": oper}
        )
        plan_rows.append(
            {
                "RULE_TIMEKEY": rule_timekey,
                "PLAN_PROD_KEY": plan,
                "OPER_ID": oper,
                "START_TIME": start,
                "END_TIME": end,
                "PLAN_QTY": plan_qty * (1 + 0.1 * i),
            }
        )
        for j, model in enumerate(models):
            uph = 50 + j * 20 + i * 5
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
                    "AVAIL_YN": "Y" if j <= i + 1 else "N",
                }
            )
            eqp_rows.append(
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "BATCH_ID": batch_id,
                    "EQP_MODEL_CD": model,
                    "EQP_QTY": max(1, eqp_per_model - j),
                }
            )
            tool_rows.append(
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "BATCH_ID": batch_id,
                    "EQP_MODEL_CD": model,
                    "TOOL_QTY": 2,
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
    solver = ImprovedGreedySolver()
    conversions = solver.solve(ds)
    sim = SchedulingSimulator(ds)
    result = sim.simulate(conversions)

    gt_rows = [c.to_row() for c in conversions]
    payload = {
        "RULE_TIMEKEY": rule_timekey,
        "expected_avg_achievement": result.avg_achievement_rate,
        "conversions": gt_rows,
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    specs = [
        ("2026051707000000", 2, ["M1", "M2"], 400, 3),
        ("2026051807000000", 2, ["M1", "M2", "M3"], 500, 4),
        ("2026051907000000", 3, ["M1", "M2"], 600, 3),
        ("2026052007000000", 3, ["M1", "M2", "M3"], 450, 5),
        ("2026052107000000", 2, ["T5833", "MAGNUM5"], 800, 4),
        ("2026052207000000", 4, ["M1", "M2"], 350, 2),
        ("2026052307000000", 4, ["M1", "M2", "M3"], 700, 6),
    ]
    root = Path("benchmarks")
    for idx, (rtk, n_opers, models, plan_qty, eqp) in enumerate(specs, start=1):
        products = []
        for i in range(n_opers):
            products.append((f"B{i+1}", f"P{idx}/PROD{i+1}", f"OP{i+1:03d}"))
        _write_benchmark(root / f"benchmark_{idx:02d}", rtk, products, models, plan_qty, eqp)
        print(f"benchmark_{idx:02d} created")


if __name__ == "__main__":
    main()
