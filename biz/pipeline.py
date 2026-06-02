"""End-to-end orchestration: load -> simulate/train/infer -> persist + report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from core.domain import SchedulingProblem
from core.evaluation import evaluate_all_benchmark_datasets
from core.heuristic import greedy_allocate
from core.report import render_html, render_markdown
from core.rl_infer import infer as rl_infer

from .data_loader import (
    latest_rule_timekey,
    list_rule_timekeys,
    load_problem_from_csv_dir,
    load_problem_from_oracle,
)
from .output_writer import build_conversion_rows, write_csv, write_oracle


def load_settings(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
def _connect(settings: dict):
    from core.db import connect
    o = settings["oracle"]
    return connect(user=o["user"], password=o["password"], dsn=o["dsn"])


def _problems_for_training(
    settings: dict,
    from_timekey: Optional[str],
    to_timekey: Optional[str],
    rule_timekey: Optional[str],
    benchmark_dataset: Optional[str],
) -> List[SchedulingProblem]:
    if benchmark_dataset:
        return [load_problem_from_csv_dir(benchmark_dataset)]
    if not (from_timekey or to_timekey or rule_timekey):
        raise ValueError("Provide --from-timekey/--to-timekey, --timekey, or --benchmark-dataset.")
    if rule_timekey:
        from_timekey = from_timekey or rule_timekey
        to_timekey = to_timekey or rule_timekey
    conn = _connect(settings)
    try:
        table = settings["oracle"]["source_table"]
        keys = list_rule_timekeys(conn, table, from_timekey, to_timekey)
        if not keys:
            return []
        groups = settings.get("tool_groups", {})
        return [load_problem_from_oracle(conn, table, k, groups) for k in keys]
    finally:
        conn.close()


def run_train(
    settings: dict,
    from_timekey: Optional[str] = None,
    to_timekey: Optional[str] = None,
    rule_timekey: Optional[str] = None,
    benchmark_dataset: Optional[str] = None,
    steps: Optional[int] = None,
) -> dict:
    from core.rl_train import train

    problems = _problems_for_training(settings, from_timekey, to_timekey, rule_timekey, benchmark_dataset)
    if not problems:
        raise RuntimeError("No training problems found for the given range.")

    model_cfg = settings["model"]
    reward_cfg = settings.get("reward", {})
    save_path = train(
        problems=problems,
        artifact_dir=model_cfg["artifact_dir"],
        policy_name=model_cfg["policy_name"],
        imitation_epochs=int(model_cfg.get("imitation_epochs", 30)),
        ppo_total_steps=int(steps or model_cfg.get("ppo_total_steps", 50000)),
        ppo_n_steps=int(model_cfg.get("ppo_n_steps", 512)),
        ppo_batch_size=int(model_cfg.get("ppo_batch_size", 64)),
        ppo_learning_rate=float(model_cfg.get("ppo_learning_rate", 3e-4)),
        ppo_gamma=float(model_cfg.get("ppo_gamma", 0.99)),
        switch_penalty=float(reward_cfg.get("switch_penalty", 0.02)),
        achievement_weight=float(reward_cfg.get("achievement_weight", 1.0)),
        seed=int(model_cfg.get("seed", 7)),
    )

    # benchmark + reports
    bench = settings.get("benchmark", {})
    results = evaluate_all_benchmark_datasets(bench.get("dataset_root", "benchmarks"), model_path=save_path)
    html_path = render_html(results, bench.get("report_path", "artifacts/reports/benchmark.html"))
    md_path = render_markdown(results, bench.get("summary_md", "MODEL_BENCHMARK.md"))
    return {
        "model_path": save_path,
        "report_html": html_path,
        "report_md": md_path,
        "n_problems": len(problems),
        "n_benchmarks": len(results),
    }


# ---------------------------------------------------------------------------
def run_infer(
    settings: dict,
    rule_timekey: Optional[str] = None,
    benchmark_dataset: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> dict:
    model_cfg = settings["model"]
    model_path = str(Path(model_cfg["artifact_dir"]) / f"{model_cfg['policy_name']}.zip")

    if benchmark_dataset:
        problem = load_problem_from_csv_dir(benchmark_dataset)
        allocation = rl_infer(problem, model_path=model_path)
        rows = build_conversion_rows(problem.rule_timekey, None, allocation)
        if not output_csv:
            output_csv = str(Path("artifacts/inference") / f"{problem.rule_timekey}.csv")
        write_csv(output_csv, rows)
        return {"mode": "benchmark", "rule_timekey": problem.rule_timekey, "rows": len(rows), "output": output_csv}

    conn = _connect(settings)
    try:
        oracle = settings["oracle"]
        rk = rule_timekey or latest_rule_timekey(conn, oracle["source_table"])
        if not rk:
            raise RuntimeError("No RULE_TIMEKEY found in source table.")
        problem = load_problem_from_oracle(conn, oracle["source_table"], rk, settings.get("tool_groups", {}))
        allocation = rl_infer(problem, model_path=model_path)
        # previous snapshot for diff (latest prior RULE_TIMEKEY)
        prev_keys = list_rule_timekeys(conn, oracle["source_table"], "00000000000000", rk)
        previous_alloc = None  # the system persists conversion rows, not prior allocations
        rows = build_conversion_rows(rk, previous_alloc, allocation)
        write_oracle(
            conn,
            output_table=oracle["output_table"],
            history_table=oracle.get("history_table", ""),
            rule_timekey=rk,
            rows=rows,
        )
        return {"mode": "oracle", "rule_timekey": rk, "rows": len(rows)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
def run_eval(settings: dict) -> dict:
    model_cfg = settings["model"]
    bench = settings.get("benchmark", {})
    model_path = str(Path(model_cfg["artifact_dir"]) / f"{model_cfg['policy_name']}.zip")
    if not Path(model_path).exists():
        model_path = None  # type: ignore
    results = evaluate_all_benchmark_datasets(bench.get("dataset_root", "benchmarks"), model_path=model_path)
    html_path = render_html(results, bench.get("report_path", "artifacts/reports/benchmark.html"))
    md_path = render_markdown(results, bench.get("summary_md", "MODEL_BENCHMARK.md"))
    return {
        "report_html": html_path,
        "report_md": md_path,
        "n_benchmarks": len(results),
        "avg_optimal": (sum(r.optimal.avg_achievement for r in results) / len(results)) if results else 0.0,
        "avg_rl": (sum(r.rl.avg_achievement for r in results) / len(results)) if results else 0.0,
        "avg_heuristic": (sum(r.heuristic.avg_achievement for r in results) / len(results)) if results else 0.0,
    }
