"""End-to-end orchestration: load -> simulate/train/infer -> persist + report.

Three scheduling models are exposed as the ``mode`` option:

  plan-only  — calculate purely from the plan, WIP is treated as unlimited
               (the original phase-0 behaviour).
  wip-static — single-snapshot with WIP cap (phase 1; the default).
  dynamic    — multi-period WIP-flow + switch cost (phase 2/3). Time is
               sliced into `num_slots` of `slot_hours` and equipment can
               move between slots, paying `switch_time_hours` per move.

Training, inference and evaluation are dispatched by mode; models are
saved under mode-suffixed names so a snapshot can carry all three.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from core.domain import AllocationSet, SchedulingProblem
from core.evaluation.benchmark import (
    evaluate_all_benchmark_datasets,
    evaluate_all_benchmark_datasets_dynamic,
)
from core.policy.heuristic import greedy_allocate
from core.evaluation.report import render_html, render_markdown
from core.rl.infer import infer as rl_infer

from .data_loader import (
    latest_rule_timekey,
    list_rule_timekeys,
    load_problem_from_csv_dir,
    load_problem_from_oracle,
)
from .output_writer import build_conversion_rows, write_csv, write_oracle

MODES = ("plan-only", "wip-static", "dynamic")


def load_settings(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def resolve_mode(settings: dict, override: Optional[str]) -> str:
    mode = (override or settings.get("model", {}).get("mode", "wip-static")).lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


def model_path_for(settings: dict, mode: str) -> str:
    """Mode-suffixed path so all three models can coexist on disk."""
    m = settings["model"]
    suffix = mode.replace("-", "_")
    return str(Path(m["artifact_dir"]) / f"{m['policy_name']}_{suffix}.zip")


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
        query_dir = settings["oracle"].get("query_dir", "config/queries")
        keys = list_rule_timekeys(conn, query_dir, from_timekey, to_timekey)
        if not keys:
            return []
        groups = settings.get("tool_groups", {})
        return [load_problem_from_oracle(conn, query_dir, k, groups) for k in keys]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
def run_train(
    settings: dict,
    from_timekey: Optional[str] = None,
    to_timekey: Optional[str] = None,
    rule_timekey: Optional[str] = None,
    benchmark_dataset: Optional[str] = None,
    steps: Optional[int] = None,
    mode: Optional[str] = None,
) -> dict:
    """선택한 모드의 학습 파이프라인. 학습 후 벤치마크 평가까지 수행.

    Args:
        settings: settings.json 로드 결과.
        from_timekey / to_timekey / rule_timekey: Oracle 구간 (또는 단일 키).
        benchmark_dataset: 벤치마크 CSV 폴더 (DB 대신).
        steps: PPO total steps override.
        mode: plan-only | wip-static | dynamic ('all'은 학습에선 불가).

    Returns:
        {"mode", "model_path", "n_problems",
         "report_html", "report_md", "n_benchmarks",
         "avg_optimal", "avg_rl", "avg_heuristic"}

    Example:
        result = run_train(load_settings("config/settings.json"),
                           benchmark_dataset="benchmarks/benchmark_01",
                           steps=50000, mode="wip-static")
    """
    mode = resolve_mode(settings, mode)
    problems = _problems_for_training(settings, from_timekey, to_timekey, rule_timekey, benchmark_dataset)
    if not problems:
        raise RuntimeError("No training problems found for the given range.")

    model_cfg = settings["model"]
    reward_cfg = settings.get("reward", {})
    save_path = model_path_for(settings, mode)

    speed_cfg = settings.get("speed", {})
    num_envs = int(speed_cfg.get("num_envs", 1))
    device = str(speed_cfg.get("device", "auto"))
    imit_loss_target = float(speed_cfg.get("imitation_loss_target", 0.05))

    if mode == "dynamic":
        from core.rl.train_mp import train_multiperiod
        dyn = settings.get("dynamic", {})
        train_multiperiod(
            problems=problems,
            num_slots=int(dyn.get("num_slots", 4)),
            slot_hours=float(dyn.get("slot_hours", 1.0)),
            switch_time_hours=float(dyn.get("switch_time_hours", 0.0)),
            artifact_dir=str(Path(save_path).parent),
            policy_name=Path(save_path).stem,
            imitation_epochs=int(model_cfg.get("imitation_epochs_dynamic", 1500)),
            ppo_total_steps=int(steps or model_cfg.get("ppo_total_steps_dynamic", 0)),
            ppo_n_steps=int(model_cfg.get("ppo_n_steps", 256)),
            ppo_batch_size=int(model_cfg.get("ppo_batch_size", 64)),
            ppo_learning_rate=float(model_cfg.get("ppo_learning_rate", 3e-4)),
            ppo_gamma=float(model_cfg.get("ppo_gamma", 0.99)),
            ppo_ent_coef=float(model_cfg.get("ppo_ent_coef", 0.01)),
            seed=int(model_cfg.get("seed", 7)),
            num_envs=num_envs,
            device=device,
            imitation_loss_target=imit_loss_target,
        )
    else:
        from core.rl.train import train
        train(
            problems=problems,
            artifact_dir=str(Path(save_path).parent),
            policy_name=Path(save_path).stem,
            imitation_epochs=int(model_cfg.get("imitation_epochs", 30)),
            ppo_total_steps=int(steps or model_cfg.get("ppo_total_steps", 50000)),
            ppo_n_steps=int(model_cfg.get("ppo_n_steps", 512)),
            ppo_batch_size=int(model_cfg.get("ppo_batch_size", 64)),
            ppo_learning_rate=float(model_cfg.get("ppo_learning_rate", 3e-4)),
            ppo_gamma=float(model_cfg.get("ppo_gamma", 0.99)),
            switch_penalty=float(reward_cfg.get("switch_penalty", 0.02)),
            achievement_weight=float(reward_cfg.get("achievement_weight", 1.0)),
            ignore_wip=(mode == "plan-only"),
            seed=int(model_cfg.get("seed", 7)),
            num_envs=num_envs,
            device=device,
            imitation_loss_target=imit_loss_target,
        )

    # benchmark + reports for this mode
    eval_result = _eval_for_mode(settings, mode, save_path)
    return {
        "mode": mode,
        "model_path": save_path,
        "n_problems": len(problems),
        **eval_result,
    }


# ---------------------------------------------------------------------------
def _eval_for_mode(settings: dict, mode: str, model_path: Optional[str]) -> dict:
    bench = settings.get("benchmark", {})
    root = bench.get("dataset_root", "benchmarks")
    if model_path and not Path(model_path).exists():
        model_path = None
    if mode == "dynamic":
        dyn = settings.get("dynamic", {})
        results = evaluate_all_benchmark_datasets_dynamic(
            root,
            model_path=model_path,
            num_slots=int(dyn.get("num_slots", 4)),
            slot_hours=float(dyn.get("slot_hours", 1.0)),
            switch_time_hours=float(dyn.get("switch_time_hours", 0.0)),
        )
    else:
        results = evaluate_all_benchmark_datasets(
            root, model_path=model_path, ignore_wip=(mode == "plan-only"),
        )
    suffix = mode.replace("-", "_")
    html_path = bench.get("report_path", "artifacts/reports/benchmark.html").replace(
        ".html", f"_{suffix}.html"
    )
    md_path = bench.get("summary_md", "MODEL_BENCHMARK.md").replace(
        ".md", f"_{suffix}.md"
    )
    render_html(results, html_path)
    render_markdown(results, md_path)
    return {
        "report_html": html_path,
        "report_md": md_path,
        "n_benchmarks": len(results),
        "avg_optimal": (sum(r.optimal.avg_achievement for r in results) / len(results)) if results else 0.0,
        "avg_rl": (sum(r.rl.avg_achievement for r in results) / len(results)) if results else 0.0,
        "avg_heuristic": (sum(r.heuristic.avg_achievement for r in results) / len(results)) if results else 0.0,
    }


# ---------------------------------------------------------------------------
def run_infer(
    settings: dict,
    rule_timekey: Optional[str] = None,
    benchmark_dataset: Optional[str] = None,
    output_csv: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """선택한 모드의 추론. 벤치마크면 CSV 출력, DB면 RTD_CONV_INF/HIS 기록.

    Args:
        settings: settings.json 로드 결과.
        rule_timekey: 특정 시각 (None이면 DB MAX).
        benchmark_dataset: 벤치마크 폴더 (DB 대신).
        output_csv: 벤치마크 추론 출력 CSV 경로.
        mode: plan-only | wip-static | dynamic.

    Returns:
        {"mode", "source": "benchmark"|"oracle", "rule_timekey", "rows", "output"?}

    Example:
        result = run_infer(load_settings("config/settings.json"),
                           benchmark_dataset="benchmarks/benchmark_11",
                           mode="dynamic")
        # 동일 (pk, op) 묶기 + 첫 슬롯 할당만 출력
    """
    mode = resolve_mode(settings, mode)
    model_path = model_path_for(settings, mode)
    if not Path(model_path).exists():
        model_path = None

    if benchmark_dataset:
        problem = load_problem_from_csv_dir(benchmark_dataset)
        allocation = _infer_one(problem, model_path, mode, settings)
        rows = build_conversion_rows(problem.rule_timekey, None, allocation)
        if not output_csv:
            output_csv = str(Path("artifacts/inference") / f"{problem.rule_timekey}_{mode}.csv")
        write_csv(output_csv, rows)
        return {"mode": mode, "source": "benchmark",
                "rule_timekey": problem.rule_timekey, "rows": len(rows), "output": output_csv}

    conn = _connect(settings)
    try:
        oracle = settings["oracle"]
        query_dir = oracle.get("query_dir", "config/queries")
        rk = rule_timekey or latest_rule_timekey(conn, query_dir)
        if not rk:
            raise RuntimeError("No RULE_TIMEKEY found in source query.")
        problem = load_problem_from_oracle(
            conn, query_dir, rk, settings.get("tool_groups", {}),
        )
        allocation = _infer_one(problem, model_path, mode, settings)
        rows = build_conversion_rows(rk, None, allocation)
        write_oracle(
            conn,
            query_dir=query_dir,
            rule_timekey=rk,
            rows=rows,
            write_history=bool(oracle.get("write_history", True)),
        )
        return {"mode": mode, "source": "oracle", "rule_timekey": rk, "rows": len(rows)}
    finally:
        conn.close()


def _infer_one(problem: SchedulingProblem, model_path: Optional[str], mode: str, settings: dict) -> AllocationSet:
    """Single allocation for the next time window. For `dynamic` mode this is
    the first slot of the multi-period plan — the next snapshot re-decides."""
    if mode == "dynamic":
        return _infer_dynamic_first_slot(problem, model_path, settings)
    if mode == "plan-only":
        return rl_infer(problem, model_path=model_path, ignore_wip=True)
    return rl_infer(problem, model_path=model_path, ignore_wip=False)


def _infer_dynamic_first_slot(problem, model_path, settings) -> AllocationSet:
    """Roll the dynamic policy for one slot and return that allocation."""
    from core.sim.flow import MultiPeriodSimulator, dynamic_greedy_policy
    dyn = settings.get("dynamic", {})
    num_slots = int(dyn.get("num_slots", 4))
    slot_hours = float(dyn.get("slot_hours", 1.0))
    switch_time_hours = float(dyn.get("switch_time_hours", 0.0))
    sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch_time_hours)

    if model_path and Path(model_path).exists():
        try:
            from sb3_contrib import MaskablePPO
            from core.rl.env_mp import MultiPeriodDispatchEnv
            import torch
            torch.distributions.Distribution.set_default_validate_args(False)
            model = MaskablePPO.load(model_path)
            env = MultiPeriodDispatchEnv(
                [problem], num_slots=num_slots, slot_hours=slot_hours,
                switch_time_hours=switch_time_hours, seed=0,
            )
            env._load_problem(problem)
            obs = env._observation()
            while env.slot_idx < 1:
                mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                obs, _, term, trunc, _ = env.step(int(action))
                if term or trunc:
                    break
            if env._prev_alloc is not None:
                return env._prev_alloc
        except Exception:
            pass
    # fallback: dynamic greedy for the first slot
    plan = {(pk, op): qty for pk, op, qty in problem.plan_targets()}
    wip = {(pk, op): problem.wip_of(pk, op) for pk, op, _ in problem.plan_targets()}
    return dynamic_greedy_policy(problem, wip, plan, None, 0)


# ---------------------------------------------------------------------------
def run_eval(settings: dict, mode: Optional[str] = None) -> dict:
    """벤치마크 평가. 모드 'all'이면 세 모드 나란히.

    Args:
        settings: settings.json 로드 결과.
        mode: plan-only | wip-static | dynamic | all.

    Returns:
        단일 모드: {"mode", "report_html", "report_md", "n_benchmarks",
                    "avg_optimal", "avg_rl", "avg_heuristic"}
        all      : {"modes": {<mode>: 위 dict, ...}}

    Example:
        result = run_eval(load_settings("config/settings.json"), mode="all")
        # → result["modes"]["dynamic"]["avg_rl"] == 1.0
    """
    mode = (mode or settings.get("model", {}).get("mode", "wip-static")).lower()
    if mode == "all":
        out = {}
        for m in MODES:
            mp = model_path_for(settings, m)
            out[m] = _eval_for_mode(settings, m, mp)
        return {"modes": out}
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES + ('all',)}, got {mode!r}")
    mp = model_path_for(settings, mode)
    return {"mode": mode, **_eval_for_mode(settings, mode, mp)}
