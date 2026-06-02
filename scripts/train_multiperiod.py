"""Train PPO on the multi-period (WIP-flow + switch cost) environment and
compare against the dynamic-greedy and optimal policies.

Run:
    python scripts/train_multiperiod.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.domain import AllocationSet, SchedulingProblem  # noqa: E402
from core.flow import (  # noqa: E402
    MultiPeriodSimulator,
    dynamic_greedy_policy,
    multiperiod_optimal,
    static_policy,
)
from core.rl_env_mp import MultiPeriodDispatchEnv  # noqa: E402

from test_multiperiod import (  # noqa: E402
    build_buildahead_problem,
    build_thrashing_problem,
)


def _replay_from_env(env: MultiPeriodDispatchEnv, model, problem: SchedulingProblem):
    """Run the trained policy deterministically and return the slot schedule."""
    env._load_problem(problem)
    obs = env._observation()
    schedule: List[AllocationSet] = []
    done = False
    safety = env.MAX_BUCKETS * (env.MAX_TARGETS + 1) * env.num_slots
    while not done and safety > 0:
        action, _ = model.predict(obs, deterministic=True)
        prev_slot = env.slot_idx
        obs, _, term, trunc, _ = env.step(int(action))
        # capture committed slot allocations as the slot advances
        if env.slot_idx > prev_slot:
            if env._prev_alloc is not None:
                schedule.append(env._prev_alloc)
        done = term or trunc
        safety -= 1
    while len(schedule) < env.num_slots:
        schedule.append(AllocationSet(rule_timekey=problem.rule_timekey, allocations=[]))
    return schedule


def _evaluate_schedule(problem, schedule, num_slots, slot_hours, switch_time_hours):
    sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch_time_hours)
    idx = {"i": 0}

    def replay(*a, **kw):
        r = schedule[idx["i"]] if idx["i"] < len(schedule) else AllocationSet(
            rule_timekey=problem.rule_timekey, allocations=[]
        )
        idx["i"] += 1
        return r

    return sim.run(replay)


def main() -> int:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        print(f"stable-baselines3 required: {exc}")
        return 1

    scenarios = [
        ("build-ahead", build_buildahead_problem(), 2, 1.0, 0.0),
        ("thrashing",    build_thrashing_problem(), 4, 1.0, 0.5),
    ]

    print(f"{'scenario':<14} {'policy':<14} {'avg_achv':>9} {'switches':>9}")
    print("-" * 56)

    for name, problem, num_slots, slot_hours, switch_time in scenarios:
        # Baselines.
        sim = MultiPeriodSimulator(problem, num_slots, slot_hours, switch_time)
        static = sim.run(static_policy)
        dyn = sim.run(dynamic_greedy_policy)
        opt = multiperiod_optimal(problem, num_slots, slot_hours, switch_time)

        # Train PPO on this scenario.
        def make_env():
            return MultiPeriodDispatchEnv(
                [problem],
                num_slots=num_slots,
                slot_hours=slot_hours,
                switch_time_hours=switch_time,
                seed=7,
            )

        vec = DummyVecEnv([make_env])
        model = PPO(
            "MlpPolicy", vec,
            learning_rate=5e-4,
            n_steps=256, batch_size=64, gamma=0.99,
            ent_coef=0.05, seed=7, verbose=0,
        )
        model.learn(total_timesteps=150000)

        eval_env = make_env()
        schedule = _replay_from_env(eval_env, model, problem)
        rl = _evaluate_schedule(problem, schedule, num_slots, slot_hours, switch_time)

        for label, r in (("static", static), ("dyn-greedy", dyn), ("PPO", rl), ("optimal", opt)):
            print(f"{name:<14} {label:<14} {r.avg_achievement:>9.3f} {r.total_switches:>9}")
        print("  PPO schedule:")
        for i, alloc in enumerate(rl.schedule):
            items = [(a.plan_prod_key, a.oper_id, a.batch_id, a.eqp_qty) for a in alloc.allocations] or ["idle"]
            print(f"    slot {i}: {items}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
