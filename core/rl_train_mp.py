"""Imitation warm-start + PPO training for the multi-period env.

The optimal multi-period schedule (`multiperiod_optimal`) is replayed against
the env to produce (observation, action) pairs. Those pairs initialise the
PPO policy via cross-entropy, then PPO refines with on-policy updates.
Cuts the search problem for PPO from "discover the time-ordering" to
"keep the demonstrated time-ordering on novel observations."
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .domain import AllocationSet, SchedulingProblem
from .flow import multiperiod_optimal
from .rl_env_mp import MultiPeriodDispatchEnv


def _schedule_to_actions(
    env: MultiPeriodDispatchEnv,
    schedule: List[AllocationSet],
) -> List[int]:
    """Convert a per-slot optimal schedule into the env's substep actions.

    For each slot: emit one substep action per unit, then a NO-OP to commit.
    Bucket index is the slot of (batch, model) in env.bucket_keys; target index
    is the slot of (plan_prod_key, oper_id) in env.target_keys.
    """
    actions: List[int] = []
    for alloc in schedule:
        for a in alloc.allocations:
            try:
                target_idx = env.target_keys.index((a.plan_prod_key, a.oper_id))
            except ValueError:
                continue
            # find a bucket whose (batch, model) matches; otherwise any same-model
            bucket_idx = -1
            for i, (b_id, model) in enumerate(env.bucket_keys):
                if model == a.eqp_model_cd:
                    bucket_idx = i
                    break
            if bucket_idx < 0:
                continue
            # one action per unit
            for _ in range(int(a.eqp_qty)):
                actions.append(bucket_idx * (env.MAX_TARGETS + 1) + target_idx)
        # commit slot
        actions.append(0 * (env.MAX_TARGETS + 1) + env.MAX_TARGETS)
    return actions


def _record_imitation_rollout(
    env: MultiPeriodDispatchEnv,
    problem: SchedulingProblem,
    schedule: List[AllocationSet],
) -> Tuple[List[np.ndarray], List[int]]:
    """Replay the teacher schedule in the env and capture (obs, action) pairs."""
    env._load_problem(problem)
    obs = env._observation()
    actions = _schedule_to_actions(env, schedule)
    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    for a in actions:
        obs_buf.append(obs.copy())
        act_buf.append(int(a))
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            break
    return obs_buf, act_buf


def train_multiperiod(
    problems: List[SchedulingProblem],
    num_slots: int,
    slot_hours: float,
    switch_time_hours: float,
    artifact_dir: str,
    policy_name: str,
    imitation_epochs: int = 30,
    ppo_total_steps: int = 80000,
    ppo_n_steps: int = 256,
    ppo_batch_size: int = 64,
    ppo_learning_rate: float = 3e-4,
    ppo_gamma: float = 0.99,
    ppo_ent_coef: float = 0.02,
    achievement_weight: float = 1.0,
    seed: int = 7,
) -> str:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required; install pjt_scheduler[rl].") from exc

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(artifact_dir) / f"{policy_name}.zip")

    def make_env():
        return MultiPeriodDispatchEnv(
            problems,
            num_slots=num_slots,
            slot_hours=slot_hours,
            switch_time_hours=switch_time_hours,
            achievement_weight=achievement_weight,
            seed=seed,
        )

    vec = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy", vec,
        learning_rate=ppo_learning_rate,
        n_steps=ppo_n_steps, batch_size=ppo_batch_size, gamma=ppo_gamma,
        ent_coef=ppo_ent_coef, seed=seed, verbose=0,
    )

    # --- imitation warm-start from multiperiod_optimal -------------------
    teacher_env = make_env()
    obs_dataset: List[np.ndarray] = []
    act_dataset: List[int] = []
    for problem in problems:
        opt = multiperiod_optimal(problem, num_slots, slot_hours, switch_time_hours)
        obs_seq, act_seq = _record_imitation_rollout(teacher_env, problem, opt.schedule)
        obs_dataset.extend(obs_seq)
        act_dataset.extend(act_seq)

    if obs_dataset:
        obs_tensor = torch.as_tensor(np.array(obs_dataset), dtype=torch.float32)
        act_tensor = torch.as_tensor(np.array(act_dataset), dtype=torch.long)
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        model.policy.train()
        for _ in range(int(imitation_epochs)):
            dist = model.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits
            loss = loss_fn(logits, act_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- PPO continuation ------------------------------------------------
    model.learn(total_timesteps=int(ppo_total_steps))
    model.save(save_path)
    return save_path
