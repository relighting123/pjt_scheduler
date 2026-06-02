"""Imitation-learning warm-start + PPO training.

The teacher policy is the greedy heuristic — it picks (bucket, target) pairs
with the highest marginal contribution. We replay the teacher's trajectories
into a supervised classifier head to warm-start PPO, then continue with
on-policy PPO updates.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .domain import SchedulingProblem
from .heuristic import greedy_allocate
from .rl_env import DispatchEnv


def _teacher_rollout(env: DispatchEnv) -> Tuple[List[np.ndarray], List[int]]:
    """Roll the greedy teacher once and return (observations, actions) pairs.

    Assumes the env is already reset to a problem by the caller.
    """
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    if env.problem is None:
        return obs_list, act_list

    cur_obs = env._observation()
    while True:
        best = None
        best_score = -1.0
        for i in range(len(env.bucket_keys)):
            if env.bucket_free[i] <= 0:
                continue
            for j in range(len(env.target_keys)):
                if (
                    env.avail_matrix[i, j] <= 0.0
                    or env.target_shortfall[j] <= 0
                    or env.target_wip[j] <= 0
                ):
                    continue
                # marginal score == min(uph, shortfall, wip) scaled by uph_matrix
                # which is already uph/plan_scale.
                score = float(env.uph_matrix[i, j])
                if score > best_score:
                    best_score = score
                    best = (i, j)
        if best is None:
            action = 0 * (env.MAX_TARGETS + 1) + env.MAX_TARGETS
            obs_list.append(cur_obs.copy())
            act_list.append(action)
            env.step(action)
            break
        i, j = best
        action = i * (env.MAX_TARGETS + 1) + j
        obs_list.append(cur_obs.copy())
        act_list.append(action)
        cur_obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    return obs_list, act_list


def train(
    problems: List[SchedulingProblem],
    artifact_dir: str,
    policy_name: str,
    imitation_epochs: int = 30,
    ppo_total_steps: int = 50000,
    ppo_n_steps: int = 512,
    ppo_batch_size: int = 64,
    ppo_learning_rate: float = 3e-4,
    ppo_gamma: float = 0.99,
    switch_penalty: float = 0.02,
    achievement_weight: float = 1.0,
    ignore_wip: bool = False,
    seed: int = 7,
) -> str:
    """Run imitation warm-start + PPO. Returns the path to the saved policy."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for training; install pjt_scheduler[rl].") from exc

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(artifact_dir) / f"{policy_name}.zip")

    def make_env() -> DispatchEnv:
        return DispatchEnv(
            problems,
            switch_penalty=switch_penalty,
            achievement_weight=achievement_weight,
            ignore_wip=ignore_wip,
            seed=seed,
        )

    vec = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=ppo_learning_rate,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        gamma=ppo_gamma,
        seed=seed,
        verbose=0,
    )

    # --- imitation warm-start --------------------------------------------
    teacher_env = make_env()
    obs_dataset: List[np.ndarray] = []
    act_dataset: List[int] = []
    # iterate over each training problem deterministically so coverage is full
    for problem in problems:
        teacher_env._load_problem(problem)
        obs_seq, act_seq = _teacher_rollout(teacher_env)
        obs_dataset.extend(obs_seq)
        act_dataset.extend(act_seq)

    if obs_dataset:
        obs_tensor = torch.as_tensor(np.array(obs_dataset), dtype=torch.float32)
        act_tensor = torch.as_tensor(np.array(act_dataset), dtype=torch.long)
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        model.policy.train()
        for _ in range(max(1, int(imitation_epochs))):
            dist = model.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits
            loss = loss_fn(logits, act_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- PPO continuation -------------------------------------------------
    model.learn(total_timesteps=int(ppo_total_steps))
    model.save(save_path)
    return save_path


def load_policy(model_path: str):
    """Load a saved PPO policy. Returns None if SB3 isn't available."""
    try:
        from stable_baselines3 import PPO
    except Exception:
        return None
    if not os.path.exists(model_path):
        return None
    return PPO.load(model_path)
