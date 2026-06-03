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
    num_envs: int = 1,
    device: str = "auto",
    imitation_loss_target: float = 0.05,
) -> str:
    """Imitation warm-start + MaskablePPO 학습. 저장된 모델 경로 반환.

    Workflow:
      1. greedy 휴리스틱을 teacher로 (obs, action) 쌍 수집
      2. cross-entropy로 PPO policy 워밍업 (조기 종료 지원)
      3. MaskablePPO로 ppo_total_steps만큼 추가 학습 (마스킹된 액션만 샘플)

    Args:
        problems: 학습 스냅샷 리스트.
        artifact_dir / policy_name: 저장 위치 (→ <dir>/<name>.zip).
        imitation_epochs: CE 최대 epoch 수.
        ppo_total_steps: PPO 추가 학습 step (0이면 imitation만).
        num_envs: SubprocVecEnv 병렬 환경 수 (CPU 가속).
        device: "auto"/"cpu"/"cuda"/"mps".
        imitation_loss_target: CE loss가 이 값보다 작으면 조기 종료.
        switch_penalty / achievement_weight: 환경의 reward 계수.
        ignore_wip: True면 plan-only 모드 환경으로 학습.

    Returns:
        저장된 모델 .zip 경로 (str).

    Example:
        save = train(
            problems=[problem1, ...],
            artifact_dir="artifacts/models", policy_name="ppo_dispatch_wip_static",
            imitation_epochs=200, ppo_total_steps=50000,
            num_envs=4, device="auto",
        )
        # → "artifacts/models/ppo_dispatch_wip_static.zip"
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        import torch
        # MaskableCategorical's super().__init__ runs validation BEFORE masking;
        # float32 softmax over 528 dims can be off from 1.0 by ~1e-5 which trips
        # PyTorch's Simplex check on newer torch (2.12+). Disable global
        # distribution validation — we control the inputs and masking happens
        # immediately after construction.
        torch.distributions.Distribution.set_default_validate_args(False)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "stable-baselines3 + sb3-contrib are required for training; "
            "install pjt_scheduler[rl] (and sb3-contrib)."
        ) from exc

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(artifact_dir) / f"{policy_name}.zip")

    def make_env(rank: int = 0):
        def _init():
            return DispatchEnv(
                problems,
                switch_penalty=switch_penalty,
                achievement_weight=achievement_weight,
                ignore_wip=ignore_wip,
                seed=seed + rank,
            )
        return _init

    n = max(1, int(num_envs))
    if n == 1:
        vec = DummyVecEnv([make_env(0)])
    else:
        vec = SubprocVecEnv([make_env(i) for i in range(n)])

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec,
        learning_rate=ppo_learning_rate,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        gamma=ppo_gamma,
        device=device,
        seed=seed,
        verbose=0,
    )

    # --- imitation warm-start --------------------------------------------
    teacher_env = DispatchEnv(
        problems, switch_penalty=switch_penalty,
        achievement_weight=achievement_weight, ignore_wip=ignore_wip, seed=seed,
    )
    obs_dataset: List[np.ndarray] = []
    act_dataset: List[int] = []
    for problem in problems:
        teacher_env._load_problem(problem)
        obs_seq, act_seq = _teacher_rollout(teacher_env)
        obs_dataset.extend(obs_seq)
        act_dataset.extend(act_seq)

    if obs_dataset:
        dev = next(model.policy.parameters()).device
        obs_tensor = torch.as_tensor(np.array(obs_dataset), dtype=torch.float32, device=dev)
        act_tensor = torch.as_tensor(np.array(act_dataset), dtype=torch.long, device=dev)
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
            if float(loss.item()) < imitation_loss_target:
                break

    # --- PPO continuation -------------------------------------------------
    if ppo_total_steps > 0:
        model.learn(total_timesteps=int(ppo_total_steps))
    model.save(save_path)
    vec.close()
    return save_path


def load_policy(model_path: str):
    """Load a saved MaskablePPO policy. Returns None if dependencies missing,
    file is absent, or the file was produced by an incompatible policy class
    (e.g. a pre-masking PPO checkpoint left over on disk)."""
    if not os.path.exists(model_path):
        return None
    try:
        from sb3_contrib import MaskablePPO
        import torch
        torch.distributions.Distribution.set_default_validate_args(False)
        return MaskablePPO.load(model_path)
    except Exception:
        return None
