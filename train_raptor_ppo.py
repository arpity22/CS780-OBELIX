import argparse
import os
import random
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


def env_worker(remote, obelix_py: str, args, seed: int):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    env = mod.OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=seed,
    )

    success_bonus = 2000

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                s, r, d = env.step(data, render=False)
                is_success = r >= success_bonus
                if d:
                    s = env.reset()
                remote.send((s, r, d, is_success))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                remote.close()
                break
        except EOFError:
            break


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float64)
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


class RecurrentActorCritic(nn.Module):
    def __init__(self, in_dim: int = 43, n_actions: int = 5, hidden_dim: int = 192):
        super().__init__()
        self.in_dim = in_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 192),
            nn.SiLU(),
        )
        self.gru = nn.GRU(192, hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.dynamics_head = nn.Linear(hidden_dim, 18)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(self, obs_t: torch.Tensor, hidden: torch.Tensor):
        # obs_t: [B, in_dim]
        z = self.encoder(obs_t)
        z, hidden = self.gru(z.unsqueeze(0), hidden)
        z = z.squeeze(0)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        pred_next_obs = self.dynamics_head(z)
        return logits, value, pred_next_obs, hidden

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        starts_seq: torch.Tensor,
        actions_seq: torch.Tensor,
    ):
        # obs_seq: [T, B, in_dim], starts_seq: [T, B] where 1 means episode start
        t_steps, batch_size = obs_seq.shape[0], obs_seq.shape[1]
        hidden = self.initial_hidden(batch_size, obs_seq.device)

        logits_list = []
        values_list = []
        pred_next_list = []

        for t in range(t_steps):
            reset_mask = (1.0 - starts_seq[t]).view(1, batch_size, 1)
            hidden = hidden * reset_mask
            logits_t, values_t, pred_next_t, hidden = self.step(obs_seq[t], hidden)
            logits_list.append(logits_t)
            values_list.append(values_t)
            pred_next_list.append(pred_next_t)

        logits = torch.stack(logits_list, dim=0)
        values = torch.stack(values_list, dim=0)
        pred_next = torch.stack(pred_next_list, dim=0)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_seq)
        entropy = dist.entropy()
        return log_probs, entropy, values, pred_next


class RNDModule(nn.Module):
    def __init__(self, in_dim: int = 43, feat_dim: int = 128):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def intrinsic(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(obs)
        pred = self.predictor(obs)
        return torch.mean((pred - target) ** 2, dim=-1)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(obs)
        pred = self.predictor(obs)
        return torch.mean((pred - target) ** 2)


@dataclass
class TrackerState:
    last_action: np.ndarray
    prev_obs: np.ndarray
    steps_since_seen: np.ndarray
    ep_steps: np.ndarray
    ep_starts: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_obs(
    obs: np.ndarray,
    prev_obs: np.ndarray,
    last_action: np.ndarray,
    steps_since_seen: np.ndarray,
    ep_steps: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    # Augment with motion cue and temporal context for partial observability.
    delta = obs - prev_obs
    one_hot = np.zeros((obs.shape[0], 5), dtype=np.float32)
    one_hot[np.arange(obs.shape[0]), last_action] = 1.0
    seen = np.minimum(1.0, steps_since_seen.astype(np.float32) / 120.0).reshape(-1, 1)
    progress = np.minimum(1.0, ep_steps.astype(np.float32) / float(max_steps)).reshape(-1, 1)
    aug = np.concatenate([obs, delta, one_hot, seen, progress], axis=-1)
    return aug.astype(np.float32)


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    horizon, num_envs = rewards.shape
    advantages = np.zeros((horizon, num_envs), dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAPTOR-PPO: Recurrent Action-conditioned PPO + RND for OBELIX"
    )
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--total_steps", type=int, default=5_000_000)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.7)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.8)

    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--envs_per_batch", type=int, default=4)

    parser.add_argument("--intrinsic_coef", type=float, default=0.25)
    parser.add_argument("--rnd_coef", type=float, default=0.5)
    parser.add_argument("--dyn_coef", type=float, default=0.05)

    parser.add_argument("--save_dir", type=str, default="submission_raptor_ppo")
    return parser.parse_args()


def train():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 43

    actor_critic = RecurrentActorCritic(in_dim=obs_dim, n_actions=5, hidden_dim=192).to(device)
    rnd_module = RNDModule(in_dim=obs_dim, feat_dim=128).to(device)

    optimizer = optim.Adam(
        list(actor_critic.parameters()) + list(rnd_module.predictor.parameters()),
        lr=args.lr,
    )

    rms_intrinsic = RunningMeanStd()

    remotes, work_remotes = zip(*[mp.Pipe() for _ in range(args.num_envs)])
    workers = [
        mp.Process(target=env_worker, args=(wr, args.obelix_py, args, args.seed + 1000 + i))
        for i, wr in enumerate(work_remotes)
    ]

    for w in workers:
        w.start()

    for r in remotes:
        r.send(("reset", None))
    obs = np.stack([r.recv() for r in remotes]).astype(np.float32)

    tracker = TrackerState(
        last_action=np.full(args.num_envs, 2, dtype=np.int64),
        prev_obs=obs.copy(),
        steps_since_seen=np.zeros(args.num_envs, dtype=np.int64),
        ep_steps=np.zeros(args.num_envs, dtype=np.int64),
        ep_starts=np.ones(args.num_envs, dtype=np.float32),
    )

    hidden = actor_critic.initial_hidden(args.num_envs, device)

    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    reward_window = deque(maxlen=300)
    length_window = deque(maxlen=300)
    success_window = deque(maxlen=500)

    global_step = 0
    rollout_id = 0

    print(
        f"Training RAPTOR-PPO | envs={args.num_envs} horizon={args.horizon} total_steps={args.total_steps}"
    )

    try:
        while global_step < args.total_steps:
            rollout_id += 1

            obs_buf = np.zeros((args.horizon, args.num_envs, obs_dim), dtype=np.float32)
            starts_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
            actions_buf = np.zeros((args.horizon, args.num_envs), dtype=np.int64)
            logp_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
            values_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
            next_raw_buf = np.zeros((args.horizon, args.num_envs, 18), dtype=np.float32)
            rewards_ext_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
            rewards_int_raw_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
            dones_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)

            for t in range(args.horizon):
                seen_now = np.any(obs[:, :17] > 0, axis=1)
                tracker.steps_since_seen[seen_now] = 0
                tracker.steps_since_seen[~seen_now] += 1

                aug_obs = augment_obs(
                    obs,
                    tracker.prev_obs,
                    tracker.last_action,
                    tracker.steps_since_seen,
                    tracker.ep_steps,
                    args.max_steps,
                )

                obs_t = torch.tensor(aug_obs, dtype=torch.float32, device=device)
                starts_t = torch.tensor(tracker.ep_starts, dtype=torch.float32, device=device)

                with torch.no_grad():
                    reset_mask = (1.0 - starts_t).view(1, args.num_envs, 1)
                    hidden = hidden * reset_mask

                    logits, values, _, hidden = actor_critic.step(obs_t, hidden)
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    intr_raw = rnd_module.intrinsic(obs_t)

                action_idx = actions.cpu().numpy().astype(np.int64)

                obs_buf[t] = aug_obs
                starts_buf[t] = tracker.ep_starts
                actions_buf[t] = action_idx
                logp_buf[t] = log_probs.cpu().numpy()
                values_buf[t] = values.cpu().numpy()
                rewards_int_raw_buf[t] = intr_raw.cpu().numpy()

                for r, a in zip(remotes, action_idx):
                    r.send(("step", ACTIONS[int(a)]))
                results = [r.recv() for r in remotes]

                next_obs = np.zeros_like(obs)
                done_mask = np.zeros(args.num_envs, dtype=np.float32)

                for i, (s2, rew, done, success) in enumerate(results):
                    s2 = np.asarray(s2, dtype=np.float32)
                    next_obs[i] = s2

                    scaled_rew = float(np.clip(rew / 100.0, -5.0, 25.0))
                    rewards_ext_buf[t, i] = scaled_rew

                    done_mask[i] = 1.0 if done else 0.0
                    next_raw_buf[t, i] = s2

                    ep_rewards[i] += rew
                    ep_lengths[i] += 1
                    tracker.ep_steps[i] += 1

                    if done:
                        reward_window.append(float(ep_rewards[i]))
                        length_window.append(int(ep_lengths[i]))
                        success_window.append(1.0 if success else 0.0)

                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0
                        tracker.ep_steps[i] = 0
                        tracker.last_action[i] = 2
                        tracker.steps_since_seen[i] = 0
                        tracker.prev_obs[i] = s2
                        hidden[:, i : i + 1, :] = 0.0
                    else:
                        tracker.last_action[i] = action_idx[i]
                        tracker.prev_obs[i] = obs[i]

                dones_buf[t] = done_mask
                tracker.ep_starts = done_mask.copy()
                obs = next_obs

            # Bootstrap value for final state.
            seen_now = np.any(obs[:, :17] > 0, axis=1)
            tracker.steps_since_seen[seen_now] = 0
            tracker.steps_since_seen[~seen_now] += 1
            aug_obs_last = augment_obs(
                obs,
                tracker.prev_obs,
                tracker.last_action,
                tracker.steps_since_seen,
                tracker.ep_steps,
                args.max_steps,
            )

            with torch.no_grad():
                obs_last_t = torch.tensor(aug_obs_last, dtype=torch.float32, device=device)
                starts_last_t = torch.tensor(tracker.ep_starts, dtype=torch.float32, device=device)
                reset_mask = (1.0 - starts_last_t).view(1, args.num_envs, 1)
                hidden_boot = hidden * reset_mask
                _, last_values_t, _, _ = actor_critic.step(obs_last_t, hidden_boot)
                last_values = last_values_t.cpu().numpy()

            rms_intrinsic.update(rewards_int_raw_buf.reshape(-1))
            intr_scale = 1.0 / np.sqrt(rms_intrinsic.var + 1e-8)
            rewards_int = np.clip(rewards_int_raw_buf * intr_scale, 0.0, 5.0)

            rewards_total = rewards_ext_buf + args.intrinsic_coef * rewards_int

            advantages, returns = compute_gae(
                rewards_total,
                dones_buf,
                values_buf,
                last_values,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            adv_mean = float(np.mean(advantages))
            adv_std = float(np.std(advantages) + 1e-8)
            advantages = (advantages - adv_mean) / adv_std

            obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
            starts_t = torch.tensor(starts_buf, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions_buf, dtype=torch.long, device=device)
            old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
            old_values_t = torch.tensor(values_buf, dtype=torch.float32, device=device)
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
            next_raw_t = torch.tensor(next_raw_buf, dtype=torch.float32, device=device)

            n_envs = args.num_envs
            for _ in range(args.ppo_epochs):
                env_perm = np.random.permutation(n_envs)
                for start in range(0, n_envs, args.envs_per_batch):
                    mb_env = env_perm[start : start + args.envs_per_batch]
                    if mb_env.size == 0:
                        continue

                    logp_new, entropy, values_new, pred_next = actor_critic.evaluate_sequence(
                        obs_t[:, mb_env], starts_t[:, mb_env], actions_t[:, mb_env]
                    )

                    ratio = torch.exp(logp_new - old_logp_t[:, mb_env])
                    surr1 = ratio * adv_t[:, mb_env]
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * adv_t[:, mb_env]
                    policy_loss = -torch.mean(torch.min(surr1, surr2))

                    value_pred_clipped = old_values_t[:, mb_env] + torch.clamp(
                        values_new - old_values_t[:, mb_env],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    value_losses = (values_new - ret_t[:, mb_env]) ** 2
                    value_losses_clipped = (value_pred_clipped - ret_t[:, mb_env]) ** 2
                    value_loss = 0.5 * torch.mean(torch.max(value_losses, value_losses_clipped))

                    entropy_loss = torch.mean(entropy)

                    dyn_loss = torch.mean((pred_next - next_raw_t[:, mb_env]) ** 2)
                    rnd_loss = rnd_module.predictor_loss(obs_t[:, mb_env].reshape(-1, obs_dim))

                    total_loss = (
                        policy_loss
                        + args.vf_coef * value_loss
                        - args.ent_coef * entropy_loss
                        + args.rnd_coef * rnd_loss
                        + args.dyn_coef * dyn_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(actor_critic.parameters()) + list(rnd_module.predictor.parameters()),
                        args.max_grad_norm,
                    )
                    optimizer.step()

            global_step += args.horizon * args.num_envs

            if rollout_id % 5 == 0:
                avg_rew = float(np.mean(reward_window)) if reward_window else 0.0
                avg_len = float(np.mean(length_window)) if length_window else 0.0
                sr = float(np.mean(success_window) * 100.0) if success_window else 0.0
                avg_ext = float(np.mean(rewards_ext_buf))
                avg_int = float(np.mean(rewards_int))
                print(
                    f"Step {global_step}/{args.total_steps} | AvgEpRew {avg_rew:.1f} | "
                    f"AvgEpLen {avg_len:.1f} | Success {sr:.1f}% | Ext {avg_ext:.3f} | Int {avg_int:.3f}"
                )

    finally:
        os.makedirs(args.save_dir, exist_ok=True)
        weights_path = os.path.join(args.save_dir, "weights.pth")
        torch.save(
            {
                "model_state_dict": actor_critic.state_dict(),
                "obs_dim": obs_dim,
                "hidden_dim": actor_critic.hidden_dim,
                "algorithm": "RAPTOR-PPO-RND",
            },
            weights_path,
        )
        print(f"Saved model weights to {weights_path}")

        for r in remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for w in workers:
            w.terminate()
            w.join()
        print("Training shutdown complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()
