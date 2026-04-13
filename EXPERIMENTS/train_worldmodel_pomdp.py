import argparse
import os
import random
import signal
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


def env_worker(remote, obelix_py: str, args, seed: int):
    import importlib.util

    signal.signal(signal.SIGINT, signal.SIG_IGN)

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

    def is_success_transition(env_obj, reward: float, done: bool) -> bool:
        if not done:
            return False
        # OBELIX success is attached box reaching boundary.
        if bool(getattr(env_obj, "enable_push", False)):
            try:
                if bool(env_obj._box_touches_boundary(env_obj.box_center_x, env_obj.box_center_y)):
                    return True
            except Exception:
                pass
        # Fallback for compatibility with environment variants.
        return reward >= 1900.0

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                s2, r, d = env.step(data, render=False)
                is_success = bool(is_success_transition(env, float(r), bool(d)))
                reset_obs = None
                if d:
                    reset_obs = env.reset()
                remote.send((s2, r, d, is_success, reset_obs))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                remote.close()
                break
        except EOFError:
            break
        except KeyboardInterrupt:
            break


@dataclass
class Episode:
    obs: np.ndarray          # [T+1, obs_dim]
    actions: np.ndarray      # [T]
    rewards: np.ndarray      # [T]
    dones: np.ndarray        # [T]


class EpisodeReplay:
    def __init__(self, capacity_episodes: int = 12000):
        self.episodes: Deque[Episode] = deque(maxlen=capacity_episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(
        self,
        obs_list: List[np.ndarray],
        action_list: List[int],
        reward_list: List[float],
        done_list: List[float],
    ) -> None:
        if len(action_list) < 2:
            return
        obs = np.asarray(obs_list, dtype=np.float32)
        actions = np.asarray(action_list, dtype=np.int64)
        rewards = np.asarray(reward_list, dtype=np.float32)
        dones = np.asarray(done_list, dtype=np.float32)
        if obs.shape[0] != actions.shape[0] + 1:
            return
        self.episodes.append(Episode(obs=obs, actions=actions, rewards=rewards, dones=dones))

    def can_sample(self, batch_size: int, seq_len: int) -> bool:
        eligible = sum(1 for ep in self.episodes if ep.actions.shape[0] >= seq_len)
        return eligible > 0 and len(self.episodes) > 8 and batch_size > 0

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eligible = [ep for ep in self.episodes if ep.actions.shape[0] >= seq_len]
        assert eligible, "No eligible episodes to sample from."

        obs_batch = []
        actions_batch = []
        rewards_batch = []
        dones_batch = []

        for _ in range(batch_size):
            ep = random.choice(eligible)
            max_start = ep.actions.shape[0] - seq_len
            start = random.randint(0, max_start)

            obs_seq = ep.obs[start : start + seq_len + 1]
            act_seq = ep.actions[start : start + seq_len]
            rew_seq = ep.rewards[start : start + seq_len]
            done_seq = ep.dones[start : start + seq_len]

            obs_batch.append(obs_seq)
            actions_batch.append(act_seq)
            rewards_batch.append(rew_seq)
            dones_batch.append(done_seq)

        return (
            np.asarray(obs_batch, dtype=np.float32),
            np.asarray(actions_batch, dtype=np.int64),
            np.asarray(rewards_batch, dtype=np.float32),
            np.asarray(dones_batch, dtype=np.float32),
        )


class WorldModel(nn.Module):
    """
    Recurrent world model with:
    - posterior filtering from observation
    - prior transition from latent + action
    - heads for next-observation, reward, and done prediction
    """

    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 5,
        hidden_dim: int = 192,
        latent_dim: int = 96,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
        )

        self.filter_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.posterior_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

        self.prior_gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

        self.obs_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def posterior(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.obs_encoder(obs)
        h = self.filter_gru(emb, h_prev)
        z = self.posterior_proj(h)
        return h, z

    def transition(
        self,
        z_prev: torch.Tensor,
        action_one_hot: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_prev, action_one_hot], dim=-1)
        h = self.prior_gru(x, h_prev)
        z = self.prior_proj(h)
        return h, z

    def heads(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_logits = self.obs_head(z)
        reward = self.reward_head(z).squeeze(-1)
        done_logit = self.done_head(z).squeeze(-1)
        return obs_logits, reward, done_logit


class Actor(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class Critic(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.value_scale = 400.0
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        raw = self.net(feat)
        # Bound value predictions to prevent runaway bootstrapped targets.
        return self.value_scale * torch.tanh(raw / self.value_scale)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot_actions(actions: torch.Tensor, n_actions: int) -> torch.Tensor:
    return F.one_hot(actions, num_classes=n_actions).float()


def world_model_loss(
    wm: WorldModel,
    obs_seq: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # obs_seq: [B, L+1, obs_dim]
    # actions/rewards/dones: [B, L]
    bsz, seq_len = actions.shape
    device = obs_seq.device

    h, z = wm.init_state(bsz, device)
    h, z = wm.posterior(obs_seq[:, 0], h)

    loss_obs = torch.zeros((), device=device)
    loss_rew = torch.zeros((), device=device)
    loss_done = torch.zeros((), device=device)
    loss_lat = torch.zeros((), device=device)

    for t in range(seq_len):
        a_oh = one_hot_actions(actions[:, t], wm.action_dim)
        h_pred, z_pred = wm.transition(z, a_oh, h)

        obs_logits, rew_pred, done_logit = wm.heads(z_pred)

        h_next, z_next = wm.posterior(obs_seq[:, t + 1], h)

        # Observation is mostly binary sensor state; BCE-with-logits is stable here.
        loss_obs = loss_obs + F.binary_cross_entropy_with_logits(obs_logits, obs_seq[:, t + 1].clamp(0.0, 1.0))
        loss_rew = loss_rew + F.smooth_l1_loss(rew_pred, rewards[:, t])
        loss_done = loss_done + F.binary_cross_entropy_with_logits(done_logit, dones[:, t])
        loss_lat = loss_lat + F.mse_loss(z_pred, z_next.detach())

        h, z = h_next, z_next

    denom = float(seq_len)
    loss_obs = loss_obs / denom
    loss_rew = loss_rew / denom
    loss_done = loss_done / denom
    loss_lat = loss_lat / denom

    total = 1.0 * loss_obs + 1.5 * loss_rew + 0.3 * loss_done + 0.6 * loss_lat
    stats = {
        "wm_obs": float(loss_obs.item()),
        "wm_rew": float(loss_rew.item()),
        "wm_done": float(loss_done.item()),
        "wm_lat": float(loss_lat.item()),
        "wm_total": float(total.item()),
    }
    return total, stats


def posterior_last_state(wm: WorldModel, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = obs_seq.shape[0]
    h, z = wm.init_state(bsz, obs_seq.device)
    for t in range(obs_seq.shape[1]):
        h, z = wm.posterior(obs_seq[:, t], h)
    return h, z


def imagine_actor_critic_loss(
    wm: WorldModel,
    actor: Actor,
    critic: Critic,
    obs_seq: torch.Tensor,
    imag_horizon: int,
    gamma: float,
    lam: float,
    ent_coef: float,
    imag_reward_min: float,
    imag_reward_max: float,
    value_target_min: float,
    value_target_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        h, z = posterior_last_state(wm, obs_seq)

    values = []
    logps = []
    entropies = []
    rewards = []
    continues = []

    for _ in range(imag_horizon):
        feat = torch.cat([h, z], dim=-1)
        logits = actor(feat)
        dist = Categorical(logits=logits)
        action = dist.sample()

        values.append(critic(feat).squeeze(-1))
        logps.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        with torch.no_grad():
            a_oh = one_hot_actions(action, wm.action_dim)
            h, z = wm.transition(z, a_oh, h)
            _, rew_pred, done_logit = wm.heads(z)
            rew_pred = torch.clamp(rew_pred, imag_reward_min, imag_reward_max)
            cont = 1.0 - torch.sigmoid(done_logit)
            rewards.append(rew_pred)
            continues.append(cont.clamp(0.0, 1.0))

    feat_last = torch.cat([h, z], dim=-1)
    value_last = critic(feat_last).squeeze(-1)

    values_t = torch.stack(values, dim=0)           # [H, B]
    logps_t = torch.stack(logps, dim=0)             # [H, B]
    ent_t = torch.stack(entropies, dim=0)           # [H, B]
    rewards_t = torch.stack(rewards, dim=0)         # [H, B]
    cont_t = torch.stack(continues, dim=0)          # [H, B]

    next_values = torch.cat([values_t[1:], value_last.unsqueeze(0)], dim=0)

    returns = []
    g = value_last
    for t in reversed(range(imag_horizon)):
        g = rewards_t[t] + gamma * cont_t[t] * ((1.0 - lam) * next_values[t] + lam * g)
        g = torch.clamp(g, value_target_min, value_target_max)
        returns.append(g)
    returns = torch.stack(list(reversed(returns)), dim=0)

    adv = returns - values_t
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    actor_loss = -(logps_t * adv.detach()).mean() - ent_coef * ent_t.mean()
    critic_loss = F.smooth_l1_loss(values_t, returns.detach())

    stats = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "imag_rew": float(rewards_t.mean().item()),
        "imag_cont": float(cont_t.mean().item()),
    }
    return actor_loss, critic_loss, stats


@torch.no_grad()
def _deterministic_eval_action(
    actor: Actor,
    obs: np.ndarray,
    h: torch.Tensor,
    z: torch.Tensor,
    ep_step: int,
    steps_since_seen: int,
    search_turn_dir: int,
    stuck_count: int,
) -> Tuple[int, int, int, int]:
    sees_target = bool(np.any(obs[:17] > 0))
    front_active = bool(np.any(obs[4:12] > 0))

    if sees_target:
        steps_since_seen = 0
    else:
        steps_since_seen += 1

    if bool(obs[17] > 0.5):
        stuck_count += 1
        phase = stuck_count % 4
        if phase == 2:
            search_turn_dir *= -1
        if phase in (1, 2):
            action_idx = 0 if search_turn_dir > 0 else 4
        else:
            action_idx = 2
        return action_idx, steps_since_seen, search_turn_dir, stuck_count

    stuck_count = 0

    if (not sees_target) and steps_since_seen > 35:
        if (ep_step % 5) == 0:
            search_turn_dir *= -1
        if (ep_step % 5) in (0, 1):
            action_idx = 1 if search_turn_dir > 0 else 3
        else:
            action_idx = 2
        return action_idx, steps_since_seen, search_turn_dir, stuck_count

    feat = torch.cat([h, z], dim=-1)
    logits = actor(feat)[0]
    if front_active:
        action_idx = 2
    else:
        action_idx = int(torch.argmax(logits).item())
    return action_idx, steps_since_seen, search_turn_dir, stuck_count


@torch.no_grad()
def evaluate_policy(
    wm: WorldModel,
    actor: Actor,
    args,
    device: torch.device,
    episodes: int,
    seed_offset: int,
) -> Tuple[float, float, float, float]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_eval_env", args.obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    env = mod.OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed + seed_offset,
    )

    wm.eval()
    actor.eval()

    ep_returns = []
    ep_lengths = []
    ep_success = []
    ep_stuck = []

    for ep in range(episodes):
        obs = np.asarray(env.reset(seed=args.seed + seed_offset + ep), dtype=np.float32)
        h, z = wm.init_state(1, device)

        search_turn_dir = 1
        stuck_count = 0
        steps_since_seen = 0

        ret = 0.0
        stuck_steps = 0
        success = 0.0

        for t in range(args.max_steps):
            if bool(obs[17] > 0.5):
                stuck_steps += 1

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            h, z = wm.posterior(obs_t, h)

            action_idx, steps_since_seen, search_turn_dir, stuck_count = _deterministic_eval_action(
                actor=actor,
                obs=obs,
                h=h,
                z=z,
                ep_step=t,
                steps_since_seen=steps_since_seen,
                search_turn_dir=search_turn_dir,
                stuck_count=stuck_count,
            )

            obs2, rew, done = env.step(ACTIONS[action_idx], render=False)
            obs = np.asarray(obs2, dtype=np.float32)
            ret += float(rew)
            if done:
                if bool(getattr(env, "enable_push", False)):
                    try:
                        if bool(env._box_touches_boundary(env.box_center_x, env.box_center_y)):
                            success = 1.0
                    except Exception:
                        if rew >= 1900.0:
                            success = 1.0
                elif rew >= 1900.0:
                    success = 1.0
            if done:
                ep_lengths.append(t + 1)
                break
        else:
            ep_lengths.append(args.max_steps)

        ep_returns.append(ret)
        ep_success.append(success)
        ep_stuck.append(100.0 * stuck_steps / float(max(1, ep_lengths[-1])))

    wm.train()
    actor.train()
    return (
        float(np.mean(ep_returns)),
        float(np.mean(ep_lengths)),
        float(np.mean(ep_success) * 100.0),
        float(np.mean(ep_stuck)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WM-RSSM-AC: World-Model style recurrent POMDP training for OBELIX"
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
    parser.add_argument("--total_steps", type=int, default=2_500_000)

    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--latent_dim", type=int, default=96)

    parser.add_argument("--warmup_steps", type=int, default=50_000)
    parser.add_argument("--eps_start", type=float, default=0.35)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_frac", type=float, default=0.45)

    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--updates_per_rollout", type=int, default=180)
    parser.add_argument("--imag_horizon", type=int, default=16)

    parser.add_argument("--lr_wm", type=float, default=2e-4)
    parser.add_argument("--lr_actor", type=float, default=7.5e-5)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lambda_ret", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.003)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--imag_reward_min", type=float, default=-6.0)
    parser.add_argument("--imag_reward_max", type=float, default=26.0)
    parser.add_argument("--value_target_min", type=float, default=-120.0)
    parser.add_argument("--value_target_max", type=float, default=420.0)
    parser.add_argument("--max_abs_ac_loss", type=float, default=1e4)

    parser.add_argument("--eval_every_rollouts", type=int, default=25)
    parser.add_argument("--eval_episodes", type=int, default=24)

    parser.add_argument("--save_dir", type=str, default="submission_worldmodel_pomdp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wm = WorldModel(
        obs_dim=18,
        action_dim=len(ACTIONS),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    actor = Actor(feat_dim=args.hidden_dim + args.latent_dim, n_actions=len(ACTIONS)).to(device)
    critic = Critic(feat_dim=args.hidden_dim + args.latent_dim).to(device)

    wm_opt = optim.Adam(wm.parameters(), lr=args.lr_wm)
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr_critic)

    replay = EpisodeReplay(capacity_episodes=12000)

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

    # Per-env trajectory builders.
    ep_obs: List[List[np.ndarray]] = [[obs[i].copy()] for i in range(args.num_envs)]
    ep_actions: List[List[int]] = [[] for _ in range(args.num_envs)]
    ep_rewards: List[List[float]] = [[] for _ in range(args.num_envs)]
    ep_dones: List[List[float]] = [[] for _ in range(args.num_envs)]

    # Per-env policy trackers.
    h_online = torch.zeros(args.num_envs, args.hidden_dim, device=device)
    z_online = torch.zeros(args.num_envs, args.latent_dim, device=device)
    steps_since_seen = np.zeros(args.num_envs, dtype=np.int64)
    search_turn_dir = np.ones(args.num_envs, dtype=np.int64)
    stuck_count = np.zeros(args.num_envs, dtype=np.int64)

    recent_ep_rew = deque(maxlen=400)
    recent_success = deque(maxlen=500)
    recent_wm = deque(maxlen=250)
    recent_actor = deque(maxlen=250)
    recent_critic = deque(maxlen=250)

    os.makedirs(args.save_dir, exist_ok=True)
    best_eval_reward = -1e18
    best_eval_success = -1e9

    global_step = 0
    rollout_id = 0

    print(
        f"Training WM-RSSM-AC | envs={args.num_envs} horizon={args.horizon} total_steps={args.total_steps}"
    )

    try:
        while global_step < args.total_steps:
            rollout_id += 1

            for _ in range(args.horizon):
                # Posterior update from current observations.
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                    h_online, z_online = wm.posterior(obs_t, h_online)
                    feat = torch.cat([h_online, z_online], dim=-1)
                    logits = actor(feat)
                    dist = Categorical(logits=logits)
                    sampled_actions = dist.sample().cpu().numpy().astype(np.int64)

                progress = global_step / float(max(1, args.total_steps))
                eps_mix = min(1.0, progress / max(1e-6, args.eps_decay_frac))
                eps_now = args.eps_start + (args.eps_end - args.eps_start) * eps_mix

                action_idx = sampled_actions.copy()
                for i in range(args.num_envs):
                    # Observation[17] is stuck flag in OBELIX.
                    if bool(obs[i, 17] > 0.5):
                        stuck_count[i] += 1
                        phase = int(stuck_count[i] % 4)
                        if phase == 2:
                            search_turn_dir[i] *= -1
                        if phase in (1, 2):
                            action_idx[i] = 0 if search_turn_dir[i] > 0 else 4
                        else:
                            action_idx[i] = 2
                    else:
                        stuck_count[i] = 0
                        if np.any(obs[i, :17] > 0):
                            steps_since_seen[i] = 0
                        else:
                            steps_since_seen[i] += 1

                        if steps_since_seen[i] > 35:
                            if (global_step + i) % 5 == 0:
                                search_turn_dir[i] *= -1
                            action_idx[i] = (1 if search_turn_dir[i] > 0 else 3) if ((global_step + i) % 5 in (0, 1)) else 2

                        if global_step < args.warmup_steps or random.random() < eps_now:
                            action_idx[i] = random.randint(0, len(ACTIONS) - 1)

                for r, a in zip(remotes, action_idx):
                    r.send(("step", ACTIONS[int(a)]))
                results = [r.recv() for r in remotes]

                for i, (s2_terminal, rew, done, success, reset_obs) in enumerate(results):
                    s2_terminal = np.asarray(s2_terminal, dtype=np.float32)
                    scaled_rew = float(np.clip(rew / 100.0, -5.0, 25.0))

                    ep_actions[i].append(int(action_idx[i]))
                    ep_rewards[i].append(scaled_rew)
                    ep_dones[i].append(1.0 if done else 0.0)
                    ep_obs[i].append(s2_terminal.copy())

                    if done:
                        replay.add_episode(ep_obs[i], ep_actions[i], ep_rewards[i], ep_dones[i])
                        recent_ep_rew.append(float(np.sum(ep_rewards[i]) * 100.0))
                        recent_success.append(1.0 if success else 0.0)

                        next_obs = np.asarray(reset_obs, dtype=np.float32)
                        obs[i] = next_obs

                        ep_obs[i] = [next_obs.copy()]
                        ep_actions[i] = []
                        ep_rewards[i] = []
                        ep_dones[i] = []

                        h_online[i].zero_()
                        z_online[i].zero_()
                        steps_since_seen[i] = 0
                        stuck_count[i] = 0
                        search_turn_dir[i] = 1
                    else:
                        obs[i] = s2_terminal

                global_step += args.num_envs
                if global_step >= args.total_steps:
                    break

            # World-model and imagined-policy updates.
            if replay.can_sample(args.batch_size, args.seq_len):
                for _ in range(args.updates_per_rollout):
                    obs_np, act_np, rew_np, done_np = replay.sample_batch(args.batch_size, args.seq_len)

                    obs_seq = torch.tensor(obs_np, dtype=torch.float32, device=device)
                    actions = torch.tensor(act_np, dtype=torch.long, device=device)
                    rewards = torch.tensor(rew_np, dtype=torch.float32, device=device)
                    dones = torch.tensor(done_np, dtype=torch.float32, device=device)

                    wm_loss, wm_stats = world_model_loss(wm, obs_seq, actions, rewards, dones)

                    wm_opt.zero_grad(set_to_none=True)
                    wm_loss.backward()
                    nn.utils.clip_grad_norm_(wm.parameters(), args.grad_clip)
                    wm_opt.step()

                    actor_loss, critic_loss, ac_stats = imagine_actor_critic_loss(
                        wm=wm,
                        actor=actor,
                        critic=critic,
                        obs_seq=obs_seq,
                        imag_horizon=args.imag_horizon,
                        gamma=args.gamma,
                        lam=args.lambda_ret,
                        ent_coef=args.ent_coef,
                        imag_reward_min=args.imag_reward_min,
                        imag_reward_max=args.imag_reward_max,
                        value_target_min=args.value_target_min,
                        value_target_max=args.value_target_max,
                    )

                    if (
                        (not torch.isfinite(actor_loss))
                        or (not torch.isfinite(critic_loss))
                        or abs(float(actor_loss.item())) > args.max_abs_ac_loss
                        or abs(float(critic_loss.item())) > args.max_abs_ac_loss
                    ):
                        continue

                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    actor_opt.step()

                    critic_opt.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                    critic_opt.step()

                    recent_wm.append(wm_stats["wm_total"])
                    recent_actor.append(ac_stats["actor_loss"])
                    recent_critic.append(ac_stats["critic_loss"])

            if rollout_id % 5 == 0:
                avg_ep = float(np.mean(recent_ep_rew)) if recent_ep_rew else 0.0
                avg_sr = float(np.mean(recent_success) * 100.0) if recent_success else 0.0
                avg_wm = float(np.mean(recent_wm)) if recent_wm else 0.0
                avg_actor = float(np.mean(recent_actor)) if recent_actor else 0.0
                avg_critic = float(np.mean(recent_critic)) if recent_critic else 0.0
                print(
                    f"Step {global_step}/{args.total_steps} | AvgEpRew {avg_ep:.1f} | "
                    f"Success {avg_sr:.1f}% | WM {avg_wm:.4f} | Actor {avg_actor:.4f} | Critic {avg_critic:.4f}"
                )

            if args.eval_every_rollouts > 0 and (rollout_id % args.eval_every_rollouts == 0):
                eval_rew, eval_len, eval_sr, eval_stuck = evaluate_policy(
                    wm=wm,
                    actor=actor,
                    args=args,
                    device=device,
                    episodes=args.eval_episodes,
                    seed_offset=300000 + rollout_id,
                )

                # Single selection objective: maximize evaluation reward; success is tie-breaker.
                better = (eval_rew > best_eval_reward) or (
                    abs(eval_rew - best_eval_reward) <= 1e-9 and eval_sr > best_eval_success
                )
                if better:
                    best_eval_reward = eval_rew
                    best_eval_success = eval_sr
                    best_path = os.path.join(args.save_dir, "weights_eval_best.pth")
                    torch.save(
                        {
                            "world_model_state_dict": wm.state_dict(),
                            "actor_state_dict": actor.state_dict(),
                            "critic_state_dict": critic.state_dict(),
                            "obs_dim": wm.obs_dim,
                            "action_dim": wm.action_dim,
                            "hidden_dim": wm.hidden_dim,
                            "latent_dim": wm.latent_dim,
                            "algorithm": "WM-RSSM-AC",
                            "eval_success_rate": best_eval_success,
                            "eval_mean_reward": best_eval_reward,
                            "eval_stuck_rate": eval_stuck,
                        },
                        best_path,
                    )

                print(
                    f"[Eval] Rollout {rollout_id} | MeanRew {eval_rew:.1f} | "
                    f"MeanLen {eval_len:.1f} | Success {eval_sr:.1f}% | Stuck {eval_stuck:.1f}%"
                )

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoints and shutting down cleanly...")

    finally:
        weights_path = os.path.join(args.save_dir, "weights.pth")
        best_path = os.path.join(args.save_dir, "weights_eval_best.pth")

        if os.path.exists(best_path):
            payload = torch.load(best_path, map_location="cpu")
            torch.save(payload, weights_path)
            print(f"Saved eval-best model weights to {weights_path}")
        else:
            torch.save(
                {
                    "world_model_state_dict": wm.state_dict(),
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "obs_dim": wm.obs_dim,
                    "action_dim": wm.action_dim,
                    "hidden_dim": wm.hidden_dim,
                    "latent_dim": wm.latent_dim,
                    "algorithm": "WM-RSSM-AC",
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
    main()
