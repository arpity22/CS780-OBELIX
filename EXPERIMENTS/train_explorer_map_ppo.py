import argparse
import os
import random
import signal
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
ACTION_YAW_DEG = np.array([45.0, 22.5, 0.0, -22.5, -45.0], dtype=np.float32)

PHASE_FIND = 0
PHASE_PUSH = 1
PHASE_UNWEDGE = 2
N_PHASES = 3

MAP_CHANNELS = 4  # visit, reward_ema, stuck_ema, success_ema


def load_obelix_class(obelix_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_eval_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def is_success_transition(env, reward: float, done: bool) -> bool:
    if not done:
        return False
    if bool(getattr(env, "enable_push", False)):
        try:
            if bool(env._box_touches_boundary(env.box_center_x, env.box_center_y)):
                return True
        except Exception:
            pass
    return reward >= 1900.0


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

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                s2, r, d = env.step(data, render=False)
                is_success = bool(is_success_transition(env, float(r), bool(d)))
                if d:
                    s2 = env.reset()
                remote.send((s2, r, d, is_success))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                remote.close()
                break
        except EOFError:
            break
        except KeyboardInterrupt:
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
    def __init__(self, in_dim: int, n_actions: int = 5, hidden_dim: int = 192):
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

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(self, obs_t: torch.Tensor, hidden: torch.Tensor):
        z = self.encoder(obs_t)
        z, hidden = self.gru(z.unsqueeze(0), hidden)
        z = z.squeeze(0)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value, hidden

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        starts_seq: torch.Tensor,
        actions_seq: torch.Tensor,
    ):
        t_steps, batch_size = obs_seq.shape[0], obs_seq.shape[1]
        hidden = self.initial_hidden(batch_size, obs_seq.device)

        logits_list = []
        values_list = []

        for t in range(t_steps):
            reset_mask = (1.0 - starts_seq[t]).view(1, batch_size, 1)
            hidden = hidden * reset_mask
            logits_t, values_t, hidden = self.step(obs_seq[t], hidden)
            logits_list.append(logits_t)
            values_list.append(values_t)

        logits = torch.stack(logits_list, dim=0)
        values = torch.stack(values_list, dim=0)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_seq)
        entropy = dist.entropy()
        return log_probs, entropy, values


@dataclass
class TrackerState:
    last_action: np.ndarray
    prev_obs: np.ndarray
    steps_since_seen: np.ndarray
    ep_steps: np.ndarray
    ep_starts: np.ndarray
    search_turn_dir: np.ndarray
    stuck_recovery_count: np.ndarray
    heading_deg: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def linear_decay(start: float, end: float, progress: float, decay_frac: float) -> float:
    if decay_frac <= 0:
        return end
    mix = min(1.0, max(0.0, progress / decay_frac))
    return start + (end - start) * mix


def one_hot_phase(phase_idx: np.ndarray) -> np.ndarray:
    out = np.zeros((phase_idx.shape[0], N_PHASES), dtype=np.float32)
    out[np.arange(phase_idx.shape[0]), phase_idx] = 1.0
    return out


def augment_obs(
    obs: np.ndarray,
    prev_obs: np.ndarray,
    last_action: np.ndarray,
    steps_since_seen: np.ndarray,
    ep_steps: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    delta = obs - prev_obs
    one_hot = np.zeros((obs.shape[0], 5), dtype=np.float32)
    one_hot[np.arange(obs.shape[0]), last_action] = 1.0
    seen = np.minimum(1.0, steps_since_seen.astype(np.float32) / 120.0).reshape(-1, 1)
    progress = np.minimum(1.0, ep_steps.astype(np.float32) / float(max_steps)).reshape(-1, 1)
    return np.concatenate([obs, delta, one_hot, seen, progress], axis=-1).astype(np.float32)


def heading_to_bin(heading_deg: float, bins: int) -> int:
    return int(((heading_deg % 360.0) / 360.0) * bins) % bins


def map_row_for_env(map_state: np.ndarray, env_idx: int, heading_deg: float, bins: int) -> np.ndarray:
    return map_state[env_idx, heading_to_bin(heading_deg, bins)]


def map_features_for_env(map_state: np.ndarray, env_idx: int, heading_deg: float, bins: int) -> np.ndarray:
    row = map_row_for_env(map_state, env_idx, heading_deg, bins)  # [5, 4]
    visits = row[:, 0]
    rew = row[:, 1]
    stuck = row[:, 2]
    succ = row[:, 3]

    vnorm = visits / (1.0 + np.max(visits))
    rew_t = np.tanh(rew / 5.0)
    stuck_c = np.clip(stuck, 0.0, 1.0)
    succ_c = np.clip(succ, 0.0, 1.0)

    all_visits = map_state[env_idx, :, :, 0]
    explored_ratio = float(np.mean(all_visits > 0.0))
    heading_bin = heading_to_bin(heading_deg, bins)
    heading_visits = float(np.sum(map_state[env_idx, heading_bin, :, 0]))
    total_visits = float(np.sum(all_visits)) + 1e-6
    heading_visit_ratio = heading_visits / total_visits
    heading_risk = float(np.mean(stuck))
    heading_rew = float(np.mean(rew_t))
    fw_novelty = 1.0 / np.sqrt(1.0 + float(visits[2]))

    extras = np.array(
        [explored_ratio, heading_visit_ratio, heading_risk, heading_rew, fw_novelty],
        dtype=np.float32,
    )
    return np.concatenate([vnorm, rew_t, stuck_c, succ_c, extras], axis=0).astype(np.float32)


def determine_phase(obs: np.ndarray, steps_since_seen: int, map_row: np.ndarray) -> int:
    stuck = bool(obs[17] > 0.5)
    sees_target = bool(np.any(obs[:17] > 0))
    front_active = bool(np.any(obs[4:12] > 0))
    fw_risk = float(map_row[2, 2])

    if stuck or fw_risk > 0.85:
        return PHASE_UNWEDGE
    if front_active and sees_target:
        return PHASE_PUSH
    if (not sees_target) or steps_since_seen > 25:
        return PHASE_FIND
    return PHASE_PUSH


def deterministic_phase_action(
    logits: np.ndarray,
    obs: np.ndarray,
    phase: int,
    steps_since_seen: int,
    search_turn_dir: int,
    stuck_recovery_count: int,
    map_row: np.ndarray,
) -> Tuple[int, int, int]:
    if phase == PHASE_UNWEDGE or bool(obs[17] > 0.5):
        stuck_recovery_count += 1
        cycle = stuck_recovery_count % 4
        if cycle == 2:
            search_turn_dir *= -1
        if cycle in (1, 2):
            return (0 if search_turn_dir > 0 else 4), search_turn_dir, stuck_recovery_count
        return 2, search_turn_dir, stuck_recovery_count

    stuck_recovery_count = 0

    sees_target = bool(np.any(obs[:17] > 0))
    front_active = bool(np.any(obs[4:12] > 0))

    if phase == PHASE_FIND:
        if steps_since_seen > 35:
            left_score = float(np.mean(map_row[[0, 1], 0] + 2.5 * map_row[[0, 1], 2]))
            right_score = float(np.mean(map_row[[3, 4], 0] + 2.5 * map_row[[3, 4], 2]))
            if left_score + 0.02 < right_score:
                search_turn_dir = 1
            elif right_score + 0.02 < left_score:
                search_turn_dir = -1

            if (steps_since_seen % 5) in (0, 1):
                return (1 if search_turn_dir > 0 else 3), search_turn_dir, stuck_recovery_count
            return 2, search_turn_dir, stuck_recovery_count

        if (not sees_target) and float(map_row[2, 2]) > 0.55:
            return (1 if search_turn_dir > 0 else 3), search_turn_dir, stuck_recovery_count
        return int(np.argmax(logits)), search_turn_dir, stuck_recovery_count

    # Push phase.
    if front_active and (not bool(obs[17] > 0.5)):
        return 2, search_turn_dir, stuck_recovery_count
    return int(np.argmax(logits)), search_turn_dir, stuck_recovery_count


def training_action_with_phase(
    sampled_action: int,
    logits: np.ndarray,
    obs: np.ndarray,
    phase: int,
    steps_since_seen: int,
    search_turn_dir: int,
    stuck_recovery_count: int,
    map_row: np.ndarray,
    override_prob: float,
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    det_action, new_turn, new_stuck = deterministic_phase_action(
        logits=logits,
        obs=obs,
        phase=phase,
        steps_since_seen=steps_since_seen,
        search_turn_dir=search_turn_dir,
        stuck_recovery_count=stuck_recovery_count,
        map_row=map_row,
    )

    if phase == PHASE_UNWEDGE:
        return det_action, new_turn, new_stuck

    if phase == PHASE_PUSH and bool(np.any(obs[4:12] > 0)):
        return det_action, new_turn, new_stuck

    if phase == PHASE_FIND and float(rng.random()) < override_prob:
        return det_action, new_turn, new_stuck

    return int(sampled_action), new_turn, new_stuck


def update_map_and_intrinsic(
    map_state: np.ndarray,
    sensor_seen: np.ndarray,
    env_idx: int,
    heading_deg: float,
    action_idx: int,
    scaled_rew: float,
    obs_after: np.ndarray,
    success: bool,
    args,
) -> float:
    hbin = heading_to_bin(heading_deg, args.map_angle_bins)
    cell = map_state[env_idx, hbin, action_idx]

    visit_before = float(cell[0])
    alpha = float(args.map_ema_alpha)

    cell[0] = min(float(args.map_visit_cap), visit_before + 1.0)
    cell[1] = (1.0 - alpha) * float(cell[1]) + alpha * float(scaled_rew)

    stuck = 1.0 if bool(obs_after[17] > 0.5) else 0.0
    cell[2] = (1.0 - alpha) * float(cell[2]) + alpha * stuck
    cell[3] = (1.0 - alpha) * float(cell[3]) + alpha * (1.0 if success else 0.0)

    sensor_now = obs_after[:17] > 0.5
    new_bits = np.logical_and(sensor_now, np.logical_not(sensor_seen[env_idx]))
    sensor_seen[env_idx] = np.logical_or(sensor_seen[env_idx], sensor_now)

    novelty = 1.0 / np.sqrt(1.0 + visit_before)
    heading_visits = float(np.sum(map_state[env_idx, hbin, :, 0]))
    total_visits = float(np.sum(map_state[env_idx, :, :, 0])) + 1e-6
    frontier = 1.0 - heading_visits / total_visits

    intrinsic = 0.0
    intrinsic += float(args.map_novelty_coef) * novelty
    intrinsic += float(args.map_frontier_coef) * frontier
    intrinsic += float(args.map_sensor_coef) * float(np.sum(new_bits))
    intrinsic -= float(args.map_stuck_coef) * stuck
    intrinsic -= 0.20 * float(cell[2])

    return float(np.clip(intrinsic, -2.0, 2.0))


@torch.no_grad()
def run_aligned_eval(
    actor_critic: RecurrentActorCritic,
    args,
    device: torch.device,
    rollout_id: int,
) -> Tuple[float, float, float, float]:
    obelix_cls = load_obelix_class(args.obelix_py)
    env = obelix_cls(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed + 140000 + rollout_id,
    )

    actor_critic.eval()

    ep_rewards = []
    ep_lengths = []
    ep_success = []
    ep_stuck_rates = []

    for ep in range(args.eval_episodes):
        obs = np.asarray(env.reset(seed=args.seed + rollout_id * 631 + ep), dtype=np.float32)

        hidden = actor_critic.initial_hidden(1, device)
        prev_obs = obs.copy()
        last_action = np.array([2], dtype=np.int64)
        steps_since_seen = np.array([0], dtype=np.int64)
        ep_step_vec = np.array([0], dtype=np.int64)
        search_turn_dir = np.array([1], dtype=np.int64)
        stuck_recovery_count = np.array([0], dtype=np.int64)
        heading_deg = np.array([0.0], dtype=np.float32)

        map_state = np.zeros((1, args.map_angle_bins, len(ACTIONS), MAP_CHANNELS), dtype=np.float32)
        sensor_seen = np.zeros((1, 17), dtype=bool)

        total_reward = 0.0
        success = 0.0
        stuck_steps = 0

        for _ in range(args.max_steps):
            if bool(obs[17] > 0.5):
                stuck_steps += 1
            if np.any(obs[:17] > 0):
                steps_since_seen[0] = 0
            else:
                steps_since_seen[0] += 1

            map_row = map_row_for_env(map_state, 0, float(heading_deg[0]), args.map_angle_bins)
            phase = determine_phase(obs, int(steps_since_seen[0]), map_row)

            aug_obs = augment_obs(
                obs.reshape(1, -1),
                prev_obs.reshape(1, -1),
                last_action,
                steps_since_seen,
                ep_step_vec,
                args.max_steps,
            )
            map_feat = map_features_for_env(map_state, 0, float(heading_deg[0]), args.map_angle_bins).reshape(1, -1)
            phase_feat = one_hot_phase(np.array([phase], dtype=np.int64))
            model_in = np.concatenate([aug_obs, map_feat, phase_feat], axis=-1)

            obs_t = torch.tensor(model_in, dtype=torch.float32, device=device).view(1, -1)
            logits, _, hidden = actor_critic.step(obs_t, hidden)
            logits_np = logits[0].detach().cpu().numpy().astype(np.float64)

            action_idx, search_turn_dir[0], stuck_recovery_count[0] = deterministic_phase_action(
                logits=logits_np,
                obs=obs,
                phase=phase,
                steps_since_seen=int(steps_since_seen[0]),
                search_turn_dir=int(search_turn_dir[0]),
                stuck_recovery_count=int(stuck_recovery_count[0]),
                map_row=map_row,
            )

            heading_deg[0] = (heading_deg[0] + ACTION_YAW_DEG[action_idx]) % 360.0

            s2, rew, done = env.step(ACTIONS[action_idx], render=False)
            s2 = np.asarray(s2, dtype=np.float32)

            total_reward += float(rew)
            success_flag = bool(is_success_transition(env, float(rew), bool(done)))
            if success_flag:
                success = 1.0

            scaled_rew = float(np.clip(rew / 100.0, -5.0, 25.0))
            if bool(s2[17] > 0.5):
                scaled_rew -= float(args.stuck_extra_penalty)
            if action_idx == 2 and np.any(s2[4:12] > 0) and (not bool(s2[17])):
                scaled_rew += float(args.fw_tracking_bonus)

            update_map_and_intrinsic(
                map_state=map_state,
                sensor_seen=sensor_seen,
                env_idx=0,
                heading_deg=float(heading_deg[0]),
                action_idx=int(action_idx),
                scaled_rew=scaled_rew,
                obs_after=s2,
                success=success_flag,
                args=args,
            )

            ep_step_vec[0] += 1
            last_action[0] = action_idx
            prev_obs = obs.copy()
            obs = s2

            if done:
                break

        ep_rewards.append(total_reward)
        ep_lengths.append(int(ep_step_vec[0]))
        ep_success.append(success)
        ep_stuck_rates.append(100.0 * stuck_steps / float(max(1, int(ep_step_vec[0]))))

    actor_critic.train()
    return (
        float(np.mean(ep_rewards)),
        float(np.mean(ep_lengths)),
        float(np.mean(ep_success) * 100.0),
        float(np.mean(ep_stuck_rates)),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXPLORER-MAP-PPO: Phase-based recurrent PPO with intrinsic self-exploration map"
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
    parser.add_argument("--total_steps", type=int, default=2_000_000)

    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.75)
    parser.add_argument("--ent_coef", type=float, default=0.0015)
    parser.add_argument("--max_grad_norm", type=float, default=0.8)

    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--envs_per_batch", type=int, default=4)

    parser.add_argument("--intrinsic_coef", type=float, default=0.45)
    parser.add_argument("--intrinsic_coef_final", type=float, default=0.10)
    parser.add_argument("--intrinsic_decay_frac", type=float, default=0.60)

    parser.add_argument("--stuck_extra_penalty", type=float, default=2.0)
    parser.add_argument("--fw_tracking_bonus", type=float, default=0.04)

    parser.add_argument("--map_angle_bins", type=int, default=16)
    parser.add_argument("--map_ema_alpha", type=float, default=0.08)
    parser.add_argument("--map_visit_cap", type=float, default=128.0)
    parser.add_argument("--map_novelty_coef", type=float, default=0.35)
    parser.add_argument("--map_frontier_coef", type=float, default=0.20)
    parser.add_argument("--map_sensor_coef", type=float, default=0.02)
    parser.add_argument("--map_stuck_coef", type=float, default=0.90)
    parser.add_argument("--phase_override_prob", type=float, default=0.80)

    parser.add_argument("--eval_every_rollouts", type=int, default=25)
    parser.add_argument("--eval_episodes", type=int, default=24)

    parser.add_argument("--save_dir", type=str, default="submission_explorer_map_ppo1")
    return parser.parse_args()


def train():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_obs_dim = 43  # obs + delta + last_action_one_hot + seen + progress
    map_feat_dim = (len(ACTIONS) * MAP_CHANNELS) + 5
    in_dim = base_obs_dim + map_feat_dim + N_PHASES

    actor_critic = RecurrentActorCritic(in_dim=in_dim, n_actions=5, hidden_dim=192).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr)

    rms_intrinsic = RunningMeanStd()
    rng = np.random.default_rng(args.seed + 123)

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
        search_turn_dir=np.ones(args.num_envs, dtype=np.int64),
        stuck_recovery_count=np.zeros(args.num_envs, dtype=np.int64),
        heading_deg=np.zeros(args.num_envs, dtype=np.float32),
    )

    map_state = np.zeros((args.num_envs, args.map_angle_bins, len(ACTIONS), MAP_CHANNELS), dtype=np.float32)
    sensor_seen = np.zeros((args.num_envs, 17), dtype=bool)

    hidden = actor_critic.initial_hidden(args.num_envs, device)

    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)

    reward_window = deque(maxlen=300)
    length_window = deque(maxlen=300)
    success_window = deque(maxlen=500)

    best_train_score = -1e9
    best_eval_reward = -1e18
    best_eval_success = -1e9

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    rollout_id = 0

    print(
        f"Training EXPLORER-MAP-PPO | envs={args.num_envs} horizon={args.horizon} total_steps={args.total_steps}"
    )

    try:
        while global_step < args.total_steps:
            rollout_id += 1
            progress = global_step / float(max(1, args.total_steps))
            intrinsic_coef_now = linear_decay(
                args.intrinsic_coef,
                args.intrinsic_coef_final,
                progress,
                args.intrinsic_decay_frac,
            )

            obs_buf = np.zeros((args.horizon, args.num_envs, in_dim), dtype=np.float32)
            starts_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
            actions_buf = np.zeros((args.horizon, args.num_envs), dtype=np.int64)
            logp_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
            values_buf = np.zeros((args.horizon, args.num_envs), dtype=np.float32)
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

                phase_idx = np.zeros(args.num_envs, dtype=np.int64)
                map_feats = np.zeros((args.num_envs, (len(ACTIONS) * MAP_CHANNELS) + 5), dtype=np.float32)
                map_rows = np.zeros((args.num_envs, len(ACTIONS), MAP_CHANNELS), dtype=np.float32)

                for i in range(args.num_envs):
                    row = map_row_for_env(map_state, i, float(tracker.heading_deg[i]), args.map_angle_bins)
                    map_rows[i] = row
                    phase_idx[i] = determine_phase(obs[i], int(tracker.steps_since_seen[i]), row)
                    map_feats[i] = map_features_for_env(map_state, i, float(tracker.heading_deg[i]), args.map_angle_bins)

                phase_feats = one_hot_phase(phase_idx)
                model_in = np.concatenate([aug_obs, map_feats, phase_feats], axis=-1).astype(np.float32)

                obs_t = torch.tensor(model_in, dtype=torch.float32, device=device)
                starts_t = torch.tensor(tracker.ep_starts, dtype=torch.float32, device=device)

                with torch.no_grad():
                    reset_mask = (1.0 - starts_t).view(1, args.num_envs, 1)
                    hidden = hidden * reset_mask

                    logits, values, hidden = actor_critic.step(obs_t, hidden)
                    dist = Categorical(logits=logits)
                    sampled_actions = dist.sample()
                    log_probs = dist.log_prob(sampled_actions)

                logits_np = logits.detach().cpu().numpy().astype(np.float64)
                sampled_action_np = sampled_actions.cpu().numpy().astype(np.int64)
                action_idx = sampled_action_np.copy()

                for i in range(args.num_envs):
                    action_idx[i], tracker.search_turn_dir[i], tracker.stuck_recovery_count[i] = training_action_with_phase(
                        sampled_action=int(sampled_action_np[i]),
                        logits=logits_np[i],
                        obs=obs[i],
                        phase=int(phase_idx[i]),
                        steps_since_seen=int(tracker.steps_since_seen[i]),
                        search_turn_dir=int(tracker.search_turn_dir[i]),
                        stuck_recovery_count=int(tracker.stuck_recovery_count[i]),
                        map_row=map_rows[i],
                        override_prob=float(args.phase_override_prob),
                        rng=rng,
                    )
                    tracker.heading_deg[i] = (tracker.heading_deg[i] + ACTION_YAW_DEG[action_idx[i]]) % 360.0

                obs_buf[t] = model_in
                starts_buf[t] = tracker.ep_starts
                actions_buf[t] = action_idx
                logp_buf[t] = log_probs.cpu().numpy()
                values_buf[t] = values.cpu().numpy()

                for r, a in zip(remotes, action_idx):
                    r.send(("step", ACTIONS[int(a)]))
                results = [r.recv() for r in remotes]

                next_obs = np.zeros_like(obs)
                done_mask = np.zeros(args.num_envs, dtype=np.float32)

                for i, (s2, rew, done, success) in enumerate(results):
                    s2 = np.asarray(s2, dtype=np.float32)
                    next_obs[i] = s2

                    scaled_rew = float(np.clip(rew / 100.0, -5.0, 25.0))
                    if bool(s2[17]):
                        scaled_rew -= float(args.stuck_extra_penalty)
                    if action_idx[i] == 2 and np.any(s2[4:12] > 0) and (not bool(s2[17])):
                        scaled_rew += float(args.fw_tracking_bonus)

                    rewards_ext_buf[t, i] = scaled_rew

                    intrinsic_raw = update_map_and_intrinsic(
                        map_state=map_state,
                        sensor_seen=sensor_seen,
                        env_idx=i,
                        heading_deg=float(tracker.heading_deg[i]),
                        action_idx=int(action_idx[i]),
                        scaled_rew=scaled_rew,
                        obs_after=s2,
                        success=bool(success),
                        args=args,
                    )
                    rewards_int_raw_buf[t, i] = intrinsic_raw

                    done_mask[i] = 1.0 if done else 0.0

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
                        tracker.search_turn_dir[i] = 1
                        tracker.stuck_recovery_count[i] = 0
                        tracker.heading_deg[i] = 0.0
                        hidden[:, i : i + 1, :] = 0.0

                        map_state[i] = 0.0
                        sensor_seen[i] = False
                    else:
                        tracker.last_action[i] = int(action_idx[i])
                        tracker.prev_obs[i] = obs[i]

                dones_buf[t] = done_mask
                tracker.ep_starts = done_mask.copy()
                obs = next_obs

            # Bootstrap values.
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

            phase_idx_last = np.zeros(args.num_envs, dtype=np.int64)
            map_feats_last = np.zeros((args.num_envs, (len(ACTIONS) * MAP_CHANNELS) + 5), dtype=np.float32)
            for i in range(args.num_envs):
                row = map_row_for_env(map_state, i, float(tracker.heading_deg[i]), args.map_angle_bins)
                phase_idx_last[i] = determine_phase(obs[i], int(tracker.steps_since_seen[i]), row)
                map_feats_last[i] = map_features_for_env(map_state, i, float(tracker.heading_deg[i]), args.map_angle_bins)
            phase_feats_last = one_hot_phase(phase_idx_last)

            model_in_last = np.concatenate([aug_obs_last, map_feats_last, phase_feats_last], axis=-1).astype(np.float32)

            with torch.no_grad():
                obs_last_t = torch.tensor(model_in_last, dtype=torch.float32, device=device)
                starts_last_t = torch.tensor(tracker.ep_starts, dtype=torch.float32, device=device)
                reset_mask = (1.0 - starts_last_t).view(1, args.num_envs, 1)
                hidden_boot = hidden * reset_mask
                _, last_values_t, _ = actor_critic.step(obs_last_t, hidden_boot)
                last_values = last_values_t.cpu().numpy()

            rms_intrinsic.update(rewards_int_raw_buf.reshape(-1))
            intr_scale = 1.0 / np.sqrt(rms_intrinsic.var + 1e-8)
            rewards_int = np.clip(rewards_int_raw_buf * intr_scale, -2.0, 2.0)

            rewards_total = rewards_ext_buf + intrinsic_coef_now * rewards_int

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

            n_envs = args.num_envs
            for _ in range(args.ppo_epochs):
                env_perm = np.random.permutation(n_envs)
                for start in range(0, n_envs, args.envs_per_batch):
                    mb_env = env_perm[start : start + args.envs_per_batch]
                    if mb_env.size == 0:
                        continue

                    logp_new, entropy, values_new = actor_critic.evaluate_sequence(
                        obs_t[:, mb_env],
                        starts_t[:, mb_env],
                        actions_t[:, mb_env],
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

                    total_loss = (
                        policy_loss
                        + args.vf_coef * value_loss
                        - args.ent_coef * entropy_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

            global_step += args.horizon * args.num_envs

            if rollout_id % 5 == 0:
                avg_rew = float(np.mean(reward_window)) if reward_window else 0.0
                avg_len = float(np.mean(length_window)) if length_window else 0.0
                sr = float(np.mean(success_window) * 100.0) if success_window else 0.0
                avg_ext = float(np.mean(rewards_ext_buf))
                avg_int = float(np.mean(rewards_int))

                train_score = avg_rew + 5.0 * sr
                if train_score > best_train_score:
                    best_train_score = train_score
                    best_path = os.path.join(args.save_dir, "weights_train_best.pth")
                    torch.save(
                        {
                            "model_state_dict": actor_critic.state_dict(),
                            "obs_dim": in_dim,
                            "hidden_dim": actor_critic.hidden_dim,
                            "map_angle_bins": args.map_angle_bins,
                            "map_channels": MAP_CHANNELS,
                            "algorithm": "EXPLORER-MAP-PPO",
                            "train_score": best_train_score,
                        },
                        best_path,
                    )

                print(
                    f"Step {global_step}/{args.total_steps} | AvgEpRew {avg_rew:.1f} | "
                    f"AvgEpLen {avg_len:.1f} | Success {sr:.1f}% | Ext {avg_ext:.3f} | "
                    f"IntMap {avg_int:.3f} | IntrCoef {intrinsic_coef_now:.3f}"
                )

            if args.eval_every_rollouts > 0 and (rollout_id % args.eval_every_rollouts == 0):
                eval_rew, eval_len, eval_sr, eval_stuck = run_aligned_eval(
                    actor_critic=actor_critic,
                    args=args,
                    device=device,
                    rollout_id=rollout_id,
                )

                better_eval = (eval_rew > best_eval_reward) or (
                    abs(eval_rew - best_eval_reward) <= 1e-9 and eval_sr > best_eval_success
                )
                if better_eval:
                    best_eval_reward = eval_rew
                    best_eval_success = eval_sr
                    eval_path = os.path.join(args.save_dir, "weights_eval_best.pth")
                    torch.save(
                        {
                            "model_state_dict": actor_critic.state_dict(),
                            "obs_dim": in_dim,
                            "hidden_dim": actor_critic.hidden_dim,
                            "map_angle_bins": args.map_angle_bins,
                            "map_channels": MAP_CHANNELS,
                            "algorithm": "EXPLORER-MAP-PPO",
                            "eval_success_rate": best_eval_success,
                            "eval_mean_reward": best_eval_reward,
                            "eval_stuck_rate": eval_stuck,
                            "map_params": {
                                "map_ema_alpha": args.map_ema_alpha,
                                "map_visit_cap": args.map_visit_cap,
                                "map_novelty_coef": args.map_novelty_coef,
                                "map_frontier_coef": args.map_frontier_coef,
                                "map_sensor_coef": args.map_sensor_coef,
                                "map_stuck_coef": args.map_stuck_coef,
                            },
                        },
                        eval_path,
                    )

                print(
                    f"[Eval] Rollout {rollout_id} | MeanRew {eval_rew:.1f} | "
                    f"MeanLen {eval_len:.1f} | Success {eval_sr:.1f}% | Stuck {eval_stuck:.1f}%"
                )

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoints and shutting down cleanly...")

    finally:
        weights_path = os.path.join(args.save_dir, "weights.pth")
        eval_path = os.path.join(args.save_dir, "weights_eval_best.pth")
        train_path = os.path.join(args.save_dir, "weights_train_best.pth")

        if os.path.exists(eval_path):
            eval_payload = torch.load(eval_path, map_location="cpu")
            torch.save(eval_payload, weights_path)
            print(f"Saved eval-best model weights to {weights_path}")
        elif os.path.exists(train_path):
            train_payload = torch.load(train_path, map_location="cpu")
            torch.save(train_payload, weights_path)
            print(f"Saved train-best model weights to {weights_path}")
        else:
            torch.save(
                {
                    "model_state_dict": actor_critic.state_dict(),
                    "obs_dim": in_dim,
                    "hidden_dim": actor_critic.hidden_dim,
                    "map_angle_bins": args.map_angle_bins,
                    "map_channels": MAP_CHANNELS,
                    "algorithm": "EXPLORER-MAP-PPO",
                    "map_params": {
                        "map_ema_alpha": args.map_ema_alpha,
                        "map_visit_cap": args.map_visit_cap,
                        "map_novelty_coef": args.map_novelty_coef,
                        "map_frontier_coef": args.map_frontier_coef,
                        "map_sensor_coef": args.map_sensor_coef,
                        "map_stuck_coef": args.map_stuck_coef,
                    },
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
