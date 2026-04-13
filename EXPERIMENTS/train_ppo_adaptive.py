#!/usr/bin/env python3
"""
ADAPTIVE PPO with Anti-Forgetting Mechanisms

FIXES FOR YOUR TRAINING FAILURES:
1. Adaptive entropy (prevents collapse)
2. Intrinsic curiosity (keeps exploring)
3. Progressive rewards (learns step-by-step)
4. Success memory (remembers good episodes)
5. Auto LR decay (when stuck)
"""

import os
import sys
import time
import random
import argparse
import importlib.util
from pathlib import Path
from collections import deque
from multiprocessing import Pool
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

_QT_WARNING_PREFIXES = (
    "QFontDatabase:",
    "Note that Qt no longer ships fonts.",
)
_QT_WARNING_PREFIXES_BYTES = tuple(p.encode("utf-8") for p in _QT_WARNING_PREFIXES)


class _QtStderrFilter:
    def __init__(self, stream):
        self._stream = stream
        self._buffer = ""

    def write(self, data):
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if any(line.startswith(prefix) for prefix in _QT_WARNING_PREFIXES):
                continue
            self._stream.write(line + "\n")

    def flush(self):
        if self._buffer:
            if not any(self._buffer.startswith(prefix) for prefix in _QT_WARNING_PREFIXES):
                self._stream.write(self._buffer)
            self._buffer = ""
        self._stream.flush()


def _start_fd2_filter():
    original_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _reader():
        with os.fdopen(read_fd, "rb", buffering=0) as reader, os.fdopen(original_fd, "wb", buffering=0) as writer:
            buffer = b""
            while True:
                chunk = reader.read(1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if any(line.startswith(prefix) for prefix in _QT_WARNING_PREFIXES_BYTES):
                        continue
                    writer.write(line + b"\n")
            if buffer and not any(buffer.startswith(prefix) for prefix in _QT_WARNING_PREFIXES_BYTES):
                writer.write(buffer)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()


def _configure_worker_logging(render_enabled: bool, worker_id: int):
    if render_enabled and worker_id == 0:
        _start_fd2_filter()
    else:
        devnull = open(os.devnull, "w")
        os.dup2(devnull.fileno(), 2)

# ============================================================================
# ADAPTIVE WORKER WITH CURIOSITY
# ============================================================================

def parallel_rollout_worker(args_tuple):
    """Worker with curiosity and progress tracking."""
    (obelix_path, env_config, policy_state, num_steps, seed, 
     config, hidden_dim, worker_id, render_enabled) = args_tuple

    _configure_worker_logging(render_enabled, worker_id)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    from collections import deque
    
    class ActorCritic(nn.Module):
        def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, n_actions),
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
        
        def forward(self, x):
            features = self.feature(x)
            logits = self.actor(features)
            value = self.critic(features)
            return logits, value
    
    policy = ActorCritic(hidden_dim=hidden_dim)
    policy.load_state_dict(policy_state)
    policy.eval()
    
    env = OBELIX(**env_config)
    state = env.reset(seed=seed)
    
    states, actions, log_probs, rewards, dones, values, timeouts = [], [], [], [], [], [], []
    episode_returns, episode_lengths, episode_successes = [], [], []
    
    episode_reward = 0
    episode_length = 0
    
    # CURIOSITY: Track visited states
    visited_state_hashes = set()
    
    # PROGRESS: Track best sensor count seen
    best_sensor_count = 0
    steps_since_improvement = 0
    
    # MEMORY: Track recent actions to detect loops
    action_memory = deque(maxlen=8)
    
    for step in range(num_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = policy(state_t)
            probs = F.softmax(logits, dim=-1)
            
            # ANTI-LOOP: Reduce probability of repeated actions
            if len(action_memory) >= 4:
                recent_actions = list(action_memory)[-4:]
                if len(set(recent_actions)) == 1:  # All same
                    repeated_action = recent_actions[0]
                    probs[0, repeated_action] *= 0.3
                    probs = probs / probs.sum()
            
            dist = Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        
        action_idx = action_idx.item()
        log_prob = log_prob.item()
        value = value.item()
        
        next_state, reward, done = env.step(ACTIONS[action_idx], render=(render_enabled and worker_id == 0))
        is_timeout = (episode_length + 1 >= env_config['max_steps']) and done
        
        # REWARD SHAPING
        base_scaled = reward / 1000.0
        shaped_reward = base_scaled
        
        # 1. MASSIVE SUCCESS BONUS
        if done and reward >= config['success_reward_threshold']:
            shaped_reward += config['success_bonus'] / 100.0
        
        # 2. CURIOSITY BONUS (exploring new states)
        state_hash = hash(tuple(next_state))
        if state_hash not in visited_state_hashes:
            shaped_reward += config['curiosity_bonus'] / 1000.0
            visited_state_hashes.add(state_hash)
        
        # 3. PROGRESS BONUS (getting closer to box)
        curr_sensor_count = np.sum(next_state[:16] == 1)
        if curr_sensor_count > best_sensor_count:
            improvement = curr_sensor_count - best_sensor_count
            shaped_reward += (config['progress_bonus'] * improvement) / 1000.0
            best_sensor_count = curr_sensor_count
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
        
        # 4. IR BONUS (very close to box)
        if next_state[16] == 1:
            shaped_reward += config['ir_bonus'] / 1000.0
        
        # 5. ATTACHMENT BONUS (box attached!)
        if (not done and
                reward >= config['attachment_reward_threshold'] and
                reward < config['success_reward_threshold']):
            shaped_reward += config['attachment_bonus'] / 1000.0
        
        # 6. STAGNATION PENALTY (not making progress)
        if steps_since_improvement > 50:
            shaped_reward -= config['stagnation_penalty'] / 1000.0
        
        # 7. LOOP PENALTY (repeating actions)
        if len(action_memory) >= 8:
            unique_actions = len(set(action_memory))
            if unique_actions <= 2:  # Very repetitive
                shaped_reward -= config['loop_penalty'] / 1000.0
        
        action_memory.append(action_idx)
        
        states.append(state.copy())
        actions.append(action_idx)
        log_probs.append(log_prob)
        rewards.append(shaped_reward)
        dones.append(done)
        timeouts.append(is_timeout)
        values.append(value)
        
        episode_reward += reward
        episode_length += 1
        state = next_state
        
        if done:
            episode_successes.append(reward >= config['success_reward_threshold'])
            episode_returns.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Reset
            state = env.reset(seed=seed + len(episode_returns))
            episode_reward = 0
            episode_length = 0
            visited_state_hashes.clear()
            action_memory.clear()
            best_sensor_count = 0
            steps_since_improvement = 0
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'log_probs': np.array(log_probs),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'timeouts': np.array(timeouts),
        'values': np.array(values),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes,
    }

# ============================================================================
# UTILS
# ============================================================================

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    raise NotImplementedError("Use compute_gae_with_timeouts")


def compute_gae_with_timeouts(rewards, values, dones, timeouts, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value_t = next_value
        else:
            next_value_t = values[t + 1]

        if dones[t]:
            mask = 1.0 if timeouts[t] else 0.0
            if not timeouts[t]:
                next_value_t = 0.0
        else:
            mask = 1.0

        delta = rewards[t] + gamma * next_value_t * mask - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae

    returns = advantages + values
    return advantages, returns

# ============================================================================
# NETWORK
# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        features = self.feature(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

# ============================================================================
# ADAPTIVE TRAINER
# ============================================================================

class AdaptivePPOTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.policy = ActorCritic(18, 5, args.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr, eps=1e-5)
        
        # LR scheduler (decay when stuck)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50
        )
        
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        # ANTI-FORGETTING: Track best success rate
        self.best_success_rate = 0.0
        self.best_policy_state = None
        self.updates_since_improvement = 0
        
        # ADAPTIVE ENTROPY: Start high, decay over time
        self.current_ent_coef = args.ent_coef_start
        self.total_timesteps = 0
    
    def collect_rollout(self, env_config, num_steps):
        if self.args.num_workers > 1:
            return self._collect_parallel(env_config, num_steps)
        else:
            raise NotImplementedError("Use parallel mode")
    
    def _collect_parallel(self, env_config, num_steps):
        policy_state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        
        config = {
            'success_bonus': self.args.success_bonus,
            'success_reward_threshold': self.args.success_reward_threshold,
            'curiosity_bonus': self.args.curiosity_bonus,
            'progress_bonus': self.args.progress_bonus,
            'ir_bonus': self.args.ir_bonus,
            'attachment_bonus': self.args.attachment_bonus,
            'attachment_reward_threshold': self.args.attachment_reward_threshold,
            'stagnation_penalty': self.args.stagnation_penalty,
            'loop_penalty': self.args.loop_penalty,
        }
        
        worker_args = []
        for i in range(self.args.num_workers):
            seed = self.args.seed + i * 1000
            worker_args.append((
                self.args.obelix_py, env_config, policy_state,
                num_steps, seed, config, self.args.hidden_dim,
                i, self.args.render
            ))
        
        with Pool(processes=self.args.num_workers) as pool:
            results = pool.map(parallel_rollout_worker, worker_args)
        
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_advantages = []
        all_returns = []

        for r in results:
            self.episode_returns.extend(r['episode_returns'])
            self.episode_lengths.extend(r['episode_lengths'])
            self.episode_successes.extend(r['episode_successes'])

            if len(r['states']) == 0:
                continue

            if not r['dones'][-1]:
                with torch.no_grad():
                    final_state_t = torch.FloatTensor(r['states'][-1]).unsqueeze(0).to(self.device)
                    _, next_value = self.policy(final_state_t)
                    next_value = next_value.item()
            else:
                next_value = r['values'][-1] if r['timeouts'][-1] else 0.0

            advantages, returns = compute_gae_with_timeouts(
                r['rewards'], r['values'], r['dones'], r['timeouts'], next_value,
                self.args.gamma, self.args.gae_lambda
            )

            all_states.append(r['states'])
            all_actions.append(r['actions'])
            all_log_probs.append(r['log_probs'])
            all_rewards.append(r['rewards'])
            all_dones.append(r['dones'])
            all_values.append(r['values'])
            all_advantages.append(advantages)
            all_returns.append(returns)

        if not all_states:
            raise RuntimeError("No rollout data collected from workers")

        states = np.concatenate(all_states)
        actions = np.concatenate(all_actions)
        log_probs = np.concatenate(all_log_probs)
        rewards = np.concatenate(all_rewards)
        dones = np.concatenate(all_dones)
        values = np.concatenate(all_values)
        advantages = np.concatenate(all_advantages)
        returns = np.concatenate(all_returns)
        
        from dataclasses import dataclass
        
        @dataclass
        class RolloutBatch:
            states: np.ndarray
            actions: np.ndarray
            log_probs: np.ndarray
            rewards: np.ndarray
            dones: np.ndarray
            values: np.ndarray
            advantages: np.ndarray
            returns: np.ndarray
        
        return RolloutBatch(
            states, actions, log_probs,
            rewards, dones, values,
            advantages, returns
        )
    
    def update(self, batch):
        advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        
        states_t = torch.FloatTensor(batch.states).to(self.device)
        actions_t = torch.LongTensor(batch.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(batch.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(batch.returns).to(self.device)
        
        for epoch in range(self.args.ppo_epochs):
            indices = torch.randperm(len(states_t))
            
            for start in range(0, len(states_t), self.args.batch_size):
                end = start + self.args.batch_size
                idx = indices[start:end]
                
                logits, values = self.policy(states_t[idx])
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), returns_t[idx])
                
                # ADAPTIVE ENTROPY
                loss = policy_loss + self.args.vf_coef * value_loss - self.current_ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropies.append(entropy.item())
    
    def train(self):
        env_config = {
            'scaling_factor': self.args.scaling_factor,
            'arena_size': self.args.arena_size,
            'max_steps': self.args.max_steps,
            'wall_obstacles': self.args.wall_obstacles,
            'difficulty': self.args.difficulty,
            'box_speed': self.args.box_speed,
        }
        
        print("🚀 Starting ADAPTIVE PPO training...\n")
        start_time = time.time()
        
        steps_per_update = self.args.rollout_steps * self.args.num_workers
        num_updates = max(1, self.args.total_timesteps // steps_per_update)
        
        for update in range(num_updates):
            batch = self.collect_rollout(env_config, self.args.rollout_steps)
            self.update(batch)
            
            # ADAPTIVE ENTROPY DECAY (global step based)
            self.total_timesteps += steps_per_update
            progress = min(1.0, self.total_timesteps / max(1, self.args.total_timesteps))
            self.current_ent_coef = (
                self.args.ent_coef_start +
                progress * (self.args.ent_coef_end - self.args.ent_coef_start)
            )
            
            # Check for improvement
            current_success = np.mean(self.episode_successes) if self.episode_successes else 0
            
            if current_success >= self.best_success_rate and current_success > 0:
                self.best_success_rate = current_success
                self.best_policy_state = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
                self.updates_since_improvement = 0
                print(f"🎯 NEW BEST! Success: {current_success*100:.1f}%")
            else:
                self.updates_since_improvement += 1
            
            # ANTI-FORGETTING: Restore best if stuck too long
            if self.updates_since_improvement > 100 and self.best_policy_state is not None:
                print(f"⚠️  RESTORING BEST POLICY (stuck for 100 updates)")
                self.policy.load_state_dict({k: v.to(self.device) for k, v in self.best_policy_state.items()})
                self.updates_since_improvement = 0
                self.current_ent_coef = self.args.ent_coef_start  # Reset exploration
            
            # LR scheduling
            self.scheduler.step(current_success)
            
            if (update + 1) % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                mean_return = np.mean(self.episode_returns) if self.episode_returns else 0
                std_return = np.std(self.episode_returns) if self.episode_returns else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                print(f"Update {update+1:4d}/{num_updates} | "
                      f"Return: {mean_return:8.1f} ± {std_return:6.1f} | "
                      f"Length: {mean_length:5.1f} | "
                      f"Success: {current_success*100:5.1f}% | "
                      f"Entropy: {np.mean(self.entropies[-100:]) if self.entropies else 0:5.3f} | "
                      f"EntCoef: {self.current_ent_coef:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {elapsed/60:.1f}m")
            
            if (update + 1) % self.args.save_interval == 0:
                self.save_checkpoint(update + 1)
        
        self.save_checkpoint(num_updates, final=True)
        
        print(f"\n{'='*70}")
        print(f"✅ Training Complete!")
        print(f"{'='*70}")
        print(f"Time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Best Success Rate: {self.best_success_rate*100:.1f}%")
        print(f"Final Success Rate: {current_success*100:.1f}%")
    
    def save_checkpoint(self, update, final=False):
        submission_dir = Path(f"submission_{self.args.agent_id}")
        submission_dir.mkdir(exist_ok=True)
        
        if final:
            # Save best policy, not final
            if self.best_policy_state is not None:
                torch.save(self.best_policy_state, submission_dir / "weights.pth")
                print(f"✓ BEST model saved (Success: {self.best_success_rate*100:.1f}%)")
            else:
                torch.save(self.policy.state_dict(), submission_dir / "weights.pth")
                print(f"✓ Final model saved")
        else:
            checkpoint = {
                'policy_state_dict': self.policy.state_dict(),
                'best_policy_state_dict': self.best_policy_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'update': update,
                'best_success_rate': self.best_success_rate,
                'current_ent_coef': self.current_ent_coef,
            }
            torch.save(checkpoint, submission_dir / f"checkpoint_update{update}.pth")

# ============================================================================
# MAIN
# ============================================================================

def suppress_qt_warnings(render_enabled: bool):
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    if render_enabled:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ["XDG_SESSION_TYPE"] = "x11"

    sys.stderr = _QtStderrFilter(sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--agent_id", type=str, default="ppo_adaptive")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    
    # Training
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=6)
    
    # PPO
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef_start", type=float, default=0.05, help="Start entropy coef")
    parser.add_argument("--ent_coef_end", type=float, default=0.001, help="End entropy coef")
    parser.add_argument("--ent_coef_decay", type=float, default=0.999, help="Entropy decay rate")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    
    # REWARDS
    parser.add_argument("--success_bonus", type=float, default=250000)
    parser.add_argument("--success_reward_threshold", type=float, default=2000)
    parser.add_argument("--curiosity_bonus", type=float, default=5)
    parser.add_argument("--progress_bonus", type=float, default=100)
    parser.add_argument("--ir_bonus", type=float, default=50)
    parser.add_argument("--attachment_bonus", type=float, default=1000)
    parser.add_argument("--attachment_reward_threshold", type=float, default=100)
    parser.add_argument("--stagnation_penalty", type=float, default=10)
    parser.add_argument("--loop_penalty", type=float, default=50)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="Show live visualization window (worker 0 only)")
    
    args = parser.parse_args()

    suppress_qt_warnings(args.render)
    
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"ADAPTIVE PPO - Anti-Forgetting + Curiosity + Progress")
    print(f"{'='*70}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"\n🎯 KEY FEATURES:")
    print(f"  ✓ Success Bonus: +{args.success_bonus:,}")
    print(f"  ✓ Curiosity Exploration: +{args.curiosity_bonus} per new state")
    print(f"  ✓ Progress Rewards: +{args.progress_bonus} per sensor improvement")
    print(f"  ✓ Adaptive Entropy: {args.ent_coef_start} → {args.ent_coef_end}")
    print(f"  ✓ Auto LR Decay: Reduces when stuck")
    print(f"  ✓ Anti-Forgetting: Restores best policy if stuck 100 updates")
    print(f"{'='*70}\n")
    
    trainer = AdaptivePPOTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()
