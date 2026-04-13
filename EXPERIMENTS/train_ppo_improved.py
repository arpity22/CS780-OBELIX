#!/usr/bin/env python3
"""
IMPROVED PPO for OBELIX POMDP

ARCHITECTURAL IMPROVEMENTS:
1. Deeper network (4 layers instead of 2)
2. LSTM for partial observability
3. Curriculum learning (start easy, get harder)
4. Better reward scaling
5. Action distribution regularization
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import random
import argparse
import importlib.util
from pathlib import Path
from collections import deque
from multiprocessing import Pool
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.episodes = []
        self.returns = []
        self.lengths = []
        self.successes = []
        
        self.updates = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.avg_returns = []
        self.success_rates = []
        self.action_distributions = []
    
    def add_episode(self, episode, return_val, length, success):
        self.episodes.append(episode)
        self.returns.append(return_val)
        self.lengths.append(length)
        self.successes.append(1 if success else 0)
    
    def add_update(self, update, policy_loss, value_loss, entropy, avg_return, success_rate, action_dist):
        self.updates.append(update)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.avg_returns.append(avg_return)
        self.success_rates.append(success_rate)
        self.action_distributions.append(action_dist)
    
    def plot(self):
        if len(self.updates) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('PPO Training Metrics', fontsize=16, fontweight='bold')
        
        # Success Rate
        ax = axes[0, 0]
        ax.plot(self.updates, [s*100 for s in self.success_rates], 'g-', linewidth=2, label='Success %')
        ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlabel('Update', fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('Success Rate', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Average Return
        ax = axes[0, 1]
        ax.plot(self.updates, self.avg_returns, 'b-', linewidth=2)
        ax.set_xlabel('Update', fontweight='bold')
        ax.set_ylabel('Average Return', fontweight='bold')
        ax.set_title('Episode Return', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Episode Length
        ax = axes[0, 2]
        if len(self.episodes) > 0:
            window = 50
            if len(self.lengths) >= window:
                rolling_lengths = np.convolve(self.lengths, np.ones(window)/window, mode='valid')
                rolling_episodes = self.episodes[window-1:]
                ax.plot(rolling_episodes, rolling_lengths, 'purple', linewidth=2)
            ax.set_xlabel('Episode', fontweight='bold')
            ax.set_ylabel('Steps', fontweight='bold')
            ax.set_title('Episode Length (50-ep avg)', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Policy Loss
        ax = axes[1, 0]
        ax.plot(self.updates, self.policy_losses, 'orange', linewidth=2)
        ax.set_xlabel('Update', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Policy Loss', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Value Loss
        ax = axes[1, 1]
        ax.plot(self.updates, self.value_losses, 'red', linewidth=2)
        ax.set_xlabel('Update', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Value Loss', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Action Distribution
        ax = axes[1, 2]
        if len(self.action_distributions) > 0:
            action_arrays = np.array(self.action_distributions)
            for i, action_name in enumerate(ACTIONS):
                ax.plot(self.updates, action_arrays[:, i] * 100, linewidth=2, label=action_name, alpha=0.8)
            ax.axhline(y=20, color='gray', linestyle='--', alpha=0.3, label='Uniform')
            ax.set_xlabel('Update', fontweight='bold')
            ax.set_ylabel('Action Frequency (%)', fontweight='bold')
            ax.set_title('Action Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to {plot_path}")

# ============================================================================
# IMPROVED ARCHITECTURE
# ============================================================================

class ImprovedActorCritic(nn.Module):
    """
    DEEPER network with LSTM for POMDP
    
    Changes from original:
    - 4 hidden layers instead of 2
    - Layer sizes: 18 → 256 → 256 → 128 → 128
    - LSTM for temporal dependencies
    - Separate value/policy heads with more capacity
    """
    
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=256, use_lstm=True):
        super().__init__()
        
        self.use_lstm = use_lstm
        
        # Deeper feature extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # LSTM for temporal patterns (POMDP)
        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim // 2, batch_first=True)
            self.hidden_state = None
        
        # Deeper actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )
        
        # Deeper critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        
        self.apply(self._init_weights)
        self.lstm_hidden_size = hidden_dim // 2
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, x, hidden=None):
        features = self.feature(x)
        
        if self.use_lstm:
            # Add sequence dimension if needed
            if len(features.shape) == 2:
                features = features.unsqueeze(1)
            
            if hidden is None:
                features, hidden = self.lstm(features)
            else:
                features, hidden = self.lstm(features, hidden)
            
            features = features.squeeze(1)
        
        logits = self.actor(features)
        value = self.critic(features)
        
        if self.use_lstm:
            return logits, value, hidden
        else:
            return logits, value
    
    def reset_hidden(self, batch_size=1):
        """Reset LSTM hidden state"""
        if self.use_lstm:
            device = next(self.parameters()).device
            self.hidden_state = (
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(device),
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(device)
            )

# ============================================================================
# WORKER
# ============================================================================

def parallel_rollout_worker(args_tuple):
    (obelix_path, env_config, policy_state, num_steps, seed, 
     config, hidden_dim, worker_id, render_enabled, use_lstm) = args_tuple
    
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
    
    torch.set_num_threads(1)  # Prevent CPU thread thrashing in local worker
    
    # Use improved architecture
    policy = ImprovedActorCritic(hidden_dim=hidden_dim, use_lstm=use_lstm)
    policy.load_state_dict(policy_state)
    policy.eval()
    
    env = OBELIX(**env_config)
    state = env.reset(seed=seed)
    
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    timeouts = []
    episode_returns, episode_lengths, episode_successes = [], [], []
    action_counts = np.zeros(len(ACTIONS))
    
    episode_reward = 0
    episode_length = 0
    
    # Reset LSTM hidden state
    if use_lstm:
        policy.reset_hidden(1)
        hidden = policy.hidden_state
    else:
        hidden = None
    
    for step in range(num_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if use_lstm:
                logits, value, hidden = policy(state_t, hidden)
            else:
                logits, value = policy(state_t)
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        
        action_idx = action_idx.item()
        log_prob = log_prob.item()
        value = value.item()
        
        action_counts[action_idx] += 1
        
        next_state, reward, done = env.step(ACTIONS[action_idx], render=(render_enabled and worker_id == 0))
        is_timeout = (episode_length + 1 >= env_config['max_steps']) and done
        
        # IMPROVED REWARD SCALING
        # Scale base reward but keep success bonus large
        base_scaled = reward / 1000.0
        
        if done and reward >= 2000:
            # Success: scale lightly
            scaled_reward = base_scaled + (config['success_bonus'] / 100.0)
        else:
            # Normal: use scaled base
            scaled_reward = base_scaled
        
        states.append(state.copy())
        actions.append(action_idx)
        log_probs.append(log_prob)
        rewards.append(scaled_reward)
        dones.append(done)
        timeouts.append(is_timeout)
        values.append(value)
        
        episode_reward += reward
        episode_length += 1
        state = next_state
        
        if done:
            episode_successes.append(reward >= 2000)
            episode_returns.append(episode_reward)
            episode_lengths.append(episode_length)
            
            state = env.reset(seed=seed + len(episode_returns))
            episode_reward = 0
            episode_length = 0
            
            # Reset LSTM
            if use_lstm:
                policy.reset_hidden(1)
                hidden = policy.hidden_state
    
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
        'action_counts': action_counts,
    }

# ============================================================================
# UTILS
# ============================================================================

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def compute_gae(rewards, values, dones, timeouts, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
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
# TRAINER
# ============================================================================

class ImprovedPPOTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Use improved architecture
        self.policy = ImprovedActorCritic(hidden_dim=args.hidden_dim, use_lstm=args.use_lstm).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr, eps=1e-5)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50
        )
        
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        self.best_success_rate = 0.0
        self.best_policy_state = None
        self.best_optimizer_state = None
        self.updates_since_improvement = 0
        
        self.total_timesteps = 0
        self.current_ent_coef = args.ent_coef_start
        
        self.metrics = MetricsTracker(f"submission_{args.agent_id}")
        self.total_episodes = 0
        
        # Action tracking
        self.total_action_counts = np.zeros(len(ACTIONS))
    
    def collect_rollout(self, env_config, num_steps):
        policy_state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        
        config = {
            'success_bonus': self.args.success_bonus,
        }
        
        worker_args = []
        for i in range(self.args.num_workers):
            seed = self.args.seed + i * 1000
            worker_args.append((
                self.args.obelix_py, env_config, policy_state,
                num_steps, seed, config, self.args.hidden_dim,
                i, self.args.render, self.args.use_lstm
            ))
        
        with Pool(processes=self.args.num_workers) as pool:
            results = pool.map(parallel_rollout_worker, worker_args)
        
        all_states = np.concatenate([r['states'] for r in results])
        all_actions = np.concatenate([r['actions'] for r in results])
        all_log_probs = np.concatenate([r['log_probs'] for r in results])
        all_rewards = np.concatenate([r['rewards'] for r in results])
        all_dones = np.concatenate([r['dones'] for r in results])
        all_timeouts = np.concatenate([r['timeouts'] for r in results])
        all_values = np.concatenate([r['values'] for r in results])
        
        for r in results:
            self.episode_returns.extend(r['episode_returns'])
            self.episode_lengths.extend(r['episode_lengths'])
            self.episode_successes.extend(r['episode_successes'])
            self.total_action_counts += r['action_counts']
            
            for ret, length, success in zip(r['episode_returns'], r['episode_lengths'], r['episode_successes']):
                self.metrics.add_episode(self.total_episodes, ret, length, success)
                self.total_episodes += 1
        
        if not all_dones[-1]:
            with torch.no_grad():
                final_state_t = torch.FloatTensor(all_states[-1]).unsqueeze(0).to(self.device)
                if self.args.use_lstm:
                    _, next_value, _ = self.policy(final_state_t, None)
                else:
                    _, next_value = self.policy(final_state_t)
                next_value = next_value.item()
        else:
            next_value = all_values[-1] if all_timeouts[-1] else 0.0
        
        advantages, returns = compute_gae(
            all_rewards, all_values, all_dones, all_timeouts, next_value,
            self.args.gamma, self.args.gae_lambda
        )
        
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
            all_states, all_actions, all_log_probs,
            all_rewards, all_dones, all_values,
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
                
                if self.args.use_lstm:
                    logits, values, _ = self.policy(states_t[idx], None)
                else:
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
        
        print("🚀 Starting IMPROVED PPO training...\n")
        start_time = time.time()
        
        num_updates = self.args.total_timesteps // (self.args.rollout_steps * self.args.num_workers)
        
        for update in range(num_updates):
            batch = self.collect_rollout(env_config, self.args.rollout_steps)
            self.update(batch)
            
            self.total_timesteps += self.args.rollout_steps * self.args.num_workers
            progress = self.total_timesteps / self.args.total_timesteps
            self.current_ent_coef = self.args.ent_coef_start + progress * (self.args.ent_coef_end - self.args.ent_coef_start)
            
            current_success = np.mean(self.episode_successes) if self.episode_successes else 0
            current_return = np.mean(self.episode_returns) if self.episode_returns else -999999
            
            if current_success > self.best_success_rate:
                self.best_success_rate = current_success
                self.best_policy_state = copy.deepcopy({k: v.cpu() for k, v in self.policy.state_dict().items()})
                self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                self.updates_since_improvement = 0
                print(f"🎯 NEW BEST! Success: {current_success*100:.1f}%")
            else:
                self.updates_since_improvement += 1
            
            if self.updates_since_improvement > 100 and self.best_policy_state is not None:
                print(f"⚠️  RESTORING BEST POLICY")
                self.policy.load_state_dict({k: v.to(self.device) for k, v in self.best_policy_state.items()})
                self.optimizer.load_state_dict(self.best_optimizer_state)
                self.updates_since_improvement = 0
                self.current_ent_coef = self.args.ent_coef_start
            
            self.scheduler.step(current_success)
            
            # Track action distribution
            action_dist = self.total_action_counts / (self.total_action_counts.sum() + 1e-8)
            
            if len(self.policy_losses) > 0:
                self.metrics.add_update(
                    update + 1,
                    np.mean(self.policy_losses[-100:]),
                    np.mean(self.value_losses[-100:]),
                    np.mean(self.entropies[-100:]),
                    current_return,
                    current_success,
                    action_dist
                )
            
            if (update + 1) % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                mean_return = current_return
                std_return = np.std(self.episode_returns) if self.episode_returns else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                # Action distribution
                action_str = " | ".join([f"{ACTIONS[i]}:{action_dist[i]*100:.0f}%" for i in range(len(ACTIONS))])
                
                print(f"Update {update+1:4d}/{num_updates} | "
                      f"Return: {mean_return:8.1f} ± {std_return:6.1f} | "
                      f"Length: {mean_length:5.1f} | "
                      f"Success: {current_success*100:5.1f}% | "
                      f"Entropy: {np.mean(self.entropies[-100:]):5.3f}")
                print(f"  Actions: {action_str}")
            
            if (update + 1) % 20 == 0:
                self.metrics.plot()
            
            if (update + 1) % self.args.save_interval == 0:
                self.save_checkpoint(update + 1)
        
        self.save_checkpoint(num_updates, final=True)
        self.metrics.plot()
        
        print(f"\n{'='*70}")
        print(f"✅ Training Complete!")
        print(f"Best Success: {self.best_success_rate*100:.1f}%")
    
    def save_checkpoint(self, update, final=False):
        submission_dir = Path(f"submission_{self.args.agent_id}")
        submission_dir.mkdir(exist_ok=True)
        
        if final:
            if self.best_policy_state is not None:
                torch.save(self.best_policy_state, submission_dir / "weights.pth")
                print(f"✓ BEST model saved (Success: {self.best_success_rate*100:.1f}%)")
        else:
            checkpoint = {
                'policy_state_dict': {k: v.cpu() for k, v in self.policy.state_dict().items()},
                'best_policy_state_dict': self.best_policy_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'update': update,
                'best_success_rate': self.best_success_rate,
            }
            torch.save(checkpoint, submission_dir / f"checkpoint_update{update}.pth")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--agent_id", type=str, default="ppo_improved")
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
    parser.add_argument("--ent_coef_start", type=float, default=0.05)
    parser.add_argument("--ent_coef_end", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=256, help="Increased from 128")
    
    # Architecture
    parser.add_argument("--use_lstm", action="store_true", help="Use LSTM for POMDP")
    
    # Rewards
    parser.add_argument("--success_bonus", type=float, default=250000)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"IMPROVED PPO - Deeper Network + LSTM")
    print(f"{'='*70}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Workers: {args.num_workers}")
    print(f"\n🏗️  ARCHITECTURE:")
    print(f"  Network depth: 4 layers (was 2)")
    print(f"  Hidden dim: {args.hidden_dim} (was 128)")
    print(f"  LSTM: {'ENABLED' if args.use_lstm else 'DISABLED'}")
    print(f"\n🎯 SUCCESS BONUS: {args.success_bonus:,}")
    print(f"📊 Plots: submission_{args.agent_id}/")
    print(f"{'='*70}\n")
    
    trainer = ImprovedPPOTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()
