#!/usr/bin/env python3
"""
PPO with Visualization (DDQN-style) + Corrected Reward Scaling

KEY FIX: Reward scaling applied SELECTIVELY:
- Success bonus: NOT scaled (keep massive)
- Environment penalties: Scaled (for stability)
- Result: Success becomes overwhelmingly important again
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
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ============================================================================
# METRICS TRACKER (like DDQN)
# ============================================================================

class MetricsTracker:
    """Track and plot training metrics."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.episodes = []
        self.returns = []
        self.lengths = []
        self.successes = []
        self.avg_returns = []
        self.success_rates = []
        
        # Per-update metrics
        self.updates = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.entropy_coefs = []
    
    def add_episode(self, episode, return_val, length, success):
        """Add single episode metrics."""
        self.episodes.append(episode)
        self.returns.append(return_val)
        self.lengths.append(length)
        self.successes.append(1 if success else 0)
    
    def add_update(self, update, policy_loss, value_loss, entropy, ent_coef, avg_return, success_rate):
        """Add update-level metrics."""
        self.updates.append(update)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.entropy_coefs.append(ent_coef)
        self.avg_returns.append(avg_return)
        self.success_rates.append(success_rate)
    
    def plot(self):
        """Generate and save plots."""
        if len(self.updates) < 2:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('PPO Training Metrics', fontsize=16)
        
        # 1. Success Rate
        ax = axes[0, 0]
        ax.plot(self.updates, [s*100 for s in self.success_rates], 'g-', linewidth=2)
        ax.set_xlabel('Update')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Over Training')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=95, color='r', linestyle='--', label='Efficiency Threshold')
        ax.legend()
        
        # 2. Average Return
        ax = axes[0, 1]
        ax.plot(self.updates, self.avg_returns, 'b-', linewidth=2)
        ax.set_xlabel('Update')
        ax.set_ylabel('Average Return')
        ax.set_title('Average Episode Return')
        ax.grid(True, alpha=0.3)
        
        # 3. Episode Length
        ax = axes[1, 0]
        if len(self.episodes) > 0:
            # Rolling average
            window = 50
            if len(self.lengths) >= window:
                rolling_lengths = np.convolve(self.lengths, np.ones(window)/window, mode='valid')
                rolling_episodes = self.episodes[window-1:]
                ax.plot(rolling_episodes, rolling_lengths, 'purple', linewidth=2, label=f'{window}-ep avg')
            ax.plot(self.episodes, self.lengths, 'purple', alpha=0.3, label='Raw')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
            ax.set_title('Episode Length')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 4. Policy Loss
        ax = axes[1, 1]
        ax.plot(self.updates, self.policy_losses, 'orange', linewidth=2)
        ax.set_xlabel('Update')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
        
        # 5. Value Loss
        ax = axes[2, 0]
        ax.plot(self.updates, self.value_losses, 'red', linewidth=2)
        ax.set_xlabel('Update')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss (Critic)')
        ax.grid(True, alpha=0.3)
        
        # 6. Entropy & Coefficient
        ax = axes[2, 1]
        ax2 = ax.twinx()
        ax.plot(self.updates, self.entropies, 'cyan', linewidth=2, label='Entropy')
        ax2.plot(self.updates, self.entropy_coefs, 'magenta', linewidth=2, label='Ent Coef')
        ax.set_xlabel('Update')
        ax.set_ylabel('Entropy', color='cyan')
        ax2.set_ylabel('Entropy Coefficient', color='magenta')
        ax.set_title('Exploration Metrics')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to {plot_path}")

# ============================================================================
# WORKER (CORRECTED REWARD SCALING)
# ============================================================================

def parallel_rollout_worker(args_tuple):
    """
    FIXED: Selective reward scaling.
    - Success bonus: Keep massive (not scaled)
    - Env penalties: Scale down (for stability)
    - Rendering: Only worker 0 shows visualization
    """
    (obelix_path, env_config, policy_state, num_steps, seed, 
     config, hidden_dim, worker_id, render_enabled) = args_tuple
    
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
                nn.Linear(hidden_dim, hidden_dim),  # FIX: hidden_dim, not in_dim
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
    
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    timeouts = []
    episode_returns, episode_lengths, episode_successes = [], [], []
    
    episode_reward = 0
    episode_length = 0
    action_memory = deque(maxlen=8)
    
    for step in range(num_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = policy(state_t)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        
        action_idx = action_idx.item()
        log_prob = log_prob.item()
        value = value.item()
        
        next_state, reward, done = env.step(ACTIONS[action_idx], render=(render_enabled and worker_id == 0))
        is_timeout = (episode_length + 1 >= env_config['max_steps']) and done
        
        # REWARD SHAPING
        # Start with scaled base reward
        base_scaled = reward / 1000.0  # Scale env rewards (-200, +5, etc.)
        shaped_reward = base_scaled
        
        # Add shaped bonuses
        # CRITICAL: Success bonus scaled separately (less aggressive)
        if done and reward >= 2000:
            # Success: add massive bonus (scaled lightly)
            shaped_reward += config['success_bonus'] / 100.0  # 250k -> 2500
        
        # Other bonuses (already small, scale normally)
        if next_state[16] == 1:
            shaped_reward += config['ir_bonus'] / 1000.0
        if next_state[17] == 1:
            shaped_reward += config['attachment_bonus'] / 1000.0
        
        # Loop penalty
        if len(action_memory) >= 8 and len(set(action_memory)) <= 2:
            shaped_reward -= config['loop_penalty'] / 1000.0
        
        action_memory.append(action_idx)
        
        # Use shaped_reward directly (already scaled)
        scaled_reward = shaped_reward
        
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
            action_memory.clear()
    
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
# TRAINER
# ============================================================================

class VisualizedPPOTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.policy = ActorCritic(18, 5, args.hidden_dim).to(device)
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
        self.optimization_stage = 1
        
        # Metrics tracker
        self.metrics = MetricsTracker(f"submission_{args.agent_id}")
        
        # Episode counter for visualization
        self.total_episodes = 0
    
    def collect_rollout(self, env_config, num_steps):
        policy_state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        
        config = {
            'success_bonus': self.args.success_bonus,
            'ir_bonus': self.args.ir_bonus,
            'attachment_bonus': self.args.attachment_bonus,
            'loop_penalty': self.args.loop_penalty,
        }
        
        worker_args = []
        for i in range(self.args.num_workers):
            seed = self.args.seed + i * 1000
            worker_args.append((
                self.args.obelix_py, env_config, policy_state,
                num_steps, seed, config, self.args.hidden_dim,
                i,  # worker_id
                self.args.render  # render_enabled flag
            ))
        
        with Pool(processes=self.args.num_workers) as pool:
            results = pool.map(parallel_rollout_worker, worker_args)
        
        # Concatenate ALL worker data
        all_states = np.concatenate([r['states'] for r in results])
        all_actions = np.concatenate([r['actions'] for r in results])
        all_log_probs = np.concatenate([r['log_probs'] for r in results])
        all_rewards = np.concatenate([r['rewards'] for r in results])
        all_dones = np.concatenate([r['dones'] for r in results])
        all_timeouts = np.concatenate([r['timeouts'] for r in results])
        all_values = np.concatenate([r['values'] for r in results])
        
        # Track episodes
        for r in results:
            self.episode_returns.extend(r['episode_returns'])
            self.episode_lengths.extend(r['episode_lengths'])
            self.episode_successes.extend(r['episode_successes'])
            
            # Add to visualization
            for ret, length, success in zip(r['episode_returns'], r['episode_lengths'], r['episode_successes']):
                self.metrics.add_episode(self.total_episodes, ret, length, success)
                self.total_episodes += 1
        
        # Bootstrap
        if not all_dones[-1]:
            with torch.no_grad():
                final_state_t = torch.FloatTensor(all_states[-1]).unsqueeze(0).to(self.device)
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
        
        print("🚀 Starting PPO with Visualization...\n")
        start_time = time.time()
        
        num_updates = self.args.total_timesteps // (self.args.rollout_steps * self.args.num_workers)
        
        for update in range(num_updates):
            batch = self.collect_rollout(env_config, self.args.rollout_steps)
            self.update(batch)
            
            # Entropy decay
            self.total_timesteps += self.args.rollout_steps * self.args.num_workers
            progress = self.total_timesteps / self.args.total_timesteps
            self.current_ent_coef = self.args.ent_coef_start + progress * (self.args.ent_coef_end - self.args.ent_coef_start)
            
            current_success = np.mean(self.episode_successes) if self.episode_successes else 0
            current_return = np.mean(self.episode_returns) if self.episode_returns else -999999
            
            # Check improvement
            if current_success > self.best_success_rate:
                self.best_success_rate = current_success
                self.best_policy_state = copy.deepcopy({k: v.cpu() for k, v in self.policy.state_dict().items()})
                self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                self.updates_since_improvement = 0
                print(f"🎯 NEW BEST! Success: {current_success*100:.1f}%")
            else:
                self.updates_since_improvement += 1
            
            # Anti-forgetting
            if self.updates_since_improvement > 100 and self.best_policy_state is not None:
                print(f"⚠️  RESTORING BEST POLICY")
                self.policy.load_state_dict({k: v.to(self.device) for k, v in self.best_policy_state.items()})
                self.optimizer.load_state_dict(self.best_optimizer_state)
                self.updates_since_improvement = 0
                self.current_ent_coef = self.args.ent_coef_start
            
            self.scheduler.step(current_success)
            
            # Add metrics for visualization
            if len(self.policy_losses) > 0:
                self.metrics.add_update(
                    update + 1,
                    np.mean(self.policy_losses[-100:]),
                    np.mean(self.value_losses[-100:]),
                    np.mean(self.entropies[-100:]),
                    self.current_ent_coef,
                    current_return,
                    current_success
                )
            
            if (update + 1) % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                mean_return = current_return
                std_return = np.std(self.episode_returns) if self.episode_returns else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                print(f"Update {update+1:4d}/{num_updates} | "
                      f"Return: {mean_return:8.1f} ± {std_return:6.1f} | "
                      f"Length: {mean_length:5.1f} | "
                      f"Success: {current_success*100:5.1f}% | "
                      f"Entropy: {np.mean(self.entropies[-100:]):5.3f} | "
                      f"Time: {elapsed/60:.1f}m")
            
            # Plot every 20 updates
            if (update + 1) % 20 == 0:
                self.metrics.plot()
            
            if (update + 1) % self.args.save_interval == 0:
                self.save_checkpoint(update + 1)
        
        self.save_checkpoint(num_updates, final=True)
        self.metrics.plot()  # Final plot
        
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
                'best_optimizer_state_dict': self.best_optimizer_state,
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
    parser.add_argument("--agent_id", type=str, default="ppo_viz")
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
    parser.add_argument("--hidden_dim", type=int, default=128)
    
    # REWARDS (NO SCALING on success!)
    parser.add_argument("--success_bonus", type=float, default=250000)
    parser.add_argument("--ir_bonus", type=float, default=50)
    parser.add_argument("--attachment_bonus", type=float, default=1000)
    parser.add_argument("--loop_penalty", type=float, default=100)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="Show live visualization window (worker 0 only)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"PPO with Visualization + Complete Reward Scaling")
    print(f"{'='*70}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Workers: {args.num_workers}")
    if args.render:
        print(f"🎮 Live Pygame Window: ENABLED (worker 0 only)")
        print(f"   ⚠️  Training will be slower with rendering")
    else:
        print(f"🎮 Live Pygame Window: DISABLED (use --render to enable)")
    print(f"\n🔧 REWARD SCALING STRATEGY:")
    print(f"  Base env rewards: /1000 (-0.2 wall, +0.005 sensor, +2.0 success)")
    print(f"  Success bonus: {args.success_bonus:,} /100 = {args.success_bonus/100:,.0f}")
    print(f"  IR bonus: {args.ir_bonus} /1000 = {args.ir_bonus/1000:.3f}")
    print(f"  Attachment: {args.attachment_bonus} /1000 = {args.attachment_bonus/1000:.1f}")
    print(f"  Loop penalty: {args.loop_penalty} /1000 = {args.loop_penalty/1000:.1f}")
    print(f"\n💡 RESULT:")
    print(f"  Success episode: ~+{2 + args.success_bonus/100:,.0f}")
    print(f"  Failed episode: ~-{200*1500/1000:.0f}")
    print(f"  Ratio: {(2 + args.success_bonus/100) / (200*1500/1000):.1f}x (SUCCESS DOMINATES!)")
    print(f"\n📊 Plots will be saved to: submission_{args.agent_id}/")
    print(f"{'='*70}\n")
    
    trainer = VisualizedPPOTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()