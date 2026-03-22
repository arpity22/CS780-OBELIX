#!/usr/bin/env python3
"""
ADVANCED PPO with Memory, Success Bonus, and Phased Exploration

KEY IMPROVEMENTS:
1. Short-term memory (detects loops/stuck states)
2. MASSIVE success bonus (200k reward)
3. Phased exploration (explore → greedy transition)
4. Optimized for speed
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ============================================================================
# MEMORY-ENHANCED PARALLEL WORKER
# ============================================================================

def parallel_rollout_worker(args_tuple):
    """Worker with short-term memory to detect stuck states."""
    (obelix_path, env_config, policy_state, num_steps, seed, 
     config, hidden_dim) = args_tuple
    
    # Import OBELIX
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    # Recreate policy
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
    
    # Create environment
    env = OBELIX(**env_config)
    state = env.reset(seed=seed)
    
    # Storage
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    episode_returns, episode_lengths, episode_successes = [], [], []
    
    episode_reward = 0
    episode_length = 0
    
    # SHORT-TERM MEMORY (detect loops/stuck states)
    memory_size = 4
    state_action_memory = deque(maxlen=memory_size)
    
    # Phase tracking
    box_found = False
    steps_since_box_found = 0
    
    for step in range(num_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = policy(state_t)
            
            # PHASED EXPLORATION
            # Phase 1: Explore (before finding box)
            # Phase 2: Exploit (after finding box)
            if not box_found:
                # High temperature = more exploration
                temperature = config['explore_temperature']
            else:
                # Low temperature = more greedy
                temperature = config['exploit_temperature']
                # Decay temperature as we get closer
                temp_decay = max(0.1, 1.0 - (steps_since_box_found / 500))
                temperature = temperature * temp_decay
            
            probs = F.softmax(logits / temperature, dim=-1)
            
            # LOOP DETECTION: Check if stuck in pattern
            stuck_penalty = 0.0
            if len(state_action_memory) == memory_size:
                # Check if repeating same action
                recent_actions = [sa[1] for sa in state_action_memory]
                if len(set(recent_actions)) == 1:  # All same action
                    stuck_penalty = -config['stuck_penalty']
                    # Force different action
                    last_action = recent_actions[0]
                    probs[0, last_action] *= 0.1  # Reduce prob of repeating
                    probs = probs / probs.sum()  # Renormalize
            
            dist = Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        
        action_idx = action_idx.item()
        log_prob = log_prob.item()
        value = value.item()
        
        # Take action
        next_state, reward, done = env.step(ACTIONS[action_idx], render=False)
        
        # Track if box found
        if not box_found and np.any(next_state[:16] == 1):
            box_found = True
            steps_since_box_found = 0
        if box_found:
            steps_since_box_found += 1
        
        # REWARD ENGINEERING
        shaped_reward = reward
        
        # 1. MASSIVE SUCCESS BONUS (200k comparable to wall penalties)
        if done and reward > 0:  # Success!
            shaped_reward += config['success_bonus']
        
        # 2. Loop/stuck penalty
        shaped_reward += stuck_penalty
        
        # 3. Exploration bonus (only before finding box)
        if not box_found:
            # Reward finding box area
            sensor_count = np.sum(next_state[:16] == 1)
            if sensor_count > 0:
                shaped_reward += config['explore_sensor_bonus']
            if next_state[16] == 1:  # IR sensor
                shaped_reward += config['explore_ir_bonus']
        else:
            # After finding box: reward staying close
            sensor_count = np.sum(next_state[:16] == 1)
            if sensor_count > 0:
                shaped_reward += config['exploit_sensor_bonus']
            if next_state[16] == 1:
                shaped_reward += config['exploit_ir_bonus']
        
        # Update memory
        state_action_memory.append((state.copy(), action_idx))
        
        # Store
        states.append(state.copy())
        actions.append(action_idx)
        log_probs.append(log_prob)
        rewards.append(shaped_reward)
        dones.append(done)
        values.append(value)
        
        episode_reward += reward  # Track base reward
        episode_length += 1
        state = next_state
        
        if done:
            episode_successes.append(reward > 0)
            episode_returns.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Reset
            state = env.reset(seed=seed + len(episode_returns))
            episode_reward = 0
            episode_length = 0
            state_action_memory.clear()
            box_found = False
            steps_since_box_found = 0
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'log_probs': np.array(log_probs),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'values': np.array(values),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes,
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
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

class PPOTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.policy = ActorCritic(18, 5, args.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr, eps=1e-5)
        
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
    
    def collect_rollout(self, env_config, num_steps):
        if self.args.num_workers > 1:
            return self._collect_parallel(env_config, num_steps)
        else:
            return self._collect_sequential(env_config, num_steps)
    
    def _collect_parallel(self, env_config, num_steps):
        policy_state = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        
        config = {
            'success_bonus': self.args.success_bonus,
            'stuck_penalty': self.args.stuck_penalty,
            'explore_temperature': self.args.explore_temperature,
            'exploit_temperature': self.args.exploit_temperature,
            'explore_sensor_bonus': self.args.explore_sensor_bonus,
            'explore_ir_bonus': self.args.explore_ir_bonus,
            'exploit_sensor_bonus': self.args.exploit_sensor_bonus,
            'exploit_ir_bonus': self.args.exploit_ir_bonus,
        }
        
        worker_args = []
        for i in range(self.args.num_workers):
            seed = self.args.seed + i * 1000
            worker_args.append((
                self.args.obelix_py, env_config, policy_state,
                num_steps, seed, config, self.args.hidden_dim
            ))
        
        with Pool(processes=self.args.num_workers) as pool:
            results = pool.map(parallel_rollout_worker, worker_args)
        
        result = results[0]
        for r in results:
            self.episode_returns.extend(r['episode_returns'])
            self.episode_lengths.extend(r['episode_lengths'])
            self.episode_successes.extend(r['episode_successes'])
        
        advantages, returns = compute_gae(
            result['rewards'], result['values'], result['dones'],
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
            result['states'], result['actions'], result['log_probs'],
            result['rewards'], result['dones'], result['values'],
            advantages, returns
        )
    
    def _collect_sequential(self, env_config, num_steps):
        # Fallback sequential implementation
        pass
    
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
                loss = policy_loss + self.args.vf_coef * value_loss - self.args.ent_coef * entropy
                
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
        
        print("🚀 Starting ADVANCED PPO training...\n")
        start_time = time.time()
        
        num_updates = self.args.total_timesteps // self.args.rollout_steps
        
        for update in range(num_updates):
            batch = self.collect_rollout(env_config, self.args.rollout_steps)
            self.update(batch)
            
            if (update + 1) % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                mean_return = np.mean(self.episode_returns) if self.episode_returns else 0
                std_return = np.std(self.episode_returns) if self.episode_returns else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                success_rate = np.mean(self.episode_successes) if self.episode_successes else 0
                
                print(f"Update {update+1:4d}/{num_updates} | "
                      f"Return: {mean_return:8.1f} ± {std_return:6.1f} | "
                      f"Length: {mean_length:5.1f} | "
                      f"Success: {success_rate*100:5.1f}% | "
                      f"Entropy: {np.mean(self.entropies[-100:]) if self.entropies else 0:5.3f} | "
                      f"Time: {elapsed/60:.1f}m")
            
            if (update + 1) % self.args.save_interval == 0:
                self.save_checkpoint(update + 1)
        
        self.save_checkpoint(num_updates, final=True)
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"Time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Final Success Rate: {np.mean(self.episode_successes)*100:.1f}%")
        print(f"Final Return: {np.mean(self.episode_returns):.1f}")
    
    def save_checkpoint(self, update, final=False):
        submission_dir = Path(f"submission_{self.args.agent_id}")
        submission_dir.mkdir(exist_ok=True)
        
        if final:
            torch.save(self.policy.state_dict(), submission_dir / "weights.pth")
            print(f"✓ Final model saved to {submission_dir}/weights.pth")
        else:
            checkpoint = {
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'update': update,
                'stats': {
                    'mean_return': np.mean(self.episode_returns) if self.episode_returns else 0,
                    'success_rate': np.mean(self.episode_successes) if self.episode_successes else 0,
                }
            }
            torch.save(checkpoint, submission_dir / f"checkpoint_update{update}.pth")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ADVANCED PPO for OBELIX')
    
    # Environment
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--agent_id", type=str, default="ppo_adv")
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
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    
    # ADVANCED FEATURES
    parser.add_argument("--success_bonus", type=float, default=200000, help="MASSIVE bonus for success")
    parser.add_argument("--stuck_penalty", type=float, default=100, help="Penalty for stuck/loop")
    parser.add_argument("--explore_temperature", type=float, default=1.5, help="High temp = explore")
    parser.add_argument("--exploit_temperature", type=float, default=0.5, help="Low temp = greedy")
    parser.add_argument("--explore_sensor_bonus", type=float, default=50, help="Finding box bonus")
    parser.add_argument("--explore_ir_bonus", type=float, default=100, help="Close to box bonus")
    parser.add_argument("--exploit_sensor_bonus", type=float, default=20, help="Staying close bonus")
    parser.add_argument("--exploit_ir_bonus", type=float, default=50, help="Maintaining contact bonus")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"ADVANCED PPO - Memory + Success Bonus + Phased Exploration")
    print(f"{'='*70}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Parallel Workers: {args.num_workers}")
    print(f"\n🎯 ADVANCED FEATURES:")
    print(f"  ✓ Success Bonus: +{args.success_bonus:,} (MASSIVE!)")
    print(f"  ✓ Loop Detection: -{args.stuck_penalty} penalty")
    print(f"  ✓ Phased Temperature: {args.explore_temperature} → {args.exploit_temperature}")
    print(f"  ✓ Short-term Memory: 4 state-action pairs")
    print(f"{'='*70}\n")
    
    trainer = PPOTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()
