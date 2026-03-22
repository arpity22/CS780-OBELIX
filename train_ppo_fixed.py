#!/usr/bin/env python3
"""
CORRECTED ADAPTIVE PPO - All 10 Critical Issues Fixed

FIXES:
1. ✅ Use ALL worker data (not just results[0])
2. ✅ Remove probability hacking (use rewards only)
3. ✅ Reward scaling (divide by 1000)
4. ✅ Proper GAE bootstrap (use critic for mid-episode)
5. ✅ Distinguish timeout from true termination
6. ✅ Restore optimizer state with policy
7. ✅ Remove sensor-count progress bonus
8. ✅ Remove state-hashing curiosity
9. ✅ Fix entropy decay schedule (per timestep)
10. ✅ Fix success condition (reward >= 2000)
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

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ============================================================================
# WORKER (FIXED)
# ============================================================================

def parallel_rollout_worker(args_tuple):
    """
    Fixed worker:
    - No probability hacking (removed)
    - No curiosity hashing (removed)
    - No sensor-count progress (removed)
    - Reward scaling (divide by 1000)
    - Proper timeout detection
    """
    (obelix_path, env_config, policy_state, num_steps, seed, 
     config, hidden_dim) = args_tuple
    
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
    
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    timeouts = []  # FIX 5: Track timeouts separately
    episode_returns, episode_lengths, episode_successes = [], [], []
    
    episode_reward = 0
    episode_length = 0
    
    # FIX 2: No probability modification - just track actions for penalty
    action_memory = deque(maxlen=8)
    
    for step in range(num_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = policy(state_t)
            probs = F.softmax(logits, dim=-1)
            
            # FIX 2: NO probability hacking - sample from true policy
            dist = Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
        
        action_idx = action_idx.item()
        log_prob = log_prob.item()
        value = value.item()
        
        next_state, reward, done = env.step(ACTIONS[action_idx], render=False)
        
        # Check if timeout (FIX 5)
        is_timeout = (episode_length + 1 >= env_config['max_steps']) and done
        
        # REWARD SHAPING
        shaped_reward = reward
        
        # FIX 1: MASSIVE SUCCESS BONUS (kept, but will be scaled)
        if done and reward >= 2000:  # FIX 10: Proper success detection
            shaped_reward += config['success_bonus']
        
        # FIX 7: REMOVED sensor-count progress bonus
        # (Rely on env's built-in +1/+2/+5 rewards)
        
        # IR bonus
        if next_state[16] == 1:
            shaped_reward += config['ir_bonus']
        
        # Attachment bonus
        if next_state[17] == 1:
            shaped_reward += config['attachment_bonus']
        
        # FIX 2: Loop penalty via REWARD (not probability hacking)
        if len(action_memory) >= 8:
            unique_actions = len(set(action_memory))
            if unique_actions <= 2:  # Very repetitive
                shaped_reward -= config['loop_penalty']
        
        action_memory.append(action_idx)
        
        # FIX 3: REWARD SCALING - divide by 1000 for stable critic training
        scaled_reward = shaped_reward / 1000.0
        
        # Store
        states.append(state.copy())
        actions.append(action_idx)
        log_probs.append(log_prob)
        rewards.append(scaled_reward)  # FIX 3: Scaled reward
        dones.append(done)
        timeouts.append(is_timeout)  # FIX 5: Track timeouts
        values.append(value)
        
        episode_reward += reward  # Track UNSCALED for metrics
        episode_length += 1
        state = next_state
        
        if done:
            # FIX 10: Proper success detection (>= 2000, not > 0)
            episode_successes.append(reward >= 2000)
            episode_returns.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Reset
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
        'timeouts': np.array(timeouts),  # FIX 5
        'values': np.array(values),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes,
    }

# ============================================================================
# UTILS (FIXED)
# ============================================================================

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def compute_gae_fixed(rewards, values, dones, timeouts, next_value, gamma=0.99, gae_lambda=0.95):
    """
    FIX 4 & 5: Proper GAE with bootstrap and timeout handling.
    
    Args:
        next_value: The critic's value estimate for the state AFTER the last step
        timeouts: Boolean array indicating which dones are timeouts (not true terminals)
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # FIX 4: Use provided next_value instead of 0
            next_value_t = next_value
        else:
            next_value_t = values[t + 1]
        
        # FIX 5: If timeout, bootstrap. If true done, use 0.
        if dones[t]:
            if timeouts[t]:
                # Timeout: bootstrap from next state
                mask = 1.0
            else:
                # True termination: no future value
                mask = 0.0
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
# TRAINER (FIXED)
# ============================================================================

class FixedAdaptivePPOTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.policy = ActorCritic(18, 5, args.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr, eps=1e-5)
        
        # LR scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50
        )
        
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        # FIX 6: Save optimizer state too
        self.best_success_rate = 0.0
        self.best_policy_state = None
        self.best_optimizer_state = None  # FIX 6: Added
        self.updates_since_improvement = 0
        
        # FIX 9: Entropy decay per timestep (not per update)
        self.total_timesteps = 0
        self.current_ent_coef = args.ent_coef_start
        
        # Track which optimization stage we're in
        self.optimization_stage = 1  # 1 = success rate, 2 = efficiency
    
    def collect_rollout(self, env_config, num_steps):
        """
        FIX 1: Collect from ALL workers and concatenate.
        FIX 4: Bootstrap final value properly.
        """
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
                num_steps, seed, config, self.args.hidden_dim
            ))
        
        with Pool(processes=self.args.num_workers) as pool:
            results = pool.map(parallel_rollout_worker, worker_args)
        
        # FIX 1: CONCATENATE ALL WORKER DATA (not just results[0])
        all_states = np.concatenate([r['states'] for r in results])
        all_actions = np.concatenate([r['actions'] for r in results])
        all_log_probs = np.concatenate([r['log_probs'] for r in results])
        all_rewards = np.concatenate([r['rewards'] for r in results])
        all_dones = np.concatenate([r['dones'] for r in results])
        all_timeouts = np.concatenate([r['timeouts'] for r in results])
        all_values = np.concatenate([r['values'] for r in results])
        
        # Track episode metrics from all workers
        for r in results:
            self.episode_returns.extend(r['episode_returns'])
            self.episode_lengths.extend(r['episode_lengths'])
            self.episode_successes.extend(r['episode_successes'])
        
        # FIX 4: Get bootstrap value for final state
        # Check if last step is mid-episode
        if not all_dones[-1]:
            # Mid-episode: need to bootstrap
            final_state = all_states[-1]
            with torch.no_grad():
                final_state_t = torch.FloatTensor(final_state).unsqueeze(0).to(self.device)
                _, next_value = self.policy(final_state_t)
                next_value = next_value.item()
        else:
            # Episode ended naturally
            if all_timeouts[-1]:
                # Timeout: should bootstrap but we don't have next state
                # Approximate with final value
                next_value = all_values[-1]
            else:
                # True terminal: value = 0
                next_value = 0.0
        
        # FIX 4 & 5: Compute GAE with proper bootstrap
        advantages, returns = compute_gae_fixed(
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
                
                # Use current adaptive entropy coefficient
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
        
        print("🚀 Starting FIXED ADAPTIVE PPO training...\n")
        start_time = time.time()
        
        num_updates = self.args.total_timesteps // (self.args.rollout_steps * self.args.num_workers)
        
        for update in range(num_updates):
            batch = self.collect_rollout(env_config, self.args.rollout_steps)
            self.update(batch)
            
            # FIX 9: Entropy decay per TIMESTEP (not per update)
            self.total_timesteps += self.args.rollout_steps * self.args.num_workers
            progress = self.total_timesteps / self.args.total_timesteps
            self.current_ent_coef = self.args.ent_coef_start + progress * (self.args.ent_coef_end - self.args.ent_coef_start)
            self.current_ent_coef = max(self.args.ent_coef_end, self.current_ent_coef)
            
            # Check for improvement
            current_success = np.mean(self.episode_successes) if self.episode_successes else 0
            current_return = np.mean(self.episode_returns) if self.episode_returns else -999999
            
            # ADAPTIVE METRIC: Switch optimization target based on success rate
            if current_success >= 0.95:
                # Stage 2: High success - optimize for EFFICIENCY (maximize return)
                if self.optimization_stage == 1:
                    print(f"\n{'='*70}")
                    print(f"🎉 SUCCESS RATE ≥95%! Switching to EFFICIENCY optimization")
                    print(f"   Now optimizing: Episode Return (minimize steps/penalties)")
                    print(f"   Safety: Will revert if success drops below 85%")
                    print(f"{'='*70}\n")
                    self.optimization_stage = 2
                    # Reset best metric when switching stages
                    self.best_success_rate = current_return
                
                # SAFETY CHECK: Don't accept efficiency improvements that hurt success too much
                if current_success < 0.85:
                    # Success dropped too low - reject this as improvement
                    print(f"⚠️  Success dropped to {current_success*100:.1f}% - prioritizing success over efficiency")
                    improvement = False
                    # Switch back to success optimization if consistently low
                    if current_success < 0.80:
                        print(f"⚠️  Success < 80%! Reverting to SUCCESS optimization")
                        self.optimization_stage = 1
                        self.best_success_rate = 0.0  # Reset to start fresh
                        improvement = False
                else:
                    # Success is still high - can optimize for efficiency
                    optimization_metric = current_return
                    metric_name = "Return"
                    metric_value = current_return
                    
                    # At high success, compare returns (higher is better)
                    if optimization_metric > self.best_success_rate:  # Reusing variable name
                        improvement = True
                    else:
                        improvement = False
            else:
                # Stage 1: Low success - optimize for SUCCESS RATE
                optimization_metric = current_success
                metric_name = "Success"
                metric_value = current_success * 100
                
                # Compare success rates (higher is better)
                if optimization_metric > self.best_success_rate:
                    improvement = True
                else:
                    improvement = False
            
            if improvement:
                self.best_success_rate = optimization_metric  # Store best metric
                # FIX 6: Save BOTH policy and optimizer state
                self.best_policy_state = copy.deepcopy({k: v.cpu() for k, v in self.policy.state_dict().items()})
                self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                self.updates_since_improvement = 0
                print(f"🎯 NEW BEST {metric_name.upper()}! {metric_name}: {metric_value:.1f} | Success: {current_success*100:.1f}%")
            else:
                self.updates_since_improvement += 1
            
            # Anti-forgetting: Restore if stuck OR if success floor violated in efficiency mode
            should_restore = False
            restore_reason = ""
            
            if self.best_policy_state is not None:
                # Immediate restore if success drops too low during efficiency mode
                if self.optimization_stage == 2 and current_success < 0.85:
                    should_restore = True
                    restore_reason = f"success dropped to {current_success*100:.1f}% (floor: 85%)"
                # Regular restore if stuck for 100 updates
                elif self.updates_since_improvement > 100:
                    should_restore = True
                    restore_reason = "stuck for 100 updates"
            
            if should_restore:
                print(f"⚠️  RESTORING BEST POLICY ({restore_reason})")
                # FIX 6: Restore BOTH policy and optimizer
                self.policy.load_state_dict({k: v.to(self.device) for k, v in self.best_policy_state.items()})
                self.optimizer.load_state_dict(self.best_optimizer_state)
                self.updates_since_improvement = 0
                # Reset entropy to encourage exploration
                self.current_ent_coef = self.args.ent_coef_start
            
            # LR scheduling
            self.scheduler.step(current_success)
            
            if (update + 1) % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                mean_return = np.mean(self.episode_returns) if self.episode_returns else 0
                std_return = np.std(self.episode_returns) if self.episode_returns else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                stage_marker = "📈" if self.optimization_stage == 1 else "⚡"
                stage_name = "Success" if self.optimization_stage == 1 else "Efficiency"
                
                print(f"{stage_marker} Update {update+1:4d}/{num_updates} [{stage_name}] | "
                      f"Return: {mean_return:8.1f} ± {std_return:6.1f} | "
                      f"Length: {mean_length:5.1f} | "
                      f"Success: {current_success*100:5.1f}% | "
                      f"Entropy: {np.mean(self.entropies[-100:]) if self.entropies else 0:5.3f} | "
                      f"EntCoef: {self.current_ent_coef:.4f} | "
                      f"Steps: {self.total_timesteps:,} | "
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
            # Save best policy
            if self.best_policy_state is not None:
                torch.save(self.best_policy_state, submission_dir / "weights.pth")
                print(f"✓ BEST model saved (Success: {self.best_success_rate*100:.1f}%)")
            else:
                torch.save({k: v.cpu() for k, v in self.policy.state_dict().items()}, 
                          submission_dir / "weights.pth")
                print(f"✓ Final model saved")
        else:
            checkpoint = {
                'policy_state_dict': {k: v.cpu() for k, v in self.policy.state_dict().items()},
                'best_policy_state_dict': self.best_policy_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_optimizer_state_dict': self.best_optimizer_state,
                'update': update,
                'best_success_rate': self.best_success_rate,
                'current_ent_coef': self.current_ent_coef,
                'total_timesteps': self.total_timesteps,
            }
            torch.save(checkpoint, submission_dir / f"checkpoint_update{update}.pth")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FIXED ADAPTIVE PPO')
    
    # Environment
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--agent_id", type=str, default="ppo_fixed")
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
    
    # REWARDS (FIX 3: These are before scaling)
    parser.add_argument("--success_bonus", type=float, default=250000, 
                       help="Success bonus BEFORE scaling (will be divided by 1000)")
    parser.add_argument("--ir_bonus", type=float, default=50)
    parser.add_argument("--attachment_bonus", type=float, default=1000)
    parser.add_argument("--loop_penalty", type=float, default=100)
    
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
    print(f"FIXED ADAPTIVE PPO - All 10 Critical Issues Resolved")
    print(f"{'='*70}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Workers: {args.num_workers}")
    print(f"\n✅ FIXES APPLIED:")
    print(f"  1. Using ALL {args.num_workers} workers' data (not just 1)")
    print(f"  2. No probability hacking (pure PPO)")
    print(f"  3. Reward scaling (/1000 for stable critic)")
    print(f"  4. Proper GAE bootstrap (no mid-episode cutoff)")
    print(f"  5. Timeout vs termination distinction")
    print(f"  6. Optimizer state saved/restored with policy")
    print(f"  7. Removed sensor-count progress (use env rewards)")
    print(f"  8. Removed state-hashing curiosity (POMDP aliasing)")
    print(f"  9. Entropy decay per timestep (not per update)")
    print(f"  10. Success detection: reward >= 2000 (not > 0)")
    print(f"\n📊 REWARDS (before /1000 scaling):")
    print(f"  Success: +{args.success_bonus:,}")
    print(f"  IR: +{args.ir_bonus}")
    print(f"  Attachment: +{args.attachment_bonus}")
    print(f"  Loop: -{args.loop_penalty}")
    print(f"{'='*70}\n")
    
    trainer = FixedAdaptivePPOTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()
