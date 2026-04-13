#!/usr/bin/env python3
"""
Novel Approach #2: Automatic Curriculum Learning for OBELIX

Key Innovations:
1. Start with box VERY close (easy)
2. Automatically increase difficulty when agent succeeds
3. Dense reward shaping that decreases over curriculum
4. Success-based progression (not time-based)

This solves sparse reward problem by making early learning easier!
"""

import os
import sys
import time
import random
import argparse
import importlib.util
from pathlib import Path
from collections import deque
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# ============================================================================
# CURRICULUM MANAGER
# ============================================================================

class CurriculumManager:
    """
    Manages automatic curriculum progression.
    
    Starts with box close to robot, gradually increases distance.
    Increases difficulty only when agent is succeeding.
    """
    
    def __init__(self, initial_level=0, max_level=10):
        self.level = initial_level
        self.max_level = max_level
        
        # Success tracking for progression
        self.recent_successes = deque(maxlen=50)
        self.level_episode_count = 0
        
        # Curriculum parameters per level
        self.curriculum = self._build_curriculum()
    
    def _build_curriculum(self):
        """
        Build curriculum levels.
        
        Each level specifies:
        - box_distance: how far box spawns from robot
        - shaping_strength: how much reward shaping to use
        - exploration_bonus: bonus for exploration
        """
        curriculum = []
        
        for level in range(self.max_level + 1):
            # Distance increases from 50 to 400
            box_distance = 50 + level * 35
            
            # Shaping decreases from 10.0 to 0.0
            shaping_strength = max(0.0, 10.0 - level * 1.0)
            
            # Exploration bonus decreases
            exploration_bonus = max(0.0, 5.0 - level * 0.5)
            
            curriculum.append({
                'level': level,
                'box_distance': min(box_distance, 400),
                'shaping_strength': shaping_strength,
                'exploration_bonus': exploration_bonus,
                'min_episodes': 100,  # Minimum episodes before progressing
            })
        
        return curriculum
    
    def get_current_params(self):
        """Get parameters for current curriculum level."""
        return self.curriculum[self.level]
    
    def add_result(self, success):
        """Add episode result and check for progression."""
        self.recent_successes.append(success)
        self.level_episode_count += 1
        
        # Check if we should progress
        if self.should_progress():
            self.progress_level()
    
    def should_progress(self):
        """Check if we should move to next level."""
        params = self.curriculum[self.level]
        
        # Need minimum episodes at this level
        if self.level_episode_count < params['min_episodes']:
            return False
        
        # Need high success rate
        if len(self.recent_successes) < 30:
            return False
        
        success_rate = np.mean(list(self.recent_successes)[-30:])
        
        # Progression thresholds (easier levels need higher success)
        if self.level <= 3:
            threshold = 0.8
        elif self.level <= 6:
            threshold = 0.7
        else:
            threshold = 0.6
        
        return success_rate >= threshold and self.level < self.max_level
    
    def progress_level(self):
        """Progress to next curriculum level."""
        old_level = self.level
        self.level = min(self.level + 1, self.max_level)
        
        if self.level != old_level:
            print(f"\n{'='*60}")
            print(f"📈 CURRICULUM PROGRESSION: Level {old_level} → {self.level}")
            print(f"   Success rate: {np.mean(list(self.recent_successes)[-30:])*100:.1f}%")
            print(f"   New box distance: {self.curriculum[self.level]['box_distance']}")
            print(f"   New shaping strength: {self.curriculum[self.level]['shaping_strength']:.1f}")
            print(f"{'='*60}\n")
            
            self.level_episode_count = 0
            self.recent_successes.clear()

# ============================================================================
# MODIFIED ENVIRONMENT WRAPPER
# ============================================================================

class CurriculumOBELIX:
    """
    Wrapper around OBELIX that supports curriculum learning.
    
    Modifies initial box position based on curriculum level.
    """
    
    def __init__(self, obelix_class, base_config, curriculum_manager):
        self.obelix_class = obelix_class
        self.base_config = base_config
        self.curriculum = curriculum_manager
        self.env = None
    
    def reset(self, seed=None):
        """Reset with curriculum-adjusted difficulty."""
        # Create new environment
        self.env = self.obelix_class(**self.base_config)
        
        # Reset normally
        state = self.env.reset(seed=seed)
        
        # Modify box position based on curriculum
        params = self.curriculum.get_current_params()
        box_distance = params['box_distance']
        
        # Move box closer to robot
        if hasattr(self.env, 'box') and hasattr(self.env, 'robot'):
            robot_pos = np.array([self.env.robot.x, self.env.robot.y])
            robot_angle = self.env.robot.theta
            
            # Place box in front of robot at curriculum distance
            box_x = robot_pos[0] + box_distance * np.cos(robot_angle)
            box_y = robot_pos[1] + box_distance * np.sin(robot_angle)
            
            self.env.box.x = box_x
            self.env.box.y = box_y
            
            # Get new observation
            state = self.env.get_observation()
        
        return state
    
    def step(self, action, render=False):
        """Forward to underlying environment."""
        return self.env.step(action, render=render)
    
    def get_shaping_strength(self):
        """Get current shaping strength from curriculum."""
        return self.curriculum.get_current_params()['shaping_strength']
    
    def get_exploration_bonus(self):
        """Get current exploration bonus."""
        return self.curriculum.get_current_params()['exploration_bonus']

# ============================================================================
# ACTOR-CRITIC NETWORK
# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        shared_features = self.shared(x)
        logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return logits, value
    
    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value

# ============================================================================
# CURRICULUM PPO TRAINER
# ============================================================================

class CurriculumPPOTrainer:
    """PPO with automatic curriculum learning."""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Curriculum manager
        self.curriculum = CurriculumManager(
            initial_level=args.initial_level,
            max_level=args.max_level
        )
        
        # Policy network
        self.policy = ActorCritic(hidden_dim=args.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        
        # Metrics
        self.episode_returns = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        self.episodes_done = 0
    
    def compute_shaped_reward(self, base_reward, state, next_state, action_idx, shaping_strength):
        """
        Curriculum-adjusted reward shaping.
        
        Early levels: strong shaping
        Later levels: weak/no shaping
        """
        if shaping_strength == 0:
            return base_reward
        
        shaped = base_reward
        
        # Sensor activation bonus
        curr_sensors = np.sum(next_state[:16] == 1)
        prev_sensors = np.sum(state[:16] == 1)
        
        if curr_sensors > 0:
            shaped += 2.0 * shaping_strength * 0.1
        
        if next_state[16] == 1:  # IR sensor
            shaped += 5.0 * shaping_strength * 0.1
        
        # Approach bonus
        if curr_sensors > prev_sensors:
            shaped += 1.0 * shaping_strength * 0.1
        elif curr_sensors < prev_sensors and prev_sensors > 0:
            shaped -= 0.5 * shaping_strength * 0.1
        
        return shaped
    
    def collect_rollout(self, env, num_steps):
        """Collect rollout from curriculum environment."""
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        state = env.reset(seed=random.choice(self.args.seed_list if self.args.multi_seed else [self.args.seed]))
        
        episode_reward = 0
        episode_length = 0
        
        shaping_strength = env.get_shaping_strength()
        prev_state = state.copy()
        
        for step in range(num_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_idx, log_prob, _, value = self.policy.get_action_and_value(state_t)
            
            action_idx = action_idx.item()
            log_prob = log_prob.item()
            value = value.item()
            
            next_state, reward, done = env.step(ACTIONS[action_idx], render=False)
            
            # Apply curriculum shaping
            shaped_reward = self.compute_shaped_reward(
                reward, prev_state, next_state, action_idx, shaping_strength
            )
            
            states.append(state)
            actions.append(action_idx)
            log_probs.append(log_prob)
            rewards.append(shaped_reward)
            dones.append(done)
            values.append(value)
            
            episode_reward += reward
            episode_length += 1
            
            prev_state = state.copy()
            state = next_state
            
            if done:
                # Record result
                success = (reward > 0)
                self.episode_returns.append(episode_reward)
                self.episode_successes.append(success)
                self.episode_lengths.append(episode_length)
                
                # Update curriculum
                self.curriculum.add_result(success)
                
                # Reset
                state = env.reset(seed=random.choice(self.args.seed_list if self.args.multi_seed else [self.args.seed]))
                episode_reward = 0
                episode_length = 0
                shaping_strength = env.get_shaping_strength()
                prev_state = state.copy()
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
        }
    
    def train(self):
        """Main training loop with curriculum."""
        OBELIX = import_obelix(self.args.obelix_py)
        
        base_config = {
            'scaling_factor': self.args.scaling_factor,
            'arena_size': self.args.arena_size,
            'max_steps': self.args.max_steps,
            'wall_obstacles': self.args.wall_obstacles,
            'difficulty': self.args.difficulty,
            'box_speed': self.args.box_speed,
        }
        
        # Create curriculum environment
        env = CurriculumOBELIX(OBELIX, base_config, self.curriculum)
        
        print("🚀 Starting curriculum learning...\n")
        print(f"Initial level: {self.curriculum.level}")
        print(f"Initial box distance: {self.curriculum.get_current_params()['box_distance']}")
        print(f"Initial shaping: {self.curriculum.get_current_params()['shaping_strength']:.1f}\n")
        
        start_time = time.time()
        
        num_updates = self.args.total_timesteps // self.args.rollout_steps
        
        for update in range(num_updates):
            # Collect rollout
            batch = self.collect_rollout(env, self.args.rollout_steps)
            
            # Compute advantages (GAE)
            advantages, returns = self.compute_gae(batch)
            
            # PPO update
            self.ppo_update(batch, advantages, returns)
            
            # Logging
            if (update + 1) % 10 == 0:
                mean_return = np.mean(self.episode_returns) if self.episode_returns else 0
                success_rate = np.mean(self.episode_successes) * 100 if self.episode_successes else 0
                curr_params = self.curriculum.get_current_params()
                
                print(f"Update {update+1:4d}/{num_updates} | "
                      f"Level: {self.curriculum.level:2d} | "
                      f"Dist: {curr_params['box_distance']:3.0f} | "
                      f"Success: {success_rate:5.1f}% | "
                      f"Return: {mean_return:8.1f} | "
                      f"Time: {(time.time()-start_time)/60:.1f}m")
            
            # Save
            if (update + 1) % 100 == 0:
                self.save_checkpoint(update + 1)
        
        self.save_checkpoint(num_updates, final=True)
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"Final curriculum level: {self.curriculum.level}/{self.curriculum.max_level}")
        print(f"Final success rate: {np.mean(self.episode_successes)*100:.1f}%")
        print(f"{'='*60}")
    
    def compute_gae(self, batch, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation."""
        rewards = batch['rewards']
        values = batch['values']
        dones = batch['dones']
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_update(self, batch, advantages, returns):
        """PPO update."""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states_t = torch.FloatTensor(batch['states']).to(self.device)
        actions_t = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs_t = torch.FloatTensor(batch['log_probs']).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
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
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), returns_t[idx])
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
    
    def save_checkpoint(self, update, final=False):
        """Save checkpoint."""
        submission_dir = Path(f"submission_{self.args.agent_id}")
        submission_dir.mkdir(exist_ok=True)
        
        if final:
            torch.save(self.policy.state_dict(), submission_dir / "weights.pth")
            
            # Save curriculum progress
            progress = {
                'final_level': self.curriculum.level,
                'success_rate': float(np.mean(self.episode_successes)) if self.episode_successes else 0,
            }
            with open(submission_dir / "curriculum_progress.json", 'w') as f:
                json.dump(progress, f, indent=2)
            
            print(f"✓ Saved to {submission_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Curriculum Learning for OBELIX')
    
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--agent_id", type=str, default="curriculum_001")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    
    # Curriculum
    parser.add_argument("--initial_level", type=int, default=0)
    parser.add_argument("--max_level", type=int, default=10)
    
    # Training
    parser.add_argument("--total_timesteps", type=int, default=2000000)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    
    # Multi-seed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seed_list", type=int, nargs='+', default=[42, 123, 456, 789, 999])
    
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"AUTOMATIC CURRICULUM LEARNING FOR OBELIX")
    print(f"{'='*60}")
    print(f"Innovation: Box distance increases as agent improves")
    print(f"Levels: 0 → {args.max_level} (automatic progression)")
    print(f"{'='*60}\n")
    
    trainer = CurriculumPPOTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()
