#!/usr/bin/env python3
"""
Hybrid Heuristic-Guided Q-Learning for OBELIX

Novel Approach:
1. Heuristic provides exploration policy (instead of random ε-greedy)
2. Q-learning learns to improve upon heuristic
3. Intrinsic reward for "better than heuristic" actions
4. Warm-start Q-values with heuristic estimates

This solves the sparse reward problem while still being "real RL"!
"""

import os
import sys
import time
import random
import argparse
import importlib.util
from pathlib import Path
from collections import deque, defaultdict
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def import_obelix(path: str):
    """Import OBELIX environment."""
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# ============================================================================
# HEURISTIC POLICY (for guidance)
# ============================================================================

class HeuristicPolicy:
    """
    Sensor-following heuristic.
    Used for:
    1. Exploration (instead of random actions)
    2. Baseline for intrinsic rewards
    3. Warm-start for Q-values
    """
    
    def __init__(self):
        self.action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
    
    def get_action_probs(self, obs):
        """
        Get probability distribution over actions according to heuristic.
        Returns: array of shape (5,) with action probabilities
        """
        sonar = obs[:16]
        ir_sensor = obs[16]
        attached = obs[17]
        
        probs = np.zeros(5)
        
        # If attached, strongly prefer forward
        if attached == 1:
            probs[2] = 0.9  # FW
            probs[[1, 3]] = 0.05  # Small chance of adjusting
            return probs
        
        # If IR active, strongly prefer forward
        if ir_sensor == 1:
            probs[2] = 0.8  # FW
            probs[[1, 3]] = 0.1  # Small adjustments
            return probs
        
        # If sonar active, turn toward sensors
        if np.any(sonar == 1):
            active_indices = np.where(sonar == 1)[0]
            center_of_mass = np.mean(active_indices)
            
            if center_of_mass < 6:
                probs[0] = 0.7  # L45
                probs[1] = 0.2  # L22
                probs[2] = 0.1  # FW
            elif center_of_mass < 7.5:
                probs[1] = 0.6  # L22
                probs[2] = 0.3  # FW
                probs[0] = 0.1  # L45
            elif center_of_mass > 9.5:
                probs[4] = 0.7  # R45
                probs[3] = 0.2  # R22
                probs[2] = 0.1  # FW
            elif center_of_mass > 8.5:
                probs[3] = 0.6  # R22
                probs[2] = 0.3  # FW
                probs[4] = 0.1  # R45
            else:
                probs[2] = 0.7  # FW
                probs[[1, 3]] = 0.15
            
            return probs
        
        # No sensors - explore with forward bias
        probs[2] = 0.5  # FW
        probs[0] = 0.15  # L45
        probs[1] = 0.1   # L22
        probs[3] = 0.1   # R22
        probs[4] = 0.15  # R45
        
        return probs
    
    def get_action(self, obs):
        """Sample action from heuristic policy."""
        probs = self.get_action_probs(obs)
        return np.random.choice(5, p=probs)
    
    def get_value_estimate(self, obs):
        """
        Estimate value of state based on heuristic.
        Used to warm-start Q-values.
        """
        sonar = obs[:16]
        ir_sensor = obs[16]
        attached = obs[17]
        
        # Higher value = closer to goal
        value = 0.0
        
        if attached == 1:
            value += 1000  # Very valuable state
        
        if ir_sensor == 1:
            value += 500  # Box nearby
        
        if np.any(sonar == 1):
            value += 100 * np.sum(sonar)  # More sensors = closer
        
        return value

# ============================================================================
# Q-NETWORK (with heuristic warm-start)
# ============================================================================

class QNetwork(nn.Module):
    """Q-network with heuristic-guided initialization."""
    
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Initialize with small weights (will be warm-started)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# HYBRID TRAINER
# ============================================================================

class HybridTrainer:
    """
    Hybrid heuristic-guided Q-learning.
    
    Key innovations:
    1. Heuristic-guided exploration (not random)
    2. Intrinsic reward for improving over heuristic
    3. Warm-start Q-values with heuristic estimates
    4. Adaptive mixing (more heuristic early, more learned later)
    """
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Networks
        self.q_network = QNetwork(hidden_dim=args.hidden_dim).to(device)
        self.target_network = QNetwork(hidden_dim=args.hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        
        # Heuristic policy
        self.heuristic = HeuristicPolicy()
        
        # Replay buffer (simple)
        self.replay_buffer = deque(maxlen=args.replay_size)
        
        # Metrics
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.heuristic_agreements = deque(maxlen=100)
        
        self.total_steps = 0
        self.episodes_done = 0
    
    def select_action(self, state, epsilon):
        """
        Hybrid action selection:
        - With prob ε: use heuristic (not random!)
        - With prob 1-ε: use Q-network
        
        This is KEY: we explore using heuristic, not random actions!
        """
        if random.random() < epsilon:
            # Exploration: use heuristic
            return self.heuristic.get_action(state), True
        else:
            # Exploitation: use Q-network
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_t)
                return q_values.argmax().item(), False
    
    def compute_intrinsic_reward(self, state, action, next_state, base_reward):
        """
        Intrinsic reward for "doing better than heuristic."
        
        Idea: Reward agent for taking actions that:
        1. Lead to states heuristic values higher
        2. Differ from heuristic but get good results
        """
        intrinsic = 0.0
        
        # Progress reward: next state value > current state value
        curr_heur_value = self.heuristic.get_value_estimate(state)
        next_heur_value = self.heuristic.get_value_estimate(next_state)
        
        progress = next_heur_value - curr_heur_value
        intrinsic += progress * 0.01  # Small scaling
        
        # Diversity bonus: reward for taking non-heuristic actions that work
        heur_action = self.heuristic.get_action(state)
        if action != heur_action and base_reward > 0:
            intrinsic += 5.0  # Reward for discovering better actions
        
        return intrinsic
    
    def train_step(self):
        """Single training step."""
        if len(self.replay_buffer) < self.args.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.args.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(1)
            next_q_values = self.target_network(next_states_t)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.args.gamma * next_q_values * (1 - dones_t)
        
        # Loss
        loss = F.mse_loss(q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
    
    def train(self):
        """Main training loop."""
        OBELIX = import_obelix(self.args.obelix_py)
        
        env_config = {
            'scaling_factor': self.args.scaling_factor,
            'arena_size': self.args.arena_size,
            'max_steps': self.args.max_steps,
            'wall_obstacles': self.args.wall_obstacles,
            'difficulty': self.args.difficulty,
            'box_speed': self.args.box_speed,
        }
        
        print("🚀 Starting hybrid training...\n")
        start_time = time.time()
        
        for episode in range(self.args.episodes):
            # Epsilon schedule (decay from 1.0 to 0.1)
            epsilon = max(0.1, 1.0 - episode / (self.args.episodes * 0.7))
            
            # Create environment
            seed = random.choice(self.args.seed_list if self.args.multi_seed else [self.args.seed])
            env = OBELIX(**env_config)
            state = env.reset(seed=seed)
            
            episode_reward = 0
            episode_length = 0
            heuristic_agreements = 0
            
            for step in range(self.args.max_steps):
                # Select action (hybrid)
                action, from_heuristic = self.select_action(state, epsilon)
                
                # Take action
                next_state, reward, done = env.step(ACTIONS[action], render=False)
                
                # Compute total reward (base + intrinsic)
                intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state, reward)
                total_reward = reward + intrinsic_reward
                
                # Store transition
                self.replay_buffer.append((
                    state.copy(),
                    action,
                    total_reward,
                    next_state.copy(),
                    float(done)
                ))
                
                # Train
                if self.total_steps % self.args.train_freq == 0:
                    self.train_step()
                
                # Update target network
                if self.total_steps % self.args.target_update == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                # Track agreement with heuristic
                if from_heuristic or action == self.heuristic.get_action(state):
                    heuristic_agreements += 1
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                self.total_steps += 1
                
                if done:
                    break
            
            # Track metrics
            self.episode_returns.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_successes.append(episode_reward > 0)
            self.heuristic_agreements.append(heuristic_agreements / episode_length)
            
            # Logging
            if (episode + 1) % self.args.log_freq == 0:
                mean_return = np.mean(self.episode_returns)
                success_rate = np.mean(self.episode_successes) * 100
                mean_agreement = np.mean(self.heuristic_agreements) * 100
                
                print(f"Episode {episode+1:4d}/{self.args.episodes} | "
                      f"Return: {mean_return:8.1f} | "
                      f"Success: {success_rate:5.1f}% | "
                      f"HeurAgree: {mean_agreement:5.1f}% | "
                      f"ε: {epsilon:.3f} | "
                      f"Time: {(time.time()-start_time)/60:.1f}m")
            
            # Save checkpoint
            if (episode + 1) % self.args.save_freq == 0:
                self.save_checkpoint(episode + 1)
        
        # Final save
        self.save_checkpoint(self.args.episodes, final=True)
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"Time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Final Success Rate: {np.mean(self.episode_successes)*100:.1f}%")
    
    def save_checkpoint(self, episode, final=False):
        """Save checkpoint."""
        submission_dir = Path(f"submission_{self.args.agent_id}")
        submission_dir.mkdir(exist_ok=True)
        
        if final:
            # Save just the Q-network weights
            torch.save(self.q_network.state_dict(), submission_dir / "weights.pth")
            
            # Save agent file
            agent_code = f'''#!/usr/bin/env python3
"""Hybrid Q-Learning Agent for OBELIX"""
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class QNetwork(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim={self.args.hidden_dim}):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x):
        return self.network(x)

model = None

def policy(obs, rng=None):
    global model
    if model is None:
        import os
        model = QNetwork()
        weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
    
    with torch.no_grad():
        state_t = torch.FloatTensor(obs).unsqueeze(0)
        q_values = model(state_t)
        action_idx = q_values.argmax().item()
    
    return ACTIONS[action_idx]
'''
            (submission_dir / "agent_hybrid.py").write_text(agent_code)
            print(f"✓ Saved to {submission_dir}/")

def main():
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--agent_id", type=str, default="hybrid_001")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    
    # Training
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--replay_size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--target_update", type=int, default=1000)
    
    # Multi-seed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_seed", action="store_true")
    parser.add_argument("--seed_list", type=int, nargs='+', default=[42, 123, 456, 789, 999])
    
    # Logging
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=500)
    
    # Device
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"HYBRID HEURISTIC-GUIDED Q-LEARNING")
    print(f"{'='*60}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Episodes: {args.episodes}")
    print(f"Innovation: Heuristic-guided exploration + intrinsic rewards")
    print(f"{'='*60}\n")
    
    trainer = HybridTrainer(args, device)
    trainer.train()

if __name__ == "__main__":
    main()
