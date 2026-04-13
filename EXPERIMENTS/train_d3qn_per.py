"""Offline trainer: D3QN-PER with comprehensive diagnostic analysis.

This implementation combines three powerful improvements over vanilla DQN:
1. Dueling architecture: separate value and advantage streams
2. Double DQN: reduced overestimation bias
3. Prioritized Experience Replay: learn more from surprising transitions

PLUS extensive diagnostic tracking and visualization to understand agent behavior!

OPTIMIZED FOR BOTH CPU AND GPU with automatic device detection.
PARALLEL ENVIRONMENT EXECUTION for 4-8x speedup on multi-core CPUs.

Run locally to create weights.pth, then submit agent_d3qn_per.py + weights.pth.

Example:
  python train_d3qn_per.py --obelix_py ./obelix.py --agent_id 001 --episodes 3000 --difficulty 0 --wall_obstacles --num_envs 8
"""

from __future__ import annotations
import argparse
import random
import os
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Dict, Any
from pathlib import Path
from multiprocessing import Pool, cpu_count
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for efficiency
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Device configuration for CPU/GPU optimization
def get_device():
    """Detect and configure best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        # Optimize CPU threading
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"💻 Using CPU with {torch.get_num_threads()} threads")
    return device

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_NAMES = {0: "L45", 1: "L22", 2: "FW", 3: "R22", 4: "R45"}

# Worker function for parallel environment execution
def run_episode_worker(args):
    """Worker function to run a single episode in parallel.
    
    This runs in a separate process to avoid Python GIL.
    """
    obelix_module_path, episode_config, policy_fn_data, reward_shaping_config = args
    
    # Import OBELIX in worker process
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX
    
    # Create environment
    env = OBELIX(**episode_config['env_params'])
    state = env.reset(seed=episode_config['seed'])
    
    # Unpack policy network state
    network_state_dict, device_str, hidden_dim = policy_fn_data
    
    # Recreate network in worker
    import torch
    import torch.nn as nn
    
    class DuelingDQN(nn.Module):
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
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, n_actions),
            )
        
        def forward(self, x):
            features = self.feature(x)
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
            return q_values
    
    # Load network
    network = DuelingDQN(hidden_dim=hidden_dim)
    network.load_state_dict(network_state_dict)
    network.eval()
    
    # Run episode
    episode_data = {
        'transitions': [],
        'rewards': [],
        'actions': [],
        'q_values': [],
        'states': [],
        'success': False,
        'total_return': 0.0,
        'length': 0
    }
    
    epsilon = episode_config['epsilon']
    max_steps = episode_config['max_steps']
    
    for step in range(max_steps):
        # Get Q-values
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = network(state_t).squeeze(0).numpy()
        
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(5)
        else:
            action_idx = int(np.argmax(q_vals))
        
        # Step
        next_state, reward, done = env.step(ACTIONS[action_idx], render=False)
        
        # REWARD SHAPING (if enabled)
        base_reward = reward
        shaped_reward = reward
        
        if reward_shaping_config['enabled']:
            # 1. Bonus for moving forward
            if action_idx == 2:  # FW action
                shaped_reward += reward_shaping_config['forward_bonus']
            
            # 2. Bonus for ANY sensor activation
            if np.any(next_state[:16] == 1):  # Any sonar sensor active
                shaped_reward += reward_shaping_config['sensor_bonus']
            
            # 3. Extra bonus for IR sensor
            if next_state[16] == 1:  # IR sensor
                shaped_reward += reward_shaping_config['ir_bonus']
        
        # Store
        episode_data['transitions'].append({
            's': state.copy(),
            'a': action_idx,
            'r': float(shaped_reward),  # Use shaped reward
            's2': next_state.copy(),
            'done': bool(done)
        })
        episode_data['rewards'].append(base_reward)  # Track base reward for metrics
        episode_data['actions'].append(action_idx)
        episode_data['q_values'].append(q_vals.copy())
        episode_data['states'].append(state.copy())
        episode_data['total_return'] += reward
        episode_data['length'] += 1
        
        state = next_state
        
        if done:
            episode_data['success'] = reward > 0
            break
    
    return episode_data

class ParallelEnvironmentRunner:
    """Runs multiple environments in parallel to collect experience faster."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = min(num_workers, cpu_count())
        self.executor = None
        print(f"🔧 Parallel runner using {self.num_workers} workers")
    
    def run_episodes_parallel(self, obelix_path: str, network_state: dict, 
                            episode_configs: List[dict], policy_data: tuple,
                            reward_shaping_config: dict) -> List[dict]:
        """Run multiple episodes in parallel."""
        
        # Prepare arguments for workers
        worker_args = [
            (obelix_path, config, policy_data, reward_shaping_config)
            for config in episode_configs
        ]
        
        # Run in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(run_episode_worker, worker_args)
        
        return results
    
    def shutdown(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown()

class BatchedExperienceCollector:
    """Collects experience from multiple parallel episodes efficiently."""
    
    def __init__(self, num_parallel: int = 4):
        self.num_parallel = num_parallel
        self.runner = ParallelEnvironmentRunner(num_workers=num_parallel)
    
    def collect_batch(self, obelix_path: str, network, env_params: dict,
                     seeds: List[int], epsilon: float, max_steps: int,
                     hidden_dim: int, device, reward_shaping_config: dict) -> List[dict]:
        """Collect a batch of episodes in parallel."""
        
        # Prepare network state for serialization
        network_state = {k: v.cpu() for k, v in network.state_dict().items()}
        policy_data = (network_state, str(device), hidden_dim)
        
        # Prepare episode configurations
        episode_configs = []
        for seed in seeds:
            episode_configs.append({
                'env_params': env_params,
                'seed': seed,
                'epsilon': epsilon,
                'max_steps': max_steps
            })
        
        # Run in parallel
        return self.runner.run_episodes_parallel(obelix_path, network_state, 
                                                episode_configs, policy_data,
                                                reward_shaping_config)
    
    def shutdown(self):
        self.runner.shutdown()

class DuelingDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams.
    
    Optimized for both CPU and GPU with proper initialization and efficient operations.
    """
    
    def __init__(self, in_dim: int = 18, n_actions: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor with LayerNorm for stability
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Better than BatchNorm for RL
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Value stream: V(s) - how good is this state?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        
        # Advantage stream: A(s,a) - how much better is each action?
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )
        
        # Initialize weights properly (Xavier for better gradient flow)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization for stable training."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns Q(s,a) for all actions."""
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages with mean normalization
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class PrioritizedReplay:
    """Prioritized Experience Replay Buffer - optimized for CPU/GPU.
    
    Samples transitions with probability proportional to their TD error.
    Transitions with high TD error (surprising) are sampled more frequently,
    leading to more efficient learning.
    
    Uses importance sampling weights to correct for bias introduced by
    non-uniform sampling.
    
    Optimized with vectorized numpy operations for speed.
    """
    
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full)
        self.beta = beta_start  # Importance sampling weight (annealed to 1)
        self.beta_start = beta_start
        
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, transition: Transition):
        """Add transition with maximum priority (will be updated after training)."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        # Set priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Sample batch with prioritized sampling and importance weights - vectorized for speed."""
        
        buffer_size = len(self.buffer)
        
        # Convert priorities to probabilities (vectorized)
        priorities = self.priorities[:buffer_size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(buffer_size, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights (vectorized)
        weights = (buffer_size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        # Extract transitions (batch operation)
        samples = [self.buffer[i] for i in indices]
        
        # Stack arrays efficiently
        s = np.stack([t.s for t in samples], axis=0).astype(np.float32)
        a = np.array([t.a for t in samples], dtype=np.int64)
        r = np.array([t.r for t in samples], dtype=np.float32)
        s2 = np.stack([t.s2 for t in samples], axis=0).astype(np.float32)
        d = np.array([t.done for t in samples], dtype=np.float32)
        
        return s, a, r, s2, d, weights.astype(np.float32), list(indices)
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors - vectorized."""
        # Vectorized priority update
        new_priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, new_priorities.max())
    
    def anneal_beta(self, step: int, total_steps: int):
        """Linearly anneal beta from beta_start to 1.0."""
        fraction = min(1.0, step / total_steps)
        self.beta = self.beta_start + (1.0 - self.beta_start) * fraction
    
    def __len__(self):
        return len(self.buffer)

@dataclass
class EpisodeData:
    """Store detailed per-episode data for analysis."""
    episode_num: int
    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    q_values: List[np.ndarray] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    success: bool = False
    total_return: float = 0.0
    length: int = 0

class TrainingMetrics:
    """Comprehensive tracking of training metrics for deep RL diagnostics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Basic training metrics
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.avg_q_values: List[float] = []
        self.max_q_values: List[float] = []
        self.min_q_values: List[float] = []
        self.q_value_std: List[float] = []
        self.epsilons: List[float] = []
        self.success_rate: List[float] = []
        
        # Action distribution tracking
        self.action_counts: Dict[int, int] = defaultdict(int)
        self.action_counts_per_episode: List[Dict[int, int]] = []
        self.greedy_action_counts: Dict[int, int] = defaultdict(int)
        self.random_action_counts: Dict[int, int] = defaultdict(int)
        
        # Per-action performance
        self.action_rewards: Dict[int, List[float]] = defaultdict(list)
        self.action_q_values: Dict[int, List[float]] = defaultdict(list)
        
        # Reward analysis
        self.positive_rewards: List[float] = []
        self.negative_rewards: List[float] = []
        self.zero_rewards: List[int] = []
        self.reward_per_step: List[float] = []
        
        # Learning dynamics
        self.gradient_norms: List[float] = []
        self.td_errors_mean: List[float] = []
        self.td_errors_max: List[float] = []
        self.replay_beta_values: List[float] = []
        
        # Episode-level details (store last N for detailed analysis)
        self.detailed_episodes: Deque[EpisodeData] = deque(maxlen=50)
        
        # State coverage (discretized for visualization)
        self.state_visits: Dict[tuple, int] = defaultdict(int)
        
        # Per-episode tracking
        self.current_ep_data: EpisodeData = None
        self.successful_episodes = deque(maxlen=window_size)
        
        # Training phases (for analysis)
        self.training_step_to_episode: List[int] = []
        
    def start_episode(self, episode_num: int):
        self.current_ep_data = EpisodeData(episode_num=episode_num)
    
    def step(self, reward: float, action: int, q_values: np.ndarray, 
             state: np.ndarray, is_greedy: bool):
        """Record a single step."""
        self.current_ep_data.rewards.append(reward)
        self.current_ep_data.actions.append(action)
        self.current_ep_data.q_values.append(q_values.copy())
        self.current_ep_data.states.append(state.copy())
        
        # Track action usage
        self.action_counts[action] += 1
        if is_greedy:
            self.greedy_action_counts[action] += 1
        else:
            self.random_action_counts[action] += 1
        
        # Track rewards
        self.reward_per_step.append(reward)
        if reward > 0:
            self.positive_rewards.append(reward)
        elif reward < 0:
            self.negative_rewards.append(reward)
        else:
            self.zero_rewards.append(1)
        
        # Track per-action performance
        self.action_rewards[action].append(reward)
        self.action_q_values[action].append(np.max(q_values))
        
        # Discretize state for coverage tracking (first 6 sensors)
        state_key = tuple(np.round(state[:6], 1))
        self.state_visits[state_key] += 1
    
    def end_episode(self, success: bool):
        """Finalize episode data."""
        self.current_ep_data.success = success
        self.current_ep_data.total_return = sum(self.current_ep_data.rewards)
        self.current_ep_data.length = len(self.current_ep_data.rewards)
        
        # Store detailed episode
        self.detailed_episodes.append(self.current_ep_data)
        
        # Update aggregate metrics
        self.episode_returns.append(self.current_ep_data.total_return)
        self.episode_lengths.append(self.current_ep_data.length)
        self.successful_episodes.append(1.0 if success else 0.0)
        
        # Action distribution for this episode
        ep_action_counts = defaultdict(int)
        for a in self.current_ep_data.actions:
            ep_action_counts[a] += 1
        self.action_counts_per_episode.append(dict(ep_action_counts))
        
        # Compute rolling success rate
        if len(self.successful_episodes) > 0:
            self.success_rate.append(np.mean(self.successful_episodes))
    
    def log_training(self, loss: float, avg_q: float, epsilon: float, 
                    gradient_norm: float = None, td_errors: np.ndarray = None,
                    q_values_batch: torch.Tensor = None, replay_beta: float = None,
                    training_step: int = None, episode_num: int = None):
        """Log training step metrics."""
        self.losses.append(loss)
        self.avg_q_values.append(avg_q)
        self.epsilons.append(epsilon)
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        if td_errors is not None:
            self.td_errors_mean.append(np.mean(np.abs(td_errors)))
            self.td_errors_max.append(np.max(np.abs(td_errors)))
        
        if q_values_batch is not None:
            with torch.no_grad():
                self.max_q_values.append(q_values_batch.max().item())
                self.min_q_values.append(q_values_batch.min().item())
                self.q_value_std.append(q_values_batch.std().item())
        
        if replay_beta is not None:
            self.replay_beta_values.append(replay_beta)
        
        if training_step is not None and episode_num is not None:
            self.training_step_to_episode.append(episode_num)
    
    def get_recent_stats(self) -> dict:
        """Get statistics from recent episodes."""
        n = min(self.window_size, len(self.episode_returns))
        if n == 0:
            return {}
        
        recent_returns = self.episode_returns[-n:]
        recent_lengths = self.episode_lengths[-n:]
        recent_success = self.success_rate[-1] if self.success_rate else 0.0
        
        return {
            'mean_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'mean_length': np.mean(recent_lengths),
            'success_rate': recent_success,
        }

def import_obelix(obelix_py: str):
    """Dynamically import OBELIX environment."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def plot_action_diagnostics(metrics: TrainingMetrics, save_dir: Path):
    """Plot comprehensive action selection diagnostics."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Overall action distribution
    ax1 = fig.add_subplot(gs[0, 0])
    total_actions = sum(metrics.action_counts.values())
    if total_actions > 0:
        actions = sorted(metrics.action_counts.keys())
        counts = [metrics.action_counts[a] for a in actions]
        percentages = [c/total_actions*100 for c in counts]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
        bars = ax1.bar([ACTION_NAMES[a] for a in actions], percentages, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add ideal uniform line
        ideal = 100 / len(actions)
        ax1.axhline(ideal, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Uniform ({ideal:.1f}%)')
        
        ax1.set_ylabel('Percentage of Total Actions', fontsize=11)
        ax1.set_title('Overall Action Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Greedy vs Random action distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics.greedy_action_counts and metrics.random_action_counts:
        actions = sorted(set(list(metrics.greedy_action_counts.keys()) + list(metrics.random_action_counts.keys())))
        greedy = [metrics.greedy_action_counts.get(a, 0) for a in actions]
        random_acts = [metrics.random_action_counts.get(a, 0) for a in actions]
        
        x = np.arange(len(actions))
        width = 0.35
        
        ax2.bar(x - width/2, greedy, width, label='Greedy (Policy)', color='green', alpha=0.8, edgecolor='black')
        ax2.bar(x + width/2, random_acts, width, label='Random (Explore)', color='orange', alpha=0.8, edgecolor='black')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([ACTION_NAMES[a] for a in actions])
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Greedy vs Random Action Selection', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Action distribution over time (heatmap)
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics.action_counts_per_episode:
        # Create matrix: episodes x actions
        n_episodes = len(metrics.action_counts_per_episode)
        n_actions = 5
        action_matrix = np.zeros((n_episodes, n_actions))
        
        for ep_idx, ep_actions in enumerate(metrics.action_counts_per_episode):
            total_ep_actions = sum(ep_actions.values())
            if total_ep_actions > 0:
                for action_idx in range(n_actions):
                    action_matrix[ep_idx, action_idx] = ep_actions.get(action_idx, 0) / total_ep_actions * 100
        
        # Sample every Nth episode for readability
        sample_rate = max(1, n_episodes // 50)
        sampled_matrix = action_matrix[::sample_rate, :]
        
        im = ax3.imshow(sampled_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_yticks(range(n_actions))
        ax3.set_yticklabels([ACTION_NAMES[i] for i in range(n_actions)])
        ax3.set_xlabel('Episode', fontsize=11)
        ax3.set_title('Action Usage Evolution (%)', fontsize=13, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('% of Episode', rotation=270, labelpad=20)
    
    # 4. Per-action average reward
    ax4 = fig.add_subplot(gs[1, 0])
    if metrics.action_rewards:
        actions = sorted(metrics.action_rewards.keys())
        avg_rewards = [np.mean(metrics.action_rewards[a]) if metrics.action_rewards[a] else 0 
                      for a in actions]
        std_rewards = [np.std(metrics.action_rewards[a]) if len(metrics.action_rewards[a]) > 1 else 0 
                      for a in actions]
        
        colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in avg_rewards]
        bars = ax4.bar([ACTION_NAMES[a] for a in actions], avg_rewards, 
                      yerr=std_rewards, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.set_ylabel('Average Reward', fontsize=11)
        ax4.set_title('Per-Action Average Reward ± Std', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Per-action average Q-value
    ax5 = fig.add_subplot(gs[1, 1])
    if metrics.action_q_values:
        actions = sorted(metrics.action_q_values.keys())
        avg_q = [np.mean(metrics.action_q_values[a]) if metrics.action_q_values[a] else 0 
                for a in actions]
        std_q = [np.std(metrics.action_q_values[a]) if len(metrics.action_q_values[a]) > 1 else 0 
                for a in actions]
        
        bars = ax5.bar([ACTION_NAMES[a] for a in actions], avg_q, 
                      yerr=std_q, capsize=5, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax5.set_ylabel('Average Q-value', fontsize=11)
        ax5.set_title('Per-Action Average Q-value ± Std', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Action transition matrix (which actions follow which)
    ax6 = fig.add_subplot(gs[1, 2])
    if metrics.detailed_episodes:
        # Build transition matrix
        transition_matrix = np.zeros((5, 5))
        for ep_data in metrics.detailed_episodes:
            for i in range(len(ep_data.actions) - 1):
                current_action = ep_data.actions[i]
                next_action = ep_data.actions[i + 1]
                transition_matrix[current_action, next_action] += 1
        
        # Normalize by row
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums * 100
        
        im = ax6.imshow(transition_matrix, cmap='Blues', interpolation='nearest')
        ax6.set_xticks(range(5))
        ax6.set_yticks(range(5))
        ax6.set_xticklabels([ACTION_NAMES[i] for i in range(5)], rotation=45)
        ax6.set_yticklabels([ACTION_NAMES[i] for i in range(5)])
        ax6.set_xlabel('Next Action', fontsize=11)
        ax6.set_ylabel('Current Action', fontsize=11)
        ax6.set_title('Action Transition Matrix (%)', fontsize=13, fontweight='bold')
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                text = ax6.text(j, i, f'{transition_matrix[i, j]:.0f}',
                              ha="center", va="center", color="black" if transition_matrix[i, j] < 50 else "white",
                              fontsize=9, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax6)
        cbar.set_label('Probability (%)', rotation=270, labelpad=20)
    
    # 7. Action sequence length (how many times repeated)
    ax7 = fig.add_subplot(gs[2, 0])
    if metrics.detailed_episodes:
        repeat_lengths = []
        for ep_data in metrics.detailed_episodes:
            if not ep_data.actions:
                continue
            current_action = ep_data.actions[0]
            repeat_count = 1
            for action in ep_data.actions[1:]:
                if action == current_action:
                    repeat_count += 1
                else:
                    repeat_lengths.append(repeat_count)
                    current_action = action
                    repeat_count = 1
            repeat_lengths.append(repeat_count)
        
        if repeat_lengths:
            ax7.hist(repeat_lengths, bins=range(1, max(repeat_lengths)+2), 
                    color='purple', alpha=0.7, edgecolor='black', linewidth=1)
            ax7.set_xlabel('Consecutive Repeats', fontsize=11)
            ax7.set_ylabel('Frequency', fontsize=11)
            ax7.set_title('Action Persistence Distribution', fontsize=13, fontweight='bold')
            ax7.axvline(np.mean(repeat_lengths), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(repeat_lengths):.1f}')
            ax7.legend()
            ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Early vs Late training action preference
    ax8 = fig.add_subplot(gs[2, 1])
    if len(metrics.action_counts_per_episode) > 100:
        n_early = 50
        n_late = 50
        
        early_actions = defaultdict(int)
        late_actions = defaultdict(int)
        
        for ep_actions in metrics.action_counts_per_episode[:n_early]:
            for action, count in ep_actions.items():
                early_actions[action] += count
        
        for ep_actions in metrics.action_counts_per_episode[-n_late:]:
            for action, count in ep_actions.items():
                late_actions[action] += count
        
        actions = range(5)
        early_pcts = [early_actions.get(a, 0) / sum(early_actions.values()) * 100 if sum(early_actions.values()) > 0 else 0 
                     for a in actions]
        late_pcts = [late_actions.get(a, 0) / sum(late_actions.values()) * 100 if sum(late_actions.values()) > 0 else 0 
                    for a in actions]
        
        x = np.arange(len(actions))
        width = 0.35
        
        ax8.bar(x - width/2, early_pcts, width, label=f'Early (Ep 1-{n_early})', 
               color='lightblue', alpha=0.8, edgecolor='black')
        ax8.bar(x + width/2, late_pcts, width, label=f'Late (Last {n_late} Ep)', 
               color='darkblue', alpha=0.8, edgecolor='black')
        
        ax8.set_xticks(x)
        ax8.set_xticklabels([ACTION_NAMES[i] for i in actions])
        ax8.set_ylabel('Percentage', fontsize=11)
        ax8.set_title('Early vs Late Training Action Preference', fontsize=13, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Action diversity over time (entropy)
    ax9 = fig.add_subplot(gs[2, 2])
    if metrics.action_counts_per_episode:
        entropies = []
        window = 20
        for i in range(window, len(metrics.action_counts_per_episode)):
            combined_counts = defaultdict(int)
            for ep_actions in metrics.action_counts_per_episode[i-window:i]:
                for action, count in ep_actions.items():
                    combined_counts[action] += count
            
            total = sum(combined_counts.values())
            if total > 0:
                probs = [combined_counts.get(a, 0) / total for a in range(5)]
                probs = [p for p in probs if p > 0]
                entropy = -sum(p * np.log(p) for p in probs)
                entropies.append(entropy)
        
        if entropies:
            ax9.plot(range(window, len(metrics.action_counts_per_episode)), entropies, 
                    color='green', linewidth=2)
            max_entropy = np.log(5)  # Maximum entropy for 5 actions
            ax9.axhline(max_entropy, color='red', linestyle='--', linewidth=2, 
                       alpha=0.7, label=f'Max Entropy ({max_entropy:.2f})')
            ax9.set_xlabel('Episode', fontsize=11)
            ax9.set_ylabel('Action Entropy', fontsize=11)
            ax9.set_title(f'Action Diversity Over Time ({window}-ep window)', fontsize=13, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            ax9.set_ylim([0, max_entropy * 1.1])
    
    plt.suptitle('Action Selection Diagnostics', fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = save_dir / 'action_diagnostics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved action diagnostics to {plot_path}")

def plot_reward_diagnostics(metrics: TrainingMetrics, save_dir: Path):
    """Plot comprehensive reward analysis."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Reward distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if metrics.reward_per_step:
        rewards = np.array(metrics.reward_per_step)
        ax1.hist(rewards, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(rewards):.3f}')
        ax1.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(rewards):.3f}')
        ax1.set_xlabel('Reward Value', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Reward Distribution (All Steps)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
    
    # 2. Reward composition (positive/negative/zero)
    ax2 = fig.add_subplot(gs[0, 1])
    n_positive = len(metrics.positive_rewards)
    n_negative = len(metrics.negative_rewards)
    n_zero = sum(metrics.zero_rewards)
    total = n_positive + n_negative + n_zero
    
    if total > 0:
        sizes = [n_positive, n_negative, n_zero]
        labels = [f'Positive\n({n_positive}, {n_positive/total*100:.1f}%)',
                 f'Negative\n({n_negative}, {n_negative/total*100:.1f}%)',
                 f'Zero\n({n_zero}, {n_zero/total*100:.1f}%)']
        colors = ['green', 'red', 'gray']
        explode = (0.05, 0.05, 0.05)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax2.set_title('Reward Composition', fontsize=13, fontweight='bold')
    
    # 3. Reward per step over training
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics.reward_per_step:
        window = 1000
        if len(metrics.reward_per_step) >= window:
            smoothed = np.convolve(metrics.reward_per_step, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(metrics.reward_per_step)), smoothed, 
                    color='blue', linewidth=2)
            ax3.set_xlabel('Training Step', fontsize=11)
            ax3.set_ylabel('Avg Reward (per step)', fontsize=11)
            ax3.set_title(f'Reward Per Step ({window}-step MA)', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # 4. Episode return progression
    ax4 = fig.add_subplot(gs[1, 0])
    if metrics.episode_returns:
        episodes = np.arange(len(metrics.episode_returns))
        returns = np.array(metrics.episode_returns)
        
        # Plot raw returns
        ax4.scatter(episodes, returns, alpha=0.3, s=10, color='blue', label='Raw')
        
        # Rolling stats
        window = 50
        if len(returns) >= window:
            rolling_mean = np.convolve(returns, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(returns[max(0, i-window):i+1]) 
                                   for i in range(window-1, len(returns))])
            
            x = np.arange(window-1, len(returns))
            ax4.plot(x, rolling_mean, color='red', linewidth=2, label=f'{window}-ep MA')
            ax4.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std,
                           alpha=0.2, color='red', label='±1 STD')
        
        ax4.set_xlabel('Episode', fontsize=11)
        ax4.set_ylabel('Episode Return', fontsize=11)
        ax4.set_title('Episode Return Progression', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Reward timing within episodes (early vs late rewards)
    ax5 = fig.add_subplot(gs[1, 1])
    if metrics.detailed_episodes:
        early_rewards = []
        late_rewards = []
        
        for ep_data in metrics.detailed_episodes:
            if len(ep_data.rewards) > 10:
                mid = len(ep_data.rewards) // 2
                early_rewards.extend(ep_data.rewards[:mid])
                late_rewards.extend(ep_data.rewards[mid:])
        
        if early_rewards and late_rewards:
            data = [early_rewards, late_rewards]
            labels = ['Early\n(1st half)', 'Late\n(2nd half)']
            
            bp = ax5.boxplot(data, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
            
            ax5.set_ylabel('Reward Value', fontsize=11)
            ax5.set_title('Reward Timing: Early vs Late Episode', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # 6. Cumulative reward curves (recent episodes)
    ax6 = fig.add_subplot(gs[1, 2])
    if metrics.detailed_episodes:
        n_plot = min(10, len(metrics.detailed_episodes))
        cmap = plt.cm.viridis(np.linspace(0, 1, n_plot))
        
        for idx, ep_data in enumerate(list(metrics.detailed_episodes)[-n_plot:]):
            cumulative = np.cumsum(ep_data.rewards)
            ax6.plot(cumulative, color=cmap[idx], alpha=0.7, linewidth=2)
        
        ax6.set_xlabel('Step within Episode', fontsize=11)
        ax6.set_ylabel('Cumulative Reward', fontsize=11)
        ax6.set_title(f'Cumulative Reward Curves (Last {n_plot} Episodes)', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=len(metrics.detailed_episodes)-n_plot, 
                                                    vmax=len(metrics.detailed_episodes)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax6)
        cbar.set_label('Episode Number', rotation=270, labelpad=20)
    
    plt.suptitle('Reward Analysis Diagnostics', fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = save_dir / 'reward_diagnostics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved reward diagnostics to {plot_path}")

def plot_qvalue_diagnostics(metrics: TrainingMetrics, save_dir: Path):
    """Plot Q-value analysis for detecting overestimation and instability."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Q-value statistics over time
    ax1 = fig.add_subplot(gs[0, 0])
    if metrics.avg_q_values:
        steps = np.arange(len(metrics.avg_q_values))
        ax1.plot(steps, metrics.avg_q_values, label='Mean Q', linewidth=1.5, color='blue', alpha=0.7)
        
        if metrics.max_q_values:
            ax1.plot(steps, metrics.max_q_values, label='Max Q', linewidth=1.5, color='red', alpha=0.7)
        if metrics.min_q_values:
            ax1.plot(steps, metrics.min_q_values, label='Min Q', linewidth=1.5, color='green', alpha=0.7)
        
        ax1.set_xlabel('Training Step', fontsize=11)
        ax1.set_ylabel('Q-value', fontsize=11)
        ax1.set_title('Q-value Statistics Over Training', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Q-value spread (std) over time - detects overestimation
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics.q_value_std:
        window = 500
        steps = np.arange(len(metrics.q_value_std))
        ax2.plot(steps, metrics.q_value_std, alpha=0.5, linewidth=0.5, color='purple')
        
        if len(metrics.q_value_std) >= window:
            smoothed = np.convolve(metrics.q_value_std, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(metrics.q_value_std)), smoothed, 
                    color='darkblue', linewidth=2, label=f'{window}-step MA')
        
        ax2.set_xlabel('Training Step', fontsize=11)
        ax2.set_ylabel('Q-value Std Dev', fontsize=11)
        ax2.set_title('Q-value Spread (Overestimation Indicator)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Q-value distribution (recent training steps)
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics.detailed_episodes:
        all_q_values = []
        for ep_data in list(metrics.detailed_episodes)[-20:]:
            for q_vals in ep_data.q_values:
                all_q_values.extend(q_vals)
        
        if all_q_values:
            ax3.hist(all_q_values, bins=50, color='orange', alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(all_q_values), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(all_q_values):.2f}')
            ax3.axvline(np.median(all_q_values), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(all_q_values):.2f}')
            ax3.set_xlabel('Q-value', fontsize=11)
            ax3.set_ylabel('Frequency', fontsize=11)
            ax3.set_title('Q-value Distribution (Last 20 Episodes)', fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Q-value vs Actual Return (calibration check)
    ax4 = fig.add_subplot(gs[1, 0])
    if metrics.detailed_episodes:
        initial_q_values = []
        actual_returns = []
        
        for ep_data in metrics.detailed_episodes:
            if ep_data.q_values:
                initial_q = np.max(ep_data.q_values[0])
                initial_q_values.append(initial_q)
                actual_returns.append(ep_data.total_return)
        
        if initial_q_values and actual_returns:
            ax4.scatter(initial_q_values, actual_returns, alpha=0.5, s=30, color='steelblue')
            
            # Perfect calibration line
            min_val = min(min(initial_q_values), min(actual_returns))
            max_val = max(max(initial_q_values), max(actual_returns))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                    label='Perfect Calibration')
            
            # Fit line
            z = np.polyfit(initial_q_values, actual_returns, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(initial_q_values), max(initial_q_values), 100)
            ax4.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, 
                    label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
            
            ax4.set_xlabel('Initial Q-value Estimate', fontsize=11)
            ax4.set_ylabel('Actual Episode Return', fontsize=11)
            ax4.set_title('Q-value Calibration Check', fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    # 5. Per-action Q-value evolution
    ax5 = fig.add_subplot(gs[1, 1])
    if metrics.action_q_values:
        for action in range(5):
            if action in metrics.action_q_values and len(metrics.action_q_values[action]) > 100:
                q_vals = metrics.action_q_values[action]
                window = 100
                smoothed = np.convolve(q_vals, np.ones(window)/window, mode='valid')
                ax5.plot(smoothed, label=ACTION_NAMES[action], linewidth=2, alpha=0.8)
        
        ax5.set_xlabel('Sample (smoothed)', fontsize=11)
        ax5.set_ylabel('Average Q-value', fontsize=11)
        ax5.set_title('Per-Action Q-value Evolution', fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Q-value change rate (detects instability)
    ax6 = fig.add_subplot(gs[1, 2])
    if len(metrics.avg_q_values) > 1:
        q_changes = np.abs(np.diff(metrics.avg_q_values))
        window = 500
        
        steps = np.arange(len(q_changes))
        ax6.plot(steps, q_changes, alpha=0.3, linewidth=0.5, color='red')
        
        if len(q_changes) >= window:
            smoothed = np.convolve(q_changes, np.ones(window)/window, mode='valid')
            ax6.plot(range(window-1, len(q_changes)), smoothed, 
                    color='darkred', linewidth=2, label=f'{window}-step MA')
        
        ax6.set_xlabel('Training Step', fontsize=11)
        ax6.set_ylabel('|ΔQ|', fontsize=11)
        ax6.set_title('Q-value Change Rate (Instability Indicator)', fontsize=13, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
    
    plt.suptitle('Q-value Analysis Diagnostics', fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = save_dir / 'qvalue_diagnostics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Q-value diagnostics to {plot_path}")

def plot_learning_dynamics(metrics: TrainingMetrics, save_dir: Path):
    """Plot learning dynamics: loss, gradients, TD errors, PER stats."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    if metrics.losses:
        steps = np.arange(len(metrics.losses))
        ax1.plot(steps, metrics.losses, alpha=0.3, linewidth=0.5, color='purple')
        
        window = 1000
        if len(metrics.losses) >= window:
            smoothed = np.convolve(metrics.losses, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(metrics.losses)), smoothed, 
                    color='darkred', linewidth=2, label=f'{window}-step MA')
        
        ax1.set_xlabel('Training Step', fontsize=11)
        ax1.set_ylabel('Loss (Smooth L1)', fontsize=11)
        ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # 2. Gradient norms (detects exploding/vanishing gradients)
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics.gradient_norms:
        steps = np.arange(len(metrics.gradient_norms))
        ax2.plot(steps, metrics.gradient_norms, alpha=0.3, linewidth=0.5, color='blue')
        
        window = 500
        if len(metrics.gradient_norms) >= window:
            smoothed = np.convolve(metrics.gradient_norms, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(metrics.gradient_norms)), smoothed, 
                    color='darkblue', linewidth=2, label=f'{window}-step MA')
        
        ax2.set_xlabel('Training Step', fontsize=11)
        ax2.set_ylabel('Gradient Norm', fontsize=11)
        ax2.set_title('Gradient Magnitudes', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Add threshold lines
        ax2.axhline(10.0, color='orange', linestyle='--', linewidth=1, 
                   alpha=0.5, label='Clip threshold')
        ax2.axhline(0.01, color='red', linestyle='--', linewidth=1, 
                   alpha=0.5, label='Vanishing threshold')
    
    # 3. TD error statistics
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics.td_errors_mean and metrics.td_errors_max:
        steps = np.arange(len(metrics.td_errors_mean))
        ax3.plot(steps, metrics.td_errors_mean, label='Mean |TD|', 
                linewidth=2, color='blue', alpha=0.8)
        ax3.plot(steps, metrics.td_errors_max, label='Max |TD|', 
                linewidth=2, color='red', alpha=0.8)
        
        ax3.set_xlabel('Training Step', fontsize=11)
        ax3.set_ylabel('TD Error Magnitude', fontsize=11)
        ax3.set_title('TD Error Statistics', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # 4. PER beta annealing
    ax4 = fig.add_subplot(gs[1, 0])
    if metrics.replay_beta_values:
        steps = np.arange(len(metrics.replay_beta_values))
        ax4.plot(steps, metrics.replay_beta_values, color='green', linewidth=2)
        ax4.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Target (β=1)')
        ax4.set_xlabel('Training Step', fontsize=11)
        ax4.set_ylabel('Beta (β)', fontsize=11)
        ax4.set_title('PER Importance Sampling Weight (β)', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1.05])
    
    # 5. Loss vs Epsilon (exploration correlation)
    ax5 = fig.add_subplot(gs[1, 1])
    if metrics.losses and metrics.epsilons and len(metrics.losses) == len(metrics.epsilons):
        # Bin by epsilon
        eps_bins = np.linspace(min(metrics.epsilons), max(metrics.epsilons), 20)
        loss_means = []
        loss_stds = []
        bin_centers = []
        
        for i in range(len(eps_bins) - 1):
            mask = (np.array(metrics.epsilons) >= eps_bins[i]) & (np.array(metrics.epsilons) < eps_bins[i+1])
            if np.sum(mask) > 10:
                losses_in_bin = np.array(metrics.losses)[mask]
                loss_means.append(np.mean(losses_in_bin))
                loss_stds.append(np.std(losses_in_bin))
                bin_centers.append((eps_bins[i] + eps_bins[i+1]) / 2)
        
        if loss_means:
            ax5.errorbar(bin_centers, loss_means, yerr=loss_stds, fmt='o-', 
                        linewidth=2, markersize=8, capsize=5, color='purple', alpha=0.8)
            ax5.set_xlabel('Epsilon (ε)', fontsize=11)
            ax5.set_ylabel('Average Loss', fontsize=11)
            ax5.set_title('Loss vs Exploration Rate', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3)
    
    # 6. Training stability over episodes
    ax6 = fig.add_subplot(gs[1, 2])
    if len(metrics.episode_returns) > 100:
        # Compute coefficient of variation in windows
        window = 50
        cv_values = []
        ep_centers = []
        
        for i in range(window, len(metrics.episode_returns), 10):
            returns_window = metrics.episode_returns[i-window:i]
            mean_ret = np.mean(returns_window)
            if mean_ret != 0:
                cv = np.std(returns_window) / abs(mean_ret)
                cv_values.append(cv)
                ep_centers.append(i)
        
        if cv_values:
            ax6.plot(ep_centers, cv_values, color='darkgreen', linewidth=2)
            ax6.set_xlabel('Episode', fontsize=11)
            ax6.set_ylabel('Coefficient of Variation', fontsize=11)
            ax6.set_title(f'Return Stability (CV, {window}-ep window)', fontsize=13, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Lower CV = more stable
            ax6.axhline(0.5, color='orange', linestyle='--', linewidth=1, 
                       alpha=0.7, label='Moderate stability')
            ax6.legend()
    
    plt.suptitle('Learning Dynamics Diagnostics', fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = save_dir / 'learning_dynamics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved learning dynamics to {plot_path}")

def plot_training_summary(metrics: TrainingMetrics, save_dir: Path):
    """Create comprehensive training summary with key insights."""
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)
    
    # 1. Episode returns
    ax1 = fig.add_subplot(gs[0, :2])
    if metrics.episode_returns:
        episodes = np.arange(len(metrics.episode_returns))
        ax1.scatter(episodes, metrics.episode_returns, alpha=0.2, s=5, color='blue')
        
        window = 50
        if len(metrics.episode_returns) >= window:
            rolling_mean = np.convolve(metrics.episode_returns, np.ones(window)/window, mode='valid')
            ax1.plot(np.arange(window-1, len(metrics.episode_returns)), 
                    rolling_mean, color='red', linewidth=3, label=f'{window}-ep MA')
        
        ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Episode Return', fontsize=12, fontweight='bold')
        ax1.set_title('Training Progress: Episode Returns', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
    
    # 2. Success rate
    ax2 = fig.add_subplot(gs[0, 2:])
    if metrics.success_rate:
        ax2.plot(metrics.success_rate, color='green', linewidth=3)
        ax2.axhline(0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='70% target')
        ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Success Rate (100-ep Rolling)', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    # 3. Action distribution
    ax3 = fig.add_subplot(gs[1, 0])
    total_actions = sum(metrics.action_counts.values())
    if total_actions > 0:
        actions = sorted(metrics.action_counts.keys())
        percentages = [metrics.action_counts[a]/total_actions*100 for a in actions]
        colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
        bars = ax3.bar([ACTION_NAMES[a] for a in actions], percentages, color=colors, edgecolor='black')
        ideal = 100 / len(actions)
        ax3.axhline(ideal, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_ylabel('Percentage', fontsize=11, fontweight='bold')
        ax3.set_title('Action Distribution', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Per-action rewards
    ax4 = fig.add_subplot(gs[1, 1])
    if metrics.action_rewards:
        actions = sorted(metrics.action_rewards.keys())
        avg_rewards = [np.mean(metrics.action_rewards[a]) if metrics.action_rewards[a] else 0 for a in actions]
        colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in avg_rewards]
        ax4.bar([ACTION_NAMES[a] for a in actions], avg_rewards, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.set_ylabel('Avg Reward', fontsize=11, fontweight='bold')
        ax4.set_title('Per-Action Reward', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Q-value progression
    ax5 = fig.add_subplot(gs[1, 2])
    if metrics.avg_q_values:
        window = 500
        steps = np.arange(len(metrics.avg_q_values))
        ax5.plot(steps, metrics.avg_q_values, alpha=0.3, linewidth=0.5, color='orange')
        if len(metrics.avg_q_values) >= window:
            smoothed = np.convolve(metrics.avg_q_values, np.ones(window)/window, mode='valid')
            ax5.plot(range(window-1, len(metrics.avg_q_values)), smoothed, color='darkblue', linewidth=2)
        ax5.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Avg Q-value', fontsize=11, fontweight='bold')
        ax5.set_title('Q-value Evolution', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Loss
    ax6 = fig.add_subplot(gs[1, 3])
    if metrics.losses:
        window = 500
        steps = np.arange(len(metrics.losses))
        ax6.plot(steps, metrics.losses, alpha=0.3, linewidth=0.5, color='purple')
        if len(metrics.losses) >= window:
            smoothed = np.convolve(metrics.losses, np.ones(window)/window, mode='valid')
            ax6.plot(range(window-1, len(metrics.losses)), smoothed, color='darkred', linewidth=2)
        ax6.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax6.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
    
    # 7. Reward composition pie
    ax7 = fig.add_subplot(gs[2, 0])
    n_positive = len(metrics.positive_rewards)
    n_negative = len(metrics.negative_rewards)
    n_zero = sum(metrics.zero_rewards)
    total = n_positive + n_negative + n_zero
    if total > 0:
        sizes = [n_positive, n_negative, n_zero]
        labels = ['Positive', 'Negative', 'Zero']
        colors = ['green', 'red', 'gray']
        ax7.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax7.set_title('Reward Composition', fontsize=13, fontweight='bold')
    
    # 8. Episode length distribution
    ax8 = fig.add_subplot(gs[2, 1])
    if metrics.episode_lengths:
        ax8.hist(metrics.episode_lengths, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax8.axvline(np.mean(metrics.episode_lengths), color='red', linestyle='--', linewidth=2)
        ax8.set_xlabel('Episode Length', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax8.set_title('Episode Length Dist.', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. TD errors
    ax9 = fig.add_subplot(gs[2, 2])
    if metrics.td_errors_mean:
        ax9.plot(metrics.td_errors_mean, color='blue', linewidth=2, label='Mean')
        if metrics.td_errors_max:
            ax9.plot(metrics.td_errors_max, color='red', linewidth=2, alpha=0.7, label='Max')
        ax9.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax9.set_ylabel('|TD Error|', fontsize=11, fontweight='bold')
        ax9.set_title('TD Error Magnitude', fontsize=13, fontweight='bold')
        ax9.legend(fontsize=10)
        ax9.grid(True, alpha=0.3)
        ax9.set_yscale('log')
    
    # 10. Gradient norms
    ax10 = fig.add_subplot(gs[2, 3])
    if metrics.gradient_norms:
        window = 500
        steps = np.arange(len(metrics.gradient_norms))
        ax10.plot(steps, metrics.gradient_norms, alpha=0.3, linewidth=0.5, color='blue')
        if len(metrics.gradient_norms) >= window:
            smoothed = np.convolve(metrics.gradient_norms, np.ones(window)/window, mode='valid')
            ax10.plot(range(window-1, len(metrics.gradient_norms)), smoothed, color='darkblue', linewidth=2)
        ax10.axhline(10.0, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax10.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax10.set_ylabel('Gradient Norm', fontsize=11, fontweight='bold')
        ax10.set_title('Gradient Magnitude', fontsize=13, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        ax10.set_yscale('log')
    
    # 11. Statistics summary (text)
    ax11 = fig.add_subplot(gs[3, :2])
    ax11.axis('off')
    
    stats = metrics.get_recent_stats()
    total_actions = sum(metrics.action_counts.values())
    
    # Compute additional stats
    if metrics.episode_returns:
        best_return = max(metrics.episode_returns)
        best_ep = np.argmax(metrics.episode_returns)
        final_100_mean = np.mean(metrics.episode_returns[-100:]) if len(metrics.episode_returns) >= 100 else 0
    else:
        best_return = 0
        best_ep = 0
        final_100_mean = 0
    
    summary_text = f"""
    TRAINING SUMMARY
    {'='*60}
    
    PERFORMANCE METRICS (Last 100 Episodes):
      • Mean Return:        {stats.get('mean_return', 0):.2f} ± {stats.get('std_return', 0):.2f}
      • Success Rate:       {stats.get('success_rate', 0)*100:.1f}%
      • Mean Length:        {stats.get('mean_length', 0):.1f} steps
    
    OVERALL STATISTICS:
      • Total Episodes:     {len(metrics.episode_returns)}
      • Total Steps:        {sum(metrics.episode_lengths) if metrics.episode_lengths else 0}
      • Best Return:        {best_return:.2f} (Episode {best_ep})
      • Total Actions:      {total_actions}
    
    LEARNING DYNAMICS:
      • Final ε:            {metrics.epsilons[-1] if metrics.epsilons else 0:.4f}
      • Final Avg Q:        {metrics.avg_q_values[-1] if metrics.avg_q_values else 0:.2f}
      • Final Loss:         {metrics.losses[-1] if metrics.losses else 0:.4f}
      • Avg Grad Norm:      {np.mean(metrics.gradient_norms[-1000:]) if len(metrics.gradient_norms) >= 1000 else 0:.4f}
    
    REWARD ANALYSIS:
      • Positive Rewards:   {len(metrics.positive_rewards)} ({len(metrics.positive_rewards)/max(1, len(metrics.reward_per_step))*100:.1f}%)
      • Negative Rewards:   {len(metrics.negative_rewards)} ({len(metrics.negative_rewards)/max(1, len(metrics.reward_per_step))*100:.1f}%)
      • Zero Rewards:       {sum(metrics.zero_rewards)} ({sum(metrics.zero_rewards)/max(1, len(metrics.reward_per_step))*100:.1f}%)
    """
    
    ax11.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
    # 12. Key insights
    ax12 = fig.add_subplot(gs[3, 2:])
    ax12.axis('off')
    
    insights = []
    
    # Analyze action distribution
    if total_actions > 0:
        action_probs = [metrics.action_counts[a]/total_actions for a in range(5)]
        max_prob = max(action_probs)
        if max_prob > 0.4:
            dominant_action = ACTION_NAMES[np.argmax(action_probs)]
            insights.append(f"⚠ Action bias: {dominant_action} used {max_prob*100:.1f}% of time")
        
        # Check exploration
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in action_probs)
        max_entropy = np.log(5)
        if entropy < max_entropy * 0.6:
            insights.append(f"⚠ Low action diversity (entropy: {entropy:.2f}/{max_entropy:.2f})")
    
    # Check Q-value overestimation
    if metrics.detailed_episodes:
        initial_qs = [np.max(ep.q_values[0]) if ep.q_values else 0 for ep in metrics.detailed_episodes]
        actual_returns = [ep.total_return for ep in metrics.detailed_episodes]
        if initial_qs and actual_returns:
            mean_q_est = np.mean(initial_qs)
            mean_actual = np.mean(actual_returns)
            if mean_q_est > mean_actual * 1.5:
                insights.append(f"⚠ Q-value overestimation: Est={mean_q_est:.1f}, Actual={mean_actual:.1f}")
    
    # Check training stability
    if len(metrics.episode_returns) > 200:
        recent_std = np.std(metrics.episode_returns[-100:])
        early_std = np.std(metrics.episode_returns[:100])
        if recent_std > early_std * 1.5:
            insights.append(f"⚠ Increasing instability: STD {early_std:.1f}→{recent_std:.1f}")
        elif recent_std < early_std * 0.5:
            insights.append(f"✓ Improving stability: STD {early_std:.1f}→{recent_std:.1f}")
    
    # Check success rate trend
    if len(metrics.success_rate) > 100:
        recent_trend = np.mean(metrics.success_rate[-50:]) - np.mean(metrics.success_rate[-100:-50])
        if recent_trend > 0.1:
            insights.append(f"✓ Strong improvement: +{recent_trend*100:.1f}% success rate")
        elif recent_trend < -0.1:
            insights.append(f"⚠ Performance decline: {recent_trend*100:.1f}% success rate")
    
    # Check gradient health
    if len(metrics.gradient_norms) > 1000:
        recent_grad_mean = np.mean(metrics.gradient_norms[-1000:])
        if recent_grad_mean > 5.0:
            insights.append(f"⚠ High gradients: {recent_grad_mean:.2f} (may explode)")
        elif recent_grad_mean < 0.01:
            insights.append(f"⚠ Vanishing gradients: {recent_grad_mean:.4f}")
        else:
            insights.append(f"✓ Healthy gradients: {recent_grad_mean:.2f}")
    
    if not insights:
        insights.append("✓ No major issues detected")
    
    insights_text = "KEY INSIGHTS & WARNINGS\n" + "="*40 + "\n\n" + "\n\n".join(insights)
    
    ax12.text(0.05, 0.5, insights_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral' if any('⚠' in i for i in insights) else 'lightgreen', 
                      alpha=0.8, pad=1))
    
    plt.suptitle('D3QN-PER Training Summary Dashboard', fontsize=18, fontweight='bold', y=0.998)
    
    plot_path = save_dir / 'training_summary.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training summary to {plot_path}")

def plot_learning_curves(metrics: TrainingMetrics, save_dir: Path):
    """Create focused learning curve plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Returns with confidence interval
    ax = axes[0, 0]
    if len(metrics.episode_returns) > 0:
        episodes = np.arange(len(metrics.episode_returns))
        returns = np.array(metrics.episode_returns)
        
        # Plot raw returns
        ax.plot(episodes, returns, alpha=0.2, linewidth=0.5, color='blue', label='Raw')
        
        # Rolling statistics
        window = 100
        if len(returns) >= window:
            rolling_mean = np.convolve(returns, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(returns[max(0, i-window):i+1]) 
                                   for i in range(window-1, len(returns))])
            
            x = np.arange(window-1, len(returns))
            ax.plot(x, rolling_mean, color='red', linewidth=2, label=f'{window}-ep MA')
            ax.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std,
                           alpha=0.3, color='red')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Return', fontsize=12)
        ax.set_title('Learning Curve: Returns', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Success rate with smoothing
    ax = axes[0, 1]
    if len(metrics.success_rate) > 0:
        episodes = np.arange(len(metrics.success_rate))
        ax.plot(episodes, metrics.success_rate, color='green', linewidth=2)
        ax.axhline(0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% threshold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Learning Curve: Success Rate', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Q-value progression
    ax = axes[1, 0]
    if len(metrics.avg_q_values) > 0:
        steps = np.arange(len(metrics.avg_q_values))
        ax.plot(steps, metrics.avg_q_values, alpha=0.3, linewidth=0.5, color='orange')
        
        # Smooth
        window = 1000
        if len(metrics.avg_q_values) >= window:
            smooth = np.convolve(metrics.avg_q_values, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(metrics.avg_q_values)), 
                   smooth, color='darkblue', linewidth=2)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Avg Q-value', fontsize=12)
        ax.set_title('Q-value Progression', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 4. Loss with smoothing
    ax = axes[1, 1]
    if len(metrics.losses) > 0:
        steps = np.arange(len(metrics.losses))
        ax.plot(steps, metrics.losses, alpha=0.3, linewidth=0.5, color='purple')
        
        # Smooth
        window = 1000
        if len(metrics.losses) >= window:
            smooth = np.convolve(metrics.losses, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(metrics.losses)), 
                   smooth, color='darkred', linewidth=2)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('D3QN-PER Learning Curves', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plot_path = save_dir / 'learning_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved learning curves to {plot_path}")

def main():
    ap = argparse.ArgumentParser(description='Train D3QN-PER agent for OBELIX (CPU/GPU optimized)')
    
    # Environment args
    ap.add_argument("--obelix_py", type=str, required=True, help="Path to obelix.py")
    ap.add_argument("--agent_id", type=str, default="001", help="Unique agent identifier")
    ap.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    ap.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    ap.add_argument("--difficulty", type=int, default=0, help="Difficulty level (0-3)")
    ap.add_argument("--wall_obstacles", action="store_true", help="Enable wall obstacles")
    ap.add_argument("--box_speed", type=int, default=2, help="Box movement speed")
    ap.add_argument("--scaling_factor", type=int, default=5, help="Arena scaling factor")
    ap.add_argument("--arena_size", type=int, default=500, help="Arena size")
    
    # Device & optimization args
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                   help="Device to use (auto=detect best)")
    ap.add_argument("--use_amp", action="store_true", 
                   help="Use automatic mixed precision (GPU only, faster training)")
    ap.add_argument("--num_envs", type=int, default=1,
                   help="Number of parallel environments (1=sequential, 4-8=parallel, uses multiprocessing)")
    ap.add_argument("--num_workers", type=int, default=0, 
                   help="Number of data loading workers (0=main thread)")
    ap.add_argument("--pin_memory", action="store_true",
                   help="Pin memory for faster GPU transfer")
    
    # D3QN-PER hyperparameters
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    ap.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    ap.add_argument("--batch", type=int, default=256, help="Batch size")
    ap.add_argument("--replay_capacity", type=int, default=100000, help="Replay buffer capacity")
    ap.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    ap.add_argument("--warmup", type=int, default=2000, help="Warmup steps before training")
    ap.add_argument("--target_sync", type=int, default=2000, help="Target network sync frequency")
    
    # Exploration
    ap.add_argument("--eps_start", type=float, default=1.0, help="Initial epsilon")
    ap.add_argument("--eps_end", type=float, default=0.02, help="Final epsilon")
    ap.add_argument("--eps_decay_steps", type=int, default=250000, help="Epsilon decay steps")
    
    # PER hyperparameters
    ap.add_argument("--per_alpha", type=float, default=0.6, help="PER alpha (prioritization)")
    ap.add_argument("--per_beta_start", type=float, default=0.4, help="PER beta start")
    
    # Multi-seed training for better generalization (CRITICAL for competition)
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--multi_seed", action="store_true", 
                   help="Enable multi-seed training (cycles through seeds for better generalization)")
    ap.add_argument("--seed_list", type=int, nargs='+', default=[42, 123, 456, 789, 999],
                   help="List of seeds to cycle through if multi_seed enabled")
    
    # Reward shaping for sparse reward problems (CRITICAL for OBELIX)
    ap.add_argument("--reward_shaping", action="store_true",
                   help="Enable reward shaping to help agent learn in sparse reward environment")
    ap.add_argument("--shape_forward_bonus", type=float, default=0.5,
                   help="Bonus reward for moving forward (encourages exploration)")
    ap.add_argument("--shape_sensor_bonus", type=float, default=2.0,
                   help="Bonus reward when any sensor activates (finding box area)")
    ap.add_argument("--shape_ir_bonus", type=float, default=5.0,
                   help="Bonus reward for IR sensor activation (very close to box)")
    
    # Curriculum training and checkpoint loading
    ap.add_argument("--load_checkpoint", type=str, default=None,
                   help="Load checkpoint from previous training phase (e.g., 'phase1' to load from submission_phase1/)")
    ap.add_argument("--checkpoint_episode", type=int, default=None,
                   help="Specific checkpoint episode to load (e.g., 1500 for checkpoint_ep1500.pth). If None, loads weights.pth")
    ap.add_argument("--curriculum_phase", type=int, default=1,
                   help="Current curriculum phase number (1-4). Used for documentation only.")
    ap.add_argument("--reset_optimizer", action="store_true",
                   help="Reset optimizer when loading checkpoint (useful for changing learning rate)")
    
    ap.add_argument("--save_freq", type=int, default=250, help="Save model every N episodes")
    
    args = ap.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            torch.set_num_threads(os.cpu_count() or 4)
            print(f"💻 Using CPU with {torch.get_num_threads()} threads")
    
    # Set seeds (including CUDA if applicable)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Multi-seed setup
    if args.multi_seed:
        training_seeds = args.seed_list
        print(f"🎲 Multi-seed training enabled with seeds: {training_seeds}")
        print(f"   This improves generalization to hidden test seeds!")
    else:
        training_seeds = [args.seed]
        print(f"⚠️  Single-seed training (seed={args.seed})")
        print(f"   Consider --multi_seed for better generalization on hidden test seeds!")
    
    # Mixed precision training setup (GPU only)
    use_amp = args.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("⚡ Using Automatic Mixed Precision (AMP) for faster training")
    
    # Create submission directory
    submission_dir = Path(f"submission_{args.agent_id}")
    submission_dir.mkdir(exist_ok=True)
    
    # Import environment  
    OBELIX = import_obelix(args.obelix_py)
    
    # Initialize networks and move to device
    q_network = DuelingDQN(in_dim=18, n_actions=5, hidden_dim=args.hidden_dim).to(device)
    target_network = DuelingDQN(in_dim=18, n_actions=5, hidden_dim=args.hidden_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    # Optimizer with gradient clipping built-in
    optimizer = optim.Adam(q_network.parameters(), lr=args.lr, eps=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=args.lr/10)
    
    # Load checkpoint if specified (for curriculum training)
    start_episode = 0
    if args.load_checkpoint:
        checkpoint_dir = Path(f"submission_{args.load_checkpoint}")
        
        if args.checkpoint_episode:
            checkpoint_path = checkpoint_dir / f"checkpoint_ep{args.checkpoint_episode}.pth"
        else:
            checkpoint_path = checkpoint_dir / "weights.pth"
        
        if checkpoint_path.exists():
            print(f"\n{'='*60}")
            print(f"📂 LOADING CHECKPOINT FROM PREVIOUS PHASE")
            print(f"{'='*60}")
            print(f"Checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load network weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Full checkpoint with metadata
                q_network.load_state_dict(checkpoint['state_dict'])
                target_network.load_state_dict(checkpoint['state_dict'])
                
                # Load optimizer and scheduler if not resetting
                if not args.reset_optimizer and 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f"✓ Loaded optimizer state")
                else:
                    print(f"✓ Using fresh optimizer (--reset_optimizer or no saved optimizer)")
                
                if not args.reset_optimizer and 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    print(f"✓ Loaded scheduler state")
                
                # Track starting episode for logging
                if 'episode' in checkpoint:
                    start_episode = checkpoint['episode']
                    print(f"✓ Previous training ended at episode {start_episode}")
                
                # Show previous stats if available
                if 'stats' in checkpoint:
                    prev_stats = checkpoint['stats']
                    print(f"✓ Previous performance:")
                    print(f"   - Success rate: {prev_stats.get('success_rate', 0)*100:.1f}%")
                    print(f"   - Mean return: {prev_stats.get('mean_return', 0):.1f}")
            else:
                # weights.pth only (just state dict)
                q_network.load_state_dict(checkpoint)
                target_network.load_state_dict(checkpoint)
                print(f"✓ Loaded network weights only")
            
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  WARNING: Checkpoint not found: {checkpoint_path}")
            print(f"   Starting training from scratch instead.\n")
    
    # Print training configuration banner
    print(f"\n{'='*60}")
    print(f"D3QN-PER Training for OBELIX")
    if args.load_checkpoint:
        print(f"CURRICULUM TRAINING - Phase {args.curriculum_phase}")
    print(f"{'='*60}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Submission Dir: {submission_dir}")
    print(f"Episodes: {args.episodes}")
    if args.load_checkpoint and start_episode > 0:
        print(f"Continuing from: submission_{args.load_checkpoint} (completed {start_episode} episodes)")
    print(f"Difficulty: {args.difficulty}")
    print(f"Wall Obstacles: {args.wall_obstacles}")
    print(f"Device: {device}")
    print(f"Mixed Precision: {use_amp}")
    if args.reward_shaping:
        print(f"Reward Shaping: ENABLED ⚡")
        print(f"  - Forward bonus: +{args.shape_forward_bonus}")
        print(f"  - Sensor bonus: +{args.shape_sensor_bonus}")
        print(f"  - IR bonus: +{args.shape_ir_bonus}")
    else:
        print(f"Reward Shaping: DISABLED (raw environment rewards)")
    print(f"{'='*60}\n")
    
    # Initialize networks and move to device
    q_network = DuelingDQN(in_dim=18, n_actions=5, hidden_dim=args.hidden_dim).to(device)
    target_network = DuelingDQN(in_dim=18, n_actions=5, hidden_dim=args.hidden_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    # Optimizer with gradient clipping built-in
    optimizer = optim.Adam(q_network.parameters(), lr=args.lr, eps=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=args.lr/10)
    
    # Load checkpoint if specified (for curriculum training)
    start_episode = 0
    if args.load_checkpoint:
        checkpoint_dir = Path(f"submission_{args.load_checkpoint}")
        
        if args.checkpoint_episode:
            checkpoint_path = checkpoint_dir / f"checkpoint_ep{args.checkpoint_episode}.pth"
        else:
            checkpoint_path = checkpoint_dir / "weights.pth"
        
        if checkpoint_path.exists():
            print(f"\n{'='*60}")
            print(f"📂 LOADING CHECKPOINT FROM PREVIOUS PHASE")
            print(f"{'='*60}")
            print(f"Checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load network weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Full checkpoint with metadata
                q_network.load_state_dict(checkpoint['state_dict'])
                target_network.load_state_dict(checkpoint['state_dict'])
                
                # Load optimizer and scheduler if not resetting
                if not args.reset_optimizer and 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f"✓ Loaded optimizer state")
                else:
                    print(f"✓ Using fresh optimizer (--reset_optimizer or no saved optimizer)")
                
                if not args.reset_optimizer and 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    print(f"✓ Loaded scheduler state")
                
                # Track starting episode for logging
                if 'episode' in checkpoint:
                    start_episode = checkpoint['episode']
                    print(f"✓ Previous training ended at episode {start_episode}")
                
                # Show previous stats if available
                if 'stats' in checkpoint:
                    prev_stats = checkpoint['stats']
                    print(f"✓ Previous performance:")
                    print(f"   - Success rate: {prev_stats.get('success_rate', 0)*100:.1f}%")
                    print(f"   - Mean return: {prev_stats.get('mean_return', 0):.1f}")
            else:
                # weights.pth only (just state dict)
                q_network.load_state_dict(checkpoint)
                target_network.load_state_dict(checkpoint)
                print(f"✓ Loaded network weights only")
            
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  WARNING: Checkpoint not found: {checkpoint_path}")
            print(f"   Starting training from scratch instead.\n")
    
    # Prioritized replay buffer
    replay = PrioritizedReplay(
        capacity=args.replay_capacity,
        alpha=args.per_alpha,
        beta_start=args.per_beta_start
    )
    
    # Parallel environment collector (if num_envs > 1)
    use_parallel = args.num_envs > 1
    if use_parallel:
        collector = BatchedExperienceCollector(num_parallel=args.num_envs)
        print(f"⚡ Using {args.num_envs} parallel environments for {args.num_envs}x speedup!")
        print(f"   Episodes will be collected in batches of {args.num_envs}")
    else:
        collector = None
        print("📝 Using sequential environment execution")
    
    # Environment parameters for parallel execution
    env_params = {
        'scaling_factor': args.scaling_factor,
        'arena_size': args.arena_size,
        'max_steps': args.max_steps,
        'wall_obstacles': args.wall_obstacles,
        'difficulty': args.difficulty,
        'box_speed': args.box_speed,
    }
    
    # Reward shaping configuration
    reward_shaping_config = {
        'enabled': args.reward_shaping,
        'forward_bonus': args.shape_forward_bonus,
        'sensor_bonus': args.shape_sensor_bonus,
        'ir_bonus': args.shape_ir_bonus,
    }
    
    # Metrics tracker
    metrics = TrainingMetrics(window_size=100)
    
    # Training state
    total_steps = 0
    best_success_rate = 0.0
    
    def epsilon_schedule(step: int) -> float:
        """Epsilon decay schedule."""
        if step >= args.eps_decay_steps:
            return args.eps_end
        frac = step / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)
    
    print("Starting training...\n")
    start_time = time.time()
    
    # Training loop
    episode = 0
    while episode < args.episodes:
        
        # Determine how many episodes to run in this batch
        if use_parallel:
            batch_size = min(args.num_envs, args.episodes - episode)
            # Multi-seed: cycle through seed list
            seeds = [training_seeds[(episode + i) % len(training_seeds)] for i in range(batch_size)]
            epsilon = epsilon_schedule(total_steps)
            
            # Collect batch of episodes in parallel
            episode_results = collector.collect_batch(
                obelix_path=args.obelix_py,
                network=q_network,
                env_params=env_params,
                seeds=seeds,
                epsilon=epsilon,
                max_steps=args.max_steps,
                hidden_dim=args.hidden_dim,
                device=device,
                reward_shaping_config=reward_shaping_config
            )
            
            # Process all episodes in the batch
            for ep_idx, ep_data in enumerate(episode_results):
                current_episode = episode + ep_idx
                metrics.start_episode(current_episode)
                
                # Add all transitions to replay
                for trans in ep_data['transitions']:
                    replay.add(Transition(
                        s=trans['s'],
                        a=trans['a'],
                        r=trans['r'],
                        s2=trans['s2'],
                        done=trans['done']
                    ))
                    total_steps += 1
                
                # Track metrics
                for i in range(len(ep_data['rewards'])):
                    metrics.step(
                        reward=ep_data['rewards'][i],
                        action=ep_data['actions'][i],
                        q_values=ep_data['q_values'][i],
                        state=ep_data['states'][i],
                        is_greedy=(np.random.rand() >= epsilon)  # Approximate
                    )
                
                # End episode
                metrics.end_episode(ep_data['success'])
            
            episode += batch_size
            
        else:
            # Sequential mode (original implementation)
            # Multi-seed: cycle through seed list
            current_seed = training_seeds[episode % len(training_seeds)]
            
            # Create environment
            env = OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=args.wall_obstacles,
                difficulty=args.difficulty,
                box_speed=args.box_speed,
                seed=current_seed,
            )
            
            state = env.reset(seed=current_seed)
            metrics.start_episode(episode)
            episode_success = False
            
            # Episode loop
            for step in range(args.max_steps):
                # Select action (epsilon-greedy)
                epsilon = epsilon_schedule(total_steps)
                
                # Get Q-values for current state
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_network(state_t).squeeze(0).cpu().numpy()
                
                # Epsilon-greedy action selection
                is_greedy = np.random.rand() >= epsilon
                if is_greedy:
                    action_idx = int(np.argmax(q_values))
                else:
                    action_idx = np.random.randint(len(ACTIONS))
                
                # Take action
                next_state, reward, done = env.step(ACTIONS[action_idx], render=False)
                
                # REWARD SHAPING (if enabled) - helps with sparse reward problem
                base_reward = reward
                shaped_reward = reward
                
                if args.reward_shaping:
                    # 1. Bonus for moving forward (encourages exploration)
                    if action_idx == 2:  # FW action
                        shaped_reward += args.shape_forward_bonus
                    
                    # 2. Bonus for ANY sensor activation (robot is near box)
                    if np.any(next_state[:16] == 1):  # Any of 16 sonar sensors active
                        shaped_reward += args.shape_sensor_bonus
                    
                    # 3. Extra bonus for IR sensor (robot is very close/aligned with box)
                    if next_state[16] == 1:  # IR sensor (index 16)
                        shaped_reward += args.shape_ir_bonus
                
                # Check success (attached box reached boundary)
                episode_success = done and base_reward > 0  # Use base reward for success check
                
                # Store transition with shaped reward
                replay.add(Transition(
                    s=state,
                    a=action_idx,
                    r=float(shaped_reward),  # Use shaped reward for training
                    s2=next_state,
                    done=bool(done)
                ))
                
                # Track metrics with BASE reward (to see true performance)
                metrics.step(base_reward, action_idx, q_values, state, is_greedy)
                state = next_state
                total_steps += 1
                
                if done:
                    break
            
            # End episode
            metrics.end_episode(episode_success)
            episode += 1
        
        # Training step (after collecting experience - works for both modes)
        # Perform multiple training updates per batch of episodes collected
        num_updates = args.num_envs if use_parallel else 1
        for _ in range(num_updates):
            if len(replay) >= max(args.warmup, args.batch):
                # Sample from replay with priorities
                s_batch, a_batch, r_batch, s2_batch, d_batch, weights, indices = replay.sample(args.batch)
                
                # Convert to tensors and move to device
                s_t = torch.from_numpy(s_batch).to(device)
                a_t = torch.from_numpy(a_batch).to(device)
                r_t = torch.from_numpy(r_batch).to(device)
                s2_t = torch.from_numpy(s2_batch).to(device)
                d_t = torch.from_numpy(d_batch).to(device)
                w_t = torch.from_numpy(weights).to(device)
                
                # Training with mixed precision if enabled
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # Double DQN target computation
                        with torch.no_grad():
                            # Use online network to select best action
                            next_q = q_network(s2_t)
                            next_actions = torch.argmax(next_q, dim=1)
                            
                            # Use target network to evaluate that action
                            next_q_target = target_network(s2_t)
                            next_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                            
                            # Compute target
                            targets = r_t + args.gamma * (1.0 - d_t) * next_values
                        
                        # Current Q-values
                        current_q = q_network(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                        
                        # Weighted loss (importance sampling)
                        loss = (w_t * nn.functional.smooth_l1_loss(current_q, targets, reduction='none')).mean()
                    
                    # TD errors (for priority update) - must be outside autocast
                    with torch.no_grad():
                        td_errors = (targets - current_q).cpu().numpy()
                    
                    # Optimize with gradient scaling
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    scaler.scale(loss).backward()
                    
                    # Compute gradient norm before clipping
                    scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    # Standard precision training
                    # Double DQN target computation
                    with torch.no_grad():
                        # Use online network to select best action
                        next_q = q_network(s2_t)
                        next_actions = torch.argmax(next_q, dim=1)
                        
                        # Use target network to evaluate that action
                        next_q_target = target_network(s2_t)
                        next_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        
                        # Compute target
                        targets = r_t + args.gamma * (1.0 - d_t) * next_values
                    
                    # Current Q-values
                    current_q = q_network(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                    
                    # TD errors (for priority update)
                    td_errors = (targets - current_q).detach().cpu().numpy()
                    
                    # Weighted loss (importance sampling)
                    loss = (w_t * nn.functional.smooth_l1_loss(current_q, targets, reduction='none')).mean()
                    
                    # Optimize
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    # Compute gradient norm before clipping
                    total_norm = 0.0
                    for p in q_network.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
                    optimizer.step()
                
                # Update priorities
                replay.update_priorities(indices, td_errors)
                
                # Anneal beta
                replay.anneal_beta(total_steps, args.eps_decay_steps)
                
                # Log comprehensive metrics
                with torch.no_grad():
                    all_q_vals = q_network(s_t)
                
                avg_q = current_q.mean().item()
                metrics.log_training(
                    loss=loss.item(),
                    avg_q=avg_q,
                    epsilon=epsilon,
                    gradient_norm=total_norm if isinstance(total_norm, float) else total_norm.item(),
                    td_errors=td_errors,
                    q_values_batch=all_q_vals.cpu(),
                    replay_beta=replay.beta,
                    training_step=total_steps,
                    episode_num=episode
                )
                
                # Sync target network
                if total_steps % args.target_sync == 0:
                    target_network.load_state_dict(q_network.state_dict())
        
        # Learning rate scheduling (after each batch)
        scheduler.step()
        
        # Logging (use the last completed episode number)
        if episode % 50 == 0 and episode > 0:
            stats = metrics.get_recent_stats()
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            epsilon = epsilon_schedule(total_steps)
            
            print(f"Episode {episode:4d}/{args.episodes} | "
                  f"Return: {stats['mean_return']:7.1f} ± {stats['std_return']:5.1f} | "
                  f"Length: {stats['mean_length']:5.1f} | "
                  f"Success: {stats['success_rate']*100:5.1f}% | "
                  f"ε: {epsilon:.3f} | "
                  f"Replay: {len(replay):6d} | "
                  f"Speed: {eps_per_sec:.2f} ep/s")
        
        # Save checkpoints
        if episode % args.save_freq == 0 and episode > 0:
            stats = metrics.get_recent_stats()
            current_success = stats.get('success_rate', 0.0)
            
            # Save current checkpoint (move to CPU for compatibility)
            checkpoint_path = submission_dir / f"checkpoint_ep{episode}.pth"
            torch.save({
                'episode': episode,
                'state_dict': {k: v.cpu() for k, v in q_network.state_dict().items()},
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'stats': stats,
                'device': str(device),
                'use_amp': use_amp,
                'num_envs': args.num_envs,
            }, checkpoint_path)
            
            # Save best model
            if current_success > best_success_rate:
                best_success_rate = current_success
                best_path = submission_dir / "weights.pth"
                # Save to CPU for universal loading
                torch.save({k: v.cpu() for k, v in q_network.state_dict().items()}, best_path)
                print(f"  → New best model saved! Success rate: {current_success*100:.1f}%")
    
    # Cleanup parallel resources
    if use_parallel:
        collector.shutdown()
    
    # Final save (move to CPU for compatibility)
    final_path = submission_dir / "weights.pth"
    torch.save({k: v.cpu() for k, v in q_network.state_dict().items()}, final_path)
    print(f"\n✓ Final model saved to {final_path} (CPU-compatible)")
    
    # Clear GPU cache if using CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Save agent file to submission directory
    agent_src = Path(__file__).parent / "agent_d3qn_per.py"
    agent_dst = submission_dir / "agent_d3qn_per.py"
    if agent_src.exists():
        import shutil
        shutil.copy(agent_src, agent_dst)
        print(f"✓ Agent file copied to {agent_dst}")
    
    # Generate plots
    print("\nGenerating comprehensive diagnostic plots...")
    print("This may take a few minutes...")
    
    plot_action_diagnostics(metrics, submission_dir)
    plot_reward_diagnostics(metrics, submission_dir)
    plot_qvalue_diagnostics(metrics, submission_dir)
    plot_learning_dynamics(metrics, submission_dir)
    plot_training_summary(metrics, submission_dir)
    
    print("\n" + "="*60)
    print("All diagnostic plots generated!")
    print("="*60)
    
    # Save training log with device and performance info
    log_path = submission_dir / "training_log.txt"
    training_time = time.time() - start_time
    with open(log_path, 'w') as f:
        f.write(f"D3QN-PER Training Log\n")
        f.write(f"{'='*60}\n\n")
        
        # Configuration
        f.write(f"CONFIGURATION:\n")
        f.write(f"  Agent ID: {args.agent_id}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Mixed Precision: {use_amp}\n")
        f.write(f"  Hidden Dim: {args.hidden_dim}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Batch Size: {args.batch}\n")
        f.write(f"  Gamma: {args.gamma}\n")
        f.write(f"  PER Alpha: {args.per_alpha}\n")
        f.write(f"  PER Beta Start: {args.per_beta_start}\n\n")
        
        # Training stats
        f.write(f"TRAINING STATISTICS:\n")
        f.write(f"  Total Episodes: {args.episodes}\n")
        f.write(f"  Total Steps: {total_steps}\n")
        f.write(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f} min)\n")
        f.write(f"  Steps/Second: {total_steps/training_time:.1f}\n")
        f.write(f"  Episodes/Second: {args.episodes/training_time:.2f}\n\n")
        
        # Performance
        final_stats = metrics.get_recent_stats()
        f.write(f"FINAL PERFORMANCE (last 100 episodes):\n")
        f.write(f"  Mean Return: {final_stats['mean_return']:.2f} ± {final_stats['std_return']:.2f}\n")
        f.write(f"  Mean Length: {final_stats['mean_length']:.1f}\n")
        f.write(f"  Success Rate: {final_stats['success_rate']*100:.1f}%\n")
        f.write(f"  Best Success Rate: {best_success_rate*100:.1f}%\n\n")
        
        # Action statistics
        total_actions = sum(metrics.action_counts.values())
        if total_actions > 0:
            f.write(f"ACTION DISTRIBUTION:\n")
            for action_idx in range(5):
                count = metrics.action_counts.get(action_idx, 0)
                pct = count / total_actions * 100
                f.write(f"  {ACTION_NAMES[action_idx]}: {count:7d} ({pct:5.1f}%)\n")
            f.write(f"\n")
        
        # Reward statistics
        if metrics.reward_per_step:
            f.write(f"REWARD STATISTICS:\n")
            f.write(f"  Total Rewards: {len(metrics.reward_per_step)}\n")
            f.write(f"  Positive: {len(metrics.positive_rewards)} ({len(metrics.positive_rewards)/len(metrics.reward_per_step)*100:.1f}%)\n")
            f.write(f"  Negative: {len(metrics.negative_rewards)} ({len(metrics.negative_rewards)/len(metrics.reward_per_step)*100:.1f}%)\n")
            f.write(f"  Zero: {sum(metrics.zero_rewards)} ({sum(metrics.zero_rewards)/len(metrics.reward_per_step)*100:.1f}%)\n")
            f.write(f"  Mean Reward: {np.mean(metrics.reward_per_step):.3f}\n\n")
        
        # Device-specific stats
        if device.type == "cuda":
            f.write(f"GPU STATISTICS:\n")
            f.write(f"  GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"  Peak Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB\n")
            f.write(f"  Memory Reserved: {torch.cuda.max_memory_reserved(0) / 1e9:.2f} GB\n")
    
    print(f"✓ Training log saved to {log_path}")
    
    # Performance summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Speed: {total_steps/training_time:.1f} steps/s, {args.episodes/training_time:.2f} ep/s")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    print(f"\nSubmission directory: {submission_dir}")
    print(f"Files created:")
    print(f"  - weights.pth (CPU-compatible model)")
    print(f"  - agent_d3qn_per.py (agent code)")
    print(f"  - action_diagnostics.png (9 plots)")
    print(f"  - reward_diagnostics.png (6 plots)")
    print(f"  - qvalue_diagnostics.png (6 plots)")
    print(f"  - learning_dynamics.png (6 plots)")
    print(f"  - training_summary.png (12 plots)")
    print(f"  - training_log.txt (detailed stats)")
    print(f"  - checkpoint_epXXXX.pth (periodic saves)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()