#!/usr/bin/env python3
"""
PPO Agent for OBELIX Competition
Loads trained weights and provides policy for evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Must match architecture used in training!
    """
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )
        
        # Critic head (value function) - not used during eval but needed for loading
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        """Forward pass returns policy logits and value."""
        features = self.feature(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, x, deterministic=True):
        """
        Get action from policy.
        
        Args:
            x: State tensor
            deterministic: If True, take argmax (greedy). If False, sample from distribution.
        """
        logits, _ = self.forward(x)
        
        if deterministic:
            # Greedy action (for competition evaluation)
            action = torch.argmax(logits, dim=-1)
        else:
            # Sample from policy distribution
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
        
        return action

# Global model instance
model = None

def policy(obs, rng=None):
    """
    Policy function for OBELIX competition.
    
    Args:
        obs: 18-bit observation from environment
        rng: Random number generator (unused, kept for compatibility)
    
    Returns:
        action: String action from ACTIONS list
    """
    global model
    
    # Lazy initialization - load model on first call
    if model is None:
        # Load trained weights
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "weights.pth")
        
        # Initialize model
        model = ActorCritic(in_dim=18, n_actions=5, hidden_dim=128)
        
        # Load weights
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✓ Loaded PPO model from {weights_path}")
        except FileNotFoundError:
            print(f"⚠️ Warning: weights.pth not found at {weights_path}")
            print(f"   Using random initialization!")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load weights: {e}")
            print(f"   Using random initialization!")
    
    # Convert observation to tensor
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    # Get action (deterministic for competition)
    with torch.no_grad():
        action_idx = model.get_action(obs_tensor, deterministic=True).item()
    
    # Return action string
    return ACTIONS[action_idx]

# Alternative: Stochastic policy (for testing/exploration)
def stochastic_policy(obs, rng=None):
    """
    Stochastic version of policy (samples from distribution).
    Useful for testing exploration vs exploitation.
    """
    global model
    
    if model is None:
        # Initialize (same as deterministic policy)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "weights.pth")
        
        model = ActorCritic(in_dim=18, n_actions=5, hidden_dim=128)
        
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
        except:
            pass
    
    # Convert observation to tensor
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    # Get action (stochastic)
    with torch.no_grad():
        action_idx = model.get_action(obs_tensor, deterministic=False).item()
    
    return ACTIONS[action_idx]

# For local testing
if __name__ == "__main__":
    print("PPO Agent for OBELIX")
    print("="*50)
    
    # Test with random observation
    test_obs = np.random.randint(0, 2, size=18)
    print(f"Test observation: {test_obs}")
    
    # Test deterministic policy
    action = policy(test_obs)
    print(f"Deterministic action: {action}")
    
    # Test stochastic policy
    action_stoch = stochastic_policy(test_obs)
    print(f"Stochastic action: {action_stoch}")
    
    # Test multiple times to see distribution
    print("\nAction distribution (10 samples, stochastic):")
    from collections import Counter
    actions = [stochastic_policy(test_obs) for _ in range(10)]
    print(Counter(actions))