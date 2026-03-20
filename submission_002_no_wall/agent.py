"""D3QN-PER Agent for OBELIX (Dueling Double DQN + Prioritized Experience Replay)

This agent is *evaluation-only*: it loads pretrained weights from a file
placed next to agent.py inside the submission zip (weights.pth).

Architecture improvements over vanilla DQN:
- Dueling architecture: separate value and advantage streams
- Double DQN target computation for reduced overestimation
- Greedy evaluation with action smoothing for stability

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class DuelingDQN(nn.Module):
    """Dueling DQN architecture separates state value and action advantages.
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """
    def __init__(self, in_dim: int = 18, n_actions: int = 5, hidden_dim: int = 64):
        super().__init__()
        
        # Shared feature extractor (UPDATED: Added LayerNorm to match training script)
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages with mean normalization
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

_model: Optional[DuelingDQN] = None
_last_action: Optional[int] = None
_repeat_count: int = 0

# Hyperparameters for action smoothing
_MAX_REPEAT = 3
_CLOSE_Q_DELTA = 0.08

def _load_once():
    """Load model weights once on first call."""
    global _model
    if _model is not None:
        return
    
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            f"weights.pth not found at {wpath}. "
            "Train offline and include it in the submission zip."
        )
    
    m = DuelingDQN()
    sd = torch.load(wpath, map_location="cpu")
    
    # Handle different save formats
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m
    print(f"[D3QN-PER Agent] Loaded weights from {wpath}")

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Greedy policy with action smoothing for stable evaluation."""
    global _last_action, _repeat_count
    
    _load_once()
    
    # Forward pass through network
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()
    best = int(np.argmax(q))
    
    # Action smoothing: if top-2 Q-values are close, prefer previous action
    if _last_action is not None:
        order = np.argsort(-q)
        best_q = float(q[order[0]])
        second_q = float(q[order[1]])
        
        if (best_q - second_q) < _CLOSE_Q_DELTA:
            # Q-values are close, check if we should repeat previous action
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                # Exceeded max repeats, take new action and reset counter
                _repeat_count = 0
        else:
            # Clear winner, reset counter
            _repeat_count = 0
    
    _last_action = best
    return ACTIONS[best]