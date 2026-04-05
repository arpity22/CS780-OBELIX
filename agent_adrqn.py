import os, torch, numpy as np
import torch.nn as nn
from typing import List, Optional

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class ADRQN(nn.Module):
    """
    Action-Specific Deep Recurrent Q-Network.
    Input dimension (24) = 18 (Sensors) + 5 (Last Action One-Hot) + 1 (Time Since Seen)
    """
    def __init__(self, in_dim: int = 24, n_actions: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, hidden=None):
        if x.dim() == 2: x = x.unsqueeze(1)
            
        feat = self.feature(x)
        lstm_out, hidden = self.lstm(feat, hidden)
        
        val = self.value_head(lstm_out)
        adv = self.adv_head(lstm_out)
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q, hidden

_model = None
_hidden = None
_last_action_idx = 2 # Default to FW (Index 2)
_steps_since_seen = 0
_step_count = 0

def _load_once():
    global _model
    if _model is not None: return
    _model = ADRQN(in_dim=24, n_actions=5, hidden_dim=128)
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if os.path.exists(wpath):
        sd = torch.load(wpath, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        _model.load_state_dict(sd)
    _model.eval()

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden, _last_action_idx, _steps_since_seen, _step_count
    _load_once()
    
    # Reset temporal features at the start of a new episode
    if _step_count == 0:
        _hidden = None
        _last_action_idx = 2
        _steps_since_seen = 0

    # POMDP Feature Extraction: Track visibility for blinking/moving box
    if np.any(obs[:17] > 0):
        _steps_since_seen = 0
    else:
        _steps_since_seen += 1

    # Augment observation vector
    a_one_hot = np.zeros(5)
    a_one_hot[_last_action_idx] = 1.0
    time_feat = min(1.0, _steps_since_seen / 100.0)
    
    aug_obs = np.concatenate([obs, a_one_hot, [time_feat]])
    x_tensor = torch.from_numpy(aug_obs).float().view(1, 1, -1)
    
    # Forward Pass through LSTM
    q_out, _hidden = _model(x_tensor, _hidden)
    best_idx = int(q_out[0, -1].argmax().item())
    
    # Update state trackers
    _last_action_idx = best_idx
    _step_count += 1
    if _step_count >= 2000: _step_count = 0
    
    return ACTIONS[best_idx]