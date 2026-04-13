"""
R-D3QN Agent for OBELIX.
Handles 18-bit POMDP observations using an LSTM memory.
"""
from __future__ import annotations
import os, torch, numpy as np
import torch.nn as nn
from typing import List, Optional

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class R_D3QN(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Dueling Heads
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None):
        # x shape: (batch, seq_len, in_dim)
        feat = self.feature(x)
        lstm_out, hidden = self.lstm(feat, hidden)
        
        val = self.value_head(lstm_out)
        adv = self.adv_head(lstm_out)
        
        # Dueling combination
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q, hidden

_model: Optional[R_D3QN] = None
_hidden: Optional[tuple] = None
_step_count: int = 0

def _load_once():
    global _model
    if _model is not None: return
    _model = R_D3QN()
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if os.path.exists(wpath):
        sd = torch.load(wpath, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        _model.load_state_dict(sd)
    _model.eval()

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden, _step_count
    _load_once()
    
    # Reset hidden state if it's likely a new episode (step 0)
    # Note: In some evaluation scripts, we manually reset this if env.reset is detected
    if _step_count == 0 or _step_count > 2000:
        _hidden = None
    
    # Prep observation: (Batch=1, Seq=1, Dim=18)
    x = torch.from_numpy(obs).float().view(1, 1, -1)
    
    # Forward pass with LSTM memory
    q_out, _hidden = _model(x, _hidden)
    
    # Greedy action selection
    best_idx = int(q_out[0, -1].argmax().item())
    
    _step_count += 1
    if _step_count > 2000: _step_count = 0 # Horizon limit [cite: 142]
    
    return ACTIONS[best_idx]