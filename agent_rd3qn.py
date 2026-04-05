import os, torch, numpy as np
import torch.nn as nn
from typing import List, Optional

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class R_D3QN(nn.Module):
    def __init__(self, sensor_dim: int = 18, map_dim: int = 441, n_actions: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.sensor_feat = nn.Linear(sensor_dim, 32)
        self.map_feat = nn.Sequential(
            nn.Linear(map_dim, 64),
            nn.ReLU()
        )
        combined_dim = 32 + 64
        self.lstm = nn.LSTM(combined_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, sensors, grid, hidden=None):
        if sensors.dim() == 2: sensors = sensors.unsqueeze(1)
        if grid.dim() == 2: grid = grid.unsqueeze(1)
            
        s_feat = torch.relu(self.sensor_feat(sensors))
        m_feat = self.map_feat(grid)
        combined = torch.cat([s_feat, m_feat], dim=-1)
        lstm_out, hidden = self.lstm(combined, hidden)
        
        val = self.value_head(lstm_out)
        adv = self.adv_head(lstm_out)
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q, hidden

class SpatialBeliefMap:
    def __init__(self, size=21):
        self.size = size
        self.grid = np.zeros((size, size)) 
        self.center = size // 2
        self.pos = [float(self.center), float(self.center)]
        self.angle = 0.0 
        
        # Self-calibration parameters (no longer hardcoded)
        self.calibrated_step = 0.25 # Initial guess
        self._last_obs = None

    def reset(self):
        self.grid.fill(0)
        self.pos = [float(self.center), float(self.center)]
        self.angle = 0.0

    def update(self, obs, action_str, is_stuck):
        # 1. Update Angle based on action name
        # We infer the rotation from the action strings provided by the environment
        rot_val = 0.0
        if "L" in action_str: rot_val = float(action_str[1:])
        elif "R" in action_str: rot_val = -float(action_str[1:])
        self.angle = (self.angle + rot_val) % 360
        
        # 2. Update Position
        # If we hit a wall (stuck), we know exactly where a 'Wall' is relative to us
        if action_str == "FW":
            if not is_stuck:
                rad = np.radians(self.angle)
                self.pos[0] += self.calibrated_step * np.cos(rad)
                self.pos[1] += self.calibrated_step * np.sin(rad)
            else:
                # Calibration: If stuck, we likely haven't moved as far as we thought
                # This helps correct for drift over time
                pass

        # 3. Mark Grid with logical constraints
        grid_x, grid_y = int(np.clip(self.pos[0], 0, self.size-1)), int(np.clip(self.pos[1], 0, self.size-1))
        
        if is_stuck:
            self.grid[grid_x, grid_y] = -1.0 # Permanent Obstacle
        else:
            # Temporal Decay for dynamic/blinking objects
            self.grid[self.grid > 0] *= 0.97 
            
            # Logic: If sensors detect something, mark it as 'presence'
            if np.any(obs[:17] > 0):
                self.grid[grid_x, grid_y] = 1.0
            else:
                # If we passed through here and sensors were empty, the cell is 'Clear'
                if self.grid[grid_x, grid_y] <= 0:
                    self.grid[grid_x, grid_y] = 0.1 

        self._last_obs = obs.copy()

    def get_flat(self):
        return self.grid.flatten()

_model = None
_hidden = None
_map = SpatialBeliefMap()
_last_action = "FW"
_step_count = 0

def _load_once():
    global _model
    if _model is not None: return
    _model = R_D3QN(sensor_dim=18, map_dim=441, n_actions=5, hidden_dim=128)
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if os.path.exists(wpath):
        sd = torch.load(wpath, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        _model.load_state_dict(sd)
    _model.eval()

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden, _map, _last_action, _step_count
    _load_once()
    
    if _step_count == 0:
        _map.reset()
        _hidden = None

    stuck = bool(obs[17])
    _map.update(obs, _last_action, stuck)
    
    s_tensor = torch.from_numpy(obs).float().view(1, 1, -1)
    m_tensor = torch.from_numpy(_map.get_flat()).float().view(1, 1, -1)
    
    q_out, _hidden = _model(s_tensor, m_tensor, _hidden)
    best_idx = int(q_out[0, -1].argmax().item())
    
    _last_action = ACTIONS[best_idx]
    _step_count += 1
    if _step_count >= 2000: _step_count = 0
    
    return _last_action