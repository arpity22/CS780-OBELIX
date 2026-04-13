import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
ACTION_YAW_DEG = np.array([45.0, 22.5, 0.0, -22.5, -45.0], dtype=np.float32)

PHASE_FIND = 0
PHASE_PUSH = 1
PHASE_UNWEDGE = 2
N_PHASES = 3
MAP_CHANNELS_DEFAULT = 4


class RecurrentActorCritic(nn.Module):
    def __init__(self, in_dim: int, n_actions: int = 5, hidden_dim: int = 192):
        super().__init__()
        self.in_dim = in_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 192),
            nn.SiLU(),
        )
        self.gru = nn.GRU(192, hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(self, obs_t: torch.Tensor, hidden: torch.Tensor):
        z = self.encoder(obs_t)
        z, hidden = self.gru(z.unsqueeze(0), hidden)
        z = z.squeeze(0)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value, hidden


_model: Optional[RecurrentActorCritic] = None
_hidden: Optional[torch.Tensor] = None

_obs_dim = 71
_hidden_dim = 192
_map_angle_bins = 16
_map_channels = MAP_CHANNELS_DEFAULT
_map_params: Dict[str, float] = {
    "map_ema_alpha": 0.08,
    "map_visit_cap": 128.0,
    "map_novelty_coef": 0.35,
    "map_frontier_coef": 0.20,
    "map_sensor_coef": 0.02,
    "map_stuck_coef": 0.90,
}

_last_action_idx = 2
_prev_obs: Optional[np.ndarray] = None
_steps_since_seen = 0
_ep_step = 0
_MAX_EP_STEPS_BEFORE_RESET = 900

_search_turn_dir = 1
_stuck_recovery_count = 0
_heading_deg = 0.0

_map_state: Optional[np.ndarray] = None
_sensor_seen: Optional[np.ndarray] = None


def heading_to_bin(heading_deg: float, bins: int) -> int:
    return int(((heading_deg % 360.0) / 360.0) * bins) % bins


def one_hot_phase(phase: int) -> np.ndarray:
    out = np.zeros(N_PHASES, dtype=np.float32)
    out[phase] = 1.0
    return out


def augment_obs(obs: np.ndarray) -> np.ndarray:
    global _last_action_idx, _prev_obs, _steps_since_seen, _ep_step

    if _prev_obs is None:
        _prev_obs = obs.copy()

    if np.any(obs[:17] > 0):
        _steps_since_seen = 0
    else:
        _steps_since_seen += 1

    delta = obs - _prev_obs

    one_hot = np.zeros(5, dtype=np.float32)
    one_hot[_last_action_idx] = 1.0

    seen_feat = min(1.0, _steps_since_seen / 120.0)
    progress_feat = min(1.0, _ep_step / 1000.0)

    return np.concatenate([obs, delta, one_hot, [seen_feat, progress_feat]], axis=0).astype(np.float32)


def map_row() -> np.ndarray:
    return _map_state[heading_to_bin(_heading_deg, _map_angle_bins)]


def map_features() -> np.ndarray:
    row = map_row()
    visits = row[:, 0]
    rew = row[:, 1]
    stuck = row[:, 2]
    succ = row[:, 3]

    vnorm = visits / (1.0 + np.max(visits))
    rew_t = np.tanh(rew / 5.0)
    stuck_c = np.clip(stuck, 0.0, 1.0)
    succ_c = np.clip(succ, 0.0, 1.0)

    all_visits = _map_state[:, :, 0]
    explored_ratio = float(np.mean(all_visits > 0.0))
    hbin = heading_to_bin(_heading_deg, _map_angle_bins)
    heading_visits = float(np.sum(_map_state[hbin, :, 0]))
    total_visits = float(np.sum(all_visits)) + 1e-6
    heading_visit_ratio = heading_visits / total_visits
    heading_risk = float(np.mean(stuck))
    heading_rew = float(np.mean(rew_t))
    fw_novelty = 1.0 / np.sqrt(1.0 + float(visits[2]))

    extras = np.array(
        [explored_ratio, heading_visit_ratio, heading_risk, heading_rew, fw_novelty],
        dtype=np.float32,
    )
    return np.concatenate([vnorm, rew_t, stuck_c, succ_c, extras], axis=0).astype(np.float32)


def determine_phase(obs: np.ndarray) -> int:
    row = map_row()
    stuck = bool(obs[17] > 0.5)
    sees_target = bool(np.any(obs[:17] > 0))
    front_active = bool(np.any(obs[4:12] > 0))
    fw_risk = float(row[2, 2])

    if stuck or fw_risk > 0.85:
        return PHASE_UNWEDGE
    if front_active and sees_target:
        return PHASE_PUSH
    if (not sees_target) or _steps_since_seen > 25:
        return PHASE_FIND
    return PHASE_PUSH


def deterministic_phase_action(logits: torch.Tensor, obs: np.ndarray, phase: int) -> int:
    global _search_turn_dir, _stuck_recovery_count

    row = map_row()

    if phase == PHASE_UNWEDGE or bool(obs[17] > 0.5):
        _stuck_recovery_count += 1
        cycle = _stuck_recovery_count % 4
        if cycle == 2:
            _search_turn_dir *= -1
        if cycle in (1, 2):
            return 0 if _search_turn_dir > 0 else 4
        return 2

    _stuck_recovery_count = 0

    sees_target = bool(np.any(obs[:17] > 0))
    front_active = bool(np.any(obs[4:12] > 0))

    if phase == PHASE_FIND:
        if _steps_since_seen > 35:
            left_score = float(np.mean(row[[0, 1], 0] + 2.5 * row[[0, 1], 2]))
            right_score = float(np.mean(row[[3, 4], 0] + 2.5 * row[[3, 4], 2]))
            if left_score + 0.02 < right_score:
                _search_turn_dir = 1
            elif right_score + 0.02 < left_score:
                _search_turn_dir = -1

            if (_steps_since_seen % 5) in (0, 1):
                return 1 if _search_turn_dir > 0 else 3
            return 2

        if (not sees_target) and float(row[2, 2]) > 0.55:
            return 1 if _search_turn_dir > 0 else 3
        return int(torch.argmax(logits).item())

    if front_active and (not bool(obs[17] > 0.5)):
        return 2
    return int(torch.argmax(logits).item())


def update_map_from_previous_outcome(obs: np.ndarray) -> None:
    global _map_state, _sensor_seen

    if _map_state is None or _sensor_seen is None:
        return

    # Update map with outcome of previous action using observable signals.
    action_idx = int(_last_action_idx)
    hbin = heading_to_bin(_heading_deg, _map_angle_bins)
    cell = _map_state[hbin, action_idx]

    visit_before = float(cell[0])
    alpha = float(_map_params.get("map_ema_alpha", 0.08))

    cell[0] = min(float(_map_params.get("map_visit_cap", 128.0)), visit_before + 1.0)

    stuck = bool(obs[17] > 0.5)

    sensor_now = obs[:17] > 0.5
    new_bits = np.logical_and(sensor_now, np.logical_not(_sensor_seen))
    _sensor_seen = np.logical_or(_sensor_seen, sensor_now)

    pseudo_rew = -0.01
    pseudo_rew += 0.02 * float(np.sum(new_bits))
    if action_idx == 2 and np.any(obs[4:12] > 0) and (not stuck):
        pseudo_rew += 0.05
    if stuck:
        pseudo_rew -= 2.0

    cell[1] = (1.0 - alpha) * float(cell[1]) + alpha * float(pseudo_rew)
    cell[2] = (1.0 - alpha) * float(cell[2]) + alpha * (1.0 if stuck else 0.0)
    cell[3] = (1.0 - alpha) * float(cell[3]) + alpha * 0.0


def soft_reset(obs: np.ndarray) -> None:
    global _hidden, _last_action_idx, _prev_obs, _steps_since_seen, _ep_step
    global _search_turn_dir, _stuck_recovery_count, _heading_deg
    global _map_state, _sensor_seen

    _hidden = _model.initial_hidden(1, torch.device("cpu"))
    _last_action_idx = 2
    _prev_obs = obs.copy()
    _steps_since_seen = 0
    _ep_step = 0
    _search_turn_dir = 1
    _stuck_recovery_count = 0
    _heading_deg = 0.0

    _map_state = np.zeros((_map_angle_bins, len(ACTIONS), _map_channels), dtype=np.float32)
    _sensor_seen = np.zeros(17, dtype=bool)


def likely_episode_reset(obs: np.ndarray) -> bool:
    if _prev_obs is None:
        return True
    if _ep_step < 12:
        return False
    jump = float(np.sum(np.abs(obs - _prev_obs)))
    return jump >= 7.0


def load_once() -> None:
    global _model, _hidden, _obs_dim, _hidden_dim, _map_angle_bins, _map_channels, _map_params

    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent_explorer_map_ppo.py.")

    payload = torch.load(wpath, map_location="cpu")

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError("Unsupported checkpoint format for explorer-map agent.")

    _obs_dim = int(payload.get("obs_dim", 71))
    _hidden_dim = int(payload.get("hidden_dim", 192))
    _map_angle_bins = int(payload.get("map_angle_bins", 16))
    _map_channels = int(payload.get("map_channels", MAP_CHANNELS_DEFAULT))
    if isinstance(payload.get("map_params"), dict):
        _map_params = {**_map_params, **payload["map_params"]}

    model = RecurrentActorCritic(in_dim=_obs_dim, n_actions=5, hidden_dim=_hidden_dim)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    _model = model
    _hidden = _model.initial_hidden(1, torch.device("cpu"))


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden, _last_action_idx, _prev_obs, _ep_step, _steps_since_seen, _heading_deg

    load_once()

    obs = np.asarray(obs, dtype=np.float32)

    if _ep_step == 0 or likely_episode_reset(obs):
        soft_reset(obs)

    update_map_from_previous_outcome(obs)

    aug = augment_obs(obs)
    phase = determine_phase(obs)
    mfeat = map_features()
    pfeat = one_hot_phase(phase)

    model_in = np.concatenate([aug, mfeat, pfeat], axis=0).astype(np.float32)
    obs_t = torch.from_numpy(model_in).float().view(1, -1)

    logits, _, _hidden = _model.step(obs_t, _hidden)
    action_idx = deterministic_phase_action(logits[0], obs, phase)

    _last_action_idx = int(action_idx)
    _heading_deg = (_heading_deg + ACTION_YAW_DEG[_last_action_idx]) % 360.0

    _prev_obs = obs.copy()
    _ep_step += 1

    if _ep_step >= _MAX_EP_STEPS_BEFORE_RESET:
        _ep_step = 0

    return ACTIONS[_last_action_idx]


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    fake_obs = np.zeros(18, dtype=np.float32)
    try:
        print("Smoke test action:", policy(fake_obs, rng))
    except FileNotFoundError as exc:
        print(exc)
