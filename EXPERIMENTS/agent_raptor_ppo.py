import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class RecurrentActorCritic(nn.Module):
    def __init__(self, in_dim: int = 43, n_actions: int = 5, hidden_dim: int = 192):
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
        self.dynamics_head = nn.Linear(hidden_dim, 18)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def step(self, obs_t: torch.Tensor, hidden: torch.Tensor):
        z = self.encoder(obs_t)
        z, hidden = self.gru(z.unsqueeze(0), hidden)
        z = z.squeeze(0)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        pred_next_obs = self.dynamics_head(z)
        return logits, value, pred_next_obs, hidden


_model: Optional[RecurrentActorCritic] = None
_hidden: Optional[torch.Tensor] = None
_last_action_idx = 2
_prev_obs: Optional[np.ndarray] = None
_steps_since_seen = 0
_ep_step = 0
_MAX_EP_STEPS_BEFORE_RESET = 700
_search_turn_dir = 1
_stuck_recovery_count = 0


def _augment_obs(obs: np.ndarray) -> np.ndarray:
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

    aug = np.concatenate([obs, delta, one_hot, [seen_feat, progress_feat]])
    return aug.astype(np.float32)


def _soft_reset(obs: np.ndarray) -> None:
    global _hidden, _last_action_idx, _prev_obs, _steps_since_seen, _ep_step
    global _search_turn_dir, _stuck_recovery_count
    _hidden = _model.initial_hidden(1, torch.device("cpu"))
    _last_action_idx = 2
    _prev_obs = obs.copy()
    _steps_since_seen = 0
    _ep_step = 0
    _search_turn_dir = 1
    _stuck_recovery_count = 0


def _likely_episode_reset(obs: np.ndarray) -> bool:
    if _prev_obs is None:
        return True
    if _ep_step < 12:
        return False
    # New episodes usually produce large observation jumps from random respawn.
    jump = float(np.sum(np.abs(obs - _prev_obs)))
    return jump >= 7.0


def _select_action_deterministic(logits: torch.Tensor, obs: np.ndarray) -> int:
    # Match train_raptor_ppo.py aligned eval behavior with deterministic recovery/search.
    global _search_turn_dir, _stuck_recovery_count

    if bool(obs[17] > 0.5):
        _stuck_recovery_count += 1
        phase = _stuck_recovery_count % 4
        if phase == 2:
            _search_turn_dir *= -1
        if phase in (1, 2):
            return 0 if _search_turn_dir > 0 else 4
        return 2

    _stuck_recovery_count = 0

    sees_target = bool(np.any(obs[:17] > 0))
    front_active = bool(np.any(obs[4:12] > 0))

    if (not sees_target) and _steps_since_seen > 35:
        if (_steps_since_seen % 7) == 0:
            _search_turn_dir *= -1
        if (_steps_since_seen % 5) in (0, 1):
            return 1 if _search_turn_dir > 0 else 3
        return 2

    if front_active:
        return 2

    return int(torch.argmax(logits).item())


def _load_once() -> None:
    global _model, _hidden

    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent_raptor_ppo.py."
        )

    state = torch.load(wpath, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
        in_dim = int(state.get("obs_dim", 43))
        hidden_dim = int(state.get("hidden_dim", 192))
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        in_dim = 43
        hidden_dim = 192
    else:
        sd = state
        in_dim = 43
        hidden_dim = 192

    model = RecurrentActorCritic(in_dim=in_dim, n_actions=5, hidden_dim=hidden_dim)
    model.load_state_dict(sd, strict=True)
    model.eval()

    _model = model
    _hidden = _model.initial_hidden(1, torch.device("cpu"))


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hidden, _last_action_idx, _prev_obs, _ep_step, _steps_since_seen

    _load_once()

    if _ep_step == 0 or _likely_episode_reset(obs):
        _soft_reset(obs)

    aug_obs = _augment_obs(obs)
    obs_t = torch.from_numpy(aug_obs).float().view(1, -1)

    logits, _, _, _hidden = _model.step(obs_t, _hidden)
    action_idx = _select_action_deterministic(logits[0], obs)

    _last_action_idx = action_idx
    _prev_obs = obs.copy()
    _ep_step += 1

    # Evaluator does not provide done signal; hard-reset recurrent state periodically.
    if _ep_step >= _MAX_EP_STEPS_BEFORE_RESET:
        _ep_step = 0

    return ACTIONS[action_idx]


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    fake_obs = np.zeros(18, dtype=np.float32)
    try:
        act = policy(fake_obs, rng)
        print(f"Smoke test action: {act}")
    except FileNotFoundError as exc:
        print(exc)
