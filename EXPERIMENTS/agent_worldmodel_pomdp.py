import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 5,
        hidden_dim: int = 192,
        latent_dim: int = 96,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
        )

        self.filter_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.posterior_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def posterior(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.obs_encoder(obs)
        h = self.filter_gru(emb, h_prev)
        z = self.posterior_proj(h)
        return h, z


class Actor(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


_wm: Optional[WorldModel] = None
_actor: Optional[Actor] = None
_h: Optional[torch.Tensor] = None
_z: Optional[torch.Tensor] = None

_prev_obs: Optional[np.ndarray] = None
_ep_step = 0
_steps_since_seen = 0
_search_turn_dir = 1
_stuck_recovery_count = 0

_MAX_EP_STEPS_BEFORE_RESET = 1200


def _reset_state(obs: np.ndarray) -> None:
    global _h, _z, _prev_obs, _ep_step, _steps_since_seen, _search_turn_dir, _stuck_recovery_count

    _h, _z = _wm.init_state(1, torch.device("cpu"))
    _prev_obs = obs.copy()
    _ep_step = 0
    _steps_since_seen = 0
    _search_turn_dir = 1
    _stuck_recovery_count = 0


def _likely_episode_reset(obs: np.ndarray) -> bool:
    if _prev_obs is None:
        return True
    if _ep_step < 12:
        return False
    jump = float(np.sum(np.abs(obs - _prev_obs)))
    return jump >= 7.0


def _stuck_recovery_action(obs: np.ndarray) -> Optional[int]:
    global _search_turn_dir, _stuck_recovery_count

    if bool(obs[17] > 0.5):
        _stuck_recovery_count += 1
        # Stuck may persist through turn-only actions; force turn-turn-forward cycles.
        phase = _stuck_recovery_count % 4
        if phase == 2:
            _search_turn_dir *= -1
        if phase in (1, 2):
            return 0 if _search_turn_dir > 0 else 4
        return 2

    _stuck_recovery_count = 0
    return None


def _structured_search_action() -> int:
    global _search_turn_dir
    if (_ep_step % 5) == 0:
        _search_turn_dir *= -1
    if (_ep_step % 5) in (0, 1):
        return 1 if _search_turn_dir > 0 else 3
    return 2


def _load_once() -> None:
    global _wm, _actor, _h, _z

    if _wm is not None and _actor is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent_worldmodel_pomdp.py.")

    payload = torch.load(wpath, map_location="cpu")

    obs_dim = int(payload.get("obs_dim", 18))
    action_dim = int(payload.get("action_dim", 5))
    hidden_dim = int(payload.get("hidden_dim", 192))
    latent_dim = int(payload.get("latent_dim", 96))

    wm = WorldModel(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    actor = Actor(feat_dim=hidden_dim + latent_dim, n_actions=action_dim)

    if "world_model_state_dict" in payload and "actor_state_dict" in payload:
        wm.load_state_dict(payload["world_model_state_dict"], strict=True)
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
    elif "state_dict" in payload:
        # Minimal compatibility fallback.
        actor.load_state_dict(payload["state_dict"], strict=False)
    else:
        raise RuntimeError("Unsupported checkpoint format for world-model agent.")

    wm.eval()
    actor.eval()

    _wm = wm
    _actor = actor
    _h, _z = _wm.init_state(1, torch.device("cpu"))


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _h, _z, _prev_obs, _ep_step, _steps_since_seen

    _load_once()

    obs = np.asarray(obs, dtype=np.float32)
    if _ep_step == 0 or _likely_episode_reset(obs):
        _reset_state(obs)

    # Posterior update with current observation.
    obs_t = torch.from_numpy(obs).float().view(1, -1)
    _h, _z = _wm.posterior(obs_t, _h)

    if np.any(obs[:17] > 0):
        _steps_since_seen = 0
    else:
        _steps_since_seen += 1

    # First apply hard safety recovery if stuck.
    action_idx = _stuck_recovery_action(obs)

    if action_idx is None:
        sees_target = bool(np.any(obs[:17] > 0))
        front_active = bool(np.any(obs[4:12] > 0))

        if not sees_target and _steps_since_seen > 35:
            action_idx = _structured_search_action()
        else:
            feat = torch.cat([_h, _z], dim=-1)
            logits = _actor(feat)[0]
            if front_active:
                action_idx = 2
            else:
                action_idx = int(torch.argmax(logits).item())

    _prev_obs = obs.copy()
    _ep_step += 1

    if _ep_step >= _MAX_EP_STEPS_BEFORE_RESET:
        _ep_step = 0

    return ACTIONS[action_idx]


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    fake = np.zeros(18, dtype=np.float32)
    try:
        print("Smoke test action:", policy(fake, rng))
    except FileNotFoundError as exc:
        print(exc)
