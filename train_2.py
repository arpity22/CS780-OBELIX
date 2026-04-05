"""High-Speed Trainer: Accelerated DDQN for OBELIX."""

import argparse, random, os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class FastDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )
    def forward(self, x): return self.net(x)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    # FAST-TRAIN HYPERPARAMETERS
    ap.add_argument("--episodes", type=int, default=1000) # Target 1k episodes
    ap.add_argument("--lr", type=float, default=1e-3)     # Higher LR for faster learning
    ap.add_argument("--batch", type=int, default=64)     # Smaller batch = more frequent updates
    ap.add_argument("--eps_decay", type=int, default=50000) # Faster decay
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    q, tgt = FastDQN(), FastDQN()
    tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = deque(maxlen=20000) # Smaller buffer for fresher data
    
    steps, gamma = 0, 0.95 # Lower gamma for faster focus on immediate rewards

    for ep in range(args.episodes):
        env = OBELIX(scaling_factor=5, arena_size=500, max_steps=1000, seed=ep)
        s = env.reset(seed=ep)
        ep_ret = 0

        for _ in range(1000):
            eps = max(0.05, 1.0 - steps / args.eps_decay)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    a = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax().item()

            s2, r, done = env.step(ACTIONS[a])
            replay.append((s, a, r, s2, done))
            s, ep_ret, steps = s2, ep_ret + r, steps + 1

            if len(replay) >= args.batch:
                # Optimized sampling
                idx = np.random.choice(len(replay), args.batch)
                batch = [replay[i] for i in idx]
                sb, ab, rb, s2b, db = [np.array([x[i] for x in batch]) for i in range(5)]
                
                sb_t, ab_t, rb_t, s2b_t, db_t = map(lambda x: torch.tensor(x, dtype=torch.float32), (sb, ab, rb, s2b, db))
                
                with torch.no_grad():
                    best_a = q(s2b_t).argmax(dim=1, keepdim=True)
                    max_next_q = tgt(s2b_t).gather(1, best_a).squeeze()
                    y = rb_t + gamma * (1 - db_t) * max_next_q

                curr_q = q(sb_t).gather(1, ab_t.long().unsqueeze(1)).squeeze()
                loss = nn.functional.mse_loss(curr_q, y)
                
                opt.zero_grad(); loss.backward(); opt.step()

                if steps % 500 == 0: # Faster target synchronization
                    tgt.load_state_dict(q.state_dict())
            if done: break

        if (ep + 1) % 50 == 0:
            print(f"Ep {ep+1} | Ret: {ep_ret:.1f} | Steps: {steps}")

    torch.save(q.state_dict(), args.out)

if __name__ == "__main__":
    main()