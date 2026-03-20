import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

# --- AMD CPU STABILITY FIX ---
# Set to 2 or 4 instead of max to prevent "starving" the OBELIX engine
torch.set_num_threads(2) 
device = torch.device("cpu")

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx
        if s <= self.tree[left]: return self._retrieve(left, s)
        else: return self._retrieve(right, s - self.tree[left])

    def total(self): return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity: self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

Transition = namedtuple('Transition', ('s', 'a', 'r', 's2', 'done'))

class PrioritizedReplay:
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4):
        self.tree = SumTree(capacity)
        self.alpha, self.beta = alpha, beta
        self.epsilon = 0.01

    def add(self, s, a, r, s2, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        p = max_p if max_p > 0 else 1.0
        self.tree.add(p, Transition(s, a, r, s2, done))

    def sample(self, n):
        batch, idxs, weights = [], [], []
        segment = self.tree.total() / n
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            weights.append((1.0 / (self.tree.n_entries * p / self.tree.total()))**self.beta)
        
        weights = np.array(weights) / max(weights)
        s = torch.FloatTensor(np.array([t.s for t in batch]))
        a = torch.LongTensor([t.a for t in batch])
        r = torch.FloatTensor([t.r for t in batch])
        s2 = torch.FloatTensor(np.array([t.s2 for t in batch]))
        done = torch.FloatTensor([float(t.done) for t in batch])
        w = torch.FloatTensor(weights)
        return s, a, r, s2, done, w, idxs

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        f = self.feature(x)
        v, a = self.value(f), self.adv(f)
        return v + (a - a.mean(dim=-1, keepdim=True))

def import_obelix(obelix_py):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    q_net = DuelingDQN().to(device)
    tgt_net = DuelingDQN().to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=5e-4)
    replay = PrioritizedReplay(capacity=50000)
    
    total_steps, best_ret = 0, -float('inf')
    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

    print("--- SYSTEM START: Environment and Networks Loaded ---")

    for ep in range(args.episodes):
        env = OBELIX(scaling_factor=args.scaling_factor, arena_size=args.arena_size, max_steps=args.max_steps)
        s = env.reset(seed=ep)
        ep_ret = 0
        
        for t in range(args.max_steps):
            eps = max(0.05, 1.0 - total_steps / 200000)
            if random.random() < eps:
                a_idx = random.randint(0, 4)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(s).unsqueeze(0))
                    a_idx = q_vals.argmax().item()
            
            s2, r, done = env.step(ACTIONS[a_idx], render=False)
            replay.add(s, a_idx, r, s2, done)
            s, ep_ret, total_steps = s2, ep_ret + r, total_steps + 1

            # FIX: Use n_entries so we don't train on empty data
            if replay.tree.n_entries >= args.batch and total_steps > 1000:
                s_b, a_b, r_b, s2_b, d_b, w_b, idxs = replay.sample(args.batch)
                with torch.no_grad():
                    next_a = q_net(s2_b).argmax(1).unsqueeze(1)
                    next_q = tgt_net(s2_b).gather(1, next_a).squeeze()
                    target = r_b + 0.99 * (1 - d_b) * next_q
                
                curr_q = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze()
                td_error = target - curr_q
                loss = (w_b * nn.functional.smooth_l1_loss(curr_q, target, reduction='none')).mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                replay.update_priorities(idxs, td_error.detach().numpy())

                if total_steps % 2000 == 0:
                    tgt_net.load_state_dict(q_net.state_dict())
            if done: break

        # LOG EVERY EPISODE FOR DEBUGGING
        print(f"EP {ep+1:3d} | Steps: {total_steps:6d} | Ret: {ep_ret:7.1f} | Buffer: {replay.tree.n_entries:5d}")

        if ep_ret > best_ret:
            best_ret = ep_ret
            torch.save(q_net.state_dict(), "weights.pth")

if __name__ == "__main__":
    main()