import argparse, os, random, multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from agent_adrqn import ADRQN, ACTIONS

def env_worker(remote, obelix_py, args, seed):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.OBELIX(scaling_factor=args.scaling_factor, arena_size=args.arena_size, 
                     max_steps=args.max_steps, wall_obstacles=args.wall_obstacles,
                     difficulty=args.difficulty, box_speed=args.box_speed, seed=seed)
    
    SUCCESS_BONUS = 2000 
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                s, r, d = env.step(data, render=False)
                is_success = (r >= SUCCESS_BONUS)
                if d: s = env.reset()
                remote.send((s, r, d, is_success))
            elif cmd == "reset": 
                remote.send(env.reset())
            elif cmd == "close": 
                remote.close()
                break
        except EOFError:
            break

class PrioritizedReplay:
    def __init__(self, cap=300000):
        self.buffer = deque(maxlen=cap)
        self.good_buffer = deque(maxlen=cap // 5) 
        
    def add(self, seq): 
        self.buffer.append(seq)
        # If any step has scaled reward > 0.03 (indicates sensor hit or goal success)
        # Index 2 is the reward (s, a, r, s2, d)
        if any(step_data[2] > 0.03 for step_data in seq):
            self.good_buffer.append(seq)
            
    def sample(self, n):
        n_good = min(len(self.good_buffer), n // 4) # 25% from good experiences
        n_normal = n - n_good
        
        batch = []
        if n_good > 0:
            idx_good = np.random.choice(len(self.good_buffer), n_good, replace=False)
            batch.extend([self.good_buffer[i] for i in idx_good])
            
        idx_normal = np.random.choice(len(self.buffer), n_normal, replace=False)
        batch.extend([self.buffer[i] for i in idx_normal])
        random.shuffle(batch)
        
        s = torch.tensor(np.array([[t[0] for t in seq] for seq in batch])).float()
        a = torch.tensor(np.array([[t[1] for t in seq] for seq in batch])).long()
        r = torch.tensor(np.array([[t[2] for t in seq] for seq in batch])).float()
        s2 = torch.tensor(np.array([[t[3] for t in seq] for seq in batch])).float()
        d = torch.tensor(np.array([[t[4] for t in seq] for seq in batch])).float()
        return s, a, r, s2, d

def get_augmented_state(obs, last_a_idx, steps_since):
    a_one_hot = np.zeros(5)
    a_one_hot[last_a_idx] = 1.0
    time_feat = min(1.0, steps_since / 100.0)
    return np.concatenate([obs, a_one_hot, [time_feat]])

def save_plots(history, save_path):
    steps = [h['step'] for h in history]
    rewards = [h['reward'] for h in history]
    lengths = [h['length'] for h in history]
    successes = [1 if h['success'] else 0 for h in history]
    
    window = 20
    def moving_avg(data, w):
        if len(data) < w: return data
        return np.convolve(data, np.ones(w), 'valid') / w

    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(steps, rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) >= window:
        plt.plot(steps[window-1:], moving_avg(rewards, window), color='darkblue', linewidth=2, label=f'MA({window})')
    plt.title('Cumulative Reward per Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(steps, lengths, alpha=0.3, color='orange', label='Raw')
    if len(lengths) >= window:
        plt.plot(steps[window-1:], moving_avg(lengths, window), color='darkorange', linewidth=2, label=f'MA({window})')
    plt.title('Episode Length (Steps)')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    if len(successes) >= window:
        plt.plot(steps[window-1:], moving_avg(successes, window) * 100, color='green', linewidth=2)
    plt.title(f'Success Rate % (Moving Window {window})')
    plt.xlabel('Global Steps')
    plt.ylabel('Success %')
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=6)
    parser.add_argument("--total_steps", type=int, default=5000000)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    q_net = ADRQN()
    target_net = ADRQN()
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    buffer = PrioritizedReplay()

    remotes, work_remotes = zip(*[mp.Pipe() for _ in range(args.num_envs)])
    workers = [mp.Process(target=env_worker, args=(wr, args.obelix_py, args, 42+i)) for i, wr in enumerate(work_remotes)]
    for w in workers: w.start()

    for r in remotes: r.send(("reset", None))
    raw_states = [r.recv() for r in remotes]
    
    # State tracking arrays
    env_hiddens = [None] * args.num_envs
    histories = [deque(maxlen=10) for _ in range(args.num_envs)]
    last_actions = [2] * args.num_envs # 2 is FW
    steps_since_seen = [0] * args.num_envs
    
    ep_rewards = [0.0] * args.num_envs
    ep_steps = [0] * args.num_envs
    recent_rewards = deque(maxlen=300)
    recent_losses = deque(maxlen=1000)
    success_history = deque(maxlen=500)

    behaviour_log = []

    steps, eps = 0, 1.0
    print(f"Starting Training: {args.total_steps} steps on {args.num_envs} envs.")

    while steps < args.total_steps:
        actions_idx = []
        aug_states = []
        
        # Prepare states and select actions
        with torch.no_grad():
            for i in range(args.num_envs):
                # Update visual tracking
                if np.any(raw_states[i][:17] > 0):
                    steps_since_seen[i] = 0
                else:
                    steps_since_seen[i] += 1
                
                aug_s = get_augmented_state(raw_states[i], last_actions[i], steps_since_seen[i])
                aug_states.append(aug_s)
                
                if random.random() < eps: 
                    # Escaper: Force turn if stuck
                    if bool(raw_states[i][17]):
                        actions_idx.append(random.choice([0, 4]))
                    else:
                        actions_idx.append(random.randint(0, 4))
                else:
                    s_t = torch.tensor(aug_s).float().view(1, 1, -1)
                    q, env_hiddens[i] = q_net(s_t, env_hiddens[i])
                    actions_idx.append(q.argmax().item())

        for r, i in zip(remotes, actions_idx): 
            r.send(("step", ACTIONS[i]))
        
        results = [r.recv() for r in remotes]
        
        for i, (s2_raw, rew, done, was_success) in enumerate(results):
            ep_rewards[i] += rew
            ep_steps[i] += 1
            
            # Neural network stability scaling
            shaped_rew = rew / 100.0
            
            # Exploration Heuristic: Encourage FW movement if not stuck and box is missing
            if actions_idx[i] == 2 and not bool(s2_raw[17]) and steps_since_seen[i] > 0:
                shaped_rew += 0.005 # Small bonus to break "spinning" tie
            
            # Lookahead state generation for storage
            next_steps_since = 0 if np.any(s2_raw[:17] > 0) else steps_since_seen[i] + 1
            aug_s2 = get_augmented_state(s2_raw, actions_idx[i], next_steps_since)
            
            histories[i].append((aug_states[i], actions_idx[i], shaped_rew, aug_s2, done))
            if len(histories[i]) == 10: 
                buffer.add(list(histories[i]))
            
            raw_states[i] = s2_raw
            last_actions[i] = actions_idx[i]
            
            if done: 
                recent_rewards.append(ep_rewards[i])
                success_history.append(1.0 if was_success else 0.0)
                behaviour_log.append({
                    'step': steps,
                    'reward': ep_rewards[i],
                    'length': ep_steps[i],
                    'success': was_success
                })
                ep_rewards[i] = 0.0
                ep_steps[i] = 0
                env_hiddens[i] = None
                steps_since_seen[i] = 0
                last_actions[i] = 2

        steps += args.num_envs
        eps = max(0.03, 1.0 - (steps / (args.total_steps * 0.85)))

        if len(buffer.buffer) > 1000 and steps % 16 == 0:
            s, a, r, s2, d = buffer.sample(32)
            with torch.no_grad():
                qn, _ = q_net(s2)
                max_a = qn.argmax(dim=-1, keepdim=True)
                qt, _ = target_net(s2)
                y = r + 0.995 * (1-d) * qt.gather(-1, max_a).squeeze(-1)
            q_v, _ = q_net(s)
            loss = nn.SmoothL1Loss()(q_v.gather(-1, a.unsqueeze(-1)).squeeze(-1), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            recent_losses.append(loss.item())

        if steps % 2500 < args.num_envs:
            target_net.load_state_dict(q_net.state_dict())
            
        if steps % 10000 < args.num_envs:
            avg_rew = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            sr = np.mean(success_history) * 100 if success_history else 0.0
            print(f"Step {steps}/{args.total_steps} | Eps {eps:.2f} | AvgEpRew {avg_rew:.1f} | Loss {avg_loss:.4f} | Success {sr:.1f}%")

    print("Training finished. Saving weights and plots...")
    save_dir = "submission_adrqn.py"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(q_net.state_dict(), os.path.join(save_dir, "weights.pth"))
    
    plot_path = os.path.join(save_dir, "behaviour_plots.png")
    if behaviour_log:
        save_plots(behaviour_log, plot_path)
        print(f"Plots saved to {plot_path}")

    for r in remotes:
        try: r.send(("close", None))
        except: pass
    for w in workers:
        w.terminate()
        w.join()
    print("Done.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True); train()