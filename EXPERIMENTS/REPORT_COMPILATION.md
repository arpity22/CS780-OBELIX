# CS780-OBELIX Report Compilation Draft (Template-Aligned)

Generated on: 2026-04-11
Primary template followed: `Report_template/Report.tex`

This file compiles the available approach and result evidence from the repository to accelerate writing the final capstone report.

## Repository Coverage Snapshot

- Total files (excluding `.git`): 370
- Total directories (excluding `.git`): 32
- Dominant artifact types: 219 `.pth`, 55 `.png`, 52 `.py`, 13 `.zip`, 11 `.txt`

Top-level folder density (direct files only):

| Folder | Direct files |
|---|---:|
| `.` | 46 |
| `Report_template` | 3 |
| `submission_001` | 26 |
| `submission_001_no_wall` | 38 |
| `submission_001_with_wall` | 13 |
| `submission_002` | 9 |
| `submission_002_no_wall` | 29 |
| `submission_003` | 52 |
| `submission_FIXED_shaped` | 24 |
| `submission_phase1_heavy` | 17 |
| `submission_phase2_medium` | 15 |
| `submission_phase3_light` | 13 |
| `submission_phase4_final` | 13 |
| `submission_raptor_ppo` | 6 |
| (other submission folders) | 1-12 each |

## 1) Competition Result (Template Section)

### 1.1 Raw leaderboard evidence currently present

From `leaderboard.csv`:

| timestamp_utc | agent_name | mean_score | std_score | runs | max_steps | wall_obstacles | difficulty | box_speed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-11T10:26:20.778346+00:00 | agent_template | -13931.000000 | 8342.381594 | 3 | 200 | 1 | (not logged in this row) | (not logged in this row) |
| 2026-02-11T10:30:09.707166+00:00 | agent_template | -16100.666667 | 10503.018656 | 3 | 200 | 1 | 3 | 3 |

### 1.2 Required fields in template that still need manual fill

- Codabench Username: TODO
- Final leaderboard rank (Test Phase): TODO
- Leaderboard rank Level 1: TODO
- Leaderboard rank Level 2: TODO
- Leaderboard rank Level 3: TODO

### 1.3 Number of valid submissions (repo evidence)

Submission folders detected (29):

`submission_001`, `submission_001_no_wall`, `submission_001_with_wall`, `submission_002`, `submission_002_no_wall`, `submission_003`, `submission_adrqn.py`, `submission_debug_tmp`, `submission_FIXED_shaped`, `submission_improved_lstm`, `submission_phase1_heavy`, `submission_phase2_medium`, `submission_phase3_light`, `submission_phase4_final`, `submission_ppo_adaptive`, `submission_ppo_adaptive_v1`, `submission_ppo_adaptive_v2`, `submission_ppo_adv_final`, `submission_ppo_safe_efficient`, `submission_ppo_test`, `submission_ppo_test2`, `submission_raptor_ppo`, `submission_raptor_ppo1`, `submission_raptor_ppo2`, `submission_raptor_ppo3`, `submission_raptor_ppo4`, `submission_rd3qn`, `submission_rdq3qn.py`, `submission_worldmodel_pomdp1`

## 2) Introduction and Problem Description (Template Section)

Use these facts from the environment implementation (`obelix.py`):

- Core task framing: search/find box, attach, push, and terminate when attached box reaches boundary.
- Observation space: 18-dimensional binary/flag signal.
  - 16 sonar-related bits
  - 1 IR bit
  - 1 stuck flag (`obs[17]`)
- Difficulty setup:
  - `difficulty=0`: static box
  - `difficulty>=2`: blinking/visible-invisible box
  - `difficulty>=3`: moving + blinking box
- Optional obstacles: `wall_obstacles` flag.
- Episode limit: `max_steps` (varies by training script, often 1000 or 2000).
- Reward mechanics visible in code:
  - one-time sensor bonuses
  - per-step penalty `-1`
  - stuck penalty `-200`
  - attach transition bonus +100 (with immediate push-step adjustment)
  - success bonus +2000 when attached box touches boundary

POMDP justification for report: blinking box plus partial local sensing and stuck dynamics make latent-state reasoning important.

## 3) Approach and Model Evolution (Template Section)

### 3.1 High-level evolution inferred from scripts

1. Value-based baseline: Double DQN (`train_ddqn.py`, `train_2.py`)
2. Enhanced value learning: Dueling + Double + PER (`train_d3qn_per.py`, `train_d3qn_per_gemini.py`)
3. Recurrent/value POMDP variants: ADRQN and RD3QN (`train_adrqn.py`, `train_rd3qn.py`)
4. PPO family and stabilizations (`train_ppo.py`, `train_ppo_fixed.py`, `train_ppo_improved.py`, `train_ppo_advanced.py`, `train_ppo_adaptive.py`, `train_ppo_visualized.py`)
5. Advanced recurrent on-policy + intrinsic learning (`train_raptor_ppo.py`)
6. World-model POMDP approach (`train_worldmodel_pomdp.py`)
7. Meta strategies: curriculum and hybrid heuristic-RL (`train_curriculum.py`, `train_hybrid.py`)

### 3.2 Script-by-script method summary with key defaults

| Script | Algorithm family | Key ideas | Key defaults from argparse/code |
|---|---|---|---|
| `train_ddqn.py` | Double DQN | online/target Q-networks + replay | episodes=2000, max_steps=1000, gamma=0.99, lr=1e-3, batch=256, replay=100000, warmup=2000, target_sync=2000 |
| `train_2.py` | Lightweight DQN | small net, fast decay, smaller replay | episodes=1000, lr=1e-3, batch=64, eps_decay=50000, gamma=0.95, replay=20000 |
| `train_d3qn_per.py` | Dueling Double DQN + PER | LayerNorm dueling network, prioritized replay, optional parallel env collection | episodes=3000, max_steps=1000, num_envs=1 default, num_workers=0 default, gamma=0.99, lr=5e-4, batch=256, replay_capacity=100000, hidden_dim=128, per_alpha=0.6, per_beta_start=0.4 |
| `train_d3qn_per_gemini.py` | D3QN+PER variant | simpler config and CPU stability edits | episodes=2000, max_steps=1000, batch=128, optimizer lr=5e-4, replay capacity=50000 |
| `train_adrqn.py` | Action-augmented recurrent DQN | LSTM with action/history augmentation and multi-env loop | difficulty=2 default, num_envs=6, total_steps=5000000, max_steps=1000, optimizer lr=1e-4 |
| `train_rd3qn.py` | Recurrent D3QN + spatial belief map | map-enhanced state + recurrent Q updates | difficulty=2 default, num_envs=6, total_steps=600000, max_steps=1000, optimizer lr=1e-4 |
| `train_ppo.py` | PPO baseline | parallel rollout workers, GAE, clipping, reward-shaping knobs | total_timesteps=2000000, rollout_steps=2048, batch_size=256, ppo_epochs=10, num_workers=6, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, hidden_dim=128 |
| `train_ppo_fixed.py` | PPO corrected | timeout-aware GAE, entropy schedule, stabilization fixes | total_timesteps=1000000, rollout_steps=2048, batch_size=256, ppo_epochs=10, num_workers=6, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef_start=0.05 -> ent_coef_end=0.001, hidden_dim=128 |
| `train_ppo_improved.py` | PPO improved + optional LSTM | deeper network, optional recurrent policy, scheduler | total_timesteps=1000000, rollout_steps=2048, batch_size=256, ppo_epochs=10, num_workers=6, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, hidden_dim=256, `--use_lstm` |
| `train_ppo_advanced.py` | PPO with exploration/exploitation heuristics | temperature-based action behavior and heavy task-specific bonuses | total_timesteps=1000000, rollout_steps=2048, batch_size=256, ppo_epochs=10, num_workers=6, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, hidden_dim=128 |
| `train_ppo_adaptive.py` | PPO adaptive | adaptive entropy, curiosity/progress style reward knobs | total_timesteps=1000000, rollout_steps=2048, batch_size=256, ppo_epochs=10, num_workers=6, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef_start=0.05 -> ent_coef_end=0.001, hidden_dim=128 |
| `train_ppo_visualized.py` | PPO with diagnostics emphasis | visualization/logging and tuned shaping/schedules | total_timesteps=1000000, rollout_steps=2048, batch_size=256, ppo_epochs=10, num_workers=6, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, hidden_dim=128 |
| `train_raptor_ppo.py` | Recurrent actor-critic PPO + RND + dynamics auxiliary loss | richer 43D feature input, GRU memory, intrinsic reward annealing, frequent eval/checkpoints | difficulty=3 default, num_envs=8, horizon=512, total_steps=2000000, lr=3e-4, gamma=0.995, gae_lambda=0.95, clip_coef=0.2, vf_coef=0.7, ent_coef=0.002, ppo_epochs=4, intrinsic_coef=0.15->0.0, rnd_coef=0.15, dyn_coef=0.05 |
| `train_worldmodel_pomdp.py` | World model + actor-critic | latent dynamics learning with imagined rollouts | difficulty=3 default, num_envs=8, horizon=256, total_steps=2500000, hidden_dim=192, latent_dim=96, warmup_steps=50000, seq_len=32, batch_size=64, updates_per_rollout=180, imag_horizon=16, lr_wm=2e-4, lr_actor=1.5e-4, lr_critic=2e-4, gamma=0.995 |
| `train_curriculum.py` | PPO + curriculum | auto progression of difficulty/shaping level | total_timesteps=2000000, rollout_steps=2048, batch_size=256, ppo_epochs=10, lr=3e-4, hidden_dim=128, max_level=10 |
| `train_hybrid.py` | Heuristic-guided Q-learning hybrid | combines heuristic action guidance with intrinsic improvement reward | episodes=2000, batch_size=128, replay_size=50000, lr=1e-3, gamma=0.99, hidden_dim=128, train_freq=4, target_update=1000 |

## 4) Experiments and Implementation Details (Template Section)

### 4.1 Environment and evaluation pipeline

- Local evaluation script: `evaluate.py` (writes to `leaderboard.csv`)
- Codabench-style evaluator: `evaluate_on_codabench.py`
  - fixed levels `[0,2,3]`
  - `wall_obstacles=True`
  - 10 runs per level by default
- Submission packaging utility: `export_submission_raptor.py`

### 4.2 Key implementation families found in repository

- Value-learning with replay (`train_ddqn.py`, `train_d3qn_per.py`, `train_d3qn_per_gemini.py`, `train_2.py`)
- Recurrent value learning for partial observability (`train_adrqn.py`, `train_rd3qn.py`)
- PPO on-policy training with many reward and schedule variants (`train_ppo*.py`)
- Intrinsic motivation / auxiliary prediction (`train_raptor_ppo.py`)
- Latent world model with imagination rollouts (`train_worldmodel_pomdp.py`)
- Curriculum and hybrid exploration (`train_curriculum.py`, `train_hybrid.py`)

### 4.3 Useful artifacts for report figures

- Multiple `training_summary.png`, `learning_dynamics.png`, `action_diagnostics.png`, `reward_diagnostics.png`, `qvalue_diagnostics.png` files in submission directories.
- Candidate visualization artifacts:
  - `submission_001/training_summary.png`
  - `submission_001_no_wall/training_summary.png`
  - `submission_phase4_final/training_summary.png`
  - `submission_debug_tmp/training_metrics.png`
  - `submission_improved_lstm/training_metrics.png`

## 5) Results (Template Section)

### 5.1 Training-log-derived summary table

Extracted from `submission_*/training_log.txt` files:

| Log file | Episodes | Total Steps | Training Time | Mean Return | Success Rate | Best Success Rate | Mean step reward |
|---|---:|---:|---|---:|---:|---:|---:|
| `submission_001_no_wall/training_log.txt` | 3000 | 2946640 | 7384.9s (123.1 min) | -6124.54 +/- 22024.20 | 0.0% | 19.0% | -31.924 |
| `submission_001/training_log.txt` | 3000 | 2793433 | 9475.2s (157.9 min) | -108260.35 +/- 81515.95 | 16.0% | 32.0% | -55.129 |
| `submission_001_with_wall/training_log.txt` | 2000 | 1961258 | 9906.7s (165.1 min) | -72925.39 +/- 68761.13 | 5.0% | 6.0% | -41.677 |
| `submission_002_no_wall/training_log.txt` | 2000 | 1919720 | 4493.5s (74.9 min) | -1185.10 +/- 1948.48 | 0.0% | 24.0% | -31.871 |
| `submission_002/training_log.txt` | 300 | 297686 | 10902.7s | -53906.20 +/- 51928.94 | 3.0% | 1.0% | (not printed) |
| `submission_FIXED_shaped/training_log.txt` | 4000 | 3886969 | 7424.0s (123.7 min) | -30453.24 +/- 66782.47 | 7.0% | 30.0% | -28.581 |
| `submission_phase1_heavy/training_log.txt` | 2500 | 2387269 | 4505.4s (75.1 min) | -177220.45 +/- 24592.99 | 5.0% | 20.0% | -137.226 |
| `submission_phase2_medium/training_log.txt` | 2000 | 1879339 | 3649.6s (60.8 min) | -116957.02 +/- 76091.05 | 16.0% | 17.0% | -48.862 |
| `submission_phase3_light/training_log.txt` | 1500 | 1453833 | 2864.3s (47.7 min) | -163758.77 +/- 29043.81 | 6.0% | 12.0% | -140.643 |
| `submission_phase4_final/training_log.txt` | 1000 | 946103 | 1966.9s (32.8 min) | -27509.60 +/- 62750.89 | 7.0% | 23.0% | -23.400 |

### 5.2 Immediate interpretation points for report writing

- Reward distribution is strongly negative-dominant in many runs (stuck penalties and time penalties are heavy).
- Successful episodes are sparse, but several variants record non-zero best success rates.
- Performance appears sensitive to shaping, walls, and exploration strategy.
- No comprehensive test-phase rank table is stored in repo; only local/dev evidence exists.

## 6) Error Analysis and Discussion (Template Section)

Use these repository-grounded observations:

- Success metric mismatch risk: some scripts infer success from high reward threshold; environment success is fundamentally terminal condition based on attached box touching boundary.
- Recurrent inference fragility: hidden-state reset behavior can differ between train/eval setups and produce deployment drift.
- Stuck handling is critical: environment applies severe stuck penalty (`-200`) via stuck flag.
- Wall-obstacle robustness requires explicit anti-stuck and recovery behavior.
- Checkpoint selection policy matters: best-eval checkpoint is generally safer than last checkpoint for submission export.

Potential discussion framing:

- Why Level-2 and Level-3 are harder: partial observability, target blinking, and nonstationary target motion.
- Why walls increase failure risk: dead-ends, collision loops, and poor local minima around turning behavior.
- Why recurrent/intrinsic methods may help: memory for intermittently observable targets and richer exploration signals.

## 7) Conclusion (Template Section)

A concise, evidence-backed conclusion can state:

- The project explored a broad RL design space (value-based, recurrent, PPO family, intrinsic motivation, world models, and curriculum).
- The strongest practical improvements came from improved stability, recurrent memory, and better checkpointing/evaluation discipline.
- Future gains likely require cleaner success metrics, stronger level-wise evaluation, and explicit anti-stuck/wall-aware policy behavior.

## 8) LLM Usage Declaration (Template Section Draft)

Suggested draft skeleton to complete manually:

- LLM used: TODO
- Usage scope: code understanding, report structuring, debugging support, plotting support (as applicable)
- Non-LLM sources: papers/blogs/repos used, if any
- Personal contribution: own experimentation, implementation, and analysis details

---

## Appendix A: Core File Taxonomy Walkthrough

### A.1 Environment and evaluation

- `obelix.py`
- `manual_play.py`
- `evaluate.py`
- `evaluate_on_codabench.py`
- `export_submission_raptor.py`
- `compute_observation_states.py`

### A.2 Agent entry files

- `agent.py`
- `agent_template.py`
- `agent_d3qn_per.py`
- `agent_adrqn.py`
- `agent_rd3qn.py`
- `agent_ppo.py`
- `agent_raptor_ppo.py`
- `agent_worldmodel_pomdp.py`
- `agent_gemini.py`

### A.3 Training scripts

- `train_ddqn.py`
- `train_d3qn_per.py`
- `train_d3qn_per_gemini.py`
- `train_2.py`
- `train_adrqn.py`
- `train_rd3qn.py`
- `train_ppo.py`
- `train_ppo_fixed.py`
- `train_ppo_improved.py`
- `train_ppo_advanced.py`
- `train_ppo_adaptive.py`
- `train_ppo_visualized.py`
- `train_raptor_ppo.py`
- `train_worldmodel_pomdp.py`
- `train_curriculum.py`
- `train_hybrid.py`

### A.4 Documentation and template

- `README.md`
- `README_D3QN_PER.md`
- `leaderboard.csv`
- `requirements.txt`
- `Report_template/Report.tex`
- `Report_template/references.bib`

## Appendix B: Full File Inventory (non-.git)

```text
agent_adrqn.py
agent_d3qn_per.py
agent_gemini.py
agent_ppo.py
agent.py
agent_raptor_ppo.py
agent_rd3qn.py
agent_template.py
agent_worldmodel_pomdp.py
arpit.ipynb
compute_observation_states.py
CS780_ Capstone Project-1.pdf
evaluate_on_codabench.py
evaluate.py
export_submission_raptor.py
FIXED.log
.gitignore
leaderboard.csv
manual_play.py
OBELIX.png
obelix.py
phase1.log
phase2.log
ppo_adv.log
__pycache__/agent_adrqn.cpython-310.pyc
__pycache__/agent.cpython-310.pyc
__pycache__/agent_r3dqn.cpython-310.pyc
__pycache__/agent_rd3qn.cpython-310.pyc
__pycache__/obelix.cpython-310.pyc
__pycache__/train_ppo_improved.cpython-310.pyc
README_D3QN_PER.md
README.md
REPORT_COMPILATION.md
Report_template/references.bib
Report_template/Report.pdf
Report_template/Report.tex
requirements.txt
submission_001/action_diagnostics.png
submission_001/agent_d3qn_per.py
submission_001/checkpoint_ep1000.pth
submission_001/checkpoint_ep1200.pth
submission_001/checkpoint_ep1400.pth
submission_001/checkpoint_ep1600.pth
submission_001/checkpoint_ep1800.pth
submission_001/checkpoint_ep2000.pth
submission_001/checkpoint_ep200.pth
submission_001/checkpoint_ep2200.pth
submission_001/checkpoint_ep2400.pth
submission_001/checkpoint_ep250.pth
submission_001/checkpoint_ep2600.pth
submission_001/checkpoint_ep2800.pth
submission_001/checkpoint_ep3000.pth
submission_001/checkpoint_ep400.pth
submission_001/checkpoint_ep500.pth
submission_001/checkpoint_ep600.pth
submission_001/checkpoint_ep750.pth
submission_001/checkpoint_ep800.pth
submission_001/learning_dynamics.png
submission_001_no_wall/action_diagnostics.png
submission_001_no_wall/agent_d3qn_per.py
submission_001_no_wall/checkpoint_ep1000.pth
submission_001_no_wall/checkpoint_ep100.pth
submission_001_no_wall/checkpoint_ep1100.pth
submission_001_no_wall/checkpoint_ep1200.pth
submission_001_no_wall/checkpoint_ep1300.pth
submission_001_no_wall/checkpoint_ep1400.pth
submission_001_no_wall/checkpoint_ep1500.pth
submission_001_no_wall/checkpoint_ep1600.pth
submission_001_no_wall/checkpoint_ep1700.pth
submission_001_no_wall/checkpoint_ep1800.pth
submission_001_no_wall/checkpoint_ep1900.pth
submission_001_no_wall/checkpoint_ep2000.pth
submission_001_no_wall/checkpoint_ep200.pth
submission_001_no_wall/checkpoint_ep2100.pth
submission_001_no_wall/checkpoint_ep2200.pth
submission_001_no_wall/checkpoint_ep2300.pth
submission_001_no_wall/checkpoint_ep2400.pth
submission_001_no_wall/checkpoint_ep2500.pth
submission_001_no_wall/checkpoint_ep2600.pth
submission_001_no_wall/checkpoint_ep2700.pth
submission_001_no_wall/checkpoint_ep2800.pth
submission_001_no_wall/checkpoint_ep2900.pth
submission_001_no_wall/checkpoint_ep3000.pth
submission_001_no_wall/checkpoint_ep300.pth
submission_001_no_wall/checkpoint_ep400.pth
submission_001_no_wall/checkpoint_ep500.pth
submission_001_no_wall/checkpoint_ep600.pth
submission_001_no_wall/checkpoint_ep700.pth
submission_001_no_wall/checkpoint_ep800.pth
submission_001_no_wall/checkpoint_ep900.pth
submission_001_no_wall/learning_dynamics.png
submission_001_no_wall/qvalue_diagnostics.png
submission_001_no_wall/reward_diagnostics.png
submission_001_no_wall/training_log.txt
submission_001_no_wall/training_summary.png
submission_001_no_wall/weights.pth
submission_001/qvalue_diagnostics.png
submission_001/reward_diagnostics.png
submission_001/training_log.txt
submission_001/training_summary.png
submission_001/weights.pth
submission_001_with_wall/action_diagnostics.png
submission_001_with_wall/agent.py
submission_001_with_wall/checkpoint_ep1000.pth
submission_001_with_wall/checkpoint_ep1500.pth
submission_001_with_wall/checkpoint_ep2000.pth
submission_001_with_wall/checkpoint_ep500.pth
submission_001_with_wall/learning_dynamics.png
submission_001_with_wall/qvalue_diagnostics.png
submission_001_with_wall/reward_diagnostics.png
submission_001_with_wall/submission.zip
submission_001_with_wall/training_log.txt
submission_001_with_wall/training_summary.png
submission_001_with_wall/weights.pth
submission_002/action_diagnostics.png
submission_002/agent_d3qn_per.py
submission_002/checkpoint_ep250.pth
submission_002/learning_dynamics.png
submission_002_no_wall/action_diagnostics.png
submission_002_no_wall/agent.py
submission_002_no_wall/checkpoint_ep1000.pth
submission_002_no_wall/checkpoint_ep100.pth
submission_002_no_wall/checkpoint_ep1100.pth
submission_002_no_wall/checkpoint_ep1200.pth
submission_002_no_wall/checkpoint_ep1300.pth
submission_002_no_wall/checkpoint_ep1400.pth
submission_002_no_wall/checkpoint_ep1500.pth
submission_002_no_wall/checkpoint_ep1600.pth
submission_002_no_wall/checkpoint_ep1700.pth
submission_002_no_wall/checkpoint_ep1800.pth
submission_002_no_wall/checkpoint_ep1900.pth
submission_002_no_wall/checkpoint_ep2000.pth
submission_002_no_wall/checkpoint_ep200.pth
submission_002_no_wall/checkpoint_ep300.pth
submission_002_no_wall/checkpoint_ep400.pth
submission_002_no_wall/checkpoint_ep500.pth
submission_002_no_wall/checkpoint_ep600.pth
submission_002_no_wall/checkpoint_ep700.pth
submission_002_no_wall/checkpoint_ep800.pth
submission_002_no_wall/checkpoint_ep900.pth
submission_002_no_wall/learning_dynamics.png
submission_002_no_wall/qvalue_diagnostics.png
submission_002_no_wall/reward_diagnostics.png
submission_002_no_wall/submission.zip
submission_002_no_wall/training_log.txt
submission_002_no_wall/training_summary.png
submission_002_no_wall/weights.pth
submission_002/qvalue_diagnostics.png
submission_002/reward_diagnostics.png
submission_002/training_log.txt
submission_002/training_summary.png
submission_002/weights.pth
submission_003/checkpoint_ep1000.pth
submission_003/checkpoint_ep100.pth
submission_003/checkpoint_ep1050.pth
submission_003/checkpoint_ep1100.pth
submission_003/checkpoint_ep1150.pth
submission_003/checkpoint_ep1200.pth
submission_003/checkpoint_ep1250.pth
submission_003/checkpoint_ep1300.pth
submission_003/checkpoint_ep1350.pth
submission_003/checkpoint_ep1400.pth
submission_003/checkpoint_ep1450.pth
submission_003/checkpoint_ep1500.pth
submission_003/checkpoint_ep150.pth
submission_003/checkpoint_ep1550.pth
submission_003/checkpoint_ep1600.pth
submission_003/checkpoint_ep1650.pth
submission_003/checkpoint_ep1700.pth
submission_003/checkpoint_ep1750.pth
submission_003/checkpoint_ep1800.pth
submission_003/checkpoint_ep1850.pth
submission_003/checkpoint_ep1900.pth
submission_003/checkpoint_ep1950.pth
submission_003/checkpoint_ep2000.pth
submission_003/checkpoint_ep200.pth
submission_003/checkpoint_ep2050.pth
submission_003/checkpoint_ep2100.pth
submission_003/checkpoint_ep2150.pth
submission_003/checkpoint_ep2200.pth
submission_003/checkpoint_ep2250.pth
submission_003/checkpoint_ep2300.pth
submission_003/checkpoint_ep2350.pth
submission_003/checkpoint_ep2400.pth
submission_003/checkpoint_ep2450.pth
submission_003/checkpoint_ep2500.pth
submission_003/checkpoint_ep250.pth
submission_003/checkpoint_ep2550.pth
submission_003/checkpoint_ep300.pth
submission_003/checkpoint_ep350.pth
submission_003/checkpoint_ep400.pth
submission_003/checkpoint_ep450.pth
submission_003/checkpoint_ep500.pth
submission_003/checkpoint_ep50.pth
submission_003/checkpoint_ep550.pth
submission_003/checkpoint_ep600.pth
submission_003/checkpoint_ep650.pth
submission_003/checkpoint_ep700.pth
submission_003/checkpoint_ep750.pth
submission_003/checkpoint_ep800.pth
submission_003/checkpoint_ep850.pth
submission_003/checkpoint_ep900.pth
submission_003/checkpoint_ep950.pth
submission_003/weights.pth
submission_adrqn.py/agent.py
submission_adrqn.py/behaviour_plots.png
submission_adrqn.py/submission.zip
submission_adrqn.py/weights.pth
submission_debug_tmp/training_metrics.png
submission_FIXED_shaped/action_diagnostics.png
submission_FIXED_shaped/agent.py
submission_FIXED_shaped/checkpoint_ep1200.pth
submission_FIXED_shaped/checkpoint_ep1500.pth
submission_FIXED_shaped/checkpoint_ep1800.pth
submission_FIXED_shaped/checkpoint_ep200.pth
submission_FIXED_shaped/checkpoint_ep2100.pth
submission_FIXED_shaped/checkpoint_ep2400.pth
submission_FIXED_shaped/checkpoint_ep2700.pth
submission_FIXED_shaped/checkpoint_ep3000.pth
submission_FIXED_shaped/checkpoint_ep300.pth
submission_FIXED_shaped/checkpoint_ep3300.pth
submission_FIXED_shaped/checkpoint_ep3600.pth
submission_FIXED_shaped/checkpoint_ep3900.pth
submission_FIXED_shaped/checkpoint_ep4000.pth
submission_FIXED_shaped/checkpoint_ep600.pth
submission_FIXED_shaped/checkpoint_ep900.pth
submission_FIXED_shaped/learning_dynamics.png
submission_FIXED_shaped/qvalue_diagnostics.png
submission_FIXED_shaped/reward_diagnostics.png
submission_FIXED_shaped/submission.zip
submission_FIXED_shaped/training_log.txt
submission_FIXED_shaped/training_summary.png
submission_FIXED_shaped/weights.pth
submission_improved_lstm/training_metrics.png
submission_phase1_heavy/action_diagnostics.png
submission_phase1_heavy/agent_d3qn_per.py
submission_phase1_heavy/checkpoint_ep1200.pth
submission_phase1_heavy/checkpoint_ep1500.pth
submission_phase1_heavy/checkpoint_ep1800.pth
submission_phase1_heavy/checkpoint_ep2100.pth
submission_phase1_heavy/checkpoint_ep2400.pth
submission_phase1_heavy/checkpoint_ep2500.pth
submission_phase1_heavy/checkpoint_ep300.pth
submission_phase1_heavy/checkpoint_ep600.pth
submission_phase1_heavy/checkpoint_ep900.pth
submission_phase1_heavy/learning_dynamics.png
submission_phase1_heavy/qvalue_diagnostics.png
submission_phase1_heavy/reward_diagnostics.png
submission_phase1_heavy/training_log.txt
submission_phase1_heavy/training_summary.png
submission_phase1_heavy/weights.pth
submission_phase2_medium/action_diagnostics.png
submission_phase2_medium/agent_d3qn_per.py
submission_phase2_medium/checkpoint_ep1200.pth
submission_phase2_medium/checkpoint_ep1500.pth
submission_phase2_medium/checkpoint_ep1800.pth
submission_phase2_medium/checkpoint_ep2000.pth
submission_phase2_medium/checkpoint_ep300.pth
submission_phase2_medium/checkpoint_ep600.pth
submission_phase2_medium/checkpoint_ep900.pth
submission_phase2_medium/learning_dynamics.png
submission_phase2_medium/qvalue_diagnostics.png
submission_phase2_medium/reward_diagnostics.png
submission_phase2_medium/training_log.txt
submission_phase2_medium/training_summary.png
submission_phase2_medium/weights.pth
submission_phase3_light/action_diagnostics.png
submission_phase3_light/agent_d3qn_per.py
submission_phase3_light/checkpoint_ep1200.pth
submission_phase3_light/checkpoint_ep1500.pth
submission_phase3_light/checkpoint_ep300.pth
submission_phase3_light/checkpoint_ep600.pth
submission_phase3_light/checkpoint_ep900.pth
submission_phase3_light/learning_dynamics.png
submission_phase3_light/qvalue_diagnostics.png
submission_phase3_light/reward_diagnostics.png
submission_phase3_light/training_log.txt
submission_phase3_light/training_summary.png
submission_phase3_light/weights.pth
submission_phase4_final/action_diagnostics.png
submission_phase4_final/agent.py
submission_phase4_final/checkpoint_ep1000.pth
submission_phase4_final/checkpoint_ep300.pth
submission_phase4_final/checkpoint_ep600.pth
submission_phase4_final/checkpoint_ep900.pth
submission_phase4_final/learning_dynamics.png
submission_phase4_final/qvalue_diagnostics.png
submission_phase4_final/reward_diagnostics.png
submission_phase4_final/submission.zip
submission_phase4_final/training_log.txt
submission_phase4_final/training_summary.png
submission_phase4_final/weights.pth
submission_ppo_adaptive_v1/agent.py
submission_ppo_adaptive_v1/checkpoint_update100.pth
submission_ppo_adaptive_v1/checkpoint_update200.pth
submission_ppo_adaptive_v1/checkpoint_update300.pth
submission_ppo_adaptive_v1/checkpoint_update400.pth
submission_ppo_adaptive_v1/submission.zip
submission_ppo_adaptive_v1/weights.pth
submission_ppo_adaptive_v2/agent.py
submission_ppo_adaptive_v2/submission.zip
submission_ppo_adaptive_v2/weights.pth
submission_ppo_adaptive/weights.pth
submission_ppo_adv_final/checkpoint_update100.pth
submission_ppo_adv_final/checkpoint_update200.pth
submission_ppo_adv_final/checkpoint_update300.pth
submission_ppo_adv_final/checkpoint_update400.pth
submission_ppo_safe_efficient/weights.pth
submission_ppo_test2/agent.py
submission_ppo_test2/checkpoint_update100.pth
submission_ppo_test2/checkpoint_update200.pth
submission_ppo_test2/checkpoint_update300.pth
submission_ppo_test2/checkpoint_update400.pth
submission_ppo_test2/checkpoint_update500.pth
submission_ppo_test2/checkpoint_update600.pth
submission_ppo_test2/checkpoint_update700.pth
submission_ppo_test2/checkpoint_update800.pth
submission_ppo_test2/checkpoint_update900.pth
submission_ppo_test2/submission.zip
submission_ppo_test2/weights.pth
submission_ppo_test/agent.py
submission_ppo_test/checkpoint_update100.pth
submission_ppo_test/checkpoint_update200.pth
submission_ppo_test/submission.zip
submission_ppo_test/weights.pth
submission_raptor_ppo1/weights_best.pth
submission_raptor_ppo1/weights.pth
submission_raptor_ppo1/weights_submission_best.pth
submission_raptor_ppo2/weights_best.pth
submission_raptor_ppo2/weights.pth
submission_raptor_ppo2/weights_submission_best.pth
submission_raptor_ppo3/agent.py
submission_raptor_ppo3/submission.zip
submission_raptor_ppo3/weights_best.pth
submission_raptor_ppo3/weights.pth
submission_raptor_ppo3/weights_submission_best.pth
submission_raptor_ppo4/weights_best.pth
submission_raptor_ppo4/weights_eval_best.pth
submission_raptor_ppo/agent.py
submission_raptor_ppo/submission.zip
submission_raptor_ppo/weights_best.pth
submission_raptor_ppo/weights_eval_best.pth
submission_raptor_ppo/weights.pth
submission_raptor_ppo/weights_submission_best.pth
submission_rd3qn/agent.py
submission_rd3qn/submission.zip
submission_rd3qn/weights.pth
submission_rdq3qn.py/agent.py
submission_rdq3qn.py/behaviour_plots.png
submission_rdq3qn.py/submission.zip
submission_rdq3qn.py/weights.pth
submission_template1.py
submission_template2.py
submission_worldmodel_pomdp1/weights_eval_best.pth
train_2.py
train_adrqn.py
train_curriculum.py
train_d3qn_per_gemini.py
train_d3qn_per.py
train_ddqn.py
train_hybrid.py
training.log
train_ppo_adaptive.py
train_ppo_advanced.py
train_ppo_fixed.py
train_ppo_improved.py
train_ppo.py
train_ppo_visualized.py
train_raptor_ppo.py
train_rd3qn.py
train_worldmodel_pomdp.py
```
