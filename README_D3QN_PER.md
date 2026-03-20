# D3QN-PER Agent for OBELIX Environment

## Overview

This implementation combines three powerful reinforcement learning improvements over vanilla DQN:

1. **Dueling Architecture**: Separates state value estimation from action advantage estimation
2. **Double DQN**: Reduces overestimation bias in Q-value updates
3. **Prioritized Experience Replay (PER)**: Learns more efficiently from important transitions

## Algorithm Details

### 1. Dueling Deep Q-Network (Dueling DQN)

**Problem with Standard DQN**: The network directly outputs Q(s,a), which can be inefficient because:
- The value of many states doesn't depend much on which action is taken
- The network has to learn state values and action advantages simultaneously

**Dueling Solution**: Split the network into two streams:
```
                     ┌─→ Value Stream V(s)
Feature Extractor ──┤
                     └─→ Advantage Stream A(s,a)

Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
```

**Benefits**:
- Better learning of state values independent of actions
- Faster convergence in environments where actions don't always matter
- More stable gradient flow

### 2. Double Deep Q-Network (Double DQN)

**Problem with Standard DQN**: Overestimation bias
```
Standard DQN: target = r + γ × max_a Q_target(s', a)
Problem: The same network both selects and evaluates actions
```

**Double DQN Solution**: Separate selection and evaluation
```
Online network: selects best action
Target network: evaluates that action

target = r + γ × Q_target(s', argmax_a Q_online(s', a))
```

**Benefits**:
- Dramatically reduces overestimation
- More stable learning
- Better final policies

### 3. Prioritized Experience Replay (PER)

**Problem with Uniform Replay**: All transitions sampled equally
- Some transitions are more important than others
- Rare but important events get sampled infrequently

**PER Solution**: Sample transitions proportional to their TD error
```
Priority: p_i = (|TD_error_i| + ε)^α
Sampling probability: P(i) = p_i / Σ_j p_j

Where α controls prioritization (0 = uniform, 1 = full prioritization)
```

**Importance Sampling Weights**: Correct bias from non-uniform sampling
```
w_i = (N × P(i))^(-β)
Normalize: w_i = w_i / max_j(w_j)

Where β is annealed from β_start to 1.0 over training
```

**Benefits**:
- Learn more from surprising transitions
- Faster convergence
- Better sample efficiency

## Architecture

### Network Structure
```
Input (18D sensor readings)
    ↓
Feature Extractor (128 → 128)
    ↓
    ├─→ Value Stream (128 → 64 → 1)
    └─→ Advantage Stream (128 → 64 → 5)
         ↓
    Q(s,a) = V(s) + (A(s,a) - mean(A))
```

### Key Components

**Dueling DQN Module**:
- Shared feature extractor (2 layers, 128 hidden units)
- Value stream: estimates V(s)
- Advantage stream: estimates A(s,a)
- Mean normalization for stability

**Prioritized Replay Buffer**:
- Capacity: 100,000 transitions
- Alpha (α = 0.6): prioritization strength
- Beta (β = 0.4 → 1.0): importance sampling correction
- Efficient priority updates

**Action Smoothing** (Evaluation):
- Prevents oscillation when Q-values are close
- Repeats previous action up to 3 times if Q-values differ by < 0.08
- Improves evaluation stability

## Training

### Hyperparameters (Default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 3000 | Total training episodes |
| Max Steps | 1000 | Steps per episode |
| Batch Size | 256 | Training batch size |
| Learning Rate | 5e-4 | Adam optimizer |
| Gamma (γ) | 0.99 | Discount factor |
| Hidden Dim | 128 | Network hidden size |
| Replay Capacity | 100,000 | Buffer size |
| Warmup Steps | 2000 | Steps before training |
| Target Sync | 2000 | Target network update frequency |
| ε Start | 1.0 | Initial exploration |
| ε End | 0.02 | Final exploration |
| ε Decay Steps | 250,000 | Exploration decay |
| PER Alpha (α) | 0.6 | Prioritization strength |
| PER Beta Start (β) | 0.4 | Initial IS correction |

### Training Command

```bash
python train_d3qn_per.py \
    --obelix_py ./obelix.py \
    --agent_id 001 \
    --episodes 3000 \
    --difficulty 0 \
    --wall_obstacles \
    --hidden_dim 128 \
    --lr 5e-4 \
    --batch 256 \
    --per_alpha 0.6 \
    --per_beta_start 0.4
```

### Output Files

Training creates a `submission_{agent_id}/` directory with:

1. **weights.pth** - Final trained model weights
2. **agent_d3qn_per.py** - Agent code for evaluation
3. **training_diagnostics.png** - Comprehensive diagnostic plots
4. **learning_curves.png** - Focused learning curves
5. **training_log.txt** - Training statistics
6. **checkpoint_epXXXX.pth** - Periodic checkpoints (every 250 episodes)

## Diagnostic Plots

### Training Diagnostics (8 subplots)

1. **Episode Returns**: Raw and rolling average (100-episode window)
2. **Success Rate**: Rolling success rate over training
3. **Training Loss**: Smooth L1 loss with moving average
4. **Average Q-values**: Q-value progression
5. **Epsilon Decay**: Exploration rate schedule
6. **Episode Length Distribution**: Histogram of episode lengths
7. **Return Distribution**: Last 500 episodes
8. **Summary Statistics**: Key metrics

### Learning Curves (4 subplots)

1. **Returns with Confidence**: Mean ± standard deviation
2. **Success Rate**: With 80% threshold line
3. **Q-value Progression**: Smoothed average Q-values
4. **Training Loss**: Smoothed loss curve

## Evaluation

The agent uses greedy action selection (ε = 0) with action smoothing:

```python
def policy(obs, rng):
    # Greedy Q-value selection
    q_values = model(obs)
    best_action = argmax(q_values)
    
    # Action smoothing for stability
    if Q-values are close:
        repeat previous action (up to 3 times)
    else:
        take new best action
    
    return action
```

## Key Improvements Over Standard DQN

| Aspect | Standard DQN | D3QN-PER |
|--------|--------------|----------|
| Architecture | Single stream | Dueling (V + A streams) |
| Target Computation | Single network | Double (select + evaluate) |
| Replay Sampling | Uniform | Prioritized by TD error |
| Sample Efficiency | Baseline | 2-3× faster convergence |
| Overestimation | High | Significantly reduced |
| Stability | Moderate | High |

## Performance Expectations

Based on the algorithm improvements:

- **Faster Convergence**: 30-50% fewer episodes to reach good performance
- **Higher Success Rate**: 10-20% improvement over vanilla DQN
- **Lower Variance**: More consistent episode returns
- **Better Final Policy**: Higher maximum achievable performance

## Troubleshooting

### High Loss / Unstable Training
- Reduce learning rate (try 1e-4)
- Increase target sync frequency
- Lower PER alpha (less aggressive prioritization)

### Low Success Rate
- Increase training episodes
- Adjust reward shaping
- Check epsilon decay schedule

### Memory Issues
- Reduce replay buffer capacity
- Reduce batch size
- Use gradient checkpointing

## References

1. **Dueling DQN**: ["Dueling Network Architectures for Deep Reinforcement Learning"](https://arxiv.org/abs/1511.06581) - Wang et al., 2016
2. **Double DQN**: ["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/abs/1509.06461) - van Hasselt et al., 2015
3. **PER**: ["Prioritized Experience Replay"](https://arxiv.org/abs/1511.05952) - Schaul et al., 2015

## License

This implementation is for educational purposes as part of NPTEL's Reinforcement Learning course.

## Author

Agent ID: 001  
Algorithm: D3QN-PER (Dueling Double DQN + Prioritized Experience Replay)  
Environment: OBELIX Robot Simulation
