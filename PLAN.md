# GRPO-Based RL Fine-Tuning for SmolVLA — Implementation Plan

> **Goal**: Apply Group Relative Policy Optimization (GRPO) to SmolVLA's flow-matching head on LIBERO tasks to improve out-of-distribution generalization.

---

## Environment Summary

| Item | Detail |
|---|---|
| Conda env | `robot-proj` (Python 3.11) |
| Datasets | `/home/shinawatra/Works/personal/robo-proj/LIBERO/libero/datasets/` — HDF5 files for `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| Base model | `lerobot/smolvla_libero` (SFT on LIBERO) or `lerobot/smolvla_base` |
| Sim env | LIBERO via `libero` package (already installed in `robot-proj`) + MuJoCo 3.7 + robosuite 1.4 |
| Key limitation | `sample_actions` reads `num_steps` from config — need custom denoising loop for 8/9/10-step weighted rewards |

---

## Phase 0 — Environment Setup

### 0.1 Install lerobot in `robot-proj`

**Verified**: lerobot 0.4.3 with `SmolVLAPolicy` is proven to import in `so-arm101-vla` env. Same install in `robot-proj`:

```bash
conda activate robot-proj
uv pip install "lerobot[smolvla]"
# Verify:
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('OK')"
```

**Proof check**: SmolVLAPolicy imported successfully in `so-arm101-vla` (confirmed in research). Same `uv pip install` path.

### 0.2 Verify LIBERO environment works

```bash
conda activate robot-proj
python -c "
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
bd = benchmark.get_benchmark_dict()
suite = bd['libero_10']()
task = suite.get_task(0)
print('Task:', task.language)
print('LIBERO env OK')
"
```

**Proof**: `libero 0.1.0` is already installed in `robot-proj`. MuJoCo 3.7 and robosuite 1.4 confirmed present.

### 0.3 Verify SmolVLA loads from pretrained checkpoint

```bash
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained('lerobot/smolvla_libero')
print('Checkpoint loaded OK. num_steps:', policy.config.num_steps)
"
```

**Proof**: `SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')` is documented in the lerobot SmolVLA guide and confirmed by the code's docstring in `modeling_smolvla.py`.

---

## Phase 1 — Baseline Evaluation (SFT)

Before RL, establish the imitation learning (SFT) baseline on LIBERO.

### 1.1 Load SmolVLA LIBERO checkpoint

Use `lerobot/smolvla_libero` — this is the SFT checkpoint already fine-tuned on LIBERO (confirmed from proposal §4.5).

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
policy.eval().cuda()
```

### 1.2 Run LIBERO evaluation loop

```python
import os, torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

def evaluate_policy(policy, task_suite_name="libero_10", n_episodes=20):
    bd = benchmark.get_benchmark_dict()
    suite = bd[task_suite_name]()
    success_rates = []

    for task_id in range(len(suite)):
        task = suite.get_task(task_id)
        bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file, camera_heights=256, camera_widths=256
        )
        init_states = suite.get_task_init_states(task_id)
        successes = 0
        for ep in range(n_episodes):
            env.reset()
            env.set_init_state(init_states[ep % len(init_states)])
            # run policy rollout ...
            # (see Phase 3 for rollout logic)
        success_rates.append(successes / n_episodes)
    return success_rates
```

**Evaluation suites**: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (in-distribution), `libero_90` (OOD generalization test).

---

## Phase 2 — Core Components Implementation

### 2.1 Custom Denoising Loop (for variable `num_steps`)

`VLAFlowMatching.sample_actions` reads `num_steps` from config. We need a wrapper that runs 8, 9, or 10 denoising steps with the **same noise vector** per trajectory.

```python
def rollout_with_n_steps(flow_model, prefix_embs_cache, noise, num_steps):
    """
    Run Euler integration for `num_steps` steps from a fixed noise tensor.
    Returns action chunk of shape (B, chunk_size, action_dim).
    
    flow_model: VLAFlowMatching instance (policy.model)
    prefix_embs_cache: past_key_values from embed_prefix (computed once, reused)
    noise: (B, chunk_size, action_dim) — fixed initial noise
    num_steps: int — 8, 9, or 10
    """
    dt = -1.0 / num_steps
    x_t = noise.clone()
    bsize = noise.shape[0]
    device = noise.device

    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
        v_t = flow_model.denoise_step(
            x_t=x_t,
            prefix_pad_masks=prefix_embs_cache["prefix_pad_masks"],
            past_key_values=prefix_embs_cache["past_key_values"],
            timestep=time_tensor,
        )
        x_t = x_t + dt * v_t
    return x_t
```

**Key insight**: The `past_key_values` KV cache from `embed_prefix` is computed once per observation and reused across all 3 denoising horizons (8, 9, 10 steps) for the same noise vector. This dramatically reduces compute.

### 2.2 Trajectory Sampling with Independent Noise (GRPO Group)

For each observation context, sample `n` independent noise vectors:

```python
def sample_group_trajectories(policy, obs_batch, n_group=8):
    """
    Sample n trajectories from independent noise vectors.
    Returns list of (noise_i, actions_i) tuples.
    """
    model = policy.model  # VLAFlowMatching
    B = 1  # per-observation rollout (one env step at a time)
    action_shape = (B, model.config.chunk_size, model.config.max_action_dim)

    # Embed observation prefix once (expensive VLM forward pass)
    prefix_cache = compute_prefix_cache(model, obs_batch)

    group = []
    for i in range(n_group):
        zi = model.sample_noise(action_shape, device=obs_batch.device)
        # Three denoising horizons per noise vector
        a8  = rollout_with_n_steps(model, prefix_cache, zi, num_steps=8)
        a9  = rollout_with_n_steps(model, prefix_cache, zi, num_steps=9)
        a10 = rollout_with_n_steps(model, prefix_cache, zi, num_steps=10)
        group.append({"noise": zi, "actions_8": a8, "actions_9": a9, "actions_10": a10})
    return group
```

### 2.3 Weighted Denoising Reward

Execute each action chunk in the LIBERO env and collect binary success:

```python
def compute_weighted_reward(env, task, group_trajectory):
    """
    R(τ_i) = 0.1 * r(τ_i^8) + 0.2 * r(τ_i^9) + 0.7 * r(τ_i^10)
    Returns scalar reward in [0, 1].
    """
    r8  = execute_and_score(env, group_trajectory["actions_8"])   # binary {0, 1}
    r9  = execute_and_score(env, group_trajectory["actions_9"])   # binary {0, 1}
    r10 = execute_and_score(env, group_trajectory["actions_10"])  # binary {0, 1}
    return 0.1 * r8 + 0.2 * r9 + 0.7 * r10
```

### 2.4 GRPO Advantage Computation

```python
def compute_grpo_advantages(rewards, eps=1e-8):
    """
    Â_i = (r_i - mean(r)) / (std(r) + ε)
    rewards: list of n scalar rewards
    Returns tensor of shape (n,)
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    advantages = (r - r.mean()) / (r.std() + eps)
    return advantages
```

**Zero-update property**: When all trajectories succeed or all fail, `std(r) = 0` → advantages ≈ 0 → no gradient update. This is the built-in curriculum from GRPO (§2.2 of proposal).

### 2.5 Flow-Matching Log Probability for Policy Ratio

The policy gradient requires `log π_θ(τ_i)` for the importance sampling ratio `ρ_i(θ) = π_θ / π_θ_old`. For flow-matching, this is computed via the flow-matching loss over the denoising path:

```python
def flow_matching_log_prob(policy, obs_batch, noise, actions_target):
    """
    Approximate log π_θ(τ | o) via the flow-matching regression loss
    (score function estimator approach from §4.4 of proposal).
    
    L_FM(θ) = E_{t, a~p1, z~p0} [ ||v_θ(x_t, t) - (a - z)||^2 ]
    log π_θ ≈ -L_FM (negative loss as proxy log-prob)
    """
    loss_dict = policy.forward(obs_batch, noise=noise)
    return -loss_dict["loss"]  # higher log-prob = lower FM loss
```

### 2.6 GRPO Policy Update

```python
def grpo_update(policy, policy_ref, optimizer, obs_batch, group_data, advantages, clip_eps=0.2):
    """
    L_GRPO = -E[min(ρ_i * Â_i, clip(ρ_i, 1-ε, 1+ε) * Â_i)]
    where ρ_i = π_θ(τ_i) / π_θ_old(τ_i)
    """
    total_loss = 0
    for traj, adv in zip(group_data, advantages):
        log_prob_new = flow_matching_log_prob(policy, obs_batch, traj["noise"], traj["actions_10"])
        with torch.no_grad():
            log_prob_old = flow_matching_log_prob(policy_ref, obs_batch, traj["noise"], traj["actions_10"])

        ratio = torch.exp(log_prob_new - log_prob_old)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss = -torch.min(ratio * adv, clipped_ratio * adv)
        total_loss += loss

    total_loss = total_loss / len(group_data)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()
```

---

## Phase 3 — Full GRPO Training Loop

### 3.1 Training Configuration

```python
GRPO_CONFIG = {
    "model_id": "lerobot/smolvla_libero",     # SFT checkpoint
    "task_suite": "libero_10",                # training tasks
    "n_group": 8,                             # trajectories per observation
    "learning_rate_backbone": 1e-5,           # low LR for VLM backbone
    "learning_rate_head": 1e-4,               # higher LR for action expert
    "clip_eps": 0.2,                          # GRPO clipping threshold
    "kl_coeff": 0.01,                         # KL regularization against SFT prior
    "total_steps": 5000,                      # RL fine-tuning steps
    "eval_every": 500,                        # evaluation frequency
    "denoising_weights": {8: 0.1, 9: 0.2, 10: 0.7},
    "output_dir": "./checkpoints/grpo_smolvla",
}
```

### 3.2 Main Training Loop Pseudocode

```python
def train_grpo():
    # 1. Load SFT policy (trainable) and frozen reference policy
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").cuda()
    policy_ref = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").cuda()
    policy_ref.requires_grad_(False)

    # 2. Optimizer with differential LR
    optimizer = torch.optim.AdamW([
        {"params": policy.model.vlm_with_expert.vlm.parameters(), "lr": 1e-5},
        {"params": policy.model.vlm_with_expert.action_expert.parameters(), "lr": 1e-4},
    ])

    # 3. LIBERO environment setup
    suite = get_libero_suite("libero_10")

    for step in range(GRPO_CONFIG["total_steps"]):
        # Sample a task and initial state
        task = random.choice(suite.tasks)
        env = make_env(task)
        obs = reset_env(env)

        # Sample group of n trajectories
        group_data = sample_group_trajectories(policy, obs, n_group=8)

        # Compute weighted rewards for each trajectory
        rewards = []
        for traj in group_data:
            env.reset(); env.set_init_state(init_state)
            r = compute_weighted_reward(env, task, traj)
            rewards.append(r)

        # GRPO advantages (zero update if all succeed or all fail)
        advantages = compute_grpo_advantages(rewards)

        # Policy gradient update
        loss = grpo_update(policy, policy_ref, optimizer, obs, group_data, advantages)

        if step % GRPO_CONFIG["eval_every"] == 0:
            evaluate_policy(policy, "libero_10")

        print(f"Step {step}: loss={loss:.4f}, mean_reward={sum(rewards)/len(rewards):.3f}")
```

---

## Phase 4 — Project File Structure

```
robo-proj/
├── grpo_smolvla/
│   ├── __init__.py
│   ├── env_utils.py          # LIBERO env wrapper, rollout execution
│   ├── flow_utils.py         # custom denoising loop, prefix cache
│   ├── grpo.py               # advantage computation, GRPO loss
│   ├── rewards.py            # weighted denoising reward
│   ├── train_grpo.py         # main training script (Phase 3.2)
│   └── evaluate.py           # LIBERO benchmark evaluation
├── configs/
│   └── grpo_config.yaml      # training hyperparameters
├── checkpoints/
│   └── grpo_smolvla/         # saved GRPO checkpoints
└── PLAN.md                   # this file
```

---

## Phase 5 — Evaluation Protocol

### 5.1 Metrics

| Metric | Description |
|---|---|
| Task success rate | Binary {0,1} per episode, averaged over N episodes per task |
| Mean reward | Weighted denoising reward (training signal) |
| OOD generalization | Eval on `libero_goal`, `libero_spatial`, `libero_object` (unseen during GRPO) |

### 5.2 Evaluation Command

```bash
python grpo_smolvla/evaluate.py \
  --checkpoint checkpoints/grpo_smolvla/step_5000 \
  --suites libero_10 libero_spatial libero_object libero_goal \
  --n_episodes 20
```

### 5.3 Baseline Comparison

| Model | libero_10 | libero_spatial | libero_object | libero_goal |
|---|---|---|---|---|
| SFT (`smolvla_libero`) | ? | ? | ? | ? |
| GRPO-SmolVLA (ours) | ? | ? | ? | ? |

---

## Installation Steps (In Order)

```bash
# Step 1: Activate environment
conda activate robot-proj

# Step 2: Install lerobot with smolvla extras
uv pip install "lerobot[smolvla]"
# Verify
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('lerobot SmolVLA OK')"

# Step 3: Verify LIBERO environment
python -c "from libero.libero import benchmark; print('LIBERO OK')"

# Step 4: Download SmolVLA LIBERO checkpoint (first run only — cached by HF hub)
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; SmolVLAPolicy.from_pretrained('lerobot/smolvla_libero')"

# Step 5: Create project structure
mkdir -p grpo_smolvla configs checkpoints/grpo_smolvla
```

---

## Key Technical Decisions

### Decision 1: `num_steps` Override Strategy

The current `VLAFlowMatching.sample_actions` reads `num_steps` from `self.config.num_steps`. We will implement `rollout_with_n_steps()` (Phase 2.1) that directly runs the Euler loop, bypassing config. This avoids mutating config state and is thread-safe.

### Decision 2: KV Cache Reuse Across Denoising Horizons

For each observation, `embed_prefix` (the expensive VLM forward pass) is called once. The `past_key_values` KV cache is then reused for all 3 denoising horizons (8, 9, 10 steps) per noise vector `z_i`. This reduces compute by ~3x per trajectory.

### Decision 3: Log-Probability via Flow-Matching Loss

The exact log-probability of a continuous-flow trajectory is intractable. We use the flow-matching regression loss as a proxy (score function estimator), consistent with §4.4 of the proposal. Specifically: `log π_θ(τ_i) ≈ -L_FM(θ; noise_i, actions_i)`.

### Decision 4: Differential Learning Rates

VLM backbone gets `lr=1e-5`, action expert head gets `lr=1e-4`. This preserves the pretrained visual-language representations while allowing the action expert to adapt quickly to RL signal (§4.5 of proposal).

### Decision 5: Use `robot-proj` env, install lerobot fresh

Rather than switching to `so-arm101-vla` (which has lerobot 0.4.3), we install lerobot into `robot-proj` via `uv pip install`. This keeps all LIBERO simulation dependencies (already in `robot-proj`) in the same env.

---

## Risk Items & Mitigations

| Risk | Mitigation |
|---|---|
| GRPO advantage collapse (all fail or all succeed) | Built into GRPO: std=0 → zero gradient. Tasks should have ~30-70% success for meaningful signal. Start with `libero_10` which has the highest baseline success. |
| Backprop through Euler integration is expensive | Only backprop through 10-step trajectory for the GRPO loss. 8-step and 9-step rewards are computed `torch.no_grad()` for reward signal only. |
| LIBERO environment reset is slow | Pre-sample all initial states. Use `OffScreenRenderEnv` (headless). Parallelize environments if multiple GPUs available. |
| Policy drifts too far from SFT prior | KL divergence regularization term against frozen `policy_ref` (§4.5). |
| `lerobot` version conflicts with existing packages | Install with `uv pip install` which handles conflicts better; `so-arm101-vla` confirms 0.4.3 works. |
