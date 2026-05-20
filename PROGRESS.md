# GRPO-SmolVLA Implementation Progress

> Caveman backup — update this after each phase completes.
> If session dies, resume from last "✅ Done" item.

---

## Phase 0 — Environment Setup

- [x] 0.1 Install lerobot in `robot-proj` ✅ lerobot 0.4.x with SmolVLAPolicy OK
- [x] 0.2 Verify LIBERO environment ✅ libero_10 tasks load fine
- [x] 0.3 Verify SmolVLA loads from pretrained checkpoint ✅ num_steps=10

## Phase 1 — Baseline Evaluation (SFT)

- [x] Understand evaluate structure (done in plan) ✅

## Phase 2 — Core Components Implementation

- [x] 2.1 `flow_utils.py` — custom denoising loop + prefix cache ✅
- [x] 2.2 `flow_utils.py` — group trajectory sampling ✅
- [x] 2.3 `rewards.py` — weighted denoising reward ✅
- [x] 2.4 `grpo.py` — GRPO advantage computation ✅
- [x] 2.5 `grpo.py` — flow-matching log probability ✅
- [x] 2.6 `grpo.py` — GRPO policy update ✅
- [x] `env_utils.py` — LIBERO env wrapper + obs preprocessing ✅

## Phase 3 — Full GRPO Training Loop

- [x] 3.1 `configs/grpo_config.yaml` — training config ✅
- [x] 3.2 `train_grpo.py` — main training loop ✅

## Phase 4 — Project File Structure

- [x] Create all module files under `grpo_smolvla/` ✅
- [x] Create `configs/` and `checkpoints/` dirs ✅

## Phase 5 — Evaluation

- [x] `evaluate.py` — LIBERO benchmark evaluation script ✅

## Phase 7 — GRPO Training

### Implementation status (2026-05-07)

All core components fixed and smoke-tested. 3-step training run completed successfully.

**Bugs fixed in this phase:**

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `flow_utils.py` | `compute_prefix_cache` passed dict to `embed_prefix` (expects tensors) | Rewrote to use `policy.prepare_images/state` + correct `embed_prefix` + KV-cache from `vlm_with_expert.forward` |
| 2 | `grpo.py` | `policy.forward()` returns `(loss, loss_dict)` not dict; `loss_dict["loss"]` is `.item()` (no grad) | Unpacked tuple, returned `-loss` tensor directly |
| 3 | `grpo.py` | `policy.forward(batch)` needs `ACTION` key in batch | Added `batch = {**obs_batch, ACTION: actions_target}` |
| 4 | `flow_utils.py` | Denoised actions are 32-dim (`max_action_dim`); postprocessor expects 7-dim | Trim to `policy.config.action_feature.shape[0]` (=7) after denoising, matching `_get_action_chunk` |
| 5 | `train_grpo.py` | Used `env.get_obs()` (doesn't exist), wrong `obs_to_policy_input` signature | Rewrote to use `LiberoEnv` + lerobot preprocessing pipeline (mirrors `evaluate.py`) |
| 6 | `train_grpo.py` | `len(suite)` raises TypeError | Changed to `suite.get_num_tasks()` |
| 7 | `train_grpo.py` | `policy.model.vlm_with_expert.action_expert` doesn't exist | Changed to `lm_expert` (confirmed via `SmolVLMWithExpertModel` children) |
| 8 | `rewards.py` | Actions not denormalized before `env.step()` | Added `postprocessor` param; denormalize before executing |
| 9 | `rewards.py` | No sim state save/restore between sub-trajectories | Save `sim.get_state()` + `env.timestep` after reset; restore + clear obs cache before each horizon |

**Verified:**
- `flow_matching_log_prob` returns tensor with `requires_grad=True` ✅
- Prefix KV-cache computed correctly via `embed_prefix` + `vlm_with_expert.forward` ✅
- 3 denoising horizons (8/9/10 steps) from same noise vector ✅
- GRPO update with real LIBERO env: 3 steps, checkpoint saved ✅

**To run full training:**
```bash
conda activate robot-proj
python -m grpo_smolvla.train_grpo --config configs/grpo_config.yaml
```

**Config:** `configs/grpo_config.yaml` — 5000 steps, n_group=8, libero_10.

---

## Notes

- LIBERO: ✅ installed in `robot-proj`
- lerobot: ✅ installed in `robot-proj`
- MuJoCo 3.7, robosuite 1.4: ✅ confirmed present
- Dataset path: `/home/shinawatra/Works/personal/robo-proj/LIBERO/libero/datasets/`

## Phase 6 — Baseline Evaluation ✅ COMPLETE

### Final SFT Baseline Results (2026-05-06)

**Command:**
```bash
conda activate robot-proj
python -m grpo_smolvla.evaluate --checkpoint lerobot/smolvla_libero --suites libero_10 --n_episodes 20
```

**Results (libero_10, 20 episodes/task, 10 tasks):**

| Task | Success Rate | Description |
|------|-------------|-------------|
| 0 | 20.00% | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 60.00% | put both the cream cheese box and the butter in the basket |
| 2 | 85.00% | turn on the stove and put the moka pot on it |
| 3 | 95.00% | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 25.00% | put the white mug on the left plate and put the yellow and white mug... |
| 5 | 70.00% | pick up the book and place it in the back compartment of the caddy |
| 6 | 30.00% | put the white mug on the plate and put the chocolate pudding... |
| 7 | 40.00% | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 35.00% | put both moka pots on the stove |
| 9 | 55.00% | put the yellow and white mug in the microwave and close it |
| **Mean** | **51.50%** | |

**Benchmark comparison (libero_10 / Long):**

| Source | Success Rate | Notes |
|--------|-------------|-------|
| Paper (SmolVLA) | 71% | Official paper result |
| HF Leaderboard | 60% | HuggingFaceVLA/libero-vla-leaderboard |
| **Our SFT baseline** | **51.50%** | Official lerobot pipeline, 20 eps/task |
| Community #3287 | 44.8% | 100k steps finetune from scratch, 10 eps/task |
| Community #2354 | 43% | `n_action_steps=1`, 10 eps/task |

**Analysis:**
- 51.5% is above all community reproductions in GitHub issues #2354 and #3287
- Gap to leaderboard (60%) likely due to: evaluation randomness (only 50 init states), possible lerobot version differences, or leaderboard using more episodes
- GitHub issue #2354 (still open) confirms this is a known gap — not our bug

**Next: Phase 7 — GRPO Training (in progress)**

## Phase 6 — Baseline Evaluation Debugging (history)

Command to run:
```bash
conda activate robot-proj
python -m grpo_smolvla.evaluate --checkpoint lerobot/smolvla_libero --suites libero_10 --n_episodes 20
```

### Bug log (all fixed except 0% success rate — root cause still under investigation)

| # | Error | Root Cause | Fix Applied |
|---|-------|-----------|-------------|
| 1 | `ModuleNotFoundError: grpo_smolvla` | Ran as script instead of module | Use `python -m grpo_smolvla.evaluate` |
| 2 | `TypeError: object of type 'LIBERO_10' has no len()` | LIBERO suite has no `__len__` | Replace `len(suite)` → `suite.get_num_tasks()` in `evaluate.py` and `env_utils.py` |
| 3 | `AttributeError: 'OffScreenRenderEnv' has no 'get_obs'` | LIBERO env returns obs from `reset()` / `step()`, no `get_obs()` | Changed to `raw_obs = env.set_init_state(...)` which returns obs directly |
| 4 | `ValueError: All image features missing` | We sent `agentview`/`eye_in_hand` keys; policy expects `camera1`/`camera2` | Renamed image keys in `obs_to_policy_input` |
| 5 | `ValueError: All image features missing` (same) | Also: state was wrong shape — policy trained on `(6,)` (believed at the time) | Mapped cameras + attempted 6-dim state |
| 6 | `KeyError: observation.language.tokens` | Policy requires pre-tokenised language, not raw string `"task"` key | Added tokenisation with `policy.model.vlm_with_expert.processor.tokenizer`; outputs `observation.language.tokens` and `.attention_mask` |
| 7 | `RuntimeError: where expected boolean tensor, got Long` | Tokenizer attention mask is `int64`, policy attention code calls `torch.where` expecting bool | Added `.bool()` cast on attention mask |
| 8 | `ValueError: executing action in terminated episode` | In LIBERO `bddl_base_domain.step()`, `done = _check_success()` (only True on success). Robosuite's internal `self.done` is set True on max steps, but the returned `done` stays False → loop keeps stepping a terminated env | Added `MAX_EPISODE_STEPS=600` cap + `try/except ValueError` + `policy.reset()` between episodes |
| 9 | **0% success rate (current issue)** | Three suspected causes:<br>① State was 8-dim in training (`eef_pos+axis_angle+gripper_qpos`) not 6-dim — confirmed by normalization stats in checkpoint<br>② No MEAN_STD normalisation applied to state before feeding policy<br>③ No MEAN_STD denormalisation on policy action output before sending to env<br>④ Controller not set to relative mode (`use_delta=True`); no stabilisation no-ops after reset | Applied fixes: `load_norm_stats()` loads stats from checkpoint safetensors; state normalised; actions denormalised; `reset_env()` sets `use_delta=True` + 10 no-op steps. **Still 0% — deeper issue likely remains.** |

### Root causes of 0% success — CONFIRMED via online research (2026-05-06)

Sources: lerobot source code (`src/lerobot/processor/env_processor.py`, `src/lerobot/envs/libero.py`),
HF docs (huggingface.co/docs/lerobot/libero), GitHub issue #2354.

| # | Root cause | Confirmed by | Fix needed |
|---|-----------|-------------|-----------|
| A | **Images NOT flipped 180°** | `LiberoProcessorStep._process_observation` does `torch.flip(img, dims=[2,3])` — our code never flips | ✅ Fixed: `agentview[::-1,::-1,:]` + `wrist[::-1,::-1,:]` (HWC numpy flip) in `obs_to_policy_input` |
| B | **Image key names** | train_config rename_map: `image`→`camera1`, `image2`→`camera2`; policy config uses `camera1`/`camera2`/`camera3`. Original keys were correct. `camera3` is absent in LIBERO (from other datasets) and `empty_cameras=0` means it is simply skipped. | ✅ Keys correctly kept as `camera1`/`camera2` (reverted mistaken rename to `image`/`image2`) |
| C | **State is 8-dim confirmed** | `LiberoProcessorStep.transform_features` sets `shape=(8,)`, HF docs confirm; our 8-dim construction is correct | No change needed |
| D | **`use_delta` order** | LiberoEnv sets `use_delta=True` AFTER no-op steps, our code set it before | ✅ Fixed: moved `use_delta = True` loop to after the no-op stabilisation loop in `reset_env` |

#### Online findings summary
- **HF LIBERO docs**: `observation.state` = 8-dim (eef_pos 3 + axis_angle 3 + gripper_qpos 2); cameras → `observation.images.image` + `.image2`; actions `Box(-1,1,(7,))`; control_mode `relative` is default.
- **`LiberoProcessorStep`**: images flipped 180° with `torch.flip(img, dims=[2,3])` — **this is the primary bug causing 0%**.
- **GitHub issue #2354**: Known reproduction gap even with official `lerobot-eval` CLI (open, unresolved). Even after fixes, results may be below paper. But 0% is definitely our bug.
- **`TASK_SUITE_MAX_STEPS`** from LiberoEnv: libero_spatial=280, libero_object=280, libero_goal=300, libero_10=520. Our 600-step cap is fine.


### Status after full rewrite (2026-05-06) — **FIXED** ✅

Rewrote `evaluate.py` to use the official lerobot pipeline end-to-end:
- `LiberoEnv` (from lerobot) → handles reset, control_mode, init_states
- `preprocess_observation` → numpy→tensor, channel-first images
- `PolicyProcessorPipeline([LiberoProcessorStep()])` → flips images, constructs 8-dim state
- Manual rename `image→camera1`, `image2→camera2` (training rename_map)
- `make_smolvla_pre_post_processors` → tokenize, normalize state, unnormalize action

**Result: 3 tasks × 3 episodes → 66.67% mean success rate** (task 0: 33%, task 1: 100%, task 2: 67%)
Expected range for libero_10: ~43% full-suite from GitHub #2354 — our partial result is consistent.

Root causes that remained after original A/B/D fixes:
- E: `_quat_to_axisangle` different from lerobot (ours used abs(w)+sign, lerobot uses plain w)
- F: Language tokenization missing "\n" suffix (SmolVLANewLineProcessor)
- G: Bypassed `LiberoProcessorStep` entirely (state construction + image flip happened inside)
- H: `make_smolvla_pre_post_processors` needed training rename_map `image→camera1`

All resolved by using the lerobot pipeline directly.

---

#### Community findings (from Firecrawl search, 2026-05-06)

**GitHub #2354** — "Cannot reproduce SmolVLA results on LIBERO benchmark" (open):
```
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --policy.num_steps=10 --policy.n_action_steps=1 ...
```
Multiple community members report:
| | Spatial | Object | Goal | Long (libero_10) |
|---|---|---|---|---|
| Leaderboard | 0.9 | 1.0 | 1.0 | 0.6 |
| Paper | 0.90 | 0.96 | 0.92 | 0.71 |
| Community repro | 0.73–0.83 | 0.91–0.96 | 0.83–0.87 | 0.38–0.43 |

Note: Using `n_action_steps=10` (default) vs `n_action_steps=1` significantly affects libero_10 long-horizon tasks.

**GitHub #3287** — "Inquiry about Training Configurations for Replicating SmolVLA on LIBERO":
- User fine-tuned from scratch (batch=64, 100k steps): libero_10=44.8%, spatial=83%, object=70%, goal=70%
- Still below paper — suggests training config gap, not just eval bug

**LIBERO-PRO paper (arXiv 2510.03827)**: notes that VLA policies can fail under simple perturbations despite >90% success on standard LIBERO — our 51.5% SFT baseline is a realistic starting point for GRPO improvement.
