# Design: LIBERO Image Sequence Visualizer

**Date:** 2026-05-06
**Status:** Approved

---

## Goal

A standalone script that produces a single PNG showing:
1. A row of 8 evenly-sampled initial environment states for a chosen task
2. N rows of 8 evenly-spaced frames from full SmolVLA policy rollouts, labelled with success/fail

---

## CLI

```bash
python -m grpo_smolvla.visualize \
  --suite libero_10 \
  --task_id 0 \
  --n_rollouts 3 \
  --checkpoint lerobot/smolvla_libero \
  --output vis_libero_10_task0.png
```

| Argument | Default | Description |
|---|---|---|
| `--suite` | `libero_10` | LIBERO benchmark suite name |
| `--task_id` | `0` | Task index within the suite |
| `--n_rollouts` | `3` | Number of policy rollout episodes to show |
| `--checkpoint` | `lerobot/smolvla_libero` | HF repo or local path |
| `--output` | `vis_{suite}_task{id}.png` | Output PNG path |

---

## Output Layout

```
Title: "Task {id}: {task_description}"

Row 0  [Init states]   img img img img img img img img   ← 8 evenly-sampled init states
Row 1  [Ep 0  ✓]       img img img img img img img img   ← 8 evenly-spaced rollout frames
Row 2  [Ep 1  ✗]       img img img img img img img img
Row 3  [Ep 2  ✓]       img img img img img img img img
                        t=0 t=65 t=130 ...               ← step labels under rollout rows
```

- Each row has exactly 8 columns (consistent grid width)
- Row labels on the left: `"Init states"` (black), `"Ep {i} ✓"` (green), `"Ep {i} ✗"` (red)
- Step labels below each rollout frame: `t=0`, `t=N`, `t=2N`, ...
- Images flipped vertically at display time (`frame[::-1]`) — raw LIBERO images are upside-down
- `agentview_image` camera used for all frames (256×256)
- Saved with `dpi=150`, `bbox_inches="tight"`

---

## File

Single new file: `grpo_smolvla/visualize.py`

No new dependencies beyond what `evaluate.py` already uses (`matplotlib`, `torch`, `torchvision`, `PIL` already available).

---

## Components

### `collect_init_frames(suite, task_id, n=8) → list[ndarray]`

Mirrors the reference LIBERO example code:
- Creates `OffScreenRenderEnv` with `camera_heights=256`, `camera_widths=256`
- Loads all init states for the task via `suite.get_task_init_states(task_id)`
- Picks `n` evenly-spaced indices from all init states
- For each: calls `env.set_init_state(...)`, takes 5 no-op steps (`[0.]*7`), captures `obs["agentview_image"]`
- Returns list of `n` numpy `(H, W, C)` uint8 arrays
- Closes env after collection

### `collect_rollout_frames(env, policy, env_preprocessor, preprocessor, postprocessor, max_steps) → (list[ndarray], bool)`

Reuses the exact inference pipeline from `evaluate.py`:
- Runs `env.reset()` then the full SmolVLA loop:
  `preprocess_observation → batch_dim_fix → env_preprocessor → rename → preprocessor → select_action → postprocessor → env.step`
- Records every `agentview_image` (raw, before flip) during the episode
- On termination or `max_steps`, samples 8 evenly-spaced frames from all recorded frames
- Returns `(frames: list[8 ndarray], success: bool)` where success comes from `info["is_success"]`

### `plot_and_save(init_frames, rollout_results, task_description, output_path)`

- `rollout_results: list[tuple[list[ndarray], bool]]`
- Creates `matplotlib` figure: `(1 + n_rollouts)` rows × 8 columns
- Row 0: init frames, label `"Init states"`
- Rows 1–N: rollout frames, label `"Ep {i} ✓"` (green) or `"Ep {i} ✗"` (red)
- Step numbers rendered as `ax.text(...)` below each rollout frame cell
- `ax.imshow(frame[::-1])` on every cell; all axes set `axis("off")`
- Left-side row labels via `fig.text(...)` at the correct y-position
- Saves with `plt.savefig(output_path, dpi=150, bbox_inches="tight")`

### `main()`

1. Parse args
2. Load policy + dataset stats + build preprocessors (same as `evaluate.py` main)
3. Build `LiberoEnv` for the task (reuse for all rollouts)
4. Call `collect_rollout_frames` × `n_rollouts`
5. Call `collect_init_frames` (separate `OffScreenRenderEnv`, no policy needed)
6. Call `plot_and_save`
7. Print output path

---

## Data Flow

```
OffScreenRenderEnv
    └─ collect_init_frames() → [8 × (H,W,C) ndarray]

LiberoEnv + SmolVLA pipeline
    └─ collect_rollout_frames() × n_rollouts → [(8 frames, success)] × n

plot_and_save()
    └─ matplotlib figure → output.png
```

---

## Key Details

- **Image flip**: raw LIBERO `agentview_image` is upside-down; flip at display time with `frame[::-1]` (not during collection, to stay consistent with env state)
- **Init state sampling**: `indices = np.linspace(0, len(init_states)-1, 8, dtype=int)` — evenly spaced, no randomness
- **Rollout frame sampling**: record all frames, then `np.linspace(0, len(frames)-1, 8, dtype=int)` at end
- **Max steps**: uses `env._max_episode_steps` (520 for libero_10) as the rollout cap
- **Policy reset**: `policy.reset()` called before each rollout episode
- **No video output** — PNG only in this version
