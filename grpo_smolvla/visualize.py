"""LIBERO image sequence visualizer.

Saves a PNG with init-state frames (row 0) and SmolVLA rollout frames (rows 1-N).
Usage:
    python -m grpo_smolvla.visualize --suite libero_10 --task_id 0 --n_rollouts 3
"""

import argparse
import os

import numpy as _np_compat
if not hasattr(_np_compat, "Inf"):
    _np_compat.Inf = _np_compat.inf  # matplotlib 3.5.x uses removed np.Inf on numpy 2.x
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep

from grpo_smolvla.env_utils import get_libero_path
from grpo_smolvla.evaluate import load_dataset_stats

N_FRAMES = 8  # fixed columns per row


def collect_init_frames(suite, task_id: int, n: int = N_FRAMES) -> list:
    """Return n evenly-spaced init-state frames as (H,W,C) uint8 numpy arrays.

    Mirrors the reference LIBERO example: set init state, run 5 no-op steps, capture.
    Images are raw (upside-down) — caller flips at display time.
    """
    task = suite.get_task(task_id)
    bddl_root = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
    )

    init_states = suite.get_task_init_states(task_id)
    indices = np.linspace(0, len(init_states) - 1, n, dtype=int)

    frames = []
    env.reset()
    for idx in indices:
        env.set_init_state(init_states[idx])
        for _ in range(5):
            obs, _, _, _ = env.step([0.0] * 7)
        frames.append(obs["agentview_image"].copy())

    env.close()
    return frames


def collect_rollout_frames(
    env: LiberoEnv,
    policy,
    env_preprocessor,
    preprocessor,
    postprocessor,
    max_steps: int,
    n_frames: int = N_FRAMES,
) -> tuple:
    """Run one episode with SmolVLA and return (frames, step_indices, success).

    frames: list of n_frames (H,W,C) uint8 numpy arrays, evenly spaced from episode.
    step_indices: list of int, the actual step numbers of the sampled frames.
    success: bool from info["is_success"].

    Reuses the exact inference pipeline from evaluate.py.
    """
    policy.reset()
    obs, info = env.reset()
    task_language = env.task_description

    all_frames = []
    success = False

    for _ in range(max_steps):
        # Capture raw agentview frame (upside-down) before policy step
        all_frames.append(obs["pixels"]["image"].copy())

        # --- inference pipeline (identical to evaluate.py) ---
        obs_t = preprocess_observation(obs)

        rs_key = "observation.robot_state"
        if rs_key in obs_t:
            def _add_batch(d):
                return {
                    k: (_add_batch(v) if isinstance(v, dict) else v.unsqueeze(0))
                    for k, v in d.items()
                }
            obs_t[rs_key] = _add_batch(obs_t[rs_key])

        obs_t["task"] = task_language
        obs_t = env_preprocessor(obs_t)

        if "observation.images.image" in obs_t:
            obs_t["observation.images.camera1"] = obs_t.pop("observation.images.image")
        if "observation.images.image2" in obs_t:
            obs_t["observation.images.camera2"] = obs_t.pop("observation.images.image2")

        obs_t = preprocessor(obs_t)

        with torch.inference_mode():
            action = policy.select_action(obs_t)

        action = postprocessor(action)
        act_np = action.squeeze(0).cpu().numpy()
        # --- end inference pipeline ---

        obs, _, terminated, truncated, info = env.step(act_np)
        if terminated or truncated:
            success = info.get("is_success", False)
            break

    # Sample n_frames evenly from all recorded frames
    total = len(all_frames)
    if total == 0:
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        return [blank] * n_frames, list(range(n_frames)), False

    step_indices = list(np.linspace(0, total - 1, n_frames, dtype=int))
    sampled = [all_frames[i] for i in step_indices]
    return sampled, step_indices, success


def plot_and_save(init_frames, rollout_results, task_description, output_path):
    """Save a PNG grid: row 0 = init states, rows 1-N = rollout episodes.

    init_frames: list of N_FRAMES (H,W,C) uint8 numpy arrays.
    rollout_results: list of (frames, step_indices, success) tuples.
    task_description: string for the figure title.
    output_path: path for the saved PNG.
    """
    n_rows = 1 + len(rollout_results)
    n_cols = N_FRAMES

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.5),
        gridspec_kw={"hspace": 0.05, "wspace": 0.02},
    )
    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = task_description if len(task_description) <= 90 else task_description[:87] + "..."
    fig.suptitle(title, fontsize=11, y=1.01)

    # --- Row 0: init states ---
    for col, frame in enumerate(init_frames):
        axes[0, col].imshow(frame[::-1])
        axes[0, col].axis("off")
    axes[0, 0].set_ylabel(
        "Init\nstates", rotation=0, labelpad=55, va="center", fontsize=9, color="black"
    )

    # --- Rollout rows ---
    for i, (frames, step_indices, success) in enumerate(rollout_results):
        row = i + 1
        badge = "✓" if success else "✗"
        color = "#2a7a2a" if success else "#bb2222"
        label = f"Ep {i}\n{badge}"

        for col, (frame, step) in enumerate(zip(frames, step_indices)):
            axes[row, col].imshow(frame[::-1])
            axes[row, col].axis("off")
            axes[row, col].set_xlabel(f"t={step}", fontsize=7, labelpad=2)
            axes[row, col].xaxis.set_label_position("bottom")

        axes[row, 0].set_ylabel(
            label, rotation=0, labelpad=55, va="center", fontsize=9,
            color=color, fontweight="bold",
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize LIBERO init states + SmolVLA rollouts")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--n_rollouts", type=int, default=3)
    parser.add_argument("--checkpoint", default="lerobot/smolvla_libero")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: vis_{suite}_task{id}.png)")
    args = parser.parse_args()

    output_path = args.output or f"vis_{args.suite}_task{args.task_id}.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load policy + processors (same as evaluate.py) ---
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device)
    policy.eval()

    dataset_stats = load_dataset_stats(args.checkpoint)
    policy.config.device = device
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset_stats
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    # --- Build suite and env ---
    bd = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()

    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
    )
    task_description = env.task_description
    max_steps = env._max_episode_steps

    # --- Collect rollout frames ---
    rollout_results = []
    for i in range(args.n_rollouts):
        print(f"Running rollout {i + 1}/{args.n_rollouts}...")
        frames, step_indices, success = collect_rollout_frames(
            env, policy, env_preprocessor, preprocessor, postprocessor, max_steps
        )
        rollout_results.append((frames, step_indices, success))
        status = "✓ success" if success else "✗ failed"
        print(f"  Rollout {i}: {status} (episode length: {step_indices[-1]} steps)")
    env.close()

    # --- Collect init frames (separate env) ---
    print("Collecting init state frames...")
    init_frames = collect_init_frames(suite, args.task_id)

    # --- Plot and save ---
    print(f"Saving plot to: {output_path}")
    plot_and_save(init_frames, rollout_results, task_description, output_path)
    print(f"Done. Saved: {output_path}")


# --- smoke tests ---
def _test_init_frames():
    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_10"]()
    frames = collect_init_frames(suite, task_id=0, n=8)
    assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"
    assert frames[0].shape == (256, 256, 3), f"Expected (256,256,3), got {frames[0].shape}"
    assert frames[0].dtype.name == "uint8", f"Expected uint8, got {frames[0].dtype}"
    print("collect_init_frames: OK")


def _test_rollout_frames_shape():
    fake_frames = [np.zeros((256, 256, 3), dtype=np.uint8)] * 30
    total = len(fake_frames)
    step_indices = list(np.linspace(0, total - 1, N_FRAMES, dtype=int))
    sampled = [fake_frames[i] for i in step_indices]
    assert len(sampled) == N_FRAMES
    assert len(step_indices) == N_FRAMES
    assert step_indices[0] == 0
    assert step_indices[-1] == total - 1
    print("collect_rollout_frames shape logic: OK")


def _test_plot_and_save(tmp_path="/tmp/vis_test.png"):
    rng = np.random.default_rng(0)
    init_frames = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(N_FRAMES)]
    rollout_results = [
        ([rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(N_FRAMES)],
         list(range(0, 520, 65)),
         True),
        ([rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(N_FRAMES)],
         list(range(0, 520, 65)),
         False),
    ]
    plot_and_save(init_frames, rollout_results, "test task description", tmp_path)
    assert os.path.exists(tmp_path), f"Output file not created: {tmp_path}"
    size = os.path.getsize(tmp_path)
    assert size > 10_000, f"Output file suspiciously small: {size} bytes"
    print(f"plot_and_save: OK — saved {size} bytes to {tmp_path}")


if __name__ == "__main__":
    main()
