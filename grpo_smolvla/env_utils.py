"""LIBERO environment utilities: task loading, env creation, obs preprocessing."""

import os
import random

import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


CAMERA_H = 256
CAMERA_W = 256
NUM_STEPS_WAIT = 10  # stabilization no-ops after reset, matching lerobot LiberoEnv


def get_libero_path(key):
    """Return the path to a LIBERO resource directory."""
    import libero.libero as libero_pkg
    pkg_root = os.path.dirname(os.path.abspath(libero_pkg.__file__))
    paths = {
        "bddl_files": os.path.join(pkg_root, "bddl_files"),
        "init_states": os.path.join(pkg_root, "init_states"),
    }
    return paths[key]


def make_libero_suite(task_suite_name):
    """Return a LIBERO benchmark suite by name."""
    bd = benchmark.get_benchmark_dict()
    return bd[task_suite_name]()


def make_env(task, bddl_root=None):
    """Create an OffScreenRenderEnv for a LIBERO task."""
    if bddl_root is None:
        bddl_root = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=CAMERA_H,
        camera_widths=CAMERA_W,
    )
    return env


def reset_env(env):
    """Reset env, enable relative control, and return stabilised obs."""
    raw_obs = env.reset()
    # Stabilise physics first (matching lerobot LiberoEnv order), then enable delta
    noop = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(NUM_STEPS_WAIT):
        raw_obs, _, _, _ = env.step(noop)
    # Set relative (delta) control mode AFTER no-ops — matches smolvla_libero training
    for robot in env.robots:
        robot.controller.use_delta = True
    return raw_obs


def load_norm_stats(checkpoint_id):
    """Load state normalization and action denormalization stats from a checkpoint."""
    pre_path = hf_hub_download(checkpoint_id, "policy_preprocessor_step_5_normalizer_processor.safetensors")
    post_path = hf_hub_download(checkpoint_id, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    pre = load_file(pre_path)
    post = load_file(post_path)
    return {
        "state_mean": pre["observation.state.mean"],
        "state_std":  pre["observation.state.std"],
        "action_mean": post["action.mean"],
        "action_std":  post["action.std"],
    }


def _quat_to_axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle (3,)."""
    quat = np.array(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    x, y, z, w = quat
    angle = 2.0 * np.arccos(np.clip(np.abs(w), 0.0, 1.0))
    if w < 0:
        angle = -angle
    sin_half = np.sqrt(1.0 - np.clip(w * w, 0.0, 1.0))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z]) / sin_half
    return (axis * angle).astype(np.float32)


def obs_to_policy_input(obs, task_language, device, tokenizer, norm_stats,
                        tokenizer_max_length=48):
    """
    Convert a raw LIBERO obs dict into the format expected by SmolVLAPolicy.

    Camera keys: camera1 (agentview), camera2 (wrist).
    State: eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8-dim, MEAN_STD normalised.
    Language: pre-tokenised to observation.language.tokens / .attention_mask.
    """
    def to_tensor(x):
        arr = np.array(x, dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    eef_axisangle = _quat_to_axisangle(obs["robot0_eef_quat"])
    state_raw = np.concatenate([
        obs["robot0_eef_pos"],       # (3,)
        eef_axisangle,               # (3,)
        obs["robot0_gripper_qpos"],  # (2,)
    ])

    state_tensor = to_tensor(state_raw)
    state_mean = norm_stats["state_mean"].to(device)
    state_std  = norm_stats["state_std"].to(device)
    state_norm = (state_tensor - state_mean) / (state_std + 1e-8)

    tokens = tokenizer(
        task_language,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
    )

    # Images must be flipped 180° to match lerobot LiberoProcessorStep._process_observation
    agentview = obs["agentview_image"][::-1, ::-1, :]       # HWC flip → still HWC
    wrist      = obs["robot0_eye_in_hand_image"][::-1, ::-1, :]

    return {
        "observation.images.camera1": to_tensor(agentview.transpose(2, 0, 1) / 255.0),
        "observation.images.camera2": to_tensor(wrist.transpose(2, 0, 1)     / 255.0),
        "observation.state": state_norm,
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }


def denormalize_action(action_tensor, norm_stats, device):
    """Denormalize policy output (MEAN_STD) back to robot action space."""
    mean = norm_stats["action_mean"].to(device)
    std  = norm_stats["action_std"].to(device)
    return action_tensor * std + mean


def sample_random_task_and_init(suite):
    """Sample a random task and a random initial state from the suite."""
    n_tasks = suite.get_num_tasks()
    task_id = random.randint(0, n_tasks - 1)
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    init_state = init_states[random.randint(0, len(init_states) - 1)]
    return task, init_state
