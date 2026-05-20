"""Main GRPO training loop for SmolVLA on LIBERO."""

import argparse
import os
import random

import torch
import yaml
from huggingface_hub import hf_hub_download
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
from safetensors.torch import load_file

from grpo_smolvla.flow_utils import sample_group_trajectories
from grpo_smolvla.grpo import compute_grpo_advantages, grpo_update
from grpo_smolvla.rewards import compute_weighted_reward


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset_stats(checkpoint_id):
    """Parse flat safetensors stats into the nested {feature: {stat: tensor}} format."""
    path = hf_hub_download(
        checkpoint_id,
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    flat = load_file(path)
    stat_names = {"mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99", "count"}
    stats = {}
    for key, val in flat.items():
        for stat in stat_names:
            if key.endswith(f".{stat}"):
                feature = key[: -len(f".{stat}")]
                stats.setdefault(feature, {})[stat] = val
                break
    return stats


def preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device):
    """Convert LiberoEnv obs dict to SmolVLA policy input batch.

    Mirrors the exact preprocessing pipeline from evaluate.py:
    preprocess_observation → LiberoProcessorStep → rename → smolvla_preprocessor
    """
    obs_t = preprocess_observation(obs)

    # Add batch dim to robot_state sub-dict tensors (preprocess_observation skips this)
    rs_key = "observation.robot_state"
    if rs_key in obs_t:
        def _add_batch(d):
            return {k: (_add_batch(v) if isinstance(v, dict) else v.unsqueeze(0)) for k, v in d.items()}
        obs_t[rs_key] = _add_batch(obs_t[rs_key])

    # SmolVLANewLineProcessor appends '\n' to task string inside preprocessor
    obs_t["task"] = task_language

    # LiberoProcessorStep: flip images 180°, build 8-dim observation.state
    obs_t = env_preprocessor(obs_t)

    # Apply training rename_map (image → camera1, image2 → camera2)
    if "observation.images.image" in obs_t:
        obs_t["observation.images.camera1"] = obs_t.pop("observation.images.image")
    if "observation.images.image2" in obs_t:
        obs_t["observation.images.camera2"] = obs_t.pop("observation.images.image2")

    # SmolVLA preprocessor: batch dim, tokenize language, normalize state
    obs_t = preprocessor(obs_t)
    return obs_t


def build_optimizer(policy, cfg):
    vlm_params = list(policy.model.vlm_with_expert.vlm.parameters())
    head_params = list(policy.model.vlm_with_expert.lm_expert.parameters())
    return torch.optim.AdamW([
        {"params": vlm_params, "lr": cfg["learning_rate_backbone"]},
        {"params": head_params, "lr": cfg["learning_rate_head"]},
    ])


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading policy from {cfg['model_id']} ...")
    policy = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device)
    policy_ref = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device)
    policy_ref.requires_grad_(False)
    policy.train()

    # Build lerobot preprocessing pipeline (identical to evaluate.py)
    dataset_stats = load_dataset_stats(cfg["model_id"])
    policy.config.device = device
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset_stats
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    optimizer = build_optimizer(policy, cfg)

    bd = benchmark.get_benchmark_dict()
    suite = bd[cfg["task_suite"]]()
    n_tasks = suite.get_num_tasks()

    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"Starting GRPO training for {cfg['total_steps']} steps ...")
    for step in range(cfg["total_steps"]):
        task_id = random.randint(0, n_tasks - 1)

        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=cfg["task_suite"],
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
        )

        try:
            obs, info = env.reset()
            task_language = env.task_description
        except Exception as e:
            print(f"  [step {step}] env reset failed: {e} — skipping")
            env.close()
            continue

        obs_batch = preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device)

        # Sample group of n trajectories (prefix KV-cache computed once for all n)
        group_data = sample_group_trajectories(policy, obs_batch, n_group=cfg["n_group"])

        # Compute weighted rewards from the shared initial sim state
        raw_env = env._env  # OffScreenRenderEnv — needed for sim state save/restore
        rewards = []
        for traj in group_data:
            r = compute_weighted_reward(raw_env, traj, postprocessor)
            rewards.append(r)

        advantages = compute_grpo_advantages(rewards)

        loss = grpo_update(
            policy, policy_ref, optimizer, obs_batch, group_data, advantages,
            clip_eps=cfg["clip_eps"], kl_coeff=cfg["kl_coeff"],
        )

        mean_r = sum(rewards) / len(rewards)
        print(f"Step {step:5d} | loss={loss:.4f} | mean_reward={mean_r:.3f} | task={task_language[:40]}")

        if step > 0 and step % cfg["eval_every"] == 0:
            _quick_eval(policy, suite, n_tasks, cfg, env_preprocessor, preprocessor, postprocessor, device)

        if step > 0 and step % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["output_dir"], f"step_{step}")
            policy.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        env.close()

    final_path = os.path.join(cfg["output_dir"], f"step_{cfg['total_steps']}")
    policy.save_pretrained(final_path)
    print(f"Training complete. Final checkpoint → {final_path}")


def _quick_eval(policy, suite, n_tasks, cfg, env_preprocessor, preprocessor, postprocessor, device):
    """Quick eval on 3 random tasks, 5 episodes each, using the same pipeline as evaluate.py."""
    policy.eval()
    n_sample = min(3, n_tasks)
    task_ids = random.sample(range(n_tasks), n_sample)
    results = []
    for tid in task_ids:
        env = LiberoEnv(
            task_suite=suite,
            task_id=tid,
            task_suite_name=cfg["task_suite"],
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
        )
        n_ep = cfg.get("n_eval_episodes", 5)
        success_count = 0
        for _ in range(n_ep):
            policy.reset()
            obs, info = env.reset()
            task_language = env.task_description
            done = False
            for _step in range(env._max_episode_steps):
                obs_t = preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device)
                with torch.inference_mode():
                    action = policy.select_action(obs_t)
                action = postprocessor(action)
                act_np = action.squeeze(0).cpu().numpy()
                obs, _, terminated, truncated, info = env.step(act_np)
                done = terminated or truncated
                if done:
                    if info.get("is_success", False):
                        success_count += 1
                    break
        results.append(success_count / n_ep)
        env.close()
    mean_sr = sum(results) / len(results)
    print(f"  [Quick eval] mean success rate = {mean_sr:.3f}")
    policy.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
