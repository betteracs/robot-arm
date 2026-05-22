"""Rollout-only training loop: runs n steps of group rollouts but NEVER calls
optimizer.step(). If rewards are non-zero across most steps, the rollout works
fine and the bug is in the gradient update.
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
import torch
import yaml
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep

from grpo_smolvla.flow_utils import sample_group_trajectories
from grpo_smolvla.rewards import compute_episode_reward
from grpo_smolvla.train_grpo import load_dataset_stats, preprocess_obs, cast_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--n_steps", type=int, default=15)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading policy from {cfg['model_id']} ...")
    policy = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device=device, dtype=torch.bfloat16)
    policy.eval()  # rollout-only, no training mode
    policy.config.device = device

    dataset_stats = load_dataset_stats(cfg["model_id"])
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset_stats
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    bd = benchmark.get_benchmark_dict()
    suite = bd[cfg["task_suite"]]()
    n_tasks = suite.get_num_tasks()
    n_train_inits = cfg.get("n_train_init_states", 40)

    print(f"Running {args.n_steps} rollout-only steps (NO gradient updates) ...")
    for step in range(args.n_steps):
        task_id = random.randint(0, n_tasks - 1)
        episode_start = (step * cfg["n_group"]) % n_train_inits
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=cfg["task_suite"],
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
            episode_index=episode_start,
        )

        model_dtype = next(policy.parameters()).dtype

        def preprocess_obs_fn(obs, lang):
            return cast_batch(
                preprocess_obs(obs, lang, env_preprocessor, preprocessor, device),
                model_dtype,
            )

        group_episodes = []
        raw_env = env._env
        for _ in range(cfg["n_group"]):
            obs, info = env.reset()
            task_language = env.task_description
            obs_batch_i = preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device)
            obs_batch_i = cast_batch(obs_batch_i, model_dtype)
            traj_i = sample_group_trajectories(policy, obs_batch_i, n_group=1)[0]
            group_episodes.append({
                "obs_batch": obs_batch_i,
                "traj": traj_i,
                "sim_state": raw_env.sim.get_state(),
                "saved_timestep": raw_env.env.timestep,
                "task_language": task_language,
            })

        rewards = []
        for ep in group_episodes:
            r = compute_episode_reward(
                policy, ep["traj"]["actions_10"], postprocessor,
                preprocess_obs_fn, env, ep["task_language"],
                raw_env, ep["sim_state"], ep["saved_timestep"],
                env._max_episode_steps,
            )
            rewards.append(r)

        mean_r = sum(rewards) / len(rewards)
        rewards_str = "[" + ",".join(f"{r:.1f}" for r in rewards) + "]"
        print(f"Step {step:3d} | mean_reward={mean_r:.3f} | rewards={rewards_str} | task={task_language[:40]}")

        env.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
