"""LIBERO benchmark evaluation for SmolVLA using the official lerobot pipeline.

Uses LiberoEnv + LiberoProcessorStep + SmolVLA pre/post-processors to exactly
replicate the lerobot-eval inference path, fixing the 0% success rate caused by
manual obs construction bugs.
"""

import argparse

import torch
from huggingface_hub import hf_hub_download
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
from safetensors.torch import load_file


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


def evaluate_policy(policy, preprocessor, postprocessor, task_suite_name,
                    n_episodes, device, n_tasks=None):
    """Evaluate policy on all tasks in a suite. Returns {task_language: success_rate}."""
    policy.eval()
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    bd = benchmark.get_benchmark_dict()
    suite = bd[task_suite_name]()
    n_tasks_total = suite.get_num_tasks()
    n_tasks = n_tasks_total if n_tasks is None else min(n_tasks, n_tasks_total)

    results = {}
    for task_id in range(n_tasks):
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=task_suite_name,
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
        )
        task_language = env.task_description
        max_steps = env._max_episode_steps

        successes = 0
        for ep in range(n_episodes):
            policy.reset()
            obs, info = env.reset()

            done = False
            for _ in range(max_steps):
                # Convert numpy obs to tensors (channel-first float [0,1])
                obs_t = preprocess_observation(obs)
                # Add batch dim to robot_state tensors (preprocess_observation skips this for non-vectorized envs)
                rs_key = "observation.robot_state"
                if rs_key in obs_t:
                    def _add_batch(d):
                        return {k: (_add_batch(v) if isinstance(v, dict) else v.unsqueeze(0)) for k, v in d.items()}
                    obs_t[rs_key] = _add_batch(obs_t[rs_key])
                # Add task string (SmolVLANewLineProcessor appends \n inside preprocessor)
                obs_t["task"] = task_language

                # env_preprocessor: flip images + build observation.state from robot_state
                obs_t = env_preprocessor(obs_t)

                # Apply training rename_map (image→camera1, image2→camera2).
                # make_smolvla_pre_post_processors sets rename_map={} by default;
                # the trained checkpoint used this mapping during SFT.
                if "observation.images.image" in obs_t:
                    obs_t["observation.images.camera1"] = obs_t.pop("observation.images.image")
                if "observation.images.image2" in obs_t:
                    obs_t["observation.images.camera2"] = obs_t.pop("observation.images.image2")

                # SmolVLA preprocessor: batch dim, tokenize, normalize state
                obs_t = preprocessor(obs_t)

                with torch.inference_mode():
                    action = policy.select_action(obs_t)

                # postprocessor: unnormalize action
                action = postprocessor(action)
                act_np = action.squeeze(0).cpu().float().numpy()

                obs, reward, terminated, truncated, info = env.step(act_np)
                done = terminated or truncated
                if done:
                    if info.get("is_success", False):
                        successes += 1
                    break

        sr = successes / n_episodes
        results[task_language] = sr
        print(f"  [{task_suite_name}] task {task_id:2d}: {sr:.2%}  {task_language[:60]}")
        env.close()

    mean_sr = sum(results.values()) / len(results)
    print(f"  [{task_suite_name}] Mean success rate: {mean_sr:.2%}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on LIBERO benchmarks")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_libero")
    parser.add_argument("--suites", nargs="+",
                        default=["libero_10", "libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--n_tasks", type=int, default=None,
                        help="Max tasks per suite (default: all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device)

    dataset_stats = load_dataset_stats(args.checkpoint)
    policy.config.device = device
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset_stats
    )

    all_results = {}
    for suite_name in args.suites:
        print(f"\n=== Evaluating on {suite_name} ===")
        suite_results = evaluate_policy(
            policy, preprocessor, postprocessor,
            suite_name, args.n_episodes, device, args.n_tasks,
        )
        all_results[suite_name] = suite_results

    print("\n=== Summary ===")
    print(f"{'Suite':<20} {'Mean Success Rate':>18}")
    print("-" * 40)
    for suite_name, results in all_results.items():
        mean_sr = sum(results.values()) / len(results)
        print(f"{suite_name:<20} {mean_sr:>17.2%}")


if __name__ == "__main__":
    main()
