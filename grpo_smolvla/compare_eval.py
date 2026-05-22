"""Compare pretrained vs finetuned SmolVLA on LIBERO-10.

Usage:
    python -m grpo_smolvla.compare_eval
    python -m grpo_smolvla.compare_eval --checkpoint lerobot/smolvla_libero \
        --finetuned checkpoints/grpo_smolvla/step_5000 \
        --n_episodes 20 --n_tasks 10
"""

import argparse
import os

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


def load_policy(checkpoint_id, local_weights_path, device):
    """Load policy from HF checkpoint, optionally overriding weights from a local path."""
    policy = SmolVLAPolicy.from_pretrained(checkpoint_id).to(device)
    if local_weights_path:
        weights_file = os.path.join(local_weights_path, "model.safetensors")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"No model.safetensors found at {weights_file}")
        local_weights = load_file(weights_file, device=str(device))
        missing, unexpected = policy.load_state_dict(local_weights, strict=False)
        if missing:
            print(f"  [warn] {len(missing)} missing keys when loading finetuned weights")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys when loading finetuned weights")
    return policy


def evaluate_policy(policy, preprocessor, postprocessor, suite, suite_name,
                    n_episodes, device, n_tasks=None, init_start=40):
    policy.eval()
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    n_tasks_total = suite.get_num_tasks()
    n_tasks = n_tasks_total if n_tasks is None else min(n_tasks, n_tasks_total)

    results = {}
    for task_id in range(n_tasks):
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
            episode_index=init_start,
        )
        task_language = env.task_description
        max_steps = env._max_episode_steps

        successes = 0
        for _ in range(n_episodes):
            policy.reset()
            obs, _ = env.reset()
            for _ in range(max_steps):
                obs_t = preprocess_observation(obs)
                rs_key = "observation.robot_state"
                if rs_key in obs_t:
                    def _add_batch(d):
                        return {k: (_add_batch(v) if isinstance(v, dict) else v.unsqueeze(0))
                                for k, v in d.items()}
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
                act_np = action.squeeze(0).cpu().float().numpy()
                obs, _, terminated, truncated, info = env.step(act_np)
                if terminated or truncated:
                    if info.get("is_success", False):
                        successes += 1
                    break

        sr = successes / n_episodes
        results[task_id] = (sr, task_language)
        env.close()

    return results


def print_comparison(pretrained_results, finetuned_results, label_pt, label_ft):
    task_ids = sorted(pretrained_results.keys())
    col = 60

    print(f"\n{'Task':>4}  {'Description':<{col}}  {label_pt:>12}  {label_ft:>12}  {'Delta':>8}")
    print("-" * (4 + 2 + col + 2 + 12 + 2 + 12 + 2 + 8))

    total_pt, total_ft = 0.0, 0.0
    for tid in task_ids:
        sr_pt, lang = pretrained_results[tid]
        sr_ft, _ = finetuned_results[tid]
        delta = sr_ft - sr_pt
        arrow = "+" if delta > 0 else ""
        print(f"{tid:>4}  {lang[:col]:<{col}}  {sr_pt:>11.1%}  {sr_ft:>11.1%}  {arrow}{delta:>+7.1%}")
        total_pt += sr_pt
        total_ft += sr_ft

    n = len(task_ids)
    mean_pt = total_pt / n
    mean_ft = total_ft / n
    delta_mean = mean_ft - mean_pt
    print("-" * (4 + 2 + col + 2 + 12 + 2 + 12 + 2 + 8))
    print(f"{'Mean':>4}  {'':^{col}}  {mean_pt:>11.1%}  {mean_ft:>11.1%}  {delta_mean:>+8.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lerobot/smolvla_libero",
                        help="HF checkpoint ID for pretrained model")
    parser.add_argument("--finetuned", default="checkpoints/grpo_smolvla/step_5000",
                        help="Local path to finetuned checkpoint directory")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of episodes per task (held-out init states)")
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--init_start", type=int, default=40,
                        help="First held-out init state index (Option A: train on [0..init_start-1])")
    parser.add_argument("--skip_pretrained", action="store_true",
                        help="Skip pretrained eval (use saved baseline results)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading dataset stats from {args.checkpoint} ...")
    dataset_stats = load_dataset_stats(args.checkpoint)

    bd = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()

    pretrained_results = None
    if not args.skip_pretrained:
        print(f"\n=== Evaluating PRETRAINED: {args.checkpoint} ===")
        policy_pt = load_policy(args.checkpoint, None, device)
        policy_pt.config.device = device
        preprocessor_pt, postprocessor_pt = make_smolvla_pre_post_processors(
            policy_pt.config, dataset_stats=dataset_stats
        )
        pretrained_results = evaluate_policy(
            policy_pt, preprocessor_pt, postprocessor_pt,
            suite, args.suite, args.n_episodes, device, args.n_tasks,
            init_start=args.init_start,
        )
        del policy_pt
        torch.cuda.empty_cache()
    else:
        # Baseline provided by user (libero_10, 20 episodes)
        baseline = [
            (0, 0.20, "put both the alphabet soup and the tomato sauce in the basket"),
            (1, 0.60, "put both the cream cheese box and the butter in the basket"),
            (2, 0.85, "turn on the stove and put the moka pot on it"),
            (3, 0.95, "put the black bowl in the bottom drawer of the cabinet and close it"),
            (4, 0.25, "put the white mug on the left plate and put the yellow and white mug..."),
            (5, 0.70, "pick up the book and place it in the back compartment of the caddy"),
            (6, 0.30, "put the white mug on the plate and put the chocolate pudding..."),
            (7, 0.40, "put both the alphabet soup and the cream cheese box in the basket"),
            (8, 0.35, "put both moka pots on the stove"),
            (9, 0.55, "put the yellow and white mug in the microwave and close it"),
        ]
        pretrained_results = {tid: (sr, lang) for tid, sr, lang in baseline}
        print("Using provided baseline results for pretrained model.")
        print("  [warn] Baseline was measured on init states [0..19] (full range),")
        print(f"         but finetuned eval uses held-out init states [{args.init_start}..49].")
        print("         Re-run pretrained eval (drop --skip_pretrained) for a fair comparison.")

    print(f"\n=== Evaluating FINETUNED: {args.finetuned} ===")
    policy_ft = load_policy(args.checkpoint, args.finetuned, device)
    policy_ft.config.device = device
    preprocessor_ft, postprocessor_ft = make_smolvla_pre_post_processors(
        policy_ft.config, dataset_stats=dataset_stats
    )
    finetuned_results = evaluate_policy(
        policy_ft, preprocessor_ft, postprocessor_ft,
        suite, args.suite, args.n_episodes, device, args.n_tasks,
        init_start=args.init_start,
    )
    del policy_ft
    torch.cuda.empty_cache()

    label_pt = "Pretrained" if not args.skip_pretrained else "Baseline"
    label_ft = f"FT step_{args.finetuned.split('step_')[-1]}" if "step_" in args.finetuned else "Finetuned"
    print_comparison(pretrained_results, finetuned_results, label_pt, label_ft)


if __name__ == "__main__":
    main()
