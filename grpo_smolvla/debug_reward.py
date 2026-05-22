"""Debug: compare compute_episode_reward (training-time) vs simple closed-loop eval.

Run on the pretrained model on a high-baseline task (e.g. task_id=2, "turn on the
stove...") across several init states. If the eval-style rollout succeeds but
compute_episode_reward returns 0, the training-time rollout has a bug.
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import torch
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep

from grpo_smolvla.flow_utils import sample_group_trajectories
from grpo_smolvla.rewards import compute_episode_reward
from grpo_smolvla.train_grpo import load_dataset_stats, preprocess_obs, cast_batch


def simple_eval_rollout(policy, env, env_preprocessor, preprocessor, postprocessor, device):
    """Closed-loop eval — same as _quick_eval/compare_eval."""
    policy.eval()
    policy.reset()
    obs, _ = env.reset()
    task_lang = env.task_description
    for _ in range(env._max_episode_steps):
        obs_t = preprocess_obs(obs, task_lang, env_preprocessor, preprocessor, device)
        obs_t = cast_batch(obs_t, next(policy.parameters()).dtype)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(obs_t)
        act_np = postprocessor(action).squeeze(0).cpu().float().numpy()
        obs, _, terminated, truncated, info = env.step(act_np)
        if terminated or truncated:
            return 1.0 if info.get("is_success", False) else 0.0
    return 0.0


def training_style_rollout(policy, env, env_preprocessor, preprocessor, postprocessor, device):
    """Use compute_episode_reward exactly as training does."""
    policy.eval()
    policy.reset()
    obs, _ = env.reset()
    task_lang = env.task_description
    raw_env = env._env
    sim_state = raw_env.sim.get_state()
    saved_timestep = raw_env.env.timestep

    obs_batch = preprocess_obs(obs, task_lang, env_preprocessor, preprocessor, device)
    obs_batch = cast_batch(obs_batch, next(policy.parameters()).dtype)

    traj = sample_group_trajectories(policy, obs_batch, n_group=1)[0]
    model_dtype = next(policy.parameters()).dtype

    def preprocess_obs_fn(o, lang):
        return cast_batch(
            preprocess_obs(o, lang, env_preprocessor, preprocessor, device),
            model_dtype,
        )

    return compute_episode_reward(
        policy, traj["actions_10"], postprocessor,
        preprocess_obs_fn, env, task_lang,
        raw_env, sim_state, saved_timestep,
        env._max_episode_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=2,
                        help="LIBERO-10 task id (default 2 = stove + moka pot, 85% baseline)")
    parser.add_argument("--n_inits", type=int, default=4)
    parser.add_argument("--init_start", type=int, default=40)
    parser.add_argument("--suite", default="libero_10")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: {args.task_id} ({args.suite}), init range [{args.init_start}..{args.init_start + args.n_inits - 1}]")

    print("Loading pretrained policy ...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero").to(device=device, dtype=torch.bfloat16)
    policy.config.device = device

    dataset_stats = load_dataset_stats("lerobot/smolvla_libero")
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset_stats
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    bd = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()

    print("\n=== Eval-style closed-loop rollout (ground truth) ===")
    eval_results = []
    for i in range(args.n_inits):
        env = LiberoEnv(
            task_suite=suite,
            task_id=args.task_id,
            task_suite_name=args.suite,
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
            episode_index=args.init_start + i,
        )
        r = simple_eval_rollout(policy, env, env_preprocessor, preprocessor, postprocessor, device)
        eval_results.append(r)
        print(f"  init={args.init_start + i}: reward={r}")
        env.close()
        torch.cuda.empty_cache()

    print(f"\nEval-style: {sum(eval_results)}/{len(eval_results)} success")

    print("\n=== Training-style rollout (compute_episode_reward) ===")
    train_results = []
    for i in range(args.n_inits):
        env = LiberoEnv(
            task_suite=suite,
            task_id=args.task_id,
            task_suite_name=args.suite,
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
            episode_index=args.init_start + i,
        )
        r = training_style_rollout(policy, env, env_preprocessor, preprocessor, postprocessor, device)
        train_results.append(r)
        print(f"  init={args.init_start + i}: reward={r}")
        env.close()
        torch.cuda.empty_cache()

    print(f"\nTraining-style: {sum(train_results)}/{len(train_results)} success")

    print("\n=== Comparison ===")
    print(f"  Eval-style:     {eval_results}")
    print(f"  Training-style: {train_results}")
    if eval_results != train_results:
        print("  ⚠️  MISMATCH — compute_episode_reward likely has a bug")
    else:
        print("  ✓ Match — model behavior is identical in both rollouts")


if __name__ == "__main__":
    main()
