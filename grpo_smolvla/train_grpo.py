"""Main GRPO training loop for SmolVLA on LIBERO."""

import argparse
import os
import random

# Must be set before the CUDA allocator initialises (before first .to("cuda"))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Cap BLAS thread counts BEFORE numpy / scipy / torch are imported, otherwise
# each library spawns one thread per CPU core (often 64). With n_group worker
# processes, that hits the container's process/thread limit and crashes with
# `pthread_create failed: Resource temporarily unavailable`.
# With fork mode these env vars also propagate to worker processes.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import datetime
import multiprocessing as _mp
from queue import Empty as _QueueEmpty

import torch
torch.set_num_threads(1)
import torch.distributed as dist
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
from grpo_smolvla.parallel_rollout import ParallelEnvPool, parallel_compute_rewards
from grpo_smolvla.rewards import compute_episode_reward


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


def cast_batch(batch, dtype):
    """Cast all floating-point tensors in a nested dict to dtype."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = cast_batch(v, dtype)
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


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

    # === STEP 1: spawn worker pool BEFORE touching CUDA ===
    # With start_method="fork", workers inherit main's already-imported modules
    # (libero, lerobot, transformers) without re-importing, so spawn is near-instant.
    # But fork after CUDA init causes worker CUDA-context corruption — so we must
    # create the pool BEFORE policy.to("cuda").
    use_parallel = cfg.get("use_parallel_rollout", False)
    pool = None
    if use_parallel:
        print(f"Spawning ParallelEnvPool with {cfg['n_group']} workers (fork mode) ...")
        pool = ParallelEnvPool(cfg["n_group"], cfg["task_suite"], start_method="fork")

    # === STEP 2: load policy and move to GPU (initializes CUDA in main only) ===
    print(f"Loading policy from {cfg['model_id']} ...")
    policy = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device=device, dtype=torch.bfloat16)
    policy_ref = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device=device, dtype=torch.bfloat16)
    policy_ref.requires_grad_(False)

    # Gradient checkpointing on the VLM backbone recomputes activations during
    # backward instead of storing them, trading ~2x compute for ~60% less activation
    # memory. Only needed on the trainable policy (policy_ref never calls backward).
    vlm = policy.model.vlm_with_expert.vlm
    if hasattr(vlm, "gradient_checkpointing_enable"):
        vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

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

    # Multi-denoise reward: each member is rolled out once per denoise level and
    # combined as a weighted sum. denoise_levels and weights come from config.
    denoising_weights = {int(k): float(v) for k, v in cfg["denoising_weights"].items()}
    denoise_levels = sorted(denoising_weights.keys())
    primary_denoise = max(denoise_levels)
    print(f"Multi-denoise reward: levels={denoise_levels}, weights={denoising_weights}, "
          f"primary (GRPO log-prob) level={primary_denoise}")

    print(f"Starting GRPO training for {cfg['total_steps']} steps ...")
    for step in range(cfg["total_steps"]):
        task_id = random.randint(0, n_tasks - 1)

        # Train/eval init-state split: training only sees init states
        # [0..n_train_init_states-1]; eval uses [n_train_init_states..49]
        # so success-rate gains measure held-out generalisation.
        n_train_inits = cfg.get("n_train_init_states", 40)
        episode_start = (step * cfg["n_group"]) % n_train_inits

        model_dtype = next(policy.parameters()).dtype

        def preprocess_obs_fn(obs, lang):
            return cast_batch(
                preprocess_obs(obs, lang, env_preprocessor, preprocessor, device),
                model_dtype,
            )

        env = None
        if pool is not None:
            # --- Parallel rollout path ---
            episode_indices = [(episode_start + i) % n_train_inits for i in range(cfg["n_group"])]
            try:
                obs_list, task_language, max_steps = pool.reset_to_task(task_id, episode_indices)
            except Exception as e:
                print(f"  [step {step}] pool reset failed (task_id={task_id}): {e} — skipping step")
                torch.cuda.empty_cache()
                continue

            # Cap training-time rollout horizon (eval path is unaffected).
            if cfg.get("max_episode_steps_train") is not None:
                max_steps = min(max_steps, int(cfg["max_episode_steps_train"]))

            group_episodes = []
            for obs in obs_list:
                obs_batch_i = preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device)
                obs_batch_i = cast_batch(obs_batch_i, model_dtype)
                traj_i = sample_group_trajectories(
                    policy, obs_batch_i, denoise_levels=denoise_levels, n_group=1,
                )[0]
                group_episodes.append({"obs_batch": obs_batch_i, "traj": traj_i})

            # Snapshot every worker's fresh init state — we restore to this before
            # each per-denoise-level rollout so all denoise levels start identically.
            try:
                saved_states = pool.save_states()
            except Exception as e:
                print(f"  [step {step}] pool save_states failed: {e} — skipping step")
                torch.cuda.empty_cache()
                continue

            # Roll out each denoise level in turn → per-level binary rewards.
            per_level_rewards = {}
            for d in denoise_levels:
                try:
                    pool.restore_states(saved_states)
                except Exception as e:
                    print(f"  [step {step}] pool restore_states (d={d}) failed: {e} — skipping step")
                    per_level_rewards = None
                    break
                chunks_np = [
                    postprocessor(ep["traj"]["actions"][d]).squeeze(0).cpu().float().numpy()
                    for ep in group_episodes
                ]
                per_level_rewards[d] = parallel_compute_rewards(
                    pool, policy, chunks_np, task_language,
                    preprocess_obs_fn, postprocessor, max_steps,
                )

            if per_level_rewards is None:
                torch.cuda.empty_cache()
                continue

            rewards = [
                sum(denoising_weights[d] * per_level_rewards[d][i] for d in denoise_levels)
                for i in range(cfg["n_group"])
            ]
        else:
            # --- Serial rollout path (original) ---
            try:
                env = LiberoEnv(
                    task_suite=suite,
                    task_id=task_id,
                    task_suite_name=cfg["task_suite"],
                    obs_type="pixels_agent_pos",
                    observation_height=256,
                    observation_width=256,
                    episode_index=episode_start,
                )
            except Exception as e:
                print(f"  [step {step}] env construction failed (task_id={task_id}): {e} — skipping step")
                torch.cuda.empty_cache()
                continue

            # Cap training-time rollout horizon (eval path is unaffected).
            if cfg.get("max_episode_steps_train") is not None:
                env._max_episode_steps = min(
                    env._max_episode_steps, int(cfg["max_episode_steps_train"])
                )

            group_episodes = []
            raw_env = env._env
            skip_step = False
            for _ in range(cfg["n_group"]):
                try:
                    obs, info = env.reset()
                    task_language = env.task_description
                except Exception as e:
                    print(f"  [step {step}] env reset failed: {e} — skipping step")
                    skip_step = True
                    break

                obs_batch_i = preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device)
                obs_batch_i = cast_batch(obs_batch_i, model_dtype)
                traj_i = sample_group_trajectories(
                    policy, obs_batch_i, denoise_levels=denoise_levels, n_group=1,
                )[0]

                group_episodes.append({
                    "obs_batch": obs_batch_i,
                    "traj": traj_i,
                    "sim_state": raw_env.sim.get_state(),
                    "saved_timestep": raw_env.env.timestep,
                    "task_language": task_language,
                })

            if skip_step:
                env.close()
                torch.cuda.empty_cache()
                continue

            # compute_episode_reward restores sim_state on every call, so calling
            # it len(denoise_levels) times per member is safe — each rollout starts
            # from the same fresh init state.
            rewards = []
            for ep in group_episodes:
                weighted = 0.0
                for d in denoise_levels:
                    r = compute_episode_reward(
                        policy, ep["traj"]["actions"][d], postprocessor,
                        preprocess_obs_fn, env, ep["task_language"],
                        raw_env, ep["sim_state"], ep["saved_timestep"],
                        env._max_episode_steps,
                    )
                    weighted += denoising_weights[d] * r
                rewards.append(weighted)

        advantages = compute_grpo_advantages(rewards)

        mean_r = sum(rewards) / len(rewards)
        rewards_str = "[" + ",".join(f"{r:.1f}" for r in rewards) + "]"

        # Skip update when all rewards are identical — zero advantage means no
        # learning signal, and the KL-only gradient would cause policy drift.
        if len(set(rewards)) == 1:
            print(f"Step {step:5d} | skip (uniform rewards={rewards_str}) | task={task_language[:40]}")
            if env is not None:
                env.close()
            torch.cuda.empty_cache()
            continue

        loss = grpo_update(
            policy, policy_ref, optimizer, group_episodes, advantages,
            clip_eps=cfg["clip_eps"], kl_coeff=cfg["kl_coeff"],
        )

        print(f"Step {step:5d} | loss={loss:.4f} | mean_reward={mean_r:.3f} | rewards={rewards_str} | task={task_language[:40]}")

        if step > 0 and step % cfg["eval_every"] == 0:
            _quick_eval(policy, suite, n_tasks, cfg, env_preprocessor, preprocessor, postprocessor, device)

        if step > 0 and step % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["output_dir"], f"step_{step}")
            policy.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        if env is not None:
            env.close()
        torch.cuda.empty_cache()

    if pool is not None:
        print("Shutting down ParallelEnvPool ...")
        pool.close()

    final_path = os.path.join(cfg["output_dir"], f"step_{cfg['total_steps']}")
    policy.save_pretrained(final_path)
    print(f"Training complete. Final checkpoint → {final_path}")


def _quick_eval(policy, suite, n_tasks, cfg, env_preprocessor, preprocessor, postprocessor, device):
    """Quick eval on 3 random tasks, n_eval_episodes each, on the held-out init states."""
    policy.eval()
    n_sample = min(3, n_tasks)
    task_ids = random.sample(range(n_tasks), n_sample)
    results = []
    n_train_inits = cfg.get("n_train_init_states", 40)
    for tid in task_ids:
        env = LiberoEnv(
            task_suite=suite,
            task_id=tid,
            task_suite_name=cfg["task_suite"],
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
            episode_index=n_train_inits,
        )
        n_ep = min(cfg.get("n_eval_episodes", 5), 50 - n_train_inits)
        success_count = 0
        for _ in range(n_ep):
            policy.reset()
            obs, info = env.reset()
            task_language = env.task_description
            done = False
            for _step in range(env._max_episode_steps):
                obs_t = preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device)
                obs_t = cast_batch(obs_t, next(policy.parameters()).dtype)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    action = policy.select_action(obs_t)
                action = postprocessor(action)
                act_np = action.squeeze(0).cpu().float().numpy()
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


def train_multi_gpu(cfg):
    """Multi-GPU GRPO: trainer on GPU 0, one rollout worker per denoise level on GPUs 1-N.

    Rollout workers run parallel_compute_rewards simultaneously — env sim across all
    three denoise levels is fully parallel. Weight sync uses NCCL dist.broadcast after
    each grpo_update (skipped when no gradient was computed).
    """
    device = "cuda:0"
    master_addr = cfg.get("dist_master_addr", "127.0.0.1")
    master_port = str(cfg.get("dist_master_port", 29500))

    denoise_levels = sorted(int(k) for k in cfg["denoising_weights"])
    denoising_weights = {int(k): float(v) for k, v in cfg["denoising_weights"].items()}
    primary_level = max(denoise_levels)
    n_workers = len(denoise_levels)
    world_size = n_workers + 1  # trainer + one worker per level

    ctx = _mp.get_context("spawn")
    send_queues = [ctx.Queue() for _ in range(n_workers)]
    recv_queues = [ctx.Queue() for _ in range(n_workers)]

    from grpo_smolvla.rollout_worker import worker_main
    worker_procs = []
    for i, level in enumerate(denoise_levels):
        rank = i + 1
        is_primary = (level == primary_level)
        p = ctx.Process(
            target=worker_main,
            args=(rank, world_size, level, is_primary, cfg,
                  send_queues[i], recv_queues[i], master_addr, master_port),
        )
        p.start()
        worker_procs.append(p)

    print("Loading trainer policy ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device=device, dtype=torch.bfloat16)
    policy_ref = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device=device, dtype=torch.bfloat16)
    policy_ref.requires_grad_(False)

    vlm = policy.model.vlm_with_expert.vlm
    if hasattr(vlm, "gradient_checkpointing_enable"):
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()

    dataset_stats = load_dataset_stats(cfg["model_id"])
    policy.config.device = device
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset_stats
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    optimizer = build_optimizer(policy, cfg)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=10),
    )
    try:
        for i, rq in enumerate(recv_queues):
            msg = rq.get(timeout=300)
            if msg[0] != "ready":
                raise RuntimeError(f"Worker {i} (level={denoise_levels[i]}) failed startup: {msg}")
        print("All workers ready. Starting training ...", flush=True)

        bd = benchmark.get_benchmark_dict()
        suite = bd[cfg["task_suite"]]()
        n_tasks = suite.get_num_tasks()
        n_train_inits = cfg.get("n_train_init_states", 40)
        model_dtype = next(policy.parameters()).dtype
        os.makedirs(cfg["output_dir"], exist_ok=True)

        for step in range(cfg["total_steps"]):
            task_id = random.randint(0, n_tasks - 1)
            episode_start = (step * cfg["n_group"]) % n_train_inits
            episode_indices = [(episode_start + i) % n_train_inits for i in range(cfg["n_group"])]
            max_steps_cap = int(cfg.get("max_episode_steps_train", 520))

            # ── Phase 1: reset all workers ───────────────────────────────────────
            for sq in send_queues:
                sq.put(("reset", task_id, episode_indices, max_steps_cap))

            obs_list = None
            task_language = None
            max_steps = max_steps_cap
            reset_ok = True
            for i, rq in enumerate(recv_queues):
                try:
                    msg = rq.get(timeout=120)
                except _QueueEmpty:
                    print(f"  [step {step}] Worker {i} timed out during reset (crashed?) — aborting", flush=True)
                    reset_ok = False
                    for drain_rq in recv_queues[i + 1:]:
                        try:
                            drain_rq.get(timeout=5)
                        except _QueueEmpty:
                            pass
                    break
                if msg[0] == "error":
                    print(f"  [step {step}] Worker {i} reset error: {msg[1]} — skipping step")
                    reset_ok = False
                    for drain_rq in recv_queues[i + 1:]:
                        try:
                            drain_rq.get(timeout=120)
                        except _QueueEmpty:
                            pass
                    break
                if msg[0] == "obs":
                    _, obs_list, task_language, env_max = msg
                    max_steps = min(max_steps_cap, env_max)

            if not reset_ok or obs_list is None:
                if obs_list is None and reset_ok:
                    print(f"  [step {step}] Primary worker failed to return obs — skipping step")
                continue

            # ── Phase 2: sample trajectories on trainer GPU ──────────────────────
            group_episodes = []
            for obs in obs_list:
                obs_batch_i = preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device)
                obs_batch_i = cast_batch(obs_batch_i, model_dtype)
                traj_i = sample_group_trajectories(
                    policy, obs_batch_i, denoise_levels=denoise_levels, n_group=1,
                )[0]
                group_episodes.append({"obs_batch": obs_batch_i, "traj": traj_i})

            # ── Phase 3: send first chunks to each worker ────────────────────────
            for i, level in enumerate(denoise_levels):
                first_chunks = [
                    postprocessor(ep["traj"]["actions"][level]).squeeze(0).cpu().float().numpy()
                    for ep in group_episodes
                ]
                send_queues[i].put(("rollout", first_chunks, task_language, max_steps))

            # ── Phase 4: collect rewards (workers ran in parallel) ───────────────
            per_level_rewards = {}
            rollout_ok = True
            for i, (level, rq) in enumerate(zip(denoise_levels, recv_queues)):
                try:
                    msg = rq.get(timeout=300)
                except _QueueEmpty:
                    print(f"  [step {step}] Worker {i} timed out during rollout (crashed?) — aborting", flush=True)
                    rollout_ok = False
                    for drain_rq in recv_queues[i + 1:]:
                        try:
                            drain_rq.get(timeout=5)
                        except _QueueEmpty:
                            pass
                    break
                if msg[0] == "error":
                    print(f"  [step {step}] Worker {i} rollout error: {msg[1]} — skipping step")
                    rollout_ok = False
                    for drain_rq in recv_queues[i + 1:]:
                        try:
                            drain_rq.get(timeout=300)
                        except _QueueEmpty:
                            pass
                    break
                per_level_rewards[level] = msg[1]

            if not rollout_ok:
                continue

            rewards = [
                sum(denoising_weights[d] * per_level_rewards[d][i] for d in denoise_levels)
                for i in range(cfg["n_group"])
            ]
            advantages = compute_grpo_advantages(rewards)
            rewards_str = "[" + ",".join(f"{r:.1f}" for r in rewards) + "]"

            if len(set(rewards)) == 1:
                print(f"Step {step:5d} | skip (uniform rewards={rewards_str}) | task={task_language[:40]}")
                continue

            # ── Phase 5: gradient update ─────────────────────────────────────────
            loss = grpo_update(
                policy, policy_ref, optimizer, group_episodes, advantages,
                clip_eps=cfg["clip_eps"], kl_coeff=cfg["kl_coeff"],
            )
            mean_r = sum(rewards) / len(rewards)
            print(f"Step {step:5d} | loss={loss:.4f} | mean_reward={mean_r:.3f} "
                  f"| rewards={rewards_str} | task={task_language[:40]}")

            # ── Phase 6: broadcast updated weights ───────────────────────────────
            for sq in send_queues:
                sq.put(("sync_weights",))
            for param in policy.parameters():
                dist.broadcast(param.data, src=0)
            for rq in recv_queues:
                sync_resp = rq.get(timeout=60)
                if sync_resp[0] != "synced":
                    print(f"  [warning] Unexpected sync response: {sync_resp}", flush=True)

            if step > 0 and step % cfg["eval_every"] == 0:
                _quick_eval(policy, suite, n_tasks, cfg, env_preprocessor,
                            preprocessor, postprocessor, device)

            if step > 0 and step % cfg["save_every"] == 0:
                ckpt_path = os.path.join(cfg["output_dir"], f"step_{step}")
                policy.save_pretrained(ckpt_path)
                print(f"  Checkpoint saved → {ckpt_path}")

        final_path = os.path.join(cfg["output_dir"], f"step_{cfg['total_steps']}")
        policy.save_pretrained(final_path)
        print(f"Training complete. Final checkpoint → {final_path}")
    finally:
        for sq in send_queues:
            try:
                sq.put(("stop",))
            except Exception:
                pass
        for p in worker_procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("use_multi_gpu_rollout", False):
        train_multi_gpu(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
