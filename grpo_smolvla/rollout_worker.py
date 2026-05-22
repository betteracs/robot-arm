"""Rollout worker process for multi-GPU GRPO training.

Each worker process:
  1. Creates a ParallelEnvPool via fork (BEFORE any CUDA init).
  2. Loads policy_inference (read-only SmolVLAPolicy) to its assigned GPU.
  3. Builds the preprocessing pipeline identical to train_grpo.
  4. Calls dist.init_process_group(backend="nccl") — collective with trainer.
  5. Sends ("ready",) to trainer via recv_q.
  6. Enters a command loop handling: reset, rollout, sync_weights, stop.
"""

import datetime
import os

# Must be set before any numpy/torch imports in this process.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
torch.set_num_threads(1)

import torch.distributed as dist
from huggingface_hub import hf_hub_download
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
from safetensors.torch import load_file

from grpo_smolvla.parallel_rollout import ParallelEnvPool, parallel_compute_rewards


# ---------------------------------------------------------------------------
# Private helper functions (mirrored from train_grpo, prefixed with _)
# ---------------------------------------------------------------------------

def _load_dataset_stats(checkpoint_id):
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


def _cast_batch(batch, dtype):
    """Cast all floating-point tensors in a nested dict to dtype."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = _cast_batch(v, dtype)
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


def _preprocess_obs(obs, task_language, env_preprocessor, preprocessor, device, dtype):
    """Convert LiberoEnv obs dict to SmolVLA policy input batch.

    Mirrors train_grpo.preprocess_obs exactly, then casts to model dtype.
    """
    obs_t = preprocess_observation(obs)

    # Add batch dim to robot_state sub-dict tensors (preprocess_observation skips this).
    rs_key = "observation.robot_state"
    if rs_key in obs_t:
        def _add_batch(d):
            return {k: (_add_batch(v) if isinstance(v, dict) else v.unsqueeze(0)) for k, v in d.items()}
        obs_t[rs_key] = _add_batch(obs_t[rs_key])

    obs_t["task"] = task_language

    # LiberoProcessorStep: flip images 180°, build 8-dim observation.state.
    obs_t = env_preprocessor(obs_t)

    # Apply training rename_map (image → camera1, image2 → camera2).
    if "observation.images.image" in obs_t:
        obs_t["observation.images.camera1"] = obs_t.pop("observation.images.image")
    if "observation.images.image2" in obs_t:
        obs_t["observation.images.camera2"] = obs_t.pop("observation.images.image2")

    # SmolVLA preprocessor: batch dim, tokenize language, normalize state.
    obs_t = preprocessor(obs_t)

    # Cast all floating-point tensors to model dtype.
    obs_t = _cast_batch(obs_t, dtype)
    return obs_t


def broadcast_weights_from_trainer(policy):
    """Broadcast every parameter from rank-0 (trainer) to this worker (collective)."""
    for param in policy.parameters():
        dist.broadcast(param.data, src=0)


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def worker_main(
    rank: int,
    world_size: int,
    denoise_level: int,
    is_primary: bool,
    cfg: dict,
    send_q,   # multiprocessing.Queue — worker reads commands from here
    recv_q,   # multiprocessing.Queue — worker writes responses here
    master_addr: str,
    master_port: str,
):
    """Entry point for a rollout worker process.

    Parameters
    ----------
    rank:         NCCL rank for this worker (trainer is rank 0).
    world_size:   Total NCCL world size (1 trainer + N workers).
    denoise_level: Which denoising level this worker handles.
    is_primary:   True for the highest denoise-level worker — it is the one
                  that sends obs/task info back to the trainer on reset.
    cfg:          Training config dict (same as passed to train_grpo.train).
    send_q:       Queue the trainer writes commands to; this worker reads from it.
    recv_q:       Queue this worker writes responses to; the trainer reads from it.
    master_addr:  NCCL master address (same as trainer).
    master_port:  NCCL master port (same as trainer).
    """

    # ------------------------------------------------------------------
    # STEP 1: Create ParallelEnvPool via fork — BEFORE any CUDA init.
    # ------------------------------------------------------------------
    # Force headless GL so the forked env workers survive on headless servers.
    os.environ.setdefault("MUJOCO_GL", "egl")

    n_env_workers = cfg["n_group"]
    pool = ParallelEnvPool(n_env_workers, cfg["task_suite"], start_method="fork")

    # ------------------------------------------------------------------
    # STEP 2: Load policy to this worker's GPU (CUDA initialises here).
    # ------------------------------------------------------------------
    device = f"cuda:{rank}"  # rank 0 is the trainer; workers use their own rank GPU
    policy_inference = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(
        device=device, dtype=torch.bfloat16
    )
    policy_inference.requires_grad_(False)
    policy_inference.eval()
    model_dtype = next(policy_inference.parameters()).dtype

    # ------------------------------------------------------------------
    # STEP 3: Build preprocessing pipeline (identical to train_grpo).
    # ------------------------------------------------------------------
    dataset_stats = _load_dataset_stats(cfg["model_id"])
    policy_inference.config.device = device
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy_inference.config, dataset_stats=dataset_stats
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    def preprocess_obs_fn(obs, lang):
        return _preprocess_obs(obs, lang, env_preprocessor, preprocessor, device, model_dtype)

    # ------------------------------------------------------------------
    # STEP 4: Join the NCCL process group (collective — blocks until
    #         the trainer also calls dist.init_process_group).
    # ------------------------------------------------------------------
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=10),
    )

    # ------------------------------------------------------------------
    # STEP 5: Signal readiness to trainer.
    # ------------------------------------------------------------------
    recv_q.put(("ready",))

    # ------------------------------------------------------------------
    # STEP 6: Command loop.
    # ------------------------------------------------------------------
    try:
        while True:
            msg = send_q.get()
            cmd = msg[0]

            if cmd == "reset":
                _, task_id, episode_indices, max_steps = msg
                try:
                    obs_list, task_language, env_max_steps = pool.reset_to_task(
                        task_id, episode_indices
                    )
                    if is_primary:
                        recv_q.put(("obs", obs_list, task_language, env_max_steps))
                    else:
                        recv_q.put(("reset_ok",))
                except Exception as e:
                    recv_q.put(("error", str(e)))

            elif cmd == "rollout":
                _, first_chunks_np, task_language, max_steps = msg
                try:
                    rewards = parallel_compute_rewards(
                        pool,
                        policy_inference,
                        first_chunks_np,
                        task_language,
                        preprocess_obs_fn,
                        postprocessor,
                        max_steps,
                    )
                    recv_q.put(("rewards", rewards))
                except Exception as e:
                    recv_q.put(("error", str(e)))

            elif cmd == "sync_weights":
                broadcast_weights_from_trainer(policy_inference)
                recv_q.put(("synced",))

            elif cmd == "stop":
                break

            else:
                recv_q.put(("error", f"unknown cmd: {cmd}"))

    finally:
        pool.close()
        dist.destroy_process_group()
