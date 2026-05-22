"""Parallel rollout pool for GRPO group-episode reward computation.

n_group worker processes each hold a LIBERO env. The main process orchestrates:
- reset_to_task: all workers reset to the same task with different init states (parallel)
- step_chunks_parallel: each worker applies a fixed action chunk (parallel)
- step_one_parallel: each alive worker takes one step (parallel)
- batched policy forward in main process (GPU-efficient)

Compared to serial rollout (~520 sim steps × n_group sequential), this gives roughly
n_group× wall-clock speedup on the env-bound phase.
"""

import multiprocessing as mp
import os
import sys
import traceback
from typing import Any

import numpy as np
import torch


def _worker_loop(conn, suite_name):
    """Worker process: owns one LiberoEnv at a time. Responds to messages on `conn`.

    Sets headless GL + disables CUDA for the worker BEFORE importing libero/lerobot,
    otherwise on headless servers (Vast.ai, Docker) the worker dies during import.
    """
    # Force headless GL backend. EGL is the most reliable on Linux servers without X.
    os.environ.setdefault("MUJOCO_GL", "egl")
    # Don't clobber CUDA_VISIBLE_DEVICES — robosuite's EGL renderer uses it to pick
    # the GPU device for the offscreen render context, and an empty value crashes
    # with "invalid literal for int() with base 10: ''".
    # Workers won't allocate CUDA memory unless they explicitly call .cuda(); they don't.
    # Avoid OpenMP / MKL thread oversubscription when N workers run in parallel.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    def _send(*args):
        conn.send(args)

    try:
        # Lazy import so env vars above take effect.
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv
        bd = benchmark.get_benchmark_dict()
        suite = bd[suite_name]()
    except Exception as e:
        # Startup failed — report the full traceback to main before dying.
        tb = traceback.format_exc()
        try:
            _send("startup_error", f"{type(e).__name__}: {e}\n{tb}")
        except Exception:
            print(f"[worker] startup error (could not send): {e}\n{tb}", file=sys.stderr, flush=True)
        return

    # Signal readiness so main knows the worker survived imports.
    _send("ready")

    env = None
    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]

            if cmd == "init":
                _, task_id, episode_index = msg
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                try:
                    env = LiberoEnv(
                        task_suite=suite,
                        task_id=task_id,
                        task_suite_name=suite_name,
                        obs_type="pixels_agent_pos",
                        observation_height=256,
                        observation_width=256,
                        episode_index=episode_index,
                    )
                    obs, _ = env.reset()
                    _send("ok", obs, env.task_description, env._max_episode_steps)
                except Exception as e:
                    env = None
                    _send("error", repr(e))

            elif cmd == "save_state":
                raw = env._env
                _send("ok", raw.sim.get_state(), raw.env.timestep)

            elif cmd == "restore_state":
                _, sim_state, timestep = msg
                raw = env._env
                raw.sim.set_state(sim_state)
                raw.sim.forward()
                raw.env.timestep = timestep
                raw.env.done = False
                raw.env._obs_cache = {}
                _send("ok")

            elif cmd == "step_chunk":
                _, action_chunk = msg
                raw = env._env
                done = False
                err = False
                raw_obs = None
                for act in action_chunk:
                    try:
                        raw_obs, _, done, _ = raw.step(act)
                    except ValueError:
                        err = True
                        break
                    if done:
                        break
                if err:
                    _send("error")
                elif done:
                    _send("done")
                else:
                    _send("continue", env._format_raw_obs(raw_obs))

            elif cmd == "step_one":
                _, action = msg
                raw = env._env
                try:
                    raw_obs, _, done, _ = raw.step(action)
                except ValueError:
                    _send("error")
                    continue
                if done:
                    _send("done")
                else:
                    _send("continue", env._format_raw_obs(raw_obs))

            elif cmd == "close":
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                _send("ok")
                break

            else:
                _send("error", f"unknown cmd: {cmd}")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


class ParallelEnvPool:
    """Pool of n_workers LiberoEnv worker processes.

    Use as a context manager or call .close() explicitly.
    """

    def __init__(self, n_workers: int, suite_name: str, startup_timeout: float = 120.0):
        self.n_workers = n_workers
        self.suite_name = suite_name
        ctx = mp.get_context("spawn")
        self.parents = []
        self.workers = []
        for _ in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_loop, args=(child, suite_name), daemon=True)
            p.start()
            self.parents.append(parent)
            self.workers.append(p)

        # Wait for every worker to signal readiness (post-import). If any worker
        # crashed during import, surface the traceback here instead of failing
        # silently later with BrokenPipeError on the first send.
        for i, parent in enumerate(self.parents):
            if not parent.poll(timeout=startup_timeout):
                self._terminate_all()
                raise RuntimeError(
                    f"worker {i} did not signal ready within {startup_timeout}s — "
                    "check stderr for crash traceback (likely MuJoCo GL or import error)"
                )
            resp = parent.recv()
            if resp[0] == "ready":
                continue
            if resp[0] == "startup_error":
                self._terminate_all()
                raise RuntimeError(f"worker {i} startup failed:\n{resp[1]}")
            self._terminate_all()
            raise RuntimeError(f"worker {i} sent unexpected message during startup: {resp}")

    def _terminate_all(self):
        for p in self.workers:
            if p.is_alive():
                p.terminate()
        for p in self.workers:
            p.join(timeout=2)

    def reset_to_task(self, task_id: int, episode_indices: list[int]):
        """Reset all workers to the same task, each with its own init state.

        Returns (obs_list, task_description, max_episode_steps).
        Raises RuntimeError if any worker fails.
        """
        assert len(episode_indices) == self.n_workers

        # Verify all workers are still alive before sending, so we get a clear
        # error message instead of BrokenPipeError on the first dead one.
        dead = [i for i, p in enumerate(self.workers) if not p.is_alive()]
        if dead:
            raise RuntimeError(f"worker(s) {dead} are dead (exit codes: "
                               f"{[self.workers[i].exitcode for i in dead]})")

        for i, (parent, ep_idx) in enumerate(zip(self.parents, episode_indices)):
            try:
                parent.send(("init", task_id, ep_idx))
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(f"send to worker {i} failed ({e}); worker likely died")

        obs_list = []
        task_desc = None
        max_steps = None
        for i, parent in enumerate(self.parents):
            try:
                resp = parent.recv()
            except (EOFError, OSError) as e:
                raise RuntimeError(f"recv from worker {i} failed ({e}); worker died mid-init")
            if resp[0] != "ok":
                raise RuntimeError(f"worker {i} init failed: {resp[1] if len(resp) > 1 else ''}")
            obs_list.append(resp[1])
            task_desc = resp[2]
            max_steps = resp[3]
        return obs_list, task_desc, max_steps

    def save_states(self):
        """Returns list of (sim_state, timestep) tuples."""
        for parent in self.parents:
            parent.send(("save_state",))
        out = []
        for parent in self.parents:
            resp = parent.recv()
            out.append((resp[1], resp[2]))
        return out

    def restore_states(self, states):
        for parent, (sim_state, ts) in zip(self.parents, states):
            parent.send(("restore_state", sim_state, ts))
        for parent in self.parents:
            parent.recv()

    def step_chunks_parallel(self, action_chunks):
        """Each worker applies action_chunks[i] (T, action_dim) sequentially.

        Returns list of (status, last_obs_or_None) where status in {'done','continue','error'}.
        """
        for parent, chunk in zip(self.parents, action_chunks):
            parent.send(("step_chunk", chunk))
        results = []
        for parent in self.parents:
            resp = parent.recv()
            if resp[0] == "continue":
                results.append(("continue", resp[1]))
            else:
                results.append((resp[0], None))
        return results

    def step_one_parallel(self, actions, alive_mask):
        """Send one action to each alive worker, recv (status, obs_or_None).

        actions: list[np.ndarray | None] of length n_workers (None for dead)
        alive_mask: list[bool] of length n_workers
        Returns list[(status, obs_or_None) | None] (None for dead workers).
        """
        for i, parent in enumerate(self.parents):
            if alive_mask[i]:
                parent.send(("step_one", actions[i]))
        results: list[Any] = [None] * self.n_workers
        for i, parent in enumerate(self.parents):
            if alive_mask[i]:
                resp = parent.recv()
                if resp[0] == "continue":
                    results[i] = ("continue", resp[1])
                else:
                    results[i] = (resp[0], None)
        return results

    def close(self):
        for parent in self.parents:
            try:
                parent.send(("close",))
                parent.recv()
            except Exception:
                pass
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _stack_obs_batches(obs_batches):
    """Stack a list of batch-size-1 obs dicts into a single batch.

    obs_batches: list of dicts with keys like 'observation.state', 'observation.images.cam1', etc.
    Returns a single dict with stacked tensors.
    """
    out = {}
    keys = obs_batches[0].keys()
    for k in keys:
        v0 = obs_batches[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.cat([ob[k] for ob in obs_batches], dim=0)
        else:
            # non-tensor (e.g. string list): take first (all envs have same task language)
            out[k] = v0
    return out


def batched_sample_actions(policy, big_obs_batch):
    """Run policy forward with batch_size=N and return action chunks.

    Bypasses select_action's per-policy action queue. Returns un-postprocessed
    actions, shape (N, chunk_size, original_action_dim).
    """
    model = policy.model
    images, img_masks = policy.prepare_images(big_obs_batch)
    state = policy.prepare_state(big_obs_batch)
    lang_tokens = big_obs_batch["observation.language.tokens"]
    lang_masks = big_obs_batch["observation.language.attention_mask"]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actions = model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
    # Trim padded action dims to the env's true action dim (lerobot convention).
    original_action_dim = policy.config.action_feature.shape[0]
    return actions[:, :, :original_action_dim]


def parallel_compute_rewards(
    pool: ParallelEnvPool,
    policy,
    first_chunks_np: list[np.ndarray],
    task_language: str,
    preprocess_obs_fn,
    postprocessor,
    max_episode_steps: int,
):
    """Run n_workers rollouts in parallel and return per-worker binary reward.

    first_chunks_np[i]: numpy array (T, action_dim) — initial chunk from GRPO trajectory.
                       Each worker applies its own chunk; pool's sim state must already
                       be at the rollout start (post-reset).
    preprocess_obs_fn(gym_obs, task_language) -> single-env preprocessed obs batch (dim-0 size 1).
    postprocessor: lerobot policy postprocessor (handles batch dim).

    Returns list[float] of per-worker rewards in {0.0, 1.0}.
    """
    n = pool.n_workers
    rewards = [0.0] * n

    # Phase 1: apply the GRPO trajectory's first chunk on every worker in parallel.
    chunk_results = pool.step_chunks_parallel(first_chunks_np)

    alive = []
    last_obs: dict[int, Any] = {}
    for i, (status, ob) in enumerate(chunk_results):
        if status == "done":
            rewards[i] = 1.0
        elif status == "continue":
            alive.append(i)
            last_obs[i] = ob
        # status == "error": reward stays 0

    if not alive:
        return rewards

    steps_done = len(first_chunks_np[0])  # all chunks have same length (50)

    # Phase 2: closed-loop with batched policy forward.
    was_training = policy.training
    policy.eval()
    try:
        while alive and steps_done < max_episode_steps:
            # Generate a fresh chunk for every alive worker (batched).
            obs_batches = [preprocess_obs_fn(last_obs[i], task_language) for i in alive]
            big_batch = _stack_obs_batches(obs_batches)
            action_chunks = batched_sample_actions(policy, big_batch)  # (n_alive, T, A)
            action_chunks = postprocessor(action_chunks)  # batched post
            chunk_len = action_chunks.shape[1]
            action_chunks_np = action_chunks.cpu().float().numpy()

            chunks: dict[int, np.ndarray] = {
                alive[j]: action_chunks_np[j] for j in range(len(alive))
            }

            # Step through this chunk one action at a time across all alive workers.
            for k in range(chunk_len):
                if not alive or steps_done >= max_episode_steps:
                    break

                alive_mask = [i in chunks for i in range(n)]
                actions_to_send = [chunks[i][k] if i in chunks else None for i in range(n)]

                results = pool.step_one_parallel(actions_to_send, alive_mask)

                new_alive = []
                for i in alive:
                    res = results[i]
                    if res is None:
                        continue
                    status, ob = res
                    if status == "done":
                        rewards[i] = 1.0
                        chunks.pop(i, None)
                    elif status == "error":
                        chunks.pop(i, None)
                    else:
                        last_obs[i] = ob
                        new_alive.append(i)
                alive = new_alive
                steps_done += 1
    finally:
        if was_training:
            policy.train()

    return rewards
