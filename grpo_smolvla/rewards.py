"""Weighted denoising reward: R = 0.1*r8 + 0.2*r9 + 0.7*r10."""

import numpy as np


def execute_and_score(raw_env, actions_np):
    """
    Execute an action chunk in a LIBERO OffScreenRenderEnv and return binary success {0, 1}.

    raw_env:    OffScreenRenderEnv (not LiberoEnv); returns 4-tuple from step()
    actions_np: ndarray of shape (chunk_size, action_dim) — already denormalized
    """
    done = False
    success = 0
    for act in actions_np:
        if done:
            break
        try:
            obs, reward, done, info = raw_env.step(act)
        except ValueError:
            # "executing action in terminated episode" — treat as not successful
            break
        if info.get("success", False):
            success = 1
            break
    return success


def compute_weighted_reward(raw_env, group_trajectory, postprocessor):
    """
    R(τ_i) = 0.1 * r(τ_i^8) + 0.2 * r(τ_i^9) + 0.7 * r(τ_i^10)

    raw_env:          OffScreenRenderEnv (already reset and stabilized via LiberoEnv.reset())
    group_trajectory: dict with actions_8, actions_9, actions_10 tensors (normalized)
    postprocessor:    callable that unnormalizes action tensors before env.step()

    For each denoising horizon, the sim state is restored to the post-stabilization
    state so all three sub-trajectories start from the same configuration.
    """
    # Save MuJoCo sim state and robosuite env counters after stabilization
    sim_state = raw_env.sim.get_state()
    saved_timestep = raw_env.env.timestep

    def run_horizon(actions_tensor):
        # Restore sim state
        raw_env.sim.set_state(sim_state)
        raw_env.sim.forward()
        # Reset robosuite counters so step() doesn't raise "terminated episode"
        raw_env.env.timestep = saved_timestep
        raw_env.env.done = False
        raw_env.env._obs_cache = {}

        # Denormalize policy actions to robot action space before stepping env
        actions_denorm = postprocessor(actions_tensor)
        actions_np = actions_denorm.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
        return execute_and_score(raw_env, actions_np)

    r8  = run_horizon(group_trajectory["actions_8"])
    r9  = run_horizon(group_trajectory["actions_9"])
    r10 = run_horizon(group_trajectory["actions_10"])

    return 0.1 * r8 + 0.2 * r9 + 0.7 * r10
