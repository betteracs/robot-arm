"""Episode-level success reward for GRPO training on LIBERO."""
import torch


def compute_episode_reward(
    policy,
    first_chunk_actions,
    postprocessor,
    preprocess_obs_fn,
    env,
    task_language,
    raw_env,
    sim_state,
    saved_timestep,
    max_episode_steps,
):
    """
    Roll out a full episode and return binary success {0.0, 1.0}.

    First chunk comes from a GRPO-sampled denoising trajectory (so the policy
    gradient can flow through it when the primary chunk is rolled out). All
    subsequent chunks re-query the policy closed-loop from the current
    observation, matching eval behaviour.

    preprocess_obs_fn: callable(gym_obs, task_language) -> policy input batch
    env:               LiberoEnv — used only for _format_raw_obs conversion
    raw_env:           OffScreenRenderEnv — stepped directly (no auto-reset)
    sim_state:         MjSimState saved right after env.reset()
    saved_timestep:    robosuite timestep counter at that same point
    """
    # Restore every group member to the same initial state
    raw_env.sim.set_state(sim_state)
    raw_env.sim.forward()
    raw_env.env.timestep = saved_timestep
    raw_env.env.done = False
    raw_env.env._obs_cache = {}

    # --- First chunk: from the GRPO trajectory ---
    first_chunk = postprocessor(first_chunk_actions).squeeze(0).cpu().float().numpy()

    raw_obs = None
    for act in first_chunk:
        try:
            raw_obs, _, done, _ = raw_env.step(act)
        except ValueError:
            return 0.0
        if done:
            return 1.0

    if raw_obs is None:
        return 0.0

    # --- Remaining chunks: closed-loop policy re-query ---
    was_training = policy.training
    policy.eval()
    policy.reset()  # clear the action queue

    steps_done = len(first_chunk)
    try:
        while steps_done < max_episode_steps:
            gym_obs = env._format_raw_obs(raw_obs)
            obs_batch = preprocess_obs_fn(gym_obs, task_language)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                action = policy.select_action(obs_batch)
            act_np = postprocessor(action).squeeze(0).cpu().float().numpy()

            try:
                raw_obs, _, done, _ = raw_env.step(act_np)
            except ValueError:
                break
            if done:
                return 1.0
            steps_done += 1
    finally:
        if was_training:
            policy.train()

    return 0.0
