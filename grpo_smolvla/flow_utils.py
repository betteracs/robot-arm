"""Custom denoising loop, prefix KV-cache computation, and group trajectory sampling."""

import torch
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def compute_prefix_cache(policy, obs_batch):
    """
    Run the expensive VLM prefix forward pass once and cache the KV tensors.

    policy: SmolVLAPolicy instance
    obs_batch: preprocessed observation dict (from lerobot pipeline)
    Returns dict with past_key_values and prefix_pad_masks.
    """
    model = policy.model  # VLAFlowMatching

    # Extract tensors using policy's own prepare methods (handles padding, batch dim, etc.)
    images, img_masks = policy.prepare_images(obs_batch)
    state = policy.prepare_state(obs_batch)
    lang_tokens = obs_batch["observation.language.tokens"]
    lang_masks = obs_batch["observation.language.attention_mask"]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )

    return {
        "past_key_values": past_key_values,
        "prefix_pad_masks": prefix_pad_masks,
    }


def rollout_with_n_steps(flow_model, prefix_cache, noise, num_steps):
    """
    Run Euler integration for num_steps steps from a fixed noise tensor.
    Returns action chunk of shape (B, chunk_size, action_dim).

    Bypasses policy.config.num_steps so we can request 8, 9, or 10 steps
    without mutating config state (thread-safe).
    """
    dt = -1.0 / num_steps
    x_t = noise.clone()
    bsize = noise.shape[0]
    device = noise.device

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=noise.dtype, device=device).expand(bsize)
            v_t = flow_model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_cache["prefix_pad_masks"],
                past_key_values=prefix_cache["past_key_values"],
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
    return x_t


def sample_group_trajectories(policy, obs_batch, n_group=8):
    """
    Sample n_group independent trajectories from the policy for a single observation.

    Calls embed_prefix once (expensive VLM + KV-cache forward) and reuses the KV
    cache across all n_group noise vectors and all 3 denoising horizons per vector.

    Actions are trimmed to original_action_dim (e.g. 7 for LIBERO) — matching what
    SmolVLAPolicy._get_action_chunk does — so they are compatible with the postprocessor.
    The noise tensor stays at max_action_dim (32) for correct FM loss computation.

    Returns list of dicts, each with keys: noise, actions_8, actions_9, actions_10.
    """
    model = policy.model  # VLAFlowMatching
    B = 1
    action_shape = (B, model.config.chunk_size, model.config.max_action_dim)
    device = next(policy.parameters()).device
    model_dtype = next(policy.parameters()).dtype
    # Trim denoised actions to the original (non-padded) action dim, matching lerobot's
    # _get_action_chunk which does: actions = actions[:, :, :original_action_dim]
    original_action_dim = policy.config.action_feature.shape[0]

    prefix_cache = compute_prefix_cache(policy, obs_batch)

    group = []
    for _ in range(n_group):
        zi = torch.randn(action_shape, device=device, dtype=model_dtype)
        a8  = rollout_with_n_steps(model, prefix_cache, zi, num_steps=8)[:, :, :original_action_dim]
        a9  = rollout_with_n_steps(model, prefix_cache, zi, num_steps=9)[:, :, :original_action_dim]
        a10 = rollout_with_n_steps(model, prefix_cache, zi, num_steps=10)[:, :, :original_action_dim]
        group.append({"noise": zi, "actions_8": a8, "actions_9": a9, "actions_10": a10})
    return group
