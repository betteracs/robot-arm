"""GRPO advantage computation, flow-matching log-probability proxy, and policy update."""

import torch
from lerobot.policies.smolvla.modeling_smolvla import ACTION


def compute_grpo_advantages(rewards, eps=1e-8):
    """
    Â_i = (r_i - mean(r)) / (std(r) + ε)

    Zero-update property: when all rewards are identical (all succeed or all fail),
    std=0 → advantages ≈ 0 → no gradient. Built-in curriculum from GRPO.

    rewards: list of n scalar floats
    Returns: FloatTensor of shape (n,)
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    advantages = (r - r.mean()) / (r.std() + eps)
    return advantages


def flow_matching_log_prob(policy, obs_batch, noise, actions_target):
    """
    Approximate log π_θ(τ | o) via the flow-matching regression loss.

    log π_θ ≈ -L_FM(θ; noise, actions)

    Higher log-prob = lower FM loss. This is the score-function estimator
    approach described in §4.4 of the proposal.

    Returns: scalar tensor (requires_grad for policy update, detached for reference)
    """
    model_dtype = next(policy.parameters()).dtype
    batch = {**obs_batch, ACTION: actions_target.to(model_dtype)}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch, noise=noise.to(model_dtype))
    return -loss  # tensor with gradient path intact


def grpo_update(policy, policy_ref, optimizer, obs_batch, group_data, advantages, clip_eps=0.2, kl_coeff=0.01):
    """
    L_GRPO = -E[min(ρ_i * Â_i, clip(ρ_i, 1-ε, 1+ε) * Â_i)] + kl_coeff * KL(π_θ || π_ref)

    ρ_i = exp(log π_θ(τ_i) - log π_θ_old(τ_i))

    policy:     trainable SmolVLAPolicy
    policy_ref: frozen SFT reference policy (requires_grad=False)
    optimizer:  AdamW with differential LRs
    obs_batch:  observation dict
    group_data: list of dicts with keys noise, actions_8, actions_9, actions_10
    advantages: FloatTensor of shape (n,) from compute_grpo_advantages
    clip_eps:   PPO clipping threshold (default 0.2)
    kl_coeff:   KL regularization weight against SFT prior

    Returns: scalar loss float
    """
    device = next(policy.parameters()).device
    n = len(group_data)
    running_loss = 0.0

    optimizer.zero_grad()
    for traj, adv in zip(group_data, advantages):
        adv = adv.to(device)

        log_prob_new = flow_matching_log_prob(
            policy, obs_batch, traj["noise"], traj["actions_10"]
        )
        with torch.no_grad():
            log_prob_ref = flow_matching_log_prob(
                policy_ref, obs_batch, traj["noise"], traj["actions_10"]
            )

        ratio = torch.exp(log_prob_new - log_prob_ref.detach())
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)
        kl_penalty = kl_coeff * (log_prob_new - log_prob_ref.detach())

        # Divide by n here so accumulated gradients equal the mean over the group
        step_loss = (policy_loss + kl_penalty) / n
        step_loss.backward()
        running_loss += step_loss.item()

    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    return running_loss
