"""Batched GRPO engine for Isaac Sim environments."""

import torch
from lerobot.policies.smolvla.modeling_smolvla import ACTION


def compute_batched_grpo_advantages(rewards, eps=1e-8):
    """
    Compute GRPO advantages by normalizing rewards across the batch of environments.
    
    rewards: FloatTensor of shape (B,)
    Returns: FloatTensor of shape (B,)
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    
    mean = rewards.mean()
    std = rewards.std()
    
    # If all rewards are the same, std is 0. Advantages should be 0.
    if std < eps:
        return torch.zeros_like(rewards)
        
    return (rewards - mean) / (std + eps)


def flow_matching_log_prob(policy, obs_batch, noise, actions_target):
    """
    Approximate log π_θ(τ | o) via the flow-matching regression loss (vectorized).
    
    log π_θ ≈ -L_FM(θ; noise, actions)
    
    obs_batch: dict of tensors with batch dimension B
    noise: FloatTensor of shape (B, chunk_size, action_dim)
    actions_target: FloatTensor of shape (B, chunk_size, action_dim)
    
    Returns: FloatTensor of shape (B,) with gradient path intact.
    """
    model_dtype = next(policy.parameters()).dtype
    device = next(policy.parameters()).device
    
    # Prepare batch for policy forward
    # Ensure all tensors in obs_batch are on the correct device and dtype
    def _to_device(d):
        if isinstance(d, dict):
            return {k: _to_device(v) for k, v in d.items()}
        if isinstance(d, torch.Tensor):
            return d.to(device=device)
        return d

    batch = _to_device(obs_batch)
    batch[ACTION] = actions_target.to(device=device, dtype=model_dtype)
    
    # Use reduction="none" to get per-sample loss
    with torch.autocast("cuda", dtype=torch.bfloat16):
        per_sample_loss, _ = policy.forward(batch, noise=noise.to(device=device, dtype=model_dtype), reduction="none")
    
    return -per_sample_loss


def grpo_update(policy, policy_ref, optimizer, obs_batch, trajectories, advantages, clip_eps=0.2, kl_coeff=0.01):
    """
    Perform a batched GRPO update step.
    
    policy: trainable SmolVLAPolicy
    policy_ref: frozen reference SmolVLAPolicy
    optimizer: optimizer for policy
    obs_batch: dict of tensors with batch dimension B
    trajectories: dict with 'noise' and 'actions' tensors of shape (B, ...)
    advantages: FloatTensor of shape (B,)
    clip_eps: PPO clipping epsilon
    kl_coeff: KL regularization coefficient
    
    Returns: scalar loss value
    """
    optimizer.zero_grad()
    
    noise = trajectories["noise"]
    actions = trajectories["actions"]
    
    # Compute log probabilities for current and reference policies
    log_prob_new = flow_matching_log_prob(policy, obs_batch, noise, actions)
    
    with torch.no_grad():
        log_prob_ref = flow_matching_log_prob(policy_ref, obs_batch, noise, actions)
    
    # Importance sampling ratio
    # ratio = exp(log_prob_new - log_prob_ref)
    ratio = torch.exp(log_prob_new - log_prob_ref.detach())
    
    # Ensure advantages is on the same device as log_prob_new
    advantages = advantages.to(log_prob_new.device)
    
    # PPO-style clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    # KL regularization: KL(pi || pi_ref) approx log(pi/pi_ref)
    # Note: This is a simplified KL term often used in GRPO/PPO
    kl_penalty = kl_coeff * (log_prob_new - log_prob_ref.detach())
    
    # Total loss (mean over batch)
    loss = (policy_loss + kl_penalty).mean()
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()
