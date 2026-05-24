import argparse
import os
import random
import yaml
import torch
from datetime import datetime

# Isaac Sim/Isaac Lab requires AppLauncher before other imports
from isaaclab.app import AppLauncher

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

# Initialize AppLauncher
parser = argparse.ArgumentParser(description="GRPO Training for SmolVLA in Isaac Sim.")
parser.add_argument("--config", default="grpo_isaac/config.yaml", help="Path to config file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Now we can import the rest
import gymnasium as gym
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from grpo_isaac.env_adapter import IsaacSmolVLAAdapter
from grpo_isaac.grpo_engine import grpo_update, compute_batched_grpo_advantages
from grpo_smolvla.train_grpo import load_dataset_stats

def main():
    cfg = load_config(args_cli.config)
    device = cfg["device"]
    
    # 1. Setup Environment using LW-BenchHub logic
    # (Simplified for the repo; in production we use the registry)
    from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode
    
    # Example for Libero-10 task
    task_name = "L10K3TurnOnTheStoveAndPutTheMokaPotOnIt"
    env_cfg = parse_env_cfg(
        scene_backend="robocasa",
        task_backend="robocasa",
        task_name=task_name,
        robot_name="LeRobot-RL",
        scene_name="robocasakitchen",
        robot_scale=1.0,
        execute_mode=ExecuteMode.TRAIN,
        num_envs=cfg["num_envs"],
        device=device,
        headless_mode=cfg["headless"]
    )
    
    env = gym.make(f"Robocasa-Task-{task_name}", cfg=env_cfg)
    
    # 2. Load Policy
    policy = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device=device, dtype=torch.bfloat16)
    policy_ref = SmolVLAPolicy.from_pretrained(cfg["model_id"]).to(device=device, dtype=torch.bfloat16)
    policy_ref.requires_grad_(False)
    
    # 3. Adapter
    dataset_stats = load_dataset_stats(cfg["model_id"])
    adapter = IsaacSmolVLAAdapter(policy.config, dataset_stats, device)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW([
        {"params": policy.model.vlm_with_expert.vlm.parameters(), "lr": cfg["learning_rate_backbone"]},
        {"params": policy.model.vlm_with_expert.lm_expert.parameters(), "lr": cfg["learning_rate_head"]},
    ])
    
    # 5. Training Loop
    obs, info = env.reset()
    task_description = "Turn on the stove and put the moka pot on it." # From task meta
    
    for step in range(cfg["total_steps"]):
        # a. Preprocess vectorized obs
        obs_batch = adapter.preprocess(obs, [task_description] * cfg["num_envs"])
        
        # b. Sample trajectories (Flow Matching)
        # Note: rollout logic from grpo_smolvla/flow_utils.py can be used here
        # For simplicity in this template, we assume a single step policy
        with torch.no_grad():
            actions = policy.select_action(obs_batch) # B, chunk, dim
            
        # c. Step Sim in parallel
        actions_np = actions.cpu().numpy()
        obs, rewards, terminated, truncated, info = env.step(actions_np)
        
        # d. Compute Advantages and Update
        advantages = compute_batched_grpo_advantages(rewards)
        
        loss = grpo_update(
            policy, policy_ref, optimizer, obs_batch, actions, advantages,
            clip_eps=cfg["clip_eps"], kl_coeff=cfg["kl_coeff"]
        )
        
        print(f"Step {step} | Loss: {loss:.4f} | Mean Reward: {rewards.mean():.3f}")
        
        if step % cfg["save_every"] == 0:
            policy.save_pretrained(os.path.join(cfg["output_dir"], f"step_{step}"))

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
