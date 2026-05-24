import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

def quat_to_axisangle(quat):
    """Convert wxyz quaternion tensor to axis-angle."""
    w, x, y, z = quat.unbind(-1)
    angle = 2.0 * torch.acos(torch.clamp(w.abs(), 0.0, 1.0))
    angle = torch.where(w < 0, -angle, angle)
    sin_half = torch.sqrt(1.0 - torch.clamp(w * w, 0.0, 1.0))
    mask = sin_half < 1e-6
    axis = torch.stack([x, y, z], dim=-1) / (sin_half.unsqueeze(-1) + 1e-12)
    res = axis * angle.unsqueeze(-1)
    res[mask] = 0.0
    return res

class IsaacSmolVLAAdapter:
    def __init__(self, policy_config, dataset_stats, device):
        self.device = device
        self.preprocessor, self.postprocessor = make_smolvla_pre_post_processors(
            policy_config, dataset_stats=dataset_stats
        )

    def preprocess(self, obs_dict, task_description):
        """
        Convert Isaac Lab/LW-BenchHub observation tensors to SmolVLA batch.
        
        Args:
            obs_dict: Dict of tensors from vectorized env (e.g., from env.step())
            task_description: List of task strings (one per environment)
        """
        batch = {}
        
        # Image Processing: Convert (B, H, W, C) uint8/float to (B, C, H, W) normalized
        # We apply a 180-degree flip to match the LiberoProcessorStep used in training.
        if "global_camera" in obs_dict:
            img = obs_dict["global_camera"].to(self.device).float() / 255.0
            img = torch.flip(img, dims=[1, 2]) # Flip H and W
            batch["observation.images.camera1"] = img.permute(0, 3, 1, 2)
            
        if "hand_camera" in obs_dict:
            img = obs_dict["hand_camera"].to(self.device).float() / 255.0
            img = torch.flip(img, dims=[1, 2])
            batch["observation.images.camera2"] = img.permute(0, 3, 1, 2)

        # State Processing: [eef_pos(3), axis_angle(3), gripper(2)]
        eef_pos = obs_dict["eef_pos"].to(self.device)
        eef_quat = obs_dict["eef_quat"].to(self.device) # Expected wxyz
        gripper = obs_dict["gripper_pos"].to(self.device)
        
        eef_aa = quat_to_axisangle(eef_quat)
        state = torch.cat([eef_pos, eef_aa, gripper], dim=-1)
        batch["observation.state"] = state

        # Language Processing
        batch["task"] = task_description
        
        # Run standard SmolVLA preprocessor (tokenization and normalization)
        return self.preprocessor(batch)
