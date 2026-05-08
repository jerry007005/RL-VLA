"""
Subgoal-conditioned Actor Network.

Inputs (all pre-encoded / numpy features — no vision encoder inside):
  imgs           : (B, 4, patch_dim)  mean-pooled features for
                   [curr_main, curr_wrist, sg_main, sg_wrist]
  current_proprio: (B, proprio_dim)   current EEF state
  subgoal_proprio: (B, proprio_dim)   subgoal EEF state

Output:
  action : (B, action_dim)   single-step action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


def build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers) -> nn.Sequential:
    layers, in_dim = [], input_dim
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class Executor(nn.Module):
    """Subgoal-conditioned actor: outputs one action per call."""

    def __init__(
        self,
        num_imgs: int = 4,           # curr_main, curr_wrist, sg_main, sg_wrist
        patch_dim: int = 2048,       # pi0.5 SigLIP projection dim
        proprio_dim: int = 8,        # EEF state dim (same for current & subgoal)
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_hidden_layers: int = 5,
        log_std_init: float = -2.0,
    ):
        super().__init__()
        self.action_dim = action_dim

        mlp_input_dim = num_imgs * patch_dim + 2 * proprio_dim
        self.mlp = build_mlp(mlp_input_dim, hidden_dim, action_dim, num_hidden_layers)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(
        self,
        imgs: torch.Tensor,            # (B, 4, patch_dim)
        current_proprio: torch.Tensor, # (B, proprio_dim)
        subgoal_proprio: torch.Tensor, # (B, proprio_dim)
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        """
        Returns:
            action   : (B, action_dim)
            log_prob : (B,)
            dist     : Normal
        """
        B = imgs.shape[0]
        img_feat = imgs.reshape(B, -1)  # (B, 4*patch_dim)

        x = torch.cat([img_feat, current_proprio, subgoal_proprio], dim=-1)
        mean = self.mlp(x)  # (B, action_dim)

        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)

        return action, log_prob, dist

    def compute_bc_loss(
        self,
        imgs: torch.Tensor,
        current_proprio: torch.Tensor,
        subgoal_proprio: torch.Tensor,
        target_action: torch.Tensor,   # (B, action_dim)
    ) -> Tuple[torch.Tensor, dict]:
        """Behavioral cloning loss: MSE between predicted and ground-truth action."""
        action, _, dist = self.forward(imgs, current_proprio, subgoal_proprio)
        loss = F.l1_loss(action, target_action)
        info = {
            "loss/bc":     loss.item(),
            "train/std":   dist.stddev.mean().item(),
        }
        return loss, info
