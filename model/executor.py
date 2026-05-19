"""
Subgoal-conditioned Actor Network.

Inputs (all pre-encoded / numpy features — no vision encoder inside):
  imgs           : (B, 4, patch_dim)  encoder latent features laid out as:
                     [0] curr_main   — SubgoalAutoencoder latent z_curr[:, :patch_dim]
                     [1] curr_wrist  — SubgoalAutoencoder latent z_curr[:, patch_dim:]
                     [2] sg_main     — SubgoalAutoencoder latent z_sg[:, :patch_dim]
                     [3] sg_wrist    — SubgoalAutoencoder latent z_sg[:, patch_dim:]
                   All 4 slots come from SubgoalAutoencoder.encode(concat(main, wrist patches)).
  current_proprio: (B, proprio_dim)   current EEF state
  subgoal_proprio: (B, proprio_dim)   subgoal EEF state

Output:
  action : (B, action_dim)   single-step action
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.distributions import Normal
from typing import Tuple


def build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers, dropout: float = 0.0) -> nn.Sequential:
    layers, in_dim = [], input_dim
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class Executor(nn.Module):
    """Subgoal-conditioned actor: outputs one action per call."""

    def __init__(
        self,
        num_imgs: int = 4,           # curr_main, curr_wrist, sg_main, sg_wrist
        patch_dim: int = 2048,       # SAE slot latent dim
        proprio_dim: int = 8,        # EEF state dim (same for current & subgoal)
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_hidden_layers: int = 5,
        log_std_init: float = -2.0,
        norm_stats_path: str = None,  # pi0.5 norm_stats.json
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim

        mlp_input_dim = num_imgs * patch_dim + 2 * proprio_dim
        self.mlp = build_mlp(mlp_input_dim, hidden_dim, action_dim, num_hidden_layers, dropout=dropout)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        # Action quantile norm stats from pi0.5 (required).
        ns = json.loads(Path(norm_stats_path).read_text())["norm_stats"]["actions"]
        self.register_buffer("action_q01", torch.tensor(ns["q01"][:action_dim], dtype=torch.float32))
        self.register_buffer("action_q99", torch.tensor(ns["q99"][:action_dim], dtype=torch.float32))
        print(f"[Executor] Loaded action norm_stats from {norm_stats_path}")

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_q01) / (self.action_q99 - self.action_q01) * 2.0 - 1.0

    def _unnormalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a + 1.0) / 2.0 * (self.action_q99 - self.action_q01) + self.action_q01

    def forward(
        self,
        imgs: torch.Tensor,            # (B, 4, patch_dim)
        current_proprio: torch.Tensor, # (B, proprio_dim)
        subgoal_proprio: torch.Tensor, # (B, proprio_dim)
        deterministic: bool = False,
        unnormalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        """
        MLP outputs action in normalized space ([-1, 1]).
        If unnormalize=True (inference), returns action in raw env scale.
        If unnormalize=False (loss computation), returns action in normalized space.
        """
        B = imgs.shape[0]
        img_feat = imgs.reshape(B, -1)  # (B, 4*patch_dim)

        x = torch.cat([img_feat, current_proprio, subgoal_proprio], dim=-1)
        mean = self.mlp(x)  # (B, action_dim) — normalized space

        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)

        if unnormalize:
            action = self._unnormalize_action(action)
        return action, log_prob, dist

    def compute_bc_loss(
        self,
        imgs: torch.Tensor,
        current_proprio: torch.Tensor,
        subgoal_proprio: torch.Tensor,
        target_action: torch.Tensor,   # (B, action_dim) — raw env scale
    ) -> Tuple[torch.Tensor, dict]:
        """BC loss in normalized action space ([-1, 1])."""
        action_norm, _, dist = self.forward(
            imgs, current_proprio, subgoal_proprio,
            deterministic=True, unnormalize=False,
        )
        target_norm = self._normalize_action(target_action)
        loss = F.l1_loss(action_norm, target_norm)
        info = {
            "loss/bc":   loss.item(),
            "train/std": dist.stddev.mean().item(),
        }
        return loss, info
