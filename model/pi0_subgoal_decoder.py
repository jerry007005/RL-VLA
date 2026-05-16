"""
PI0WithGoalExpert
=================
Inherits PI0Pytorch and adds a flow-matching goal expert that generates
subgoal features directly compatible with the Executor.

Architecture
------------
  Frozen PI0 backbone (inherited)
    _prefix_kv_cache():
      embed_image(main)  → (B, 256, 2048)  ┐
      embed_image(wrist) → (B, 256, 2048)  ├→ concat → (B, 560, 2048)
      embed_language()   → (B,  48, 2048)  ┘
      frozen PaliGemma LM → prefix KV cache

  Trainable goal expert (GemmaForCausalLM, gemma_300m, bfloat16)
    Suffix: 5 tokens, fully bidirectional
      token 0: curr_main_in_proj( z_curr[:2048] )   (B, W)  ← current obs conditioning
      token 1: curr_wrist_in_proj( z_curr[2048:] )  (B, W)  ← current obs conditioning
      token 2: sg_main_in_proj( noisy_sg_main  )    (B, W)
      token 3: sg_wrist_in_proj( noisy_sg_wrist )   (B, W)
      token 4: sg_state_in_proj( noisy_sg_state )   (B, W)
    Timestep → AdaRMS conditioning
    All 5 tokens attend to each other + full prefix KV

  Output heads (applied only to tokens 2, 3, 4)
    sg_main_out_proj  : (B, W) → (B, 2048)
    sg_wrist_out_proj : (B, W) → (B, 2048)
    sg_state_out_proj : (B, W) → (B, 8)

Training: flow matching (same as PI0)
  noisy = t * noise + (1-t) * target
  predict u = noise - target
  loss = MSE over (sg_main, sg_wrist, sg_state)

Inference: sample_goal(curr_z) → (sg_main_emb, sg_wrist_emb, sg_state) float32
  Directly stackable as Executor's imgs (B, 4, 2048) + subgoal_proprio (B, 8)
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "openpi" / "src"))

from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING

from openpi.models import gemma as _gemma
from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    sample_beta,
)


class PI0WithGoalExpert(PI0Pytorch):
    """
    PI0 + flow-matching goal expert.

    sample_goal(curr_z) returns (sg_main_emb, sg_wrist_emb, sg_state) that plug
    directly into the Executor:
        imgs = torch.stack([curr_main, curr_wrist, sg_main_emb, sg_wrist_emb], dim=1)
        subgoal_proprio = sg_state
    """

    def __init__(
        self,
        config,
        patch_dim:      int  = 2048,
        proprio_dim:    int  = 8,
        freeze_pi0:     bool = True,
        expert_variant: str  = "gemma_2b",
        norm_stats_path: str = None,  # pi0.5 norm_stats.json
    ):
        super().__init__(config=config)

        self.patch_dim   = patch_dim
        self.proprio_dim = proprio_dim

        # Action quantile norm stats from pi0.5 (required).
        import json as _json
        ns = _json.loads(Path(norm_stats_path).read_text())["norm_stats"]["actions"]
        self.register_buffer("action_q01", torch.tensor(ns["q01"], dtype=torch.float32))
        self.register_buffer("action_q99", torch.tensor(ns["q99"], dtype=torch.float32))
        print(f"[PI0WithGoalExpert] Loaded action norm_stats from {norm_stats_path}")

        if freeze_pi0:
            for p in self.parameters():
                p.requires_grad_(False)

        expert_cfg   = _gemma.get_config(expert_variant)
        W            = expert_cfg.width   # 1024 for gemma_300m
        self._W      = W

        expert_hf_cfg = CONFIG_MAPPING["gemma"](
            head_dim              = expert_cfg.head_dim,
            hidden_size           = W,
            intermediate_size     = expert_cfg.mlp_dim,
            num_attention_heads   = expert_cfg.num_heads,
            num_hidden_layers     = expert_cfg.depth,
            num_key_value_heads   = expert_cfg.num_kv_heads,
            vocab_size            = 257152,
            hidden_activation     = "gelu_pytorch_tanh",
            torch_dtype           = "bfloat16",
            use_adarms            = True,
            adarms_cond_dim       = W,
            attn_implementation   = "sdpa",
        )
        self.goal_expert = GemmaForCausalLM(expert_hf_cfg).to(torch.bfloat16)
        self.goal_expert.model.embed_tokens = None

        # Timestep MLP for AdaRMS
        self.goal_time_mlp_in  = nn.Linear(W, W, dtype=torch.bfloat16)
        self.goal_time_mlp_out = nn.Linear(W, W, dtype=torch.bfloat16)

        # Current obs conditioning projections (z_curr → expert width)
        self.curr_main_in_proj  = nn.Linear(patch_dim, W, dtype=torch.bfloat16)
        self.curr_wrist_in_proj = nn.Linear(patch_dim, W, dtype=torch.bfloat16)

        # Noisy target input projections
        self.sg_main_in_proj  = nn.Linear(patch_dim,   W, dtype=torch.bfloat16)
        self.sg_wrist_in_proj = nn.Linear(patch_dim,   W, dtype=torch.bfloat16)
        self.sg_state_in_proj = nn.Linear(proprio_dim, W, dtype=torch.bfloat16)

        # Output projections (expert hidden → original dims, tokens 2/3/4 only).
        # 2-layer MLP with SiLU to break rank bottleneck when W < patch_dim.
        self.sg_main_out_proj  = nn.Sequential(
            nn.Linear(W, patch_dim,         dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(patch_dim, patch_dim, dtype=torch.bfloat16),
        )
        self.sg_wrist_out_proj = nn.Sequential(
            nn.Linear(W, patch_dim,         dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(patch_dim, patch_dim, dtype=torch.bfloat16),
        )
        self.sg_state_out_proj = nn.Linear(W, proprio_dim, dtype=torch.bfloat16)

        # Horizon head: predicts (sg_idx - curr_idx) / MAX_HORIZON from token 0.
        # Token 0 is the curr_main conditioning token — it sees the full prefix
        # KV + all suffix tokens bidirectionally and is well-placed to regress
        # the temporal distance to the subgoal.
        self.MAX_HORIZON   = 200
        self.horizon_head  = nn.Linear(W, 1, dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Noise / time helpers
    # ------------------------------------------------------------------

    def _sample_noise(self, shape, device):
        return torch.randn(*shape, dtype=torch.float32, device=device)

    def _sample_time(self, B, device):
        t = sample_beta(1.5, 1.0, B, device)
        return (t * 0.95 + 0.05).to(dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Action norm helpers (pi0.5 action quantiles)
    # ------------------------------------------------------------------

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_q01) / (self.action_q99 - self.action_q01) * 2.0 - 1.0

    def _unnormalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a + 1.0) / 2.0 * (self.action_q99 - self.action_q01) + self.action_q01

    # ------------------------------------------------------------------
    # Prefix: current obs → KV cache
    # ------------------------------------------------------------------

    def _build_prefix(self, main_img, wrist_img, lang_tokens, lang_mask):
        B = main_img.shape[0]
        embs, pad_masks = [], []

        for img in [main_img, wrist_img]:
            img_emb = self.paligemma_with_expert.embed_image(img).to(torch.bfloat16)
            n = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(torch.ones(B, n, dtype=torch.bool, device=img.device))

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens).to(torch.bfloat16)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_mask)

        embs      = torch.cat(embs, dim=1)          # (B, 560, 2048)
        pad_masks = torch.cat(pad_masks, dim=1)     # (B, 560)
        att_masks = torch.zeros(B, embs.shape[1], dtype=torch.float32, device=embs.device)
        return embs, pad_masks, att_masks

    def _prefix_kv_cache(self, main_img, wrist_img, lang_tokens, lang_mask):
        prefix_embs, prefix_pad, prefix_att = self._build_prefix(
            main_img, wrist_img, lang_tokens, lang_mask
        )
        prefix_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad.long(), dim=1) - 1
        prefix_4d  = self._prepare_attention_masks_4d(prefix_2d).to(torch.bfloat16)

        _, kv = self.paligemma_with_expert.forward(
            attention_mask  = prefix_4d,
            position_ids    = prefix_pos,
            past_key_values = None,
            inputs_embeds   = [prefix_embs, None],
            use_cache       = True,
        )
        return kv, prefix_pad

    # ------------------------------------------------------------------
    # Goal suffix: 2 conditioning + 3 noisy tokens → suffix embeddings
    # ------------------------------------------------------------------

    def _embed_goal_suffix(
        self,
        noisy_main:  torch.Tensor,   # (B, 2048) float32
        noisy_wrist: torch.Tensor,   # (B, 2048) float32
        noisy_state: torch.Tensor,   # (B, 8)    float32
        curr_main:   torch.Tensor,   # (B, 2048) float32  ← z_curr[:, :2048]
        curr_wrist:  torch.Tensor,   # (B, 2048) float32  ← z_curr[:, 2048:]
        time:        torch.Tensor,   # (B,)      float32
    ):
        B      = noisy_main.shape[0]
        device = noisy_main.device

        # Conditioning tokens (current obs, not denoised)
        tok_curr_main  = self.curr_main_in_proj(curr_main.to(torch.bfloat16))    # (B, W)
        tok_curr_wrist = self.curr_wrist_in_proj(curr_wrist.to(torch.bfloat16))  # (B, W)

        # Noisy target tokens
        tok_main  = self.sg_main_in_proj(noisy_main.to(torch.bfloat16))   # (B, W)
        tok_wrist = self.sg_wrist_in_proj(noisy_wrist.to(torch.bfloat16)) # (B, W)
        tok_state = self.sg_state_in_proj(noisy_state.to(torch.bfloat16)) # (B, W)

        # [curr_main, curr_wrist, sg_main, sg_wrist, sg_state]
        suffix_embs = torch.stack(
            [tok_curr_main, tok_curr_wrist, tok_main, tok_wrist, tok_state], dim=1
        )  # (B, 5, W)

        time_emb = create_sinusoidal_pos_embedding(
            time, self._W, min_period=4e-3, max_period=4.0, device=device
        ).to(torch.bfloat16)
        adarms_cond = F.silu(self.goal_time_mlp_out(self.goal_time_mlp_in(time_emb)))  # (B, W)

        pad_masks = torch.ones(B, 5,  dtype=torch.bool,    device=device)
        att_masks = torch.zeros(B, 5, dtype=torch.float32, device=device)
        return suffix_embs, pad_masks, att_masks, adarms_cond

    # ------------------------------------------------------------------
    # One denoising step
    # ------------------------------------------------------------------

    def _denoise_step(
        self,
        noisy_main:  torch.Tensor,
        noisy_wrist: torch.Tensor,
        noisy_state: torch.Tensor,
        curr_main:   torch.Tensor,
        curr_wrist:  torch.Tensor,
        time:        torch.Tensor,
        prefix_kv,
        prefix_pad:  torch.Tensor,
    ):
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self._embed_goal_suffix(
            noisy_main, noisy_wrist, noisy_state, curr_main, curr_wrist, time
        )
        B, prefix_len = prefix_pad.shape

        prefix_pad_2d = prefix_pad[:, None, :].expand(B, 5, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)  # (B, 5, prefix+5)
        full_att_4d   = self._prepare_attention_masks_4d(full_att_2d).to(torch.bfloat16)

        prefix_offsets = prefix_pad.sum(dim=-1)[:, None]
        suffix_pos_ids = prefix_offsets + torch.cumsum(suffix_pad.long(), dim=1) - 1

        out = self.goal_expert.model.forward(
            inputs_embeds   = suffix_embs,
            attention_mask  = full_att_4d,
            position_ids    = suffix_pos_ids,
            past_key_values = prefix_kv,
            use_cache       = False,
            adarms_cond     = adarms_cond,
        ).last_hidden_state  # (B, 5, W) bfloat16

        # Project only the 3 noisy tokens (indices 2, 3, 4).
        # Visual tokens use residual parameterization: model predicts the correction
        # over noisy/t, so the target becomes -delta/t instead of noise-delta.
        # This bypasses the rank bottleneck for the dominant noise component.
        t_col   = time[:, None].float()                                        # (B, 1)
        v_main  = self.sg_main_out_proj(out[:, 2]).float()  + noisy_main.float()  / t_col
        v_wrist = self.sg_wrist_out_proj(out[:, 3]).float() + noisy_wrist.float() / t_col
        v_state = self.sg_state_out_proj(out[:, 4]).float()                    # (B, 8)

        # Horizon prediction from token 0 (curr_main conditioning).
        # Output is normalized steps in (0, MAX_HORIZON); sigmoid keeps it positive.
        horizon = (torch.sigmoid(self.horizon_head(out[:, 0])).float().squeeze(-1)
                   * self.MAX_HORIZON)                                         # (B,)
        return v_main, v_wrist, v_state, horizon

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_goal(
        self,
        main_img:    torch.Tensor,   # (B, 3, H, W) float32
        wrist_img:   torch.Tensor,   # (B, 3, H, W) float32
        lang_tokens: torch.Tensor,   # (B, T) int64
        lang_mask:   torch.Tensor,   # (B, T) bool
        curr_main:   torch.Tensor,   # (B, 2048) float32  ← z_curr[:, :2048]
        curr_wrist:  torch.Tensor,   # (B, 2048) float32  ← z_curr[:, 2048:]
        sg_main:     torch.Tensor,   # (B, 2048) float32  ← encoder latent target
        sg_wrist:    torch.Tensor,   # (B, 2048) float32  ← encoder latent target
        sg_state:    torch.Tensor,   # (B, 8)    float32  ← robot state target
        horizon:     torch.Tensor | None = None,  # (B,) int  sg_idx - curr_idx
        noise_main:  torch.Tensor | None = None,
        noise_wrist: torch.Tensor | None = None,
        noise_state: torch.Tensor | None = None,
        time:        torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B      = main_img.shape[0]
        device = main_img.device

        if noise_main  is None: noise_main  = self._sample_noise(sg_main.shape,  device)
        if noise_wrist is None: noise_wrist = self._sample_noise(sg_wrist.shape, device)
        if noise_state is None: noise_state = self._sample_noise(sg_state.shape, device)
        if time        is None: time        = self._sample_time(B, device)

        # Flow matching on delta (sg - curr) for visual tokens; state stays absolute.
        delta_main  = sg_main  - curr_main
        delta_wrist = sg_wrist - curr_wrist

        t = time[:, None]
        noisy_main  = t * noise_main  + (1 - t) * delta_main
        noisy_wrist = t * noise_wrist + (1 - t) * delta_wrist
        noisy_state = t * noise_state + (1 - t) * sg_state
        u_main  = noise_main  - delta_main
        u_wrist = noise_wrist - delta_wrist
        u_state = noise_state - sg_state

        prefix_kv, prefix_pad = self._prefix_kv_cache(
            main_img, wrist_img, lang_tokens, lang_mask
        )

        v_main, v_wrist, v_state, horizon_pred = self._denoise_step(
            noisy_main, noisy_wrist, noisy_state,
            curr_main, curr_wrist,
            time, prefix_kv, prefix_pad
        )

        loss_main  = F.mse_loss(v_main,  u_main)
        loss_wrist = F.mse_loss(v_wrist, u_wrist)
        loss_state = F.mse_loss(v_state, u_state)

        # Horizon loss: Huber for robustness to outliers (long episodes).
        # Target normalized to [0, 1] by MAX_HORIZON.
        if horizon is not None:
            horizon_norm = horizon.float() / self.MAX_HORIZON
            loss_horizon = F.huber_loss(horizon_pred / self.MAX_HORIZON, horizon_norm)
        else:
            loss_horizon = torch.zeros(1, device=device).squeeze()

        loss = loss_main + loss_wrist + loss_state + loss_horizon

        return loss, {
            "loss/total":      loss.item(),
            "loss/sg_main":    loss_main.item(),
            "loss/sg_wrist":   loss_wrist.item(),
            "loss/sg_state":   loss_state.item(),
            "loss/horizon":    loss_horizon.item(),
        }

    def forward(self, main_img, wrist_img, lang_tokens, lang_mask,
                curr_main, curr_wrist, sg_main, sg_wrist, sg_state, **kwargs):
        """Standard forward for DDP compatibility — delegates to forward_goal."""
        return self.forward_goal(main_img, wrist_img, lang_tokens, lang_mask,
                                 curr_main, curr_wrist, sg_main, sg_wrist, sg_state, **kwargs)

    # ------------------------------------------------------------------
    # Inference: Euler sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_goal(
        self,
        main_img:    torch.Tensor,   # (B, 3, H, W) float32
        wrist_img:   torch.Tensor,   # (B, 3, H, W) float32
        lang_tokens: torch.Tensor,   # (B, T) int64
        lang_mask:   torch.Tensor,   # (B, T) bool
        curr_z:      torch.Tensor,   # (B, 4096) float32  ← encoder latent of current frame
        num_steps:   int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (sg_main_emb, sg_wrist_emb, sg_state) all float32.

        Plug directly into Executor:
            imgs = torch.stack([curr_main, curr_wrist, sg_main_emb, sg_wrist_emb], dim=1)
            action, _, _ = executor(imgs, curr_state, sg_state)
        """
        B      = main_img.shape[0]
        device = main_img.device

        curr_main  = curr_z[:, :self.patch_dim].float()
        curr_wrist = curr_z[:, self.patch_dim:].float()

        prefix_kv, prefix_pad = self._prefix_kv_cache(
            main_img, wrist_img, lang_tokens, lang_mask
        )

        x_main  = self._sample_noise((B, self.patch_dim),   device)
        x_wrist = self._sample_noise((B, self.patch_dim),   device)
        x_state = self._sample_noise((B, self.proprio_dim), device)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        t  = torch.tensor(1.0,              dtype=torch.float32, device=device)

        horizon_pred = None
        while t >= -dt / 2:
            v_main, v_wrist, v_state, horizon_pred = self._denoise_step(
                x_main, x_wrist, x_state,
                curr_main, curr_wrist,
                t.expand(B), prefix_kv, prefix_pad
            )
            x_main  = x_main  + dt * v_main
            x_wrist = x_wrist + dt * v_wrist
            x_state = x_state + dt * v_state
            t = t + dt

        # x_main/x_wrist are deltas; recover absolute subgoal by adding curr.
        # horizon_pred is in steps (float), from the final denoising step.
        return curr_main + x_main, curr_wrist + x_wrist, x_state, horizon_pred

    @torch.no_grad()
    def sample_all(
        self,
        main_img:    torch.Tensor,   # (B, 3, H, W)
        wrist_img:   torch.Tensor,   # (B, 3, H, W)
        lang_tokens: torch.Tensor,   # (B, T) int64
        lang_mask:   torch.Tensor,   # (B, T) bool
        curr_z:      torch.Tensor,   # (B, 4096)
        state:       torch.Tensor,   # (B, 8)
        num_steps:   int = 10,
    ) -> tuple:
        """
        Shared prefix KV: goal expert denoising + PI0.5 action denoising in one call.
        Returns (actions, sg_main, sg_wrist, sg_state, horizon_pred)
          actions: (B, action_horizon, action_dim) float32
        """
        B      = main_img.shape[0]
        device = main_img.device

        curr_main  = curr_z[:, :self.patch_dim].float()
        curr_wrist = curr_z[:, self.patch_dim:].float()

        # Build prefix KV once — shared for goal and action denoising
        prefix_kv, prefix_pad = self._prefix_kv_cache(main_img, wrist_img, lang_tokens, lang_mask)

        # ---- Goal expert denoising ----
        x_main  = self._sample_noise((B, self.patch_dim),   device)
        x_wrist = self._sample_noise((B, self.patch_dim),   device)
        x_state = self._sample_noise((B, self.proprio_dim), device)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        t  = torch.tensor(1.0,              dtype=torch.float32, device=device)

        horizon_pred = None
        while t >= -dt / 2:
            v_main, v_wrist, v_state, horizon_pred = self._denoise_step(
                x_main, x_wrist, x_state, curr_main, curr_wrist,
                t.expand(B), prefix_kv, prefix_pad,
            )
            x_main  = x_main  + dt * v_main
            x_wrist = x_wrist + dt * v_wrist
            x_state = x_state + dt * v_state
            t = t + dt

        sg_main  = curr_main + x_main
        sg_wrist = curr_wrist + x_wrist
        sg_state = x_state

        # ---- PI0.5 action denoising (reuses same prefix KV) ----
        action_shape = (B, self.config.action_horizon, self.config.action_dim)
        x_t = self.sample_noise(action_shape, device)

        state_in = state.to(torch.bfloat16)
        dt_a = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        t_a  = torch.tensor(1.0,              dtype=torch.float32, device=device)

        while t_a >= -dt_a / 2:
            v_t = self.denoise_step(state_in, prefix_pad, prefix_kv, x_t, t_a.expand(B))
            x_t = x_t + dt_a * v_t
            t_a = t_a + dt_a

        actions = x_t.float()[:, :, :self.action_q01.shape[0]]   # trim padding dims
        actions = self._unnormalize_action(actions)
        return actions, sg_main, sg_wrist, sg_state, horizon_pred

    @torch.no_grad()
    def sample_action(
        self,
        main_img:    torch.Tensor,   # (B, 3, H, W)
        wrist_img:   torch.Tensor,   # (B, 3, H, W)
        lang_tokens: torch.Tensor,   # (B, T) int64
        lang_mask:   torch.Tensor,   # (B, T) bool
        state:       torch.Tensor,   # (B, 8)
        num_steps:   int = 10,
    ) -> torch.Tensor:
        """
        PI0.5 action denoising only — no goal expert, no executor.
        Returns actions: (B, action_horizon, action_dim) float32
        """
        B      = main_img.shape[0]
        device = main_img.device

        prefix_kv, prefix_pad = self._prefix_kv_cache(main_img, wrist_img, lang_tokens, lang_mask)

        action_shape = (B, self.config.action_horizon, self.config.action_dim)
        x_t = self.sample_noise(action_shape, device)

        state_in = state.to(torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        t  = torch.tensor(1.0,              dtype=torch.float32, device=device)

        while t >= -dt / 2:
            v_t = self.denoise_step(state_in, prefix_pad, prefix_kv, x_t, t.expand(B))
            x_t = x_t + dt * v_t
            t = t + dt

        actions = x_t.float()[:, :, :self.action_q01.shape[0]]   # trim padding dims
        actions = self._unnormalize_action(actions)
        return actions
