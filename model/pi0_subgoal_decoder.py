"""
PI0WithGoalExpert
=================
Inherits PI0Pytorch and adds a flow-matching goal expert that generates
mean-pooled subgoal features directly compatible with the Executor.

Architecture
------------
  Frozen PI0 backbone (inherited)
    _prefix_kv_cache():
      embed_image(main)  → (B, 256, 2048)  ┐
      embed_image(wrist) → (B, 256, 2048)  ├→ concat → (B, 560, 2048)
      embed_language()   → (B,  48, 2048)  ┘
      frozen PaliGemma LM → prefix KV cache

  Trainable goal expert (GemmaForCausalLM, gemma_300m, bfloat16)
    Suffix: 3 tokens, fully bidirectional
      token 0: sg_main_in_proj( noisy_sg_main  )   (B, W)
      token 1: sg_wrist_in_proj( noisy_sg_wrist )  (B, W)
      token 2: sg_state_in_proj( noisy_sg_state )  (B, W)
    Timestep → AdaRMS conditioning
    All 3 tokens attend to each other + full prefix KV

  Output heads
    sg_main_out_proj  : (B, W) → (B, 2048)
    sg_wrist_out_proj : (B, W) → (B, 2048)
    sg_state_out_proj : (B, W) → (B, 8)

Training: flow matching (same as PI0)
  noisy = t * noise + (1-t) * target
  predict u = noise - target
  loss = MSE over (sg_main, sg_wrist, sg_state)

Inference: sample_goal() → (sg_main_emb, sg_wrist_emb, sg_state) float32
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

    sample_goal() returns (sg_main_emb, sg_wrist_emb, sg_state) that plug
    directly into the Executor:
        imgs = torch.stack([curr_main, curr_wrist, sg_main_emb, sg_wrist_emb], dim=1)
        subgoal_proprio = sg_state
    """

    def __init__(
        self,
        config,
        patch_dim:    int  = 2048,
        proprio_dim:  int  = 8,
        freeze_pi0:   bool = True,
        expert_variant: str = "gemma_300m",
    ):
        super().__init__(config=config)

        self.patch_dim   = patch_dim
        self.proprio_dim = proprio_dim

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

        # Input projections  (noisy targets → expert width)
        self.sg_main_in_proj  = nn.Linear(patch_dim,   W, dtype=torch.bfloat16)
        self.sg_wrist_in_proj = nn.Linear(patch_dim,   W, dtype=torch.bfloat16)
        self.sg_state_in_proj = nn.Linear(proprio_dim, W, dtype=torch.bfloat16)

        # Output projections (expert hidden → original dims)
        self.sg_main_out_proj  = nn.Linear(W, patch_dim,   dtype=torch.bfloat16)
        self.sg_wrist_out_proj = nn.Linear(W, patch_dim,   dtype=torch.bfloat16)
        self.sg_state_out_proj = nn.Linear(W, proprio_dim, dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Noise / time helpers
    # ------------------------------------------------------------------

    def _sample_noise(self, shape, device):
        return torch.randn(*shape, dtype=torch.float32, device=device)

    def _sample_time(self, B, device):
        t = sample_beta(1.5, 1.0, B, device)
        return (t * 0.999 + 0.001).to(dtype=torch.float32, device=device)

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
        # all-zero att_masks → fully bidirectional prefix attention
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
    # Goal suffix: 3 noisy tokens → suffix embeddings
    # ------------------------------------------------------------------

    def _embed_goal_suffix(
        self,
        noisy_main:  torch.Tensor,   # (B, 2048) float32
        noisy_wrist: torch.Tensor,   # (B, 2048) float32
        noisy_state: torch.Tensor,   # (B, 8)    float32
        time:        torch.Tensor,   # (B,)      float32
    ):
        B      = noisy_main.shape[0]
        device = noisy_main.device

        tok_main  = self.sg_main_in_proj(noisy_main.to(torch.bfloat16))   # (B, W)
        tok_wrist = self.sg_wrist_in_proj(noisy_wrist.to(torch.bfloat16)) # (B, W)
        tok_state = self.sg_state_in_proj(noisy_state.to(torch.bfloat16)) # (B, W)
        suffix_embs = torch.stack([tok_main, tok_wrist, tok_state], dim=1) # (B, 3, W)

        time_emb = create_sinusoidal_pos_embedding(
            time, self._W, min_period=4e-3, max_period=4.0, device=device
        ).to(torch.bfloat16)
        adarms_cond = F.silu(self.goal_time_mlp_out(self.goal_time_mlp_in(time_emb)))  # (B, W)

        # all-zero att_masks → 3 tokens attend to each other fully (bidirectional)
        pad_masks = torch.ones(B, 3,  dtype=torch.bool,    device=device)
        att_masks = torch.zeros(B, 3, dtype=torch.float32, device=device)
        return suffix_embs, pad_masks, att_masks, adarms_cond

    # ------------------------------------------------------------------
    # One denoising step
    # ------------------------------------------------------------------

    def _denoise_step(
        self,
        noisy_main:  torch.Tensor,
        noisy_wrist: torch.Tensor,
        noisy_state: torch.Tensor,
        time:        torch.Tensor,
        prefix_kv,
        prefix_pad:  torch.Tensor,
    ):
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self._embed_goal_suffix(
            noisy_main, noisy_wrist, noisy_state, time
        )
        B, prefix_len = prefix_pad.shape

        prefix_pad_2d = prefix_pad[:, None, :].expand(B, 3, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)  # (B, 3, prefix+3)
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
        ).last_hidden_state  # (B, 3, W) bfloat16

        v_main  = self.sg_main_out_proj(out[:, 0]).float()   # (B, 2048)
        v_wrist = self.sg_wrist_out_proj(out[:, 1]).float()  # (B, 2048)
        v_state = self.sg_state_out_proj(out[:, 2]).float()  # (B, 8)
        return v_main, v_wrist, v_state

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_goal(
        self,
        main_img:    torch.Tensor,   # (B, 3, H, W) float32
        wrist_img:   torch.Tensor,   # (B, 3, H, W) float32
        lang_tokens: torch.Tensor,   # (B, T) int64
        lang_mask:   torch.Tensor,   # (B, T) bool
        sg_main:     torch.Tensor,   # (B, 2048) float32  ← mean-pooled SigLIP target
        sg_wrist:    torch.Tensor,   # (B, 2048) float32  ← mean-pooled SigLIP target
        sg_state:    torch.Tensor,   # (B, 8)    float32  ← robot state target
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

        t = time[:, None]
        noisy_main  = t * noise_main  + (1 - t) * sg_main
        noisy_wrist = t * noise_wrist + (1 - t) * sg_wrist
        noisy_state = t * noise_state + (1 - t) * sg_state
        u_main  = noise_main  - sg_main
        u_wrist = noise_wrist - sg_wrist
        u_state = noise_state - sg_state

        with torch.no_grad():
            prefix_kv, prefix_pad = self._prefix_kv_cache(
                main_img, wrist_img, lang_tokens, lang_mask
            )

        v_main, v_wrist, v_state = self._denoise_step(
            noisy_main, noisy_wrist, noisy_state, time, prefix_kv, prefix_pad
        )

        loss_main  = F.mse_loss(v_main,  u_main)
        loss_wrist = F.mse_loss(v_wrist, u_wrist)
        loss_state = F.mse_loss(v_state, u_state)
        loss = loss_main + loss_wrist + loss_state

        return loss, {
            "loss/total":      loss.item(),
            "loss/sg_main":    loss_main.item(),
            "loss/sg_wrist":   loss_wrist.item(),
            "loss/sg_state":   loss_state.item(),
        }

    def forward(self, main_img, wrist_img, lang_tokens, lang_mask,
                sg_main, sg_wrist, sg_state, **kwargs):
        """Standard forward for DDP compatibility — delegates to forward_goal."""
        return self.forward_goal(main_img, wrist_img, lang_tokens, lang_mask,
                                 sg_main, sg_wrist, sg_state, **kwargs)

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

        prefix_kv, prefix_pad = self._prefix_kv_cache(
            main_img, wrist_img, lang_tokens, lang_mask
        )

        x_main  = self._sample_noise((B, self.patch_dim),   device)
        x_wrist = self._sample_noise((B, self.patch_dim),   device)
        x_state = self._sample_noise((B, self.proprio_dim), device)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        t  = torch.tensor(1.0,              dtype=torch.float32, device=device)

        while t >= -dt / 2:
            v_main, v_wrist, v_state = self._denoise_step(
                x_main, x_wrist, x_state, t.expand(B), prefix_kv, prefix_pad
            )
            x_main  = x_main  + dt * v_main
            x_wrist = x_wrist + dt * v_wrist
            x_state = x_state + dt * v_state
            t = t + dt

        return x_main, x_wrist, x_state
