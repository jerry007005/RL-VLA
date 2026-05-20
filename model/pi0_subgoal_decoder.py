"""
PI0WithGoalExpert  (multi-slot, S = slots_per_view)
====================================================
Inherits PI0Pytorch and adds a flow-matching goal expert that generates
multi-slot subgoal features (S slots per view) directly compatible with
the Executor.

Architecture
------------
  Frozen PI0 backbone (inherited)
    _prefix_kv_cache():
      embed_image(main)  → (B, 256, 2048)
      embed_image(wrist) → (B, 256, 2048)
      embed_language()   → (B,  48, 2048)
      frozen PaliGemma LM → prefix KV cache

  Trainable goal expert (GemmaForCausalLM, bfloat16)
    Suffix layout (4S + 2 tokens), fully bidirectional:
      [0    .. S-1]      curr_main slots   (S tokens)   ← current obs conditioning
      [S    .. 2S-1]     curr_wrist slots  (S tokens)   ← current obs conditioning
      [2S]               curr_state                          ← current obs conditioning
      [2S+1 .. 3S]       sg_main slots (noisy)  (S tokens)
      [3S+1 .. 4S]       sg_wrist slots (noisy) (S tokens)
      [4S+1]             sg_state (noisy)

    Shared per-view projectors (applied per slot):
      curr_main_in_proj : (S, 2048) → (S, W)
      sg_main_in_proj   : same shape
      (same for wrist + state)

    Timestep → AdaRMS conditioning, all suffix tokens see each other + prefix KV.

  Output heads (applied only to noisy sg tokens)
    sg_main_out_proj  : (B, S, W) → (B, S, 2048)
    sg_wrist_out_proj : (B, S, W) → (B, S, 2048)
    sg_state_out_proj : (B, W)    → (B, 8)
    Horizon head      : (B, W)    → (B, 1)   from curr_main slot 0 token

Training: flow matching (same as PI0)
  noisy = t * noise + (1-t) * target
  predict u = noise - target
  loss = MSE over (sg_main, sg_wrist, sg_state)

Inference: sample_goal(curr_z) returns
  sg_main : (B, S, 2048) float32
  sg_wrist: (B, S, 2048) float32
  sg_state: (B, 8)       float32 raw env scale
  horizon : (B,)         float32
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
    PI0 + flow-matching goal expert with S slots per view.

    sample_goal(curr_z) returns (sg_main, sg_wrist, sg_state, horizon) where
        sg_main  : (B, S, patch_dim)
        sg_wrist : (B, S, patch_dim)
        sg_state : (B, 8) raw env scale
    These plug directly into the Executor.
    """

    def __init__(
        self,
        config,
        patch_dim:      int  = 2048,
        proprio_dim:    int  = 8,
        slots_per_view: int  = 8,
        freeze_pi0:     bool = True,
        expert_variant: str  = "gemma_300m",
        norm_stats_path: str = None,
    ):
        super().__init__(config=config)

        self.patch_dim      = patch_dim
        self.proprio_dim    = proprio_dim
        self.slots_per_view = slots_per_view

        S = slots_per_view
        # Suffix token offsets (within suffix only, not including prefix length)
        self.CURR_MAIN_OFF  = 0
        self.CURR_WRIST_OFF = S
        self.CURR_STATE_IDX = 2 * S
        self.SG_MAIN_OFF    = 2 * S + 1
        self.SG_WRIST_OFF   = 3 * S + 1
        self.SG_STATE_IDX   = 4 * S + 1
        self.SUFFIX_LEN     = 4 * S + 2

        # Action + state quantile norm stats from pi0.5
        import json as _json
        ns = _json.loads(Path(norm_stats_path).read_text())["norm_stats"]
        self.register_buffer("action_q01", torch.tensor(ns["actions"]["q01"], dtype=torch.float32))
        self.register_buffer("action_q99", torch.tensor(ns["actions"]["q99"], dtype=torch.float32))
        self.register_buffer("state_q01",  torch.tensor(ns["state"]["q01"],   dtype=torch.float32))
        self.register_buffer("state_q99",  torch.tensor(ns["state"]["q99"],   dtype=torch.float32))
        print(f"[PI0WithGoalExpert] Loaded action+state norm_stats from {norm_stats_path}")

        if freeze_pi0:
            for p in self.parameters():
                p.requires_grad_(False)

        expert_cfg   = _gemma.get_config(expert_variant)
        W            = expert_cfg.width
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
        self.goal_expert.lm_head = None

        # Timestep MLP for AdaRMS
        self.goal_time_mlp_in  = nn.Linear(W, W, dtype=torch.bfloat16)
        self.goal_time_mlp_out = nn.Linear(W, W, dtype=torch.bfloat16)

        # Conditioning projections — applied per-slot via broadcast.
        self.curr_main_in_proj  = nn.Linear(patch_dim, W, dtype=torch.bfloat16)
        self.curr_wrist_in_proj = nn.Linear(patch_dim, W, dtype=torch.bfloat16)
        self.curr_state_in_proj = nn.Linear(proprio_dim, W, dtype=torch.bfloat16)

        # Noisy target input projections (same weights shared across S slots)
        self.sg_main_in_proj  = nn.Linear(patch_dim, W, dtype=torch.bfloat16)
        self.sg_wrist_in_proj = nn.Linear(patch_dim, W, dtype=torch.bfloat16)
        self.sg_state_in_proj = nn.Linear(proprio_dim, W, dtype=torch.bfloat16)

        # Output projections. 2-layer MLP with SiLU to break rank bottleneck (W < patch_dim).
        self.sg_main_out_proj = nn.Sequential(
            nn.Linear(W, patch_dim, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(patch_dim, patch_dim, dtype=torch.bfloat16),
        )
        self.sg_wrist_out_proj = nn.Sequential(
            nn.Linear(W, patch_dim, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(patch_dim, patch_dim, dtype=torch.bfloat16),
        )
        self.sg_state_out_proj = nn.Linear(W, proprio_dim, dtype=torch.bfloat16)

        # Horizon head: predicts (sg_idx - curr_idx) / MAX_HORIZON from token 0 (curr_main slot 0).
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
    # Action / state norm helpers
    # ------------------------------------------------------------------

    def _normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_q01) / (self.action_q99 - self.action_q01) * 2.0 - 1.0

    def _unnormalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a + 1.0) / 2.0 * (self.action_q99 - self.action_q01) + self.action_q01

    def _normalize_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.state_q01) / (self.state_q99 - self.state_q01) * 2.0 - 1.0

    def _unnormalize_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s + 1.0) / 2.0 * (self.state_q99 - self.state_q01) + self.state_q01

    # ------------------------------------------------------------------
    # Checkpoint helpers (save/load only trainable + buffers)
    # ------------------------------------------------------------------

    def trainable_state_dict(self) -> dict:
        full = self.state_dict()
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        keep |= {n for n, _ in self.named_buffers()}
        return {k: v for k, v in full.items() if k in keep}

    def load_trainable_state(self, state: dict) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        buffer_names    = {n for n, _ in self.named_buffers()}
        expected_keys   = trainable_names | buffer_names
        real_missing    = [k for k in missing if k in expected_keys]
        if real_missing:
            raise RuntimeError(f"Missing trainable keys: {real_missing[:5]}...")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in ckpt: {unexpected[:5]}...")

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

        embs      = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
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
    # Goal suffix: (4S+2) tokens
    # ------------------------------------------------------------------

    def _embed_goal_suffix(
        self,
        noisy_main:  torch.Tensor,   # (B, S, 2048) float32
        noisy_wrist: torch.Tensor,   # (B, S, 2048) float32
        noisy_state: torch.Tensor,   # (B, 8) float32 normalized
        curr_main:   torch.Tensor,   # (B, S, 2048) float32
        curr_wrist:  torch.Tensor,   # (B, S, 2048) float32
        curr_state:  torch.Tensor,   # (B, 8) float32 normalized
        time:        torch.Tensor,   # (B,) float32
    ):
        B = noisy_main.shape[0]
        device = noisy_main.device

        # Conditioning slot tokens (per-slot Linear via broadcast)
        tok_curr_main  = self.curr_main_in_proj(curr_main.to(torch.bfloat16))     # (B, S, W)
        tok_curr_wrist = self.curr_wrist_in_proj(curr_wrist.to(torch.bfloat16))   # (B, S, W)
        tok_curr_state = self.curr_state_in_proj(curr_state.to(torch.bfloat16))   # (B, W)
        tok_curr_state = tok_curr_state.unsqueeze(1)                              # (B, 1, W)

        # Noisy target slot tokens
        tok_sg_main  = self.sg_main_in_proj(noisy_main.to(torch.bfloat16))    # (B, S, W)
        tok_sg_wrist = self.sg_wrist_in_proj(noisy_wrist.to(torch.bfloat16))  # (B, S, W)
        tok_sg_state = self.sg_state_in_proj(noisy_state.to(torch.bfloat16))  # (B, W)
        tok_sg_state = tok_sg_state.unsqueeze(1)                              # (B, 1, W)

        # Concat: [curr_main(S), curr_wrist(S), curr_state(1), sg_main(S), sg_wrist(S), sg_state(1)]
        suffix_embs = torch.cat(
            [tok_curr_main, tok_curr_wrist, tok_curr_state,
             tok_sg_main, tok_sg_wrist, tok_sg_state], dim=1
        )  # (B, 4S+2, W)

        time_emb = create_sinusoidal_pos_embedding(
            time, self._W, min_period=4e-3, max_period=4.0, device=device
        ).to(torch.bfloat16)
        adarms_cond = F.silu(self.goal_time_mlp_out(self.goal_time_mlp_in(time_emb)))

        L = self.SUFFIX_LEN
        pad_masks = torch.ones(B, L,  dtype=torch.bool,    device=device)
        att_masks = torch.zeros(B, L, dtype=torch.float32, device=device)
        return suffix_embs, pad_masks, att_masks, adarms_cond

    # ------------------------------------------------------------------
    # One denoising step
    # ------------------------------------------------------------------

    def _denoise_step(
        self,
        noisy_main:  torch.Tensor,   # (B, S, 2048)
        noisy_wrist: torch.Tensor,   # (B, S, 2048)
        noisy_state: torch.Tensor,   # (B, 8) normalized
        curr_main:   torch.Tensor,   # (B, S, 2048)
        curr_wrist:  torch.Tensor,   # (B, S, 2048)
        curr_state:  torch.Tensor,   # (B, 8) normalized
        time:        torch.Tensor,   # (B,)
        prefix_kv,
        prefix_pad:  torch.Tensor,
    ):
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self._embed_goal_suffix(
            noisy_main, noisy_wrist, noisy_state, curr_main, curr_wrist, curr_state, time
        )
        B, prefix_len = prefix_pad.shape
        L = self.SUFFIX_LEN

        prefix_pad_2d = prefix_pad[:, None, :].expand(B, L, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
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
        ).last_hidden_state  # (B, 4S+2, W) bfloat16

        S = self.slots_per_view
        # Slice sg slot outputs
        sg_main_out  = out[:, self.SG_MAIN_OFF  : self.SG_MAIN_OFF  + S]   # (B, S, W)
        sg_wrist_out = out[:, self.SG_WRIST_OFF : self.SG_WRIST_OFF + S]   # (B, S, W)
        sg_state_out = out[:, self.SG_STATE_IDX]                            # (B, W)

        # Project + residual on visual (per-slot residual via /t broadcast)
        t_col3 = time[:, None, None].float()                               # (B, 1, 1)
        v_main  = self.sg_main_out_proj(sg_main_out).float()  + noisy_main.float()  / t_col3
        v_wrist = self.sg_wrist_out_proj(sg_wrist_out).float() + noisy_wrist.float() / t_col3
        v_state = self.sg_state_out_proj(sg_state_out).float()             # (B, 8)

        # Horizon: from curr_main slot 0 token (suffix index 0)
        horizon = (torch.sigmoid(self.horizon_head(out[:, 0])).float().squeeze(-1)
                   * self.MAX_HORIZON)
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
        curr_main:   torch.Tensor,   # (B, S, 2048) float32
        curr_wrist:  torch.Tensor,   # (B, S, 2048) float32
        curr_state:  torch.Tensor,   # (B, 8) float32 raw
        sg_main:     torch.Tensor,   # (B, S, 2048) float32
        sg_wrist:    torch.Tensor,   # (B, S, 2048) float32
        sg_state:    torch.Tensor,   # (B, 8) float32 raw
        horizon:     torch.Tensor | None = None,
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

        sg_state_n   = self._normalize_state(sg_state)
        curr_state_n = self._normalize_state(curr_state)

        # Visual noise broadcasts: time (B,) → (B, 1, 1)
        t3 = time[:, None, None]
        t1 = time[:, None]
        noisy_main  = t3 * noise_main  + (1 - t3) * sg_main
        noisy_wrist = t3 * noise_wrist + (1 - t3) * sg_wrist
        noisy_state = t1 * noise_state + (1 - t1) * sg_state_n
        u_main  = noise_main  - sg_main
        u_wrist = noise_wrist - sg_wrist
        u_state = noise_state - sg_state_n

        prefix_kv, prefix_pad = self._prefix_kv_cache(
            main_img, wrist_img, lang_tokens, lang_mask
        )

        v_main, v_wrist, v_state, horizon_pred = self._denoise_step(
            noisy_main, noisy_wrist, noisy_state,
            curr_main, curr_wrist, curr_state_n,
            time, prefix_kv, prefix_pad
        )

        loss_main  = F.mse_loss(v_main,  u_main)
        loss_wrist = F.mse_loss(v_wrist, u_wrist)
        loss_state = F.mse_loss(v_state, u_state)

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
                curr_main, curr_wrist, curr_state,
                sg_main, sg_wrist, sg_state, **kwargs):
        return self.forward_goal(main_img, wrist_img, lang_tokens, lang_mask,
                                 curr_main, curr_wrist, curr_state,
                                 sg_main, sg_wrist, sg_state, **kwargs)

    # ------------------------------------------------------------------
    # Helper: split flat curr_z into (curr_main, curr_wrist) per-slot tensors
    # ------------------------------------------------------------------

    def _split_curr_z(self, curr_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        curr_z: (B, 2*S*patch_dim) flat SAE latent.
        Returns (curr_main, curr_wrist) each (B, S, patch_dim).
        Layout matches SubgoalAutoencoder.split_z.
        """
        B = curr_z.shape[0]
        S = self.slots_per_view
        D = self.patch_dim
        z_view = curr_z.reshape(B, 2, S, D)
        return z_view[:, 0].float(), z_view[:, 1].float()

    # ------------------------------------------------------------------
    # Inference: Euler sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_goal(
        self,
        main_img:    torch.Tensor,   # (B, 3, H, W)
        wrist_img:   torch.Tensor,   # (B, 3, H, W)
        lang_tokens: torch.Tensor,   # (B, T) int64
        lang_mask:   torch.Tensor,   # (B, T) bool
        curr_z:      torch.Tensor,   # (B, 2*S*patch_dim) SAE latent of current frame
        curr_state:  torch.Tensor,   # (B, 8)
        num_steps:   int = 10,
    ) -> tuple:
        """
        Returns:
          sg_main      : (B, S, 2048) float32
          sg_wrist     : (B, S, 2048) float32
          sg_state     : (B, 8)       float32 raw env scale
          horizon_pred : (B,)         float32
        """
        B      = main_img.shape[0]
        device = main_img.device
        S      = self.slots_per_view
        D      = self.patch_dim

        curr_main, curr_wrist = self._split_curr_z(curr_z)         # (B, S, D) each
        curr_state_n = self._normalize_state(curr_state)

        prefix_kv, prefix_pad = self._prefix_kv_cache(
            main_img, wrist_img, lang_tokens, lang_mask
        )

        x_main  = self._sample_noise((B, S, D),                device)
        x_wrist = self._sample_noise((B, S, D),                device)
        x_state = self._sample_noise((B, self.proprio_dim),    device)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        t  = torch.tensor(1.0,              dtype=torch.float32, device=device)

        horizon_pred = None
        while t >= -dt / 2:
            v_main, v_wrist, v_state, horizon_pred = self._denoise_step(
                x_main, x_wrist, x_state,
                curr_main, curr_wrist, curr_state_n,
                t.expand(B), prefix_kv, prefix_pad
            )
            x_main  = x_main  + dt * v_main
            x_wrist = x_wrist + dt * v_wrist
            x_state = x_state + dt * v_state
            t = t + dt

        return x_main, x_wrist, self._unnormalize_state(x_state), horizon_pred

    @torch.no_grad()
    def sample_all(
        self,
        main_img:    torch.Tensor,
        wrist_img:   torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask:   torch.Tensor,
        curr_z:      torch.Tensor,
        state:       torch.Tensor,
        num_steps:   int = 10,
    ) -> tuple:
        """
        Returns (actions, sg_main, sg_wrist, sg_state, horizon_pred)
          actions  : (B, action_horizon, action_dim) float32 raw env scale
          sg_main  : (B, S, 2048) float32
          sg_wrist : (B, S, 2048) float32
          sg_state : (B, 8)       float32 raw env scale
        """
        B      = main_img.shape[0]
        device = main_img.device
        S      = self.slots_per_view
        D      = self.patch_dim

        curr_main, curr_wrist = self._split_curr_z(curr_z)
        curr_state_n = self._normalize_state(state)

        prefix_kv, prefix_pad = self._prefix_kv_cache(main_img, wrist_img, lang_tokens, lang_mask)

        # ---- Goal expert denoising ----
        x_main  = self._sample_noise((B, S, D),                device)
        x_wrist = self._sample_noise((B, S, D),                device)
        x_state = self._sample_noise((B, self.proprio_dim),    device)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        t  = torch.tensor(1.0,              dtype=torch.float32, device=device)

        horizon_pred = None
        while t >= -dt / 2:
            v_main, v_wrist, v_state, horizon_pred = self._denoise_step(
                x_main, x_wrist, x_state,
                curr_main, curr_wrist, curr_state_n,
                t.expand(B), prefix_kv, prefix_pad,
            )
            x_main  = x_main  + dt * v_main
            x_wrist = x_wrist + dt * v_wrist
            x_state = x_state + dt * v_state
            t = t + dt

        sg_main  = x_main
        sg_wrist = x_wrist
        sg_state = self._unnormalize_state(x_state)

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

        actions = x_t.float()[:, :, :self.action_q01.shape[0]]
        actions = self._unnormalize_action(actions)
        return actions, sg_main, sg_wrist, sg_state, horizon_pred

    @torch.no_grad()
    def sample_action(
        self,
        main_img:    torch.Tensor,
        wrist_img:   torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask:   torch.Tensor,
        state:       torch.Tensor,
        num_steps:   int = 10,
    ) -> torch.Tensor:
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

        actions = x_t.float()[:, :, :self.action_q01.shape[0]]
        actions = self._unnormalize_action(actions)
        return actions
