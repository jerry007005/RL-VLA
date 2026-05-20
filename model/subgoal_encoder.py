"""
SubgoalAutoencoder (8-slot per view)
====================================
Multi-slot SAE that compresses SigLIP patch tokens into 2 * S * patch_dim latent.

Architecture
------------
  Encoder (per frame):
    input  : (B, 512, 2048)  — 256 main patches + 256 wrist patches (concat)
    spatial self-attn over 512 patches (cross-view exchange)
    split into main (256) + wrist (256) patches
    per-view learnable queries (S queries each) cross-attend → (B, S, 2048) per view
    concat: (B, 2*S, 2048)  e.g. S=8 → 16 slots × 2048

  Temporal (per slot, over T-frame window):
    self-attn over T frames per slot
    pool: center frame

  Output LayerNorm per slot → ~unit variance latent
    z: (B, 2*S, 2048) flattened to (B, 2*S*2048)
       main slots = z.view(B, 2, S, D)[:, 0] = z[:, :S*D].view(B, S, D)
       wrist slots = z[:, S*D:].view(B, S, D)

  Decoder:
    z (B, 2*S*D) → reshape (B, 2*S, D)
    512 learnable pos queries cross-attend to 2*S slots → (B, 512, 2048)

Training (Phase A):
    loss = MSE(decoder(encoder(patches)), patches)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + pre-norm FFN (residual around each)."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff      = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class _SpatialAttnBlock(nn.Module):
    """Pre-norm self-attention over spatial tokens + FFN (residual each)."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class _TemporalAttnBlock(nn.Module):
    """Pre-norm self-attention over temporal dim + FFN (residual each)."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SubgoalPatchDecoder(nn.Module):
    """
    (B, n_slots * patch_dim) → (B, n_patches, patch_dim).
    Default for 8-slot SAE: (B, 16 * 2048) → (B, 512, 2048).
    """

    def __init__(
        self,
        patch_dim: int = 2048,
        n_queries: int = 16,          # 2 views × slots_per_view
        n_patches: int = 512,
        n_heads:   int = 16,
        n_layers:  int = 2,
        ffn_mult:  int = 4,
    ):
        super().__init__()
        self._n_queries  = n_queries
        self._patch_dim  = patch_dim
        self.pos_queries = nn.Parameter(torch.randn(n_patches, patch_dim) * 0.02)
        self.layers      = nn.ModuleList([
            _CrossAttnBlock(patch_dim, n_heads, ffn_mult) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(patch_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, n_queries * patch_dim) → patches: (B, n_patches, patch_dim)"""
        B  = z.shape[0]
        kv = z.reshape(B, self._n_queries, self._patch_dim)
        q  = self.pos_queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, kv)
        return self.out_norm(q)


# ---------------------------------------------------------------------------
# Multi-slot Autoencoder
# ---------------------------------------------------------------------------

class SubgoalAutoencoder(nn.Module):
    """
    Multi-slot SAE: each view → S slots × D dim.

    Spatial: per-frame self-attn over 512 patches (cross-view info exchange).
    Split: first 256 patches → main view, second 256 → wrist view.
    Slot extraction: per-view learnable queries cross-attend to that view's patches.
                     Shared cross-attn block, separate query parameters per view.
    Temporal: per-slot self-attn over T frames; pool center frame.
    Output: per-slot LayerNorm.

    encode() input modes:
      (B, P, D)        single frame — temporal layers bypassed
      (B, T, P, D)     T-frame window (T == self.temporal_window)
    Output z: (B, 2*S*D) — main slots flat, then wrist slots flat.
              z[:, :S*D] = main slots,  z[:, S*D:] = wrist slots.
    """

    def __init__(
        self,
        patch_dim:        int = 2048,
        slots_per_view:   int = 8,
        n_patches:        int = 512,
        n_heads:          int = 16,
        enc_layers:       int = 2,
        dec_layers:       int = 2,
        ffn_mult:         int = 4,
        temporal_window:  int = 5,
        temporal_layers:  int = 2,
        # Legacy compat
        n_queries:        int = None,
        latent_dim:       int = None,
    ):
        super().__init__()
        if n_queries is not None:
            assert n_queries == 2 * slots_per_view, (
                f"n_queries={n_queries} but slots_per_view={slots_per_view} (expected 2*S)"
            )

        S = slots_per_view

        # Spatial self-attn over all 512 patches (cross-view info exchange)
        self.spatial_blocks = nn.ModuleList([
            _SpatialAttnBlock(patch_dim, n_heads, ffn_mult)
            for _ in range(enc_layers)
        ])
        self.spatial_norm = nn.LayerNorm(patch_dim)

        # Per-view learnable queries + shared cross-attn block
        self.main_queries  = nn.Parameter(torch.randn(S, patch_dim) * 0.02)
        self.wrist_queries = nn.Parameter(torch.randn(S, patch_dim) * 0.02)
        self.slot_cross_attn = _CrossAttnBlock(patch_dim, n_heads, ffn_mult)

        # Temporal self-attn (per slot, over T frames)
        self.temporal_blocks = nn.ModuleList([
            _TemporalAttnBlock(patch_dim, n_heads, ffn_mult)
            for _ in range(temporal_layers)
        ])
        self.temporal_pos_emb = nn.Parameter(
            torch.randn(temporal_window, patch_dim) * 0.02
        )

        # Per-slot LayerNorm at output: latent ~unit variance per dim
        self.out_norm = nn.LayerNorm(patch_dim)

        # Decoder reconstructs full patches from 2*S slots
        self.decoder = SubgoalPatchDecoder(
            patch_dim, n_queries=2 * S, n_patches=n_patches,
            n_heads=16, n_layers=dec_layers, ffn_mult=ffn_mult,
        )

        self.temporal_window = temporal_window
        self.patch_dim       = patch_dim
        self.n_patches       = n_patches
        self.slots_per_view  = S
        self.n_queries       = 2 * S
        self._latent_dim     = 2 * S * patch_dim

    # ------------------------------------------------------------------
    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    def split_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z: (B, 2*S*D)  →  (main_slots, wrist_slots) each (B, S, D).
        """
        B = z.shape[0]
        S = self.slots_per_view
        D = self.patch_dim
        z_view = z.reshape(B, 2, S, D)
        return z_view[:, 0], z_view[:, 1]

    # ------------------------------------------------------------------
    def _spatial_encode(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, P, D)  →  (B, 2*S, D) via spatial self-attn + per-view cross-attn queries.
        Assumes first P/2 patches = main view, second P/2 = wrist view.
        """
        x = patches
        for block in self.spatial_blocks:
            x = block(x)
        x = self.spatial_norm(x)

        half = x.shape[1] // 2
        main_patches  = x[:, :half]                              # (B, P/2, D)
        wrist_patches = x[:, half:]                              # (B, P/2, D)

        B = x.shape[0]
        main_q  = self.main_queries.unsqueeze(0).expand(B, -1, -1)   # (B, S, D)
        wrist_q = self.wrist_queries.unsqueeze(0).expand(B, -1, -1)  # (B, S, D)

        main_slots  = self.slot_cross_attn(main_q,  main_patches)    # (B, S, D)
        wrist_slots = self.slot_cross_attn(wrist_q, wrist_patches)   # (B, S, D)

        return torch.cat([main_slots, wrist_slots], dim=1)           # (B, 2*S, D)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches:
          (B, P, D)        single frame; bypass temporal layers
          (B, T, P, D)     temporal window of T frames

        Returns z (B, 2*S*D), per-slot LayerNorm applied.
        Layout: z[:, :S*D] = main slots flat,  z[:, S*D:] = wrist slots flat.
        """
        S = self.slots_per_view
        D = self.patch_dim
        n_slots = 2 * S

        if patches.dim() == 3:
            z = self._spatial_encode(patches)                # (B, 2*S, D)
            z = self.out_norm(z)                              # per-slot LayerNorm
            return z.reshape(z.shape[0], -1)                 # (B, 2*S*D)

        assert patches.dim() == 4, f"expected (B,T,P,D), got {tuple(patches.shape)}"
        B, T, P, _ = patches.shape
        assert T == self.temporal_window, f"T mismatch: got {T}, expected {self.temporal_window}"

        # Spatial per frame: (B*T, P, D) → (B*T, 2*S, D)
        z_per_frame = self._spatial_encode(patches.reshape(B * T, P, D))
        z_per_frame = z_per_frame.reshape(B, T, n_slots, D)               # (B, T, 2*S, D)

        # Temporal: per-slot self-attn over T axis
        z_temp = z_per_frame.permute(0, 2, 1, 3).reshape(B * n_slots, T, D)  # (B*2S, T, D)
        z_temp = z_temp + self.temporal_pos_emb.unsqueeze(0)
        for block in self.temporal_blocks:
            z_temp = block(z_temp)
        z_temp = z_temp.reshape(B, n_slots, T, D).permute(0, 2, 1, 3)        # (B, T, 2*S, D)

        # Pool: take center frame + per-slot LayerNorm
        center = T // 2
        z_out  = self.out_norm(z_temp[:, center])             # (B, 2*S, D)
        return z_out.reshape(B, n_slots * D)                  # (B, 2*S*D)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 2*S*D)  →  patches: (B, n_patches, patch_dim)"""
        return self.decoder(z)

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches: (B, T, P, D) or (B, P, D)
        Reconstructs center frame (or single frame); loss = MSE(recon, target).
        """
        z = self.encode(patches)
        recon = self.decode(z)
        if patches.dim() == 4:
            center = patches.shape[1] // 2
            target = patches[:, center]
        else:
            target = patches
        loss = F.mse_loss(recon, target)
        return z, recon, loss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SubgoalAutoencoder(slots_per_view=8).to(device)

    print(f"Total params : {model.n_params() / 1e6:.1f}M")
    print(f"Latent dim   : {model.latent_dim} (= {model.n_queries} slots × {model.patch_dim})")

    patches = torch.randn(4, 512, 2048, device=device)
    z, recon, loss = model(patches)

    print(f"Input  : {patches.shape}")
    print(f"z      : {z.shape}")
    print(f"recon  : {recon.shape}")
    print(f"loss   : {loss.item():.4f}")

    main_slots, wrist_slots = model.split_z(z)
    print(f"main_slots  : {main_slots.shape}")
    print(f"wrist_slots : {wrist_slots.shape}")

    # Temporal window test
    patches_t = torch.randn(2, 5, 512, 2048, device=device)
    z_t = model.encode(patches_t)
    print(f"Temporal z : {z_t.shape}")
