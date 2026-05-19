"""
SubgoalAutoencoder
==================
Cross-attention encoder/decoder that compresses SigLIP patch tokens into a
compact latent compatible with the executor's existing interface.

Architecture
------------
  Encoder:
    input  : (B, 512, 2048)  — 256 main patches + 256 wrist patches (concat)
    2 learnable queries cross-attend to 512 patch tokens
    output : (B, 4096)       — flatten of (B, 2, 2048)
               └─ sg_main_emb (B, 2048)  | sg_wrist_emb (B, 2048)

  Decoder:
    input  : (B, 4096)  →  reshape  →  (B, 2, 2048)
    512 learnable position queries cross-attend to 2 encoder tokens
    output : (B, 512, 2048)  — reconstructed patch tokens

Training (Phase A):
    loss = MSE(decoder(encoder(patches)), patches)

Downstream use (Phase B — PI0WithGoalExpert):
    z = encoder(patches).detach()   ← fixed target for flow matching
    split: sg_main = z[:, :2048], sg_wrist = z[:, 2048:]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
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
        # cross-attention
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + attn_out
        # FFN
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
        # x: (B, P, D)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SubgoalPatchEncoder(nn.Module):
    """
    (B, N_patches, patch_dim) → (B, n_queries * patch_dim)

    Default: (B, 512, 2048) → (B, 4096)
    """     

    def __init__(
        self,
        patch_dim: int = 2048,
        n_queries: int = 2,
        n_heads:   int = 16,
        n_layers:  int = 2,
        ffn_mult:  int = 4,
    ):
        super().__init__()
        self.queries  = nn.Parameter(torch.randn(n_queries, patch_dim) * 0.02)
        self.layers   = nn.ModuleList([
            _CrossAttnBlock(patch_dim, n_heads, ffn_mult) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(patch_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, 512, 2048)  →  z: (B, 4096)"""
        B = patches.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, 2, 2048)
        for layer in self.layers:
            q = layer(q, patches)
        return self.out_norm(q).reshape(B, -1)             # (B, 4096)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SubgoalPatchDecoder(nn.Module):
    """
    (B, n_queries * patch_dim) → (B, N_patches, patch_dim)

    Default: (B, 4096) → (B, 512, 2048)
    """

    def __init__(
        self,
        patch_dim: int = 2048,
        n_queries: int = 2,
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
        """z: (B, 4096)  →  patches: (B, 512, 2048)"""
        B  = z.shape[0]
        kv = z.reshape(B, self._n_queries, self._patch_dim)        # (B, 2, 2048)
        q  = self.pos_queries.unsqueeze(0).expand(B, -1, -1)       # (B, 512, 2048)
        for layer in self.layers:
            q = layer(q, kv)
        return self.out_norm(q)                                     # (B, 512, 2048)


# ---------------------------------------------------------------------------
# Combined autoencoder
# ---------------------------------------------------------------------------

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
        # x: (B, T, D)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class SubgoalAutoencoder(nn.Module):
    """
    Two-slot Spatial + Temporal SAE.

    Spatial: per-frame self-attention over patches + split-mean pool
             First P/2 patches → main slot, second P/2 → wrist slot.
             Structurally enforces main/wrist slot separation.
    Temporal: self-attention over T-frame window per slot.
    Pool: take center frame of temporally-attended slots.

    encode() input modes:
      (B, P, D)        single frame  — no temporal smoothing
      (B, T, P, D)     temporal window (T == self.temporal_window)
    Output z: (B, 2 * patch_dim) where 2 = (main, wrist).

    Latent z splits cleanly:  z[:, :patch_dim] = main slot,  z[:, patch_dim:] = wrist slot
    """

    def __init__(
        self,
        patch_dim:        int = 2048,
        n_queries:        int = 2,       # always 2 (main, wrist) — kept for backward compat
        n_patches:        int = 512,
        n_heads:          int = 16,
        enc_layers:       int = 2,
        dec_layers:       int = 2,
        ffn_mult:         int = 4,
        temporal_window:  int = 5,
        temporal_layers:  int = 2,
        # legacy
        latent_dim:       int = None,
    ):
        super().__init__()
        assert n_queries == 2, "SubgoalAutoencoder uses 2 slots (main, wrist)"
        # Spatial self-attn over patches (operates in patch_dim space, no down-project)
        self.spatial_blocks = nn.ModuleList([
            _SpatialAttnBlock(patch_dim, n_heads, ffn_mult)
            for _ in range(enc_layers)
        ])
        self.spatial_norm = nn.LayerNorm(patch_dim)
        # Temporal self-attn (per-slot, over T frames)
        self.temporal_blocks = nn.ModuleList([
            _TemporalAttnBlock(patch_dim, n_heads, ffn_mult)
            for _ in range(temporal_layers)
        ])
        self.temporal_pos_emb = nn.Parameter(
            torch.randn(temporal_window, patch_dim) * 0.02
        )
        # Output LayerNorm: makes latent ~unit variance per dim (per slot)
        # so downstream goal expert flow matching loss is on a sane scale.
        self.out_norm = nn.LayerNorm(patch_dim)
        # Decoder reconstructs full patches from 2-slot latent
        self.decoder = SubgoalPatchDecoder(
            patch_dim, n_queries=2, n_patches=n_patches,
            n_heads=16, n_layers=dec_layers, ffn_mult=ffn_mult,
        )

        self.temporal_window = temporal_window
        self.patch_dim       = patch_dim
        self.n_patches       = n_patches
        self.n_queries       = n_queries
        self._latent_dim     = n_queries * patch_dim

    # ------------------------------------------------------------------
    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    def _spatial_encode(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, P, D)  →  (B, 2, D) via self-attn + split-mean pool.
        Assumes first P/2 patches = main, second P/2 = wrist.
        """
        x = patches
        for block in self.spatial_blocks:
            x = block(x)
        x = self.spatial_norm(x)
        half = x.shape[1] // 2
        main_slot  = x[:, :half].mean(dim=1)     # (B, D)
        wrist_slot = x[:, half:].mean(dim=1)
        return torch.stack([main_slot, wrist_slot], dim=1)   # (B, 2, D)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches:
          (B, P, D)        — single frame; bypass temporal layers
          (B, T, P, D)     — temporal window of T frames
        Returns z (B, 2 * patch_dim), per-slot LayerNorm applied.
        """
        if patches.dim() == 3:
            z = self._spatial_encode(patches)                # (B, 2, D)
            z = self.out_norm(z)                              # per-slot LayerNorm
            return z.reshape(z.shape[0], -1)                 # (B, 2*D)

        assert patches.dim() == 4, f"expected (B,T,P,D), got {tuple(patches.shape)}"
        B, T, P, D = patches.shape
        assert T == self.temporal_window, f"T mismatch: got {T}, expected {self.temporal_window}"

        # Spatial per frame: (B*T, P, D) → (B*T, 2, D)
        z_per_frame = self._spatial_encode(patches.reshape(B * T, P, D))
        z_per_frame = z_per_frame.reshape(B, T, 2, D)         # (B, T, 2, D)

        # Temporal: per-slot self-attn over T axis
        z_temp = z_per_frame.permute(0, 2, 1, 3).reshape(B * 2, T, D)  # (B*2, T, D)
        z_temp = z_temp + self.temporal_pos_emb.unsqueeze(0)            # broadcast pos
        for block in self.temporal_blocks:
            z_temp = block(z_temp)
        z_temp = z_temp.reshape(B, 2, T, D).permute(0, 2, 1, 3)        # (B, T, 2, D)

        # Pool: take center frame + per-slot LayerNorm
        center = T // 2
        z_out  = self.out_norm(z_temp[:, center])             # (B, 2, D)
        return z_out.reshape(B, 2 * D)                        # (B, 2*D)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 2*patch_dim)  →  patches: (B, n_patches, patch_dim)"""
        return self.decoder(z)

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches: (B, T, P, D) or (B, P, D)
        Reconstructs the center frame (or single frame); loss = MSE(recon, target).
        Returns z (B, 2*D), recon (B, P, D), loss scalar.
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
    model  = SubgoalAutoencoder().to(device)

    print(f"Total params : {model.n_params() / 1e6:.1f}M")
    print(f"Latent dim   : {model.latent_dim}")

    patches = torch.randn(4, 512, 2048, device=device)
    z, recon, loss = model(patches)

    print(f"Input  : {patches.shape}")
    print(f"z      : {z.shape}")
    print(f"recon  : {recon.shape}")
    print(f"loss   : {loss.item():.4f}")

    sg_main  = z[:, :2048]
    sg_wrist = z[:, 2048:]
    print(f"sg_main  : {sg_main.shape}")
    print(f"sg_wrist : {sg_wrist.shape}")
