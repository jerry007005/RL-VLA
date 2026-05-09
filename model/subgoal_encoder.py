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

class SubgoalAutoencoder(nn.Module):
    """
    Encoder + Decoder for Phase-A training.

    forward() returns (z, recon, loss) for training.
    encode() returns z for latent caching / Phase-B target generation.

    The latent z splits cleanly:
        sg_main_emb  = z[:, :2048]
        sg_wrist_emb = z[:, 2048:]
    """

    def __init__(
        self,
        patch_dim:  int = 2048,
        n_queries:  int = 2,
        n_patches:  int = 512,
        n_heads:    int = 16,
        enc_layers: int = 2,
        dec_layers: int = 2,
        ffn_mult:   int = 4,
    ):
        super().__init__()
        self.encoder = SubgoalPatchEncoder(
            patch_dim, n_queries, n_heads, enc_layers, ffn_mult
        )
        self.decoder = SubgoalPatchDecoder(
            patch_dim, n_queries, n_patches, n_heads, dec_layers, ffn_mult
        )
        self._latent_dim = n_queries * patch_dim

    # ------------------------------------------------------------------
    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, 512, 2048)  →  z: (B, 4096)"""
        return self.encoder(patches)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 4096)  →  patches: (B, 512, 2048)"""
        return self.decoder(z)

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches : (B, 512, 2048)
        returns : z (B, 4096), recon (B, 512, 2048), loss scalar
        """
        z     = self.encode(patches)
        recon = self.decode(z)
        loss  = F.mse_loss(recon, patches)
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
