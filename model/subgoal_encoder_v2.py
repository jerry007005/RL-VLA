"""
SubgoalEncoderV2
================
Combines main + wrist patches into a single z (2048,) via one learnable query.

Training objective: contrastive + reconstruction (jointly).
  Contrastive: in-batch pairwise spatial contrastive loss (frames closer in space → higher similarity)
  Reconstruction: z → MLP → reconstruct mean normalized patch (forces z to retain visual content)

The reconstruction loss is critical for the goal expert to generate meaningful subgoals.
Without it, z only encodes "how similar frames are" but loses visual detail needed for generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class SubgoalEncoderV2(nn.Module):
    """
    encode(patches) → z (B, 2048)

    forward(patches) → z, loss, info
      patches: (B, 512, 2048) — random batch of frames, any episode/task
      loss: contrastive + rec_weight * reconstruction
    """

    def __init__(
        self,
        patch_dim:    int   = 2048,
        n_heads:      int   = 16,
        n_layers:     int   = 2,
        ffn_mult:     int   = 4,
        tau:          float = 0.5,
        change_scale: float = 100.0,
        rec_weight:   float = 1.0,    # weight for reconstruction loss
    ):
        super().__init__()
        self.tau          = tau
        self.change_scale = change_scale
        self.rec_weight   = rec_weight

        self.query = nn.Parameter(torch.randn(1, patch_dim) * 0.02)
        self.layers = nn.ModuleList([
            _CrossAttnBlock(patch_dim, n_heads, ffn_mult) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(patch_dim)

        # Reconstruction head: z → mean normalized patch
        self.rec_head = nn.Sequential(
            nn.Linear(patch_dim, patch_dim * 2),
            nn.GELU(),
            nn.Linear(patch_dim * 2, patch_dim),
        )

    # ------------------------------------------------------------------
    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, 512, 2048)  →  z: (B, 2048)"""
        B = patches.shape[0]
        q = self.query.unsqueeze(0).expand(B, -1, -1)   # (B, 1, 2048)
        for layer in self.layers:
            q = layer(q, patches)
        return self.out_norm(q).squeeze(1)               # (B, 2048)

    # ------------------------------------------------------------------
    @staticmethod
    def _mean_feat(patches: torch.Tensor) -> torch.Tensor:
        """patches (B, 512, 2048) → normalized mean feature (B, 2048)"""
        mean = F.normalize(patches, dim=-1).mean(dim=1)  # (B, 2048)
        return F.normalize(mean, dim=-1)

    @staticmethod
    def _pairwise_change(mean_feat: torch.Tensor) -> torch.Tensor:
        """mean_feat (B, 2048) normalized → cosine distance (B, B)"""
        cos_sim = mean_feat @ mean_feat.T   # (B, B)
        return 1.0 - cos_sim               # (B, B)

    # ------------------------------------------------------------------
    def forward(self, patches: torch.Tensor) -> tuple:
        """
        patches : (B, 512, 2048)
        returns : z (B, 2048), loss scalar, info dict
        """
        B = patches.shape[0]

        z = self.encode(patches)   # (B, 2048)

        # ---- Contrastive loss ----
        with torch.no_grad():
            mean_feat  = self._mean_feat(patches)           # (B, 2048)
            change     = self._pairwise_change(mean_feat)   # (B, B)

        target_sim = torch.exp(-change * self.change_scale / self.tau)  # (B, B)
        z_norm     = F.normalize(z, dim=-1)
        pred_sim   = z_norm @ z_norm.T                                   # (B, B)

        mask = torch.triu(torch.ones(B, B, device=z.device, dtype=torch.bool), diagonal=1)
        loss_contrast = F.mse_loss(pred_sim[mask], target_sim[mask])

        # ---- Reconstruction loss: z → mean normalized patch ----
        recon      = F.normalize(self.rec_head(z), dim=-1)  # (B, 2048)
        loss_rec   = F.mse_loss(recon, mean_feat)

        loss = loss_contrast + self.rec_weight * loss_rec

        info = {
            "loss_contrast":   loss_contrast.item(),
            "loss_rec":        loss_rec.item(),
            "change_mean":     change[mask].mean().item(),
            "target_sim_mean": target_sim[mask].mean().item(),
            "pred_sim_mean":   pred_sim[mask].mean().item(),
        }
        return z, loss, info

    # ------------------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SubgoalEncoderV2().to(device)
    print(f"Params: {model.n_params() / 1e6:.1f}M")

    patches = torch.randn(8, 512, 2048, device=device)
    z, loss, info = model(patches)
    print(f"z: {z.shape}  loss: {loss.item():.4f}")
    print(f"info: {info}")
