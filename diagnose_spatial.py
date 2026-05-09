"""
Spatial information diagnostic: SubgoalAutoencoder encoder vs mean-pooling.

Tests
-----
1. Attention maps  — where do the 2 encoder queries attend in the 16×16 patch grid?
   Saved to: spatial_attention.png

2. Spatial sensitivity curve — cos_sim as a function of temporal distance.
   If encoder curve < mean-pool curve → encoder is more spatially discriminative.
   Saved to: spatial_sensitivity.png

Run: python diagnose_spatial.py
"""

import os, sys, json, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "openpi" / "src"))

from model.subgoal_encoder import SubgoalAutoencoder, SubgoalPatchEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBGOAL_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/subgoal_decoder_cache"
PI05_CKPT_DIR     = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"
ENCODER_CKPT_PATH = "./checkpoints/subgoal_encoder/checkpoint.pt"

PATCH_DIM   = 2048
N_PATCHES   = 512
IMG_SIZE    = 224
PATCH_GRID  = 16        # 224 / 14 (PaliGemma patch size) = 16 patches per side
ENCODE_BS   = 16        # batch size for backbone inference
N_EPISODES  = 5        # episodes to sample for spatial sensitivity test
VIS_FRAMES  = 4        # frames to visualize attention maps
TEMPORAL_DISTS = [1, 5, 10, 20, 50]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_backbone():
    from openpi.training import config as _cfg
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch

    print("Loading PI0 backbone ...")
    cfg = _cfg.get_config("pi05_libero")
    model = PI0Pytorch(config=cfg.model)
    safetensors.torch.load_model(
        model, os.path.join(PI05_CKPT_DIR, "model.safetensors"), strict=False
    )
    backbone = model.paligemma_with_expert
    backbone.eval().to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad_(False)
    print("  Backbone loaded.")
    return backbone


def load_encoder():
    ae = SubgoalAutoencoder(
        patch_dim=PATCH_DIM, n_queries=2, n_patches=N_PATCHES,
        n_heads=16, enc_layers=2, dec_layers=2, ffn_mult=4,
    )
    ckpt = torch.load(ENCODER_CKPT_PATH, map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval().to(DEVICE)
    for p in ae.parameters():
        p.requires_grad_(False)
    print("  SubgoalAutoencoder loaded.")
    return ae

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess(imgs_uint8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, 224, 224) float32 in [-1, 1]."""
    from openpi_client import image_tools
    out = []
    for img in imgs_uint8:
        r = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
        )
        t = torch.from_numpy(r).float() / 255.0
        out.append((t * 2.0 - 1.0).permute(2, 0, 1))
    return torch.stack(out)

# ---------------------------------------------------------------------------
# Patch extraction (no pooling)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_patches(backbone, main_imgs, wrist_imgs):
    """
    main_imgs, wrist_imgs: list of (H, W, 3) uint8
    Returns: patches (N, 512, 2048), mean_pool_main (N, 2048), mean_pool_wrist (N, 2048)
    """
    main_t  = preprocess(np.stack(main_imgs)).to(DEVICE)
    wrist_t = preprocess(np.stack(wrist_imgs)).to(DEVICE)
    mp  = backbone.embed_image(main_t).float()    # (N, 256, 2048)
    wp  = backbone.embed_image(wrist_t).float()   # (N, 256, 2048)
    patches = torch.cat([mp, wp], dim=1)          # (N, 512, 2048)
    mean_pool_main  = mp.mean(dim=1)              # (N, 2048)
    mean_pool_wrist = wp.mean(dim=1)              # (N, 2048)
    return patches, mean_pool_main, mean_pool_wrist

# ---------------------------------------------------------------------------
# Encoder with attention map extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_with_attn(raw_enc: SubgoalPatchEncoder, patches: torch.Tensor):
    """
    Run encoder and capture attention weights from the last cross-attn layer.
    patches : (B, 512, 2048)
    Returns : z (B, 4096),  attn (B, 2, 512)
    """
    B = patches.shape[0]
    q = raw_enc.queries.unsqueeze(0).expand(B, -1, -1)  # (B, 2, 2048)
    last_attn = None
    for layer in raw_enc.layers:
        q_n  = layer.norm_q(q)
        kv_n = layer.norm_kv(patches)
        attn_out, attn_w = layer.attn(
            q_n, kv_n, kv_n,
            need_weights=True,
            average_attn_weights=True,   # (B, n_queries, 512)
        )
        q = q + attn_out
        q = q + layer.ff(layer.norm_ff(q))
        last_attn = attn_w
    z = raw_enc.out_norm(q).reshape(B, -1)
    return z, last_attn   # (B, 4096), (B, 2, 512)

# ---------------------------------------------------------------------------
# Load raw images from subgoal_decoder_cache
# ---------------------------------------------------------------------------

def load_episode_images(cache_dir: str, ep_idx: int):
    index = json.loads((Path(cache_dir) / "index.json").read_text())
    ep    = index[ep_idx]
    raw   = np.load(Path(cache_dir) / ep["file"])
    return raw["main_imgs"], raw["wrist_imgs"]   # (N, H, W, 3) uint8 each

# ---------------------------------------------------------------------------
# Test 1: Attention map visualization
# ---------------------------------------------------------------------------

def visualize_attention(backbone, ae, cache_dir, n_frames=VIS_FRAMES):
    print("\n[1] Generating attention maps ...")
    index = json.loads((Path(cache_dir) / "index.json").read_text())
    ep    = index[0]
    main_imgs, wrist_imgs = load_episode_images(cache_dir, 0)
    N = len(main_imgs)

    frame_ids = np.linspace(0, N - 1, n_frames, dtype=int)

    raw_enc = ae.encoder
    fig, axes = plt.subplots(3, n_frames, figsize=(4 * n_frames, 10))

    for col, t in enumerate(frame_ids):
        patches, _, _ = extract_patches(
            backbone, [main_imgs[t]], [wrist_imgs[t]]
        )
        _, attn = encode_with_attn(raw_enc, patches)  # attn: (1, 2, 512)
        attn = attn[0].cpu().numpy()  # (2, 512)

        # main image (first 256 patches → 16×16)
        img_rgb = main_imgs[t]
        h, w = img_rgb.shape[:2]
        img_show = img_rgb[..., ::-1] if img_rgb.shape[-1] == 3 else img_rgb

        axes[0, col].imshow(img_show)
        axes[0, col].set_title(f"frame {t}", fontsize=9)
        axes[0, col].axis("off")

        for q_idx in range(2):
            heat = attn[q_idx, :256].reshape(PATCH_GRID, PATCH_GRID)  # main patches
            ax = axes[q_idx + 1, col]
            ax.imshow(img_show, alpha=0.5)
            ax.imshow(
                heat, cmap="hot", alpha=0.6,
                extent=[0, img_show.shape[1], img_show.shape[0], 0],
                interpolation="bilinear",
            )
            ax.set_title(f"query {q_idx} attn", fontsize=9)
            ax.axis("off")

    axes[1, 0].set_ylabel("Query 0", fontsize=10)
    axes[2, 0].set_ylabel("Query 1", fontsize=10)
    plt.suptitle("Encoder attention on main image patches (last cross-attn layer)", fontsize=11)
    plt.tight_layout()
    plt.savefig("spatial_attention.png", dpi=120, bbox_inches="tight")
    print("  Saved spatial_attention.png")

# ---------------------------------------------------------------------------
# Test 2: Spatial sensitivity — cos_sim vs temporal distance
# ---------------------------------------------------------------------------

def spatial_sensitivity(backbone, ae, cache_dir, n_eps=N_EPISODES):
    print("\n[2] Computing spatial sensitivity ...")
    index = json.loads((Path(cache_dir) / "index.json").read_text())
    ep_ids = list(range(min(n_eps, len(index))))

    # cos_sim[method][dist] accumulator
    enc_cos  = {d: [] for d in TEMPORAL_DISTS}
    pool_cos = {d: [] for d in TEMPORAL_DISTS}

    for ep_idx in ep_ids:
        main_imgs, wrist_imgs = load_episode_images(cache_dir, ep_idx)
        N = len(main_imgs)
        print(f"  ep {ep_idx}: {N} frames", end="", flush=True)

        # Extract all frames in batches
        all_z    = []
        all_pool = []
        for i in range(0, N, ENCODE_BS):
            patches, mp, wp = extract_patches(
                backbone,
                list(main_imgs[i:i + ENCODE_BS]),
                list(wrist_imgs[i:i + ENCODE_BS]),
            )
            z, _ = encode_with_attn(ae.encoder, patches)
            all_z.append(z.cpu().numpy())
            # mean-pool: concat main+wrist pooled vectors → same 4096 dim for fair comparison
            pool = torch.cat([mp, wp], dim=-1)  # (B, 4096)
            all_pool.append(pool.cpu().numpy())

        all_z    = np.concatenate(all_z,    axis=0)   # (N, 4096)
        all_pool = np.concatenate(all_pool, axis=0)   # (N, 4096)

        # Normalise for cos_sim
        z_norm    = all_z    / (np.linalg.norm(all_z,    axis=-1, keepdims=True) + 1e-8)
        pool_norm = all_pool / (np.linalg.norm(all_pool, axis=-1, keepdims=True) + 1e-8)

        for d in TEMPORAL_DISTS:
            if N <= d:
                continue
            t_idx = np.arange(N - d)
            cos_e = (z_norm[t_idx]    * z_norm[t_idx + d]   ).sum(-1)
            cos_p = (pool_norm[t_idx] * pool_norm[t_idx + d]).sum(-1)
            enc_cos[d].extend(cos_e.tolist())
            pool_cos[d].extend(cos_p.tolist())

        print(f"  done")

    # Summary table
    print("\n  Temporal dist | mean cos_sim (encoder) | mean cos_sim (mean-pool)")
    print("  " + "-" * 58)
    enc_means  = []
    pool_means = []
    for d in TEMPORAL_DISTS:
        em = np.mean(enc_cos[d])  if enc_cos[d]  else float("nan")
        pm = np.mean(pool_cos[d]) if pool_cos[d] else float("nan")
        enc_means.append(em)
        pool_means.append(pm)
        better = "enc<pool ✓" if em < pm else "enc>=pool"
        print(f"  d={d:3d}          |   {em:.4f}               |   {pm:.4f}    {better}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(TEMPORAL_DISTS, enc_means,  "o-", label="Encoder latent", color="steelblue")
    ax.plot(TEMPORAL_DISTS, pool_means, "s--", label="Mean-pool", color="tomato")
    ax.set_xlabel("Temporal distance (frames)", fontsize=11)
    ax.set_ylabel("Mean cosine similarity", fontsize=11)
    ax.set_title("Spatial sensitivity: encoder vs mean-pool\n"
                 "(lower cos_sim = more spatially discriminative)", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("spatial_sensitivity.png", dpi=120)
    print("  Saved spatial_sensitivity.png")

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    backbone = load_backbone()
    ae       = load_encoder()

    visualize_attention(backbone, ae, SUBGOAL_CACHE_DIR)
    spatial_sensitivity(backbone, ae, SUBGOAL_CACHE_DIR)

    print("\nDone. Check spatial_attention.png and spatial_sensitivity.png")
