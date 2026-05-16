"""
Phase A: Train SubgoalAutoencoder (encoder + decoder).

Pipeline per batch:
  1. Load raw images from subgoal_decoder_cache
  2. Frozen embed_image (PI0 backbone, no_grad) → (B, 256, 2048) × 2
  3. Concat main + wrist patches → (B, 512, 2048)
  4. SubgoalAutoencoder forward → reconstruction loss
  5. Backprop through autoencoder only

Run (single GPU):
    python train_subgoal_encoder.py

Run (multi-GPU):
    torchrun --nproc_per_node=N train_subgoal_encoder.py
"""

import os, sys, json, random, math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from tqdm import tqdm

import wandb

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "openpi" / "src"))

from model.subgoal_encoder import SubgoalAutoencoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBGOAL_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/subgoal_decoder_cache"
RLDS_DATA_DIR     = "/mnt/nfs/Users/jerry007005/dataset/gripper_annotated_libero_rlds"
DATASET_NAMES     = [
    "libero_spatial_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_10_no_noops",
]
PI05_CKPT_DIR     = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"
CKPT_DIR          = "./checkpoints/subgoal_encoder"

IMG_SIZE   = 224
PATCH_DIM  = 2048
N_PATCHES  = 512   # 256 main + 256 wrist

BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500
MAX_STEPS    = 30000
LOG_STEPS    = 100
SAVE_STEPS   = 2000

WANDB_PROJECT  = "RL-VLA"
WANDB_RUN_NAME = "subgoal-encoder"


# ---------------------------------------------------------------------------
# Phase 0: build image cache from RLDS
# ---------------------------------------------------------------------------

def _resize_img(img_uint8: np.ndarray) -> np.ndarray:
    from openpi_client import image_tools
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_uint8, IMG_SIZE, IMG_SIZE)
    )


def _make_tokenizer():
    from openpi.models import tokenizer as _tok
    return _tok.PaligemmaTokenizer(max_len=48)


def build_cache():
    """Read gripper_annotated RLDS, save full per-episode npz (images, lang, states, sg_frames)."""
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    cache_path = Path(SUBGOAL_CACHE_DIR)
    index_file = cache_path / "index.json"
    if index_file.exists():
        print("Cache already exists, skipping Phase 0.")
        return

    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Building cache from {RLDS_DATA_DIR} ...")

    from convert_rlds_subgoal_2 import GripperAnnotatedLiberoDataset
    tokenizer = _make_tokenizer()

    index  = []
    ep_idx = 0
    for dataset_name in DATASET_NAMES:
        print(f"  Processing {dataset_name} ...")
        builder = GripperAnnotatedLiberoDataset(config=dataset_name, data_dir=RLDS_DATA_DIR)
        ds = builder.as_dataset(split="train")

        for episode in ds:
            steps = list(episode["steps"].as_numpy_iterator())
            n = len(steps)

            lang_str = steps[0]["language_instruction"].decode()
            lang_tokens, lang_mask = tokenizer.tokenize(lang_str, state=None)

            main_imgs  = np.stack([_resize_img(s["observation"]["image"])       for s in steps])
            wrist_imgs = np.stack([_resize_img(s["observation"]["wrist_image"]) for s in steps])
            states     = np.stack([s["observation"]["state"].astype(np.float32) for s in steps])
            sg_frames  = np.array([s["subgoal_frame"] for s in steps], dtype=np.int32)

            ep_file = cache_path / f"ep_{ep_idx:04d}.npz"
            np.savez_compressed(ep_file,
                main_imgs   = main_imgs.astype(np.uint8),
                wrist_imgs  = wrist_imgs.astype(np.uint8),
                lang_tokens = lang_tokens.astype(np.int32),
                lang_mask   = lang_mask.astype(bool),
                states      = states,
                sg_frames   = sg_frames,
            )
            index.append({"ep_idx": ep_idx, "n_steps": n, "file": ep_file.name,
                          "dataset": dataset_name})

            if ep_idx % 50 == 0:
                print(f"  ep {ep_idx:4d}: {n} steps | {lang_str[:50]}")
            ep_idx += 1

    index_file.write_text(json.dumps(index, indent=2))
    print(f"Phase 0 done. {ep_idx} episodes → {SUBGOAL_CACHE_DIR}")


# ---------------------------------------------------------------------------
# Dataset: yields raw images (no patch computation yet)
# ---------------------------------------------------------------------------

def _img_to_chw(img_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32 in [-1, 1]."""
    t = torch.from_numpy(img_uint8).float() / 255.0
    return (t * 2.0 - 1.0).permute(2, 0, 1)


class PatchImageDataset(IterableDataset):
    """
    Yields (main_chw, wrist_chw) for every frame in every episode.
    Uses a per-worker image cache to avoid repeated NFS reads.
    """

    def __init__(self, cache_dir: str):
        index = json.loads((Path(cache_dir) / "index.json").read_text())
        self.episodes  = index
        self.cache_dir = cache_dir

    def __iter__(self):
        img_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        cache_limit = 32

        while True:
            ep   = random.choice(self.episodes)
            idx  = ep["ep_idx"]

            if idx not in img_cache:
                try:
                    raw = np.load(Path(self.cache_dir) / ep["file"])
                    if len(img_cache) >= cache_limit:
                        img_cache.pop(next(iter(img_cache)))
                    img_cache[idx] = (raw["main_imgs"], raw["wrist_imgs"])
                except Exception:
                    continue

            main_imgs, wrist_imgs = img_cache[idx]
            step_order = list(range(ep["n_steps"]))
            random.shuffle(step_order)

            for t in step_order:
                yield _img_to_chw(main_imgs[t]), _img_to_chw(wrist_imgs[t])


# ---------------------------------------------------------------------------
# Load frozen embed_image from PI0 backbone
# ---------------------------------------------------------------------------

def _load_embed_fn():
    """
    Returns a function embed_image(img) that maps
    (B, 3, 224, 224) float32 → (B, 256, 2048) float32.
    The underlying backbone weights are frozen.
    """
    from openpi.training import config as _config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch

    print("Loading PI0 backbone for embed_image ...")
    train_cfg = _config.get_config("pi05_libero")
    backbone  = PI0Pytorch(config=train_cfg.model)
    safetensors.torch.load_model(
        backbone,
        os.path.join(PI05_CKPT_DIR, "model.safetensors"),
        strict=False,
    )
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    print("  Backbone loaded and frozen.")
    return backbone


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    build_cache()

    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank       = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    torch.set_float32_matmul_precision("high")

    if is_main:
        Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
        wandb.init(
            project = WANDB_PROJECT,
            name    = WANDB_RUN_NAME,
            config  = {
                "batch_size":  BATCH_SIZE,
                "lr":          LR,
                "max_steps":   MAX_STEPS,
                "patch_dim":   PATCH_DIM,
                "n_patches":   N_PATCHES,
                "world_size":  world_size,
            },
            resume="allow",
        )

    random.seed(42 + rank)

    # ---- data ----
    dataset = PatchImageDataset(SUBGOAL_CACHE_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4,
                         pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # ---- frozen backbone (embed_image only) ----
    backbone = _load_embed_fn().to(device)

    # ---- autoencoder (trainable) ----
    ae = SubgoalAutoencoder(
        patch_dim  = PATCH_DIM,
        n_queries  = 2,
        n_patches  = N_PATCHES,
        n_heads    = 16,
        enc_layers = 2,
        dec_layers = 2,
        ffn_mult   = 4,
    ).to(device)

    if use_ddp:
        ae = DDP(ae, device_ids=[local_rank])

    raw_ae = ae.module if use_ddp else ae

    if is_main:
        print(f"Autoencoder params: {raw_ae.n_params() / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(ae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- resume ----
    ckpt_path  = Path(CKPT_DIR) / "checkpoint.pt"
    start_step = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_ae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
        start_step = ckpt["step"] + 1
        if is_main:
            print(f"Resumed from step {start_step}")

    if use_ddp:
        dist.barrier()

    # ---- training loop ----
    ae.train()
    data_iter = iter(loader)
    pbar = tqdm(range(start_step, MAX_STEPS), initial=start_step, total=MAX_STEPS,
                dynamic_ncols=True, disable=not is_main)

    for step in pbar:
        main_img, wrist_img = next(data_iter)
        main_img  = main_img.to(device,  non_blocking=True)   # (B, 3, 224, 224)
        wrist_img = wrist_img.to(device, non_blocking=True)

        # embed_image: frozen backbone, no grad, output bfloat16
        with torch.no_grad():
            main_patches  = backbone.paligemma_with_expert.embed_image(main_img).float()   # (B, 256, 2048)
            wrist_patches = backbone.paligemma_with_expert.embed_image(wrist_img).float()  # (B, 256, 2048)

        patches = torch.cat([main_patches, wrist_patches], dim=1)   # (B, 512, 2048)

        optimizer.zero_grad()
        z, recon, loss = ae(patches)
        loss.backward()
        nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if is_main:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            if (step + 1) % LOG_STEPS == 0:
                # cosine similarity between original and reconstructed patches (batch mean)
                with torch.no_grad():
                    cos_sim = torch.nn.functional.cosine_similarity(
                        recon.reshape(-1, PATCH_DIM),
                        patches.reshape(-1, PATCH_DIM),
                        dim=-1
                    ).mean().item()
                print(
                    f"step {step+1:5d}/{MAX_STEPS} | "
                    f"loss={loss.item():.4f}  cos={cos_sim:.4f}  lr={lr:.2e}"
                )
                wandb.log({
                    "train/recon_loss": loss.item(),
                    "train/cos_sim":    cos_sim,
                    "train/lr":         lr,
                }, step=step + 1)

            if (step + 1) % SAVE_STEPS == 0 or step == MAX_STEPS - 1:
                torch.save({
                    "step":      step,
                    "model":     raw_ae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

    if is_main:
        wandb.finish()
        print("Phase A training complete.")

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
