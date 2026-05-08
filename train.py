"""
Train Executor with behavioral cloning on annotated_libero_rlds.

Two phases (automatic):
  1. Feature extraction  — encode images with frozen pi0.5 SigLIP encoder,
                           cache per episode as .npz
  2. Training            — load cached features, train Executor with L1 loss

Run: python train.py
"""

import os, sys, json, itertools, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
from tqdm import tqdm

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from model.executor import Executor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DST_DATA_DIR  = "/mnt/nfs/Users/jerry007005/dataset/annotated_libero_rlds"
DATASET_NAME  = "libero_spatial_no_noops"
CACHE_DIR     = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache"
CKPT_DIR      = "./checkpoints/executor"

PI05_CKPT_DIR = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"
IMG_SIZE      = 224
PATCH_DIM     = 2048   # pi0.5 SigLIP projection dim

PROPRIO_DIM   = 8
ACTION_DIM    = 7
HIDDEN_DIM    = 512
NUM_LAYERS    = 5

BATCH_SIZE          = 64
SHUFFLE_BUFFER_SIZE = 1_000
LR                  = 3e-4
WEIGHT_DECAY        = 1e-4
MAX_STEPS           = 1_000_000
LOG_STEPS           = 100
SAVE_STEPS          = 1_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WANDB_PROJECT = "RL-VLA"
WANDB_RUN_NAME = "executor"  # set to a string to name the run, or None for auto


# ---------------------------------------------------------------------------
# Phase 1: Feature extraction with pi0.5 SigLIP encoder
# ---------------------------------------------------------------------------

def _load_pi05_vision_encoder():
    """Load only the vision tower from pi0.5 checkpoint. Returns a callable."""
    from openpi.training import config as _config
    from openpi.models_pytorch import pi0_pytorch
    import safetensors.torch

    print("Loading pi0.5 vision encoder ...")
    train_config = _config.get_config("pi05_libero")
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    weight_path = os.path.join(PI05_CKPT_DIR, "model.safetensors")
    safetensors.torch.load_model(model, weight_path)

    encoder = model.paligemma_with_expert
    encoder.eval().to(DEVICE)
    for p in encoder.parameters():
        p.requires_grad_(False)

    print("  pi0.5 vision encoder loaded.")
    return encoder


def _preprocess_images(imgs_uint8: np.ndarray) -> torch.Tensor:
    """
    imgs_uint8: (N, H, W, 3) uint8
    Returns: (N, 3, 224, 224) float32 in [-1, 1]
    """
    from openpi_client import image_tools
    out = []
    for img in imgs_uint8:
        resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
        )
        t = torch.from_numpy(resized).float() / 255.0  # [0, 1] HWC
        t = t * 2.0 - 1.0                              # [-1, 1] HWC
        t = t.permute(2, 0, 1)                         # CHW
        out.append(t)
    return torch.stack(out, dim=0)  # (N, 3, H, W)


@torch.no_grad()
def _encode_batch(encoder, imgs_uint8: list[np.ndarray]) -> np.ndarray:
    """Encode a list of uint8 HWC images → (N, 2048) mean-pooled float32."""
    pixel_values = _preprocess_images(np.stack(imgs_uint8)).to(DEVICE)
    # embed_image returns (N, 256, 2048)
    patch_feats = encoder.embed_image(pixel_values).float()
    pooled = patch_feats.mean(dim=1)  # (N, 2048)
    return pooled.cpu().numpy()


def extract_features():
    """Extract and cache pi0.5 image features for every step of every episode."""
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    index_file = cache_path / "index.json"
    if index_file.exists():
        print("Feature cache already exists, skipping extraction.")
        return

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    encoder = _load_pi05_vision_encoder()

    from convert_rlds_subgoal import AnnotatedLiberoDataset
    builder = AnnotatedLiberoDataset(config=DATASET_NAME, data_dir=DST_DATA_DIR)
    ds = builder.as_dataset(split="train")

    index = []
    ENCODE_BATCH = 32

    for ep_idx, episode in enumerate(ds):
        steps = list(episode["steps"].as_numpy_iterator())
        n = len(steps)

        main_imgs  = [s["observation"]["image"]       for s in steps]
        wrist_imgs = [s["observation"]["wrist_image"] for s in steps]
        states     = np.array([s["observation"]["state"] for s in steps], dtype=np.float32)
        actions    = np.array([s["action"]               for s in steps], dtype=np.float32)
        sg_frames  = np.array([int(s["subgoal_frame"])   for s in steps], dtype=np.int32)

        def _batch_encode(imgs):
            out = []
            for i in range(0, len(imgs), ENCODE_BATCH):
                out.append(_encode_batch(encoder, imgs[i:i + ENCODE_BATCH]))
            return np.concatenate(out, axis=0)

        main_feats  = _batch_encode(main_imgs)   # (N, 2048)
        wrist_feats = _batch_encode(wrist_imgs)  # (N, 2048)

        ep_file = cache_path / f"ep_{ep_idx:04d}.npz"
        np.savez_compressed(ep_file,
            main_feats  = main_feats,
            wrist_feats = wrist_feats,
            states      = states,
            actions     = actions,
            sg_frames   = sg_frames,
        )
        index.append({"ep_idx": ep_idx, "n_steps": n, "file": ep_file.name})

        if ep_idx % 20 == 0:
            print(f"  ep {ep_idx:4d}: {n} steps cached")

    index_file.write_text(json.dumps(index, indent=2))
    print(f"Feature extraction done. {len(index)} episodes → {CACHE_DIR}")


# ---------------------------------------------------------------------------
# Phase 2: Dataset
# ---------------------------------------------------------------------------

class ExecutorDataset(IterableDataset):
    """
    Infinite iterable dataset with shuffle buffer.
    Each item: (imgs (4, 2048), curr_state (8,), sg_state (8,), action (7,))
    imgs layout: [curr_main, curr_wrist, sg_main, sg_wrist]
    """

    def __init__(self, cache_dir: str, shuffle_buffer_size: int = 1_000):
        self.samples = []
        self.shuffle_buffer_size = shuffle_buffer_size

        index = json.loads((Path(cache_dir) / "index.json").read_text())
        print("Loading feature cache ...")
        for entry in index:
            data = np.load(Path(cache_dir) / entry["file"])
            ep = {
                "main_feats":  data["main_feats"].astype(np.float32),
                "wrist_feats": data["wrist_feats"].astype(np.float32),
                "states":      data["states"].astype(np.float32),
                "actions":     data["actions"].astype(np.float32),
                "sg_frames":   data["sg_frames"].astype(np.int32),
            }
            for step_idx in range(entry["n_steps"]):
                sg_idx = int(ep["sg_frames"][step_idx])
                imgs = np.stack([
                    ep["main_feats"][step_idx],
                    ep["wrist_feats"][step_idx],
                    ep["main_feats"][sg_idx],
                    ep["wrist_feats"][sg_idx],
                ], axis=0)
                self.samples.append((
                    torch.from_numpy(imgs),
                    torch.from_numpy(ep["states"][step_idx]),
                    torch.from_numpy(ep["states"][sg_idx]),
                    torch.from_numpy(ep["actions"][step_idx]),
                ))

        print(f"  {len(self.samples)} frames total")

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        buf_size = min(self.shuffle_buffer_size, len(self.samples))
        idx_iter = itertools.cycle(random.sample(range(len(self.samples)), len(self.samples)))
        buffer = [self.samples[next(idx_iter)] for _ in range(buf_size)]
        while True:
            pos = random.randrange(buf_size)
            yield buffer[pos]
            buffer[pos] = self.samples[next(idx_iter)]


# ---------------------------------------------------------------------------
# Phase 2: Training
# ---------------------------------------------------------------------------

def train():
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "dataset":      DATASET_NAME,
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "max_steps":    MAX_STEPS,
            "hidden_dim":   HIDDEN_DIM,
            "num_layers":   NUM_LAYERS,
            "patch_dim":    PATCH_DIM,
            "proprio_dim":  PROPRIO_DIM,
            "action_dim":   ACTION_DIM,
        },
        resume="allow",
    )

    dataset = ExecutorDataset(CACHE_DIR, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)
    loader  = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        num_workers = 0,
        pin_memory  = True,
    )

    model = Executor(
        num_imgs          = 4,
        patch_dim         = PATCH_DIM,
        proprio_dim       = PROPRIO_DIM,
        action_dim        = ACTION_DIM,
        hidden_dim        = HIDDEN_DIM,
        num_hidden_layers = NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

    ckpt_path = Path(CKPT_DIR) / "checkpoint.pt"
    start_step = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {ckpt_path} (step {start_step})")

    print(f"Training Executor on {DEVICE} | {len(dataset)} frames | max_steps={MAX_STEPS}")

    model.train()
    data_iter = iter(loader)

    pbar = tqdm(range(start_step, MAX_STEPS), initial=start_step, total=MAX_STEPS,
                dynamic_ncols=True)

    for step in pbar:
        imgs, curr_state, sg_state, actions = next(data_iter)
        imgs       = imgs.to(DEVICE)
        curr_state = curr_state.to(DEVICE)
        sg_state   = sg_state.to(DEVICE)
        actions    = actions.to(DEVICE)

        optimizer.zero_grad()
        loss, info = model.compute_bc_loss(imgs, curr_state, sg_state, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(
            l1=f"{info['loss/bc']:.4f}",
            std=f"{info['train/std']:.4f}",
            lr=f"{lr:.2e}",
        )

        if (step + 1) % LOG_STEPS == 0:
            print(
                f"step {step+1:6d}/{MAX_STEPS} | "
                f"l1={info['loss/bc']:.4f} std={info['train/std']:.4f} "
                f"lr={lr:.2e}"
            )
            wandb.log(
                {
                    "train/loss_bc": info["loss/bc"],
                    "train/std":     info["train/std"],
                    "train/lr":      lr,
                },
                step=step + 1,
            )

        if (step + 1) % SAVE_STEPS == 0 or step == MAX_STEPS - 1:
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    wandb.finish()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extract_features()
    train()
