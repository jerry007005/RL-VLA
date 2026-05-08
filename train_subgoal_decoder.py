"""
Train PI0WithGoalExpert with flow-matching loss.

Phase 0  — one-time cache build
  Load annotated_libero_rlds, resize images to 224×224 (uint8), tokenize
  language instructions with PaliGemma SentencePiece tokenizer, save per-
  episode .npz files to SUBGOAL_CACHE_DIR.

Phase 1  — training
  Each batch:
    ① load raw images + lang tokens from Phase 0 cache (lazy per episode)
    ② load subgoal targets (main_feats, wrist_feats, states at sg_frame)
       from the existing executor feature cache (already on disk)
    ③ frozen PI0 prefix → KV cache
    ④ noisy [sg_main | sg_wrist | sg_state] tokens → goal expert → flow-matching loss
    ⑤ back-prop through goal expert weights only

Run: python train_subgoal_decoder.py
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

from model.pi0_subgoal_decoder import PI0WithGoalExpert

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DST_DATA_DIR      = "/mnt/nfs/Users/jerry007005/dataset/annotated_libero_rlds"
DATASET_NAME      = "libero_spatial_no_noops"
FEAT_CACHE_DIR    = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache"
SUBGOAL_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/subgoal_decoder_cache"
CKPT_DIR          = "./checkpoints/subgoal_decoder"

PI05_CKPT_DIR = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"

IMG_SIZE     = 224
MAX_LANG_LEN = 48

PATCH_DIM   = 2048
PROPRIO_DIM = 8

BATCH_SIZE    = 128
LR            = 6e-4  # linear scaling: 3e-4 * (128*2/64), conservative
WEIGHT_DECAY  = 1e-4
WARMUP_STEPS  = 2000
MAX_STEPS     = 100000
LOG_STEPS    = 100
SAVE_STEPS   = 1000

WANDB_PROJECT  = "RL-VLA"
WANDB_RUN_NAME = "goal-expert"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Phase 0: build image + language cache
# ---------------------------------------------------------------------------

def _resize_img(img_uint8: np.ndarray) -> np.ndarray:
    from openpi_client import image_tools
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_uint8, IMG_SIZE, IMG_SIZE)
    )


def _make_tokenizer():
    from openpi.models import tokenizer as _tok
    return _tok.PaligemmaTokenizer(max_len=MAX_LANG_LEN)


def extract_images_and_lang():
    """Phase 0: cache resized images + tokenized language per episode."""
    cache_path = Path(SUBGOAL_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    index_file = cache_path / "index.json"
    if index_file.exists():
        print("Phase 0 cache already exists, skipping.")
        return

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    tokenizer = _make_tokenizer()

    from convert_rlds_subgoal import AnnotatedLiberoDataset
    builder = AnnotatedLiberoDataset(config=DATASET_NAME, data_dir=DST_DATA_DIR)
    ds = builder.as_dataset(split="train")

    index = []
    for ep_idx, episode in enumerate(ds):
        steps = list(episode["steps"].as_numpy_iterator())
        n = len(steps)

        lang_str = steps[0]["language_instruction"].decode()
        lang_tokens, lang_mask = tokenizer.tokenize(lang_str, state=None)

        main_imgs  = np.stack([_resize_img(s["observation"]["image"])       for s in steps])
        wrist_imgs = np.stack([_resize_img(s["observation"]["wrist_image"]) for s in steps])

        ep_file = cache_path / f"ep_{ep_idx:04d}.npz"
        np.savez_compressed(ep_file,
            main_imgs   = main_imgs.astype(np.uint8),
            wrist_imgs  = wrist_imgs.astype(np.uint8),
            lang_tokens = lang_tokens.astype(np.int32),
            lang_mask   = lang_mask.astype(bool),
        )
        index.append({"ep_idx": ep_idx, "n_steps": n, "file": ep_file.name})

        if ep_idx % 20 == 0:
            print(f"  ep {ep_idx:4d}: {n} steps | lang: {lang_str[:60]}")

    index_file.write_text(json.dumps(index, indent=2))
    print(f"Phase 0 done. {len(index)} episodes → {SUBGOAL_CACHE_DIR}")


# ---------------------------------------------------------------------------
# Phase 1: Dataset
# ---------------------------------------------------------------------------

def _img_to_chw(img_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32 in [-1, 1]."""
    t = torch.from_numpy(img_uint8).float() / 255.0
    return (t * 2.0 - 1.0).permute(2, 0, 1)


class GoalExpertDataset(IterableDataset):
    """
    Each item:
      main_chw    : (3, 224, 224) float32  current main image
      wrist_chw   : (3, 224, 224) float32  current wrist image
      lang_tokens : (MAX_LANG_LEN,) int64
      lang_mask   : (MAX_LANG_LEN,) bool
      sg_main     : (PATCH_DIM,)  float32  mean-pooled SigLIP at subgoal frame
      sg_wrist    : (PATCH_DIM,)  float32  mean-pooled SigLIP wrist at subgoal frame
      sg_state    : (PROPRIO_DIM,) float32 robot state at subgoal frame
    """

    def __init__(self, subgoal_cache_dir: str, feat_cache_dir: str):
        sg_path   = Path(subgoal_cache_dir)
        feat_path = Path(feat_cache_dir)

        index = json.loads((feat_path / "index.json").read_text())

        self.episodes = []
        print("Loading caches ...")
        for entry in index:
            ep_idx = entry["ep_idx"]

            feat_data = np.load(feat_path / entry["file"])
            sg_file   = sg_path / f"ep_{ep_idx:04d}.npz"
            if not sg_file.exists():
                continue
            sg_data = np.load(sg_file)

            self.episodes.append({
                "ep_idx":      ep_idx,
                "n_steps":     entry["n_steps"],
                "sg_file":     str(sg_file),
                "main_feats":  feat_data["main_feats"].astype(np.float32),
                "wrist_feats": feat_data["wrist_feats"].astype(np.float32),
                "states":      feat_data["states"].astype(np.float32),
                "sg_frames":   feat_data["sg_frames"].astype(np.int32),
                "lang_tokens": torch.from_numpy(sg_data["lang_tokens"].astype(np.int64)),
                "lang_mask":   torch.from_numpy(sg_data["lang_mask"].astype(bool)),
            })

        total = sum(e["n_steps"] for e in self.episodes)
        print(f"  {len(self.episodes)} episodes | {total} total steps")

    def __iter__(self):
        while True:
            ep = random.choice(self.episodes)
            try:
                raw        = np.load(ep["sg_file"])
                main_imgs  = raw["main_imgs"]    # (N, 224, 224, 3) uint8
                wrist_imgs = raw["wrist_imgs"]
            except Exception:
                continue

            step_order = list(range(ep["n_steps"]))
            random.shuffle(step_order)

            for step_idx in step_order:
                sg_idx = int(ep["sg_frames"][step_idx])
                yield (
                    _img_to_chw(main_imgs[step_idx]),                    # (3, H, W)
                    _img_to_chw(wrist_imgs[step_idx]),                   # (3, H, W)
                    ep["lang_tokens"],                                    # (T,)
                    ep["lang_mask"],                                      # (T,)
                    torch.from_numpy(ep["main_feats"][sg_idx]),           # (2048,)
                    torch.from_numpy(ep["wrist_feats"][sg_idx]),          # (2048,)
                    torch.from_numpy(ep["states"][sg_idx]),               # (8,)
                )


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def _load_model() -> PI0WithGoalExpert:
    from openpi.training import config as _config
    import safetensors.torch

    print("Loading PI0WithGoalExpert ...")
    train_cfg = _config.get_config("pi05_libero")
    model = PI0WithGoalExpert(
        config      = train_cfg.model,
        patch_dim   = PATCH_DIM,
        proprio_dim = PROPRIO_DIM,
        freeze_pi0  = True,
    )
    # Load PI0 backbone weights; goal expert weights stay randomly initialized
    safetensors.torch.load_model(
        model,
        os.path.join(PI05_CKPT_DIR, "model.safetensors"),
        strict=False,
    )
    model.goal_expert = torch.compile(model.goal_expert)
    print("  Model loaded (goal_expert compiled).")
    return model


# ---------------------------------------------------------------------------
# Phase 1: Training
# ---------------------------------------------------------------------------

def train():
    # ---- DDP init ----
    use_ddp    = "LOCAL_RANK" in os.environ
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

    if is_main:
        Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
        wandb.init(
            project = WANDB_PROJECT,
            name    = WANDB_RUN_NAME,
            config  = {
                "batch_size":      BATCH_SIZE,
                "eff_batch_size":  BATCH_SIZE * world_size,
                "lr":              LR,
                "weight_decay":    WEIGHT_DECAY,
                "warmup_steps":    WARMUP_STEPS,
                "max_steps":       MAX_STEPS,
                "world_size":      world_size,
                "max_lang_len":    MAX_LANG_LEN,
                "img_size":        IMG_SIZE,
                "patch_dim":       PATCH_DIM,
                "proprio_dim":     PROPRIO_DIM,
            },
            resume="allow",
        )

    # 每个 rank 用不同随机种子，保证数据多样性
    random.seed(42 + rank)

    dataset = GoalExpertDataset(SUBGOAL_CACHE_DIR, FEAT_CACHE_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2,
                         pin_memory=True, persistent_workers=True)

    model = _load_model().to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if use_ddp else model
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    if is_main:
        print(f"Trainable params: {n_params/1e6:.1f}M | "
              f"world_size={world_size} | eff_batch={BATCH_SIZE * world_size}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_path  = Path(CKPT_DIR) / "checkpoint.pt"
    start_step = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            if is_main:
                print("  Scheduler state incompatible, resetting.")
        start_step = ckpt["step"] + 1
        if is_main:
            print(f"Resumed from step {start_step}")

    if use_ddp:
        dist.barrier()

    if is_main:
        print(f"Training GoalExpert | device={device} | world_size={world_size} | max_steps={MAX_STEPS}")

    model.train()
    raw_model.paligemma_with_expert.eval()   # keep frozen backbone in eval mode

    data_iter = iter(loader)
    pbar = tqdm(range(start_step, MAX_STEPS), initial=start_step, total=MAX_STEPS,
                dynamic_ncols=True, disable=not is_main)

    for step in pbar:
        batch = next(data_iter)
        main_img, wrist_img, lang_tokens, lang_mask, sg_main, sg_wrist, sg_state = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()
        loss, info = model(main_img, wrist_img, lang_tokens, lang_mask,
                           sg_main, sg_wrist, sg_state)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if is_main:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{info['loss/total']:.4f}",
                main=f"{info['loss/sg_main']:.4f}",
                wrist=f"{info['loss/sg_wrist']:.4f}",
                state=f"{info['loss/sg_state']:.4f}",
                lr=f"{lr:.2e}",
            )

            if (step + 1) % LOG_STEPS == 0:
                print(
                    f"step {step+1:6d}/{MAX_STEPS} | "
                    f"loss={info['loss/total']:.4f} "
                    f"main={info['loss/sg_main']:.4f} "
                    f"wrist={info['loss/sg_wrist']:.4f} "
                    f"state={info['loss/sg_state']:.4f} "
                    f"lr={lr:.2e}"
                )
                wandb.log(
                    {
                        "train/loss_total": info["loss/total"],
                        "train/loss_main":  info["loss/sg_main"],
                        "train/loss_wrist": info["loss/sg_wrist"],
                        "train/loss_state": info["loss/sg_state"],
                        "train/lr":         lr,
                    },
                    step=step + 1,
                )

            if (step + 1) % SAVE_STEPS == 0 or step == MAX_STEPS - 1:
                torch.save({
                    "step":      step,
                    "model":     raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

    if is_main:
        wandb.finish()
        print("Training complete.")

    if use_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        extract_images_and_lang()
    train()
