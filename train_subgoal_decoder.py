"""
Train PI0WithGoalExpert with flow-matching loss.

Phase 0   — one-time: resize images + tokenize language → SUBGOAL_CACHE_DIR
Phase 0.5 — one-time: backbone + SubgoalAutoencoder → encoder latents z → SG_ENCODER_CACHE_DIR
Phase 1   — training:
    ① raw images + lang tokens from SUBGOAL_CACHE_DIR (lazy per episode)
    ② encoder latent z[sg], states[sg] from SG_ENCODER_CACHE_DIR
    ③ frozen PI0 prefix → KV cache
    ④ noisy [sg_main | sg_wrist | sg_state] → goal expert → flow-matching loss
    ⑤ back-prop through goal expert only

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

DST_DATA_DIR         = "/mnt/nfs/Users/jerry007005/dataset/annotated_libero_rlds"
DATASET_NAME         = "libero_spatial_no_noops"
SUBGOAL_CACHE_DIR    = "/mnt/nfs/Users/jerry007005/dataset/subgoal_decoder_cache"
SG_ENCODER_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/sg_encoder_cache"
CKPT_DIR             = "./checkpoints/subgoal_decoder"

PI05_CKPT_DIR     = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"
ENCODER_CKPT_PATH = "./checkpoints/subgoal_encoder/checkpoint.pt"

IMG_SIZE     = 224
MAX_LANG_LEN = 48

PATCH_DIM   = 2048
PROPRIO_DIM = 8

BATCH_SIZE    = 64    # reduced from 128: LoRA backprop through prefix needs more memory
LR            = 3e-4

LORA_RANK  = 8
LORA_ALPHA = 16
WEIGHT_DECAY  = 1e-4
WARMUP_STEPS  = 2000
MAX_STEPS     = 100000
LOG_STEPS    = 100
SAVE_STEPS   = 1000

WANDB_PROJECT  = "RL-VLA"
WANDB_RUN_NAME = "goal-expert"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Phase 0.5 helpers: backbone + encoder for latent pre-computation
# ---------------------------------------------------------------------------

def _load_pi05_backbone():
    from openpi.training import config as _cfg
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch

    print("Loading PI0 backbone ...")
    cfg   = _cfg.get_config("pi05_libero")
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


def _load_subgoal_encoder_for_phase05():
    from model.subgoal_encoder import SubgoalAutoencoder
    ae = SubgoalAutoencoder(
        patch_dim=PATCH_DIM, n_queries=2, n_patches=512,
        n_heads=16, enc_layers=2, dec_layers=2, ffn_mult=4,
    )
    ckpt = torch.load(ENCODER_CKPT_PATH, map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval().to(DEVICE)
    for p in ae.parameters():
        p.requires_grad_(False)
    print("  SubgoalAutoencoder loaded.")
    return ae


@torch.no_grad()
def _encode_frames(backbone, encoder, main_imgs_uint8, wrist_imgs_uint8, bs=16):
    """
    (N, H, W, 3) uint8 arrays → z (N, 4096) float32 via backbone + encoder.
    """
    from openpi_client import image_tools

    def _prep(imgs):
        out = []
        for img in imgs:
            r = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
            )
            t = torch.from_numpy(r).float() / 255.0
            out.append((t * 2.0 - 1.0).permute(2, 0, 1))
        return torch.stack(out).to(DEVICE)

    N = len(main_imgs_uint8)
    z_list = []
    for i in range(0, N, bs):
        mp = backbone.embed_image(_prep(main_imgs_uint8[i:i+bs])).float()   # (B,256,2048)
        wp = backbone.embed_image(_prep(wrist_imgs_uint8[i:i+bs])).float()  # (B,256,2048)
        patches = torch.cat([mp, wp], dim=1)                                # (B,512,2048)
        z_list.append(encoder.encode(patches).cpu().numpy())
    return np.concatenate(z_list, axis=0).astype(np.float32)                # (N,4096)


def extract_sg_encoder_latents():
    """
    Phase 0.5: compute SubgoalAutoencoder latents for every frame of every episode.

    Reads:
      subgoal_decoder_cache  — raw images (main_imgs, wrist_imgs)
      executor_feat_cache    — states, sg_frames  (old format OK)

    Writes to sg_encoder_cache per episode:
      z         : (N, 4096)  encoder latent (main+wrist)
      states    : (N, 8)
      sg_frames : (N,)
    """
    # executor_feat_cache (old format) is only needed here to bootstrap
    # states and sg_frames; not used anywhere else in this script.
    _FEAT_CACHE = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache"

    out_path = Path(SG_ENCODER_CACHE_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    index_file = out_path / "index.json"

    if index_file.exists():
        print("Phase 0.5 cache already exists, skipping.")
        return

    sg_path   = Path(SUBGOAL_CACHE_DIR)
    feat_path = Path(_FEAT_CACHE)

    sg_index   = json.loads((sg_path   / "index.json").read_text())
    feat_index = json.loads((feat_path / "index.json").read_text())
    feat_map   = {e["ep_idx"]: e for e in feat_index}

    backbone = _load_pi05_backbone()
    encoder  = _load_subgoal_encoder_for_phase05()

    out_index = []
    for ep_entry in sg_index:
        ep_idx = ep_entry["ep_idx"]
        if ep_idx not in feat_map:
            continue

        sg_data   = np.load(sg_path   / ep_entry["file"])
        feat_data = np.load(feat_path / feat_map[ep_idx]["file"])

        main_imgs  = sg_data["main_imgs"]    # (N, 224, 224, 3)
        wrist_imgs = sg_data["wrist_imgs"]

        z = _encode_frames(backbone, encoder,
                           list(main_imgs), list(wrist_imgs))  # (N, 4096)

        ep_file = out_path / f"ep_{ep_idx:04d}.npz"
        np.savez_compressed(ep_file,
            z         = z,
            states    = feat_data["states"].astype(np.float32),
            sg_frames = feat_data["sg_frames"].astype(np.int32),
        )
        out_index.append({"ep_idx": ep_idx, "n_steps": ep_entry["n_steps"],
                          "file": ep_file.name})

        if ep_idx % 20 == 0:
            print(f"  ep {ep_idx:4d}: {ep_entry['n_steps']} steps encoded")

    index_file.write_text(json.dumps(out_index, indent=2))
    print(f"Phase 0.5 done. {len(out_index)} episodes → {SG_ENCODER_CACHE_DIR}")


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
      main_chw    : (3, 224, 224) float32   current main image
      wrist_chw   : (3, 224, 224) float32   current wrist image
      lang_tokens : (MAX_LANG_LEN,) int64
      lang_mask   : (MAX_LANG_LEN,) bool
      curr_main   : (PATCH_DIM,)   float32  encoder latent z[curr, :2048]
      curr_wrist  : (PATCH_DIM,)   float32  encoder latent z[curr, 2048:]
      sg_main     : (PATCH_DIM,)   float32  encoder latent z[sg, :2048]
      sg_wrist    : (PATCH_DIM,)   float32  encoder latent z[sg, 2048:]
      sg_state    : (PROPRIO_DIM,) float32  robot state at subgoal frame

    Reads raw images + lang from subgoal_cache_dir.
    Reads encoder latents + states + sg_frames from sg_enc_cache_dir
    (built by extract_sg_encoder_latents()).
    """

    def __init__(self, subgoal_cache_dir: str, sg_enc_cache_dir: str):
        sg_path  = Path(subgoal_cache_dir)
        enc_path = Path(sg_enc_cache_dir)

        enc_index = json.loads((enc_path / "index.json").read_text())
        sg_index  = {
            e["ep_idx"]: e
            for e in json.loads((sg_path / "index.json").read_text())
        }

        self.episodes = []
        print("Loading caches ...")
        for entry in enc_index:
            ep_idx  = entry["ep_idx"]
            sg_entry = sg_index.get(ep_idx)
            if sg_entry is None:
                continue

            enc_data = np.load(enc_path / entry["file"])
            sg_data  = np.load(sg_path  / sg_entry["file"])

            self.episodes.append({
                "ep_idx":      ep_idx,
                "n_steps":     entry["n_steps"],
                "sg_file":     str(sg_path / sg_entry["file"]),
                "z":           enc_data["z"].astype(np.float32),        # (N, 4096)
                "states":      enc_data["states"].astype(np.float32),   # (N, 8)
                "sg_frames":   enc_data["sg_frames"].astype(np.int32),  # (N,)
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
                sg_idx  = int(ep["sg_frames"][step_idx])
                z_curr  = ep["z"][step_idx]        # (4096,)
                z_sg    = ep["z"][sg_idx]          # (4096,)
                yield (
                    _img_to_chw(main_imgs[step_idx]),                   # (3, H, W)
                    _img_to_chw(wrist_imgs[step_idx]),                  # (3, H, W)
                    ep["lang_tokens"],                                   # (T,)
                    ep["lang_mask"],                                     # (T,)
                    torch.from_numpy(z_curr[:PATCH_DIM].copy()),        # (2048,) curr_main
                    torch.from_numpy(z_curr[PATCH_DIM:].copy()),        # (2048,) curr_wrist
                    torch.from_numpy(z_sg[:PATCH_DIM].copy()),          # (2048,) sg_main
                    torch.from_numpy(z_sg[PATCH_DIM:].copy()),          # (2048,) sg_wrist
                    torch.from_numpy(ep["states"][sg_idx].copy()),      # (8,)
                    torch.tensor(sg_idx - step_idx, dtype=torch.float32),  # () horizon
                )


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def _apply_lora(model: PI0WithGoalExpert):
    """Apply LoRA to PaliGemma language model attention layers after base weights are loaded."""
    from peft import get_peft_model, LoraConfig
    lora_cfg = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = 0.0,   # 0 so eval() on backbone doesn't affect training
        bias           = "none",
    )
    model.paligemma_with_expert.paligemma.language_model = get_peft_model(
        model.paligemma_with_expert.paligemma.language_model,
        lora_cfg,
    )
    n_lora = sum(
        p.numel() for p in model.paligemma_with_expert.parameters()
        if p.requires_grad
    )
    print(f"  LoRA applied (r={LORA_RANK}): {n_lora/1e6:.2f}M trainable backbone params")


def _load_model() -> PI0WithGoalExpert:
    from openpi.training import config as _config
    import safetensors.torch

    print("Loading PI0WithGoalExpert ...")
    train_cfg = _config.get_config("pi05_libero")
    model = PI0WithGoalExpert(
        config         = train_cfg.model,
        patch_dim      = PATCH_DIM,
        proprio_dim    = PROPRIO_DIM,
        freeze_pi0     = True,
        expert_variant = "gemma_300m",
    )
    # 1. Load PI0 base weights first (before LoRA renames keys)
    safetensors.torch.load_model(
        model,
        os.path.join(PI05_CKPT_DIR, "model.safetensors"),
        strict=False,
    )
    # 2. Apply LoRA (adds trainable A/B matrices, base weights stay frozen)
    _apply_lora(model)
    # 3. Gradient checkpointing to offset memory cost of LoRA backprop through prefix
    model.paligemma_with_expert.paligemma.language_model.gradient_checkpointing_enable()
    # 4. Compile only the goal expert (LoRA layers are too small to benefit from compile)
    model.goal_expert = torch.compile(model.goal_expert)
    print("  Model loaded (LoRA on PaliGemma LM + goal_expert compiled).")
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

    dataset = GoalExpertDataset(SUBGOAL_CACHE_DIR, SG_ENCODER_CACHE_DIR)
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
        main_img, wrist_img, lang_tokens, lang_mask, \
            curr_main, curr_wrist, sg_main, sg_wrist, sg_state, horizon = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()
        loss, info = model(main_img, wrist_img, lang_tokens, lang_mask,
                           curr_main, curr_wrist, sg_main, sg_wrist, sg_state,
                           horizon=horizon)
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
                hor=f"{info['loss/horizon']:.4f}",
                lr=f"{lr:.2e}",
            )

            if (step + 1) % LOG_STEPS == 0:
                print(
                    f"step {step+1:6d}/{MAX_STEPS} | "
                    f"loss={info['loss/total']:.4f} "
                    f"main={info['loss/sg_main']:.4f} "
                    f"wrist={info['loss/sg_wrist']:.4f} "
                    f"state={info['loss/sg_state']:.4f} "
                    f"hor={info['loss/horizon']:.4f} "
                    f"lr={lr:.2e}"
                )
                wandb.log(
                    {
                        "train/loss_total":   info["loss/total"],
                        "train/loss_main":    info["loss/sg_main"],
                        "train/loss_wrist":   info["loss/sg_wrist"],
                        "train/loss_state":   info["loss/sg_state"],
                        "train/loss_horizon": info["loss/horizon"],
                        "train/lr":           lr,
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
        extract_images_and_lang()      # Phase 0:   raw images + language cache
        extract_sg_encoder_latents()   # Phase 0.5: encoder latents cache
    train()
