"""
Evaluate Executor on two conditions:

  Test 1 — gt:   subgoal from feature cache (ground truth annotation)
  Test 2 — gen:  subgoal from PI0WithGoalExpert (flow-matching decoder)

Usage:
  python eval_executor.py
  python eval_executor.py --max_episodes 20   # limit for Test 2 (slow)
"""

import os, sys, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "openpi" / "src"))

from model.executor        import Executor
from model.pi0_subgoal_decoder import PI0WithGoalExpert

# ---------------------------------------------------------------------------
# Config (match train.py / train_subgoal_decoder.py)
# ---------------------------------------------------------------------------

FEAT_CACHE_DIR    = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache"
SUBGOAL_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/subgoal_decoder_cache"

EXECUTOR_CKPT   = "./checkpoints/executor/checkpoint.pt"
GOAL_EXPERT_CKPT = "./checkpoints/subgoal_decoder/checkpoint.pt"
PI05_CKPT_DIR   = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"

PATCH_DIM   = 2048
PROPRIO_DIM = 8
ACTION_DIM  = 7
HIDDEN_DIM  = 512
NUM_LAYERS  = 5
MAX_LANG_LEN = 48

EVAL_BATCH  = 32   # batch size for executor forward
GOAL_BATCH  = 8    # batch size for goal expert (runs PaliGemma, more expensive)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_executor() -> Executor:
    model = Executor(
        num_imgs=4, patch_dim=PATCH_DIM, proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_LAYERS,
    ).to(DEVICE)
    ckpt = torch.load(EXECUTOR_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Executor loaded (step {step})")
    return model


def load_goal_expert() -> PI0WithGoalExpert:
    from openpi.training import config as _config
    import safetensors.torch

    train_cfg = _config.get_config("pi05_libero")
    model = PI0WithGoalExpert(
        config=train_cfg.model, patch_dim=PATCH_DIM,
        proprio_dim=PROPRIO_DIM, freeze_pi0=True,
    ).to(DEVICE)

    # Load PI0 backbone weights
    safetensors.torch.load_model(
        model, os.path.join(PI05_CKPT_DIR, "model.safetensors"), strict=False,
    )
    # Load goal expert weights
    ckpt = torch.load(GOAL_EXPERT_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"GoalExpert loaded (step {step})")
    return model


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_episodes(feat_cache_dir: str, subgoal_cache_dir: str | None = None,
                  max_episodes: int | None = None):
    """
    Returns list of episode dicts.
    If subgoal_cache_dir given, also loads raw images + lang tokens for Test 2.
    """
    feat_path = Path(feat_cache_dir)
    index = json.loads((feat_path / "index.json").read_text())
    if max_episodes:
        index = index[:max_episodes]

    episodes = []
    for entry in index:
        ep_idx   = entry["ep_idx"]
        feat     = np.load(feat_path / entry["file"])
        ep = {
            "ep_idx":      ep_idx,
            "n_steps":     entry["n_steps"],
            "main_feats":  feat["main_feats"].astype(np.float32),   # (N, 2048)
            "wrist_feats": feat["wrist_feats"].astype(np.float32),  # (N, 2048)
            "states":      feat["states"].astype(np.float32),       # (N, 8)
            "actions":     feat["actions"].astype(np.float32),      # (N, 7)
            "sg_frames":   feat["sg_frames"].astype(np.int32),      # (N,)
        }
        if subgoal_cache_dir is not None:
            sg_file = Path(subgoal_cache_dir) / f"ep_{ep_idx:04d}.npz"
            if sg_file.exists():
                sg = np.load(sg_file)
                ep["main_imgs"]  = sg["main_imgs"]   # (N, 224, 224, 3) uint8
                ep["wrist_imgs"] = sg["wrist_imgs"]
                ep["lang_tokens"] = torch.from_numpy(sg["lang_tokens"].astype(np.int64))
                ep["lang_mask"]   = torch.from_numpy(sg["lang_mask"].astype(bool))
            else:
                ep["main_imgs"] = None  # flag: skip this ep in Test 2
        episodes.append(ep)

    print(f"Loaded {len(episodes)} episodes")
    return episodes


def _img_to_chw(imgs_uint8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, H, W) float32 [-1, 1]."""
    t = torch.from_numpy(imgs_uint8).float() / 255.0
    return (t * 2.0 - 1.0).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Test 1: ground-truth subgoal
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_gt(executor: Executor, episodes: list) -> dict:
    """Use annotated subgoal frames directly from feature cache."""
    all_l1, all_per_dim = [], []

    for ep in tqdm(episodes, desc="Test 1 (gt subgoal)"):
        N = ep["n_steps"]
        sg_idx = ep["sg_frames"]                          # (N,)

        curr_main  = torch.from_numpy(ep["main_feats"])   # (N, 2048)
        curr_wrist = torch.from_numpy(ep["wrist_feats"])
        curr_state = torch.from_numpy(ep["states"])
        sg_main    = torch.from_numpy(ep["main_feats"][sg_idx])
        sg_wrist   = torch.from_numpy(ep["wrist_feats"][sg_idx])
        sg_state   = torch.from_numpy(ep["states"][sg_idx])
        actions    = torch.from_numpy(ep["actions"])

        # Batch forward
        l1_ep = []
        for i in range(0, N, EVAL_BATCH):
            sl = slice(i, i + EVAL_BATCH)
            imgs = torch.stack([
                curr_main[sl], curr_wrist[sl], sg_main[sl], sg_wrist[sl]
            ], dim=1).to(DEVICE)                          # (B, 4, 2048)
            pred, _, _ = executor(
                imgs,
                curr_state[sl].to(DEVICE),
                sg_state[sl].to(DEVICE),
                deterministic=True,
            )
            l1 = F.l1_loss(pred, actions[sl].to(DEVICE), reduction="none")  # (B, 7)
            l1_ep.append(l1.cpu())

        l1_ep = torch.cat(l1_ep, dim=0)  # (N, 7)
        all_l1.append(l1_ep.mean().item())
        all_per_dim.append(l1_ep.mean(dim=0))

    per_dim = torch.stack(all_per_dim).mean(dim=0)  # (7,)
    return {
        "mean_l1":  float(np.mean(all_l1)),
        "per_dim":  per_dim.tolist(),
    }


# ---------------------------------------------------------------------------
# Test 2: goal-expert generated subgoal
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_generated(executor: Executor, goal_expert: PI0WithGoalExpert,
                   episodes: list) -> dict:
    """Generate subgoal with goal expert, then run executor."""
    all_l1, all_per_dim = [], []

    for ep in tqdm(episodes, desc="Test 2 (gen subgoal)"):
        if ep.get("main_imgs") is None:
            continue

        N          = ep["n_steps"]
        curr_main  = torch.from_numpy(ep["main_feats"])
        curr_wrist = torch.from_numpy(ep["wrist_feats"])
        curr_state = torch.from_numpy(ep["states"])
        actions    = torch.from_numpy(ep["actions"])

        # Generate subgoals in batches (PaliGemma is expensive)
        sg_main_list, sg_wrist_list, sg_state_list = [], [], []
        lang_tok  = ep["lang_tokens"].unsqueeze(0)  # (1, T) → broadcast
        lang_mask = ep["lang_mask"].unsqueeze(0)

        for i in range(0, N, GOAL_BATCH):
            sl   = slice(i, i + GOAL_BATCH)
            B    = min(GOAL_BATCH, N - i)
            imgs = _img_to_chw(ep["main_imgs"][sl]).to(DEVICE)    # (B, 3, H, W)
            wrist= _img_to_chw(ep["wrist_imgs"][sl]).to(DEVICE)

            sg_m, sg_w, sg_s = goal_expert.sample_goal(
                imgs,
                wrist,
                lang_tok.expand(B, -1).to(DEVICE),
                lang_mask.expand(B, -1).to(DEVICE),
            )
            sg_main_list.append(sg_m.cpu())
            sg_wrist_list.append(sg_w.cpu())
            sg_state_list.append(sg_s.cpu())

        sg_main  = torch.cat(sg_main_list,  dim=0)  # (N, 2048)
        sg_wrist = torch.cat(sg_wrist_list, dim=0)
        sg_state = torch.cat(sg_state_list, dim=0)  # (N, 8)

        # Executor forward with generated subgoal
        l1_ep = []
        for i in range(0, N, EVAL_BATCH):
            sl = slice(i, i + EVAL_BATCH)
            imgs = torch.stack([
                curr_main[sl], curr_wrist[sl], sg_main[sl], sg_wrist[sl]
            ], dim=1).to(DEVICE)
            pred, _, _ = executor(
                imgs,
                curr_state[sl].to(DEVICE),
                sg_state[sl].to(DEVICE),
                deterministic=True,
            )
            l1 = F.l1_loss(pred, actions[sl].to(DEVICE), reduction="none")  # (B, 7)
            l1_ep.append(l1.cpu())

        l1_ep = torch.cat(l1_ep, dim=0)  # (N, 7)
        all_l1.append(l1_ep.mean().item())
        all_per_dim.append(l1_ep.mean(dim=0))

    per_dim = torch.stack(all_per_dim).mean(dim=0)
    return {
        "mean_l1": float(np.mean(all_l1)),
        "per_dim": per_dim.tolist(),
    }


# ---------------------------------------------------------------------------
# Test 3: black (zero) subgoal — baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_black(executor: Executor, episodes: list) -> dict:
    """Replace subgoal with all-zeros to measure executor's reliance on subgoal."""
    all_l1, all_per_dim = [], []

    for ep in tqdm(episodes, desc="Test 3 (black subgoal)"):
        N = ep["n_steps"]

        curr_main  = torch.from_numpy(ep["main_feats"])
        curr_wrist = torch.from_numpy(ep["wrist_feats"])
        curr_state = torch.from_numpy(ep["states"])
        actions    = torch.from_numpy(ep["actions"])

        sg_main  = torch.zeros(N, PATCH_DIM)
        sg_wrist = torch.zeros(N, PATCH_DIM)
        sg_state = torch.zeros(N, PROPRIO_DIM)

        l1_ep = []
        for i in range(0, N, EVAL_BATCH):
            sl = slice(i, i + EVAL_BATCH)
            imgs = torch.stack([
                curr_main[sl], curr_wrist[sl], sg_main[sl], sg_wrist[sl]
            ], dim=1).to(DEVICE)
            pred, _, _ = executor(
                imgs,
                curr_state[sl].to(DEVICE),
                sg_state[sl].to(DEVICE),
                deterministic=True,
            )
            l1 = F.l1_loss(pred, actions[sl].to(DEVICE), reduction="none")
            l1_ep.append(l1.cpu())

        l1_ep = torch.cat(l1_ep, dim=0)
        all_l1.append(l1_ep.mean().item())
        all_per_dim.append(l1_ep.mean(dim=0))

    per_dim = torch.stack(all_per_dim).mean(dim=0)
    return {
        "mean_l1": float(np.mean(all_l1)),
        "per_dim": per_dim.tolist(),
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of episodes (useful for Test 2 which is slow)")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip Test 2 (goal expert generation)")
    args = parser.parse_args()

    executor = load_executor()
    episodes = load_episodes(FEAT_CACHE_DIR, max_episodes=args.max_episodes)

    # Test 1: ground truth subgoal
    gt_results = eval_gt(executor, episodes)

    # Test 2: generated subgoal
    if not args.skip_gen:
        goal_expert = load_goal_expert()
        episodes_with_imgs = load_episodes(
            FEAT_CACHE_DIR, SUBGOAL_CACHE_DIR, max_episodes=args.max_episodes
        )
        gen_results = eval_generated(executor, goal_expert, episodes_with_imgs)
    else:
        gen_results = None

    # Test 3: black (zero) subgoal
    black_results = eval_black(executor, episodes)

    # Print results
    dim_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    print("\n" + "=" * 70)
    print(f"{'Condition':<22} {'Mean L1':>8}   per-dim L1")
    print("=" * 70)

    def _row(name, res):
        dims = "  ".join(f"{v:.4f}" for v in res["per_dim"])
        print(f"{name:<22} {res['mean_l1']:>8.4f}   {dims}")

    _row("GT subgoal",    gt_results)
    if gen_results:
        _row("Gen subgoal",   gen_results)
    _row("Black subgoal", black_results)

    print("-" * 70)
    if gen_results:
        d_gen   = gen_results["mean_l1"]   - gt_results["mean_l1"]
        d_black = black_results["mean_l1"] - gt_results["mean_l1"]
        print(f"  Delta gen   - gt : {d_gen:+.4f}")
        print(f"  Delta black - gt : {d_black:+.4f}")
    else:
        d_black = black_results["mean_l1"] - gt_results["mean_l1"]
        print(f"  Delta black - gt : {d_black:+.4f}")

    print("=" * 70)
    print(f"  Dims: {dim_names}")


if __name__ == "__main__":
    main()
