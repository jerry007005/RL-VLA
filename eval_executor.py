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

FEAT_CACHE_DIR    = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache/libero_10_no_noops"
SUBGOAL_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/subgoal_decoder_cache"
SG_ENC_CACHE_DIR  = "/mnt/nfs/Users/jerry007005/dataset/sg_encoder_cache"

EXECUTOR_CKPT   = "./checkpoints/v3/executor/checkpoint.pt"
GOAL_EXPERT_CKPT = "./checkpoints/v3/goal_expert/checkpoint.pt"
PI05_CKPT_DIR   = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"
NORM_STATS_PATH = os.path.join(
    PI05_CKPT_DIR, "assets", "physical-intelligence", "libero", "norm_stats.json"
)

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
        norm_stats_path=NORM_STATS_PATH,
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
        expert_variant="gemma_300m",
        norm_stats_path=NORM_STATS_PATH,
    ).to(DEVICE)

    # Load PI0 base weights (frozen backbone)
    safetensors.torch.load_model(
        model, os.path.join(PI05_CKPT_DIR, "model.safetensors"), strict=False,
    )
    # Load trained goal expert weights (strip torch.compile prefix)
    ckpt = torch.load(GOAL_EXPERT_CKPT, map_location=DEVICE)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_trainable_state(state)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"GoalExpert loaded (step {step})")
    return model


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_episodes(feat_cache_dir: str, sg_enc_cache_dir: str,
                  subgoal_cache_dir: str | None = None,
                  max_episodes: int | None = None):
    """
    Returns list of episode dicts.
    feat_cache_dir    : executor_feat_cache  (z, actions)
    sg_enc_cache_dir  : sg_encoder_cache     (z, states, sg_frames)
    subgoal_cache_dir : subgoal_decoder_cache (raw imgs + lang) for Test 2
    """
    feat_path = Path(feat_cache_dir)
    enc_path  = Path(sg_enc_cache_dir)

    feat_index = json.loads((feat_path / "index.json").read_text())
    enc_index  = {e["ep_idx"]: e for e in json.loads((enc_path / "index.json").read_text())}

    if max_episodes:
        feat_index = feat_index[:max_episodes]

    episodes = []
    for entry in feat_index:
        ep_idx = entry["ep_idx"]
        if ep_idx not in enc_index:
            continue

        feat     = np.load(feat_path / entry["file"])
        enc_data = np.load(enc_path  / enc_index[ep_idx]["file"])

        z         = enc_data["z"].astype(np.float32)          # (N, 4096)
        ep = {
            "ep_idx":    ep_idx,
            "n_steps":   entry["n_steps"],
            "z":         z,                                    # (N, 4096)
            "states":    enc_data["states"].astype(np.float32),  # (N, 8)
            "actions":   feat["actions"].astype(np.float32),     # (N, 7)
            "sg_frames": enc_data["sg_frames"].astype(np.int32), # (N,)
        }
        if subgoal_cache_dir is not None:
            sg_file = Path(subgoal_cache_dir) / f"ep_{ep_idx:04d}.npz"
            if sg_file.exists():
                sg = np.load(sg_file)
                ep["main_imgs"]   = sg["main_imgs"]
                ep["wrist_imgs"]  = sg["wrist_imgs"]
                ep["lang_tokens"] = torch.from_numpy(sg["lang_tokens"].astype(np.int64))
                ep["lang_mask"]   = torch.from_numpy(sg["lang_mask"].astype(bool))
            else:
                ep["main_imgs"] = None
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
        sg_idx = ep["sg_frames"]                           # (N,)

        z         = ep["z"]                                # (N, 4096)
        curr_main  = torch.from_numpy(z[:, :PATCH_DIM])   # (N, 2048)
        curr_wrist = torch.from_numpy(z[:, PATCH_DIM:])   # (N, 2048)
        curr_state = torch.from_numpy(ep["states"])
        sg_main    = torch.from_numpy(z[sg_idx, :PATCH_DIM])
        sg_wrist   = torch.from_numpy(z[sg_idx, PATCH_DIM:])
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
        z          = ep["z"]                               # (N, 4096)
        curr_main  = torch.from_numpy(z[:, :PATCH_DIM])
        curr_wrist = torch.from_numpy(z[:, PATCH_DIM:])
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
            curr_z = torch.from_numpy(z[sl]).to(DEVICE)           # (B, 4096)

            sg_m, sg_w, sg_s, _ = goal_expert.sample_goal(
                imgs,
                wrist,
                lang_tok.expand(B, -1).to(DEVICE),
                lang_mask.expand(B, -1).to(DEVICE),
                curr_z,
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
# Test 3: generated image embeddings + ground-truth subgoal state
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_gen_img_gt_state(executor: Executor, goal_expert: PI0WithGoalExpert,
                          episodes: list) -> dict:
    """Gen subgoal image embeddings + GT subgoal state."""
    all_l1, all_per_dim = [], []

    for ep in tqdm(episodes, desc="Test 3 (gen img / gt state)"):
        if ep.get("main_imgs") is None:
            continue

        N         = ep["n_steps"]
        z         = ep["z"]
        curr_main  = torch.from_numpy(z[:, :PATCH_DIM])
        curr_wrist = torch.from_numpy(z[:, PATCH_DIM:])
        curr_state = torch.from_numpy(ep["states"])
        actions    = torch.from_numpy(ep["actions"])
        sg_frames  = ep["sg_frames"]
        sg_state   = torch.from_numpy(ep["states"][sg_frames])  # GT state

        lang_tok  = ep["lang_tokens"].unsqueeze(0)
        lang_mask = ep["lang_mask"].unsqueeze(0)

        sg_main_list, sg_wrist_list = [], []
        for i in range(0, N, GOAL_BATCH):
            sl = slice(i, i + GOAL_BATCH)
            B  = min(GOAL_BATCH, N - i)
            imgs  = _img_to_chw(ep["main_imgs"][sl]).to(DEVICE)
            wrist = _img_to_chw(ep["wrist_imgs"][sl]).to(DEVICE)
            sg_m, sg_w, _, _ = goal_expert.sample_goal(
                imgs, wrist,
                lang_tok.expand(B, -1).to(DEVICE),
                lang_mask.expand(B, -1).to(DEVICE),
                torch.from_numpy(z[sl]).to(DEVICE),
            )
            sg_main_list.append(sg_m.cpu())
            sg_wrist_list.append(sg_w.cpu())

        sg_main  = torch.cat(sg_main_list,  dim=0)
        sg_wrist = torch.cat(sg_wrist_list, dim=0)

        l1_ep = []
        for i in range(0, N, EVAL_BATCH):
            sl = slice(i, i + EVAL_BATCH)
            imgs = torch.stack([
                curr_main[sl], curr_wrist[sl], sg_main[sl], sg_wrist[sl]
            ], dim=1).to(DEVICE)
            pred, _, _ = executor(
                imgs, curr_state[sl].to(DEVICE), sg_state[sl].to(DEVICE),
                deterministic=True,
            )
            l1 = F.l1_loss(pred, actions[sl].to(DEVICE), reduction="none")
            l1_ep.append(l1.cpu())

        l1_ep = torch.cat(l1_ep, dim=0)
        all_l1.append(l1_ep.mean().item())
        all_per_dim.append(l1_ep.mean(dim=0))

    per_dim = torch.stack(all_per_dim).mean(dim=0)
    return {"mean_l1": float(np.mean(all_l1)), "per_dim": per_dim.tolist()}


# ---------------------------------------------------------------------------
# Test 4: ground-truth image embeddings + generated subgoal state
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_gt_img_gen_state(executor: Executor, goal_expert: PI0WithGoalExpert,
                          episodes: list) -> dict:
    """GT subgoal image embeddings + gen subgoal state."""
    all_l1, all_per_dim = [], []

    for ep in tqdm(episodes, desc="Test 4 (gt img / gen state)"):
        if ep.get("main_imgs") is None:
            continue

        N         = ep["n_steps"]
        z         = ep["z"]
        curr_main  = torch.from_numpy(z[:, :PATCH_DIM])
        curr_wrist = torch.from_numpy(z[:, PATCH_DIM:])
        curr_state = torch.from_numpy(ep["states"])
        actions    = torch.from_numpy(ep["actions"])
        sg_frames  = ep["sg_frames"]
        sg_main    = torch.from_numpy(z[sg_frames, :PATCH_DIM])  # GT img
        sg_wrist   = torch.from_numpy(z[sg_frames, PATCH_DIM:])  # GT img

        lang_tok  = ep["lang_tokens"].unsqueeze(0)
        lang_mask = ep["lang_mask"].unsqueeze(0)

        sg_state_list = []
        for i in range(0, N, GOAL_BATCH):
            sl = slice(i, i + GOAL_BATCH)
            B  = min(GOAL_BATCH, N - i)
            imgs  = _img_to_chw(ep["main_imgs"][sl]).to(DEVICE)
            wrist = _img_to_chw(ep["wrist_imgs"][sl]).to(DEVICE)
            _, _, sg_s, _ = goal_expert.sample_goal(
                imgs, wrist,
                lang_tok.expand(B, -1).to(DEVICE),
                lang_mask.expand(B, -1).to(DEVICE),
                torch.from_numpy(z[sl]).to(DEVICE),
            )
            sg_state_list.append(sg_s.cpu())

        sg_state = torch.cat(sg_state_list, dim=0)

        l1_ep = []
        for i in range(0, N, EVAL_BATCH):
            sl = slice(i, i + EVAL_BATCH)
            imgs = torch.stack([
                curr_main[sl], curr_wrist[sl], sg_main[sl], sg_wrist[sl]
            ], dim=1).to(DEVICE)
            pred, _, _ = executor(
                imgs, curr_state[sl].to(DEVICE), sg_state[sl].to(DEVICE),
                deterministic=True,
            )
            l1 = F.l1_loss(pred, actions[sl].to(DEVICE), reduction="none")
            l1_ep.append(l1.cpu())

        l1_ep = torch.cat(l1_ep, dim=0)
        all_l1.append(l1_ep.mean().item())
        all_per_dim.append(l1_ep.mean(dim=0))

    per_dim = torch.stack(all_per_dim).mean(dim=0)
    return {"mean_l1": float(np.mean(all_l1)), "per_dim": per_dim.tolist()}


# ---------------------------------------------------------------------------
# Test 5: black (zero) subgoal — baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_black(executor: Executor, episodes: list) -> dict:
    """Replace subgoal with all-zeros to measure executor's reliance on subgoal."""
    all_l1, all_per_dim = [], []

    for ep in tqdm(episodes, desc="Test 5 (black subgoal)"):
        N = ep["n_steps"]

        z          = ep["z"]                               # (N, 4096)
        curr_main  = torch.from_numpy(z[:, :PATCH_DIM])
        curr_wrist = torch.from_numpy(z[:, PATCH_DIM:])
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
    episodes = load_episodes(FEAT_CACHE_DIR, SG_ENC_CACHE_DIR, max_episodes=args.max_episodes)

    # Test 1: ground truth subgoal
    gt_results = eval_gt(executor, episodes)

    # Tests 2/3/4 all need goal expert + raw images
    gen_results = gen_img_gt_state_results = gt_img_gen_state_results = None
    if not args.skip_gen:
        goal_expert = load_goal_expert()
        episodes_with_imgs = load_episodes(
            FEAT_CACHE_DIR, SG_ENC_CACHE_DIR, SUBGOAL_CACHE_DIR, max_episodes=args.max_episodes
        )
        # Test 2: gen img + gen state
        gen_results = eval_generated(executor, goal_expert, episodes_with_imgs)
        # Test 3: gen img + GT state
        gen_img_gt_state_results = eval_gen_img_gt_state(executor, goal_expert, episodes_with_imgs)
        # Test 4: GT img + gen state
        gt_img_gen_state_results = eval_gt_img_gen_state(executor, goal_expert, episodes_with_imgs)

    # Test 5: black (zero) subgoal
    black_results = eval_black(executor, episodes)

    # Print results
    dim_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    print("\n" + "=" * 80)
    print(f"{'Condition':<28} {'Mean L1':>8}   per-dim L1")
    print("=" * 80)

    def _row(name, res):
        dims = "  ".join(f"{v:.4f}" for v in res["per_dim"])
        print(f"{name:<28} {res['mean_l1']:>8.4f}   {dims}")

    _row("T1  GT img + GT state",       gt_results)
    if gen_results:
        _row("T2  Gen img + gen state",  gen_results)
    if gen_img_gt_state_results:
        _row("T3  Gen img + GT state",   gen_img_gt_state_results)
    if gt_img_gen_state_results:
        _row("T4  GT img + gen state",   gt_img_gen_state_results)
    _row("T5  Black subgoal",            black_results)

    print("-" * 80)
    ref = gt_results["mean_l1"]
    if gen_results:
        print(f"  T2 - T1 (full gen gap)       : {gen_results['mean_l1']           - ref:+.4f}")
    if gen_img_gt_state_results:
        print(f"  T3 - T1 (img gen gap)        : {gen_img_gt_state_results['mean_l1'] - ref:+.4f}")
    if gt_img_gen_state_results:
        print(f"  T4 - T1 (state gen gap)      : {gt_img_gen_state_results['mean_l1'] - ref:+.4f}")
    print(f"  T5 - T1 (no subgoal gap)     : {black_results['mean_l1']           - ref:+.4f}")
    print("=" * 80)
    print(f"  Dims: {dim_names}")


if __name__ == "__main__":
    main()
