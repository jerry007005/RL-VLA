"""
Ablation eval: zero out each sg component to see which contributes most.
Uses GT subgoals from cache — no goal expert / SAE needed.
"""
import os, sys, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model.executor import Executor

FEAT_CACHE_DIR    = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache/libero_spatial_no_noops"
SG_ENC_CACHE_DIR  = "/mnt/nfs/Users/jerry007005/dataset/sg_encoder_cache"
EXECUTOR_CKPT     = "./checkpoints/executor/checkpoint.pt"
PI05_CKPT_DIR     = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"
NORM_STATS_PATH   = os.path.join(PI05_CKPT_DIR, "assets", "physical-intelligence", "libero", "norm_stats.json")

PATCH_DIM   = 2048
PROPRIO_DIM = 8
ACTION_DIM  = 7
HIDDEN_DIM  = 512
NUM_LAYERS  = 5
EVAL_BATCH  = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_executor() -> Executor:
    model = Executor(
        num_imgs=4, patch_dim=PATCH_DIM, proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_LAYERS,
        norm_stats_path=NORM_STATS_PATH,
    ).to(DEVICE)
    ckpt = torch.load(EXECUTOR_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Executor loaded (step {ckpt.get('step', '?')})")
    return model


def load_episodes(max_episodes: int | None = None):
    feat_path = Path(FEAT_CACHE_DIR)
    enc_path  = Path(SG_ENC_CACHE_DIR)
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
        episodes.append({
            "ep_idx":    ep_idx,
            "n_steps":   entry["n_steps"],
            "z":         enc_data["z"].astype(np.float32),
            "states":    enc_data["states"].astype(np.float32),
            "actions":   feat["actions"].astype(np.float32),
            "sg_frames": enc_data["sg_frames"].astype(np.int32),
        })
    print(f"Loaded {len(episodes)} episodes")
    return episodes


@torch.no_grad()
def eval_with_mask(
    executor: Executor, episodes: list,
    use_sg_main: bool = True,
    use_sg_wrist: bool = True,
    use_sg_state: bool = True,
    desc: str = "",
) -> dict:
    all_l1, all_per_dim = [], []
    for ep in tqdm(episodes, desc=desc):
        N = ep["n_steps"]
        sg_idx = ep["sg_frames"]
        z = ep["z"]
        curr_main  = torch.from_numpy(z[:, :PATCH_DIM])
        curr_wrist = torch.from_numpy(z[:, PATCH_DIM:])
        curr_state = torch.from_numpy(ep["states"])
        sg_main    = torch.from_numpy(z[sg_idx, :PATCH_DIM])  if use_sg_main  else torch.zeros(N, PATCH_DIM)
        sg_wrist   = torch.from_numpy(z[sg_idx, PATCH_DIM:])  if use_sg_wrist else torch.zeros(N, PATCH_DIM)
        sg_state   = torch.from_numpy(ep["states"][sg_idx])   if use_sg_state else torch.zeros(N, PROPRIO_DIM)
        actions    = torch.from_numpy(ep["actions"])

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
    return {"mean_l1": float(np.mean(all_l1)), "per_dim": per_dim.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    executor = load_executor()
    episodes = load_episodes(args.max_episodes)

    configs = [
        ("A  All GT (baseline)",      True,  True,  True),
        ("B  No sg_wrist",            True,  False, True),
        ("C  No sg_main",             False, True,  True),
        ("D  No sg_state",            True,  True,  False),
        ("E  Only sg_state",          False, False, True),
        ("F  Only sg_main",           True,  False, False),
        ("G  Only sg_wrist",          False, True,  False),
        ("H  Nothing (all zero sg)",  False, False, False),
    ]

    results = {}
    for name, um, uw, us in configs:
        results[name] = eval_with_mask(executor, episodes, um, uw, us, desc=name[:30])

    print("\n" + "=" * 90)
    print(f"{'Condition':<32} {'Mean L1':>8}   per-dim L1 (x y z rx ry rz gripper)")
    print("=" * 90)
    baseline = results[configs[0][0]]["mean_l1"]
    for name, *_ in configs:
        r = results[name]
        dims = "  ".join(f"{v:.4f}" for v in r["per_dim"])
        delta = r["mean_l1"] - baseline
        print(f"{name:<32} {r['mean_l1']:>8.4f}   {dims}  ({delta:+.4f})")
    print("=" * 90)


if __name__ == "__main__":
    main()
