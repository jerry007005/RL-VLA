"""
Diagnose SigLIP feature statistics from executor_feat_cache.
CPU-only, no GPU used.

Checks:
  1. L2 norm distribution of main_feats / wrist_feats
  2. Per-dim mean vector magnitude (dataset-level bias)
  3. Per-dim std
  4. Cosine similarity between current-frame and subgoal-frame features
  5. sg_frame distance (how far ahead is the subgoal)
"""

import json
import numpy as np
from pathlib import Path

FEAT_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache"
MAX_EPISODES   = 50   # sample this many episodes (full dataset = 291)
RNG_SEED       = 0

feat_path = Path(FEAT_CACHE_DIR)
all_files = sorted(feat_path.glob("ep_*.npz"))

rng = np.random.default_rng(RNG_SEED)
selected = rng.choice(all_files, size=min(MAX_EPISODES, len(all_files)), replace=False)
selected = sorted(selected)

print(f"Sampling {len(selected)} / {len(all_files)} episodes ...\n")

# Accumulators
all_main_sg   = []   # main_feats[sg_frame]   — the actual flow-matching targets
all_wrist_sg  = []   # wrist_feats[sg_frame]
all_states_sg = []   # states[sg_frame]
all_cos_main  = []   # cosine(curr_main, sg_main) per step
all_sg_dist   = []   # sg_frame - step_idx  (how far ahead)

for ep_file in selected:
    d = np.load(ep_file)
    main_feats  = d["main_feats"].astype(np.float32)   # (N, 2048)
    wrist_feats = d["wrist_feats"].astype(np.float32)  # (N, 2048)
    states      = d["states"].astype(np.float32)        # (N, 8)
    sg_frames   = d["sg_frames"].astype(np.int32)       # (N,)
    N = len(sg_frames)

    for t in range(N):
        sg_idx = int(sg_frames[t])
        all_main_sg.append(main_feats[sg_idx])
        all_wrist_sg.append(wrist_feats[sg_idx])
        all_states_sg.append(states[sg_idx])

        # cosine similarity between current frame and its subgoal frame
        curr = main_feats[t]
        sg   = main_feats[sg_idx]
        cos  = float(np.dot(curr, sg) / (np.linalg.norm(curr) * np.linalg.norm(sg) + 1e-8))
        all_cos_main.append(cos)
        all_sg_dist.append(sg_idx - t)

all_main_sg   = np.stack(all_main_sg)    # (M, 2048)
all_wrist_sg  = np.stack(all_wrist_sg)   # (M, 2048)
all_states_sg = np.stack(all_states_sg)  # (M, 8)
all_cos_main  = np.array(all_cos_main)
all_sg_dist   = np.array(all_sg_dist)

print("=" * 60)
print("  main_feats (sg targets)  —  shape:", all_main_sg.shape)
print("=" * 60)

norms = np.linalg.norm(all_main_sg, axis=-1)
print(f"  L2 norm  : mean={norms.mean():.2f}  std={norms.std():.2f}"
      f"  min={norms.min():.2f}  max={norms.max():.2f}")

# sqrt(2048) is the expected norm if each dim ~ N(0,1)
expected_norm = 2048 ** 0.5
print(f"  (if N(0,1) per dim, expected norm = sqrt(2048) = {expected_norm:.1f})")

dataset_mean = all_main_sg.mean(axis=0)   # (2048,)
mean_norm    = np.linalg.norm(dataset_mean)
per_dim_std  = all_main_sg.std(axis=0)    # (2048,)
print(f"  Dataset-mean L2 norm   : {mean_norm:.2f}")
print(f"    (if large → features have big shared bias across all episodes)")
print(f"  Per-dim std  : mean={per_dim_std.mean():.4f}  "
      f"min={per_dim_std.min():.4f}  max={per_dim_std.max():.4f}")

# How much of the total variance is explained by the mean vector?
total_var   = (all_main_sg ** 2).mean()
mean_sq     = (dataset_mean ** 2).mean()
residual_sq = ((all_main_sg - dataset_mean) ** 2).mean()
print(f"  Variance decomposition:")
print(f"    Total E[||x||²/D]      = {total_var:.4f}")
print(f"    Mean  E[||μ||²/D]      = {mean_sq:.4f}  ({100*mean_sq/total_var:.1f}% of total)")
print(f"    Residual E[||x-μ||²/D] = {residual_sq:.4f}  ({100*residual_sq/total_var:.1f}% of total)")
print()

print("=" * 60)
print("  wrist_feats (sg targets)")
print("=" * 60)
norms_w  = np.linalg.norm(all_wrist_sg, axis=-1)
mean_w   = all_wrist_sg.mean(axis=0)
mean_norm_w = np.linalg.norm(mean_w)
total_var_w = (all_wrist_sg ** 2).mean()
mean_sq_w   = (mean_w ** 2).mean()
res_sq_w    = ((all_wrist_sg - mean_w) ** 2).mean()
print(f"  L2 norm  : mean={norms_w.mean():.2f}  std={norms_w.std():.2f}")
print(f"  Dataset-mean L2 norm   : {mean_norm_w:.2f}")
print(f"  Variance decomposition:")
print(f"    Total E[||x||²/D]      = {total_var_w:.4f}")
print(f"    Mean  E[||μ||²/D]      = {mean_sq_w:.4f}  ({100*mean_sq_w/total_var_w:.1f}%)")
print(f"    Residual E[||x-μ||²/D] = {res_sq_w:.4f}  ({100*res_sq_w/total_var_w:.1f}%)")
print()

print("=" * 60)
print("  states (sg targets)  —  8-dim")
print("=" * 60)
states_mean = all_states_sg.mean(axis=0)
states_std  = all_states_sg.std(axis=0)
for i, (m, s) in enumerate(zip(states_mean, states_std)):
    print(f"  dim {i}: mean={m:+.4f}  std={s:.4f}")
state_total = (all_states_sg ** 2).mean()
print(f"  E[||s||²/8] = {state_total:.4f}")
print()

print("=" * 60)
print("  Subgoal distance & cosine similarity")
print("=" * 60)
print(f"  sg_frame - step_idx : mean={all_sg_dist.mean():.1f}  "
      f"std={all_sg_dist.std():.1f}  "
      f"min={all_sg_dist.min()}  max={all_sg_dist.max()}")
print(f"  cos(curr_main, sg_main) : mean={all_cos_main.mean():.4f}  "
      f"std={all_cos_main.std():.4f}  "
      f"min={all_cos_main.min():.4f}  max={all_cos_main.max():.4f}")
print()

print("=" * 60)
print("  Flow-matching baseline loss (random velocity predictor)")
print("=" * 60)
# noise ~ N(0,1), u = noise - target
# E[MSE(0, u)] = E[||noise - target||²/D] = 1 + E[||target||²/D]  (noise ⊥ target)
baseline_main  = 1.0 + total_var
baseline_wrist = 1.0 + total_var_w
baseline_state = 1.0 + state_total
print(f"  main  : {baseline_main:.4f}  (predict-zero baseline)")
print(f"  wrist : {baseline_wrist:.4f}")
print(f"  state : {baseline_state:.4f}")
print(f"  Your reported loss ~0.6 means main/wrist are at "
      f"{100*0.6/baseline_main:.1f}% / {100*0.6/baseline_wrist:.1f}% of baseline")
print()

print("=" * 60)
print("  Recommendation summary")
print("=" * 60)
bias_ratio = mean_sq / total_var
if bias_ratio > 0.3:
    print(f"  [!] Dataset mean accounts for {100*bias_ratio:.0f}% of feature variance.")
    print(f"      Subtracting the mean (centering) will make flow-matching significantly easier.")
else:
    print(f"  [OK] Dataset mean bias is small ({100*bias_ratio:.0f}%). Centering won't help much.")

norm_mean = norms.mean()
if abs(norm_mean - expected_norm) / expected_norm > 0.3:
    print(f"  [!] Feature L2 norm ({norm_mean:.1f}) differs significantly from N(0,1) expectation "
          f"({expected_norm:.1f}). L2 normalization recommended.")
else:
    print(f"  [OK] Feature norm ({norm_mean:.1f}) is close to N(0,1) scale ({expected_norm:.1f}).")

cos_mean = all_cos_main.mean()
if cos_mean > 0.95:
    print(f"  [!] cos(curr, sg_main)={cos_mean:.3f} — subgoal frames look almost identical to"
          f" current frames. The task may be too easy or sg_frames annotation is off.")
elif cos_mean < 0.5:
    print(f"  [!] cos(curr, sg_main)={cos_mean:.3f} — very different. Hard prediction task.")
else:
    print(f"  [OK] cos(curr, sg_main)={cos_mean:.3f} — reasonable difficulty.")
