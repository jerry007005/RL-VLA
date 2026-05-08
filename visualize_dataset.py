"""
Convert training episodes to MP4 videos.

For each episode in the subgoal decoder cache:
  - Side-by-side: main camera | wrist camera  (224 × 448)
  - Info bar at bottom: step number, subgoal frame index
  - Subgoal frames highlighted with a green border
  - Frames where the robot is "heading to" this subgoal are normal

Usage:
  python visualize_dataset.py                     # all episodes
  python visualize_dataset.py --ep_ids 0 1 2      # specific episodes
  python visualize_dataset.py --out_dir ./videos  # custom output dir
"""

import argparse
import json
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SUBGOAL_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/subgoal_decoder_cache"
FEAT_CACHE_DIR    = "/mnt/nfs/Users/jerry007005/dataset/executor_feat_cache"
DEFAULT_OUT_DIR   = "./videos/dataset"
FPS               = 10
BORDER            = 4    # px border thickness for subgoal highlight
BAR_H             = 32   # bottom info bar height (224+32=256, divisible by 16)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_frame(
    main_img:   np.ndarray,   # (224, 224, 3) uint8
    wrist_img:  np.ndarray,   # (224, 224, 3) uint8
    step:       int,
    n_steps:    int,
    sg_frame:   int,          # subgoal frame index for this step
    is_subgoal: bool,         # True if this step IS a subgoal frame
) -> np.ndarray:
    """Returns (224+BAR_H, 448, 3) uint8 frame."""
    H, W = main_img.shape[:2]

    # Green border on subgoal frames
    if is_subgoal:
        color = (0, 220, 80)
        for img in [main_img, wrist_img]:
            img[:BORDER,  :] = color
            img[-BORDER:, :] = color
            img[:, :BORDER]  = color
            img[:, -BORDER:] = color

    # Concatenate side by side
    frame = np.concatenate([main_img, wrist_img], axis=1)  # (H, 2W, 3)

    # Info bar
    bar = Image.new("RGB", (W * 2, BAR_H), color=(30, 30, 30))
    draw = ImageDraw.Draw(bar)
    label = f"step {step:3d}/{n_steps-1}  sg→{sg_frame}"
    if is_subgoal:
        label += "  ◆ SUBGOAL"
        draw.text((4, 3), label, fill=(0, 220, 80))
    else:
        draw.text((4, 3), label, fill=(200, 200, 200))

    bar_arr = np.array(bar)
    return np.concatenate([frame, bar_arr], axis=0)  # (H+BAR_H, 2W, 3)


# ---------------------------------------------------------------------------
# Per-episode conversion
# ---------------------------------------------------------------------------

def episode_to_video(ep_idx: int, out_dir: Path, fps: int) -> Path | None:
    sg_path   = Path(SUBGOAL_CACHE_DIR) / f"ep_{ep_idx:04d}.npz"
    feat_path = Path(FEAT_CACHE_DIR)    / f"ep_{ep_idx:04d}.npz"

    if not sg_path.exists():
        print(f"  ep {ep_idx}: subgoal cache missing, skip")
        return None
    if not feat_path.exists():
        print(f"  ep {ep_idx}: feat cache missing, skip")
        return None

    sg   = np.load(sg_path)
    feat = np.load(feat_path)

    main_imgs  = sg["main_imgs"]    # (N, 224, 224, 3) uint8
    wrist_imgs = sg["wrist_imgs"]   # (N, 224, 224, 3) uint8
    sg_frames  = feat["sg_frames"]  # (N,) int32 — subgoal frame index per step
    N = len(main_imgs)

    # Set of frame indices that ARE subgoal frames
    subgoal_set = set(sg_frames.tolist())

    frames = []
    for i in range(N):
        frame = _render_frame(
            main_img   = main_imgs[i].copy(),
            wrist_img  = wrist_imgs[i].copy(),
            step       = i,
            n_steps    = N,
            sg_frame   = int(sg_frames[i]),
            is_subgoal = (i in subgoal_set),
        )
        frames.append(frame)

    out_path = out_dir / f"ep_{ep_idx:04d}.mp4"
    iio.imwrite(str(out_path), frames, fps=fps, codec="libx264")
    print(f"  ep {ep_idx:4d}: {N} steps → {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_ids",  type=int, nargs="*",
                        help="Episode indices to convert (default: all)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps",     type=int, default=FPS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which episodes to process
    index_file = Path(SUBGOAL_CACHE_DIR) / "index.json"
    if not index_file.exists():
        print("Subgoal cache index not found. Run train_subgoal_decoder.py first.")
        return

    index = json.loads(index_file.read_text())
    all_ep_ids = [e["ep_idx"] for e in index]

    ep_ids = args.ep_ids if args.ep_ids is not None else all_ep_ids
    print(f"Converting {len(ep_ids)} episode(s) → {out_dir}")

    saved = []
    for ep_idx in ep_ids:
        path = episode_to_video(ep_idx, out_dir, args.fps)
        if path:
            saved.append(path)

    print(f"\nDone. {len(saved)} video(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
