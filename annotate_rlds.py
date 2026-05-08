"""
Read and annotate LIBERO RLDS datasets.
Data path: /mnt/nfs/Users/jerry007005/dataset/modified_libero_rlds
Run with: python annotate_rlds.py  (project venv)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], "GPU")

DATA_DIR = "/mnt/nfs/Users/jerry007005/dataset/modified_libero_rlds"
DATASET_NAMES = [
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
]


def load_dataset(dataset_name: str):
    """Load an RLDS dataset by name."""
    return tfds.load(dataset_name, data_dir=DATA_DIR, split="train")


def iter_episodes(dataset_name: str):
    """Yield episodes as lists of step dicts (numpy arrays)."""
    dataset = load_dataset(dataset_name)
    for episode in dataset:
        steps = list(episode["steps"].as_numpy_iterator())
        episode_meta = {
            k: v.numpy() for k, v in episode["episode_metadata"].items()
        }
        yield episode_meta, steps


def print_dataset_info(dataset_name: str, max_episodes: int = 3, max_steps: int = 3):
    """Print structure and sample data from a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    for ep_idx, (meta, steps) in enumerate(iter_episodes(dataset_name)):
        if ep_idx >= max_episodes:
            break

        print(f"\nEpisode {ep_idx}  ({len(steps)} steps)")
        print(f"  file_path : {meta['file_path'].decode()}")

        for step in steps[:max_steps]:
            print(f"  language  : {step['language_instruction'].decode()}")
            print(f"  image     : {step['observation']['image'].shape}  dtype={step['observation']['image'].dtype}")
            print(f"  wrist_img : {step['observation']['wrist_image'].shape}")
            print(f"  state     : {step['observation']['state']}")
            print(f"  joint_state: {step['observation']['joint_state']}")
            print(f"  action    : {step['action']}")
            print(f"  reward    : {step['reward']}  is_terminal={step['is_terminal']}")

if __name__ == "__main__":
    print_dataset_info("libero_spatial_no_noops", max_episodes=3)
