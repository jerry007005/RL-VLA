"""
Convert LIBERO RLDS datasets to add subgoal_frame field.

For each step, subgoal_frame = the index of the next subgoal frame in the episode
(determined by GPT-4o vision, cached to avoid re-spending money).
Output: tfds.load("annotated_libero_dataset", config=<name>, data_dir=DST_DATA_DIR)

Run: python convert_rlds_subgoal.py
"""

import os, json, base64, io
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from PIL import Image
from openai import OpenAI

tf.config.set_visible_devices([], "GPU")

SRC_DATA_DIR = "/mnt/nfs/Users/jerry007005/dataset/modified_libero_rlds"
DST_DATA_DIR = "/mnt/nfs/Users/jerry007005/dataset/annotated_libero_rlds"
SG_CACHE_DIR = "/mnt/nfs/Users/jerry007005/dataset/subgoal_frames_cache"

DATASET_NAMES = [
    "libero_spatial_no_noops",
]

_SPEC_CACHE: dict = {}


# ---------------------------------------------------------------------------
# GPT subgoal frame cache helpers
# ---------------------------------------------------------------------------

def _get_cached_sg_frames(dataset_name: str, ep_idx: int) -> list[int] | None:
    path = Path(SG_CACHE_DIR) / f"{dataset_name}.json"
    if path.exists():
        data = json.loads(path.read_text())
        entry = data.get(str(ep_idx))
        if entry is not None:
            return entry["subgoal_frames"]
    return None


def _save_sg_frames(dataset_name: str, ep_idx: int, n_steps: int, frames: list[int]):
    Path(SG_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(SG_CACHE_DIR) / f"{dataset_name}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[str(ep_idx)] = {"n_steps": n_steps, "subgoal_frames": frames}
    path.write_text(json.dumps(data, indent=2))


def _query_subgoal_frames(task: str, steps: list) -> list[int]:
    """Ask GPT which frame indices mark subgoal completions. Returns e.g. [15, 45, 80, n-1]."""
    n    = len(steps)
    idxs = list(range(n))

    def _b64(img):
        buf = io.BytesIO()
        Image.fromarray(img).resize((256, 256)).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    content = [{"type": "text",
                "text": (
                    f"Task: {task}\n"
                    f"The frames show a robot performing a task over time.\n"
                    f"A subgoal is a meaningful state change (e.g., grasp, lift, place).\n"
                    f"Total steps: {n}\n"
                    f"Shown frame indices: {idxs}"
                )}]
    for i in idxs:
        obs = steps[i]["observation"]
        content += [
            {"type": "text", "text": f"Frame {i} (main camera):"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_b64(obs['image'])}", "detail": "low"}},
            {"type": "text", "text": f"Frame {i} (wrist camera):"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_b64(obs['wrist_image'])}", "detail": "low"}},
        ]
    resp = OpenAI().chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content":
                "You output JSON only. No explanation.\n"
                "Format: {\"subgoal_frames\": [int, ...]}\n"
                "Rules:\n"
                "- 2 to 5 integers\n"
                "- Strictly increasing\n"
                "- Must be chosen from the provided indices\n"
                "- Last must be the final frame index\n"
            },
            {"role": "user", "content": content},
        ],
        max_completion_tokens=128,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    if "```" in text:
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    frames = json.loads(text)["subgoal_frames"]
    if frames[-1] != n - 1:
        frames.append(n - 1)
    return frames


# ---------------------------------------------------------------------------
# Feature spec helpers
# ---------------------------------------------------------------------------

def _numpy_to_feature(arr) -> tfds.features.FeatureConnector:
    if isinstance(arr, (bytes, str)):
        return tfds.features.Text()
    if arr.dtype.kind in ("U", "S", "O"):
        return tfds.features.Text()
    if arr.ndim == 0:
        return tfds.features.Scalar(dtype=tf.as_dtype(arr.dtype))
    return tfds.features.Tensor(shape=arr.shape, dtype=tf.as_dtype(arr.dtype))


def _build_step_features(first_step: dict) -> dict:
    obs = first_step["observation"]

    obs_features: dict = {}
    for k, v in obs.items():
        if k in ("image", "wrist_image"):
            obs_features[k] = tfds.features.Image(shape=v.shape)
        else:
            obs_features[k] = _numpy_to_feature(v)

    step_features: dict = {"observation": tfds.features.FeaturesDict(obs_features)}
    for k, v in first_step.items():
        if k != "observation":
            step_features[k] = _numpy_to_feature(v)

    # subgoal_frame: index of the next subgoal frame within the episode
    step_features["subgoal_frame"] = tfds.features.Scalar(dtype=tf.int32)

    return step_features


def _get_spec(dataset_name: str) -> dict:
    if dataset_name not in _SPEC_CACHE:
        ds = tfds.load(dataset_name, data_dir=SRC_DATA_DIR, split="train")
        first_ep = next(iter(ds))
        first_step = next(iter(first_ep["steps"].as_numpy_iterator()))
        _SPEC_CACHE[dataset_name] = _build_step_features(first_step)
        print(f"  Step keys    : {list(first_step.keys())}")
        print(f"  Obs keys     : {list(first_step['observation'].keys())}")
        print(f"  image shape  : {first_step['observation']['image'].shape}")
        print(f"  joint_state  : {first_step['observation']['joint_state'].shape}")
    return _SPEC_CACHE[dataset_name]


# ---------------------------------------------------------------------------
# TFDS builder
# ---------------------------------------------------------------------------

class AnnotatedLiberoDataset(tfds.core.GeneratorBasedBuilder):
    """LIBERO dataset annotated with subgoal_frame index."""

    VERSION = tfds.core.Version("2.0.0")
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name=n) for n in DATASET_NAMES
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        step_features = _get_spec(self.builder_config.name)
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
                "steps": tfds.features.Dataset(step_features),
            }),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        src_ds = tfds.load(
            self.builder_config.name, data_dir=SRC_DATA_DIR, split="train"
        )
        for ep_idx, episode in enumerate(src_ds):
            steps = list(episode["steps"].as_numpy_iterator())
            meta = {k: v.numpy() for k, v in episode["episode_metadata"].items()}

            n = len(steps)
            task = steps[0]["language_instruction"].decode()
            sg_frames = _get_cached_sg_frames(self.builder_config.name, ep_idx)
            if sg_frames is None:
                print(f"    ep {ep_idx:4d}: no cache, skipping")
                continue
            print(f"    ep {ep_idx:4d}: cache → {sg_frames}")

            annotated = []
            for i, step in enumerate(steps):
                sg = next((f for f in sg_frames if f > i), n - 1)
                annotated.append({**step, "subgoal_frame": np.int32(sg)})

            if ep_idx % 50 == 0:
                print(f"    episode {ep_idx:4d}: {len(steps)} steps")

            yield ep_idx, {
                "episode_metadata": {"file_path": meta["file_path"]},
                "steps": annotated,
            }


# ---------------------------------------------------------------------------
# Convert + verify
# ---------------------------------------------------------------------------

def convert_all():
    Path(DST_DATA_DIR).mkdir(parents=True, exist_ok=True)
    for name in DATASET_NAMES:
        print(f"\n{'='*60}")
        print(f"Converting {name} ...")
        builder = AnnotatedLiberoDataset(config=name, data_dir=DST_DATA_DIR)
        builder.download_and_prepare()
        print(f"  Saved -> {DST_DATA_DIR}/annotated_libero_dataset/{name}/")


def verify(dataset_name: str):
    """Quick sanity check on a converted dataset."""
    builder = AnnotatedLiberoDataset(config=dataset_name, data_dir=DST_DATA_DIR)
    ds = builder.as_dataset(split="train")
    ep = next(iter(ds))
    step = next(iter(ep["steps"]))
    obs = step["observation"]
    print(f"\nVerify {dataset_name}:")
    print(f"  image            : {obs['image'].shape}  {obs['image'].dtype}")
    print(f"  wrist_image      : {obs['wrist_image'].shape}")
    print(f"  joint_state      : {obs['joint_state'].shape}  {obs['joint_state'].dtype}")
    print(f"  action           : {step['action'].shape}  {step['action'].dtype}")
    print(f"  subgoal_frame    : {step['subgoal_frame'].numpy()}")
    print(f"  language_instruction: {step['language_instruction']}")


def test_one(dataset_name: str = "libero_spatial_no_noops"):
    """Run the full pipeline on a single episode to verify correctness."""
    print(f"\nTest one episode from {dataset_name}")
    ds = tfds.load(dataset_name, data_dir=SRC_DATA_DIR, split="train")
    episode = next(iter(ds))
    steps = list(episode["steps"].as_numpy_iterator())
    step = steps[0]

    task = step["language_instruction"].decode()
    print(f"  obs image shape  : {step['observation']['image'].shape}")
    print(f"  obs state shape  : {step['observation']['state'].shape}")
    print(f"  language         : {task}")
    print(f"  action shape     : {step['action'].shape}")

    sg_frames = _get_cached_sg_frames(dataset_name, 0)
    if sg_frames is None:
        sg_frames = _query_subgoal_frames(task, steps)
        _save_sg_frames(dataset_name, 0, len(steps), sg_frames)
        print(f"  subgoal frames (GPT)  : {sg_frames}")
    else:
        print(f"  subgoal frames (cache): {sg_frames}")
    sg = next((f for f in sg_frames if f > 0), len(steps) - 1)
    print(f"  step 0 → subgoal_frame={sg}")

    # Save one episode to RLDS
    print("\nSaving one episode to RLDS ...")
    Path(DST_DATA_DIR).mkdir(parents=True, exist_ok=True)
    builder = AnnotatedLiberoDataset(config=dataset_name, data_dir=DST_DATA_DIR)
    orig_gen = builder._generate_examples
    def _one():
        for item in orig_gen():
            yield item
            return
    builder._generate_examples = _one
    builder.download_and_prepare()
    verify(dataset_name)
    print("Test passed.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_one()
        sys.exit(0)
    convert_all()
    print("\n" + "="*60)
    print("Verification")
    print("="*60)
    for name in DATASET_NAMES:
        verify(name)
