"""
LIBERO evaluation for the GoalExpert + Executor pipeline.

At every replan_steps, GoalExpert generates a subgoal (sg_main, sg_wrist, sg_state)
from the current observation. At every step, the Executor takes the current
SigLIP features + fixed subgoal → single-step action.

Usage:
  python libero_eval.py
  python libero_eval.py --task_suite_name libero_spatial --num_trials_per_task 10
"""

import os, sys, math, logging, pathlib, dataclasses
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import imageio.v3 as iio

import numpy as np
import torch
import tqdm
import draccus

# LIBERO's benchmark.get_task_init_states calls torch.load() without weights_only;
# upstream LIBERO has not patched this. Default it here so torch>=2.6 doesn't error.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent / "openpi" / "src"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from model.executor            import Executor
from model.pi0_subgoal_decoder import PI0WithGoalExpert
from model.subgoal_encoder     import SubgoalAutoencoder

LIBERO_DUMMY_ACTION    = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION  = 256

PATCH_DIM      = 2048
SLOTS_PER_VIEW = 8
NUM_IMGS       = 4 * SLOTS_PER_VIEW
PROPRIO_DIM    = 8
ACTION_DIM     = 7
HIDDEN_DIM     = 1024
NUM_LAYERS     = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    task_suite_name:    str = "libero_spatial"
    num_steps_wait:     int = 10
    num_trials_per_task:int = 50
    replan_tolerance:   int = 0     # extra steps added to horizon_pred before replanning
    resize_size:        int = 224
    max_lang_len:       int = 48
    seed:               int = 7
    local_log_dir:      str = "./logs"
    run_id_note:        str = None
    executor_ckpt:      str = "./checkpoints/executor/checkpoint.pt"
    goal_expert_ckpt:   str = "./checkpoints/subgoal_decoder/checkpoint.pt"
    encoder_ckpt:       str = "./checkpoints/subgoal_encoder/checkpoint.pt"
    pi05_ckpt_dir:      str = "/mnt/nfs/Users/jerry007005/model/openpi/pi05_libero"
    save_video:         bool = True   # save per-episode MP4 to local_log_dir/videos/
    video_fps:          int  = 10
    mode:               str  = "sample_all"   # "sample_action" | "sample_goal" | "sample_all"
    action_chunk_replan: int = 5              # for sample_action: replan every N steps


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_executor(ckpt_path: str, norm_stats_path: str) -> Executor:
    model = Executor(
        num_imgs=NUM_IMGS, patch_dim=PATCH_DIM, proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_LAYERS,
        norm_stats_path=norm_stats_path,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Executor loaded from {ckpt_path} (step {ckpt.get('step','?')})")
    return model


def _load_goal_expert(goal_ckpt: str, pi05_ckpt_dir: str) -> PI0WithGoalExpert:
    from openpi.training import config as _config
    import safetensors.torch

    train_cfg = _config.get_config("pi05_libero")
    ns_path = str(
        pathlib.Path(pi05_ckpt_dir) / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    )
    model = PI0WithGoalExpert(
        config=train_cfg.model, patch_dim=PATCH_DIM,
        proprio_dim=PROPRIO_DIM, slots_per_view=SLOTS_PER_VIEW,
        freeze_pi0=True,
        expert_variant="gemma_300m",
        norm_stats_path=ns_path,
    ).to(DEVICE)
    # Load PI0 base weights (frozen backbone)
    safetensors.torch.load_model(
        model, os.path.join(pi05_ckpt_dir, "model.safetensors"), strict=False,
    )
    # Load trained goal expert weights (strip torch.compile prefix)
    ckpt = torch.load(goal_ckpt, map_location=DEVICE)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_trainable_state(state)
    model.eval()
    print(f"GoalExpert loaded from {goal_ckpt} (step {ckpt.get('step','?')})")
    return model


# ---------------------------------------------------------------------------
# Obs preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_img(img_uint8: np.ndarray, resize: int) -> torch.Tensor:
    """uint8 HWC → (1, 3, H, W) float32 [-1, 1] on DEVICE."""
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_uint8, resize, resize)
    )
    t = torch.from_numpy(img).float() / 255.0
    return (t * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def _get_state(obs: dict) -> torch.Tensor:
    """Returns (1, 8) float32 state tensor on DEVICE."""
    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ])
    return torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(DEVICE)


def _load_subgoal_encoder(ckpt_path: str) -> SubgoalAutoencoder:
    ae = SubgoalAutoencoder(
        patch_dim=PATCH_DIM, slots_per_view=SLOTS_PER_VIEW, n_patches=512,
        n_heads=16, enc_layers=2, dec_layers=2, ffn_mult=4,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval().to(DEVICE)
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"SubgoalEncoder loaded from {ckpt_path}")
    return ae


@torch.no_grad()
def _encode_obs(
    goal_expert: PI0WithGoalExpert,
    subgoal_enc: SubgoalAutoencoder,
    main_t:  torch.Tensor,   # (1, 3, H, W)
    wrist_t: torch.Tensor,   # (1, 3, H, W)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (curr_main, curr_wrist, curr_z):
      curr_main : (1, S, PATCH_DIM)   main slots
      curr_wrist: (1, S, PATCH_DIM)   wrist slots
      curr_z    : (1, 2*S*PATCH_DIM)  flat latent (matches SAE encode output)
    """
    main_patches  = goal_expert.paligemma_with_expert.embed_image(main_t).float()
    wrist_patches = goal_expert.paligemma_with_expert.embed_image(wrist_t).float()
    patches = torch.cat([main_patches, wrist_patches], dim=1)                       # (1, 512, 2048)
    z       = subgoal_enc.encode(patches)                                            # (1, 2*S*D)
    curr_main, curr_wrist = subgoal_enc.split_z(z)                                   # each (1, S, D)
    return curr_main, curr_wrist, z


def _make_tokenizer(max_len: int):
    from openpi.models import tokenizer as _tok
    return _tok.PaligemmaTokenizer(max_len=max_len)


# ---------------------------------------------------------------------------
# LIBERO env helper
# ---------------------------------------------------------------------------

def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution
    )
    env.seed(seed)
    return env, task_description


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    run_id = f"EVAL-{args.task_suite_name}-goal_executor"
    if args.run_id_note:
        run_id += f"--{args.run_id_note}"
    os.makedirs(args.local_log_dir, exist_ok=True)
    log_path = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")

    if args.save_video:
        video_dir = pathlib.Path(args.local_log_dir) / "videos" / run_id
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"Videos → {video_dir}")

    # Max steps per suite
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object":  280,
        "libero_goal":    300,
        "libero_10":      520,
        "libero_90":      400,
    }
    if args.task_suite_name not in max_steps_map:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = max_steps_map[args.task_suite_name]

    if args.mode not in ("sample_action", "sample_goal", "sample_all"):
        raise ValueError(f"Unknown mode: {args.mode}. Choose 'sample_action', 'sample_goal', or 'sample_all'.")

    # Load models
    need_executor = args.mode in ("sample_goal", "sample_all")
    need_encoder  = args.mode in ("sample_goal", "sample_all")
    ns_path = str(
        pathlib.Path(args.pi05_ckpt_dir) / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    )
    executor    = _load_executor(args.executor_ckpt, ns_path) if need_executor else None
    goal_expert = _load_goal_expert(args.goal_expert_ckpt, args.pi05_ckpt_dir)
    subgoal_enc = _load_subgoal_encoder(args.encoder_ckpt) if need_encoder  else None
    tokenizer   = _make_tokenizer(args.max_lang_len)
    print(f"Eval mode: {args.mode}")

    # Task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[args.task_suite_name]()
    print(f"Task suite: {args.task_suite_name}  ({task_suite.n_tasks} tasks)")
    log_file.write(f"Task suite: {args.task_suite_name}\n")

    total_episodes, total_successes = 0, 0
    total_replans_all, total_replans_success = 0, 0

    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Pre-tokenize the task language (same for all episodes of this task)
        lang_tokens, lang_mask = tokenizer.tokenize(task_description, state=None)
        lang_tokens = torch.from_numpy(lang_tokens.astype(np.int64)).unsqueeze(0).to(DEVICE)
        lang_mask   = torch.from_numpy(lang_mask.astype(bool)).unsqueeze(0).to(DEVICE)

        task_episodes, task_successes = 0, 0
        task_replans_all, task_replans_success = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            step_since_replan = 1  # force replan on first real step (any value >= next_replan_in=1)
            next_replan_in    = 1
            ep_replans        = 0
            sg_main = sg_wrist = sg_state = None   # current subgoal
            vla_actions       = None               # action chunk from sample_all
            frames = []                            # for video recording

            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Preprocess current observation
                    # IMPORTANT: rotate 180° to match train preprocessing
                    raw_main  = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    raw_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Collect frame: main | wrist side-by-side (both resized to 256)
                    if args.save_video:
                        frames.append(np.concatenate([raw_main, raw_wrist], axis=1))

                    main_t  = _preprocess_img(raw_main,  args.resize_size)  # (1,3,H,W)
                    wrist_t = _preprocess_img(raw_wrist, args.resize_size)
                    state_t = _get_state(obs)                                # (1, 8)

                    # Encode current obs via SigLIP + SubgoalAutoencoder (not needed for sample_action)
                    curr_main = curr_wrist = curr_z = None
                    if need_encoder:
                        with torch.no_grad():
                            curr_main, curr_wrist, curr_z = _encode_obs(
                                goal_expert, subgoal_enc, main_t, wrist_t
                            )  # curr_main/wrist: (1, S, D); curr_z: (1, 2*S*D)

                    # Replan if needed
                    if step_since_replan >= next_replan_in:
                        with torch.no_grad():
                            if args.mode == "sample_action":
                                vla_actions = goal_expert.sample_action(
                                    main_t, wrist_t, lang_tokens, lang_mask, state_t,
                                )
                                next_replan_in = args.action_chunk_replan
                            elif args.mode == "sample_all":
                                vla_actions, sg_main, sg_wrist, sg_state, horizon_pred = \
                                    goal_expert.sample_all(
                                        main_t, wrist_t, lang_tokens, lang_mask, curr_z, state_t,
                                    )
                                h = int(horizon_pred.item()) if horizon_pred is not None else 0
                                next_replan_in = h + args.replan_tolerance
                            else:  # sample_goal
                                sg_main, sg_wrist, sg_state, horizon_pred = goal_expert.sample_goal(
                                    main_t, wrist_t, lang_tokens, lang_mask, curr_z, state_t,
                                )
                                h = int(horizon_pred.item()) if horizon_pred is not None else 0
                                next_replan_in = h + args.replan_tolerance
                        step_since_replan = 0
                        ep_replans += 1

                    # Generate action
                    if args.mode == "sample_action":
                        # PI0.5 chunk only; clamp to last if somehow past end
                        act_idx = min(step_since_replan, vla_actions.shape[1] - 1)
                        action  = vla_actions[:, act_idx]
                    elif args.mode == "sample_all" and step_since_replan < vla_actions.shape[1]:
                        # Use PI0.5 action chunk while within horizon
                        action  = vla_actions[:, step_since_replan]
                    else:
                        # sample_goal, or sample_all after chunk exhausted → executor
                        # curr_main/wrist: (1, S, D); sg_main/wrist: (1, S, D) from goal expert
                        imgs = torch.cat(
                            [curr_main, curr_wrist, sg_main, sg_wrist], dim=1
                        )  # (1, 4S, D)
                        with torch.no_grad():
                            action, _, _ = executor(
                                imgs, state_t, sg_state, deterministic=True
                            )  # (1, 7)

                    # Step environment
                    obs, _, done, _ = env.step(action[0].cpu().tolist())
                    step_since_replan += 1
                    t += 1

                    if done:
                        print(f"  Episode {episode_idx+1} success at step {t}")
                        task_successes  += 1
                        total_successes += 1
                        break

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes  += 1
            total_episodes += 1
            task_replans_all   += ep_replans
            total_replans_all  += ep_replans
            if done:
                task_replans_success  += ep_replans
                total_replans_success += ep_replans

            # Save episode video
            if args.save_video and frames:
                outcome = "success" if done else "fail"
                vpath = video_dir / f"task{task_id:02d}_ep{episode_idx:03d}_{outcome}.mp4"
                iio.imwrite(str(vpath), frames, fps=args.video_fps, codec="libx264")
                print(f"  Video → {vpath.name}")

            env_steps = t - args.num_steps_wait
            print(f"  Success: {done} | replans: {ep_replans} / {env_steps} steps "
                  f"(avg {env_steps/max(ep_replans,1):.1f} steps/subgoal) | "
                  f"total {total_successes}/{total_episodes} "
                  f"({total_successes/total_episodes*100:.1f}%)")
            log_file.write(f"Success: {done}  replans: {ep_replans}  env_steps: {env_steps}\n")
            log_file.write(f"Episodes: {total_episodes}  Successes: {total_successes} "
                           f"({total_successes/total_episodes*100:.1f}%)\n")
            log_file.flush()

        sr = task_successes / task_episodes if task_episodes else 0
        avg_all     = task_replans_all     / task_episodes    if task_episodes    else 0
        avg_success = task_replans_success / task_successes   if task_successes   else float("nan")
        print(f"Task {task_id} SR: {sr:.3f}  "
              f"pi0.5 calls — success: {avg_success:.1f}  all: {avg_all:.1f}")
        log_file.write(f"Task {task_id} SR: {sr:.3f}  "
                       f"avg_replans_success: {avg_success:.1f}  avg_replans_all: {avg_all:.1f}\n")
        log_file.flush()

    final_sr    = total_successes / total_episodes  if total_episodes  else 0
    avg_all     = total_replans_all     / total_episodes   if total_episodes   else 0
    avg_success = total_replans_success / total_successes  if total_successes  else float("nan")
    print(f"\nFinal SR: {final_sr:.3f}  ({total_successes}/{total_episodes})")
    print(f"pi0.5 calls — success episodes: {avg_success:.1f}/ep  "
          f"all episodes: {avg_all:.1f}/ep")
    log_file.write(f"\nFinal SR: {final_sr:.3f}  ({total_successes}/{total_episodes})\n")
    log_file.write(f"pi0.5 calls avg_success: {avg_success:.1f}  avg_all: {avg_all:.1f}\n")
    log_file.close()


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _quat2axisangle(quat):
    if quat[3] > 1.0:  quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero()
