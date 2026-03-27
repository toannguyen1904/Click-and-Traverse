"""Run all specialist evaluations and save videos to a folder.

Usage:
    python eval_all.py                          # run everything
    python eval_all.py --output_dir my_videos   # custom output dir
    python eval_all.py --num_steps 300          # shorter episodes
    python eval_all.py --filter narrow          # only run experiments whose name contains 'narrow'
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as rt

import cat_ppo
from cat_ppo.envs.g1.play_cat import PlayG1CatEnv


# ---------------------------------------------------------------------------
# Experiment registry
# Each entry: (task, exp_name, obs_path, use_privileged_obs)
# ---------------------------------------------------------------------------

_TYPI_OBS = "data/assets/TypiObs"
_RAND_OBS = "data/assets/RandObs"

EXPERIMENTS = []

# G1Cat specialists — typical obstacles
for scene in [
    "bar3", "bend", "ceilbar0", "ceilbar1", "doubar",
    "highcorner", "lowcorner", "Mceil1", "Mceilbar0", "Mceilbar1",
    "narrow0", "narrow1", "Nbar0", "Nbar1", "pillar",
]:
    EXPERIMENTS.append(("G1Cat", f"G1Cat_{scene}", f"{_TYPI_OBS}/{scene}", False))

# G1CatPri specialists — typical obstacles
for scene in ["ceil0", "narrow0", "narrow1", "Mceil0"]:
    EXPERIMENTS.append(("G1CatPri", f"G1CatPri_{scene}", f"{_TYPI_OBS}/{scene}", True))

# G1CatPri specialists — random obstacles (auto-discovered from logs)
_log_root = Path("data/logs/origin")
for d in sorted(_log_root.glob("G1CatPri_D*")):
    scene = d.name[len("G1CatPri_"):]  # e.g. "D8G2L1O0S25"
    rand_path = f"{_RAND_OBS}/{scene}"
    if Path(rand_path).exists():
        EXPERIMENTS.append(("G1CatPri", d.name, rand_path, True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_env(task: str, obs_path: str) -> PlayG1CatEnv:
    task_cfg = cat_ppo.registry.get(task, "config")
    env_cfg = task_cfg.env_config
    env_cfg.pf_config.path = obs_path
    env = PlayG1CatEnv(task_type=env_cfg.task_type, config=env_cfg, headless=True)
    return env


def load_policy(exp_name: str) -> rt.InferenceSession:
    ckpt_path = cat_ppo.get_latest_ckpt(exp_name)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for experiment: {exp_name}")
    onnx_path = ckpt_path / "policy.onnx"
    return rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _make_camera(distance: float = 5.0, azimuth: float = 135.0, elevation: float = -20.0) -> mujoco.MjvCamera:
    """Create a free camera with a wide view suitable for capturing the full scene."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = [1.0, 0.0, 0.8]  # look slightly ahead and at torso height
    return cam


def run_episode(
    env: PlayG1CatEnv,
    policy: rt.InferenceSession,
    pri: bool,
    num_steps: int,
    render_width: int,
    render_height: int,
    cam_distance: float = 5.0,
    cam_azimuth: float = 135.0,
    cam_elevation: float = -20.0,
) -> list[np.ndarray]:
    """Run one episode and return a list of RGB frames."""
    renderer = mujoco.Renderer(env.mj_model, height=render_height, width=render_width)
    cam = _make_camera(cam_distance, cam_azimuth, cam_elevation)
    env.pri = pri
    state = env.reset()
    frames = []

    for _ in range(num_steps):
        obs = state.obs["state"].reshape(1, -1).astype(np.float32)
        action = policy.run(["continuous_actions"], {"obs": obs})[0][0]
        state = env.step(state, action)

        # Keep camera loosely tracking the robot's horizontal position
        pelv_xy = env.mj_data.qpos[:2]
        cam.lookat[0] = pelv_xy[0]
        cam.lookat[1] = pelv_xy[1]

        renderer.update_scene(env.mj_data, camera=cam)
        frames.append(renderer.render())

    renderer.close()
    return frames


def save_video(frames: list[np.ndarray], path: Path, fps: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run all evaluations and save videos.")
    parser.add_argument("--output_dir", default="eval_videos", help="Output folder for videos")
    parser.add_argument("--num_steps", type=int, default=500, help="Steps per episode (50 Hz → 500 = 10 s)")
    parser.add_argument("--width", type=int, default=640, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels")
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate")
    parser.add_argument("--cam_distance", type=float, default=5.0, help="Camera distance from lookat point")
    parser.add_argument("--cam_azimuth", type=float, default=135.0, help="Camera azimuth angle in degrees")
    parser.add_argument("--cam_elevation", type=float, default=-20.0, help="Camera elevation angle in degrees (negative = looking down)")
    parser.add_argument(
        "--filter", default="", help="Only run experiments whose name contains this string"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.filter:
        experiments = [e for e in experiments if args.filter in e[1]]
        if not experiments:
            print(f"No experiments match filter '{args.filter}'. Available:")
            for e in EXPERIMENTS:
                print(f"  {e[1]}")
            return

    print(f"Running {len(experiments)} evaluations → {output_dir}/\n")

    for i, (task, exp_name, obs_path, pri) in enumerate(experiments):
        video_path = output_dir / f"{exp_name}.mp4"
        if video_path.exists():
            print(f"[{i+1}/{len(experiments)}] SKIP (exists): {exp_name}")
            continue

        print(f"[{i+1}/{len(experiments)}] {exp_name}  (pri={pri})")
        try:
            env = build_env(task, obs_path)
            policy = load_policy(exp_name)
            frames = run_episode(
                env, policy, pri, args.num_steps, args.width, args.height,
                cam_distance=args.cam_distance,
                cam_azimuth=args.cam_azimuth,
                cam_elevation=args.cam_elevation,
            )
            save_video(frames, video_path, fps=args.fps)
            print(f"  Saved {len(frames)} frames → {video_path}")
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
