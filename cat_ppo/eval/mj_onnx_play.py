import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import json
from dataclasses import dataclass
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as rt
import tyro

import cat_ppo
from cat_ppo.eval.mj_onnx_test import _episode_status, _set_box_noise


@dataclass
class Args:
    task: str
    exp_name: str = None
    seed: int = 42
    onnx_path: str = None
    pri: bool = False
    obs_path: str = 'data/assets/TypiObs/empty'
    goal_x: float = 1.6       # base x (m) counted as a completed traversal (success)
    box_inflation: bool = True  # G1CaTra: box observes gf_inflation.npy (True) or regular gf.npy (False). Read from the checkpoint's config.json when available; this is only a fallback.
    yaw: float = 0.0          # initial robot yaw in degrees (0 = default forward direction)
    box_size: str = None      # box half-extents as "x,y,z" in metres (e.g. "0.15,0.20,0.15")
    box_mass: float = None    # box mass in kg (e.g. 1.5)
    box_noise: bool = True    # add box position/orientation tracking noise to the deployable obs (False -> ground-truth box)
    stage1_steps: int = -1    # for G1CaTra: override stage 1 length; -1 = use task default
    warmstart_states_path: str = None  # path to .npz warm-start file; starts episode in Stage 2
    warmstart_idx: int = -1   # which saved state to load (-1 = random)
    init_pos_offset: float = 0.0  # G1CaTra: random xy offset (m) for robot+box init pose in [-v, v]; 0 disables
    init_ang_offset: float = 0.0  # G1CaTra: random yaw offset (deg) for robot+box init pose in [-v, v]; 0 disables
    record: bool = False      # save rollout video to disk
    video_path: str = None    # output video path; defaults to <exp_name>/rollout.mp4
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 50       # matches ctrl_dt=0.02 s (50 Hz)
    cam_distance: float = 5.0       # camera distance from lookat point (metres)
    cam_azimuth: float = 135.0      # camera azimuth angle (degrees)
    cam_elevation: float = -20.0    # camera elevation angle (degrees)
    cam_lookat: str = None          # lookat point as "x,y,z"; defaults to robot base position


@dataclass
class State:
    info: dict
    obs: dict


def _read_box_use_inflation(exp_name, fallback):
    """Read env_config.box_use_inflation from the checkpoint's config.json so playback
    matches training. Falls back to the given value if the file/key is unavailable."""
    if not exp_name:
        return fallback
    cfg_path = cat_ppo.get_path_log(exp_name) / "checkpoints" / "config.json"
    if not cfg_path.exists():
        print(f"[mj_onnx_play] config.json not found at {cfg_path}; using --box_inflation={fallback}")
        return fallback
    try:
        saved = json.loads(cfg_path.read_text())
        return bool(saved["env_config"]["box_use_inflation"])
    except (KeyError, ValueError):
        print(f"[mj_onnx_play] box_use_inflation absent from {cfg_path}; using --box_inflation={fallback}")
        return fallback


def play(args: Args):
    env_class = cat_ppo.registry.get(args.task, "play_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    env_cfg.pf_config.path = args.obs_path
    if hasattr(env_cfg, "box_use_inflation"):
        env_cfg.box_use_inflation = _read_box_use_inflation(args.exp_name, args.box_inflation)
        print(f"[mj_onnx_play] box_use_inflation = {env_cfg.box_use_inflation}")
    if args.stage1_steps >= 0 and hasattr(env_cfg, "stage1_steps"):
        env_cfg.stage1_steps = args.stage1_steps
    if args.warmstart_states_path and hasattr(env_cfg, "warmstart_states_path"):
        env_cfg.warmstart_states_path = args.warmstart_states_path
        if args.stage1_steps < 0 and hasattr(env_cfg, "stage1_steps"):
            env_cfg.stage1_steps = 0
    _set_box_noise(env_cfg, args.box_noise)
    env = env_class(task_type=env_cfg.task_type, config=env_cfg, headless=args.record)
    env.pri = args.pri

    # Override box geometry in the MuJoCo model before reset
    if args.box_size is not None:
        half_extents = np.array([float(v) for v in args.box_size.split(",")])
        assert half_extents.shape == (3,), "--box-size must be 'x,y,z'"
        env.mj_model.geom_size[env._box_geom_id] = half_extents
    if args.box_mass is not None:
        env.mj_model.body_mass[env._box_body_id] = args.box_mass

    output_names = ["continuous_actions"]
    # Two-agent tasks load two policies (lower=legs, upper=arms) and concatenate actions.
    is_2a = hasattr(env_cfg, "num_act_lower")
    if is_2a:
        ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
        policy_lower = rt.InferenceSession(str(ckpt_path / "policy_lower.onnx"), providers=["CPUExecutionProvider"])
        policy_upper = rt.InferenceSession(str(ckpt_path / "policy_upper.onnx"), providers=["CPUExecutionProvider"])
    else:
        if args.onnx_path is not None:
            onnx_path = args.onnx_path
        else:
            ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
            onnx_path = ckpt_path / "policy.onnx"
        policy = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Set up video recorder if requested
    renderer = None
    writer = None
    cam = None
    if args.record:
        renderer = mujoco.Renderer(env.mj_model, height=args.video_height, width=args.video_width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = args.cam_distance
        cam.azimuth = args.cam_azimuth
        cam.elevation = args.cam_elevation
        if args.cam_lookat is not None:
            cam.lookat[:] = [float(v) for v in args.cam_lookat.split(",")]
        if args.video_path is not None:
            video_path = args.video_path
        else:
            ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
            video_path = str(ckpt_path / "rollout.mp4")
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=args.video_fps)
        print(f"Recording video to: {video_path}")

    if hasattr(env, '_ws_qpos'):
        state = env.reset(
            warmstart_idx=args.warmstart_idx,
            pos_offset=args.init_pos_offset,
            ang_offset_deg=args.init_ang_offset,
        )
    else:
        state = env.reset()
    if args.yaw != 0.0:
        angle = np.deg2rad(args.yaw)
        env.mj_data.qpos[3:7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]  # wxyz pure yaw quaternion
        mujoco.mj_forward(env.mj_model, env.mj_data)
    _ctr = 0
    max_steps = int(env_cfg.episode_length) if hasattr(env_cfg, "episode_length") else 10 ** 9

    try:
        while True:
            obs = state.obs["state"].reshape(1, -1).astype(np.float32)
            if is_2a:
                a_lower = policy_lower.run(output_names, {"obs": obs})[0][0]
                a_upper = policy_upper.run(output_names, {"obs": obs})[0][0]
                action = np.concatenate([a_lower, a_upper])  # lower-first, matches action ordering
            else:
                action = policy.run(output_names, {"obs": obs})[0][0]
            state = env.step(state, action)

            if renderer is not None:
                if args.cam_lookat is None:
                    cam.lookat[:] = env.mj_data.qpos[:3]  # track robot base
                renderer.update_scene(env.mj_data, camera=cam)
                frame = renderer.render()
                writer.append_data(frame)

            _ctr += 1

            # Episode outcome (matches mj_onnx_test scoring): stop and report at the terminal event.
            status, base_x = _episode_status(env, state, env_cfg, args.goal_x, max_steps)
            if status is not None:
                outcome = "SUCCESS" if status == "success" else f"FAIL ({status})"
                print(f"[mj_onnx_play] episode ended at step {int(state.info['step'])}: "
                      f"{outcome}  base_x={base_x:.3f}  goal_x={args.goal_x}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
            print(f"Video saved to: {video_path}")
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
