import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as rt
import tyro

import cat_ppo


@dataclass
class Args:
    task: str
    exp_name: str = None
    seed: int = 42
    onnx_path: str = None
    pri: bool = False
    obs_path: str = 'data/assets/TypiObs/empty'
    yaw: float = 0.0          # initial robot yaw in degrees (0 = default forward direction)
    box_size: str = None      # box half-extents as "x,y,z" in metres (e.g. "0.15,0.20,0.15")
    box_mass: float = None    # box mass in kg (e.g. 1.5)
    stage1_steps: int = -1    # for G1CaTra: override stage 1 length; -1 = use task default
    warmstart_states_path: str = None  # path to .npz warm-start file; starts episode in Stage 2
    warmstart_idx: int = -1   # which saved state to load (-1 = random)
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


def play(args: Args):
    env_class = cat_ppo.registry.get(args.task, "play_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    env_cfg.pf_config.path = args.obs_path
    if args.stage1_steps >= 0 and hasattr(env_cfg, "stage1_steps"):
        env_cfg.stage1_steps = args.stage1_steps
    if args.warmstart_states_path and hasattr(env_cfg, "warmstart_states_path"):
        env_cfg.warmstart_states_path = args.warmstart_states_path
        if args.stage1_steps < 0 and hasattr(env_cfg, "stage1_steps"):
            env_cfg.stage1_steps = 0
    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
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

    state = env.reset(warmstart_idx=args.warmstart_idx) if hasattr(env, '_ws_qpos') else env.reset()
    if args.yaw != 0.0:
        angle = np.deg2rad(args.yaw)
        env.mj_data.qpos[3:7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]  # wxyz pure yaw quaternion
        mujoco.mj_forward(env.mj_model, env.mj_data)
    _ctr = 0

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
