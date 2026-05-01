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
    record: bool = False      # save rollout video to disk
    video_path: str = None    # output video path; defaults to <exp_name>/rollout.mp4
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 50       # matches ctrl_dt=0.02 s (50 Hz)


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
    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    env.pri = args.pri

    # Override box geometry in the MuJoCo model before reset
    if args.box_size is not None:
        half_extents = np.array([float(v) for v in args.box_size.split(",")])
        assert half_extents.shape == (3,), "--box-size must be 'x,y,z'"
        env.mj_model.geom_size[env._box_geom_id] = half_extents
    if args.box_mass is not None:
        env.mj_model.body_mass[env._box_body_id] = args.box_mass

    if args.onnx_path is not None:
        onnx_path = args.onnx_path
    else:
        ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
        onnx_path = ckpt_path / "policy.onnx"
    output_names = ["continuous_actions"]
    policy = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Set up video recorder if requested
    renderer = None
    writer = None
    if args.record:
        renderer = mujoco.Renderer(env.mj_model, height=args.video_height, width=args.video_width)
        if args.video_path is not None:
            video_path = args.video_path
        else:
            ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
            video_path = str(ckpt_path / "rollout.mp4")
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=args.video_fps)
        print(f"Recording video to: {video_path}")

    state = env.reset()
    if args.yaw != 0.0:
        angle = np.deg2rad(args.yaw)
        env.mj_data.qpos[3:7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]  # wxyz pure yaw quaternion
        mujoco.mj_forward(env.mj_model, env.mj_data)
    _ctr = 0

    try:
        while True:
            obs = state.obs["state"].reshape(1, -1).astype(np.float32)
            onnx_input = {"obs": obs}
            action = policy.run(output_names, onnx_input)[0]
            action = action[0]
            state = env.step(state, action)

            if renderer is not None:
                renderer.update_scene(env.mj_data)
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
