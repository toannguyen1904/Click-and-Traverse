"""Visualize a live G1Pickup episode driven by MJX physics.

Uses G1PickupEnv.reset() / .step() (JAX/MJX) to run the simulation and
syncs qpos / qvel / mocap_pos into a CPU MuJoCo passive viewer each frame.
Zero actions are applied so the robot is uncontrolled (gravity + contacts only).

Usage:
    python check_pickup_mjx.py
    python check_pickup_mjx.py --task G1Stand
    python check_pickup_mjx.py --surface_z 0.6
    python check_pickup_mjx.py --seed 5
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
import tyro

import cat_ppo
from cat_ppo.envs.g1 import constants as consts


@dataclass
class Args:
    task: str = "G1Pickup"
    seed: int = 0
    surface_z: Optional[float] = None  # None = random from config range


def sync_to_cpu(state, mj_data):
    # Support is now a freejoint body — its position comes through qpos, not mocap_pos
    mj_data.qpos[:] = np.array(state.data.qpos)
    mj_data.qvel[:] = np.array(state.data.qvel)


def main(args: Args):
    cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = cfg.env_config
    env_class = cat_ppo.registry.get(args.task, "train_env_class")

    if args.surface_z is not None:
        env_cfg.box_surface_height_range = [args.surface_z, args.surface_z]

    env = env_class(task_type=env_cfg.task_type, config=env_cfg)

    rng = jax.random.PRNGKey(args.seed)
    reset_fn = jax.jit(env.reset)
    step_fn  = jax.jit(env.step)

    print("JIT-compiling reset + step...")
    state = reset_fn(rng)
    zero_action = jp.zeros(env.action_size)
    state = step_fn(state, zero_action)
    state.data.qpos.block_until_ready()
    print("Done.")

    # Reset again cleanly for the actual run
    state = reset_fn(rng)

    # CPU model for the viewer — same XML as training env, so layout matches exactly
    mj_model = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    mj_data  = mujoco.MjData(mj_model)

    box_body_id    = mj_model.body("carried_box").id
    support_body_id = mj_model.body("box_support").id
    box_geom_id    = mj_model.geom("box_geom").id

    sync_to_cpu(state, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    print(f"Support z           : {mj_data.xpos[support_body_id][2]:.3f} m")
    print(f"Box z               : {mj_data.xpos[box_body_id][2]:.3f} m")
    print(f"Box position        : {mj_data.xpos[box_body_id]}")
    print(f"Box size (half-ext) : {mj_model.geom_size[box_geom_id]}")
    print("\nViewer open — close the window to exit.")

    ctrl_dt = env_cfg.ctrl_dt  # 0.02 s per policy step

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            t0 = time.monotonic()

            state = step_fn(state, zero_action)

            # Auto-reset on episode termination
            if float(state.done) > 0.5:
                rng, key = jax.random.split(rng)
                state = reset_fn(key)

            sync_to_cpu(state, mj_data)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, ctrl_dt - elapsed))


if __name__ == "__main__":
    main(tyro.cli(Args))
