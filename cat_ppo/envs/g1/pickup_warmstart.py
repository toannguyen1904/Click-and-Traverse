"""Pickup-policy warm-start utilities.

Public symbols:
  load_pickup_inference_fn  – load a trained pickup policy from a Brax checkpoint
  pickup_obs_from_data      – pure function producing the pickup obs from mjx.Data
"""

import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from cat_ppo.envs.g1.env_catra import (
    BOX_QVEL_START,
    NUM_ROBOT_JOINTS,
)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_pickup_inference_fn(checkpoint_path: str) -> Callable:
    """Load the deterministic pickup policy from a Brax orbax checkpoint.

    Reads ppo_network_config.json from inside the checkpoint directory,
    reconstructs the network, and returns inference_fn(obs_dict, rng) -> (action, extra).

    obs_dict must have keys "state" (97,) and "privileged_state" (135,).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pickup checkpoint not found: {ckpt_path}")

    config_path = ckpt_path / "ppo_network_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"ppo_network_config.json not found in {ckpt_path}")

    with open(config_path) as f:
        net_cfg = json.load(f)

    observation_size = {k: tuple(v) for k, v in net_cfg["observation_size"].items()}
    action_size = net_cfg["action_size"]
    network_kwargs = net_cfg["network_factory_kwargs"]

    ppo_net = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        **network_kwargs,
    )

    params = ppo_checkpoint.load(ckpt_path)
    make_policy = ppo_networks.make_inference_fn(ppo_net)
    return make_policy(params, deterministic=True)


# ---------------------------------------------------------------------------
# Obs computation (extracted from G1PickupEnv._get_obs)
# ---------------------------------------------------------------------------

def pickup_obs_from_data(
    data: mjx.Data,
    info: dict[str, Any],
    *,
    gyro_sensor_adr: int,
    gyro_sensor_dim: int,
    pelvis_body_id: int,
    pelvis_imu_site_id: int,
    torso_imu_site_id: int,
    hands_site_id: np.ndarray,
    shlds_site_id: np.ndarray,
    box_body_id: int,
    action_joint_ids: jax.Array,
    default_qpos: jax.Array,
    noise_config: Any,
    dt: float,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Compute 97-dim state and 135-dim privileged_state from mjx.Data.

    Returns (obs_dict, updated_rng). The updated_rng reflects the noise-sampling
    splits consumed internally, so callers can thread it forward.

    This is the canonical implementation; G1PickupEnv._get_obs is a thin wrapper.
    """
    gyro_pelvis = data.sensordata[gyro_sensor_adr:gyro_sensor_adr + gyro_sensor_dim]
    gvec_pelvis = data.site_xmat[pelvis_imu_site_id].T @ jp.array([0., 0., -1.])
    joint_angles = data.qpos[7:7 + NUM_ROBOT_JOINTS]
    joint_vel = data.qvel[6:6 + NUM_ROBOT_JOINTS]

    pelvis_pos = data.xpos[pelvis_body_id]
    pelvis_rot = data.site_xmat[pelvis_imu_site_id].reshape(3, 3)
    pelvis_xquat = data.xquat[pelvis_body_id]
    box_pos_world = data.xpos[box_body_id]
    box_quat_world = data.xquat[box_body_id]
    box_pos_local = pelvis_rot.T @ (box_pos_world - pelvis_pos)
    pelvis_xquat_conj = pelvis_xquat * jp.array([1., -1., -1., -1.])
    box_quat_local = math.quat_mul(pelvis_xquat_conj, box_quat_world)

    box_size = info["box_size"]

    box_vel_local = pelvis_rot.T @ data.qvel[BOX_QVEL_START:BOX_QVEL_START + 3]
    box_angvel = data.qvel[BOX_QVEL_START + 3:BOX_QVEL_START + 6]
    left_hand_pos = data.site_xpos[hands_site_id[0]]
    right_hand_pos = data.site_xpos[hands_site_id[1]]
    left_hand_vel = (left_hand_pos - info["last_left_hand_pos"]) / dt
    right_hand_vel = (right_hand_pos - info["last_right_hand_pos"]) / dt
    pelv_site_pos = data.site_xpos[pelvis_imu_site_id]
    tors_site_pos = data.site_xpos[torso_imu_site_id]
    left_shld_pos = data.site_xpos[shlds_site_id[0]]
    right_shld_pos = data.site_xpos[shlds_site_id[1]]
    head_pos = info["head_pos"]

    privileged_state = jp.hstack([
        gyro_pelvis, gvec_pelvis,
        (joint_angles - default_qpos)[action_joint_ids],
        joint_vel[action_joint_ids],
        info["last_act"],
        info["motor_targets"][action_joint_ids],
        box_pos_local, box_quat_local, box_size,
        info["box_mass"].reshape(1),
        box_vel_local, box_angvel,
        left_hand_pos, right_hand_pos, box_pos_world,
        pelv_site_pos, tors_site_pos,
        left_shld_pos, right_shld_pos, head_pos,
        left_hand_vel, right_hand_vel,
        info["kp_scale"].reshape(1), info["kd_scale"].reshape(1),
    ])

    nl = noise_config.level
    ns = noise_config.scales
    rng, k1, k2, k3, k4 = jax.random.split(info["rng"], 5)
    noisy_gyro = gyro_pelvis + (2 * jax.random.uniform(k1, (3,)) - 1) * nl * ns.gyro
    noisy_gvec = gvec_pelvis + (2 * jax.random.uniform(k2, (3,)) - 1) * nl * ns.gravity
    noisy_ja = joint_angles + (2 * jax.random.uniform(k3, joint_angles.shape) - 1) * nl * ns.joint_pos
    noisy_jv = joint_vel + (2 * jax.random.uniform(k4, joint_vel.shape) - 1) * nl * ns.joint_vel

    state = jp.hstack([
        noisy_gyro, noisy_gvec,
        (noisy_ja - default_qpos)[action_joint_ids],
        noisy_jv[action_joint_ids],
        info["last_act"],
        info["motor_targets"][action_joint_ids],
        box_pos_local, box_quat_local, box_size,
        info["box_mass"].reshape(1),
    ])

    obs = {
        "state": jp.nan_to_num(state),
        "privileged_state": jp.nan_to_num(privileged_state),
    }
    return obs, rng

