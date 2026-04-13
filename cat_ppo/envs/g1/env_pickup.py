"""G1 Box Pickup environment: robot reaches and lifts a box from a support surface.

Phase 1 of the CaTra curriculum:
- Robot stands in place but can crouch using sagittal-plane leg joints.
- 17-DOF action space: hip pitch/knee/ankle pitch (6) + waist (3) + arms (8).
- Box placed 0.4 m in front of robot on a support surface at random height.
- Reward guides the robot to reach the box, reduce table contact force, and lift.
- Terminal states feed into the CaTra traversal policy.

Observation dimensions:
  num_obs = 85   (state, deployable with sensor noise)
  num_pri = 123  (privileged_state, built from scratch with noiseless sensors)
"""

from typing import Any, Dict, Optional, Union

import jax
import jaxlie
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src import mjx_env

import cat_ppo
from cat_ppo.envs.g1 import constants as consts
from cat_ppo.envs.g1.env_catra import (
    BOX_QPOS_START,
    BOX_QVEL_START,
    NUM_ROBOT_JOINTS,
    G1CaTraEnv,
    domain_randomize_catra,
    torque_step_catra,
)

# 17-DOF action space: sagittal-plane leg joints (6) + waist (3) + arms (8).
PICKUP_ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------

def _make_domain_randomize_pickup():
    """Factory: loads model once to get box IDs, returns the DR function."""
    _mj = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    _box_geom_id = _mj.geom("box_geom").id
    _box_body_id = _mj.body("carried_box").id
    del _mj

    FLOOR_GEOM_ID = 0
    TORSO_BODY_ID = 16

    def domain_randomize_pickup(model: mjx.Model, rng: jax.Array):
        """Base CaTra DR + per-env box size/mass randomization."""

        @jax.vmap
        def rand_dynamics(rng):
            pair_friction = model.pair_friction

            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[6:6 + NUM_ROBOT_JOINTS] * jax.random.uniform(
                key, shape=(NUM_ROBOT_JOINTS,), minval=0.9, maxval=1.1
            )
            dof_frictionloss = model.dof_frictionloss.at[6:6 + NUM_ROBOT_JOINTS].set(frictionloss)

            rng, key = jax.random.split(rng)
            armature = model.dof_armature[6:6 + NUM_ROBOT_JOINTS] * jax.random.uniform(
                key, shape=(NUM_ROBOT_JOINTS,), minval=1.0, maxval=1.05
            )
            dof_armature = model.dof_armature.at[6:6 + NUM_ROBOT_JOINTS].set(armature)

            rng, key = jax.random.split(rng)
            dpos = jax.random.uniform(key, (3,), minval=-0.1, maxval=0.1)
            body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
                model.body_ipos[TORSO_BODY_ID] + dpos
            )

            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

            rng, key = jax.random.split(rng)
            dmass_torso = jax.random.uniform(key, minval=-1.0, maxval=1.0)
            body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass_torso)

            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0
            qpos0 = qpos0.at[7:7 + NUM_ROBOT_JOINTS].set(
                qpos0[7:7 + NUM_ROBOT_JOINTS]
                + jax.random.uniform(key, shape=(NUM_ROBOT_JOINTS,), minval=-0.05, maxval=0.05)
            )

            # Box size: sample per-env half-extents
            rng, key = jax.random.split(rng)
            box_half_x = jax.random.uniform(key, minval=0.10, maxval=0.20)
            rng, key = jax.random.split(rng)
            box_half_y = jax.random.uniform(key, minval=0.10, maxval=0.25)
            rng, key = jax.random.split(rng)
            box_half_z = jax.random.uniform(key, minval=0.10, maxval=0.20)
            geom_size = model.geom_size.at[_box_geom_id].set(
                jp.array([box_half_x, box_half_y, box_half_z])
            )

            # Box mass: override the globally-scaled value
            rng, key = jax.random.split(rng)
            box_mass = jax.random.uniform(key, minval=1.0, maxval=2.0)
            body_mass = body_mass.at[_box_body_id].set(box_mass)

            return (pair_friction, dof_frictionloss, dof_armature, body_ipos, body_mass, qpos0, geom_size)

        (pair_friction, frictionloss, armature, body_ipos, body_mass, qpos0, geom_size) = rand_dynamics(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({
            "pair_friction": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "qpos0": 0,
            "geom_size": 0,
        })

        model = model.tree_replace({
            "pair_friction": pair_friction,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "qpos0": qpos0,
            "geom_size": geom_size,
        })

        return model, in_axes

    return domain_randomize_pickup


domain_randomize_pickup = _make_domain_randomize_pickup()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def g1_pickup_task_config() -> config_dict.ConfigDict:
    """Config for G1Pickup: stationary manipulation with crouching support.

    Observation dimensions:
      num_obs = 85   (state, deployable)
      num_pri = 123  (privileged_state)
    """
    env_config = config_dict.create(
        task_type="flat_terrain_catra",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=200,
        action_repeat=1,
        action_scale=0.5,
        num_obs=85,
        num_pri=123,
        num_act=17,
        soft_joint_pos_limit_factor=0.95,
        # Required by G1LocoEnv._post_init() — unused by G1PickupEnv which overrides reset/step
        history_len=15,
        restricted_joint_range=False,
        gait_config=config_dict.create(
            gait_bound=0.6,
            freq_range=[1.3, 1.5],
            foot_height_range=[0.07, 0.07],
        ),
        command_config=config_dict.create(
            resampling_time=10.0,
            stop_prob=0.2,
        ),
        push_config=config_dict.create(
            enable=False,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-0.5, 0.5],
        lin_vel_y=[-0.3, 0.3],
        ang_vel_yaw=[-0.5, 0.5],
        torso_height=[0.5, consts.DEFAULT_CHEST_Z],
        term_collision_threshold=0.04,
        box_drop_threshold=0.3,
        dm_rand_config=config_dict.create(
            enable_pd=True,
            kp_range=[0.75, 1.25],
            kd_range=[0.75, 1.25],
            enable_rfi=False,   # disabled: upper-body perturbations risk toppling the stationary robot
            rfi_lim=0.0,
            rfi_lim_range=[0.5, 1.5],
            enable_ctrl_delay=False,
            ctrl_delay_range=[0, 2],
        ),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                reach=0.0,  # 1.0 CHANGED
                lift=0.0,   # 5.0 CHANGED
                hold_stable=0.0,    # 0.1 CHANGED
                box_upright=0.0,    # 1.0 CHANGED
                upright=1.0,
                foot_contact=-0.5,
                foot_slip=-0.1,
                straight_knee=-5.0,
                joint_torque=-1e-4,
                smoothness_joint=-1e-6,
                smoothness=1e-3,
                joint_limits=1.0,
            ),
            base_height_target=0.75,
            foot_height_stance=0.0,
        ),
        box_surface_height_range=[0.4, 0.6],
        pf_config=config_dict.create(
            path='data/assets/TypiObs/empty',
            dx=0.04,
            origin=np.array([-0.5, -1.0, 0.0], dtype=np.float32),
        ),
    )

    policy_config = config_dict.create(
        num_timesteps=1_000_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        madrona_backend=False,
        augment_pixels=False,
        num_envs=32768,
        episode_length=200,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=domain_randomize_pickup,
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=1024,
        num_minibatches=32,
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 128, 64),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        seed=0,
        num_evals=6,
        eval_env=None,
        num_eval_envs=0,
        deterministic_eval=False,
        log_training_metrics=True,
        training_metrics_steps=int(1e6),
        progress_fn=lambda *args: None,
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=False,
    )

    eval_config = config_dict.create(
        duration=10.0,
    )

    return config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
        eval_config=eval_config,
    )


cat_ppo.registry.register("G1Pickup", "config")(g1_pickup_task_config())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@cat_ppo.registry.register("G1Pickup", "train_env_class")
class G1PickupEnv(G1CaTraEnv):
    """G1 humanoid reaching for and lifting a box from a support surface.

    Extends G1CaTraEnv with:
    - 17-DOF action space (sagittal-plane legs + waist + arms)
    - Compact pickup-specific observations (no HumanoidPF fields)
    - Pickup reward set: reach, lift, table_force, hold_stable, box_upright, upright
    - No gait clock, no push force, no command tracking
    """

    def __init__(
            self,
            task_type: str = "flat_terrain_catra",
            config: config_dict.ConfigDict = None,
            config_overrides: Optional[Dict[str, Union[str, Any, list[Any]]]] = None,
    ) -> None:
        super().__init__(
            task_type=task_type,
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init_pickup()

    def _post_init_pickup(self) -> None:
        """Override to 17-DOF action space and cache pickup IDs."""
        # 17-DOF action space: sagittal-plane legs + waist + arms
        self.action_joint_names = PICKUP_ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = jp.array([
            self.mj_model.actuator(name).id for name in self.action_joint_names
        ])

        # Keep _default_qpos as 29-dim (all robot joints); non-actuated joints
        # still receive their default PD targets each step.
        # (Already set correctly by _post_init_catra.)

        # Body IDs for obs/reward
        self._pelvis_body_id = self._mj_model.body("pelvis").id
        self._box_geom_id = self._mj_model.geom(consts.BOX_GEOM).id
        self._box_support_geom_id = self._mj_model.geom("box_support_col").id

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)  # 17

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset with 43-dim qpos; place box 0.4 m in front on support surface."""
        qpos = self._init_q.copy()  # (43,): robot default + box placeholder
        qvel = jp.zeros(self.mjx_model.nv)  # (41,)

        # Random root xy spawn (small offset)
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        qpos = qpos.at[2].set(0.8)

        # Random yaw in [-90°, 90°] (tighter than CaTra's [-180°, 180°])
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-np.pi / 2, maxval=np.pi / 2)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # Randomize robot joints [7:36] only (not box freejoint [36:43])
        rng, key = jax.random.split(rng)
        rand_qpos = qpos[7:7 + NUM_ROBOT_JOINTS] * jax.random.uniform(
            key, (NUM_ROBOT_JOINTS,), minval=0.5, maxval=1.5
        )
        rand_qpos = jp.clip(rand_qpos, self._soft_lowers, self._soft_uppers)
        qpos = qpos.at[7:7 + NUM_ROBOT_JOINTS].set(rand_qpos)

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:7 + NUM_ROBOT_JOINTS])

        # --- Place box 3.0 m in front of robot on support surface ---
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        forward_xy = jp.array([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z)])
        box_xy = qpos[:2] + 3.0 * forward_xy

        # Random box yaw offset relative to robot forward (±10°)
        rng, key = jax.random.split(rng)
        box_yaw_offset = jax.random.uniform(key, minval=-np.pi / 18, maxval=np.pi / 18)
        robot_yaw = yaw[0]
        box_yaw = robot_yaw + box_yaw_offset
        box_quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), jp.array([box_yaw]))

        # Random surface height
        rng, key = jax.random.split(rng)
        surface_z = jax.random.uniform(
            key,
            minval=self._config.box_surface_height_range[0],
            maxval=self._config.box_surface_height_range[1],
        )

        # Box/support half-z from current model.
        box_half_z = self.mjx_model.geom_size[self._box_geom_id][2]
        support_half_z = self.mjx_model.geom_size[self._box_support_geom_id][2]
        box_z = surface_z + support_half_z + box_half_z

        new_qpos = data.qpos.at[BOX_QPOS_START:BOX_QPOS_START + 3].set(
            jp.array([box_xy[0], box_xy[1], box_z])
        )
        new_qpos = new_qpos.at[BOX_QPOS_START + 3:BOX_QPOS_START + 7].set(box_quat)
        data = data.replace(qpos=new_qpos)

        # Reposition support surface
        new_mocap_pos = data.mocap_pos.at[self._box_support_mocap_id].set(
            jp.array([box_xy[0], box_xy[1], surface_z])
        )
        data = data.replace(mocap_pos=new_mocap_pos)

        data = mjx.forward(self.mjx_model, data)

        # --- Domain randomization scalars ---
        rng, key_kp, key_kd = jax.random.split(rng, 3)
        kp_scale = jax.random.uniform(
            key_kp,
            minval=self._config.dm_rand_config.kp_range[0],
            maxval=self._config.dm_rand_config.kp_range[1],
        )
        kp_scale = jp.where(self._config.dm_rand_config.enable_pd, kp_scale, jp.ones_like(kp_scale))
        kd_scale = jax.random.uniform(
            key_kd,
            minval=self._config.dm_rand_config.kd_range[0],
            maxval=self._config.dm_rand_config.kd_range[1],
        )
        kd_scale = jp.where(self._config.dm_rand_config.enable_pd, kd_scale, jp.ones_like(kd_scale))

        # Cache initial hand positions for velocity computation
        left_hand_pos = data.site_xpos[self._hands_site_id[0]]
        right_hand_pos = data.site_xpos[self._hands_site_id[1]]
        head_pos = data.site_xpos[self._head_site_id]

        # Box size from current (DR'd) model — stored in info so reward/obs can use it
        box_size = self.mjx_model.geom_size[self._box_geom_id]  # (3,) half-extents

        info = {
            "rng": rng,
            "step": 0,
            "last_act": jp.zeros(self.action_size),
            "motor_targets": self._default_qpos.copy(),  # 29-dim; legs hold default
            "last_joint_vel": np.zeros(NUM_ROBOT_JOINTS),
            "head_pos": head_pos.copy(),
            "last_left_hand_pos": left_hand_pos.copy(),
            "last_right_hand_pos": right_hand_pos.copy(),
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "surface_z": surface_z,
            "support_half_z": support_half_z,
            "box_size": box_size,
            # Dummy fields required by pf_utils.py training wrapper (unused by pickup)
            "command": jp.zeros(4),
            "last_command": jp.zeros(4),
            "stop_timestep": jp.array(100),
            "phase": jp.zeros(2),
            "phase_dt": jp.zeros(()),
            "gait_freq": jp.zeros(()),
            "foot_height": jp.zeros(()),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["term/fall"] = jp.zeros(())
        metrics["term/box_drop"] = jp.zeros(())
        metrics["term/nan"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step with 17-DOF action; wrists and unused leg joints stay at default."""
        # Update motor targets for controlled joints.
        lower_motor_targets = jp.clip(
            state.info["motor_targets"][self.action_joint_ids]
            + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()  # 29-dim: uncontrolled joints stay at default
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)

        # Physics step: torque_step_catra handles the extra box freejoint DOF
        rfi_zeros = jp.zeros_like(self.torque_limit)
        state.info["rng"], data = torque_step_catra(
            state.info["rng"],
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            kp_scale=state.info["kp_scale"],
            kd_scale=state.info["kd_scale"],
            rfi_lim_scale=rfi_zeros,
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )

        # Update cached body positions
        head_pos = data.site_xpos[self._head_site_id]
        left_hand_pos = data.site_xpos[self._hands_site_id[0]]
        right_hand_pos = data.site_xpos[self._hands_site_id[1]]

        state.info["motor_targets"] = motor_targets
        state.info["head_pos"] = head_pos

        obs = self._get_obs(data, state.info)
        done, term_fall, term_box_drop, term_nan = self._get_termination(data, state.info)
        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])

        rewards = self._get_reward(data, action, state.info, done, feet_contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # Update info for next step
        state.info["last_left_hand_pos"] = left_hand_pos
        state.info["last_right_hand_pos"] = right_hand_pos
        state.info["last_act"] = action.copy()
        state.info["last_joint_vel"] = data.qvel[6:6 + NUM_ROBOT_JOINTS].copy()
        state.info["step"] += 1

        timeout = state.info["step"] >= self._config.episode_length
        state.info["step"] = jp.where(done | timeout, 0, state.info["step"])
        state.info["motor_targets"] = jp.where(done, self._default_qpos, state.info["motor_targets"])

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["term/fall"] = term_fall.astype(jp.float32)
        state.metrics["term/box_drop"] = term_box_drop.astype(jp.float32)
        state.metrics["term/nan"] = term_nan.astype(jp.float32)

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        """85-dim state (deployable, noisy) and 123-dim privileged_state (noiseless + extras).

        State (85):
            gyro_pelvis[+noise](3), gvec_pelvis[+noise](3),
            joint_angles[+noise](17), joint_vel[+noise](17),
            last_action(17), motor_targets(17),
            box_pos_local(3), box_quat_local(4), box_size(3), surface_z(1)

        Privileged (123 = 85 noiseless + 38 extras):
            [same 85 fields, noiseless]
            + box_vel_local(3), box_angvel(3),
            + left_hand_pos(3), right_hand_pos(3), box_pos_world(3),
            + pelvis_pos(3), torso_pos(3), left_shld_pos(3), right_shld_pos(3), head_pos(3),
            + left_hand_vel(3), right_hand_vel(3),
            + kp_scale(1), kd_scale(1)
        """
        # --- Shared (noiseless) ---
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0., 0., -1.])
        joint_angles = data.qpos[7:7 + NUM_ROBOT_JOINTS]
        joint_vel = data.qvel[6:6 + NUM_ROBOT_JOINTS]

        # Box pose in pelvis frame
        pelvis_pos = data.xpos[self._pelvis_body_id]
        pelvis_rot = data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        pelvis_xquat = data.xquat[self._pelvis_body_id]  # wxyz
        box_pos_world = data.xpos[self._box_body_id]
        box_quat_world = data.xquat[self._box_body_id]  # wxyz
        box_pos_local = pelvis_rot.T @ (box_pos_world - pelvis_pos)
        pelvis_xquat_conj = pelvis_xquat * jp.array([1., -1., -1., -1.])
        box_quat_local = math.quat_mul(pelvis_xquat_conj, box_quat_world)

        box_size = info["box_size"]
        surface_z = info["surface_z"]

        # --- Privileged extras ---
        box_vel_local = pelvis_rot.T @ data.qvel[BOX_QVEL_START:BOX_QVEL_START + 3]
        box_angvel = data.qvel[BOX_QVEL_START + 3:BOX_QVEL_START + 6]
        left_hand_pos = data.site_xpos[self._hands_site_id[0]]
        right_hand_pos = data.site_xpos[self._hands_site_id[1]]
        left_hand_vel = (left_hand_pos - info["last_left_hand_pos"]) / self.dt
        right_hand_vel = (right_hand_pos - info["last_right_hand_pos"]) / self.dt
        pelv_site_pos = data.site_xpos[self._pelvis_imu_site_id]
        tors_site_pos = data.site_xpos[self._torso_imu_site_id]
        left_shld_pos = data.site_xpos[self._shlds_site_id[0]]
        right_shld_pos = data.site_xpos[self._shlds_site_id[1]]
        head_pos = info["head_pos"]  # updated in step() before _get_obs is called

        # --- Privileged state (noiseless, built from scratch) ---
        privileged_state = jp.hstack([
            gyro_pelvis, gvec_pelvis,
            (joint_angles - self._default_qpos)[self.action_joint_ids],
            joint_vel[self.action_joint_ids],
            info["last_act"],
            info["motor_targets"][self.action_joint_ids],
            box_pos_local, box_quat_local, box_size,
            surface_z.reshape(1),
            # Privileged-only extras
            box_vel_local, box_angvel,
            left_hand_pos, right_hand_pos, box_pos_world,
            pelv_site_pos, tors_site_pos,
            left_shld_pos, right_shld_pos, head_pos,
            left_hand_vel, right_hand_vel,
            info["kp_scale"].reshape(1), info["kd_scale"].reshape(1),
        ])

        # --- Noisy state (deployable) ---
        nl = self._config.noise_config.level
        ns = self._config.noise_config.scales
        info["rng"], k1, k2, k3, k4 = jax.random.split(info["rng"], 5)
        noisy_gyro = gyro_pelvis + (2 * jax.random.uniform(k1, (3,)) - 1) * nl * ns.gyro
        noisy_gvec = gvec_pelvis + (2 * jax.random.uniform(k2, (3,)) - 1) * nl * ns.gravity
        noisy_ja = joint_angles + (2 * jax.random.uniform(k3, joint_angles.shape) - 1) * nl * ns.joint_pos
        noisy_jv = joint_vel + (2 * jax.random.uniform(k4, joint_vel.shape) - 1) * nl * ns.joint_vel

        state = jp.hstack([
            noisy_gyro, noisy_gvec,
            (noisy_ja - self._default_qpos)[self.action_joint_ids],
            noisy_jv[self.action_joint_ids],
            info["last_act"],
            info["motor_targets"][self.action_joint_ids],
            box_pos_local, box_quat_local, box_size,
            surface_z.reshape(1),
        ])

        return {
            "state": jp.nan_to_num(state),
            "privileged_state": jp.nan_to_num(privileged_state),
        }

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]):
        """Terminate on robot fall, box drop, or NaN. Returns (done, fall, box_drop, nan_term)."""
        fall = self.get_gravity(data, "pelvis")[2] < 0.0
        fall |= info["head_pos"][2] < 0.5
        box_drop = data.xpos[self._box_body_id][2] < (info["surface_z"] - 0.1)
        nan_term = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        # return fall | box_drop | nan_term
        return fall | nan_term, fall, box_drop, nan_term

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            done: jax.Array,
            feet_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        """Pickup reward set: reach, lift, hold_stable, box_upright, upright, foot stability, energy, smoothness, joint_limits."""
        box_pos = data.xpos[self._box_body_id]
        box_z = box_pos[2]
        box_half_z = info["box_size"][2]
        box_half_y = info["box_size"][1]

        left_palm_pos = data.site_xpos[self._hands_site_id[0]]
        right_palm_pos = data.site_xpos[self._hands_site_id[1]]

        # Reach: encourage hands to approach the left/right box faces, not the center.
        reach = -(jp.linalg.norm(left_palm_pos - box_pos) + jp.linalg.norm(right_palm_pos - box_pos)) + 2 * box_half_y

        # Lift: reward lifting above surface (0 when resting, 1.0 at +10cm)
        lift_height = box_z - (info["surface_z"] + info["support_half_z"] + box_half_z)
        lift = jp.clip(lift_height, 0.0, 0.1) / 0.1

        # Hold stable: penalize box tumbling
        box_angvel = data.qvel[BOX_QVEL_START + 3:BOX_QVEL_START + 6]
        hold_stable = -jp.linalg.norm(box_angvel)

        # Box upright: keep box z-axis aligned with world z (important for CaTra handoff)
        box_quat = data.xquat[self._box_body_id]  # wxyz
        qx, qy = box_quat[1], box_quat[2]
        box_tilt_cos = jp.clip(1.0 - 2.0 * (qx ** 2 + qy ** 2), -1.0, 1.0)
        box_tilt_angle = jp.arccos(box_tilt_cos)
        box_upright = jp.exp(-box_tilt_angle ** 2)

        # Robot upright: penalize pelvis/torso roll and torso pitch at all times.
        pelvis_rot = data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        torso_rot = data.site_xmat[self._torso_imu_site_id].reshape(3, 3)
        pelvis_rpy = jp.array(jaxlie.SO3.from_matrix(pelvis_rot).as_rpy_radians())
        torso_rpy = jp.array(jaxlie.SO3.from_matrix(torso_rot).as_rpy_radians())
        err_roll = jp.abs(pelvis_rpy[0]) + jp.abs(torso_rpy[0])
        err_pitch = jp.abs(torso_rpy[1])
        upright = jp.exp(-0.5 * (err_roll + err_pitch))

        # Pickup is a stationary task: both feet should stay planted on the floor.
        stance_gait = jp.ones(2)
        foot_contact_cost = self._cost_foot_contact(data, feet_contact, stance_gait, jp.array(1.0))
        foot_slip_cost = self._cost_foot_slip(data, stance_gait)
        straight_knee = self._cost_straight_knee(data.qpos[jp.array(self._knee_indices) + 7])

        # Torque-effort and smoothness
        joint_torque = self._cost_torque(data.actuator_force)
        smoothness_joint = self._cost_smoothness_joint(data, info["last_joint_vel"])
        smoothness = -jp.sum((action - info["last_act"]) ** 2)

        # Joint limits (only robot joints; box freejoint has no meaningful range)
        joint_limits = self._cost_joint_pos_limits(data.qpos[7:7 + NUM_ROBOT_JOINTS])

        reward_dict = {
            "reach": reach,
            "lift": lift,
            "hold_stable": hold_stable,
            "box_upright": box_upright,
            "upright": upright,
            "foot_contact": foot_contact_cost,
            "foot_slip": foot_slip_cost,
            "straight_knee": straight_knee,
            "joint_torque": joint_torque,
            "smoothness_joint": smoothness_joint,
            "smoothness": smoothness,
            "joint_limits": joint_limits,
        }
        for k, v in reward_dict.items():
            reward_dict[k] = jp.where(jp.isnan(v), 0.0, v)
        return reward_dict
