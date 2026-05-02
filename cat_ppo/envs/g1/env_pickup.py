"""G1 Box Pickup environment: robot reaches and lifts a box from a support surface.

Phase 1 of the CaTra curriculum:
- Robot stands in place but can crouch using sagittal-plane leg joints.
- 23-DOF action space: all leg joints (12) + waist (3) + arms (8).
- Box placed 0.3 m in front of robot on a support surface at random height.
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
    SUPPORT_QPOS_START,
    NUM_ROBOT_JOINTS,
    G1CaTraEnv,
    domain_randomize_catra,
    torque_step_catra,
)
from cat_ppo.envs.g1.pickup_warmstart import pickup_obs_from_data

# 23-DOF action space: all leg joints (12) + waist (3) + arms (8).
PICKUP_ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
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

            # Box size: sample per-env half-extents for x, y, and z.
            # reset() uses the nominal half_z (0.15 m, the XML max) to compute box_z, so
            # DR'd boxes are always placed at or slightly above the pillar top — never embedded.
            rng, key = jax.random.split(rng)
            box_half_x = jax.random.uniform(key, minval=0.10, maxval=0.15)
            rng, key = jax.random.split(rng)
            box_half_y = jax.random.uniform(key, minval=0.10, maxval=0.20)
            rng, key = jax.random.split(rng)
            box_half_z = jax.random.uniform(key, minval=0.10, maxval=0.15)
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
      num_obs = 108  (state, deployable)
      num_pri = 147  (privileged_state)
    """
    env_config = config_dict.create(
        task_type="flat_terrain_catra",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=200,
        action_repeat=1,
        action_scale=0.5,
        num_obs=108,
        num_pri=147,
        num_act=23,
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
                reach=1.5,
                lift=2.0,
                hand_contact=2.0,
                box_pillar_contact=-1.5,
                grasp_symmetry=-2.0,
                palm_orient=2.0,
                hands_level=-1.0,
                hold_stable=0.0,
                box_yaw_stable=0.0, # -2.0
                box_centering=0.0,  # -2.0
                box_vertical=-0.5,   # -5.0
                box_upright=0.0,
                upright=3.0,    # 1.0 in CAT
                foot_contact=-0.5,
                foot_slip=-0.1,
                straight_knee=-5.0,
                joint_torque=-1e-4,
                smoothness_joint=-1e-6,
                smoothness=1e-3,
                joint_limits=-1.0,
                base_height=1.0,
                foot_balance=-30.0,
            ),
            base_height_target=0.75,
            foot_height_stance=0.0,
        ),
        box_surface_height_range=[0.3, 0.3],  # body centre=0.3m → pillar top at 0.6m
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


def g1_stand_task_config() -> config_dict.ConfigDict:
    """Config for G1Stand: leg-only standing policy (12-DOF), no box.

    Observation dimensions:
      num_obs = 54   (state, deployable)
      num_pri = 62   (privileged_state)
    """
    cfg = g1_pickup_task_config()
    cfg.env_config.num_obs = 54
    cfg.env_config.num_pri = 62
    cfg.env_config.num_act = 12
    return cfg


cat_ppo.registry.register("G1Stand", "config")(g1_stand_task_config())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@cat_ppo.registry.register("G1Pickup", "train_env_class")
class G1PickupEnv(G1CaTraEnv):
    """G1 humanoid reaching for and lifting a box from a support surface.

    Extends G1CaTraEnv with:
    - 23-DOF action space (all leg joints + waist + arms)
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
        """Override to 23-DOF action space and cache pickup IDs."""
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
        self._hand_geom_ids = np.array([
            self._mj_model.geom("left_hand_collision").id,
            self._mj_model.geom("right_hand_collision").id,
        ])

        # Gyro sensor address for pickup_obs_from_data
        _sid = self._mj_model.sensor("gyro_pelvis").id
        self._gyro_sensor_adr = int(self._mj_model.sensor_adr[_sid])
        self._gyro_sensor_dim = int(self._mj_model.sensor_dim[_sid])

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)  # 23

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset with 50-dim qpos; place box 3 m in front on support surface."""
        qpos = self._init_q.copy()  # (50,): robot default + box + support placeholders
        qvel = jp.zeros(self.mjx_model.nv)  # (47,)

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

        # --- Place box 0.3 m in front of robot on support surface ---
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        forward_xy = jp.array([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z)])
        box_xy = qpos[:2] + 0.3 * forward_xy

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

        # Reposition support pillar via qpos (freejoint body, not mocap).
        # Yaw matches box so the rectangular pillar faces the same direction.
        new_qpos = data.qpos.at[SUPPORT_QPOS_START:SUPPORT_QPOS_START + 3].set(
            jp.array([box_xy[0], box_xy[1], surface_z])
        )
        new_qpos = new_qpos.at[SUPPORT_QPOS_START + 3:SUPPORT_QPOS_START + 7].set(box_quat)
        data = data.replace(qpos=new_qpos)

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

        # Box size and mass from current (DR'd) model — stored in info so reward/obs can use it
        box_size = self.mjx_model.geom_size[self._box_geom_id]  # (3,) half-extents
        box_mass = self.mjx_model.body_mass[self._box_body_id]  # scalar

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
            "box_mass": box_mass,
            "box_xy_init": box_xy,   # pillar center XY at reset; box should rise straight up
            "box_yaw_init": box_yaw, # box yaw at reset; yaw should stay stable during grasp
            "box_z_init": box_z,     # box z at reset; used for drop termination threshold
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
        """Step with 23-DOF action; wrist joints stay at default."""
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
        # reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        reward = sum(rewards.values()) * self.dt

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
        """108-dim state (deployable, noisy) and 147-dim privileged_state (noiseless + extras).

        Thin wrapper around pickup_obs_from_data.
        """
        obs, updated_rng = pickup_obs_from_data(
            data, info,
            gyro_sensor_adr=self._gyro_sensor_adr,
            gyro_sensor_dim=self._gyro_sensor_dim,
            pelvis_body_id=self._pelvis_body_id,
            pelvis_imu_site_id=self._pelvis_imu_site_id,
            torso_imu_site_id=self._torso_imu_site_id,
            hands_site_id=self._hands_site_id,
            shlds_site_id=self._shlds_site_id,
            box_body_id=self._box_body_id,
            action_joint_ids=self.action_joint_ids,
            default_qpos=self._default_qpos,
            noise_config=self._config.noise_config,
            dt=self.dt,
        )
        info["rng"] = updated_rng
        return obs

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]):
        """Terminate on robot fall, box drop, or NaN. Returns (done, fall, box_drop, nan_term)."""
        fall = self.get_gravity(data, "pelvis")[2] < 0.0
        fall |= info["head_pos"][2] < 0.5
        box_drop = data.xpos[self._box_body_id][2] < (info["box_z_init"] - 0.1)
        nan_term = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return fall | box_drop | nan_term, fall, box_drop, nan_term

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

        # Reach: guide each hand to its respective box face (left hand → left face, right → right face).
        # The box local +Y axis maps to the right face in world frame; -Y maps to the left face.
        box_quat = data.xquat[self._box_body_id]  # wxyz
        box_left_axis = math.rotate(jp.array([0., 1., 0.]), box_quat)  # world-frame +Y of box = robot's left
        left_target = box_pos + box_left_axis * box_half_y   # left face center
        right_target = box_pos - box_left_axis * box_half_y  # right face center
        reach = -(jp.linalg.norm(left_palm_pos - left_target) + jp.linalg.norm(right_palm_pos - right_target))

        # Hand contact: reward when both hands are touching the box (0, 0.5, or 1.0).
        left_contact = geoms_colliding(data, self._hand_geom_ids[0], self._box_geom_id)
        right_contact = geoms_colliding(data, self._hand_geom_ids[1], self._box_geom_id)
        hand_contact = 0.5 * (left_contact.astype(jp.float32) + right_contact.astype(jp.float32))

        # Box-pillar contact: penalize box remaining on the pillar — encourages liftoff.
        box_pillar_contact = geoms_colliding(data, self._box_geom_id, self._box_support_geom_id).astype(jp.float32)

        # Grasp symmetry: penalize asymmetric hand placement along the box's local Z (height)
        # and X (front-back depth) axes. Both palms should be at the same height and depth.
        box_up_axis = math.rotate(jp.array([0., 0., 1.]), box_quat)    # box local Z in world
        box_fwd_axis = math.rotate(jp.array([1., 0., 0.]), box_quat)   # box local X in world
        left_rel = left_palm_pos - box_pos
        right_rel = right_palm_pos - box_pos
        height_diff = jp.dot(left_rel - right_rel, box_up_axis)
        depth_diff = jp.dot(left_rel - right_rel, box_fwd_axis)
        grasp_symmetry = height_diff ** 2 + depth_diff ** 2

        # Palm orientation: reward palm normals aligned with the inward normals of the
        # left/right box faces. The target direction is a property of the box pose only
        # (independent of current palm position), so the optimum is well-defined and
        # non-degenerate even when the palm is exactly at the target.
        # Site-frame convention: left_palm inward = -local_Y, right_palm inward = +local_Y.
        left_xmat = data.site_xmat[self._hands_site_id[0]].reshape(3, 3)
        right_xmat = data.site_xmat[self._hands_site_id[1]].reshape(3, 3)
        left_palm_normal = -left_xmat[:, 1]
        right_palm_normal = right_xmat[:, 1]
        # box_left_axis is the world-frame +Y of the box (unit length). Left face inward
        # normal = -box_left_axis; right face inward normal = +box_left_axis.
        left_dot = jp.dot(left_palm_normal, -box_left_axis)
        right_dot = jp.dot(right_palm_normal, box_left_axis)
        # Map dot in [-1, 1] -> reward in [0, 1] so there is always gradient, even when
        # the palm starts facing outward.
        palm_orient = 0.5 * (0.5 * (1.0 + left_dot) + 0.5 * (1.0 + right_dot))

        # Hands level: encourage the line between the two palms to lie in the horizontal
        # plane (angle between (left - right) and world +Z is ±90°). Uses the normalized
        # squared z-component = sin^2(angle from XY). Range [0, 1]: 0 when hands are at
        # the same height, 1 when one palm is directly above the other. Use with a
        # negative scale to penalize non-horizontal placement.
        hands_vec = left_palm_pos - right_palm_pos
        hands_level = hands_vec[2] ** 2 / (jp.dot(hands_vec, hands_vec) + 1e-6)

        # Lift: reward up to +10 cm above surface, penalize lifting higher than 10 cm.
        lift_height = box_z - (info["surface_z"] + info["support_half_z"] + box_half_z)
        lift = jp.clip(lift_height, 0.0, 0.10) / 0.10 - jp.clip(lift_height - 0.10, 0.0, None)

        # Box vertical: penalize XY drift from pillar center — box should rise straight up.
        # Gated on both hands touching the box so it only activates during an active grasp.
        both_hands = left_contact & right_contact
        box_xy_drift = jp.linalg.norm(box_pos[:2] - info["box_xy_init"])
        box_vertical = jp.where(both_hands, box_xy_drift ** 2, 0.0)

        # Hold stable: penalize box tumbling and translation
        box_linvel = data.qvel[BOX_QVEL_START:BOX_QVEL_START + 3]
        box_angvel = data.qvel[BOX_QVEL_START + 3:BOX_QVEL_START + 6]
        hold_stable = -(jp.linalg.norm(box_linvel) + jp.linalg.norm(box_angvel))

        # Box yaw stable: penalize yaw deviation from initial box yaw, gated on both hands touching.
        # Extracts current box yaw from quaternion via atan2 and compares to reset yaw.
        qw, qz = box_quat[0], box_quat[3]
        box_yaw_now = 2.0 * jp.arctan2(qz, qw)
        yaw_diff = box_yaw_now - info["box_yaw_init"]
        # Wrap to [-pi, pi]
        yaw_diff = (yaw_diff + jp.pi) % (2 * jp.pi) - jp.pi
        box_yaw_stable = jp.where(both_hands, yaw_diff ** 2, 0.0)

        # Box upright: keep box z-axis aligned with world z (important for CaTra handoff).
        # Only active once the box is off the surface — avoids free reward while resting on pillar.
        qx, qy = box_quat[1], box_quat[2]
        box_tilt_cos = jp.clip(1.0 - 2.0 * (qx ** 2 + qy ** 2), -1.0, 1.0)
        box_tilt_angle = jp.arccos(box_tilt_cos)
        box_upright = jp.where(lift_height > 0.0, jp.exp(-box_tilt_angle ** 2), 0.0)

        # Box centering: penalize box being laterally offset from the torso's forward axis.
        # Uses torso right axis (column 1 of torso_rot) to measure left-right skew.
        pelvis_rot = data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        torso_rot = data.site_xmat[self._torso_imu_site_id].reshape(3, 3)
        torso_pos = data.site_xpos[self._torso_imu_site_id]
        torso_right = torso_rot[:, 1]  # torso local Y = right direction in world frame
        lateral_offset = jp.dot(box_pos - torso_pos, torso_right)
        box_centering = lateral_offset ** 2

        # Robot upright: CAT asymmetric formula — backward lean is double-penalised.
        pelvis_rpy = jp.array(jaxlie.SO3.from_matrix(pelvis_rot).as_rpy_radians())
        torso_rpy = jp.array(jaxlie.SO3.from_matrix(torso_rot).as_rpy_radians())
        err_roll = jp.abs(pelvis_rpy[0]) + jp.abs(torso_rpy[0])
        err_pitch_back = jp.abs(jp.clip(torso_rpy[1], -jp.pi, 0.0))  # backward lean only
        err_pitch_any = jp.abs(torso_rpy[1])                          # all pitch (idle_mask=1 always)
        upright = jp.exp(-0.5 * (err_roll + err_pitch_back + err_pitch_any)) - err_pitch_back

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

        # Base height: reward standing/reaching from correct pelvis height (0.75 m target).
        base_height = self._reward_base_height(data.qpos[2], jp.zeros(()))

        # Foot balance: penalise pelvis COM drifting off-center between feet,
        # and feet getting too close together (< 0.35 m).
        pelvis_com_xy = data.subtree_com[self.body_id_pelvis][:2]
        left_foot_xy = data.site_xpos[self._feet_site_id[0]][:2]
        right_foot_xy = data.site_xpos[self._feet_site_id[1]][:2]
        foot_center = left_foot_xy + right_foot_xy - 2 * pelvis_com_xy
        centering_cost = jp.sum(jp.square(foot_center))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_dist = jp.linalg.norm(foot_pos[0] - foot_pos[1])
        spread_penalty = jp.where(foot_dist < 0.35, (0.35 - foot_dist) * 10, 0.0)
        foot_balance = centering_cost * (1 + spread_penalty)

        reward_dict = {
            "reach": reach,
            "lift": lift,
            "hand_contact": hand_contact,
            "box_pillar_contact": box_pillar_contact,
            "grasp_symmetry": grasp_symmetry,
            "palm_orient": palm_orient,
            "hands_level": hands_level,
            "hold_stable": hold_stable,
            "box_yaw_stable": box_yaw_stable,
            "box_centering": box_centering,
            "box_vertical": box_vertical,
            "box_upright": box_upright,
            "upright": upright,
            "foot_contact": foot_contact_cost,
            "foot_slip": foot_slip_cost,
            "straight_knee": straight_knee,
            "joint_torque": joint_torque,
            "smoothness_joint": smoothness_joint,
            "smoothness": smoothness,
            "joint_limits": joint_limits,
            "base_height": base_height,
            "foot_balance": foot_balance,
        }
        for k, v in reward_dict.items():
            reward_dict[k] = jp.where(jp.isnan(v), 0.0, v)
        return reward_dict


# ---------------------------------------------------------------------------
# G1Stand: leg-only standing policy (inherits from G1PickupEnv)
# ---------------------------------------------------------------------------

# 12-DOF action space: all leg joints (same as consts.ACTION_JOINT_NAMES)
STAND_ACTION_JOINT_NAMES = list(consts.ACTION_JOINT_NAMES)


@cat_ppo.registry.register("G1Stand", "train_env_class")
class G1StandEnv(G1PickupEnv):
    """Leg-only standing policy (12-DOF). No box in obs; box rewards all have scale=0.

    Inherits reset/step/rewards/termination from G1PickupEnv unchanged.
    Only the action joints and observations differ.

    State (54-dim):
        gyro_pelvis(3), gvec_pelvis(3),
        joint_angles[legs](12), joint_vel[legs](12),
        last_action(12), motor_targets[legs](12)

    Privileged (62-dim):
        [same 54 noiseless]
        + pelvis_pos(3), torso_pos(3), kp_scale(1), kd_scale(1)
    """

    def _post_init_pickup(self) -> None:
        """Override to 12-DOF leg-only action space."""
        self.action_joint_names = STAND_ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = jp.array([
            self.mj_model.actuator(name).id for name in self.action_joint_names
        ])
        self._pelvis_body_id = self._mj_model.body("pelvis").id
        self._box_geom_id = self._mj_model.geom(consts.BOX_GEOM).id
        self._box_support_geom_id = self._mj_model.geom("box_support_col").id

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)  # 12

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
        """54-dim state (deployable, noisy) and 62-dim privileged_state (noiseless + extras).

        State (54):
            gyro_pelvis[+noise](3), gvec_pelvis[+noise](3),
            joint_angles[legs, +noise](12), joint_vel[legs, +noise](12),
            last_action(12), motor_targets[legs](12)

        Privileged (62 = 54 noiseless + 8 extras):
            [same 54 fields, noiseless]
            + pelvis_pos(3), torso_pos(3), kp_scale(1), kd_scale(1)
        """
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0., 0., -1.])
        joint_angles = data.qpos[7:7 + NUM_ROBOT_JOINTS]
        joint_vel = data.qvel[6:6 + NUM_ROBOT_JOINTS]

        pelv_site_pos = data.site_xpos[self._pelvis_imu_site_id]
        tors_site_pos = data.site_xpos[self._torso_imu_site_id]

        privileged_state = jp.hstack([
            gyro_pelvis, gvec_pelvis,
            (joint_angles - self._default_qpos)[self.action_joint_ids],
            joint_vel[self.action_joint_ids],
            info["last_act"],
            info["motor_targets"][self.action_joint_ids],
            pelv_site_pos, tors_site_pos,
            info["kp_scale"].reshape(1), info["kd_scale"].reshape(1),
        ])

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
        ])

        return {
            "state": jp.nan_to_num(state),
            "privileged_state": jp.nan_to_num(privileged_state),
        }

