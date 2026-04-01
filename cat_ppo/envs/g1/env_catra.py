# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CaTra (Carry and Traverse) environment: G1 humanoid navigates while carrying a box."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math
from ml_collections import config_dict
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
import numpy as np

import cat_ppo
from cat_ppo.envs.g1.env_cat import (
    G1CatEnv,
    torque_step,
    world_to_navi_vel,
    delay_body_pos,
    delay_rootpose_noisy,
    normalize,
    base2navi_transform,
    EPS,
)
from cat_ppo.envs.g1 import constants as consts

import jaxlie

# Number of robot joints (excluding root freejoint and box freejoint)
NUM_ROBOT_JOINTS = 29

# Box freejoint qpos adds 7 elements after the robot joints.
# qpos layout: [0:7] root freejoint, [7:36] robot joints, [36:43] box freejoint
# qvel layout: [0:6] root vel, [6:35] robot joints vel, [35:41] box vel
BOX_QPOS_START = 7 + NUM_ROBOT_JOINTS   # 36
BOX_QVEL_START = 6 + NUM_ROBOT_JOINTS   # 35


def torque_step_catra(
        rng: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
        qpos_des: jax.Array,
        kps: jax.Array,
        kds: jax.Array,
        kp_scale: jax.Array,
        kd_scale: jax.Array,
        rfi_lim_scale: jax.Array,
        torque_limit: jax.Array,
        n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data]:
    """Like torque_step but slices only robot joints from qpos/qvel (ignoring box freejoint)."""
    def single_step(carry, _):
        rng, data = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        # Only use robot joint slice to avoid shape mismatch with box freejoint DOF
        pos_err = qpos_des - data.qpos[7:7 + NUM_ROBOT_JOINTS]
        vel_err = -data.qvel[6:6 + NUM_ROBOT_JOINTS]
        torque = (kp_scale * kps) * pos_err + (kd_scale * kds) * vel_err

        rfi_noise = rfi_lim_scale * jax.random.uniform(rng_rfi, shape=torque.shape, minval=-1.0, maxval=1.0)
        torque += rfi_noise
        torque = jp.clip(torque, -torque_limit, torque_limit)

        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)

        return (rng, data), None

    return jax.lax.scan(single_step, (rng, data), (), n_substeps)[0]


def domain_randomize_catra(model: mjx.Model, rng: jax.Array):
    """Domain randomization for CaTra: handles 43-dim qpos (robot 36 + box 7)."""
    FLOOR_GEOM_ID = 0
    TORSO_BODY_ID = 16

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
        dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        # Only perturb robot joints [7:36], not box freejoint [36:43]
        qpos0 = qpos0.at[7:7 + NUM_ROBOT_JOINTS].set(
            qpos0[7:7 + NUM_ROBOT_JOINTS]
            + jax.random.uniform(key, shape=(NUM_ROBOT_JOINTS,), minval=-0.05, maxval=0.05)
        )

        return (
            pair_friction,
            dof_frictionloss,
            dof_armature,
            body_ipos,
            body_mass,
            qpos0,
        )

    (
        pair_friction,
        frictionloss,
        armature,
        body_ipos,
        body_mass,
        qpos0,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "pair_friction": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "qpos0": 0,
        }
    )

    model = model.tree_replace(
        {
            "pair_friction": pair_friction,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "qpos0": qpos0,
        }
    )

    return model, in_axes


def g1_catra_task_config() -> config_dict.ConfigDict:
    """Config for CaTra: carries from g1_loco_task_config but with 23-dim action + box rewards.

    Observation dimensions:
      num_obs = 191 = 162 (base) + 11 (last_act 12→23) + 11 (motor_targets 12→23) + 7 (box PF)
      num_pri = 259 = 224 (base) + 11 + 11 + 7 + 6 (box_pos + box_vel)
    NOTE: verify these after the first training run if ONNX export fails.
    """
    env_config = config_dict.create(
        task_type="flat_terrain_catra",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.3,  # smaller than 0.5 to be gentle on arms (25 Nm vs 88-139 Nm legs)
        history_len=15,
        num_obs=191,
        num_pri=259,
        num_act=23,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        gait_config=config_dict.create(
            gait_bound=0.6,
            freq_range=[1.3, 1.5],
            foot_height_range=[0.07, 0.07],
        ),
        dm_rand_config=config_dict.create(
            enable_pd=True,
            kp_range=[0.75, 1.25],
            kd_range=[0.75, 1.25],
            enable_rfi=True,
            rfi_lim=0.1,
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
                # behavior rewards (inherited from G1CatEnv)
                tracking_orientation=2.0,
                tracking_root_field=1.0,
                body_motion=-0.5,
                body_rotation=1.0,
                foot_contact=-1.0,
                foot_clearance=-15.0,
                foot_slip=-0.5,
                foot_balance=-30,
                foot_far=-0,
                straight_knee=-30,
                # energy rewards
                smoothness_joint=-1e-6,
                smoothness_action=-1e-3,
                joint_limits=-1.0,
                joint_torque=-1e-4,
                # body HumanoidPF fields
                headgf=0.0,
                handsgf=0.0,
                feetgf=0.0,
                headdf=0.0,
                handsdf=0.0,
                feetdf=0.0,
                kneesdf=0.0,
                shldsdf=0.0,
                # box-specific rewards (activate with --box flag)
                boxgf=0.0,
                boxdf=0.0,
                # arm stability rewards (always active with negative scales)
                arm_pose=-0.5,
                arm_smoothness=-1e-3,
            ),
            base_height_target=0.75,
            foot_height_stance=0.0,
        ),
        term_collision_threshold=0.04,
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        command_config=config_dict.create(
            resampling_time=10.0,
            stop_prob=0.2,
        ),
        lin_vel_x=[-0.5, 0.5],
        lin_vel_y=[-0.3, 0.3],
        ang_vel_yaw=[-0.5, 0.5],
        torso_height=[0.5, consts.DEFAULT_CHEST_Z],
        pf_config=config_dict.create(
            path='data/assets/TypiObs/empty',
            dx=0.04,
            origin=np.array([-0.5, -1.0, 0.0], dtype=np.float32),
        ),
        box_config=config_dict.create(
            mass_range=[0.5, 3.0],
        ),
    )

    policy_config = config_dict.create(
        num_timesteps=5_000_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        madrona_backend=False,
        augment_pixels=False,
        num_envs=32768,
        episode_length=1000,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=domain_randomize_catra,
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
        duration=50.0,
        command_waypoints=np.array([[0, 0.0, 0.0, 0.0]]),
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
        eval_config=eval_config,
    )
    return config


cat_ppo.registry.register("G1CaTra", "config")(g1_catra_task_config())


@cat_ppo.registry.register("G1CaTra", "train_env_class")
class G1CaTraEnv(G1CatEnv):
    """G1 humanoid carrying a box while navigating obstacles.

    Extends G1CatEnv with:
    - 23-dim action space (12 leg + 3 waist + 8 arm joints)
    - Box PF observations (boxgf, boxbf, boxdf sampled at box_center site)
    - Box-specific rewards (boxgf alignment, boxdf SDF penalty)
    - Arm pose stabilization reward
    - Box collision termination
    - CaTra scene XML with box body + weld equality constraints
    """

    def __init__(
            self,
            task_type: str = "flat_terrain_catra",
            config: config_dict.ConfigDict = None,
            config_overrides: Optional[Dict[str, Union[str, Any, list[Any]]]] = None,
    ) -> None:
        # G1CatEnv.__init__ calls G1LocoEnv.__init__ which calls _post_init.
        # We then override what we need in _post_init_catra.
        super().__init__(
            task_type=task_type,
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init_catra()

    def _post_init_catra(self) -> None:
        """Override action joints, default qpos, and init_q for CaTra."""
        # 23-dim action space: legs + waist + arms
        self.action_joint_names = consts.CATRA_ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = jp.array([
            self.mj_model.actuator(name).id for name in self.action_joint_names
        ])

        # Fix soft limits: jnt_range[1:] now includes the box freejoint at the end.
        # We only want the 29 robot joint limits; the box freejoint has no meaningful range.
        lowers, uppers = self.mj_model.jnt_range[1:1 + NUM_ROBOT_JOINTS].T
        c = (lowers + uppers) / 2
        r = uppers - lowers
        factor = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * factor
        self._soft_uppers = c + 0.5 * r * factor

        # Carrying pose: DEFAULT_QPOS_CATRA[7:36] = robot joints (29-dim)
        self._default_qpos = jp.array(consts.DEFAULT_QPOS_CATRA[7:7 + NUM_ROBOT_JOINTS])

        # Extended init_q: robot (36-dim) + box freejoint (7-dim = pos + quat)
        # Box starts at a reasonable position; weld constraints pull it to the correct pose.
        box_default = np.array([0.35, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._init_q = jp.array(np.concatenate([consts.DEFAULT_QPOS_CATRA, box_default]))

        # Box identifiers
        self._box_site_id = self._mj_model.site(consts.BOX_SITE).id
        self._box_body_id = self._mj_model.body("carried_box").id

        # Arm joint indices for the arm_pose reward (shoulder + elbow, no wrists)
        arm_obs_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
        ]
        self._arm_actuator_ids = jp.array([self.mj_model.actuator(n).id for n in arm_obs_names])

        # Target arm angles from carrying pose (for arm_pose reward)
        self._carry_pose_arm = jp.array(consts.DEFAULT_QPOS_CATRA[7:][self._arm_actuator_ids])

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)  # 23

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset with 43-dim qpos (robot 36 + box 7) and initialize box PF fields."""
        qpos = self._init_q.copy()                              # (43,)
        qvel = jp.zeros(self.mjx_model.nv)                      # (41,) = 6+29+6

        # Random spawn xy
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        qpos = qpos.at[2].set(0.8)

        # Random initial yaw
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

        # Random root velocity
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        # ctrl is 29-dim (robot actuators only)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:7 + NUM_ROBOT_JOINTS])

        # --- Phase 3: place box at palm midpoint using FK ---
        # After init, FK is computed and site_xpos is available. Set box freejoint qpos
        # to the midpoint between the two palms so the box starts in the hands.
        left_palm_pos  = data.site_xpos[self._hands_site_id[0]]
        right_palm_pos = data.site_xpos[self._hands_site_id[1]]
        box_init_pos   = (left_palm_pos + right_palm_pos) / 2.0
        new_qpos = data.qpos.at[BOX_QPOS_START:BOX_QPOS_START + 3].set(box_init_pos)
        new_qpos = new_qpos.at[BOX_QPOS_START + 3].set(1.0)                          # quat w
        new_qpos = new_qpos.at[BOX_QPOS_START + 4:BOX_QPOS_START + 7].set(0.0)      # quat xyz
        data = data.replace(qpos=new_qpos)
        data = mjx.forward(self.mjx_model, data)

        # --- Sample HumanoidPF fields for all body parts ---
        head_pos = data.site_xpos[self._head_site_id]
        head_vel = jp.zeros_like(head_pos)
        headgf = self.sample_field(self.gf, head_pos.reshape(1, -1))
        headbf = self.sample_field(self.bf, head_pos.reshape(1, -1))
        headdf = self.sample_field(self.sdf, head_pos.reshape(1, -1))
        pelv_pos = data.site_xpos[self._pelvis_imu_site_id]
        tors_pos = data.site_xpos[self._torso_imu_site_id]
        pelvgf = self.sample_field(self.gf, pelv_pos.reshape(1, -1))
        pelvbf = self.sample_field(self.bf, pelv_pos.reshape(1, -1))
        pelvdf = self.sample_field(self.sdf, pelv_pos.reshape(1, -1))
        torsgf = self.sample_field(self.gf, tors_pos.reshape(1, -1))
        torsbf = self.sample_field(self.bf, tors_pos.reshape(1, -1))
        torsdf = self.sample_field(self.sdf, tors_pos.reshape(1, -1))
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = jp.zeros_like(feet_pos)
        feetgf = self.sample_field(self.gf, feet_pos)
        feetbf = self.sample_field(self.bf, feet_pos)
        feetdf = self.sample_field(self.sdf, feet_pos)
        hands_pos = data.site_xpos[self._hands_site_id]
        hands_vel = jp.zeros_like(hands_pos)
        handsgf = self.sample_field(self.gf, hands_pos)
        handsbf = self.sample_field(self.bf, hands_pos)
        handsdf = self.sample_field(self.sdf, hands_pos)
        knees_pos = data.site_xpos[self._knees_site_id]
        kneesgf = self.sample_field(self.gf, knees_pos)
        kneesbf = self.sample_field(self.bf, knees_pos)
        kneesdf = self.sample_field(self.sdf, knees_pos)
        shlds_pos = data.site_xpos[self._shlds_site_id]
        shldsgf = self.sample_field(self.gf, shlds_pos)
        shldsbf = self.sample_field(self.bf, shlds_pos)
        shldsdf = self.sample_field(self.sdf, shlds_pos)

        command = self.compute_cmd_from_rtf(
            pelvgf.reshape(-1),
            jp.concat([headgf, feetgf, handsgf], axis=0),
            jp.concat([headbf, feetbf, handsbf], axis=0),
        )

        # --- Box PF fields ---
        box_pos = data.site_xpos[self._box_site_id]
        box_vel = jp.zeros(3)
        boxgf = self.sample_field(self.gf, box_pos.reshape(1, -1))
        boxbf = self.sample_field(self.bf, box_pos.reshape(1, -1))
        boxdf = self.sample_field(self.sdf, box_pos.reshape(1, -1))

        # --- Push interval ---
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        # --- Gait ---
        rng, gait_freq_rng, foot_height_rng = jax.random.split(rng, 3)
        gait_freq = jax.random.uniform(
            gait_freq_rng,
            minval=self._config.gait_config.freq_range[0],
            maxval=self._config.gait_config.freq_range[1],
        )
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        rng, phase_rng = jax.random.split(rng)
        cond_phase = jax.random.bernoulli(phase_rng)
        phase = jp.where(cond_phase, self._init_phase_l, self._init_phase_r)
        foot_height = jax.random.uniform(
            foot_height_rng,
            minval=self._config.gait_config.foot_height_range[0],
            maxval=self._config.gait_config.foot_height_range[1],
        )

        # --- Domain randomization scalars ---
        rng, key_kp, key_kd, key_rfi, key_delay = jax.random.split(rng, 5)
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

        rfi_lim_noise_scale = jax.random.uniform(
            key_rfi,
            self.torque_limit.shape,
            minval=self._config.dm_rand_config.rfi_lim_range[0],
            maxval=self._config.dm_rand_config.rfi_lim_range[1],
        )
        rfi_lim_scale = self._config.dm_rand_config.rfi_lim * rfi_lim_noise_scale * self.torque_limit
        rfi_lim_scale = jp.where(self._config.dm_rand_config.enable_rfi, rfi_lim_scale, jp.zeros_like(rfi_lim_scale))

        info = {
            "rng": rng,
            "step": 0,
            "command": command,
            # history
            "last_command": jp.zeros(4),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "last_feet_vel": jp.zeros(2),
            "last_joint_vel": np.zeros(NUM_ROBOT_JOINTS),
            # push
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # state
            "motor_targets": self._default_qpos.copy(),
            "local_lin_vel": jp.zeros(3),
            "global_lin_vel": jp.zeros(3),
            "global_ang_vel": jp.zeros(3),
            "navi2world_rot": jp.eye(3),
            "navi2world_pose": jp.eye(4),
            "navi_torso_rpy": jp.zeros(3),
            "navi_torso_lin_vel": jp.zeros(3),
            "navi_torso_ang_vel": jp.zeros(3),
            "navi_pelvis_rpy": jp.zeros(3),
            "navi_pelvis_lin_vel": jp.zeros(3),
            "navi_pelvis_ang_vel": jp.zeros(3),
            # Phase
            "stop_timestep": 100,
            "phase": phase,
            "phase_dt": phase_dt,
            "gait_mask": jp.zeros(2),
            "gait_freq": gait_freq,
            "foot_height": foot_height,
            # Domain randomization
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "rfi_lim_scale": rfi_lim_scale,
            # Body HumanoidPF fields (current)
            "headgf": headgf.copy(), "headbf": headbf.copy(), "headdf": headdf.copy(),
            "pelvgf": pelvgf.copy(), "pelvbf": pelvbf.copy(), "pelvdf": pelvdf.copy(),
            "torsgf": torsgf.copy(), "torsbf": torsbf.copy(), "torsdf": torsdf.copy(),
            "feetgf": feetgf.copy(), "feetbf": feetbf.copy(), "feetdf": feetdf.copy(),
            "handsgf": handsgf.copy(), "handsbf": handsbf.copy(), "handsdf": handsdf.copy(),
            "kneesgf": kneesgf.copy(), "kneesbf": kneesbf.copy(), "kneesdf": kneesdf.copy(),
            "shldsgf": shldsgf.copy(), "shldsbf": shldsbf.copy(), "shldsdf": shldsdf.copy(),
            # Body positions/velocities
            "head_pos": head_pos.copy(), "head_vel": head_vel.copy(),
            "pelv_pos": pelv_pos.copy(), "tors_pos": tors_pos.copy(),
            "feet_pos": feet_pos.copy(), "feet_vel": feet_vel.copy(),
            "hands_pos": hands_pos.copy(), "hands_vel": hands_vel.copy(),
            "knees_pos": knees_pos.copy(), "shlds_pos": shlds_pos.copy(),
            # Delay buffer (body HumanoidPF)
            "command_delay": command, "odom_delay": qpos[:7],
            "headgf_delay": headgf.copy(), "headbf_delay": headbf.copy(), "headdf_delay": headdf.copy(),
            "pelvgf_delay": pelvgf.copy(), "pelvbf_delay": pelvbf.copy(), "pelvdf_delay": pelvdf.copy(),
            "torsgf_delay": torsgf.copy(), "torsbf_delay": torsbf.copy(), "torsdf_delay": torsdf.copy(),
            "feetgf_delay": feetgf.copy(), "feetbf_delay": feetbf.copy(), "feetdf_delay": feetdf.copy(),
            "handsgf_delay": handsgf.copy(), "handsbf_delay": handsbf.copy(), "handsdf_delay": handsdf.copy(),
            "kneesgf_delay": kneesgf.copy(), "kneesbf_delay": kneesbf.copy(), "kneesdf_delay": kneesdf.copy(),
            "shldsgf_delay": shldsgf.copy(), "shldsbf_delay": shldsbf.copy(), "shldsdf_delay": shldsdf.copy(),
            # Box PF fields
            "boxgf": boxgf.copy(), "boxbf": boxbf.copy(), "boxdf": boxdf.copy(),
            "box_pos": box_pos.copy(), "box_vel": box_vel.copy(),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step with 23-dim action; updates box PF fields after physics step."""
        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push_signal = jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= push_signal
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(qvel[:2] + push * push_magnitude)
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # Set motor targets (23-dim action, 29-dim full motor_targets)
        lower_motor_targets = jp.clip(
            state.info["motor_targets"][self.action_joint_ids]
            + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)

        # Physics step: use torque_step_catra to handle extra box freejoint DOF in qpos/qvel
        state.info["rng"], data = torque_step_catra(
            state.info["rng"],
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            kp_scale=state.info["kp_scale"],
            kd_scale=state.info["kd_scale"],
            rfi_lim_scale=state.info["rfi_lim_scale"],
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )

        # Collect body state
        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        state.info["motor_targets"] = motor_targets
        state.info["local_lin_vel"] = self.get_local_linvel(data, "pelvis")
        state.info["global_lin_vel"] = self.get_global_linvel(data, "pelvis")
        state.info["global_ang_vel"] = self.get_global_angvel(data, "pelvis")

        # Navi frame
        pelvis2world_rot = data.site_xmat[self._pelvis_imu_site_id]
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        state.info["navi2world_pose"] = state.info["navi2world_pose"].at[:3, :3].set(navi2world_rot)
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[:2, 3].set(data.site_xpos[self._pelvis_imu_site_id][:2])
        )
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[2, 3].set(self._config.reward_config.base_height_target)
        )

        pelvis2navi_rot = navi2world_rot.T @ pelvis2world_rot
        state.info["navi2world_rot"] = navi2world_rot
        state.info["navi_pelvis_rpy"] = jp.array(jaxlie.SO3.from_matrix(pelvis2navi_rot).as_rpy_radians())
        state.info["navi_pelvis_lin_vel"] = pelvis2navi_rot @ self.get_local_linvel(data, "pelvis")
        state.info["navi_pelvis_ang_vel"] = pelvis2navi_rot @ self.get_gyro(data, "pelvis")
        torso2world_rot = data.site_xmat[self._torso_imu_site_id]
        torso2navi_rot = navi2world_rot.T @ torso2world_rot
        state.info["navi_torso_rpy"] = jp.array(jaxlie.SO3.from_matrix(torso2navi_rot).as_rpy_radians())
        state.info["navi_torso_lin_vel"] = torso2navi_rot @ self.get_local_linvel(data, "torso")
        state.info["navi_torso_ang_vel"] = torso2navi_rot @ self.get_gyro(data, "torso")

        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["last_command"] = state.info["command"].copy()

        # Sample body positions and PF fields
        head_pos = data.site_xpos[self._head_site_id]
        head_vel = (head_pos - state.info["head_pos"]) / self.dt
        pelv_pos = data.site_xpos[self._pelvis_imu_site_id]
        tors_pos = data.site_xpos[self._torso_imu_site_id]
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = (feet_pos - state.info["feet_pos"]) / self.dt
        hands_pos = data.site_xpos[self._hands_site_id]
        hands_vel = (hands_pos - state.info["hands_pos"]) / self.dt
        knees_pos = data.site_xpos[self._knees_site_id]
        shlds_pos = data.site_xpos[self._shlds_site_id]
        all_poses = jp.concatenate([
            head_pos.reshape(1, -1), pelv_pos.reshape(1, -1), tors_pos.reshape(1, -1),
            feet_pos, hands_pos, knees_pos, shlds_pos,
        ], axis=0)
        all_gf = self.sample_field(self.gf, all_poses)
        all_bf = self.sample_field(self.bf, all_poses)
        all_df = self.sample_field(self.sdf, all_poses)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = jp.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = jp.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = jp.split(all_df, [1, 2, 3, 5, 7, 9], axis=0)

        command = self.compute_cmd_from_rtf(
            pelvgf.reshape(-1),
            jp.concat([headgf, feetgf, handsgf], axis=0),
            jp.concat([headbf, feetbf, handsbf], axis=0),
        )
        state.info["command"] = command.copy()

        # Delay buffer update
        update_pf = (state.info["step"] % 5) == 0
        state.info["rng"], odo_key = jax.random.split(state.info["rng"], 2)
        odom_delay = jp.where(update_pf, data.qpos[:7], state.info["odom_delay"])
        p_gt = data.qpos[:3]; q_gt = data.qpos[3:7]
        p_odom = odom_delay[:3]; q_odom = odom_delay[3:7]
        all_poses_delay = delay_body_pos(p_gt, q_gt, p_odom, q_odom, all_poses)
        all_gf_delay = self.sample_field(self.gf, all_poses_delay)
        all_bf_delay = self.sample_field(self.bf, all_poses_delay)
        all_df_delay = self.sample_field(self.sdf, all_poses_delay)

        # Gait update
        self._update_phase(state)
        move_flag = state.info["command"][0]
        all_gf = all_gf * (move_flag[None] > 0.5) / (jp.linalg.norm(all_gf, axis=-1, keepdims=True) + EPS)
        all_bf = all_bf / (jp.linalg.norm(all_bf, axis=-1, keepdims=True) + EPS)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = jp.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = jp.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)

        all_gf_delay = all_gf_delay * (move_flag[None] > 0.5) / (jp.linalg.norm(all_gf_delay, axis=-1, keepdims=True) + EPS)
        all_bf_delay = all_bf_delay / (jp.linalg.norm(all_bf_delay, axis=-1, keepdims=True) + EPS)
        headgf_delay, pelvgf_delay, torsgf_delay, feetgf_delay, handsgf_delay, kneesgf_delay, shldsgf_delay = jp.split(all_gf_delay, [1, 2, 3, 5, 7, 9], axis=0)
        headbf_delay, pelvbf_delay, torsbf_delay, feetbf_delay, handsbf_delay, kneesbf_delay, shldsbf_delay = jp.split(all_bf_delay, [1, 2, 3, 5, 7, 9], axis=0)
        headdf_delay, pelvdf_delay, torsdf_delay, feetdf_delay, handsdf_delay, kneesdf_delay, shldsdf_delay = jp.split(all_df_delay, [1, 2, 3, 5, 7, 9], axis=0)
        command_delay = self.compute_cmd_from_rtf(
            pelvgf_delay.reshape(-1),
            jp.concat([headgf_delay, feetgf_delay, handsgf_delay], axis=0),
            jp.concat([headbf_delay, feetbf_delay, handsbf_delay], axis=0),
        )

        # --- Box PF fields (sampled after physics step) ---
        box_pos = data.site_xpos[self._box_site_id]
        box_vel = (box_pos - state.info["box_pos"]) / self.dt
        boxgf = self.sample_field(self.gf, box_pos.reshape(1, -1))
        boxbf = self.sample_field(self.bf, box_pos.reshape(1, -1))
        boxdf = self.sample_field(self.sdf, box_pos.reshape(1, -1))

        # Update info
        state.info["odom_delay"] = odom_delay.copy()
        state.info["headgf_delay"] = headgf_delay.copy(); state.info["headbf_delay"] = headbf_delay.copy(); state.info["headdf_delay"] = headdf_delay.copy()
        state.info["pelvgf_delay"] = pelvgf_delay.copy(); state.info["pelvbf_delay"] = pelvbf_delay.copy(); state.info["pelvdf_delay"] = pelvdf_delay.copy()
        state.info["torsgf_delay"] = torsgf_delay.copy(); state.info["torsbf_delay"] = torsbf_delay.copy(); state.info["torsdf_delay"] = torsdf_delay.copy()
        state.info["feetgf_delay"] = feetgf_delay.copy(); state.info["feetbf_delay"] = feetbf_delay.copy(); state.info["feetdf_delay"] = feetdf_delay.copy()
        state.info["handsgf_delay"] = handsgf_delay.copy(); state.info["handsbf_delay"] = handsbf_delay.copy(); state.info["handsdf_delay"] = handsdf_delay.copy()
        state.info["kneesgf_delay"] = kneesgf_delay.copy(); state.info["kneesbf_delay"] = kneesbf_delay.copy(); state.info["kneesdf_delay"] = kneesdf_delay.copy()
        state.info["shldsgf_delay"] = shldsgf_delay.copy(); state.info["shldsbf_delay"] = shldsbf_delay.copy(); state.info["shldsdf_delay"] = shldsdf_delay.copy()
        state.info["command_delay"] = command_delay.copy()

        state.info["headgf"] = headgf.copy(); state.info["headbf"] = headbf.copy(); state.info["headdf"] = headdf.copy()
        state.info["pelvgf"] = pelvgf.copy(); state.info["pelvbf"] = pelvbf.copy(); state.info["pelvdf"] = pelvdf.copy()
        state.info["torsgf"] = torsgf.copy(); state.info["torsbf"] = torsbf.copy(); state.info["torsdf"] = torsdf.copy()
        state.info["feetgf"] = feetgf.copy(); state.info["feetbf"] = feetbf.copy(); state.info["feetdf"] = feetdf.copy()
        state.info["handsgf"] = handsgf.copy(); state.info["handsbf"] = handsbf.copy(); state.info["handsdf"] = handsdf.copy()
        state.info["kneesgf"] = kneesgf.copy(); state.info["kneesbf"] = kneesbf.copy(); state.info["kneesdf"] = kneesdf.copy()
        state.info["shldsgf"] = shldsgf.copy(); state.info["shldsbf"] = shldsbf.copy(); state.info["shldsdf"] = shldsdf.copy()
        state.info["head_pos"] = head_pos.copy(); state.info["head_vel"] = head_vel.copy()
        state.info["pelv_pos"] = pelv_pos.copy(); state.info["tors_pos"] = tors_pos.copy()
        state.info["feet_pos"] = feet_pos.copy(); state.info["feet_vel"] = feet_vel.copy()
        state.info["hands_pos"] = hands_pos.copy(); state.info["hands_vel"] = hands_vel.copy()
        state.info["knees_pos"] = knees_pos.copy(); state.info["shlds_pos"] = shlds_pos.copy()
        state.info["push"] = push; state.info["push_step"] += 1; state.info["step"] += 1

        # Box info update
        state.info["boxgf"] = boxgf.copy(); state.info["boxbf"] = boxbf.copy(); state.info["boxdf"] = boxdf.copy()
        state.info["box_pos"] = box_pos.copy(); state.info["box_vel"] = box_vel.copy()

        # Update history
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        state.info["last_joint_vel"] = data.qvel[6:6 + NUM_ROBOT_JOINTS].copy()

        obs = self._get_obs(data, state.info, feet_contact)
        done = self._get_termination(data, state.info)

        rewards = self._get_reward(data, action, state.info, done, feet_contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        timeout = state.info["step"] >= self._config.episode_length
        state.info["step"] = jp.where(done | timeout, 0, state.info["step"])
        state.info["motor_targets"] = jp.where(done, self._default_qpos, state.info["motor_targets"])

        state.info["rng"], episode_rng = jax.random.split(state.info["rng"])
        _is_resample = jp.where(done, self.resample_domain_random_param(episode_rng, state), False)

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        state.info["last_feet_vel"] = data.sensordata[self._foot_linvel_sensor_adr][..., 2]
        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        """State (191-dim) and privileged_state (259-dim) including box PF fields.

        Extends G1CatEnv._get_obs by:
        - Using 23-dim last_act and motor_targets (instead of 12-dim)
        - Appending box PF fields (boxgf=3, boxbf=3, boxdf=1) to state
        - Appending box PF + box_pos + box_vel to privileged_state
        """
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel(data, "pelvis")
        # Use robot joints only: qpos[7:36], qvel[6:35]
        joint_angles = data.qpos[7:7 + NUM_ROBOT_JOINTS]
        joint_vel = data.qvel[6:6 + NUM_ROBOT_JOINTS]
        gait_phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])

        navi2world_pose = info["navi2world_pose"]

        # --- Privileged state (noiseless) ---
        privileged_state = jp.hstack([
            gyro_pelvis, gvec_pelvis,
            (joint_angles - self._default_qpos)[self.obs_joint_ids],
            joint_vel[self.obs_joint_ids],
            info["last_act"],                               # 23-dim
            info["motor_targets"][self.action_joint_ids],   # 23-dim
            info["command"], info["foot_height"], gait_phase,
            linvel_pelvis,
            # Body PF fields (non-delayed, world frame)
            info["headgf"].reshape(-1), info["headbf"].reshape(-1), info["headdf"].reshape(-1),
            info["pelvgf"].reshape(-1), info["pelvbf"].reshape(-1), info["pelvdf"].reshape(-1),
            info["torsgf"].reshape(-1), info["torsbf"].reshape(-1), info["torsdf"].reshape(-1),
            info["feetgf"].reshape(-1), info["feetbf"].reshape(-1), info["feetdf"].reshape(-1),
            info["handsgf"].reshape(-1), info["handsbf"].reshape(-1), info["handsdf"].reshape(-1),
            info["kneesgf"].reshape(-1), info["kneesbf"].reshape(-1), info["kneesdf"].reshape(-1),
            info["shldsgf"].reshape(-1), info["shldsbf"].reshape(-1), info["shldsdf"].reshape(-1),
            # Body positions/velocities
            info["head_pos"].reshape(-1), info["head_vel"].reshape(-1),
            info["pelv_pos"].reshape(-1), info["tors_pos"].reshape(-1),
            info["feet_pos"].reshape(-1), info["feet_vel"].reshape(-1),
            info["hands_pos"].reshape(-1), info["hands_vel"].reshape(-1),
            info["knees_pos"].reshape(-1), info["shlds_pos"].reshape(-1),
            info["navi_torso_rpy"][:2], info["gait_mask"], feet_contact,
            # Domain randomization
            info["kp_scale"], info["kd_scale"], info["rfi_lim_scale"],
            # Box PF + pose (privileged extra)
            info["boxgf"].reshape(-1), info["boxbf"].reshape(-1), info["boxdf"].reshape(-1),
            info["box_pos"].reshape(-1), info["box_vel"].reshape(-1),
        ])

        # --- Noisy observations for deployable state ---
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro_pelvis = gyro_pelvis + (2 * jax.random.uniform(noise_rng, shape=gyro_pelvis.shape) - 1) \
            * self._config.noise_config.level * self._config.noise_config.scales.gyro

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gvec_pelvis = gvec_pelvis + (2 * jax.random.uniform(noise_rng, shape=gvec_pelvis.shape) - 1) \
            * self._config.noise_config.level * self._config.noise_config.scales.gravity

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = joint_angles + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1) \
            * self._config.noise_config.level * self._config.noise_config.scales.joint_pos

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = joint_vel + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1) \
            * self._config.noise_config.level * self._config.noise_config.scales.joint_vel

        # Body PF fields: delayed + nav-frame transform
        headgf = world_to_navi_vel(navi2world_pose, info["headgf_delay"].reshape(-1, 3))
        headbf = world_to_navi_vel(navi2world_pose, info["headbf_delay"].reshape(-1, 3))
        pelvgf = world_to_navi_vel(navi2world_pose, info["pelvgf_delay"].reshape(-1, 3))
        pelvbf = world_to_navi_vel(navi2world_pose, info["pelvbf_delay"].reshape(-1, 3))
        torsgf = world_to_navi_vel(navi2world_pose, info["torsgf_delay"].reshape(-1, 3))
        torsbf = world_to_navi_vel(navi2world_pose, info["torsbf_delay"].reshape(-1, 3))
        feetgf = world_to_navi_vel(navi2world_pose, info["feetgf_delay"].reshape(-1, 3))
        feetbf = world_to_navi_vel(navi2world_pose, info["feetbf_delay"].reshape(-1, 3))
        handsgf = world_to_navi_vel(navi2world_pose, info["handsgf_delay"].reshape(-1, 3))
        handsbf = world_to_navi_vel(navi2world_pose, info["handsbf_delay"].reshape(-1, 3))
        kneesgf = world_to_navi_vel(navi2world_pose, info["kneesgf_delay"].reshape(-1, 3))
        kneesbf = world_to_navi_vel(navi2world_pose, info["kneesbf_delay"].reshape(-1, 3))
        shldsgf = world_to_navi_vel(navi2world_pose, info["shldsgf_delay"].reshape(-1, 3))
        shldsbf = world_to_navi_vel(navi2world_pose, info["shldsbf_delay"].reshape(-1, 3))
        command = info["command"].copy()
        command = command.at[-3:].set(world_to_navi_vel(navi2world_pose, info["command_delay"][-3:].reshape(-1, 3)).reshape(-1))
        command = command.at[-1].set(0)

        headbf = headbf * (info["headdf_delay"] < 0.5);  headdf = jp.clip(info["headdf_delay"], -1.0, 0.5)
        pelvbf = pelvbf * (info["pelvdf_delay"] < 0.5);  pelvdf = jp.clip(info["pelvdf_delay"], -1.0, 0.5)
        torsbf = torsbf * (info["torsdf_delay"] < 0.5);  torsdf = jp.clip(info["torsdf_delay"], -1.0, 0.5)
        feetbf = feetbf * (info["feetdf_delay"] < 0.5);  feetdf = jp.clip(info["feetdf_delay"], -1.0, 0.5)
        handsbf = handsbf * (info["handsdf_delay"] < 0.5); handsdf = jp.clip(info["handsdf_delay"], -1.0, 0.5)
        kneesbf = kneesbf * (info["kneesdf_delay"] < 0.5); kneesdf = jp.clip(info["kneesdf_delay"], -1.0, 0.5)
        shldsbf = shldsbf * (info["shldsdf_delay"] < 0.5); shldsdf = jp.clip(info["shldsdf_delay"], -1.0, 0.5)

        # Box PF in nav frame (using current, not delayed)
        boxgf_navi = world_to_navi_vel(navi2world_pose, info["boxgf"].reshape(-1, 3))
        boxbf_navi = world_to_navi_vel(navi2world_pose, info["boxbf"].reshape(-1, 3))
        boxdf_clip = jp.clip(info["boxdf"], -1.0, 0.5)
        boxbf_navi = boxbf_navi * (info["boxdf"] < 0.5)

        pf = jp.hstack([
            headgf.reshape(-1), headbf.reshape(-1), headdf.reshape(-1),
            pelvgf.reshape(-1), pelvbf.reshape(-1), pelvdf.reshape(-1),
            torsgf.reshape(-1), torsbf.reshape(-1), torsdf.reshape(-1),
            feetgf.reshape(-1), feetbf.reshape(-1), feetdf.reshape(-1),
            handsgf.reshape(-1), handsbf.reshape(-1), handsdf.reshape(-1),
            kneesgf.reshape(-1), kneesbf.reshape(-1), kneesdf.reshape(-1),
            shldsgf.reshape(-1), shldsbf.reshape(-1), shldsdf.reshape(-1),
        ])

        state = jp.hstack([
            noisy_gyro_pelvis, noisy_gvec_pelvis,
            (noisy_joint_angles - self._default_qpos)[self.obs_joint_ids],
            noisy_joint_vel[self.obs_joint_ids],
            info["last_act"],                               # 23-dim
            info["motor_targets"][self.action_joint_ids],   # 23-dim
            command, info["foot_height"], gait_phase,
            pf,
            # Box PF in nav frame (deployable: no absolute positions)
            boxgf_navi.reshape(-1), boxbf_navi.reshape(-1), boxdf_clip.reshape(-1),
        ])

        state = jp.nan_to_num(state)
        privileged_state = jp.nan_to_num(privileged_state)
        return {"state": state, "privileged_state": privileged_state}

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Inherits body termination from G1CatEnv; adds box-in-obstacle termination."""
        fall_termination = self.get_gravity(data, "pelvis")[2] < 0.0
        fall_termination |= info["head_pos"][2] < 0.7
        contact_termination = collision.geoms_colliding(data, self._right_foot_geom_id, self._left_foot_geom_id)
        contact_termination |= collision.geoms_colliding(data, self._left_foot_geom_id, self._right_shin_geom_id)
        contact_termination |= collision.geoms_colliding(data, self._right_foot_geom_id, self._left_shin_geom_id)
        contact_termination |= jp.any(info['headdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['pelvdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['torsdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['feetdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['handsdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['kneesdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['shldsdf'] < -self._config.term_collision_threshold)
        # Box inside obstacle
        contact_termination |= jp.any(info['boxdf'] < -self._config.term_collision_threshold)
        contact_termination &= (info["step"] >= 50)
        return fall_termination | contact_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            done: jax.Array,
            feet_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        """Inherits all G1CatEnv rewards; adds boxgf, boxdf, arm_pose, arm_smoothness."""
        move_flag = info["command"][0]
        cmd_vel = info["command"][1:].copy()

        reward_dict = {
            # Inherited behavior rewards
            "tracking_orientation": self._reward_orientation(
                info["navi_pelvis_rpy"], info["navi_torso_rpy"],
                info["head_pos"][..., 0] > 1.5,
            ),
            "tracking_root_field": self._reward_tracking_root_field(cmd_vel, info["global_lin_vel"]),
            "body_motion": self._cost_body_motion(info["global_lin_vel"], info["navi_torso_ang_vel"], cmd_vel),
            "body_rotation": self._reward_body_rotation(data, cmd_vel, info["navi2world_rot"]),
            "foot_contact": self._cost_foot_contact(data, feet_contact, info["gait_mask"], move_flag),
            "foot_clearance": self._cost_foot_clearance(data, info["foot_height"], info["gait_mask"], move_flag),
            "foot_slip": self._cost_foot_slip(data, info["gait_mask"]),
            "foot_balance": self._cost_foot_balance(data, info["navi2world_pose"], move_flag),
            "straight_knee": self._cost_straight_knee(data.qpos[jp.array(self._knee_indices) + 7]),
            "foot_far": self._cost_foot_far(data),
            # Energy
            "joint_limits": self._cost_joint_pos_limits(data.qpos[7:7 + NUM_ROBOT_JOINTS]),
            "joint_torque": self._cost_torque(data.actuator_force),
            "smoothness_joint": self._cost_smoothness_joint(data, info["last_joint_vel"]),
            "smoothness_action": self._cost_smoothness_action(action, info["last_act"], info["last_last_act"]),
            # Body HumanoidPF
            "headgf": self._re_gf0(info["headgf"], info["head_vel"], info["headdf"],
                                    (move_flag[None] < 0.5) | (info["head_pos"][..., 0] > 1.5), tau=0.5),
            "feetgf": self._re_gf0(info["feetgf"], info["feet_vel"], info["feetdf"],
                                    (move_flag[None] < 0.5) | (info["gait_mask"] == 1) | (info["feet_pos"][..., 0] > 1.5), tau=0.3),
            "handsgf": self._re_gf0(info["handsgf"], info["hands_vel"], info["handsdf"],
                                     (move_flag[None] < 0.5) | (info["hands_pos"][..., 0] > 1.5), tau=0.5),
            "headdf": self._re_sdf(info["headdf"]),
            "feetdf": self._re_sdf(info["feetdf"]),
            "handsdf": self._re_sdf(info["handsdf"]),
            "kneesdf": self._re_sdf(info["kneesdf"]),
            "shldsdf": self._re_sdf(info["shldsdf"]),
            # Box HumanoidPF (activate with --box flag)
            "boxgf": self._re_gf0(info["boxgf"], info["box_vel"].reshape(1, 3), info["boxdf"],
                                   (move_flag[None] < 0.5) | (info["box_pos"][..., 0] > 1.5), tau=0.5),
            "boxdf": self._re_sdf(info["boxdf"]),
            # Arm stability (keep arms in carrying pose)
            "arm_pose": self._cost_arm_pose(data.qpos[7:7 + NUM_ROBOT_JOINTS]),
            "arm_smoothness": self._cost_arm_smoothness(action),
        }
        for k, v in reward_dict.items():
            reward_dict[k] = jp.where(jp.isnan(v), 0.0, v)
        return reward_dict

    def _cost_arm_pose(self, robot_qpos: jax.Array) -> jax.Array:
        """Penalize arm deviation from the carrying pose. robot_qpos is 29-dim."""
        arm_angles = robot_qpos[self._arm_actuator_ids]
        err = jp.sum(jp.square(arm_angles - self._carry_pose_arm))
        return err

    def _cost_arm_smoothness(self, action: jax.Array) -> jax.Array:
        """Penalize large arm action deltas (indices 15:23 in the 23-dim action)."""
        arm_action = action[15:]  # 8 arm joints (last 8 of 23)
        return jp.sum(jp.square(arm_action))


@cat_ppo.registry.register("G1CaTra", "command_to_reference_fn")
def command_to_reference(env_config: config_dict.ConfigDict, command: jax.Array):
    command_vel = command[1:]
    base_height = env_config.reward_config.base_height_target
    base_gvec = np.array([0.0, 0.0, 1.0])
    base_lin_vel = np.array([command_vel[0], command_vel[1], 0.0])
    base_ang_vel = np.array([0.0, 0.0, command_vel[2]])
    return {
        "base_height": base_height,
        "base_gvec": base_gvec,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
    }
