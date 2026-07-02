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
"""Single-stage CaTra environment: box carry & traverse (500 steps).

The robot is always initialized from a warm-start state (already holding the box);
there is no pickup stage. Every step uses the PF-derived command and the navigation
+ carry-maintenance rewards.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import jaxlie
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

import functools

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

# Number of robot joints (excluding root freejoint and box/support freejoints)
NUM_ROBOT_JOINTS = 29

# qpos layout: [0:7] root, [7:36] robot joints, [36:43] box freejoint, [43:50] support freejoint
# qvel layout: [0:6] root, [6:35] robot joints,  [35:41] box vel,       [41:47] support vel
BOX_QPOS_START     = 7 + NUM_ROBOT_JOINTS   # 36
BOX_QVEL_START     = 6 + NUM_ROBOT_JOINTS   # 35
SUPPORT_QPOS_START = BOX_QPOS_START + 7     # 43
SUPPORT_QVEL_START = BOX_QVEL_START + 6     # 41


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
    # Sanitize the action-derived target: a NaN/Inf policy output would otherwise
    # produce NaN torques that wedge mjx.step's solver in a non-terminating loop
    # (GPU pins at 100% forever). nan_to_num maps NaN->0, +/-Inf->finite extremes,
    # which the per-substep torque clip below then bounds to the actuator limits.
    qpos_des = jp.nan_to_num(qpos_des)

    def single_step(carry, _):
        rng, data = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        pos_err = qpos_des - data.qpos[7:7 + NUM_ROBOT_JOINTS]
        vel_err = -data.qvel[6:6 + NUM_ROBOT_JOINTS]
        torque = (kp_scale * kps) * pos_err + (kd_scale * kds) * vel_err

        rfi_noise = rfi_lim_scale * jax.random.uniform(rng_rfi, shape=torque.shape, minval=-1.0, maxval=1.0)
        torque += rfi_noise
        # nan_to_num before clip: clip alone passes NaN through (NaN compares False),
        # so a NaN here (e.g. from already-diverged qpos/qvel) would still reach ctrl.
        torque = jp.nan_to_num(torque)
        torque = jp.clip(torque, -torque_limit, torque_limit)

        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)

        return (rng, data), None

    return jax.lax.scan(single_step, (rng, data), (), n_substeps)[0]


def _make_domain_randomize_catra():
    """Factory: loads model once to get box IDs, returns the DR function."""
    _mj = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    _box_geom_id = _mj.geom("box_geom").id
    _box_body_id = _mj.body("carried_box").id
    del _mj

    TORSO_BODY_ID = 16

    def domain_randomize_catra(model: mjx.Model, rng: jax.Array):
        """Robot DR (frictionloss/armature/CoM/mass/qpos0) + box size/mass DR + RFI enabled."""

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

            rng, key = jax.random.split(rng)
            box_half_x = jax.random.uniform(key, minval=0.15, maxval=0.15)
            rng, key = jax.random.split(rng)
            box_half_y = jax.random.uniform(key, minval=0.20, maxval=0.20)
            rng, key = jax.random.split(rng)
            box_half_z = jax.random.uniform(key, minval=0.15, maxval=0.15)
            geom_size = model.geom_size.at[_box_geom_id].set(
                jp.array([box_half_x, box_half_y, box_half_z])
            )

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

    return domain_randomize_catra


domain_randomize_catra = _make_domain_randomize_catra()


def _make_domain_randomize_catra_warmstart(ws_box_mass: "jax.Array", ws_box_size: "jax.Array",
                                           indices: "jax.Array" = None):
    """Factory: returns a DR function that sources box mass/size from saved state file.

    Each env i receives state index `indices[i]` (default: i, sequential), with box_mass
    and box_size from ws_box_mass[idx] / ws_box_size[idx].  The index is encoded in qpos0[0]
    so reset() can retrieve it without knowing the env axis position.

    `indices` lets an offline generator draw a different slice of the state pool per batch
    (e.g. (arange(num_envs) + offset) % N); when None it falls back to arange(num_envs).
    """
    _mj = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    _box_geom_id = _mj.geom("box_geom").id
    _box_body_id = _mj.body("carried_box").id
    del _mj

    TORSO_BODY_ID = 16

    def domain_randomize_catra_warmstart(model: mjx.Model, rng: jax.Array):
        """Same robot DR as the default, but box mass/size come from the state file."""

        @functools.partial(jax.vmap, in_axes=(0, 0))
        def rand_dynamics(rng, idx):
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

            # Box mass/size: taken from the saved state file indexed by env index.
            geom_size = model.geom_size.at[_box_geom_id].set(ws_box_size[idx])
            body_mass = body_mass.at[_box_body_id].set(ws_box_mass[idx])

            # Encode state index into qpos0[0] so reset() can retrieve it.
            # qpos0[0] (root x) is never used after reset overwrites qpos with the saved state.
            qpos0 = qpos0.at[0].set(idx.astype(jp.float32))

            return (pair_friction, dof_frictionloss, dof_armature, body_ipos, body_mass, qpos0, geom_size)

        num_envs = rng.shape[0]
        env_indices = jp.arange(num_envs) if indices is None else indices
        (pair_friction, frictionloss, armature, body_ipos, body_mass, qpos0, geom_size) = rand_dynamics(rng, env_indices)

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

    return domain_randomize_catra_warmstart


def make_warmstart_domain_randomize_catra(states_path: str):
    """Load states_path and return a DR function that carries box mass/size per env."""
    npz = np.load(states_path)
    ws_box_mass = jp.array(npz["box_mass"])   # (N,)
    ws_box_size = jp.array(npz["box_size"])   # (N, 3)
    return _make_domain_randomize_catra_warmstart(ws_box_mass, ws_box_size)


def make_warmstart_domain_randomize_catra_indexed(states_path: str, indices: "jax.Array"):
    """Like make_warmstart_domain_randomize_catra but loads warm-start states by an explicit
    per-env `indices` array instead of arange(num_envs). Used by offline state generators that
    roll out several batches and want each batch to draw a different slice of the state pool."""
    npz = np.load(states_path)
    ws_box_mass = jp.array(npz["box_mass"])   # (N,)
    ws_box_size = jp.array(npz["box_size"])   # (N, 3)
    return _make_domain_randomize_catra_warmstart(ws_box_mass, ws_box_size, indices=indices)


def _make_warmstart_only_catra(ws_box_mass: "jax.Array", ws_box_size: "jax.Array"):
    """Factory: warm-start without robot DR.

    Same as _make_domain_randomize_catra_warmstart but drops the robot-param randomization
    (frictionloss, armature, body CoM/mass, qpos0 joint-pose noise). Keeps the two pieces
    that warm-start needs: per-env box mass/size from the saved file, and the qpos0[0]
    index encoding so reset() can decode which warm-start state to load.

    Use this for teacher policies (G1CaTraPri) where we want clean nominal dynamics for
    distillation but still want to skip the pickup stage via warm-start.
    """
    _mj = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    _box_geom_id = _mj.geom("box_geom").id
    _box_body_id = _mj.body("carried_box").id
    del _mj

    def warmstart_only_catra(model: mjx.Model, rng: jax.Array):
        @functools.partial(jax.vmap, in_axes=(0,))
        def assign(idx):
            geom_size = model.geom_size.at[_box_geom_id].set(ws_box_size[idx])
            body_mass = model.body_mass.at[_box_body_id].set(ws_box_mass[idx])
            qpos0 = model.qpos0.at[0].set(idx.astype(jp.float32))
            return geom_size, body_mass, qpos0

        num_envs = rng.shape[0]
        indices = jp.arange(num_envs)
        geom_size, body_mass, qpos0 = assign(indices)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({
            "geom_size": 0,
            "body_mass": 0,
            "qpos0": 0,
        })

        model = model.tree_replace({
            "geom_size": geom_size,
            "body_mass": body_mass,
            "qpos0": qpos0,
        })

        return model, in_axes

    return warmstart_only_catra


def make_warmstart_only_catra(states_path: str):
    """Load states_path and return a DR-free warm-start fn (box mass/size + index dispatch only)."""
    npz = np.load(states_path)
    ws_box_mass = jp.array(npz["box_mass"])   # (N,)
    ws_box_size = jp.array(npz["box_size"])   # (N, 3)
    return _make_warmstart_only_catra(ws_box_mass, ws_box_size)


def g1_catra_task_config() -> config_dict.ConfigDict:
    """Config for single-stage G1CaTra (box carry & traverse).

    Episode: 500 steps (10 s @ 50 Hz). The robot is warm-started already holding
    the box; every step uses the PF-derived command and the navigation + carry
    rewards. There is no pickup stage.

    Observation dimensions:
      num_obs = 239  (state, deployable; PF subblock delayed + nav-frame, not additively noised)
      num_pri = 333  (privileged_state, noiseless + extras)
    """
    env_config = config_dict.create(
        task_type="flat_terrain_catra",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=500,
        action_repeat=1,
        action_scale=0.5,
        history_len=15,
        num_obs=239,    # TEMP 20-DOF (no waist) + box_mass(1), no stage_flag; 23-DOF (waist) would be 251 (+3 joints x 4 obs fields)
        num_pri=333,    # TEMP 20-DOF + box_mass(1), no stage_flag; 23-DOF would be 345
        num_act=20,     # TEMP: 12 legs + 8 arms (3 waist removed)
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
                box_pos=0.05,                 # +/- 5 cm per xyz axis (box tracking error)
                box_ori=float(np.deg2rad(5.0)),  # +/- 5 deg random axis-angle perturbation
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # --- Always active ---
                joint_torque=-1e-4,
                smoothness_joint=-1e-6,
                joint_limits=-1.0,
                hip_yaw_lim=-2.0,    # penalize hip-yaw joints outside [-0.5, 0.5] rad
                # --- Navigation rewards (CAT scales) ---
                tracking_orientation=2.0,
                tracking_root_field=1.0,
                body_motion=-0.5,
                body_rotation=3.0,
                foot_contact_trav=-1.0,
                foot_clearance=-15.0,
                foot_slip_trav=-0.5,
                foot_balance_trav=-30.0,
                foot_far=0.0,
                feet_apart=-2.0,    # penalize feet farther than 0.5 m apart
                straight_knee_trav=-30.0,
                feet_rotation=1.0,           # reward clean knee+ankle alignment with nav frame; indirect anti-crouch (ported from G1CatPri)
                smoothness_action=-1e-3,
                forward_progress=5.0,   #5.0,
                upper_body_align=-0.0, #-2.0,
                headgf=0.0,    # overwritten by --overhead in train_ppo.py
                handsgf=0.0,   # overwritten by --lateralgf
                feetgf=0.0,    # overwritten by --groundgf
                headdf=0.0,    # overwritten by --overhead
                handsdf=0.0,   # overwritten by --lateraldf
                feetdf=0.0,    # overwritten by --grounddf
                kneesdf=0.0,   # overwritten by --lateraldf
                shldsdf=0.0,   # overwritten by --lateraldf
                boxdf=0.0,     # box-corner SDF collision penalty (set non-zero to enable)
                boxgf=0.0,     # box-corner inflation-GF alignment (set via --boxgf)
                # --- Carry maintenance ---
                reach_carry=3.0,
                lift_carry=0.0, #2.0,
                hand_contact_carry=2.0,
                grasp_symmetry_carry=-2.0,
                palm_orient_carry=2.0,
                hands_level_carry=-1.0,
                box_upright_carry=2.0,
            ),
            base_height_target=0.75,
            foot_height_stance=0.0,
        ),
        term_collision_threshold=0.04,
        box_drop_threshold=0.3,
        box_use_inflation=True,   # boxgf reward: True -> sample gf_inflation.npy (anticipatory), False -> regular gf.npy

        box_surface_height_range=[0.3, 0.3],   # fixed: support body centre at 0.3 m
        # Warm-start: when set, reset() loads robot+box state from a pre-generated file
        warmstart_states_path=None,
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
    )

    policy_config = config_dict.create(
        num_timesteps=5_000_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        madrona_backend=False,
        augment_pixels=False,
        num_envs=32768,
        episode_length=500,
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

    return config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
        eval_config=eval_config,
    )


cat_ppo.registry.register("G1CaTra", "config")(g1_catra_task_config())


@cat_ppo.registry.register("G1CaTra", "train_env_class")
class G1CaTraEnv(G1CatEnv):
    """Single-stage G1 humanoid: box carry & traverse.

    The robot is warm-started already holding the box and walks through the
    obstacle scene for the full 500-step episode, using CAT navigation rewards
    + carry-maintenance terms. There is no pickup stage.

    Action space: 20 DOF (12 leg + 8 arm joints; TEMP: all 3 waist joints removed).
    State:       239-dim (deployable, noisy).
    Priv state:  333-dim (noiseless + extras, critic only).
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
        self._post_init_catra()

    def _post_init_catra(self) -> None:
        """Set up 20-DOF action space, soft limits, init_q, and cached IDs for pickup rewards."""
        self.action_joint_names = consts.CATRA_ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = jp.array([
            self.mj_model.actuator(name).id for name in self.action_joint_names
        ])

        lowers, uppers = self.mj_model.jnt_range[1:1 + NUM_ROBOT_JOINTS].T
        c = (lowers + uppers) / 2
        r = uppers - lowers
        factor = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * factor
        self._soft_uppers = c + 0.5 * r * factor

        self._default_qpos = jp.array(consts.DEFAULT_QPOS_CATRA[7:7 + NUM_ROBOT_JOINTS])

        box_default     = np.array([0.35, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        support_default = np.array([0.35, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._init_q = jp.array(np.concatenate([consts.DEFAULT_QPOS_CATRA, box_default, support_default]))

        # Guidance field used by the carried box's boxgf reward. With box_use_inflation, use the
        # anticipatory inflated-obstacle field (gf_inflation.npy, fails fast if absent); otherwise
        # reuse the regular gf. Same grid/shape as self.gf (loaded in G1CatEnv.__init__).
        if self._config.box_use_inflation:
            pf_path = self._config.pf_config.path
            self.gf_box = jp.array(np.load(f"{pf_path}/gf_inflation.npy"))  # (Nx,Ny,Nz,3)
        else:
            self.gf_box = self.gf

        # IDs needed for Pickup reward computation
        self._pelvis_body_id    = self._mj_model.body("pelvis").id
        self._box_body_id       = self._mj_model.body("carried_box").id
        self._box_geom_id       = self._mj_model.geom(consts.BOX_GEOM).id
        self._box_support_geom_id = self._mj_model.geom("box_support_col").id
        self._hand_geom_ids     = np.array([
            self._mj_model.geom("left_hand_collision").id,
            self._mj_model.geom("right_hand_collision").id,
        ])
        self._thigh_geom_ids    = np.array([
            self._mj_model.geom("left_thigh").id,
            self._mj_model.geom("right_thigh").id,
        ])
        self._head_geom_id      = self._mj_model.geom("head_collision").id

        # Hip-yaw joint qpos indices (offset to drop into qpos[7:] via +7), for hip_yaw_lim reward.
        self._hip_yaw_indices = jp.array([
            self._mj_model.joint(f"{side}_hip_yaw_joint").qposadr - 7
            for side in ["left", "right"]
        ])

        # Warm-start is mandatory: single-stage CaTra always initializes from a
        # pre-generated "robot already holding box" state (there is no pickup stage).
        ws_path = getattr(self._config, "warmstart_states_path", None)
        if not ws_path:
            raise ValueError(
                "G1CaTra is single-stage (box transport only) and requires warm-start "
                "initialization: set env_config.warmstart_states_path (e.g. via "
                "train_ppo.py --warmstart_states_path <states.npz>)."
            )
        npz = np.load(ws_path)
        self._ws_qpos = jp.array(npz["qpos"])   # (N, nq)
        self._ws_qvel = jp.array(npz["qvel"])   # (N, nv)
        N = int(self._ws_qpos.shape[0])
        print(f"[G1CaTraEnv] Loaded {N} warm-start states from {ws_path}")

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)  # 20

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset: loads a pre-generated warm-start state (robot already holding box)."""
        # --- Warm-start (only) path: load saved (qpos, qvel) exactly as recorded ---
        # State index for this env is encoded in qpos0[0] by the warm-start DR fn.
        state_idx = jp.round(self.mjx_model.qpos0[0]).astype(jp.int32)
        qpos = self._ws_qpos[state_idx]      # (nq,) full state: root + joints + box + support
        qvel = self._ws_qvel[state_idx]      # (nv,)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:7 + NUM_ROBOT_JOINTS])
        data = mjx.forward(self.mjx_model, data)

        last_act_init = jp.zeros(self.action_size)
        motor_targets_init = qpos[7:7 + NUM_ROBOT_JOINTS]   # PD targets match saved pose
        last_jv_init = qvel[6:6 + NUM_ROBOT_JOINTS]

        # Ancillary scalars needed by info dict below; derive from saved state / model.
        surface_z = self._config.box_surface_height_range[0]
        support_half_z = self.mjx_model.geom_size[self._box_support_geom_id][2]
        box_xy = qpos[BOX_QPOS_START:BOX_QPOS_START + 2]
        box_quat_ws = qpos[BOX_QPOS_START + 3:BOX_QPOS_START + 7]
        box_yaw = jp.arctan2(2 * (box_quat_ws[0] * box_quat_ws[3] + box_quat_ws[1] * box_quat_ws[2]),
                              1 - 2 * (box_quat_ws[2] ** 2 + box_quat_ws[3] ** 2))

        # --- DR scalars (kp/kd/rfi, sampled fresh each reset) ---
        rng, key_kp, key_kd, key_rfi = jax.random.split(rng, 4)
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

        # Box info from DR'd model (mass/size already set correctly by DR)
        box_size = self.mjx_model.geom_size[self._box_geom_id]
        box_mass = self.mjx_model.body_mass[self._box_body_id]

        # --- Sample HumanoidPF fields (from post-rollout data) ---
        head_pos = data.site_xpos[self._head_site_id]
        head_vel = jp.zeros_like(head_pos)
        pelv_pos = data.site_xpos[self._pelvis_imu_site_id]
        tors_pos = data.site_xpos[self._torso_imu_site_id]
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = jp.zeros_like(feet_pos)
        hands_pos = data.site_xpos[self._hands_site_id]
        hands_vel = jp.zeros_like(hands_pos)
        knees_pos = data.site_xpos[self._knees_site_id]
        shlds_pos = data.site_xpos[self._shlds_site_id]

        all_poses = jp.concatenate([
            head_pos.reshape(1, -1), pelv_pos.reshape(1, -1), tors_pos.reshape(1, -1),
            feet_pos, hands_pos, knees_pos, shlds_pos,
        ], axis=0)
        all_gf = self.sample_field(self.gf, all_poses)
        if self._config.box_use_inflation:
            # Hands are the box's actuator, so they follow the same (inflation) field as the box (rows 5:7 = hands).
            all_gf = all_gf.at[5:7].set(self.sample_field(self.gf_box, hands_pos))
        all_bf = self.sample_field(self.bf, all_poses)
        all_df = self.sample_field(self.sdf, all_poses)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = jp.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = jp.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = jp.split(all_df, [1, 2, 3, 5, 7, 9], axis=0)

        # Box corner HumanoidPF: 8 corners x {gf:3, bf:3, sdf:1}
        # boxgf is sampled from self.gf_box (inflation field if box_use_inflation, else regular gf):
        # the box observes and is rewarded against the same field.
        box_corners = self._box_corners_world(data, box_size)
        boxgf = self.sample_field(self.gf_box, box_corners)
        boxbf = self.sample_field(self.bf, box_corners)
        boxdf = self.sample_field(self.sdf, box_corners)
        box_corners_vel = jp.zeros_like(box_corners)  # per-corner velocity for the boxgf reward

        # Noisy box tracking estimate (single shared perturbation) for the deployable obs.
        rng, box_pos_noisy, box_quat_noisy = self._noisy_box_pose(rng, data)
        box_corners_noisy = self._box_corners_from_pose(box_pos_noisy, box_quat_noisy, box_size)
        boxgf_delay_init = self.sample_field(self.gf_box, box_corners_noisy)
        boxbf_delay_init = self.sample_field(self.bf, box_corners_noisy)
        boxdf_delay_init = self.sample_field(self.sdf, box_corners_noisy)

        # Command is PF-derived (computed each step); zero at reset before the first step.
        command = jp.zeros(4)

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

        box_pos_init = data.xpos[self._box_body_id]

        info = {
            "rng": rng,
            "step": 0,
            "command": command,
            "last_command": jp.zeros(4),
            "last_act": last_act_init,
            "last_last_act": jp.zeros(self.action_size),
            "last_feet_vel": jp.zeros(2),
            "last_joint_vel": last_jv_init,
            # push
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # state
            "motor_targets": motor_targets_init,
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
            # Gait
            "stop_timestep": 100,
            "phase": phase,
            "phase_dt": phase_dt,
            "gait_mask": jp.zeros(2),
            "gait_freq": gait_freq,
            "foot_height": foot_height,
            # DR
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "rfi_lim_scale": rfi_lim_scale,
            # Body HumanoidPF (current, world frame, normalized)
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
            # Delay buffers (initialized to post-rollout values)
            "command_delay": command, "odom_delay": data.qpos[:7],
            "headgf_delay": headgf.copy(), "headbf_delay": headbf.copy(), "headdf_delay": headdf.copy(),
            "pelvgf_delay": pelvgf.copy(), "pelvbf_delay": pelvbf.copy(), "pelvdf_delay": pelvdf.copy(),
            "torsgf_delay": torsgf.copy(), "torsbf_delay": torsbf.copy(), "torsdf_delay": torsdf.copy(),
            "feetgf_delay": feetgf.copy(), "feetbf_delay": feetbf.copy(), "feetdf_delay": feetdf.copy(),
            "handsgf_delay": handsgf.copy(), "handsbf_delay": handsbf.copy(), "handsdf_delay": handsdf.copy(),
            "kneesgf_delay": kneesgf.copy(), "kneesbf_delay": kneesbf.copy(), "kneesdf_delay": kneesdf.copy(),
            "shldsgf_delay": shldsgf.copy(), "shldsbf_delay": shldsbf.copy(), "shldsdf_delay": shldsdf.copy(),
            # Box corner HumanoidPF (8 corners): current world frame and delayed
            "boxgf": boxgf.copy(), "boxbf": boxbf.copy(), "boxdf": boxdf.copy(),
            "boxgf_delay": boxgf_delay_init.copy(), "boxbf_delay": boxbf_delay_init.copy(), "boxdf_delay": boxdf_delay_init.copy(),
            # Box corner kinematics (boxgf reward)
            "box_corners": box_corners.copy(), "box_corners_vel": box_corners_vel.copy(),
            # Noisy box tracking estimate (shared by box_pos_local/quat_local + box PF in the deployable obs)
            "box_pos_noisy": box_pos_noisy, "box_quat_noisy": box_quat_noisy,
            # Box state
            "box_pos": box_pos_init.copy(),
            # Box metadata (for reward computation)
            "surface_z": jp.array(surface_z),
            "support_half_z": support_half_z,
            "box_size": box_size,
            "box_mass": box_mass,
            "box_xy_init": box_xy,
            "box_yaw_init": box_yaw,
        }
        info.update(self._extra_reward_info())

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, info, feet_contact)
        reward = self._initial_reward()
        done = jp.zeros(())
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step with the transport action; PF-derived command and pushes active every step."""
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
        qvel = state.data.qvel.at[:2].set(state.data.qvel[:2] + push * push_magnitude)

        state = state.replace(data=state.data.replace(qvel=qvel))

        # Motor targets
        lower_motor_targets = jp.clip(
            state.info["motor_targets"][self.action_joint_ids]
            + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)

        # Physics step
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

        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        state.info["motor_targets"] = motor_targets
        state.info["local_lin_vel"] = self.get_local_linvel(data, "pelvis")
        state.info["global_lin_vel"] = self.get_global_linvel(data, "pelvis")
        state.info["global_ang_vel"] = self.get_global_angvel(data, "pelvis")

        # Navi frame
        pelvis2world_rot = data.site_xmat[self._pelvis_imu_site_id]
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        state.info["navi2world_pose"] = state.info["navi2world_pose"].at[:3, :3].set(navi2world_rot)
        state.info["navi2world_pose"] = state.info["navi2world_pose"].at[:2, 3].set(
            data.site_xpos[self._pelvis_imu_site_id][:2]
        )
        state.info["navi2world_pose"] = state.info["navi2world_pose"].at[2, 3].set(
            self._config.reward_config.base_height_target
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
        if self._config.box_use_inflation:
            # Hands are the box's actuator, so they follow the same (inflation) field as the box (rows 5:7 = hands).
            all_gf = all_gf.at[5:7].set(self.sample_field(self.gf_box, hands_pos))
        all_bf = self.sample_field(self.bf, all_poses)
        all_df = self.sample_field(self.sdf, all_poses)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = jp.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = jp.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = jp.split(all_df, [1, 2, 3, 5, 7, 9], axis=0)

        box_corners = self._box_corners_world(data, state.info["box_size"])
        boxgf = self.sample_field(self.gf_box, box_corners)  # inflation field if box_use_inflation, else regular gf
        boxbf = self.sample_field(self.bf, box_corners)
        boxdf = self.sample_field(self.sdf, box_corners)
        box_corners_vel = (box_corners - state.info["box_corners"]) / self.dt  # per-corner velocity for the boxgf reward

        # PF-derived command (active every step)
        cmd_pf = self.compute_cmd_from_rtf(
            pelvgf.reshape(-1),
            jp.concat([headgf, feetgf], axis=0),
            jp.concat([headbf, feetbf], axis=0),
        )
        state.info["command"] = cmd_pf

        # Delay buffer update
        update_pf = (state.info["step"] % 5) == 0
        state.info["rng"], odo_key = jax.random.split(state.info["rng"], 2)
        odom_delay = jp.where(update_pf, data.qpos[:7], state.info["odom_delay"])
        p_gt = data.qpos[:3]; q_gt = data.qpos[3:7]
        p_odom = odom_delay[:3]; q_odom = odom_delay[3:7]
        all_poses_delay = delay_body_pos(p_gt, q_gt, p_odom, q_odom, all_poses)
        all_gf_delay = self.sample_field(self.gf, all_poses_delay)
        if self._config.box_use_inflation:
            # Hands follow the box's (inflation) field (rows 5:7 = hands).
            all_gf_delay = all_gf_delay.at[5:7].set(self.sample_field(self.gf_box, all_poses_delay[5:7]))
        all_bf_delay = self.sample_field(self.bf, all_poses_delay)
        all_df_delay = self.sample_field(self.sdf, all_poses_delay)

        # Deployable box PF is sampled at the NOISY box estimate (imperfect box tracking).
        # The same estimate feeds box_pos_local/box_quat_local in _get_obs (via info).
        state.info["rng"], box_pos_noisy, box_quat_noisy = self._noisy_box_pose(state.info["rng"], data)
        state.info["box_pos_noisy"] = box_pos_noisy
        state.info["box_quat_noisy"] = box_quat_noisy
        box_corners_noisy = self._box_corners_from_pose(box_pos_noisy, box_quat_noisy, state.info["box_size"])
        box_corners_delay = delay_body_pos(p_gt, q_gt, p_odom, q_odom, box_corners_noisy)
        boxgf_delay = self.sample_field(self.gf_box, box_corners_delay)
        boxbf_delay = self.sample_field(self.bf, box_corners_delay)
        boxdf_delay = self.sample_field(self.sdf, box_corners_delay)

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

        boxgf       = boxgf       * (move_flag[None] > 0.5) / (jp.linalg.norm(boxgf,       axis=-1, keepdims=True) + EPS)
        boxbf       = boxbf       / (jp.linalg.norm(boxbf,       axis=-1, keepdims=True) + EPS)
        boxgf_delay = boxgf_delay * (move_flag[None] > 0.5) / (jp.linalg.norm(boxgf_delay, axis=-1, keepdims=True) + EPS)
        boxbf_delay = boxbf_delay / (jp.linalg.norm(boxbf_delay, axis=-1, keepdims=True) + EPS)
        command_delay = self.compute_cmd_from_rtf(
            pelvgf_delay.reshape(-1),
            jp.concat([headgf_delay, feetgf_delay], axis=0),
            jp.concat([headbf_delay, feetbf_delay], axis=0),
        )

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
        state.info["boxgf"] = boxgf.copy(); state.info["boxbf"] = boxbf.copy(); state.info["boxdf"] = boxdf.copy()
        state.info["boxgf_delay"] = boxgf_delay.copy(); state.info["boxbf_delay"] = boxbf_delay.copy(); state.info["boxdf_delay"] = boxdf_delay.copy()
        state.info["box_corners"] = box_corners.copy(); state.info["box_corners_vel"] = box_corners_vel.copy()
        state.info["head_pos"] = head_pos.copy(); state.info["head_vel"] = head_vel.copy()
        state.info["pelv_pos"] = pelv_pos.copy(); state.info["tors_pos"] = tors_pos.copy()
        state.info["feet_pos"] = feet_pos.copy(); state.info["feet_vel"] = feet_vel.copy()
        state.info["hands_pos"] = hands_pos.copy(); state.info["hands_vel"] = hands_vel.copy()
        state.info["knees_pos"] = knees_pos.copy(); state.info["shlds_pos"] = shlds_pos.copy()
        state.info["push"] = push; state.info["push_step"] += 1; state.info["step"] += 1
        state.info["box_pos"] = data.xpos[self._box_body_id].copy()
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        state.info["last_joint_vel"] = data.qvel[6:6 + NUM_ROBOT_JOINTS].copy()

        obs = self._get_obs(data, state.info, feet_contact)
        done = self._get_termination(data, state.info)

        rewards = self._get_reward(data, action, state.info, done, feet_contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}

        reward = self._assemble_reward(rewards)
        self._record_agent_rewards(state.info, rewards)

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

    def _initial_reward(self) -> jax.Array:
        """Scalar reward placed in State at reset."""
        return jp.zeros(())

    def _assemble_reward(self, rewards: dict[str, jax.Array]) -> jax.Array:
        """Collapse the scaled per-term reward dict into the scalar env reward.
        The reward is clipped to [0, 1e4] exactly as in CAT. The two-agent env
        overrides this to return the (lower + upper) sum so the brax wrappers still see a
        scalar; the per-agent split is carried in info via `_record_agent_rewards`."""
        raw_reward = sum(rewards.values()) * self.dt
        return jp.clip(raw_reward, 0.0, 10000.0)

    def _extra_reward_info(self) -> dict[str, jax.Array]:
        """Extra info keys (e.g. per-agent rewards) added at reset. Empty for the
        single-agent env; the two-agent env adds 'reward_lower' / 'reward_upper'."""
        return {}

    def _record_agent_rewards(self, info: dict, rewards: dict[str, jax.Array]) -> None:
        """Hook to stash per-agent rewards into info during step. No-op for single agent."""
        pass

    def _box_corners_from_pose(self, box_pos: jax.Array, box_quat: jax.Array, box_size: jax.Array) -> jax.Array:
        # Returns (8, 3) world positions of the box's corners for an explicit pose.
        corner_signs = jp.array([
            [-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.], [-1.,  1.,  1.],
            [ 1., -1., -1.], [ 1., -1.,  1.], [ 1.,  1., -1.], [ 1.,  1.,  1.],
        ], dtype=jp.float32)
        R = math.quat_to_mat(box_quat)
        local_corners = corner_signs * box_size
        return box_pos + local_corners @ R.T

    def _box_corners_world(self, data: mjx.Data, box_size: jax.Array) -> jax.Array:
        # Ground-truth box corners (used by reward + privileged critic PF).
        return self._box_corners_from_pose(
            data.xpos[self._box_body_id], data.xquat[self._box_body_id], box_size)

    def _noisy_box_pose(self, rng: jax.Array, data: mjx.Data) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Perturb the tracked box pose to mimic imperfect box tracking at deployment.

        A single estimate per call drives BOTH box_pos_local/box_quat_local and the box-corner
        PF sampling, so the deployable policy sees one coherent noisy box estimate. Position:
        uniform +/- box_pos per xyz axis. Orientation: random-axis, angle uniform in +/- box_ori.
        Returns (rng, box_pos_noisy, box_quat_noisy).
        """
        box_pos_world = data.xpos[self._box_body_id]
        box_quat_world = data.xquat[self._box_body_id]
        lvl = self._config.noise_config.level
        rng, pos_rng, axis_rng, angle_rng = jax.random.split(rng, 4)
        box_pos_noisy = box_pos_world + (2 * jax.random.uniform(pos_rng, shape=box_pos_world.shape) - 1) \
            * lvl * self._config.noise_config.scales.box_pos
        rand_axis = jax.random.normal(axis_rng, shape=(3,))
        rand_axis = rand_axis / (jp.linalg.norm(rand_axis) + 1e-6)
        rand_angle = (2 * jax.random.uniform(angle_rng) - 1) * lvl * self._config.noise_config.scales.box_ori
        noise_quat = math.axis_angle_to_quat(rand_axis, rand_angle)
        box_quat_noisy = math.quat_mul(box_quat_world, noise_quat)
        return rng, box_pos_noisy, box_quat_noisy

    def _re_boxgf(self, gf_vel: jax.Array, lin_vel: jax.Array, sdf: jax.Array,
                  crossed: jax.Array, cmd_vel: jax.Array, tau: float = 1.5) -> jax.Array:
        """boxgf reward: identical to _re_gf0 (proximity-gated cosine alignment over the box
        corners against the inflated guidance field), but the component along the commanded
        travel direction is first removed from BOTH the guidance vector and the corner
        velocity. This way the box is rewarded only for obstacle-driven lateral/vertical
        motion and cannot pull the robot's locomotion forward.

        gf_vel:  (8, 3) — inflated-obstacle guidance field at the 8 box corners
        lin_vel: (8, 3) — world-frame velocity of each box corner
        sdf:     (8, 1) — signed distance from each corner to the nearest obstacle
        crossed: (8,) bool — fallback mask (robot stopped or corner past obstacle zone)
        cmd_vel: (3,) — [vx, vy, yaw], the same command vector forward_progress uses
        tau:     float — proximity activation radius (m)
        """
        eps = 1e-6
        cmd_dir = cmd_vel[:2] / (jp.linalg.norm(cmd_vel[:2]) + eps)  # (2,) commanded planar direction
        # Remove the along-command component in the XY plane only (Z motion preserved)
        gf_perp = gf_vel.at[:, :2].add(-jp.sum(gf_vel[:, :2] * cmd_dir, axis=-1, keepdims=True) * cmd_dir)
        v_perp  = lin_vel.at[:, :2].add(-jp.sum(lin_vel[:, :2] * cmd_dir, axis=-1, keepdims=True) * cmd_dir)
        return self._re_gf0(gf_perp, v_perp, sdf, crossed, tau=tau)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        """251-dim state (deployable; PF subblock delayed + nav-frame, not additively noised) and 345-dim privileged_state.

        State (251):
            noisy_gyro(3), noisy_gvec(3),
            noisy_joint_angles[action_ids](23), noisy_joint_vel[action_ids](23),
            last_act(23), motor_targets[action_ids](23),
            command(4), foot_height(1), gait_phase(4),
            body_pf_delayed_nav(77) + box_pf_delayed_nav(56) = 133,
            box_pos_local(3), box_quat_local(4), box_size(3), box_mass(1)

        Privileged (345 = 244 noiseless-state-block + 101 extras):
            [same fields, noiseless, world-frame PF (body 77 + box 56 = 133), no box_pos_local/quat_local]
            box_size(3), box_mass(1),
            + linvel_pelvis(3), body_positions(33), body_velocities(15),
              box_pos_world(3), box_quat_world(4), box_linvel(3), box_angvel(3),
              navi_torso_rpy[:2](2)+gait_mask(2)+feet_contact(2),
              rfi_lim_scale(29), kp_scale(1), kd_scale(1)
        """
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel(data, "pelvis")
        joint_angles = data.qpos[7:7 + NUM_ROBOT_JOINTS]
        joint_vel = data.qvel[6:6 + NUM_ROBOT_JOINTS]
        gait_phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])

        # Box pose in pelvis frame. The deployable state uses the NOISY box estimate
        # (imperfect box tracking) shared with the box-corner PF; the privileged state below
        # uses the ground-truth world pose.
        pelvis_pos = data.xpos[self._pelvis_body_id]
        pelvis_rot = data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        pelvis_xquat = data.xquat[self._pelvis_body_id]
        pelvis_xquat_conj = pelvis_xquat * jp.array([1., -1., -1., -1.])
        box_pos_world = data.xpos[self._box_body_id]
        box_quat_world = data.xquat[self._box_body_id]
        noisy_box_pos_local = pelvis_rot.T @ (info["box_pos_noisy"] - pelvis_pos)
        noisy_box_quat_local = math.quat_mul(pelvis_xquat_conj, info["box_quat_noisy"])

        # Box world-frame velocity (for privileged)
        box_linvel_world = data.qvel[BOX_QVEL_START:BOX_QVEL_START + 3]
        box_angvel_world = data.qvel[BOX_QVEL_START + 3:BOX_QVEL_START + 6]

        navi2world_pose = info["navi2world_pose"]

        # --- Build noiseless PF block (77 body + 56 box = 133 dims, world frame, non-delayed) ---
        pf_noiseless = jp.hstack([
            info["headgf"].reshape(-1), info["headbf"].reshape(-1), info["headdf"].reshape(-1),
            info["pelvgf"].reshape(-1), info["pelvbf"].reshape(-1), info["pelvdf"].reshape(-1),
            info["torsgf"].reshape(-1), info["torsbf"].reshape(-1), info["torsdf"].reshape(-1),
            info["feetgf"].reshape(-1), info["feetbf"].reshape(-1), info["feetdf"].reshape(-1),
            info["handsgf"].reshape(-1), info["handsbf"].reshape(-1), info["handsdf"].reshape(-1),
            info["kneesgf"].reshape(-1), info["kneesbf"].reshape(-1), info["kneesdf"].reshape(-1),
            info["shldsgf"].reshape(-1), info["shldsbf"].reshape(-1), info["shldsdf"].reshape(-1),
            info["boxgf"].reshape(-1), info["boxbf"].reshape(-1), info["boxdf"].reshape(-1),
        ])

        # --- Privileged state (289 dims) ---
        privileged_state = jp.hstack([
            # Noiseless state block (188 dims, without box_pos_local/quat_local)
            gyro_pelvis, gvec_pelvis,
            (joint_angles - self._default_qpos)[self.action_joint_ids],
            joint_vel[self.action_joint_ids],
            info["last_act"],
            info["motor_targets"][self.action_joint_ids],
            info["command"], info["foot_height"], gait_phase,
            pf_noiseless,
            info["box_size"],
            info["box_mass"].reshape(1),
            # Privileged extras (101 dims)
            linvel_pelvis,
            info["pelv_pos"].reshape(-1), info["tors_pos"].reshape(-1), info["head_pos"].reshape(-1),
            info["shlds_pos"].reshape(-1), info["hands_pos"].reshape(-1),
            info["knees_pos"].reshape(-1), info["feet_pos"].reshape(-1),
            info["head_vel"].reshape(-1), info["hands_vel"].reshape(-1), info["feet_vel"].reshape(-1),
            box_pos_world, box_quat_world, box_linvel_world, box_angvel_world,
            info["navi_torso_rpy"][:2], info["gait_mask"], feet_contact,
            info["rfi_lim_scale"],
            info["kp_scale"].reshape(1), info["kd_scale"].reshape(1),
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

        # Body PF: delayed + nav-frame transform
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

        boxgf = world_to_navi_vel(navi2world_pose, info["boxgf_delay"].reshape(-1, 3))
        boxbf = world_to_navi_vel(navi2world_pose, info["boxbf_delay"].reshape(-1, 3))
        boxbf = boxbf * (info["boxdf_delay"] < 0.5)
        boxdf = jp.clip(info["boxdf_delay"], -1.0, 0.5)

        pf = jp.hstack([
            headgf.reshape(-1), headbf.reshape(-1), headdf.reshape(-1),
            pelvgf.reshape(-1), pelvbf.reshape(-1), pelvdf.reshape(-1),
            torsgf.reshape(-1), torsbf.reshape(-1), torsdf.reshape(-1),
            feetgf.reshape(-1), feetbf.reshape(-1), feetdf.reshape(-1),
            handsgf.reshape(-1), handsbf.reshape(-1), handsdf.reshape(-1),
            kneesgf.reshape(-1), kneesbf.reshape(-1), kneesdf.reshape(-1),
            shldsgf.reshape(-1), shldsbf.reshape(-1), shldsdf.reshape(-1),
            boxgf.reshape(-1), boxbf.reshape(-1), boxdf.reshape(-1),
        ])

        state = jp.hstack([
            noisy_gyro_pelvis, noisy_gvec_pelvis,
            (noisy_joint_angles - self._default_qpos)[self.action_joint_ids],
            noisy_joint_vel[self.action_joint_ids],
            info["last_act"],
            info["motor_targets"][self.action_joint_ids],
            command, info["foot_height"], gait_phase,
            pf,
            noisy_box_pos_local, noisy_box_quat_local, info["box_size"],
            info["box_mass"].reshape(1),
        ])

        return {
            "state": jp.nan_to_num(state),
            "privileged_state": jp.nan_to_num(privileged_state),
        }

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Fall + body-obstacle SDF collision (active after step 50) + box drop (always active)."""
        fall_termination = self.get_gravity(data, "pelvis")[2] < 0.0
        fall_termination |= info["head_pos"][2] < 0.7

        contact_termination = collision.geoms_colliding(data, self._right_foot_geom_id, self._left_foot_geom_id)
        contact_termination |= collision.geoms_colliding(data, self._left_foot_geom_id, self._right_shin_geom_id)
        contact_termination |= collision.geoms_colliding(data, self._right_foot_geom_id, self._left_shin_geom_id)

        thr = self._config.term_collision_threshold
        contact_termination |= jp.any(info['headdf'] < -thr)
        contact_termination |= jp.any(info['pelvdf'] < -thr)
        contact_termination |= jp.any(info['torsdf'] < -thr)
        contact_termination |= jp.any(info['feetdf'] < -thr)
        contact_termination |= jp.any(info['handsdf'] < -thr)
        contact_termination |= jp.any(info['kneesdf'] < -thr)
        contact_termination |= jp.any(info['shldsdf'] < -thr)
        contact_termination |= jp.any(info['boxdf'] < -thr)

        # Body-obstacle SDF collision active only after a short settle window
        # (lets the warm-start pose stabilize before collision penalties apply).
        contact_termination &= (info["step"] >= 50)

        # Box–thigh collision (box bumps into the upper legs while carried)
        box_thigh_termination = collision.geoms_colliding(data, self._box_geom_id, self._thigh_geom_ids[0])
        box_thigh_termination |= collision.geoms_colliding(data, self._box_geom_id, self._thigh_geom_ids[1])

        # Box–head collision (box bumps into the head while carried)
        box_head_termination = collision.geoms_colliding(data, self._box_geom_id, self._head_geom_id)

        # Box dropped — active throughout the full episode
        box_drop_termination = info['box_pos'][2] < self._config.box_drop_threshold

        return fall_termination | contact_termination | box_thigh_termination | box_head_termination | box_drop_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        # return fall_termination | contact_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()   # CHANGED


    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            done: jax.Array,
            feet_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        """Single-stage reward: navigation (CAT) + carry-maintenance terms."""
        # -----------------------------------------------------------------------
        # Always-active terms
        # -----------------------------------------------------------------------
        joint_torque   = self._cost_torque(data.actuator_force)
        smoothness_joint = self._cost_smoothness_joint(data, info["last_joint_vel"])
        joint_limits   = self._cost_joint_pos_limits(data.qpos[7:7 + NUM_ROBOT_JOINTS])
        hip_yaw_lim    = self._cost_hip_yaw(data.qpos[self._hip_yaw_indices + 7])

        # -----------------------------------------------------------------------
        # Grasp quantities reused by the carry-maintenance rewards below
        # -----------------------------------------------------------------------
        box_pos = data.xpos[self._box_body_id]
        box_half_y = info["box_size"][1]

        left_palm_pos  = data.site_xpos[self._hands_site_id[0]]
        right_palm_pos = data.site_xpos[self._hands_site_id[1]]
        box_quat = data.xquat[self._box_body_id]

        # Reach
        box_left_axis = math.rotate(jp.array([0., 1., 0.]), box_quat)
        left_target  = box_pos + box_left_axis * box_half_y
        right_target = box_pos - box_left_axis * box_half_y
        reach = -(jp.linalg.norm(left_palm_pos - left_target) + jp.linalg.norm(right_palm_pos - right_target))

        # Hand contact
        left_contact  = geoms_colliding(data, self._hand_geom_ids[0], self._box_geom_id)
        right_contact = geoms_colliding(data, self._hand_geom_ids[1], self._box_geom_id)
        hand_contact  = 0.5 * (left_contact.astype(jp.float32) + right_contact.astype(jp.float32))

        # Grasp symmetry
        box_up_axis  = math.rotate(jp.array([0., 0., 1.]), box_quat)
        box_fwd_axis = math.rotate(jp.array([1., 0., 0.]), box_quat)
        left_rel  = left_palm_pos - box_pos
        right_rel = right_palm_pos - box_pos
        height_diff = jp.dot(left_rel - right_rel, box_up_axis)
        depth_diff  = jp.dot(left_rel - right_rel, box_fwd_axis)
        grasp_symmetry = height_diff ** 2 + depth_diff ** 2

        # Palm orientation
        left_xmat  = data.site_xmat[self._hands_site_id[0]].reshape(3, 3)
        right_xmat = data.site_xmat[self._hands_site_id[1]].reshape(3, 3)
        left_palm_normal  = -left_xmat[:, 1]
        right_palm_normal =  right_xmat[:, 1]
        left_dot  = jp.dot(left_palm_normal, -box_left_axis)
        right_dot = jp.dot(right_palm_normal, box_left_axis)
        palm_orient = 0.5 * (0.5 * (1.0 + left_dot) + 0.5 * (1.0 + right_dot))

        # Hands level
        hands_vec  = left_palm_pos - right_palm_pos
        hands_level = hands_vec[2] ** 2 / (jp.dot(hands_vec, hands_vec) + 1e-6)

        # Lift
        box_half_z = info["box_size"][2]
        lift_height = box_pos[2] - (info["surface_z"] + info["support_half_z"] + box_half_z)
        lift = jp.clip(lift_height, 0.0, 0.10) / 0.10 - jp.clip(lift_height - 0.10, 0.0, None)

        # Box upright (tilt angle from the box quaternion)
        qx, qy = box_quat[1], box_quat[2]
        box_tilt_cos = jp.clip(1.0 - 2.0 * (qx ** 2 + qy ** 2), -1.0, 1.0)
        box_tilt_angle = jp.arccos(box_tilt_cos)

        # -----------------------------------------------------------------------
        # Navigation (CAT) + carry maintenance
        # -----------------------------------------------------------------------
        move_flag = info["command"][0]
        cmd_vel   = info["command"][1:].copy()

        rewards = {
            "joint_torque":    joint_torque,
            "smoothness_joint": smoothness_joint,
            "joint_limits":    joint_limits,
            "hip_yaw_lim":     hip_yaw_lim,
            "tracking_orientation": self._reward_orientation(
                info["navi_pelvis_rpy"], info["navi_torso_rpy"],
                info["head_pos"][2] > (self._config.torso_height[1] + 0.1),
            ),
            "tracking_root_field": self._reward_tracking_root_field(cmd_vel, info["global_lin_vel"]),
            "body_motion":         self._cost_body_motion(info["global_lin_vel"], info["navi_torso_ang_vel"], cmd_vel),
            "body_rotation":       self._reward_body_rotation(data, cmd_vel, info["navi2world_rot"]),
            "foot_contact_trav":   self._cost_foot_contact(data, feet_contact, info["gait_mask"], move_flag),
            "foot_clearance":      self._cost_foot_clearance(data, info["foot_height"], info["gait_mask"], move_flag),
            "foot_slip_trav":      self._cost_foot_slip(data, info["gait_mask"]),
            "foot_balance_trav":   self._cost_foot_balance(data, info["navi2world_pose"], move_flag),
            "foot_far":            self._cost_foot_far(data),
            "feet_apart":          self._cost_feet_apart(data),
            "straight_knee_trav":  self._cost_straight_knee(data.qpos[jp.array(self._knee_indices) + 7]),
            "feet_rotation":       self._reward_feet_rotation(data, info["navi2world_rot"]),
            "smoothness_action":   self._cost_smoothness_action(action, info["last_act"], info["last_last_act"]),
            "forward_progress":    self._reward_forward_progress(info["global_lin_vel"], cmd_vel),
            "upper_body_align":    jp.sum(jp.square(info["tors_pos"][:2] - info["pelv_pos"][:2]))
                                 + jp.sum(jp.square(info["head_pos"][:2] - info["pelv_pos"][:2])),
            "headgf": self._re_gf0(info["headgf"], info["head_vel"], info["headdf"],
                                    (move_flag[None] < 0.5) | (info["head_pos"][..., 0] > 1.5), tau=1.5),
            "feetgf": self._re_gf0(info["feetgf"], info["feet_vel"], info["feetdf"],
                                    (move_flag[None] < 0.5) | (info["gait_mask"] == 1) | (info["feet_pos"][..., 0] > 1.5), tau=1.5),
            "handsgf": self._re_gf0(info["handsgf"], info["hands_vel"], info["handsdf"],
                                     (move_flag[None] < 0.5) | (info["hands_pos"][..., 0] > 1.5), tau=1.5),
            "headdf":  self._re_sdf(info["headdf"]),
            "feetdf":  self._re_sdf(info["feetdf"]),
            "handsdf": self._re_sdf(info["handsdf"]),
            "kneesdf": self._re_sdf(info["kneesdf"]),
            "shldsdf": self._re_sdf(info["shldsdf"]),
            "boxdf":   self._re_sdf(info["boxdf"]),
            "boxgf":   self._re_boxgf(
                info["boxgf"], info["box_corners_vel"], info["boxdf"],
                (move_flag[None] < 0.5) | (info["box_corners"][..., 0] > 1.5),
                cmd_vel, tau=1.5),
            # Carry maintenance (reuse the grasp quantities computed above)
            "reach_carry":          reach,
            "lift_carry":           lift,
            "hand_contact_carry":   hand_contact,
            "grasp_symmetry_carry": grasp_symmetry,
            "palm_orient_carry":    palm_orient,
            "hands_level_carry":    hands_level,
            "box_upright_carry":    jp.exp(-box_tilt_angle ** 2),
        }

        for k, v in rewards.items():
            rewards[k] = jp.where(jp.isnan(v), 0.0, v)
        return rewards

    def _cost_feet_apart(self, data: mjx.Data) -> jax.Array:
        """Penalize the two feet being more than 0.5 m apart (over-wide stance).
        Linear: 0 when feet are within 0.5 m, grows with the excess distance.
        Complements _cost_foot_far, which penalizes feet too CLOSE (< 0.35 m).
        returns: scalar >= 0.
        """
        foot_pos = data.site_xpos[self._feet_site_id]              # (2, 3) world positions of both feet
        foot_distance = jp.linalg.norm(foot_pos[0] - foot_pos[1])  # scalar 3D distance
        return jp.clip(foot_distance - 0.5, 0.0, None)

    def _cost_hip_yaw(self, hip_yaw_pos: jax.Array) -> jax.Array:
        """Penalize the two hip-yaw joints leaving the [-0.5, 0.5] rad range.
        Linear out-of-range violation (same shape as _cost_joint_pos_limits), summed over both hips:
        positive below -0.5 or above +0.5, zero inside.
        hip_yaw_pos: (2,) hip-yaw joint angles [left, right].
        returns: scalar >= 0.
        """
        out = jp.clip(-1.5 - hip_yaw_pos, 0.0, None) + jp.clip(hip_yaw_pos - 1.5, 0.0, None)
        return jp.sum(out)

    def _reward_forward_progress(self, global_lin_vel: jax.Array, cmd_vel: jax.Array) -> jax.Array:
        """Reward velocity in the commanded direction, linear and always non-negative.
        Unlike tracking_root_field (exp-based), this gives nonzero gradient from a dead stop.
        cmd_vel: (3,) [vx, vy, yaw]; global_lin_vel: (3,) pelvis world-frame velocity.
        """
        cmd_xy = cmd_vel[:2]
        cmd_norm = jp.linalg.norm(cmd_xy) + 1e-6
        cmd_dir = cmd_xy / cmd_norm
        v_along_cmd = jp.dot(global_lin_vel[:2], cmd_dir)
        return jp.clip(v_along_cmd, 0.0, cmd_norm)

    def _reward_feet_rotation(self, data: mjx.Data, navi2world_rot: jax.Array) -> jax.Array:
        """Reward clean knee + ankle alignment with the navigation frame.
        Ported verbatim from G1CatPriEnv._reward_feet_rotation (env_cat_pri.py:890).
        Five off-axis terms (knee roll/yaw, ankle roll/pitch/yaw) summed inside exp(-.).
        Peaks at 1.0 when legs are vertical with feet flat and toes pointing forward;
        falls off as any element drifts. Indirectly penalizes deep crouching because
        deep knee bend forces ankle pitch compensation to keep feet flat.
        """
        knees2world_rot = jp.concat([
            data.xmat[self.body_id_knee_l][None],
            data.xmat[self.body_id_knee_r][None],
        ])
        knees2navi_rot = navi2world_rot.T[None] @ knees2world_rot
        ankles2world_rot = jp.concat([
            data.xmat[self.body_id_ankle_l][None],
            data.xmat[self.body_id_ankle_r][None],
        ])
        ankles2navi_rot = navi2world_rot.T[None] @ ankles2world_rot

        knees_roll_err  = jp.sum(jp.abs(knees2navi_rot[:, 2, 1]))
        knees_yaw_err   = jp.sum(jp.abs(knees2navi_rot[:, 0, 1]))
        ankles_roll_err = jp.sum(jp.abs(ankles2navi_rot[:, 1, 2]))
        ankles_pitch_err = jp.sum(jp.abs(ankles2navi_rot[:, 0, 2]))
        ankles_yaw_err  = jp.sum(jp.square(ankles2navi_rot[:, 0, 1]))

        return jp.exp(-1.0 * (
            knees_roll_err + knees_yaw_err
            + ankles_roll_err + ankles_pitch_err + ankles_yaw_err
        ))


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
