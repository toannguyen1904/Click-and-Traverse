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
"""Joystick task for Unitree G1."""

from typing import Any, Dict, Optional, Union
import jax
import jaxlie
import jax.numpy as jp
from jax.scipy.ndimage import map_coordinates
from jax import lax
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

import cat_ppo
from cat_ppo.envs.g1.env_loco import G1LocoEnv
from cat_ppo.envs.g1 import constants as consts

ENABLE_RANDOMIZE = True
EPS = 1e-6

def g1_loco_task_config() -> config_dict.ConfigDict:
    from cat_ppo.envs.g1.randomize import domain_randomize

    # env_config: ConfigDict with all the environment configuration.
    env_config = config_dict.create(
        task_type="flat_terrain",
        ctrl_dt=0.02,   # the physics timestep. Mujoco steps the simulation at this rate
        sim_dt=0.002,   # the control timestep, meaning 50Hz control frequency
        episode_length=1000,    # each episode runs for 1000 control steps
        action_repeat=1,    # running fully at 50Hz
        action_scale=0.5,   # output action is scaled by this
        history_len=15, # the last 15 timesteps of (action, joint position, joint velocity) are stacked and included in the observation.
        # This gives the policy a short temporal window to reason about motion history 
        num_obs=162,    # 162D state vector for the actor network
        num_pri=224,    # 224 privileged_state vector fed to the critic network only
        num_act=12, # 6 left leg and 6 right leg joints
        restricted_joint_range=False,   # not being used
        soft_joint_pos_limit_factor=0.95,   # soft joint limit of 95% of the full hardware range
        gait_config=config_dict.create( # gait clock configuration, used for policy's observation and reward
            gait_bound=0.6, # cos(phase) > gait_bound: right swing; cos(phase) < -0.6: left swing; otherwise: transition/double support
            freq_range=[1.3, 1.5],  # each step is advanced by phase_dt = 2pi x 0.02 x freq; not being fixed for domain randomization
            foot_height_range=[0.07, 0.07], # used for foot_clearance reward encouranging to lift the foot to 7cm.
        ),
        dm_rand_config=config_dict.create(  # domain randomization to simulate hardware noise
            enable_pd=True,
            # understand kp and kd:
            # 1. policy outputs action a
            # 2. target = default_qpos + action_scale x a
            # 3. torque = kp x (target - current) + kd x (0 - vel), target velocity (target_vel) is 0: position controller
            kp_range=[0.75, 1.25],
            kd_range=[0.75, 1.25],
            enable_rfi=True,    # random force injection, but actually torque
            rfi_lim=0.1,    # random noise injected into each joint's torque is at most 10% of that joint's torque limit
            rfi_lim_range=[0.5, 1.5],   # the injected noise is further scaled by a random factor in [0.5, 1.5]
            enable_ctrl_delay=False,    # False means not simulating the communication latency between the policy and the motors on real hardware
            ctrl_delay_range=[0, 2],    # delay would be randomly 0, 1, or 2 control steps (0-40ms at 50Hz), but since enable_ctrl_delay is set to False then it doesn't really matter
        ),
        noise_config=config_dict.create(    # sensor noise configuration, the noise applied to each sensor is noisy_value = true_value + uniform(-1, 1) × level × scale
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(   # reward configuration
            scales=config_dict.create(
                # behavior reward
                tracking_orientation=2.0,
                tracking_root_field=1.0,
                body_motion=-0.5,
                body_rotation=1.0,
                foot_contact=-1.0,
                foot_clearance=-15.0,
                foot_slip=-0.5,
                foot_balance=-30, 
                foot_far = -0,
                straight_knee = -30,
                # energy reward
                smoothness_joint=-1e-6,
                smoothness_action=-1e-3,
                joint_limits=-1.0,
                joint_torque=-1e-4,
                # field
                headgf=0.0,
                handsgf=0.0,
                feetgf=0.0,
                headdf=0.0,
                handsdf=0.0,
                feetdf=0.0,
                kneesdf=0.0,
                shldsdf=0.0,
            ),
            base_height_target=0.75,  # expected pelvis height above ground during normal walking (m); used in orientation reward and as nav frame Z origin
            foot_height_stance=0.0,  # target foot height during stance (on ground); added to tar_foot_height for swing target: stance(0.0) + swing(0.07) = 0.07m
        ),
        term_collision_threshold=0.04,  # SDF below -0.04m (4cm inside obstacle) triggers episode termination
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],   # seconds between pushes, randomly sampled each episode
            magnitude_range=[0.1, 1.0],   # velocity impulse (m/s) added to root XY velocity — NOT force in Newtons; 1.0 m/s is a hard shove for a ~35kg robot
        ),
        command_config=config_dict.create(
            # NOTE: unused in G1Cat — command is overwritten every step by compute_cmd_from_rtf (from GF fields).
            # Inherited from G1LocoEnv where command is randomly resampled; kept for config compatibility.
            resampling_time=10.0,  # seconds between command resamples (G1Loco only)
            stop_prob=0.2,         # probability of sampling a stop command (G1Loco only)
        ),
        # NOTE: unused in G1Cat — command velocity comes from compute_cmd_from_rtf, not random sampling.
        # Used in G1LocoEnv's sample_command; kept here for config compatibility.
        lin_vel_x=[-0.5, 0.5],    # (m/s) forward/backward velocity range
        lin_vel_y=[-0.3, 0.3],    # (m/s) lateral velocity range
        ang_vel_yaw=[-0.5, 0.5],  # (rad/s) yaw rotation range
        torso_height=[0.5, consts.DEFAULT_CHEST_Z],  # [unused, upper(m)]: head above upper+0.1m = robot is upright → relaxes orientation penalty
        pf_config=config_dict.create(   # configuration for loading the precomputed HumanoidPF fields
            path='data/assets/TypiObs/empty', # NOTE
            dx=0.04,
            origin=np.array([-0.5, -1.0, 0.0], dtype=np.float32),
        ),
    )

    # policy_config: ConfigDict with all the policy configuration.
    policy_config = config_dict.create(
        num_timesteps=5_000_000_000,  # total env steps budget (overridden to 400M via CLI --num_timesteps)
        max_devices_per_host=8,       # max GPUs to use per machine
        # high-level control flow
        wrap_env=True,          # wrap env with Brax observation normalization and episode reset logic
        madrona_backend=False,  # use Madrona GPU-accelerated renderer (disabled; using MJX instead)
        augment_pixels=False,   # pixel augmentation for visual observations (unused; this env has no pixel obs)
        # environment wrapper
        num_envs=32768,       # number of parallel environments; must be divisible by device count. 8192(256*32), 16384(512*32), 32768(1024*32)
        episode_length=1000,  # max steps per episode = 20s at 50Hz; episode resets on termination or when reached
        action_repeat=1,      # policy called every step (no action repetition); effective policy frequency = sim_freq / action_repeat = 50Hz
        wrap_env_fn=None,     # custom env wrapper injected at training time via _prepare_training_params; overrides default Brax wrapper
        randomization_fn=domain_randomize if ENABLE_RANDOMIZE else None,
        # ppo params
        learning_rate=3e-4, # Brax's PPO implementation uses a single optimizer with the same learning rate for both the policy (actor) and value (critic) networks. 
        entropy_cost=0.01,  # entropy regularization weight: penalizes overconfident actions to encourage exploration
        discounting=0.97,             # PPO discount factor γ; lower than typical 0.99 to reduce variance in long episodes
        unroll_length=20,             # steps collected per env before each PPO update (0.4s of experience per rollout)
        batch_size=1024,              # number of envs per minibatch update; total batch = batch_size × num_minibatches × unroll_length. 256, 512, 1024
        num_minibatches=32,           # number of minibatches to split the rollout into per update; total envs = batch_size × num_minibatches = 32768
        num_updates_per_batch=4,      # number of gradient update passes over the same rollout data (PPO epochs)
        num_resets_per_eval=0,        # number of forced env resets before each eval run (0 = no forced resets)
        normalize_observations=False, # disable running-mean observation normalization (HumanoidPF fields have inconsistent scales)
        reward_scaling=1.0,           # scalar multiplied on all rewards before PPO loss (1.0 = no scaling)
        clipping_epsilon=0.2,         # PPO clip range: limits policy update ratio to [1-ε, 1+ε] to prevent destructive large updates
        gae_lambda=0.95,  # GAE λ: advantage = Σ(γλ)^l × δ_{t+l}; 0.95 ≈ Monte Carlo (low bias, higher variance)
        max_grad_norm=1.0,        # gradient clipping threshold: scale down gradients if norm exceeds this to prevent instability
        normalize_advantage=True, # normalize advantages to zero mean and unit std per minibatch for stable PPO loss scale
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 128, 64),    # actor MLP: 162-dim state → 256 → 128 → 64 → 12-dim action; swish activation
            value_hidden_layer_sizes=(512, 256, 128),    # critic MLP: larger than actor since privileged_state has more info; 224-dim → 512 → 256 → 128 → scalar value
            policy_obs_key="state",                      # actor reads noisy 162-dim obs (deployable on real hardware)
            value_obs_key="privileged_state",            # critic reads noiseless 224-dim obs (training only, asymmetric actor-critic)
        ),
        seed=0,  # random seed for training reproducibility
        # eval
        num_evals=6,              # total eval checkpoints: 1 at init + 5 evenly spaced → ~80M steps apart for 400M total
        eval_env=None,            # separate eval env instance; None = use training env. Overridden in train_ppo.py with a dedicated _eval_env
        num_eval_envs=0,          # parallel envs for eval; 0 falls back to 128 in train.py. Uses same scene as training env
        deterministic_eval=False, # if True, use policy mean action (no sampling) during eval; False = stochastic (more representative of training)
        # training metrics
        log_training_metrics=True,           # enable logging of PPO metrics (loss, entropy, etc.) to wandb
        training_metrics_steps=int(1e6),     # log metrics every 1M env steps
        # callbacks
        progress_fn=lambda *args: None,      # progress callback; overridden in train_ppo.py with wandb logging + ETA estimation
        # policy_params_fn=lambda *args: None,
        # checkpointing
        save_checkpoint_path=None,           # path to save checkpoints; set at runtime via _prepare_training_params
        restore_checkpoint_path=None,        # path to restore checkpoint from to resume training; set via --restore_name CLI arg
        restore_params=None,                 # optionally inject pre-loaded params directly instead of loading from path
        restore_value_fn=False,              # if True, also restore value network weights when resuming (False = reinitialize critic)
    )

    # NOTE: eval_config is currently unused — loaded in train_ppo.py but never consumed.
    # Intended for a future scripted post-training evaluation run with fixed waypoints and duration.
    # vel: move_flag[0|1], x[m], y[m], yaw[rad]
    eval_config = config_dict.create(
        duration=50.0,             # planned eval episode duration in seconds
        command_waypoints=np.array(
            [
                [0, 0.0, 0.0, 0.0],  # sequence of [move_flag, vx, vy, yaw] commands for scripted eval
            ]
        ),
    )

    # config: ConfigDict with env_config, policy_config, eval_config for the given task.
    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
        eval_config=eval_config,
    )
    return config

# This registers the config object for the task "G1Cat" with the category "config".
cat_ppo.registry.register("G1Cat", "config")(g1_loco_task_config())

@jax.jit
def world_to_navi_pos(navi2world_pose: jp.ndarray, pos: jp.ndarray) -> jp.ndarray:  # not being used
    world2navi = jp.linalg.inv(navi2world_pose)
    R = world2navi[:3, :3]
    t = world2navi[:3, 3]
    return (R @ pos.T).T + t

@jax.jit
def world_to_navi_vel(navi2world_pose: jp.ndarray, vel: jp.ndarray) -> jp.ndarray:
    # Rotate direction vectors from world frame into navigation frame (rotation only, no translation).
    # navi2world_pose: (4,4) — nav-to-world homogeneous transform;
    # vel: (N,3) — vectors in world frame;
    # returns (N,3) in nav frame
    world2navi = jp.linalg.inv(navi2world_pose)
    R = world2navi[:3, :3]
    return (R @ vel.T).T

@jax.jit
def quat_conj(q):
    # Conjugate of quaternion q = [w, x, y, z] → [w, -x, -y, -z]: reverses the rotation direction.
    # q: (..., 4) wxyz. For example, if q rotates from frame A → frame B, then quat_conj(q) rotates from frame B → frame A.
    return jp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)

@jax.jit
def quat_mul(q1, q2):
    # Multiply two quaternions q1 ⊗ q2: composes rotations (apply q2 first, then q1).
    # q1, q2: (..., 4) wxyz
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)

@jax.jit
def quat_rotate(q, v):
    # Rotate vector v by quaternion q via sandwich product q ⊗ [0,v] ⊗ q*. q: (...,4) wxyz; v: (...,3); returns (...,3)
    zeros = jp.zeros_like(v[..., :1])
    q_v = jp.concatenate([zeros, v], axis=-1)
    return quat_mul(quat_mul(q, q_v), quat_conj(q))[..., 1:]

@jax.jit
def delay_body_pos(p_gt, q_gt, p_odom, q_odom, body_pos):
    # Re-express body part positions using delayed odometry pose instead of ground truth,
    # simulating the SLAM localization lag on real hardware (odom updated every 5 steps).
    # p_gt:     (3,)   — ground-truth root position this step
    # q_gt:     (4,)   — ground-truth root orientation (wxyz) this step
    # p_odom:   (3,)   — delayed odometry root position (frozen up to 5 steps ago)
    # q_odom:   (4,)   — delayed odometry root orientation (wxyz)
    # body_pos: (N, 3) — true world positions of N body parts
    # returns:  (N, 3) — body positions as perceived through delayed odometry
    # head (1) + pelvis (1) + torso (1) + feet (2) + hands (2) + knees (2) + shoulders (2) = N=11
    body_pos_local = quat_rotate(quat_conj(q_gt), body_pos - p_gt)  # transform to root-local frame using ground truth
    return (p_odom + quat_rotate(q_odom, body_pos_local)).reshape(-1,3)  # transform back to world using delayed odom

@jax.jit
def normalize(q):
    # normalize the quaternion
    return q / jp.linalg.norm(q, axis=-1, keepdims=True)

@jax.jit
def delay_rootpose_noisy(key, qpos_root):
    # Add noise to root pose to simulate odometry drift: ±5cm position offset + ±10° yaw rotation.
    # NOTE: currently unused — commented out in favor of clean delayed pose (see odom_delay update).
    # key:       JAX random key
    # qpos_root: (7,) — root pose [x, y, z, qw, qx, qy, qz]
    # returns:   (7,) — noisy root pose
    dxyz = (jax.random.uniform(key, (3,)) * 2 - 1) * 0.05  # (3,) random ±5cm position noise

    q_gt = qpos_root[3:7]  # (4,) wxyz
    angle = (jax.random.uniform(key, ()) * 2 - 1) * jp.deg2rad(10.0)  # random ±10° yaw angle
    half = angle / 2.0

    q_dr = jp.stack([jp.cos(half), 0.0, 0.0, jp.sin(half)])  # yaw-only rotation quaternion (rotation around z-axis)

    q_new = normalize(quat_mul(q_dr, q_gt))  # apply yaw noise to ground-truth orientation

    return jp.concatenate([qpos_root[:3] + dxyz, q_new], axis=0)  # (7,)

@jax.jit
def base2navi_transform(base2world: jax.Array) -> jax.Array:
    """
    Compute the navigation frame rotation matrix from the pelvis (base) rotation.
    The navigation frame is robot-centered and yaw-only: x points in the pelvis's
    horizontal heading, z always points up. Roll and pitch are stripped out by
    projecting the pelvis x-axis onto the horizontal plane (setting z=0).
    Only the pelvis is used — the torso rotation does NOT affect this frame.

    The nav frame is defined w.r.t. the pelvis because the IMU is physically located
    in the pelvis (site="imu_in_pelvis" in the MJCF), making it the natural reference
    frame consistent between sim and real hardware deployment.

    base2world: (3, 3) — pelvis rotation matrix in world frame (from site_xmat)
    returns:    (3, 3) — navi2world rotation matrix
    """
    x = base2world[:, 0]              # pelvis forward axis in world frame
    x_proj = x.at[2].set(0.0)        # project onto horizontal plane (strip pitch/roll)
    x_proj /= jp.linalg.norm(x_proj) # normalize to unit vector
    z_axis = jp.array([0.0, 0.0, 1.0])
    y_axis = jp.cross(z_axis, x_proj)  # y = up × forward = left
    y_axis /= jp.linalg.norm(y_axis)
    x_axis = jp.cross(y_axis, z_axis)  # x = left × up = forward (re-orthogonalized)
    return jp.column_stack((x_axis, y_axis, z_axis))


def torque_step(
        rng: jax.Array,           # JAX random key for RFI noise; updated and returned each substep
        model: mjx.Model,         # MuJoCo model (static geometry, joint definitions)
        data: mjx.Data,           # current MuJoCo sim state (qpos, qvel, etc.)
        qpos_des: jax.Array,      # (29,) desired joint angles = default_qpos + policy_action * action_scale for legs, default_qpos for waist/arms
        kps: jax.Array,           # (29,) PD proportional gains
        kds: jax.Array,           # (29,) PD derivative gains
        kp_scale: jax.Array,      # scalar domain randomization scale for Kp (sampled per episode, range [0.75, 1.25])
        kd_scale: jax.Array,      # scalar domain randomization scale for Kd (sampled per episode, range [0.75, 1.25])
        rfi_lim_scale: jax.Array, # scalar RFI noise scale (sampled per episode); controls magnitude of random torque injection
        torque_limit: jax.Array,  # (29,) per-joint torque limits for clipping
        n_substeps: int = 1,      # number of physics steps per policy step (10 substeps × sim_dt=0.002s = ctrl_dt=0.02s)
) -> tuple[jax.Array, mjx.Data]:
    # Run n_substeps of PD control + physics simulation for one policy action.
    # Returns updated (rng, data) after all substeps.
    def single_step(carry, _):
        rng, data = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        # pd control
        pos_err = qpos_des - data.qpos[7:]  # (29,) position error (skip root 7-DoF)
        vel_err = -data.qvel[6:]            # (29,) velocity error: target vel=0 (position controller)
        torque = (kp_scale * kps) * pos_err + (kd_scale * kds) * vel_err  # (29,)

        # rfi noise
        rfi_noise = rfi_lim_scale * jax.random.uniform(rng_rfi, shape=torque.shape, minval=-1.0, maxval=1.0)  # (29,) random torque noise
        torque += rfi_noise

        # clip
        torque = jp.clip(torque, -torque_limit, torque_limit)  # (29,) clamp to hardware limits

        # apply torque
        data = data.replace(ctrl=torque)  # set actuator commands
        data = mjx.step(model, data)      # advance physics by sim_dt=0.002s

        return (rng, data), None

    return jax.lax.scan(single_step, (rng, data), (), n_substeps)[0]



@cat_ppo.registry.register("G1Cat", "train_env_class")  # This registers the class G1CatEnv with the category "train_env_class" for the task "G1Cat".
class G1CatEnv(G1LocoEnv):
    """Track a joystick command."""

    def __init__(
            self,
            task_type: str = "flat_terrain",
            config: config_dict.ConfigDict = None,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task_type=task_type,
            config=config,
            config_overrides=config_overrides,
        )
        pf_path = config.pf_config.path
        self.dx = config.pf_config.dx
        self.sdf = jp.array(np.load(f"{pf_path}/sdf.npy"))[...,None]  # (Nx,Ny,Nz)
        self.bf  = jp.array(np.load(f"{pf_path}/bf.npy"))    # (Nx,Ny,Nz,3)
        self.gf  = jp.array(np.load(f"{pf_path}/gf.npy"))    # (Nx,Ny,Nz,3)
        self.pf_origin = jp.array(np.array(config.pf_config.origin, dtype=np.float32), dtype=jp.float32)  # (3,) world-space XYZ of voxel grid corner [0,0,0]; used to convert world pos → grid index
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape
        self._head_site_id = self._mj_model.site("head").id  # int — site ID for "head"
        self._knees_site_id = np.array([self._mj_model.site(name).id for name in consts.KNEE_SITES])     # (2,) — site IDs for ["left_knee", "right_knee"]
        self._shlds_site_id = np.array([self._mj_model.site(name).id for name in consts.SHOULDER_SITES]) # (2,) — site IDs for ["left_shoulder", "right_shoulder"]

        # print("========================")
        # print(self._head_site_id)
        # print(self._knees_site_id)
        # print(self._shlds_site_id)
        # print("========================")
        # exit()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """
        Resets the environment to a new initial state at the start of each episode —
        randomizes the robot's starting pose, samples new domain randomization parameters
        (PD scales, RFI scale), initializes all info fields, and computes the initial HumanoidPF observations.
        """
        qpos = self._init_q.copy()          # (36,) jax float32 — [root_xyz(3), root_quat(4), joint_angles(29)]; from DEFAULT_QPOS, robot upright at z=0.8m (torso)
        qvel = jp.zeros(self.mjx_model.nv)  # (35,) jax float32 — [root_linvel(3), root_angvel(3), joint_vel(29)]; all zero (start from rest)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        # randomize starting XY position within ±1m of default position for diverse obstacle exposure each episode
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)  # (2,) random XY offset in [-1, 1]m
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)  # apply XY offset to root position
        qpos = qpos.at[2].set(0.8)  # fix torso Z at 0.8m — default leg pose places feet roughly on ground at this height

        # randomize starting yaw orientation within ±90° so the robot faces different directions each episode
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-np.pi / 2, maxval=np.pi / 2)  # random yaw angle in [-90°, 90°]
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)  # convert yaw angle to quaternion (rotation around z-axis)
        new_quat = math.quat_mul(qpos[3:7], quat)  # compose with current orientation (upright from DEFAULT_QPOS)
        qpos = qpos.at[3:7].set(new_quat)  # apply randomized yaw to root quaternion

        # randomize starting joint angles by scaling default pose by U(0.5, 1.5) — diverse initial body configurations
        rng, key = jax.random.split(rng)
        rand_qpos = qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)  # (29,) scale each joint angle by random factor in [0.5, 1.5]
        rand_qpos = jp.clip(rand_qpos, self._soft_lowers, self._soft_uppers)            # clamp to soft joint limits to keep pose physically valid
        qpos = qpos.at[7:].set(rand_qpos)  # apply randomized joint angles (skip root 7-DoF)

        # d(xyzrpy)=U(-0.5, 0.5)
        # randomize initial root velocity (linear + angular) to simulate mid-motion starts
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))  # (6,) random root linvel(3) + angvel(3) in [-0.5, 0.5] m/s or rad/s
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])  # initialize MuJoCo state; ctrl=qpos[7:] sets initial motor targets to current joint angles

        # rng, cmd_rng = jax.random.split(rng)
        head_pos = data.site_xpos[self._head_site_id]   # position of "head" site, shape (3,). data.site_xpos is a (num_sites, 3) array of world-space positions for all named sites in the model,
        head_vel = jp.zeros_like(head_pos)  # (3,) zero velocity at reset — will be computed as finite diff each step
        headgf = self.sample_field(self.gf, head_pos.reshape(1, -1))   # (1, 3) guidance field vector at head position
        headbf = self.sample_field(self.bf, head_pos.reshape(1, -1))   # (1, 3) boundary field vector at head position
        headdf = self.sample_field(self.sdf, head_pos.reshape(1, -1))  # (1, 1) signed distance to nearest obstacle at head position
        pelv_pos = data.site_xpos[self._pelvis_imu_site_id]  # (3,) world-space position of pelvis IMU site
        tors_pos = data.site_xpos[self._torso_imu_site_id]   # (3,) world-space position of torso IMU site
        pelvgf = self.sample_field(self.gf,  pelv_pos.reshape(1, -1))  # (1, 3) GF at pelvis
        pelvbf = self.sample_field(self.bf,  pelv_pos.reshape(1, -1))  # (1, 3) BF at pelvis
        pelvdf = self.sample_field(self.sdf, pelv_pos.reshape(1, -1))  # (1, 1) SDF at pelvis
        torsgf = self.sample_field(self.gf,  tors_pos.reshape(1, -1))  # (1, 3) GF at torso
        torsbf = self.sample_field(self.bf,  tors_pos.reshape(1, -1))  # (1, 3) BF at torso
        torsdf = self.sample_field(self.sdf, tors_pos.reshape(1, -1))  # (1, 1) SDF at torso
        feet_pos = data.site_xpos[self._feet_site_id]  # (2, 3) world-space positions of [left_foot, right_foot] sites
        feet_vel = jp.zeros_like(feet_pos)              # (2, 3) zero velocity at reset — computed as finite diff each step
        feetgf = self.sample_field(self.gf,  feet_pos)  # (2, 3) GF at each foot
        feetbf = self.sample_field(self.bf,  feet_pos)  # (2, 3) BF at each foot
        feetdf = self.sample_field(self.sdf, feet_pos)  # (2, 1) SDF at each foot
        hands_pos = data.site_xpos[self._hands_site_id]  # (2, 3) world-space positions of [left_palm, right_palm] sites
        hands_vel = jp.zeros_like(hands_pos)              # (2, 3) zero velocity at reset — computed as finite diff each step
        handsgf = self.sample_field(self.gf,  hands_pos)  # (2, 3) GF at each hand
        handsbf = self.sample_field(self.bf,  hands_pos)  # (2, 3) BF at each hand
        handsdf = self.sample_field(self.sdf, hands_pos)  # (2, 1) SDF at each hand
        knees_pos = data.site_xpos[self._knees_site_id]   # (2, 3) world-space positions of [left_knee, right_knee] sites
        kneesgf = self.sample_field(self.gf,  knees_pos)  # (2, 3) GF at each knee
        kneesbf = self.sample_field(self.bf,  knees_pos)  # (2, 3) BF at each knee
        kneesdf = self.sample_field(self.sdf, knees_pos)  # (2, 1) SDF at each knee
        shlds_pos = data.site_xpos[self._shlds_site_id]   # (2, 3) world-space positions of [left_shoulder, right_shoulder] sites
        shldsgf = self.sample_field(self.gf,  shlds_pos)  # (2, 3) GF at each shoulder
        shldsbf = self.sample_field(self.bf,  shlds_pos)  # (2, 3) BF at each shoulder
        shldsdf = self.sample_field(self.sdf, shlds_pos)  # (2, 1) SDF at each shoulder

        # Compute the initial velocity command for the episode
        command = self.compute_cmd_from_rtf(pelvgf.reshape(-1), jp.concat([headgf, feetgf, handsgf], axis=0), jp.concat([headbf, feetbf, handsbf], axis=0))

        # Sample push interval for domain randomization
        # sample push interval for this episode: how many steps between random force pushes
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],  # 5.0s
            maxval=self._config.push_config.interval_range[1],  # 10.0s
        )  # random push interval in [5, 10] seconds
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)  # convert to control steps: [250, 500] steps at dt=0.02s

        # initialize gait clock parameters for this episode
        rng, gait_freq_rng, foot_height_rng = jax.random.split(rng, 3)
        gait_freq = jax.random.uniform(
            gait_freq_rng,
            minval=self._config.gait_config.freq_range[0],  # 1.3 Hz
            maxval=self._config.gait_config.freq_range[1],  # 1.5 Hz
        )  # random stepping frequency in [1.3, 1.5] Hz — behavioral diversity across episodes
        phase_dt = 2 * jp.pi * self.dt * gait_freq  # phase increment per control step (rad); phase advances by this each 0.02s
        rng, phase_rng = jax.random.split(rng)
        cond_phase = jax.random.bernoulli(phase_rng)  # 50/50 coin flip for starting foot
        phase = jp.where(cond_phase, self._init_phase_l, self._init_phase_r)  # randomly start with left or right foot leading
        # phase = self._init_phase_l.copy()
        foot_height = jax.random.uniform(
            foot_height_rng,
            minval=self._config.gait_config.foot_height_range[0],  # 0.07m
            maxval=self._config.gait_config.foot_height_range[1],  # 0.07m (fixed in G1Cat)
        )  # target foot clearance during swing phase (m)

        # domain randomization: sample PD gain scales and RFI noise magnitude for this episode
        rng, key_kp, key_kd, key_rfi, key_delay = jax.random.split(rng, 5)

        # PD gain randomization: scale Kp and Kd by U(0.75, 1.25) to simulate motor uncertainty
        kp_scale = jax.random.uniform(
            key_kp,
            minval=self._config.dm_rand_config.kp_range[0],  # 0.75
            maxval=self._config.dm_rand_config.kp_range[1],  # 1.25
        )
        kp_scale = jp.where(self._config.dm_rand_config.enable_pd, kp_scale, jp.ones_like(kp_scale))  # 1.0 if disabled

        kd_scale = jax.random.uniform(
            key_kd,
            minval=self._config.dm_rand_config.kd_range[0],  # 0.75
            maxval=self._config.dm_rand_config.kd_range[1],  # 1.25
        )
        kd_scale = jp.where(self._config.dm_rand_config.enable_pd, kd_scale, jp.ones_like(kd_scale))  # 1.0 if disabled

        # RFI (Random Force Injection): per-joint torque noise = rfi_lim × U(0.5, 1.5) × torque_limit
        rfi_lim_noise_scale = jax.random.uniform(
            key_rfi,
            self.torque_limit.shape,   # (29,) per-joint scale
            minval=self._config.dm_rand_config.rfi_lim_range[0],  # 0.5
            maxval=self._config.dm_rand_config.rfi_lim_range[1],  # 1.5
        )
        rfi_lim_scale = self._config.dm_rand_config.rfi_lim * rfi_lim_noise_scale * self.torque_limit  # (29,) max noise per joint
        rfi_lim_scale = jp.where(self._config.dm_rand_config.enable_rfi, rfi_lim_scale, jp.zeros_like(rfi_lim_scale))  # zero if disabled


        info = {
            "rng": rng,         # JAX random key, carried forward each step
            "step": 0,          # current step count within episode
            "command": command, # (4,) [move_flag, vx, vy, yaw] — recomputed each step from HumanoidPF
            # history — previous step values for smoothness rewards and finite-diff velocity
            "last_command": jp.zeros(4),                 # (4,) command from previous step
            "last_act": jp.zeros(self.action_size),      # (12,) policy action from previous step
            "last_last_act": jp.zeros(self.action_size), # (12,) policy action from two steps ago
            "last_feet_vel": jp.zeros(2),                # (2,) foot speed magnitude from previous step
            "last_joint_vel": np.zeros(self.num_joints), # (29,) joint velocities from previous step for smoothness cost
            # push — external force disturbance state
            "push": jp.array([0.0, 0.0]),            # (2,) current push XY direction
            "push_step": 0,                          # steps since last push
            "push_interval_steps": push_interval_steps, # int — steps between pushes, sampled per episode in [250, 500]
            # state — robot kinematic state updated each step
            "motor_targets": self._default_qpos.copy(), # (29,) current PD target joint angles
            "local_lin_vel": jp.zeros(3),               # (3,) pelvis linear velocity in local frame
            "global_lin_vel": jp.zeros(3),              # (3,) pelvis linear velocity in world frame
            "global_ang_vel": jp.zeros(3),              # (3,) pelvis angular velocity in world frame
            "navi2world_rot": jp.eye(3),                # (3,3) nav frame rotation matrix
            "navi2world_pose": jp.eye(4),               # (4,4) nav frame homogeneous transform
            "navi_torso_rpy": jp.zeros(3),              # (3,) torso roll-pitch-yaw in nav frame
            "navi_torso_lin_vel": jp.zeros(3),          # (3,) torso linear velocity in nav frame
            "navi_torso_ang_vel": jp.zeros(3),          # (3,) torso angular velocity in nav frame
            "navi_pelvis_rpy": jp.zeros(3),             # (3,) pelvis roll-pitch-yaw in nav frame
            "navi_pelvis_lin_vel": jp.zeros(3),         # (3,) pelvis linear velocity in nav frame
            "navi_pelvis_ang_vel": jp.zeros(3),         # (3,) pelvis angular velocity in nav frame
            # gait clock
            "stop_timestep": 100,          # steps before robot starts moving (warm-up period)
            "phase": phase,                # (4,) [cos_L, cos_R, sin_L, sin_R] — current gait phase
            "phase_dt": phase_dt,          # scalar — phase increment per step (rad)
            "gait_mask": jp.zeros(2),      # (2,) [left, right] ∈ {-1=swing, 0=transition, 1=stance}
            "gait_freq": gait_freq,        # scalar — stepping frequency (Hz), sampled per episode
            "foot_height": foot_height,    # scalar — target foot clearance during swing (m)
            # domain randomization scales, sampled per episode
            "kp_scale": kp_scale,          # scalar — PD Kp multiplier in [0.75, 1.25]
            "kd_scale": kd_scale,          # scalar — PD Kd multiplier in [0.75, 1.25]
            "rfi_lim_scale": rfi_lim_scale, # (29,) — per-joint RFI noise magnitude
            # HumanoidPF fields at current body part positions (ground-truth, used for rewards)
            "headgf": headgf.copy(),   # (1,3) GF at head
            "headbf": headbf.copy(),   # (1,3) BF at head
            "headdf": headdf.copy(),   # (1,1) SDF at head
            "pelvgf": pelvgf.copy(),   # (1,3) GF at pelvis
            "pelvbf": pelvbf.copy(),   # (1,3) BF at pelvis
            "pelvdf": pelvdf.copy(),   # (1,1) SDF at pelvis
            "torsgf": torsgf.copy(),   # (1,3) GF at torso
            "torsbf": torsbf.copy(),   # (1,3) BF at torso
            "torsdf": torsdf.copy(),   # (1,1) SDF at torso
            "feetgf": feetgf.copy(),   # (2,3) GF at feet
            "feetbf": feetbf.copy(),   # (2,3) BF at feet
            "feetdf": feetdf.copy(),   # (2,1) SDF at feet
            "handsgf": handsgf.copy(), # (2,3) GF at hands
            "handsbf": handsbf.copy(), # (2,3) BF at hands
            "handsdf": handsdf.copy(), # (2,1) SDF at hands
            "kneesgf": kneesgf.copy(), # (2,3) GF at knees
            "kneesbf": kneesbf.copy(), # (2,3) BF at knees
            "kneesdf": kneesdf.copy(), # (2,1) SDF at knees
            "shldsgf": shldsgf.copy(), # (2,3) GF at shoulders
            "shldsbf": shldsbf.copy(), # (2,3) BF at shoulders
            "shldsdf": shldsdf.copy(), # (2,1) SDF at shoulders
            # body part world-space positions and velocities (for finite-diff velocity and rewards)
            "head_pos": head_pos.copy(),   # (3,)
            "head_vel": head_vel.copy(),   # (3,)
            "pelv_pos": pelv_pos.copy(),   # (3,)
            "tors_pos": tors_pos.copy(),   # (3,)
            "feet_pos": feet_pos.copy(),   # (2,3)
            "feet_vel": feet_vel.copy(),   # (2,3)
            "hands_pos": hands_pos.copy(), # (2,3)
            "hands_vel": hands_vel.copy(), # (2,3)
            "knees_pos": knees_pos.copy(), # (2,3)
            "shlds_pos": shlds_pos.copy(), # (2,3)
            # delayed versions — fields/pose sampled using odometry-delayed body positions (simulates SLAM lag for observations)
            "command_delay": command,      # (4,) delayed command
            "odom_delay": qpos[:7],        # (7,) delayed root pose [xyz, quat]; frozen snapshot updated every 5 steps
            "headgf_delay": headgf.copy(),   "headbf_delay": headbf.copy(),   "headdf_delay": headdf.copy(),
            "pelvgf_delay": pelvgf.copy(),   "pelvbf_delay": pelvbf.copy(),   "pelvdf_delay": pelvdf.copy(),
            "torsgf_delay": torsgf.copy(),   "torsbf_delay": torsbf.copy(),   "torsdf_delay": torsdf.copy(),
            "feetgf_delay": feetgf.copy(),   "feetbf_delay": feetbf.copy(),   "feetdf_delay": feetdf.copy(),
            "handsgf_delay": handsgf.copy(), "handsbf_delay": handsbf.copy(), "handsdf_delay": handsdf.copy(),
            "kneesgf_delay": kneesgf.copy(), "kneesbf_delay": kneesbf.copy(), "kneesdf_delay": kneesdf.copy(),
            "shldsgf_delay": shldsgf.copy(), "shldsbf_delay": shldsbf.copy(), "shldsdf_delay": shldsdf.copy(),
        }

        # initialize metrics dict with zero scalars for each reward term — populated each step and logged to wandb
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])  # (2,) bool — [left_foot, right_foot] floor contact at reset
        obs = self._get_obs(data, info, contact)  # compute initial observation dict (state + privileged_state)
        reward, done = jp.zeros(2)                # reward=0, done=False at episode start
        return mjx_env.State(data, obs, reward, done, metrics, info)  # pack everything into the Brax state object

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Advance environment by one control step (0.02s): apply action, run physics, update HumanoidPF fields, compute obs/reward/done.
        # state:  mjx_env.State — current env state (physics data, obs, reward, done, metrics, info)
        # action: (12,) — policy output (joint angle offsets from default pose, scaled by action_scale=0.5)
        # returns: mjx_env.State — next state after one control step
        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)

        # apply random push disturbance before action — robot must recover without warning
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)  # random push direction (angle in [0, 2π])
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],  # 0.1 m/s
            maxval=self._config.push_config.magnitude_range[1],  # 1.0 m/s
        )  # random velocity impulse magnitude
        push_signal = jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0  # True only at push interval
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])  # (2,) unit XY direction vector
        push *= push_signal          # zero out if not push time
        push *= self._config.push_config.enable  # zero out if pushes disabled
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(qvel[:2] + push * push_magnitude)  # add velocity impulse to root XY velocity
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # compute motor targets and step physics
        # action (12,) → new targets for 12 leg joints: prev_target + action * 0.5, clipped to soft limits
        lower_motor_targets = jp.clip(
            state.info["motor_targets"][self.action_joint_ids]
            + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()  # full 29-joint target array, initialized to default pose
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)  # override 12 leg joints; remaining 17 upper-body joints (waist + arms) stay at default
        state.info["rng"], data = torque_step(  # PD torques → 10 physics substeps at 500 Hz
            state.info["rng"],
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            kp_scale=state.info["kp_scale"],   # domain-randomized PD gain scales
            kd_scale=state.info["kd_scale"],
            rfi_lim_scale=state.info["rfi_lim_scale"],  # domain-randomized RFI noise scale
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )

        # collect info
        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])  # (2,) bool: [left_foot, right_foot] touching floor
        state.info["motor_targets"] = motor_targets  # (29,) full joint target array, used next step as prev_target
        state.info["local_lin_vel"] = self.get_local_linvel(data, "pelvis")   # pelvis linear velocity in pelvis local frame
        state.info["global_lin_vel"] = self.get_global_linvel(data, "pelvis")  # pelvis linear velocity in world frame
        state.info["global_ang_vel"] = self.get_global_angvel(data, "pelvis")  # pelvis angular velocity in world frame

        # update navigation frame pose (4x4) to track robot's current position and heading
        pelvis2world_rot = data.site_xmat[self._pelvis_imu_site_id]  # (3,3) pelvis IMU rotation in world frame
        navi2world_rot = base2navi_transform(pelvis2world_rot)  # (3,3) yaw-only rotation (strips pitch/roll)
        state.info["navi2world_pose"] = state.info["navi2world_pose"].at[:3, :3].set(navi2world_rot)  # update rotation block
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[:2, 3].set(data.site_xpos[self._pelvis_imu_site_id][:2])  # update XY translation to current pelvis position
        )
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[2, 3].set(self._config.reward_config.base_height_target)  # fix Z at nominal pelvis height (0.75m) for stable foot balance geometry
        )

        # pelvis projection
        pelvis2navi_rot = navi2world_rot.T @ pelvis2world_rot
        state.info["navi2world_rot"] = navi2world_rot
        state.info["navi_pelvis_rpy"] = jp.array(jaxlie.SO3.from_matrix(pelvis2navi_rot).as_rpy_radians())
        state.info["navi_pelvis_lin_vel"] = pelvis2navi_rot @ self.get_local_linvel(data, "pelvis")
        state.info["navi_pelvis_ang_vel"] = pelvis2navi_rot @ self.get_gyro(data, "pelvis")
        # torso projection
        torso2world_rot = data.site_xmat[self._torso_imu_site_id]
        torso2navi_rot = navi2world_rot.T @ torso2world_rot
        state.info["navi_torso_rpy"] = jp.array(jaxlie.SO3.from_matrix(torso2navi_rot).as_rpy_radians())
        state.info["navi_torso_lin_vel"] = torso2navi_rot @ self.get_local_linvel(data, "torso")
        state.info["navi_torso_ang_vel"] = torso2navi_rot @ self.get_gyro(data, "torso")

        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])

        state.info["last_command"] = state.info["command"].copy()
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
            head_pos.reshape(1, -1),
            pelv_pos.reshape(1, -1),
            tors_pos.reshape(1, -1),
            feet_pos,
            hands_pos,
            knees_pos,
            shlds_pos,
        ], axis=0)
        all_gf = self.sample_field(self.gf, all_poses)
        all_bf = self.sample_field(self.bf, all_poses)
        all_df = self.sample_field(self.sdf, all_poses)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = jp.split(all_gf, [1,2,3,5,7,9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = jp.split(all_bf, [1,2,3,5,7,9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = jp.split(all_df, [1,2,3,5,7,9], axis=0)

        # command is derived from HumanoidPF every step (overrides the inherited random sample_command from G1LocoEnv)
        command = self.compute_cmd_from_rtf(pelvgf.reshape(-1), jp.concat([headgf,feetgf,handsgf], axis=0), jp.concat([headbf,feetbf,handsbf], axis=0))
        state.info["command"] = command.copy()

        # update delay buffer
        update_pf = (state.info["step"] % 5) == 0
        state.info["rng"], odo_key = jax.random.split(state.info["rng"], 2)
        odo_noisy = delay_rootpose_noisy(odo_key, data.qpos[:7])
        # odom_delay = jp.where(update_pf, odo_noisy, state.info["odom_delay"]) 
        odom_delay = jp.where(update_pf, data.qpos[:7], state.info["odom_delay"])
        p_gt = data.qpos[:3]
        q_gt = data.qpos[3:7]
        p_odom = odom_delay[:3]
        q_odom = odom_delay[3:7]
        all_poses_delay = delay_body_pos(p_gt, q_gt, p_odom, q_odom, all_poses)
        all_gf_delay = self.sample_field(self.gf, all_poses_delay)
        all_bf_delay = self.sample_field(self.bf, all_poses_delay)
        all_df_delay = self.sample_field(self.sdf, all_poses_delay)

        # update gait
        self._update_phase(state)
        move_flag = state.info["command"][0]
        all_gf = all_gf * (move_flag[None] > 0.5) / (jp.linalg.norm(all_gf, axis=-1, keepdims=True) + EPS)
        all_bf = all_bf / (jp.linalg.norm(all_bf, axis=-1, keepdims=True) + EPS)

        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = jp.split(all_gf, [1,2,3,5,7,9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = jp.split(all_bf, [1,2,3,5,7,9], axis=0)
        
        all_gf_delay = all_gf_delay * (move_flag[None] > 0.5) / (jp.linalg.norm(all_gf_delay, axis=-1, keepdims=True) + EPS)
        all_bf_delay = all_bf_delay / (jp.linalg.norm(all_bf_delay, axis=-1, keepdims=True) + EPS)
        
        headgf_delay, pelvgf_delay, torsgf_delay, feetgf_delay, handsgf_delay, kneesgf_delay, shldsgf_delay = jp.split(all_gf_delay, [1,2,3,5,7,9], axis=0)
        headbf_delay, pelvbf_delay, torsbf_delay, feetbf_delay, handsbf_delay, kneesbf_delay, shldsbf_delay = jp.split(all_bf_delay, [1,2,3,5,7,9], axis=0)
        headdf_delay, pelvdf_delay, torsdf_delay, feetdf_delay, handsdf_delay, kneesdf_delay, shldsdf_delay = jp.split(all_df_delay, [1,2,3,5,7,9], axis=0)
        command_delay = self.compute_cmd_from_rtf(pelvgf_delay.reshape(-1), jp.concat([headgf_delay,feetgf_delay,handsgf_delay], axis=0), jp.concat([headbf_delay,feetbf_delay,handsbf_delay], axis=0))

        # update info
        state.info["odom_delay"] = odom_delay.copy()
        state.info["headgf_delay"] = headgf_delay.copy()
        state.info["headbf_delay"] = headbf_delay.copy()
        state.info["headdf_delay"] = headdf_delay.copy()
        state.info["pelvgf_delay"] = pelvgf_delay.copy()
        state.info["pelvbf_delay"] = pelvbf_delay.copy()
        state.info["pelvdf_delay"] = pelvdf_delay.copy()
        state.info["torsgf_delay"] = torsgf_delay.copy()
        state.info["torsbf_delay"] = torsbf_delay.copy()
        state.info["torsdf_delay"] = torsdf_delay.copy()
        state.info["feetgf_delay"] = feetgf_delay.copy()
        state.info["feetbf_delay"] = feetbf_delay.copy()
        state.info["feetdf_delay"] = feetdf_delay.copy()
        state.info["handsgf_delay"] = handsgf_delay.copy()
        state.info["handsbf_delay"] = handsbf_delay.copy()
        state.info["handsdf_delay"] = handsdf_delay.copy()
        state.info["kneesgf_delay"] = kneesgf_delay.copy()
        state.info["kneesbf_delay"] = kneesbf_delay.copy()
        state.info["kneesdf_delay"] = kneesdf_delay.copy()
        state.info["shldsgf_delay"] = shldsgf_delay.copy()
        state.info["shldsbf_delay"] = shldsbf_delay.copy()
        state.info["shldsdf_delay"] = shldsdf_delay.copy()
        state.info["command_delay"] = command_delay.copy()

        state.info["headgf"] = headgf.copy()
        state.info["headbf"] = headbf.copy()
        state.info["headdf"] = headdf.copy()
        state.info["pelvgf"] = pelvgf.copy()
        state.info["pelvbf"] = pelvbf.copy()
        state.info["pelvdf"] = pelvdf.copy()
        state.info["torsgf"] = torsgf.copy()
        state.info["torsbf"] = torsbf.copy()
        state.info["torsdf"] = torsdf.copy()
        state.info["feetgf"] = feetgf.copy()
        state.info["feetbf"] = feetbf.copy()
        state.info["feetdf"] = feetdf.copy()
        state.info["handsgf"] = handsgf.copy()
        state.info["handsbf"] = handsbf.copy()
        state.info["handsdf"] = handsdf.copy()
        state.info["kneesgf"] = kneesgf.copy()
        state.info["kneesbf"] = kneesbf.copy()
        state.info["kneesdf"] = kneesdf.copy()
        state.info["shldsgf"] = shldsgf.copy()
        state.info["shldsbf"] = shldsbf.copy()
        state.info["shldsdf"] = shldsdf.copy()

        state.info["head_pos"] = head_pos.copy()
        state.info["head_vel"] = head_vel.copy()
        state.info["pelv_pos"] = pelv_pos.copy()
        state.info["tors_pos"] = tors_pos.copy()
        state.info["feet_pos"] = feet_pos.copy()
        state.info["feet_vel"] = feet_vel.copy()
        state.info["hands_pos"] = hands_pos.copy()
        state.info["hands_vel"] = hands_vel.copy()
        state.info["knees_pos"] = knees_pos.copy()
        state.info["shlds_pos"] = shlds_pos.copy()
        state.info["push"] = push
        state.info["push_step"] += 1
        state.info["step"] += 1


        # update history
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        obs = self._get_obs(data, state.info, feet_contact)
        done = self._get_termination(data, state.info)

        rewards = self._get_reward(data, action, state.info, done, feet_contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)


        timeout = state.info["step"] >= self._config.episode_length
        state.info["step"] = jp.where(done | timeout, 0, state.info["step"])

        state.info["motor_targets"] = jp.where(
            done, self._default_qpos, state.info["motor_targets"]
        )
        # ransom
        state.info["rng"], episode_rng = jax.random.split(state.info["rng"])
        _is_resample = jp.where(
            done,
            self.resample_domain_random_param(episode_rng, state),
            False,
        )

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        state.info["last_joint_vel"] = data.qvel[6:].copy()
        state.info["last_feet_vel"] = data.sensordata[self._foot_linvel_sensor_adr][..., 2]
        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _update_phase(self, state):
        task_mask = state.info["command"][0]
        last_task_mask = state.info["last_command"][0]

        stop_timestep = state.info["stop_timestep"]
        before_stop = stop_timestep > 50
        during_stop = (~before_stop) & (stop_timestep > 0)
        after_stop = (~before_stop) & (~during_stop)
        move2stop = (last_task_mask == 1.0) & (task_mask == 0.0) & before_stop
        stop_timestep = jp.where(move2stop, 50, stop_timestep)
        stop_timestep = jp.where(during_stop, stop_timestep-1, stop_timestep)
        state.info["stop_timestep"] = stop_timestep
        command = jp.where(before_stop, state.info["command"], self._stop_cmd)
        command = command.at[0].set(jp.where(after_stop, 0.0, 1.0))
        state.info["command"] = command

        phase = state.info["phase"] + state.info["phase_dt"]
        phase = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi
        phase = jp.where(after_stop, self._stance_phase, phase)
        state.info["phase"] = phase

        # gait flag
        gait_cycle = jp.cos(phase)
        gait_mask = jp.where(gait_cycle > self._gait_bound, 1, 0)
        gait_mask = jp.where(gait_cycle < -self._gait_bound, -1, gait_mask)
        state.info["gait_mask"] = jp.float32(gait_mask)

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        # Terminates on fall (pelvis tipped or head too low) or SDF collision for all 7 body groups.
        fall_termination = self.get_gravity(data, "pelvis")[2] < 0.0
        fall_termination |= info["head_pos"][2] < 0.7
        contact_termination = collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_foot_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._left_foot_geom_id,
            self._right_shin_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_shin_geom_id,
        )
        contact_termination |= jp.any(info['headdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['pelvdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['torsdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['feetdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['handsdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['kneesdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['shldsdf'] < -self._config.term_collision_threshold)
        contact_termination &= (info["step"] >= 50)
        return fall_termination | contact_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()# | timeout

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        # state (162-dim): noisy proprioception + full HumanoidPF fields for all 7 body groups, nav-frame. No absolute positions. Deployable.
        # privileged_state (224-dim): noiseless state + all absolute body positions/velocities + domain rand params (for critic only).
        # body pose
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel(data, "pelvis")
        # joint
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        gait_phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])

        navi2world_pose = info["navi2world_pose"]
        headgf = info["headgf"].copy()
        headbf = info["headbf"].copy()
        headdf = info["headdf"].copy()
        pelvgf = info["pelvgf"].copy()
        pelvbf = info["pelvbf"].copy()
        pelvdf = info["pelvdf"].copy()
        torsgf = info["torsgf"].copy()
        torsbf = info["torsbf"].copy()
        torsdf = info["torsdf"].copy()
        feetgf = info["feetgf"].copy()
        feetbf = info["feetbf"].copy()
        feetdf = info["feetdf"].copy()
        handsgf= info["handsgf"].copy()
        handsbf= info["handsbf"].copy()
        handsdf= info["handsdf"].copy()
        kneesgf = info["kneesgf"].copy()
        kneesbf = info["kneesbf"].copy()
        kneesdf = info["kneesdf"].copy()
        shldsgf = info["shldsgf"].copy()
        shldsbf = info["shldsbf"].copy()
        shldsdf = info["shldsdf"].copy()
        headgf_delay = info["headgf_delay"].copy()
        headbf_delay = info["headbf_delay"].copy()
        headdf_delay = info["headdf_delay"].copy()
        pelvgf_delay = info["pelvgf_delay"].copy()
        pelvbf_delay = info["pelvbf_delay"].copy()
        pelvdf_delay = info["pelvdf_delay"].copy()
        torsgf_delay = info["torsgf_delay"].copy()
        torsbf_delay = info["torsbf_delay"].copy()
        torsdf_delay = info["torsdf_delay"].copy()
        feetgf_delay = info["feetgf_delay"].copy()
        feetbf_delay = info["feetbf_delay"].copy()
        feetdf_delay = info["feetdf_delay"].copy()
        handsgf_delay = info["handsgf_delay"].copy()
        handsbf_delay = info["handsbf_delay"].copy()
        handsdf_delay = info["handsdf_delay"].copy()
        kneesgf_delay = info["kneesgf_delay"].copy()
        kneesbf_delay = info["kneesbf_delay"].copy()
        kneesdf_delay = info["kneesdf_delay"].copy()        
        shldsgf_delay = info["shldsgf_delay"].copy()
        shldsbf_delay = info["shldsbf_delay"].copy()
        shldsdf_delay = info["shldsdf_delay"].copy()
        head_pos = info["head_pos"].copy()
        head_vel = info["head_vel"].copy()
        pelv_pos = info["pelv_pos"].copy()
        tors_pos = info["tors_pos"].copy()
        feet_pos = info["feet_pos"].copy()
        feet_vel = info["feet_vel"].copy()
        hands_pos = info["hands_pos"].copy()
        hands_vel = info["hands_vel"].copy()
        knees_pos = info["knees_pos"].copy()
        shlds_pos = info["shlds_pos"].copy()
        command = info["command"].copy()
        command_delay = info["command_delay"].copy()

        privileged_state = jp.hstack(
            [
                # noiseless state
                gyro_pelvis,  # (3,)
                gvec_pelvis,  # (3,)
                (joint_angles - self._default_qpos)[self.obs_joint_ids],  # (23,)
                joint_vel[self.obs_joint_ids],  # 23
                info["last_act"],  # num_actions, (12,)
                info["motor_targets"][self.action_joint_ids],  # num_actions, (12,)
                command,  # (4,)
                info["foot_height"],  # 1, ()
                gait_phase,  # (num_foot * 2), (4,)
                # hint state
                linvel_pelvis,  # (3,)
                # pelvgf.reshape(-1),
                headgf.reshape(-1), # (3,)
                headbf.reshape(-1), # (3,)
                headdf.reshape(-1), # (1,)
                pelvgf.reshape(-1), # (3,)
                pelvbf.reshape(-1), # (3,)
                pelvdf.reshape(-1), # (1,)
                torsgf.reshape(-1), # (3,)
                torsbf.reshape(-1), # (3,)
                torsdf.reshape(-1), # (1,)
                feetgf.reshape(-1), # (6,)
                feetbf.reshape(-1), # (6,)
                feetdf.reshape(-1), # (2,)
                handsgf.reshape(-1), # (6,)
                handsbf.reshape(-1), # (6,)
                handsdf.reshape(-1), # (2,)
                kneesgf.reshape(-1), # (6,)
                kneesbf.reshape(-1), # (6,)
                kneesdf.reshape(-1), # (2,)
                shldsgf.reshape(-1), # (6,)
                shldsbf.reshape(-1), # (6,)
                shldsdf.reshape(-1), # (2,)
                head_pos.reshape(-1), # (3,)
                head_vel.reshape(-1), # (3,)
                pelv_pos.reshape(-1), # (3,)
                tors_pos.reshape(-1), # (3,)
                feet_pos.reshape(-1), # (6,)
                feet_vel.reshape(-1), # (6,)
                hands_pos.reshape(-1), # (6,)
                hands_vel.reshape(-1), # (6,)
                knees_pos.reshape(-1), # (6,)
                shlds_pos.reshape(-1), # (6,)
                info["navi_torso_rpy"][:2], # (2,)
                info["gait_mask"],  # (2,)
                feet_contact,  # num_foot, (2,)
                # domain randomization
                info["kp_scale"], # 1, ()
                info["kd_scale"], # 1, ()
                info["rfi_lim_scale"], # (29,)
                # info["delay_steps"],
            ]
        )
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro_pelvis = (
                gyro_pelvis
                + (2 * jax.random.uniform(noise_rng, shape=gyro_pelvis.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gyro
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gvec_pelvis = (
                gvec_pelvis
                + (2 * jax.random.uniform(noise_rng, shape=gvec_pelvis.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gravity
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
                joint_angles
                + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_pos
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
                joint_vel
                + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_vel
        )

        headgf = world_to_navi_vel(navi2world_pose, headgf_delay.reshape(-1, 3))
        headbf = world_to_navi_vel(navi2world_pose, headbf_delay.reshape(-1, 3))
        pelvgf = world_to_navi_vel(navi2world_pose, pelvgf_delay.reshape(-1, 3))
        pelvbf = world_to_navi_vel(navi2world_pose, pelvbf_delay.reshape(-1, 3))
        torsgf = world_to_navi_vel(navi2world_pose, torsgf_delay.reshape(-1, 3))
        torsbf = world_to_navi_vel(navi2world_pose, torsbf_delay.reshape(-1, 3))
        feetgf = world_to_navi_vel(navi2world_pose, feetgf_delay.reshape(-1, 3))
        feetbf = world_to_navi_vel(navi2world_pose, feetbf_delay.reshape(-1, 3))
        handsgf = world_to_navi_vel(navi2world_pose, handsgf_delay.reshape(-1, 3))
        handsbf = world_to_navi_vel(navi2world_pose, handsbf_delay.reshape(-1, 3))
        kneesgf = world_to_navi_vel(navi2world_pose, kneesgf_delay.reshape(-1, 3))
        kneesbf = world_to_navi_vel(navi2world_pose, kneesbf_delay.reshape(-1, 3))
        shldsgf = world_to_navi_vel(navi2world_pose, shldsgf_delay.reshape(-1, 3))
        shldsbf = world_to_navi_vel(navi2world_pose, shldsbf_delay.reshape(-1, 3))
        command=command.at[-3:].set(world_to_navi_vel(navi2world_pose, command_delay[-3:].reshape(-1, 3)).reshape(-1))
        command=command.at[-1].set(0)

        headbf = headbf * (headdf_delay < 0.5)
        headdf = jp.clip(headdf_delay, -1.0, 0.5)
        pelvbf = pelvbf * (pelvdf_delay < 0.5)
        pelvdf = jp.clip(pelvdf_delay, -1.0, 0.5)
        torsbf = torsbf * (torsdf_delay < 0.5)
        torsdf = jp.clip(torsdf_delay, -1.0, 0.5)
        feetbf = feetbf * (feetdf_delay < 0.5)
        feetdf = jp.clip(feetdf_delay, -1.0, 0.5)
        handsbf = handsbf * (handsdf_delay < 0.5)
        handsdf = jp.clip(handsdf_delay, -1.0, 0.5)
        kneesbf = kneesbf * (kneesdf_delay < 0.5)
        kneesdf = jp.clip(kneesdf_delay, -1.0, 0.5)
        shldsbf = shldsbf * (shldsdf_delay < 0.5)
        shldsdf = jp.clip(shldsdf_delay, -1.0, 0.5)

        pf = jp.hstack(
            [
                headgf.reshape(-1),
                headbf.reshape(-1),
                headdf.reshape(-1),
                pelvgf.reshape(-1),
                pelvbf.reshape(-1),
                pelvdf.reshape(-1),
                torsgf.reshape(-1),
                torsbf.reshape(-1),
                torsdf.reshape(-1),
                feetgf.reshape(-1),
                feetbf.reshape(-1),
                feetdf.reshape(-1),
                handsgf.reshape(-1),
                handsbf.reshape(-1),
                handsdf.reshape(-1),
                kneesgf.reshape(-1),
                kneesbf.reshape(-1),
                kneesdf.reshape(-1),
                shldsgf.reshape(-1),
                shldsbf.reshape(-1),
                shldsdf.reshape(-1),
            ]
        )   # (77,)

        state = jp.hstack(
            [
                # noiseless state
                noisy_gyro_pelvis,  # (3,)
                noisy_gvec_pelvis,  # (3,)
                # joint state
                (noisy_joint_angles - self._default_qpos)[self.obs_joint_ids],  # (23,)
                noisy_joint_vel[self.obs_joint_ids],  # (23,)
                info["last_act"],  # num_actions, (12,)
                info["motor_targets"][self.action_joint_ids],  # num_actions, (12,)
                command,  # (4,)
                info["foot_height"],  # 1, ()
                gait_phase,  # (num_foot * 2), (4,)
                pf, # (77,)
            ]   # (162,)
        )

        # Nan to 0
        state = jp.nan_to_num(state)
        privileged_state = jp.nan_to_num(privileged_state)

        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            done: jax.Array,
            feet_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        move_flag = info["command"][0]
        cmd_vel = info["command"][1:].copy()  # [x, y, yaw]

        reward_dict = {
            # behavior reward
            "tracking_orientation": self._reward_orientation(
                info["navi_pelvis_rpy"], info["navi_torso_rpy"], info["head_pos"][2] > (self._config.torso_height[1] + 0.1)
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
            # energy reward
            "joint_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "joint_torque": self._cost_torque(data.actuator_force),
            "smoothness_joint": self._cost_smoothness_joint(data, info["last_joint_vel"]),
            "smoothness_action": self._cost_smoothness_action(action, info["last_act"], info["last_last_act"]),
            # field
            "headgf": self._re_gf0(info["headgf"], info["head_vel"], info["headdf"], (move_flag[None]<0.5) | (info["head_pos"][...,0] > 1.5), tau=0.5),
            "feetgf": self._re_gf0(info["feetgf"], info["feet_vel"], info["feetdf"], (move_flag[None]<0.5) | (info["gait_mask"] == 1) | (info["feet_pos"][...,0] > 1.5), tau=0.3),
            "handsgf": self._re_gf0(info["handsgf"], info["hands_vel"], info["handsdf"], (move_flag[None]<0.5) | (info["hands_pos"][...,0] > 1.5), tau=0.5),
            "headdf": self._re_sdf(info["headdf"]),
            "feetdf": self._re_sdf(info["feetdf"]),
            "handsdf": self._re_sdf(info["handsdf"]), # NOTE
            "kneesdf": self._re_sdf(info["kneesdf"]),
            "shldsdf": self._re_sdf(info["shldsdf"]),

        }
        for k, v in reward_dict.items():
            # replace NaN with 0
            reward_dict[k] = jp.where(jp.isnan(v), 0.0, v)

        return reward_dict

    def _re_gf0(self, gf_vel: jax.Array, lin_vel: jax.Array, sdf: jax.Array, crossed: jax.Array, tau = 0.3) -> jax.Array:
        """
        Reward a body part for moving in the direction recommended by the guidance field (GF),
        but only when it is within tau meters of an obstacle (proximity-gated).

        gf_vel:  (N, 3) — guidance field vectors at N body site positions; points toward free/safe space
        lin_vel: (N, 3) — actual world-frame velocity of each body site (finite diff of site_xpos)
        sdf:     (N, 1) — signed distance from each body site to the nearest obstacle surface
        crossed: (N,) bool — True when reward should use a fixed fallback value instead of cos_align:
                             triggered when robot is stopped (move_flag < 0.5),
                             body part is past obstacle zone (pos_x > 1.5m),
                             or foot is in stance (gait_mask == 1, feet only)
        tau:     float — activation radius in meters (0.3 for feet, 0.5 for head/hands)
        returns: scalar — mean reward over N sites, in range (-5, 5)
        """
        eps = 1e-6

        k_window    = 40.0  # sigmoid sharpness: controls how abruptly the reward activates near tau
        alpha_align = 5.0   # reward magnitude scale

        # normalize to unit vectors so only direction matters, not speed or field magnitude
        g_norm = gf_vel / (jp.linalg.norm(gf_vel, axis=-1, keepdims=True) + eps)   # (N, 3)
        v_norm = lin_vel / (jp.linalg.norm(lin_vel, axis=-1, keepdims=True) + eps)  # (N, 3)
        cos_align = jp.sum(g_norm * v_norm, axis=-1)  # (N,) cosine similarity in [-1, 1]: +1 = perfect alignment, -1 = moving into obstacle

        sdf_flat = sdf.reshape(-1)  # (N,) flatten (N,1) → 1D
        # proximity gate: ≈1 when sdf < tau (near obstacle), ≈0 when sdf > tau (far away)
        # k_window=40 makes the sigmoid transition sharp over ~0.05m
        window = jax.nn.sigmoid(k_window * (tau - sdf_flat))  # (N,)

        reward_near = window * (alpha_align * cos_align)              # (N,) proximity-weighted alignment reward
        reward_near = jp.where(crossed, alpha_align * 0.8, reward_near)  # (N,) fallback: fixed reward 4.0 for degenerate cases

        return jp.mean(reward_near)  # (scalar) average over N body sites

    def _re_sdf(self, sdf: jax.Array, sdf_safe = 0.05) -> jax.Array:
        """
        Penalize a body part for being too close to or inside an obstacle.
        Penalty activates smoothly when the SDF value drops below sdf_safe (0.05m safety margin).

        sdf:      (N, 1) — signed distance from N body sites to the nearest obstacle surface
                           positive = outside (safe), negative = inside obstacle
        returns:  scalar <= 0 — negative reward (penalty), averaged over N sites
        """
        beta_inside = 0.02       # softplus smoothing width (smaller = sharper transition at sdf_safe boundary)
        pen_inside_scale = 20.0  # penalty magnitude scale

        sdf_flat = sdf.reshape(-1)  # (N,) flatten (N,1) → 1D for element-wise ops

        # softplus((sdf_safe - sdf) / beta): ≈0 when sdf >> sdf_safe (far from obstacle),
        # rises smoothly as sdf approaches sdf_safe, large when sdf < 0 (inside obstacle)
        pen_inside = jax.nn.softplus((sdf_safe - sdf_flat) / beta_inside)  # (N,)

        penalty = pen_inside_scale * pen_inside  # (N,) scale up the raw softplus value

        re_gf = -penalty          # (N,) negate: reward is negative (cost)
        return jp.mean(re_gf)     # (scalar) average penalty over all N body sites 

    def _reward_tracking_root_field(self, cmd_vel: jax.Array, local_lin_vel: jax.Array) -> jax.Array:
        """
        Reward the robot to walk at the commanded velocity.
        cmd_vel:       (3,) — [vx (m/s), vy (m/s), yaw (rad/s)] commanded velocity (move_flag already stripped)
        local_lin_vel: (3,) — [vx, vy, vz] actual pelvis velocity in local frame
        returns:       scalar in (0, 1] — 1.0 when perfectly matching, decays exponentially with error
        """
        lin_vel_error = jp.sum(jp.square(cmd_vel[:2] - local_lin_vel[:2]))  # (scalar) squared error on vx, vy only; vz not commanded
        return jp.exp(-4.0 * lin_vel_error)
    
    def _cost_body_motion(
        self, local_lin_vel, local_ang_vel: jax.Array, cmd_vel: jax.Array
    ) -> jax.Array:
        """
        Penalize unwanted body motion — lateral drift and rocking — that isn't along the commanded direction.
        local_lin_vel: (3,) — [vx, vy, vz] actual pelvis linear velocity in local frame
        local_ang_vel: (3,) — [roll_rate, pitch_rate, yaw_rate] actual pelvis angular velocity in local frame
        cmd_vel:       (3,) — [vx, vy, yaw] commanded velocity (move_flag already stripped)
        returns:       scalar >= 0 (cost, negated by scale -0.5 in reward config)
        """
        cmd_xy = cmd_vel[:2]  # (2,) commanded direction in xy plane
        cmd_norm = jp.linalg.norm(cmd_xy)  # (scalar) magnitude of commanded velocity
        is_zero_cmd = jp.isclose(cmd_norm, 0.0)  # (scalar bool) True if standing still command
        cmd_dir = jp.where(is_zero_cmd, jp.zeros_like(cmd_xy), cmd_xy / cmd_norm)  # (2,) unit vector in commanded direction; zero if no command

        lin_xy = local_lin_vel[:2]  # (2,) actual xy velocity
        lin_xy_orth = lin_xy - jp.dot(lin_xy, cmd_dir) * cmd_dir  # (2,) component of velocity perpendicular to commanded direction (sideways drift)
        cost_lin_xy_orth = jp.where(is_zero_cmd, 0.0, jp.sum(jp.square(lin_xy_orth)))  # (scalar) squared lateral drift; 0 if standing still command

        cost = (
            1.2 * cost_lin_xy_orth            # penalize sideways drift (strongest weight)
            + 0.4 * jp.abs(local_ang_vel[0])  # penalize roll rate (rocking left/right)
            + 0.4 * jp.abs(local_ang_vel[1])  # penalize pitch rate (rocking forward/backward)
        )
        return cost

    def _reward_orientation(
        self, pelvis_rpy: jax.Array, torso_rpy: jax.Array, idle_mask: jax.Array
    ) -> jax.Array:
        """
        Reward the robot for staying upright
        pelvis_rpy and torso_rpy have shape (3,), idle_mask is a boolean value indicating whether or not the robot's in the idle mode 
        """
        err_roll = jp.abs(pelvis_rpy[0]) + jp.abs(torso_rpy[0]) # encourage pelvis roll and torso roll to be near 0: robot staying upright, no left/right tilt
        err_pitch_dire = jp.abs(jp.clip(torso_rpy[1], -np.pi, 0.0)) # only penalize backward pitch (leaning back)
        err_pitch_idle = idle_mask * jp.abs(torso_rpy[1])   # when standing tall, penalize any pitch; when ducking, this is 0
        err_ori = err_roll + err_pitch_dire + err_pitch_idle
        rew = jp.exp(-0.5 * err_ori) - err_pitch_dire
        return rew
    
    def _cost_foot_far(self, data: mjx.Data) -> jax.Array:  # currently not being used (scale=0 in reward config)
        """
        Penalize feet being too close together (< 0.35m) — narrow stance is unstable.
        Note: despite the name 'foot_far', this penalizes feet too CLOSE, not too far.
        data:    mjx.Data — full MuJoCo sim state
        returns: scalar >= 0 — linear penalty proportional to how much closer than 0.35m; 0 if feet are far enough
        """
        foot_pos = data.site_xpos[self._feet_site_id]               # (2, 3) world positions of left and right foot sites
        foot_distance = jp.linalg.norm(foot_pos[0] - foot_pos[1])  # (scalar) 3D distance between feet
        foot_spread_penalty = jp.where(
            foot_distance < 0.35,
            (0.35 - foot_distance),  # linear penalty: larger when feet are closer
            0.0
        )
        return foot_spread_penalty

    def _cost_straight_knee(self, knee_pos) -> jax.Array:
        """
        Penalize fully straight or hyperextended knees (angle < 0.1 rad ≈ 5.7°).
        Slightly bent knees are required for shock absorption and stability;
        locking knees can damage real hardware.
        knee_pos: (2,) — current knee joint angles in radians [left_knee, right_knee],
                  taken from data.qpos at knee joint indices (always positive for G1, knee only bends forward)
        returns:  scalar >= 0 — sum of linear penalties for each knee below 0.1 rad
        """
        penalty = jp.clip(0.1 - knee_pos, min=0.0)  # (2,) positive when knee_pos < 0.1 rad, 0 otherwise
        cost = jp.sum(penalty)                        # (scalar) sum over both knees
        return cost

    def _cost_foot_balance(
        self, data: mjx.Data, navi2world_pose: jax.Array, task_mask: jax.Array
    ):
        """
        Penalize two things: pelvis COM not centered between feet, and feet too close together.
        data:           mjx.Data — full MuJoCo sim state (qpos, qvel, site_xpos, subtree_com, etc.)
        navi2world_pose: (4, 4) — homogeneous transform from navigation frame to world frame
        task_mask:      (2,) — move_flag per foot (unused currently, stance_mask is computed but not applied)
        returns:        scalar >= 0
        """
        stance_mask = 1 - task_mask  # (2,) unused currently

        # Build homogeneous positions (xyz + 1) for pelvis COM and both feet in world frame
        sup2world_pos_h = jp.ones((3, 4))                                          # (3, 4) last column is 1 for homogeneous coords
        sup2world_pos_h = sup2world_pos_h.at[0, :3].set(
            data.subtree_com[self.body_id_pelvis]                                  # (3,) center of mass of pelvis subtree in world frame
        )
        sup2world_pos_h = sup2world_pos_h.at[1, :3].set(
            data.site_xpos[self._feet_site_id[0]]                                  # (3,) left foot world position
        )
        sup2world_pos_h = sup2world_pos_h.at[2, :3].set(
            data.site_xpos[self._feet_site_id[1]]                                  # (3,) right foot world position
        )
        sup2navi_pos = (jp.linalg.inv(navi2world_pose) @ sup2world_pos_h.T).T[:, :3]  # (3, 3) all three positions in navigation frame

        foot2com_err = sup2navi_pos[1:] - sup2navi_pos[0]          # (2, 3) each foot position relative to pelvis COM in navi frame
        foot_center = foot2com_err[0, :2] + foot2com_err[1, :2]    # (2,) xy only: sum of (left_foot - pelvis) + (right_foot - pelvis); = 0 when pelvis is centered
        cost_support = jp.sum(jp.square(foot_center))               # (scalar) penalize pelvis offset from foot midpoint

        # Additional penalty when feet are too close together (< 0.35m) — narrow stance is unstable
        foot_pos = data.site_xpos[self._feet_site_id]               # (2, 3) foot positions in world frame
        foot_distance = jp.linalg.norm(foot_pos[0] - foot_pos[1])  # (scalar) distance between feet
        foot_spread_penalty = jp.where(
            foot_distance < 0.35,
            (0.35 - foot_distance),                                  # linear penalty proportional to how much closer than 0.35m
            0.0
        ) * 10
        cost_support = cost_support * (1 + foot_spread_penalty)     # amplify centering cost when feet are too close
        return cost_support
    
    def _cost_smoothness_action(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        """
        Penalize large, sudden, or jerky policy action outputs using three orders of finite differences.
        act:           (12,) — current policy action output
        last_act:      (12,) — action from previous step, stored in info
        last_last_act: (12,) — action from two steps ago, stored in info
        returns:       scalar >= 0
        """
        smooth_0th = jp.square(act)                              # (12,) penalize large action magnitude — stay near default pose
        smooth_1st = jp.square(act - last_act)                   # (12,) penalize sudden action changes (1st difference ≈ action velocity)
        smooth_2nd = jp.square(act - 2 * last_act + last_last_act)  # (12,) penalize change in action change rate (2nd difference ≈ action acceleration/jerk)
        cost = jp.sum(smooth_0th + smooth_1st + smooth_2nd)      # (scalar) sum over all 12 action dims
        return cost

    def _reward_body_rotation(self, data: mjx.Data, cmd_vel: jax.Array, navi2world_rot: jax.Array) -> jax.Array:
        """
        Reward leg bodies for staying properly aligned — no sideways roll, no toe-out/in when going straight.
        cmd_vel:       (3,) — [vx, vy, yaw]
        navi2world_rot: (3,3) — rotation matrix from navigation frame to world frame
        returns:       scalar in (0, 1]
        """
        cmd_max = jp.abs(self._config.ang_vel_yaw[1]) + 1e-6
        cmd_decay = jp.clip((cmd_max - jp.abs(cmd_vel[2])) / cmd_max, 0.0, 1.0) ** 2  # (scalar) 1.0 when going straight, 0.0 when turning at max yaw; relaxes yaw penalty during turns

        legs2world_rot = jp.concat([data.xmat[self.body_ids_left_leg], data.xmat[self.body_ids_right_leg]])  # (N, 3, 3) rotation matrices of all leg body segments
        legs2navi_rot = navi2world_rot.T[None] @ legs2world_rot  # (N, 3, 3) leg rotations in navigation frame

        axis_roll_err = jp.mean(jp.abs(legs2navi_rot[:, 2, 1]))               # (scalar) R[2,1]: how much leg lateral axis projects onto world up — measures leg roll (bow-legged)
        axis_yaw_err = jp.mean(cmd_decay * jp.abs(legs2navi_rot[:, 0, 1]))    # (scalar) R[0,1]: how much leg lateral axis projects onto forward axis — measures toe-out/in; only penalized when going straight
        axis_rew = jp.exp(-5.0 * (axis_roll_err + axis_yaw_err))              # peaks at 1.0 when both errors are 0
        return axis_rew
        

    def world_to_grid(self, pos):
        rel = pos - self.pf_origin
        idx = rel / self.dx
        return idx

    def sample_field(self, field, pos):
        idx = self.world_to_grid(pos)                  # (N,3)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]     # (N,)

        x = jp.clip(x, 0, self.Nx - 2)
        y = jp.clip(y, 0, self.Ny - 2)
        z = jp.clip(z, 0, self.Nz - 2)

        xi = jp.floor(x).astype(jp.int32)             # (N,)
        yi = jp.floor(y).astype(jp.int32)
        zi = jp.floor(z).astype(jp.int32)
        xd = x - xi                                    # (N,)
        yd = y - yi
        zd = z - zi

        offsets = jp.array([
            [0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1]
        ], dtype=jp.int32)                             # (8,3)

        base = jp.stack([xi, yi, zi], axis=1)         # (N,3)
        corners = base[:, None, :] + offsets[None, :, :]     # (N,8,3)

        vals = field[corners[..., 0], corners[..., 1], corners[..., 2], :]  # (N,8,C)

        wx = jp.stack([1.0 - xd, xd], axis=1)         # (N,2)
        wy = jp.stack([1.0 - yd, yd], axis=1)         # (N,2)
        wz = jp.stack([1.0 - zd, zd], axis=1)         # (N,2)

        w = (wx[:, :, None, None] *
            wy[:, None, :, None] *
            wz[:, None, None, :]).reshape(-1, 8)      # (N,8)

        out = jp.einsum('ne,nec->nc', w, vals)        # (N,C)
        return out

    def compute_cmd_from_rtf(self, rtf, cgf, cbf):
        """
        Auto-compute velocity command from HumanoidPF fields (replaces joystick input during training).
        Starts from pelvis GF direction, then corrects it so each body part satisfies its obstacle constraint.

        rtf: (3,)   — guidance field at pelvis position (root trajectory field); base direction to move
        cgf: (M, 3) — guidance field vectors for M body sites (head + feet + hands, M=5)
        cbf: (M, 3) — boundary field vectors for M body sites (obstacle surface normals)
        returns: (4,) — command [move_flag, vx, vy, yaw]
        """
        v = rtf[:2] * 0.7  # (2,) initial 2D velocity from pelvis GF, scaled down

        b_hat = cbf[:, :2] / (jp.linalg.norm(cbf[:, :2], axis=-1, keepdims=True) + 1e-9)  # (M,2) normalized obstacle surface normals

        Ls = jp.sum(b_hat * cgf[:, :2], axis=-1)  # (M,) how much GF pushes along each constraint direction
        bv = jp.sum(b_hat * v, axis=-1)            # (M,) how much current velocity satisfies each constraint

        diff = (Ls - bv)[:, None] / (jp.sum(b_hat * b_hat, axis=-1, keepdims=True) + 1e-9)
        delta = diff * b_hat  # (M,2) correction vector per body part along its constraint direction

        mask = (Ls > bv)[:, None]           # (M,1) only correct when constraint is not yet satisfied
        delta = jp.where(mask, delta, 0.0)  # (M,2) zero out corrections that are already satisfied

        v_new = v + jp.mean(delta, axis=0)  # (2,) apply averaged correction from all M body parts

        command = jp.hstack([1.0, v_new[0], v_new[1], 0.0]) * 0.75  # (4,) pack as [move_flag=0.75, vx, vy, yaw=0]

        small_cond = jp.linalg.norm(command[1:4]) < 0.2
        # move_flag: 0.75 = moving, 0.0 = stop (policy treats < 0.5 as stopped)
        command = jp.where(small_cond, self._stop_cmd, command)  # if velocity too small, replace with [0,0,0,0]
        return command
    
@cat_ppo.registry.register("G1Cat", "command_to_reference_fn")
def command_to_reference(env_config: config_dict.ConfigDict, command: jax.Array):
    """
    Unpack the 4-dim command vector into a reference state dict.
    Registered for future use (e.g. visualization tools); not called during training.

    command: (4,) — [move_flag, vx, vy, yaw] produced by compute_cmd_from_rtf
    returns: dict of reference quantities (base_height, base_gvec, base_lin_vel, base_ang_vel)
    """
    command_vel = command[1:]  # (3,) strip move_flag → [vx, vy, yaw]
    base_height = env_config.reward_config.base_height_target  # target torso height (fixed, not from command)
    base_gvec = np.array([0.0, 0.0, 1.0])                      # upright orientation target (gravity points down)
    base_lin_vel = np.array([command_vel[0], command_vel[1], 0.0])  # commanded horizontal velocity, z=0
    base_ang_vel = np.array([0.0, 0.0, command_vel[2]])             # commanded yaw rotation only

    return {
        "base_height": base_height,
        "base_gvec": base_gvec,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
    }
