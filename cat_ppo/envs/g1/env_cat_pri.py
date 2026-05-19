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

ENABLE_RANDOMIZE = False


def g1_loco_task_config() -> config_dict.ConfigDict:
    from cat_ppo.envs.g1.randomize import domain_randomize

    env_config = config_dict.create(
        task_type="flat_terrain",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=15,
        num_obs=175,
        num_pri=209,
        num_act=12,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        gait_config=config_dict.create(
            gait_bound=0.6,
            freq_range=[1.3, 1.5],
            foot_height_range=[0.05, 0.05],
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
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # behavior reward
                tracking_orientation=2.0,
                tracking_root_field=1.0,
                body_motion=-0.5,
                body_rotation=1.0,
                feet_rotation=1.0,
                foot_contact=-1.0,
                foot_clearance=-15.0, 
                foot_slip=-0.5, 
                foot_balance=-10,
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
    )

    policy_config = config_dict.create(
        num_timesteps=5_000_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        madrona_backend=False,
        augment_pixels=False,
        # environment wrapper
        num_envs=32768,  # 8192(256*32), 16384(512*32), 32768(1024*32)
        episode_length=1000,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=domain_randomize if ENABLE_RANDOMIZE else None,
        # ppo params
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
        # eval
        num_evals=6,
        eval_env=None,
        num_eval_envs=0,
        deterministic_eval=False,
        # training metrics
        log_training_metrics=True,
        training_metrics_steps=int(1e6),  # 1M
        # callbacks
        progress_fn=lambda *args: None,
        # policy_params_fn=lambda *args: None,
        # checkpointing
        save_checkpoint_path=None,
        restore_checkpoint_path=None, 
        restore_params=None,
        restore_value_fn=False,
    )

    # vel: move_flag[0|1], x[m], y[m], yaw[rad]
    eval_config = config_dict.create(
        duration=50.0,
        command_waypoints=np.array(
            [
                [0, 0.0, 0.0, 0.0],
            ]
        ),
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
        eval_config=eval_config,
    )
    return config

cat_ppo.registry.register("G1CatPri", "config")(g1_loco_task_config())

from cat_ppo.envs.g1.env_cat import G1CatEnv

@cat_ppo.registry.register("G1CatPri", "train_env_class")
class G1CatPriEnv(G1CatEnv):
    """G1Cat dynamics with the cat privileged observation exposed to policy and value."""
    
    def _get_obs(
        self,
        data: mjx.Data,
        info: dict,
        feet_contact: jax.Array,
    ) -> mjx_env.Observation:
        obs = super()._get_obs(self, data, info, feet_contact)
        privileged_state = jp.nan_to_num(obs["privileged_state"])
        return {"state": privileged_state, "privileged_state": privileged_state}


@cat_ppo.registry.register("G1CatPri", "command_to_reference_fn")
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
