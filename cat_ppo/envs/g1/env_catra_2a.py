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
"""Two-agent CaTra: lower body (legs) and upper body (arms) use separate actor/critic networks.

Both actors share the same actor observation; both critics share the same privileged
observation. Termination is shared. The environment emits a length-2 reward vector
[lower, upper]: SHARED reward terms go to both agents, LOWER-only and UPPER-only terms
go to the corresponding agent. Whole-joint regularizers (joint_torque, joint_limits,
smoothness_joint, smoothness, smoothness_action) are split per joint-group so each agent
only pays for the joints / action dims it controls.

`G1CaTra2A`    — deployable actor obs (239-dim noisy `state`), DR on.
`G1CaTra2APri` — privileged actor obs (302-dim `priv[:-31]` slice), DR off; teacher variant.

Action ordering: CATRA_ACTION_JOINT_NAMES is legs(0-11), waist(12-14), arms(15-22). The two
policies output a_lower(15 = legs+waist) and a_upper(8 = arms); training concatenates
[a_lower, a_upper] -> 23-dim action. Waist is part of the lower group.
"""

import jax
import jax.numpy as jp
from mujoco import mjx

import cat_ppo
from cat_ppo.envs.g1 import constants as consts
from cat_ppo.envs.g1.env_catra import (
    G1CaTraEnv,
    g1_catra_task_config,
    command_to_reference,
    NUM_ROBOT_JOINTS,
)
from cat_ppo.envs.g1.env_catra_pri import _DR_TAIL_DIMS

# Regularizers that act on all joints / all action dims and must be split per joint-group.
_SPLIT_REGULARIZERS = ("joint_torque", "joint_limits", "smoothness_joint", "smoothness", "smoothness_action")

# Reward grouping (excluding the split regularizers, which are classified by their _lower/_upper suffix).
# SHARED terms go to BOTH agents; LOWER/UPPER terms go to the corresponding agent only.
_SHARED_KEYS = frozenset({
    "lift", "lift_carry", "box_pillar_contact", "box_vertical", "hold_stable",
    "box_yaw_stable", "box_centering", "box_upright", "box_upright_carry", "boxdf", "boxgf",
    # hands/shoulders are articulated by the arms (upper) but their world position also
    # depends on torso/pelvis pose driven by the legs (lower) -> obstacle terms are shared.
    "handsgf", "handsdf", "shldsdf",
})
_UPPER_KEYS = frozenset({
    "reach", "reach_carry", "hand_contact", "hand_contact_carry",
    "grasp_symmetry", "grasp_symmetry_carry", "palm_orient", "palm_orient_carry",
    "hands_level", "hands_level_carry", "upper_body_align",
})
_LOWER_KEYS = frozenset({
    "foot_contact", "foot_contact_trav", "foot_slip", "foot_slip_trav",
    "straight_knee", "straight_knee_trav", "foot_balance", "foot_balance_trav",
    "body_rotation", "foot_clearance", "foot_far", "feet_rotation", "bent_knee_trav",
    "hip_yaw_lim", "feet_apart",
    "feetgf", "feetdf", "kneesdf",
    # head is driven by torso/pelvis pose (waist frozen, arms cannot move it) -> lower-only.
    "headgf", "headdf",
    # root / locomotion (assigned to the lower agent per design decision)
    "tracking_root_field", "tracking_orientation", "body_motion",
    "forward_progress", "base_height", "upright",
})


def _split_reward_scales(scales):
    """Return a new scales ConfigDict with the whole-joint regularizers replaced by per-group
    (_lower / _upper) variants carrying the same scale value."""
    new = scales.copy_and_resolve_references()
    for key in _SPLIT_REGULARIZERS:
        val = float(new[key])
        del new[key]
        new[f"{key}_lower"] = val
        new[f"{key}_upper"] = val
    return new


def g1_catra_2a_task_config():
    cfg = g1_catra_task_config()
    cfg.env_config.reward_config.scales = _split_reward_scales(cfg.env_config.reward_config.scales)
    cfg.env_config.num_act_lower = 12  # TEMP: 12 legs (waist removed; would be 15 with waist)
    cfg.env_config.num_act_upper = 8
    return cfg


def g1_catra_2a_pri_task_config():
    cfg = g1_catra_2a_task_config()
    cfg.env_config.num_obs = cfg.env_config.num_pri - _DR_TAIL_DIMS  # 302
    cfg.policy_config.randomization_fn = None
    return cfg


cat_ppo.registry.register("G1CaTra2A", "config")(g1_catra_2a_task_config())
cat_ppo.registry.register("G1CaTra2A", "command_to_reference_fn")(command_to_reference)
cat_ppo.registry.register("G1CaTra2APri", "config")(g1_catra_2a_pri_task_config())
cat_ppo.registry.register("G1CaTra2APri", "command_to_reference_fn")(command_to_reference)


@cat_ppo.registry.register("G1CaTra2A", "train_env_class")
class G1CaTra2AEnv(G1CaTraEnv):
    """G1CaTra with the action+reward split into a lower (legs) and upper (arms) agent."""

    def _post_init_catra(self) -> None:
        super()._post_init_catra()

        # Identify arm action dims by joint name; confirm legs precede arms contiguously.
        arm_dims = [i for i, n in enumerate(consts.CATRA_ACTION_JOINT_NAMES)
                    if ("shoulder" in n) or ("elbow" in n)]
        leg_dims = [i for i, n in enumerate(consts.CATRA_ACTION_JOINT_NAMES)
                    if i not in arm_dims]
        n_lower = len(leg_dims)
        n_upper = len(arm_dims)
        assert leg_dims == list(range(n_lower)), "legs must be the first action dims"
        assert arm_dims == list(range(n_lower, n_lower + n_upper)), "arms must follow legs contiguously"
        self._n_lower = n_lower            # 12
        self._n_upper = n_upper            # 8
        self._leg_slice = slice(0, n_lower)
        self._arm_slice = slice(n_lower, n_lower + n_upper)

        # Actuator ids per group (subset of action_joint_ids; index 29-long joint arrays).
        self._leg_act_ids = self.action_joint_ids[self._leg_slice]
        self._arm_act_ids = self.action_joint_ids[self._arm_slice]

        # Build per-agent reward-key sets from the actual config scale keys; assert full coverage.
        lower, upper = set(), set()
        for k in self._config.reward_config.scales.keys():
            if k.endswith("_lower"):
                lower.add(k)
            elif k.endswith("_upper"):
                upper.add(k)
            elif k in _SHARED_KEYS:
                lower.add(k); upper.add(k)
            elif k in _LOWER_KEYS:
                lower.add(k)
            elif k in _UPPER_KEYS:
                upper.add(k)
            else:
                raise ValueError(f"[G1CaTra2A] reward key '{k}' is not classified upper/lower/shared")
        self._lower_reward_keys = frozenset(lower)
        self._upper_reward_keys = frozenset(upper)

    @property
    def action_size_lower(self) -> int:
        return self._n_lower

    @property
    def action_size_upper(self) -> int:
        return self._n_upper

    def _split_regularizers(self, data: mjx.Data, action: jax.Array, info: dict) -> dict:
        leg, arm = self._leg_act_ids, self._arm_act_ids

        # joint_torque: sum of squared actuator torque, per group
        tau = data.actuator_force
        jt_lower = jp.sum(jp.square(tau[leg]))
        jt_upper = jp.sum(jp.square(tau[arm]))

        # joint_limits: soft-limit violation, per group
        qpos = data.qpos[7:7 + NUM_ROBOT_JOINTS]
        def _limit(ids):
            q = qpos[ids]
            v = -jp.clip(q - self._soft_lowers[ids], None, 0.0) + jp.clip(q - self._soft_uppers[ids], 0.0, None)
            return jp.sum(v)
        jl_lower = _limit(leg)
        jl_upper = _limit(arm)

        # smoothness_joint: 0.01*vel^2 + acc^2, per group
        qvel = data.qvel[6:6 + NUM_ROBOT_JOINTS]
        last_qvel = info["last_joint_vel"]
        def _smj(ids):
            v = qvel[ids]
            acc = (last_qvel[ids] - v) / self.dt
            return jp.sum(0.01 * jp.square(v) + jp.square(acc))
        smj_lower = _smj(leg)
        smj_upper = _smj(arm)

        # smoothness (stage 1): -sum((act - last_act)^2), per action-dim group
        act, la, lla = action, info["last_act"], info["last_last_act"]
        def _sm1(sl):
            return -jp.sum((act[sl] - la[sl]) ** 2)
        sm_lower = _sm1(self._leg_slice)
        sm_upper = _sm1(self._arm_slice)

        # smoothness_action (stage 2): sum(sq(1st diff) + sq(2nd diff)), per action-dim group
        def _sma(sl):
            return jp.sum(jp.square(act[sl] - la[sl]) + jp.square(act[sl] - 2 * la[sl] + lla[sl]))
        sma_lower = _sma(self._leg_slice)
        sma_upper = _sma(self._arm_slice)

        return {
            "joint_torque_lower": jt_lower, "joint_torque_upper": jt_upper,
            "joint_limits_lower": jl_lower, "joint_limits_upper": jl_upper,
            "smoothness_joint_lower": smj_lower, "smoothness_joint_upper": smj_upper,
            "smoothness_lower": sm_lower, "smoothness_upper": sm_upper,
            "smoothness_action_lower": sma_lower, "smoothness_action_upper": sma_upper,
        }

    def _get_reward(self, data, action, info, done, feet_contact):
        rewards = super()._get_reward(data, action, info, done, feet_contact)
        for key in _SPLIT_REGULARIZERS:
            del rewards[key]
        rewards.update(self._split_regularizers(data, action, info))
        return rewards

    def _agent_rewards(self, rewards: dict, in_stage2: jax.Array):
        """Return the (lower, upper) scalar rewards. SHARED terms count toward both;
        each is scaled by dt and stage-2 clipped exactly like the single-agent reward."""
        lower_raw = sum(v for k, v in rewards.items() if k in self._lower_reward_keys) * self.dt
        upper_raw = sum(v for k, v in rewards.items() if k in self._upper_reward_keys) * self.dt
        lower = jp.where(in_stage2, jp.clip(lower_raw, 0.0, 10000.0), lower_raw)
        upper = jp.where(in_stage2, jp.clip(upper_raw, 0.0, 10000.0), upper_raw)
        return lower, upper

    def _extra_reward_info(self):
        # Per-agent rewards carried in info (read by the trainer via extra_fields).
        return {"reward_lower": jp.zeros(()), "reward_upper": jp.zeros(())}

    def _record_agent_rewards(self, info: dict, rewards: dict, in_stage2: jax.Array) -> None:
        lower, upper = self._agent_rewards(rewards, in_stage2)
        info["reward_lower"] = lower
        info["reward_upper"] = upper

    def _assemble_reward(self, rewards: dict, in_stage2: jax.Array) -> jax.Array:
        # Scalar reward for brax wrappers/metrics; per-agent split lives in info.
        lower, upper = self._agent_rewards(rewards, in_stage2)
        return lower + upper


@cat_ppo.registry.register("G1CaTra2APri", "train_env_class")
class G1CaTra2APriEnv(G1CaTra2AEnv):
    """Two-agent CaTra with privileged actor observation (same split as G1CaTraPri)."""

    def _get_obs(self, data, info, feet_contact):
        obs = super()._get_obs(data, info, feet_contact)
        priv = obs["privileged_state"]
        return {"state": priv[:-_DR_TAIL_DIMS], "privileged_state": priv}
