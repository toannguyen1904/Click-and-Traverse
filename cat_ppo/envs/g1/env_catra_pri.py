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
"""Privileged-actor variant of G1CaTra (mirrors the G1Cat -> G1CatPri relationship).

The actor receives the same noiseless world-frame signals as the critic — current
(non-delayed) PF, absolute body positions/velocities, box world pose + velocity,
pelvis linvel, navi_torso_rpy[:2], gait_mask, feet_contact — minus only the 31
domain-randomization dims (rfi_lim_scale(29) + kp_scale(1) + kd_scale(1)).
The critic's privileged_state is unchanged from G1CaTra.

Domain randomization is disabled by default (randomization_fn=None), matching
G1CatPri's ENABLE_RANDOMIZE=False convention. When --warmstart_states_path is set,
train_ppo.py detects randomization_fn=None and installs `make_warmstart_only_catra`
instead of the DR-on warm-start fn — preserving warm-start initialization (box
mass/size + per-env state-index dispatch via qpos0[0]) while keeping robot DR off.
"""

import cat_ppo
from cat_ppo.envs.g1.env_catra import (
    G1CaTraEnv,
    g1_catra_task_config,
    command_to_reference,
)

# Trailing block stripped from priv to form the actor state.
# Matches the tail order in env_catra.py _get_obs: ..., rfi_lim_scale(29), kp_scale(1), kd_scale(1).
_DR_TAIL_DIMS = 29 + 1 + 1  # 31


def g1_catra_pri_task_config():
    cfg = g1_catra_task_config()
    cfg.env_config.num_obs = cfg.env_config.num_pri - _DR_TAIL_DIMS
    cfg.policy_config.randomization_fn = None
    return cfg


cat_ppo.registry.register("G1CaTraPri", "config")(g1_catra_pri_task_config())
cat_ppo.registry.register("G1CaTraPri", "command_to_reference_fn")(command_to_reference)


@cat_ppo.registry.register("G1CaTraPri", "train_env_class")
class G1CaTraPriEnv(G1CaTraEnv):
    """G1CaTra with a privileged actor (noiseless world-frame state, no DR scales)."""

    def _get_obs(self, data, info, feet_contact):
        obs = super()._get_obs(data, info, feet_contact)
        priv = obs["privileged_state"]
        state = priv[:-_DR_TAIL_DIMS]
        return {"state": state, "privileged_state": priv}
