# Copyright 2025 The Brax Authors.
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
"""Two-agent PPO loss: independent PPO objective per agent (lower / upper), summed.

Each agent has its own reward stream (env emits reward shape [..., 2] = [lower, upper]),
its own value baseline, and its own action distribution. Both agents read the shared
observation. Losses are summed so a single optimizer updates all four networks.
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.ppo.losses import compute_gae
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp

from cat_ppo.learning.policy.ppo import networks_2a as ppo_networks_2a


@flax.struct.dataclass
class PPONetworkParams2A:
  """Training params for the two-agent learner (4 networks)."""

  policy_lower: Params
  policy_upper: Params
  value_lower: Params
  value_upper: Params


def _agent_loss(
    policy_logits, baseline, bootstrap_value, rewards, truncation, termination,
    raw_action, behaviour_log_prob, dist, rng,
    entropy_cost, discounting, gae_lambda, clipping_epsilon, normalize_advantage,
):
  """Standard single-agent PPO objective (mirrors brax compute_ppo_loss body)."""
  target_log_prob = dist.log_prob(policy_logits, raw_action)

  vs, advantages = compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=gae_lambda,
      discount=discounting,
  )
  if normalize_advantage:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  rho_s = jnp.exp(target_log_prob - behaviour_log_prob)
  surrogate1 = rho_s * advantages
  surrogate2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
  policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

  v_error = vs - baseline
  v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

  entropy = jnp.mean(dist.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total = policy_loss + v_loss + entropy_loss
  return total, policy_loss, v_loss, entropy_loss


def compute_ppo_loss_2a(
    params: PPONetworkParams2A,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks_2a.PPONetworks2A,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Sum of the lower-agent and upper-agent PPO losses."""
  dist_l = ppo_network.parametric_action_distribution_lower
  dist_u = ppo_network.parametric_action_distribution_upper

  # Put time dimension first (data arrives [B, T, ...]).
  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

  obs = data.observation
  terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)

  # Per-agent logits / baselines.
  logits_l = ppo_network.policy_network_lower.apply(normalizer_params, params.policy_lower, obs)
  logits_u = ppo_network.policy_network_upper.apply(normalizer_params, params.policy_upper, obs)
  baseline_l = ppo_network.value_network_lower.apply(normalizer_params, params.value_lower, obs)
  baseline_u = ppo_network.value_network_upper.apply(normalizer_params, params.value_upper, obs)
  bootstrap_l = ppo_network.value_network_lower.apply(normalizer_params, params.value_lower, terminal_obs)
  bootstrap_u = ppo_network.value_network_upper.apply(normalizer_params, params.value_upper, terminal_obs)

  # Per-agent reward streams are carried in info (state.reward stays scalar for the
  # brax wrappers). They are pulled into extras via the trainer's extra_fields.
  rewards_l = data.extras['state_extras']['reward_lower'] * reward_scaling
  rewards_u = data.extras['state_extras']['reward_upper'] * reward_scaling

  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  rng, rng_l, rng_u = jax.random.split(rng, 3)
  total_l, pl_l, vl_l, el_l = _agent_loss(
      logits_l, baseline_l, bootstrap_l, rewards_l, truncation, termination,
      data.extras['policy_extras']['raw_action_lower'],
      data.extras['policy_extras']['log_prob_lower'],
      dist_l, rng_l,
      entropy_cost, discounting, gae_lambda, clipping_epsilon, normalize_advantage,
  )
  total_u, pl_u, vl_u, el_u = _agent_loss(
      logits_u, baseline_u, bootstrap_u, rewards_u, truncation, termination,
      data.extras['policy_extras']['raw_action_upper'],
      data.extras['policy_extras']['log_prob_upper'],
      dist_u, rng_u,
      entropy_cost, discounting, gae_lambda, clipping_epsilon, normalize_advantage,
  )

  total_loss = total_l + total_u
  return total_loss, {
      'total_loss': total_loss,
      'lower/total_loss': total_l, 'lower/policy_loss': pl_l, 'lower/v_loss': vl_l, 'lower/entropy_loss': el_l,
      'upper/total_loss': total_u, 'upper/policy_loss': pl_u, 'upper/v_loss': vl_u, 'upper/entropy_loss': el_u,
  }
