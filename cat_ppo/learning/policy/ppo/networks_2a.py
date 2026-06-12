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
"""Two-agent PPO networks: separate actor + critic for lower body (legs) and upper body (arms).

Both actors read `policy_obs_key`; both critics read `value_obs_key`. The two actors have
their own action distributions (sizes differ: 12 legs, 8 arms). The combined policy outputs a
concatenated action [a_lower, a_upper] matching the env's action ordering (legs then arms).
"""

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
from flax import linen


@flax.struct.dataclass
class PPONetworks2A:
  policy_network_lower: networks.FeedForwardNetwork
  policy_network_upper: networks.FeedForwardNetwork
  value_network_lower: networks.FeedForwardNetwork
  value_network_upper: networks.FeedForwardNetwork
  parametric_action_distribution_lower: distribution.ParametricDistribution
  parametric_action_distribution_upper: distribution.ParametricDistribution


def make_ppo_networks_2a(
    observation_size: types.ObservationSize,
    action_size_lower: int,
    action_size_upper: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 128, 64),
    value_hidden_layer_sizes: Sequence[int] = (512, 256, 128),
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'privileged_state',
) -> PPONetworks2A:
  """Build the 4 networks (2 actors, 2 critics) for the two-agent PPO setup."""
  dist_lower = distribution.NormalTanhDistribution(event_size=action_size_lower)
  dist_upper = distribution.NormalTanhDistribution(event_size=action_size_upper)

  policy_lower = networks.make_policy_network(
      dist_lower.param_size, observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation, obs_key=policy_obs_key,
  )
  policy_upper = networks.make_policy_network(
      dist_upper.param_size, observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation, obs_key=policy_obs_key,
  )
  value_lower = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation, obs_key=value_obs_key,
  )
  value_upper = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation, obs_key=value_obs_key,
  )

  return PPONetworks2A(
      policy_network_lower=policy_lower,
      policy_network_upper=policy_upper,
      value_network_lower=value_lower,
      value_network_upper=value_upper,
      parametric_action_distribution_lower=dist_lower,
      parametric_action_distribution_upper=dist_upper,
  )


def make_inference_fn_2a(ppo_networks: PPONetworks2A):
  """Create the combined-policy inference function for the two-agent PPO agent.

  params layout: (normalizer_params, policy_lower_params, policy_upper_params).
  """

  def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
    norm = params[0]
    plower = params[1]
    pupper = params[2]
    net_l = ppo_networks.policy_network_lower
    net_u = ppo_networks.policy_network_upper
    dist_l = ppo_networks.parametric_action_distribution_lower
    dist_u = ppo_networks.parametric_action_distribution_upper

    def policy(observations: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits_l = net_l.apply(norm, plower, observations)
      logits_u = net_u.apply(norm, pupper, observations)
      if deterministic:
        action = jnp.concatenate([dist_l.mode(logits_l), dist_u.mode(logits_u)], axis=-1)
        return action, {}
      key_l, key_u = jax.random.split(key_sample)
      raw_l = dist_l.sample_no_postprocessing(logits_l, key_l)
      raw_u = dist_u.sample_no_postprocessing(logits_u, key_u)
      logp_l = dist_l.log_prob(logits_l, raw_l)
      logp_u = dist_u.log_prob(logits_u, raw_u)
      act_l = dist_l.postprocess(raw_l)
      act_u = dist_u.postprocess(raw_u)
      action = jnp.concatenate([act_l, act_u], axis=-1)  # lower-first, matches env action order
      return action, {
          'log_prob_lower': logp_l, 'raw_action_lower': raw_l,
          'log_prob_upper': logp_u, 'raw_action_upper': raw_u,
      }

    return policy

  return make_policy
