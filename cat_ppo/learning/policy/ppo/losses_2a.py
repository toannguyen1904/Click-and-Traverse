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

from typing import Any, Callable, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# DAgger x RL distillation for the two-agent student                          #
#                                                                             #
# Distill two-agent privileged specialists (G1CaTra2APri) into one generalist #
# two-agent G1CaTra2A student. Teachers are routed per scene by `pf_id` and    #
# supervise the student head-for-head (lower<-lower, upper<-upper), so no logit #
# reassembly is needed. Schedule mirrors the single-agent case: KL imitation   #
# for the first `dagger_timesteps` env steps, then two-agent PPO.              #
#                                                                             #
# Consistency rule (enforced by train_ppo_dagger): a two-agent student is only #
# distilled from two-agent teachers (two-two), never from single-agent ones.   #
# --------------------------------------------------------------------------- #


def _gauss_kl(student_dist, teacher_dist, kl_eps: float) -> jnp.ndarray:
  """Mean diagonal-Gaussian KL(student || teacher), summed over action dims."""
  student_std = jnp.maximum(student_dist.scale, kl_eps)
  teacher_std = jnp.maximum(teacher_dist.scale, kl_eps)
  log_term = jnp.log(teacher_std / student_std + kl_eps)
  numerator = jnp.square(student_std) + jnp.square(student_dist.loc - teacher_dist.loc)
  denominator = 2.0 * jnp.square(teacher_std)
  kl = jnp.sum(log_term + numerator / denominator - 0.5, axis=-1)
  return jnp.mean(kl), jnp.mean(student_std), jnp.mean(teacher_std)


def _select_teacher_logits(logits_all, pf_id, num_teachers):
  """One-hot select per-env teacher logits: logits_all[t? , T, B, A] -> [T, B, A]."""
  pf_id = jnp.clip(pf_id.astype(jnp.int32), 0, num_teachers - 1)
  weight = jax.nn.one_hot(pf_id, num_teachers, dtype=logits_all.dtype)
  return jnp.einsum("tbn,ntba->tba", weight, logits_all)


def compute_dagger_loss_2a(
    params: PPONetworkParams2A,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks_2a.PPONetworks2A,
    teacher_normalizer_params: Any,
    teacher_policy_lower_params: Any,
    teacher_policy_upper_params: Any,
    num_teachers: int,
    kl_eps: float = 1e-5,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    actor_loss_scale: float = 1.0,
    value_loss_scale: float = 1.0,
    teacher_obs_key: Optional[str] = None,
    teacher_privileged_obs_key: Optional[str] = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Per-head KL imitation toward the selected two-agent teacher + value losses."""
  from cat_ppo.learning.policy.ppo.train import _teacher_observation

  del rng
  dist_l = ppo_network.parametric_action_distribution_lower
  dist_u = ppo_network.parametric_action_distribution_upper
  apply_l = ppo_network.policy_network_lower.apply
  apply_u = ppo_network.policy_network_upper.apply

  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
  obs = data.observation
  terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)

  student_logits_l = apply_l(normalizer_params, params.policy_lower, obs)
  student_logits_u = apply_u(normalizer_params, params.policy_upper, obs)
  baseline_l = ppo_network.value_network_lower.apply(normalizer_params, params.value_lower, obs)
  baseline_u = ppo_network.value_network_upper.apply(normalizer_params, params.value_upper, obs)
  bootstrap_l = ppo_network.value_network_lower.apply(normalizer_params, params.value_lower, terminal_obs)
  bootstrap_u = ppo_network.value_network_upper.apply(normalizer_params, params.value_upper, terminal_obs)

  # The teacher (G1CaTra2APri) shares the student's per-head architecture, so we
  # reuse the student's apply with the teacher's params on the teacher obs.
  teacher_obs = _teacher_observation(obs, teacher_obs_key, teacher_privileged_obs_key)

  def teacher_apply(t_norm, t_pl, t_pu):
    return apply_l(t_norm, t_pl, teacher_obs), apply_u(t_norm, t_pu, teacher_obs)

  teacher_logits_all_l, teacher_logits_all_u = jax.vmap(teacher_apply)(
      teacher_normalizer_params, teacher_policy_lower_params, teacher_policy_upper_params
  )
  pf_id = data.extras["state_extras"]["pf_id"]
  teacher_logits_l = _select_teacher_logits(teacher_logits_all_l, pf_id, num_teachers)
  teacher_logits_u = _select_teacher_logits(teacher_logits_all_u, pf_id, num_teachers)

  kl_l, sstd_l, tstd_l = _gauss_kl(
      dist_l.create_dist(student_logits_l), dist_l.create_dist(teacher_logits_l), kl_eps)
  kl_u, sstd_u, tstd_u = _gauss_kl(
      dist_u.create_dist(student_logits_u), dist_u.create_dist(teacher_logits_u), kl_eps)

  truncation = data.extras["state_extras"]["truncation"]
  termination = (1 - data.discount) * (1 - truncation)

  def _value_loss(rewards, baseline, bootstrap):
    vs, _ = compute_gae(
        truncation=truncation, termination=termination,
        rewards=rewards * reward_scaling, values=baseline,
        bootstrap_value=bootstrap, lambda_=gae_lambda, discount=discounting,
    )
    return jnp.mean(jnp.square(vs - baseline)) * 0.5 * 0.5

  vloss_l = _value_loss(data.extras["state_extras"]["reward_lower"], baseline_l, bootstrap_l)
  vloss_u = _value_loss(data.extras["state_extras"]["reward_upper"], baseline_u, bootstrap_u)

  al_l, al_u = actor_loss_scale * kl_l, actor_loss_scale * kl_u
  vl_l, vl_u = value_loss_scale * vloss_l, value_loss_scale * vloss_u
  total_l, total_u = al_l + vl_l, al_u + vl_u
  total_loss = total_l + total_u
  zero = jnp.zeros_like(total_loss)
  # Mirror compute_ppo_loss_2a's metric schema exactly (+ the 4 dagger keys) so
  # the jax.lax.cond branches return identical pytrees.
  return total_loss, {
      "total_loss": total_loss,
      "lower/total_loss": total_l, "lower/policy_loss": al_l, "lower/v_loss": vl_l, "lower/entropy_loss": zero,
      "upper/total_loss": total_u, "upper/policy_loss": al_u, "upper/v_loss": vl_u, "upper/entropy_loss": zero,
      "dagger_kl": kl_l + kl_u,
      "student_std": 0.5 * (sstd_l + sstd_u),
      "teacher_std": 0.5 * (tstd_l + tstd_u),
      "loss_mode": jnp.ones_like(total_loss),
  }


def compute_dagger_then_ppo_loss_2a(
    params: PPONetworkParams2A,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    dagger_phase: jnp.ndarray,
    ppo_network: ppo_networks_2a.PPONetworks2A,
    teacher_normalizer_params: Any,
    teacher_policy_lower_params: Any,
    teacher_policy_upper_params: Any,
    num_teachers: int,
    kl_eps: float = 1e-5,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    actor_loss_scale: float = 1.0,
    value_loss_scale: float = 1.0,
    teacher_obs_key: Optional[str] = None,
    teacher_privileged_obs_key: Optional[str] = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Switch between two-agent DAgger imitation and two-agent PPO via `dagger_phase`."""

  def dagger_loss(_):
    return compute_dagger_loss_2a(
        params, normalizer_params, data, rng, ppo_network=ppo_network,
        teacher_normalizer_params=teacher_normalizer_params,
        teacher_policy_lower_params=teacher_policy_lower_params,
        teacher_policy_upper_params=teacher_policy_upper_params,
        num_teachers=num_teachers, kl_eps=kl_eps, discounting=discounting,
        reward_scaling=reward_scaling, gae_lambda=gae_lambda,
        actor_loss_scale=actor_loss_scale, value_loss_scale=value_loss_scale,
        teacher_obs_key=teacher_obs_key, teacher_privileged_obs_key=teacher_privileged_obs_key,
    )

  def ppo_loss(_):
    total_loss, metrics = compute_ppo_loss_2a(
        params, normalizer_params, data, rng, ppo_network=ppo_network,
        entropy_cost=entropy_cost, discounting=discounting, reward_scaling=reward_scaling,
        gae_lambda=gae_lambda, clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
    )
    metrics = {
        **metrics,
        "dagger_kl": jnp.zeros_like(total_loss),
        "student_std": jnp.zeros_like(total_loss),
        "teacher_std": jnp.zeros_like(total_loss),
        "loss_mode": jnp.zeros_like(total_loss),
    }
    return total_loss, metrics

  return jax.lax.cond(dagger_phase, dagger_loss, ppo_loss, operand=None)


def compute_blended_dagger_ppo_loss_2a(
    params: PPONetworkParams2A,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    lambda_dagger: jnp.ndarray,
    ppo_network: ppo_networks_2a.PPONetworks2A,
    teacher_normalizer_params: Any,
    teacher_policy_lower_params: Any,
    teacher_policy_upper_params: Any,
    num_teachers: int,
    kl_eps: float = 1e-5,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    actor_loss_scale: float = 1.0,
    value_loss_scale: float = 1.0,
    teacher_obs_key: Optional[str] = None,
    teacher_privileged_obs_key: Optional[str] = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Curriculum-blended two-agent objective: lambda_ppo*L_PPO + lambda_dagger*L_DAgger.

  DAgger is never turned off (lambda_dagger is floored upstream). Blends the whole
  two-agent PPO and DAgger losses; since the weights sum to 1 and both carry the
  per-head value regression, the critics stay trained while the actors hand off
  from per-head imitation to PPO.
  """
  rng_ppo, rng_dagger = jax.random.split(rng)
  lambda_ppo = 1.0 - lambda_dagger

  ppo_total, ppo_metrics = compute_ppo_loss_2a(
      params, normalizer_params, data, rng_ppo, ppo_network=ppo_network,
      entropy_cost=entropy_cost, discounting=discounting, reward_scaling=reward_scaling,
      gae_lambda=gae_lambda, clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
  )
  dagger_total, dagger_metrics = compute_dagger_loss_2a(
      params, normalizer_params, data, rng_dagger, ppo_network=ppo_network,
      teacher_normalizer_params=teacher_normalizer_params,
      teacher_policy_lower_params=teacher_policy_lower_params,
      teacher_policy_upper_params=teacher_policy_upper_params,
      num_teachers=num_teachers, kl_eps=kl_eps, discounting=discounting,
      reward_scaling=reward_scaling, gae_lambda=gae_lambda,
      actor_loss_scale=actor_loss_scale, value_loss_scale=value_loss_scale,
      teacher_obs_key=teacher_obs_key, teacher_privileged_obs_key=teacher_privileged_obs_key,
  )

  total_loss = lambda_ppo * ppo_total + lambda_dagger * dagger_total
  metrics = {
      **ppo_metrics,
      "total_loss": total_loss,
      "ppo_total_loss": ppo_total,
      "dagger_total_loss": dagger_total,
      "dagger_kl": dagger_metrics["dagger_kl"],
      "student_std": dagger_metrics["student_std"],
      "teacher_std": dagger_metrics["teacher_std"],
      "lambda_dagger": lambda_dagger,
      "lambda_ppo": lambda_ppo,
      "loss_mode": jnp.full_like(total_loss, 2.0),
  }
  return total_loss, metrics
