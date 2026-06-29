# Copyright 2024 The Brax Authors.
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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Optional[envs.Env] = None,
):
    """Validates arguments for Madrona-MJX."""
    if madrona_backend:
        if eval_env:
            raise ValueError("Madrona-MJX doesn't support multiple env instances")
        if num_eval_envs != num_envs:
            raise ValueError("Madrona-MJX requires a fixed batch size")
        if action_repeat != 1:
            raise ValueError(
                "Implement action_repeat using PipelineEnv's _n_frames to avoid unnecessary rendering!"
            )


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    local_device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError("episode_length must be specified in ppo.train")
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    else:
        wrap_for_training = envs.training.wrap
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations.

    The same shift is applied across the unroll_length (T) dimension.

    Args:
      obs: a dictionary of observations
      key: a PRNGKey

    Returns:
      A dictionary of observations with translated pixels
    """

    @jax.vmap
    def rt_all_views(
        ub_obs: Mapping[str, jax.Array], key: PRNGKey
    ) -> Mapping[str, jax.Array]:
        # Expects dictionary of unbatched observations.
        def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:  # TxHxWxC
            # Randomly translates a set of pixel inputs.
            # Adapted from
            # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="edge",
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith("pixels/"):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


# --------------------------------------------------------------------------- #
# DAgger x RL distillation helpers                                            #
#                                                                             #
# Distill multiple privileged specialists (G1CaTraPri single-agent, or        #
# G1CaTra2APri two-agent) into a single generalist G1CaTra student. The       #
# student rollout env (G1CaTraDagger) routes each episode's potential fields  #
# to one scene and tags it with `pf_id`; the matching specialist supervises   #
# that env. Schedule is two-phase: KL imitation for the first                 #
# `dagger_timesteps` env steps, then standard PPO.                            #
# --------------------------------------------------------------------------- #


def _cfg_get(config: Optional[Any], name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _stack_trees(trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)


def _teacher_observation(
    observation: Any,
    teacher_obs_key: Optional[str],
    teacher_privileged_obs_key: Optional[str],
) -> Any:
    """Remap the student rollout observation into a teacher policy observation."""
    if teacher_obs_key is None:
        return observation
    if not isinstance(observation, Mapping):
        raise ValueError("teacher_obs_key requires dict observations")
    if teacher_obs_key not in observation:
        raise ValueError(f"Missing DAgger teacher obs key: {teacher_obs_key}")

    teacher_obs = {"state": observation[teacher_obs_key]}
    if teacher_privileged_obs_key is not None:
        if teacher_privileged_obs_key not in observation:
            raise ValueError(
                f"Missing DAgger teacher privileged obs key: {teacher_privileged_obs_key}"
            )
        teacher_obs["privileged_state"] = observation[teacher_privileged_obs_key]
    elif "privileged_state" in observation:
        teacher_obs["privileged_state"] = observation["privileged_state"]
    return teacher_obs


def _uint64_lt(value: types.UInt64, other: int) -> jnp.ndarray:
    other_hi = jnp.uint32(other >> 32)
    other_lo = jnp.uint32(other & 0xFFFFFFFF)
    return (value.hi < other_hi) | ((value.hi == other_hi) & (value.lo < other_lo))


def _uint64_to_float(value: types.UInt64) -> jnp.ndarray:
    """env_steps (UInt64 hi/lo) -> float for the blend-mode curriculum schedule.

    float32 is fine here: the value only feeds a progress ratio, where a few-step
    rounding error over hundreds of millions of steps is negligible.
    """
    return value.hi.astype(jnp.float32) * float(1 << 32) + value.lo.astype(jnp.float32)


def _teacher_logits_all(
    teacher_kind: str,
    teacher_observation: Any,
    student_policy_apply: Callable,
    teacher_normalizer_params: Any,
    teacher_policy_params: Any,
    teacher_policy_apply_lower: Optional[Callable],
    teacher_policy_apply_upper: Optional[Callable],
) -> jnp.ndarray:
    """Per-teacher action logits in the student's distribution layout.

    Single-agent teachers reuse the student's 20-dim policy apply directly.
    Two-agent teachers run their lower (12) + upper (8) policy nets and
    reassemble the two NormalTanh param vectors into the student's
    `[loc(20), scale(20)]` layout (lower-first, matching the env action order).
    """
    if teacher_kind == "2a":
        policy_lower, policy_upper = teacher_policy_params

        def teacher_apply(norm, p_lower, p_upper):
            logits_l = teacher_policy_apply_lower(norm, p_lower, teacher_observation)
            logits_u = teacher_policy_apply_upper(norm, p_upper, teacher_observation)
            loc_l, scale_l = jnp.split(logits_l, 2, axis=-1)
            loc_u, scale_u = jnp.split(logits_u, 2, axis=-1)
            return jnp.concatenate([loc_l, loc_u, scale_l, scale_u], axis=-1)

        return jax.vmap(teacher_apply)(
            teacher_normalizer_params, policy_lower, policy_upper
        )

    def teacher_apply(norm, policy):
        return student_policy_apply(norm, policy, teacher_observation)

    return jax.vmap(teacher_apply)(teacher_normalizer_params, teacher_policy_params)


def compute_dagger_loss(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    teacher_normalizer_params: Any,
    teacher_policy_params: Any,
    num_teachers: int,
    teacher_kind: str = "single",
    teacher_policy_apply_lower: Optional[Callable] = None,
    teacher_policy_apply_upper: Optional[Callable] = None,
    kl_eps: float = 1e-5,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    actor_loss_scale: float = 1.0,
    value_loss_scale: float = 1.0,
    teacher_obs_key: Optional[str] = None,
    teacher_privileged_obs_key: Optional[str] = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """KL-imitation loss toward the per-env selected teacher + a value loss."""
    del rng
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    parametric_action_distribution = ppo_network.parametric_action_distribution

    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    student_logits = policy_apply(normalizer_params, params.policy, data.observation)
    baseline = value_apply(normalizer_params, params.value, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)

    teacher_observation = _teacher_observation(
        data.observation, teacher_obs_key, teacher_privileged_obs_key
    )

    teacher_logits_all = _teacher_logits_all(
        teacher_kind,
        teacher_observation,
        policy_apply,
        teacher_normalizer_params,
        teacher_policy_params,
        teacher_policy_apply_lower,
        teacher_policy_apply_upper,
    )
    pf_id = data.extras["state_extras"]["pf_id"].astype(jnp.int32)
    pf_id = jnp.clip(pf_id, 0, num_teachers - 1)
    teacher_weight = jax.nn.one_hot(pf_id, num_teachers, dtype=student_logits.dtype)
    teacher_logits = jnp.einsum("tbn,ntba->tba", teacher_weight, teacher_logits_all)

    student_dist = parametric_action_distribution.create_dist(student_logits)
    teacher_dist = parametric_action_distribution.create_dist(teacher_logits)
    student_std = jnp.maximum(student_dist.scale, kl_eps)
    teacher_std = jnp.maximum(teacher_dist.scale, kl_eps)
    log_term = jnp.log(teacher_std / student_std + kl_eps)
    numerator = jnp.square(student_std) + jnp.square(student_dist.loc - teacher_dist.loc)
    denominator = 2.0 * jnp.square(teacher_std)
    dagger_kl = jnp.sum(log_term + numerator / denominator - 0.5, axis=-1)
    dagger_kl = jnp.mean(dagger_kl)

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)
    vs, _ = ppo_losses.compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    v_loss = jnp.mean(jnp.square(vs - baseline)) * 0.5 * 0.5

    actor_loss = actor_loss_scale * dagger_kl
    value_loss = value_loss_scale * v_loss
    total_loss = actor_loss + value_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": actor_loss,
        "v_loss": value_loss,
        "entropy_loss": jnp.zeros_like(total_loss),
        "dagger_kl": dagger_kl,
        "student_std": jnp.mean(student_std),
        "teacher_std": jnp.mean(teacher_std),
        "loss_mode": jnp.ones_like(total_loss),
    }


def compute_dagger_then_ppo_loss(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    dagger_phase: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    teacher_normalizer_params: Any,
    teacher_policy_params: Any,
    num_teachers: int,
    teacher_kind: str = "single",
    teacher_policy_apply_lower: Optional[Callable] = None,
    teacher_policy_apply_upper: Optional[Callable] = None,
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
    """Switch between DAgger KL imitation and PPO based on `dagger_phase`."""

    def dagger_loss(_):
        return compute_dagger_loss(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network=ppo_network,
            teacher_normalizer_params=teacher_normalizer_params,
            teacher_policy_params=teacher_policy_params,
            num_teachers=num_teachers,
            teacher_kind=teacher_kind,
            teacher_policy_apply_lower=teacher_policy_apply_lower,
            teacher_policy_apply_upper=teacher_policy_apply_upper,
            kl_eps=kl_eps,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            actor_loss_scale=actor_loss_scale,
            value_loss_scale=value_loss_scale,
            teacher_obs_key=teacher_obs_key,
            teacher_privileged_obs_key=teacher_privileged_obs_key,
        )

    def ppo_loss(_):
        total_loss, metrics = ppo_losses.compute_ppo_loss(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
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


def compute_blended_dagger_ppo_loss(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    lambda_dagger: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    teacher_normalizer_params: Any,
    teacher_policy_params: Any,
    num_teachers: int,
    teacher_kind: str = "single",
    teacher_policy_apply_lower: Optional[Callable] = None,
    teacher_policy_apply_upper: Optional[Callable] = None,
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
    """Curriculum-blended objective: ``lambda_ppo*L_PPO + lambda_dagger*L_DAgger``.

    DAgger is never turned off (lambda_dagger is floored upstream). Because the two
    weights sum to 1 and both losses carry the same value-regression term, the
    critic stays trained throughout while the actor smoothly hands off from
    imitation to PPO.
    """
    rng_ppo, rng_dagger = jax.random.split(rng)
    lambda_ppo = 1.0 - lambda_dagger

    ppo_total, ppo_metrics = ppo_losses.compute_ppo_loss(
        params, normalizer_params, data, rng_ppo,
        ppo_network=ppo_network, entropy_cost=entropy_cost, discounting=discounting,
        reward_scaling=reward_scaling, gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon, normalize_advantage=normalize_advantage,
    )
    dagger_total, dagger_metrics = compute_dagger_loss(
        params, normalizer_params, data, rng_dagger,
        ppo_network=ppo_network,
        teacher_normalizer_params=teacher_normalizer_params,
        teacher_policy_params=teacher_policy_params,
        num_teachers=num_teachers, teacher_kind=teacher_kind,
        teacher_policy_apply_lower=teacher_policy_apply_lower,
        teacher_policy_apply_upper=teacher_policy_apply_upper,
        kl_eps=kl_eps, discounting=discounting, reward_scaling=reward_scaling,
        gae_lambda=gae_lambda, actor_loss_scale=actor_loss_scale,
        value_loss_scale=value_loss_scale, teacher_obs_key=teacher_obs_key,
        teacher_privileged_obs_key=teacher_privileged_obs_key,
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


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    training_metrics_buffer_size: int = 10,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = False,
    dagger_config: Optional[Any] = None,
):
    """PPO training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      max_devices_per_host: maximum number of chips to use per host process
      wrap_env: If True, wrap the environment for training. Otherwise use the
        environment as is.
      madrona_backend: whether to use Madrona backend for training
      augment_pixels: whether to add image augmentation to pixel inputs
      num_envs: the number of parallel environments to use for rollouts
        NOTE: `num_envs` must be divisible by the total number of chips since each
          chip gets `num_envs // total_number_of_chips` environments to roll out
        NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
          data generated by `num_envs` parallel envs gets used for gradient
          updates over `num_minibatches` of data, where each minibatch has a
          leading dimension of `batch_size`
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      wrap_env_fn: a custom function that wraps the environment for training. If
        not specified, the environment is wrapped with the default training
        wrapper.
      randomization_fn: a user-defined callback function that generates randomized
        environments
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy of
        the policy
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each environment. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new environment rollout
      num_resets_per_eval: the number of environment resets to run between each
        eval. The environment resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      max_grad_norm: gradient clipping norm value. If None, no clipping is done
      normalize_advantage: whether to normalize advantage estimate
      network_factory: function that generates networks for policy and value
        functions
      seed: random seed
      num_evals: the number of evals to run during the entire training run.
        Increasing the number of evals increases total training time
      eval_env: an optional environment for eval only, defaults to `environment`
      num_eval_envs: the number of envs to use for evluation. Each env will run 1
        episode, and all envs run in parallel during eval.
      deterministic_eval: whether to run the eval with a deterministic policy
      log_training_metrics: whether to log training metrics and callback to
        progress_fn
      training_metrics_steps: the number of environment steps between logging
        training metrics
      training_metrics_buffer_size: log buf size
      progress_fn: a user-defined callback function for reporting/plotting metrics
      policy_params_fn: a user-defined callback function that can be used for
        saving custom policy checkpoints or creating policy rollouts and videos
      save_checkpoint_path: the path used to save checkpoints. If None, no
        checkpoints are saved.
      restore_checkpoint_path: the path used to restore previous model params
      restore_params: raw network parameters to restore the TrainingState from.
        These override `restore_checkpoint_path`. These paramaters can be obtained
        from the return values of ppo.train().
      restore_value_fn: whether to restore the value function from the checkpoint
        or use a random initialization

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    assert batch_size * num_minibatches % num_envs == 0
    _validate_madrona_args(
        madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
    )

    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key)
    del global_key

    assert num_envs % device_count == 0

    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        local_device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )
    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)
    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # ---- DAgger distillation setup (load + stack teacher specialists) ----
    use_dagger = bool(_cfg_get(dagger_config, "enable", False))
    teacher_checkpoint_paths = list(_cfg_get(dagger_config, "teacher_checkpoint_paths", []))
    dagger_timesteps = int(_cfg_get(dagger_config, "dagger_timesteps", 0))
    teacher_kind = str(_cfg_get(dagger_config, "teacher_kind", "single"))
    # Schedule: "two_phase" (DAgger then PPO) or "blend" (curriculum-weighted sum).
    dagger_mode = str(_cfg_get(dagger_config, "dagger_mode", "two_phase"))
    blend_lambda_floor = float(_cfg_get(dagger_config, "blend_lambda_floor", 0.1))
    blend_anneal_timesteps = float(
        _cfg_get(dagger_config, "blend_anneal_timesteps", 0) or (num_timesteps // 2)
    )
    teacher_normalizer_params = None
    teacher_policy_params = None
    teacher_policy_apply_lower = None
    teacher_policy_apply_upper = None
    if use_dagger:
        loss_name = _cfg_get(dagger_config, "loss", "kl")
        if loss_name != "kl":
            raise ValueError(f"Unsupported DAgger loss: {loss_name!r}; only 'kl' is implemented")
        if teacher_kind not in ("single", "2a"):
            raise ValueError(f"Unsupported dagger teacher_kind: {teacher_kind!r}")
        if dagger_mode not in ("two_phase", "blend"):
            raise ValueError(f"Unsupported dagger_mode: {dagger_mode!r}")
        if not teacher_checkpoint_paths:
            raise ValueError("dagger_config.enable=True requires teacher_checkpoint_paths")
        teacher_params = tuple(checkpoint.load(path) for path in teacher_checkpoint_paths)
        teacher_normalizer_params = _stack_trees([p[0] for p in teacher_params])
        if teacher_kind == "2a":
            # Two-agent checkpoint layout:
            # (normalizer, policy_lower, policy_upper, value_lower, value_upper).
            from cat_ppo.learning.policy.ppo import networks_2a

            action_size_lower = int(_cfg_get(dagger_config, "teacher_action_size_lower", 12))
            action_size_upper = int(_cfg_get(dagger_config, "teacher_action_size_upper", 8))
            teacher_hidden = tuple(
                _cfg_get(dagger_config, "teacher_policy_hidden_layer_sizes", []) or (256, 128, 64)
            )
            teacher_2a_net = networks_2a.make_ppo_networks_2a(
                obs_shape,
                action_size_lower=action_size_lower,
                action_size_upper=action_size_upper,
                preprocess_observations_fn=normalize,
                policy_hidden_layer_sizes=teacher_hidden,
                # teacher_observation is remapped to {"state": <302-dim>, ...}.
                policy_obs_key="state",
            )
            teacher_policy_apply_lower = teacher_2a_net.policy_network_lower.apply
            teacher_policy_apply_upper = teacher_2a_net.policy_network_upper.apply
            teacher_policy_params = (
                _stack_trees([p[1] for p in teacher_params]),
                _stack_trees([p[2] for p in teacher_params]),
            )
        else:
            teacher_policy_params = _stack_trees([p[1] for p in teacher_params])

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    if use_dagger:
        # Both schedules thread one scalar through the SGD step (the slot the
        # two-phase `dagger_phase` bool occupies): blend mode passes lambda_dagger.
        dagger_loss_builder = (
            compute_blended_dagger_ppo_loss
            if dagger_mode == "blend"
            else compute_dagger_then_ppo_loss
        )
        loss_fn = functools.partial(
            dagger_loss_builder,
            ppo_network=ppo_network,
            teacher_normalizer_params=teacher_normalizer_params,
            teacher_policy_params=teacher_policy_params,
            num_teachers=len(teacher_checkpoint_paths),
            teacher_kind=teacher_kind,
            teacher_policy_apply_lower=teacher_policy_apply_lower,
            teacher_policy_apply_upper=teacher_policy_apply_upper,
            kl_eps=_cfg_get(dagger_config, "kl_eps", 1e-5),
            teacher_obs_key=_cfg_get(dagger_config, "teacher_obs_key", None),
            teacher_privileged_obs_key=_cfg_get(dagger_config, "teacher_privileged_obs_key", None),
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
            actor_loss_scale=_cfg_get(dagger_config, "actor_loss_scale", 1.0),
            value_loss_scale=_cfg_get(dagger_config, "value_loss_scale", 1.0),
        )
    else:
        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
        )

    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        buffer_size=training_metrics_buffer_size,
        steps_between_logging=training_metrics_steps or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    ckpt_config = checkpoint.network_config(
        observation_size=obs_shape,
        action_size=env.action_size,
        normalize_observations=normalize_observations,
        network_factory=network_factory,
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        dagger_phase: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        if use_dagger:
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                dagger_phase,
                optimizer_state=optimizer_state,
            )
        else:
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state=optimizer_state,
            )

        return (optimizer_state, params, key), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        dagger_phase: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(_random_translate_pixels, key=key_rt)
            data = types.Transition(
                observation=r_translate(data.observation),
                action=data.action,
                reward=data.reward,
                discount=data.discount,
                next_observation=r_translate(data.next_observation),
                extras=data.extras,
            )

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step,
                normalizer_params=normalizer_params,
                dagger_phase=dagger_phase,
            ),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t, *, enable_metrics: bool
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )

        extra_fields = ("truncation", "episode_metrics", "episode_done")
        if use_dagger:
            extra_fields = extra_fields + ("pf_id",)

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=extra_fields,
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        if enable_metrics:  # log unroll metrics
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras["state_extras"]["episode_metrics"],
                data.extras["state_extras"]["episode_done"],
            )

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            _remove_pixels(data.observation),
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        # Scalar threaded into the loss. two_phase: boolean DAgger-vs-PPO switch
        # (imitate while env_steps < dagger_timesteps). blend: lambda_dagger from
        # the curriculum max(floor, 1 - env_steps/anneal_timesteps).
        if dagger_mode == "blend":
            progress = _uint64_to_float(training_state.env_steps) / blend_anneal_timesteps
            dagger_phase = jnp.maximum(blend_lambda_floor, 1.0 - progress)
        else:
            dagger_phase = _uint64_lt(training_state.env_steps, dagger_timesteps)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=data,
                normalizer_params=normalizer_params,
                dagger_phase=dagger_phase,
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey, *, enable_metrics: bool
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            functools.partial(training_step, enable_metrics=enable_metrics),
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch_with_metrics = jax.pmap(
        functools.partial(training_epoch, enable_metrics=log_training_metrics),
        axis_name=_PMAP_AXIS_NAME,
    )
    training_epoch_without_metrics = jax.pmap(
        functools.partial(training_epoch, enable_metrics=False),
        axis_name=_PMAP_AXIS_NAME,
    )

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        key: PRNGKey,
        pmapped_training_epoch: Callable[[TrainingState, envs.State, PRNGKey], Tuple[TrainingState, envs.State, Metrics]],
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = pmapped_training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )

    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(
            init_params
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(_remove_pixels(obs_shape)),
        env_steps=types.UInt64(hi=0, lo=0),
    )

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        value_params = params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(policy=params[1], value=value_params),
        )

    if restore_params is not None:
        logging.info("Restoring TrainingState from `restore_params`.")
        value_params = restore_params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=restore_params[0],
            params=training_state.params.replace(
                policy=restore_params[1], value=value_params
            ),
        )

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    eval_env = _maybe_wrap_env(
        eval_env or environment,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        local_device_count=1,  # eval on the host only
        key_env=eval_key,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
    )
    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    disable_metrics_after_checkpoint = 4
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            disable_metrics_this_epoch = (
                device_count > 1
                and log_training_metrics
                and it >= disable_metrics_after_checkpoint
            )
            if disable_metrics_this_epoch and it == disable_metrics_after_checkpoint:
                logging.warning(
                    "Disabling in-epoch training metrics after checkpoint %s on multi-GPU runs "
                    "to avoid late jax.debug.callback stalls under pmap.",
                    disable_metrics_after_checkpoint,
                )
            pmapped_training_epoch = (
                training_epoch_without_metrics
                if disable_metrics_this_epoch
                else training_epoch_with_metrics
            )
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys, pmapped_training_epoch
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        # Process id == 0.
        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )

        policy_params_fn(current_step, make_policy, params)

        if save_checkpoint_path is not None:
            checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

        if num_evals > 0:
            metrics = evaluator.run_evaluation(
                params,
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f"Total steps {total_steps} is less than `num_timesteps`= {num_timesteps}."
        )

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
