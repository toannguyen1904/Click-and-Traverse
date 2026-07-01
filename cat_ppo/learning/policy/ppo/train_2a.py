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
from cat_ppo.learning.policy.ppo import losses_2a
from cat_ppo.learning.policy.ppo import networks_2a
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
    params: losses_2a.PPONetworkParams2A
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
    # network_factory is pre-bound with action_size_lower / action_size_upper (see train_ppo dispatch).
    ppo_network = network_factory(
        obs_shape, preprocess_observations_fn=normalize
    )
    make_policy = networks_2a.make_inference_fn_2a(ppo_network)

    # ---- DAgger distillation setup (two-agent teachers -> two-agent student) ----
    from cat_ppo.learning.policy.ppo.train import (
        _cfg_get, _stack_trees, _uint64_lt, _uint64_to_float,
    )

    use_dagger = bool(_cfg_get(dagger_config, "enable", False))
    teacher_checkpoint_paths = list(_cfg_get(dagger_config, "teacher_checkpoint_paths", []))
    dagger_timesteps = int(_cfg_get(dagger_config, "dagger_timesteps", 0))
    # Schedule: "two_phase" (DAgger then PPO) or "blend" (curriculum-weighted sum).
    dagger_mode = str(_cfg_get(dagger_config, "dagger_mode", "two_phase"))
    blend_lambda_floor = float(_cfg_get(dagger_config, "blend_lambda_floor", 0.1))
    blend_anneal_timesteps = float(
        _cfg_get(dagger_config, "blend_anneal_timesteps", 0) or (num_timesteps // 2)
    )
    teacher_normalizer_params = None
    teacher_policy_lower_params = None
    teacher_policy_upper_params = None
    if use_dagger:
        if _cfg_get(dagger_config, "loss", "kl") != "kl":
            raise ValueError("Only the 'kl' DAgger loss is implemented")
        if str(_cfg_get(dagger_config, "teacher_kind", "2a")) != "2a":
            raise ValueError(
                "The two-agent student requires two-agent teachers (teacher_kind='2a')."
            )
        if dagger_mode not in ("two_phase", "blend"):
            raise ValueError(f"Unsupported dagger_mode: {dagger_mode!r}")
        if not teacher_checkpoint_paths:
            raise ValueError("dagger_config.enable=True requires teacher_checkpoint_paths")
        # Two-agent checkpoint: (normalizer, policy_lower, policy_upper, value_l, value_u).
        teacher_params = tuple(checkpoint.load(path) for path in teacher_checkpoint_paths)
        teacher_normalizer_params = _stack_trees([p[0] for p in teacher_params])
        teacher_policy_lower_params = _stack_trees([p[1] for p in teacher_params])
        teacher_policy_upper_params = _stack_trees([p[2] for p in teacher_params])

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    if use_dagger:
        # Both schedules thread one scalar through the SGD step: blend mode passes
        # lambda_dagger; two_phase passes the boolean DAgger-vs-PPO switch.
        dagger_loss_builder = (
            losses_2a.compute_blended_dagger_ppo_loss_2a
            if dagger_mode == "blend"
            else losses_2a.compute_dagger_then_ppo_loss_2a
        )
        loss_fn = functools.partial(
            dagger_loss_builder,
            ppo_network=ppo_network,
            teacher_normalizer_params=teacher_normalizer_params,
            teacher_policy_lower_params=teacher_policy_lower_params,
            teacher_policy_upper_params=teacher_policy_upper_params,
            num_teachers=len(teacher_checkpoint_paths),
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
            losses_2a.compute_ppo_loss_2a,
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
                training_state.params.policy_lower,
                training_state.params.policy_upper,
            )
        )

        extra_fields = (
            "truncation", "episode_metrics", "episode_done",
            "reward_lower", "reward_upper",
        )
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
        # Scalar threaded into the loss. two_phase: boolean DAgger-vs-PPO switch.
        # blend: lambda_dagger = max(floor, 1 - env_steps/anneal_timesteps).
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

    # Initialize model params and training state (4 networks).
    key_pl, key_pu, key_vl, key_vu = jax.random.split(key_policy, 4)
    init_params = losses_2a.PPONetworkParams2A(
        policy_lower=ppo_network.policy_network_lower.init(key_pl),
        policy_upper=ppo_network.policy_network_upper.init(key_pu),
        value_lower=ppo_network.value_network_lower.init(key_vl),
        value_upper=ppo_network.value_network_upper.init(key_vu),
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

    def _restore_from(p):
        # p layout: (normalizer, policy_lower, policy_upper, value_lower, value_upper)
        vl = p[3] if restore_value_fn else init_params.value_lower
        vu = p[4] if restore_value_fn else init_params.value_upper
        return training_state.replace(
            normalizer_params=p[0],
            params=training_state.params.replace(
                policy_lower=p[1], policy_upper=p[2], value_lower=vl, value_upper=vu
            ),
        )

    if restore_checkpoint_path is not None:
        training_state = _restore_from(checkpoint.load(restore_checkpoint_path))

    if restore_params is not None:
        logging.info("Restoring TrainingState from `restore_params`.")
        training_state = _restore_from(restore_params)

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy_lower,
                training_state.params.policy_upper,
                training_state.params.value_lower,
                training_state.params.value_upper,
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
                    training_state.params.policy_lower,
                    training_state.params.policy_upper,
                    training_state.params.value_lower,
                    training_state.params.value_upper,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    disable_metrics_after_checkpoint = 3
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
                training_state.params.policy_lower,
                training_state.params.policy_upper,
                training_state.params.value_lower,
                training_state.params.value_upper,
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
            training_state.params.policy_lower,
            training_state.params.policy_upper,
            training_state.params.value_lower,
            training_state.params.value_upper,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
