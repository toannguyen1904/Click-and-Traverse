from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tree
from brax.envs.wrappers import training as brax_training
from cat_ppo.utils.logger import LOGGER  # noqa: F401
from mujoco import mjx
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
from ml_collections import config_dict

import cat_ppo.envs.g1  # noqa: F401


def _split_any(keys):
    """
    Split PRNG keys into (main, sample).

    keys shape  (2,)   -> returns (2,), (2,)
          (B,2) -> returns (B,2), (B,2)
    """
    if keys.ndim == 1:  # scalar key
        k1, k2 = jax.random.split(keys)
        return k1, k2
    elif keys.ndim == 2:  # batched keys
        split = jax.vmap(jax.random.split)(keys)  # (B,2,2)
        return split[:, 0], split[:, 1]  # two (B,2) arrays
    else:
        raise ValueError(f"PRNG key must be shape (2,) or (B,2); got {keys.shape}")


def _randint_any(keys, lo, hi):
    """
    Uniform ints using scalar or batched keys.

    keys (2,)   -> scalar int
    keys (B,2)  -> (B,) ints
    """
    if keys.ndim == 1:
        return jax.random.randint(keys, (), lo, hi)
    elif keys.ndim == 2:
        return jax.vmap(lambda k: jax.random.randint(k, (), lo, hi))(keys)
    else:
        raise ValueError(f"PRNG key must be shape (2,) or (B,2); got {keys.shape}")


def _take_cache(cache, idx):
    """cache: {name: [T,...]}, idx: scalar or (B,) -> {name: [...]} or {name: [B,...]}"""
    return {k: jnp.take(v, idx, axis=0, mode="clip") for k, v in cache.items()}


def _to_batch(x, mask):
    """Broadcast scalar x to (B,...) if mask is (B,)."""
    if x.ndim == 0 and mask.ndim == 1:
        return jnp.broadcast_to(x, mask.shape)
    return x


class SamplePFWrapper(wrapper.Wrapper):
    """
    Loads mocap trajectories from npz files with keys:
      - qpos: [T, 7+J]
      - qvel: [T, 6+J]
      - kpt_npose: [T, K, 4, 4]
      - kpt_cvel: [T, K, 6]

    Caches data to device and resamples per-episode reference when episode ends.
    """

    def __init__(self, env):
        super().__init__(env)

    def _reset_with_pf_id(self, rng, pf_id):
        node = self.env
        while hasattr(node, "env"):
            if hasattr(node, "_mjx_model_v") and hasattr(node.env, "reset_with_pf_id"):
                def reset(mjx_model, key, scene_id):
                    env = node._env_fn(mjx_model=mjx_model)
                    return env.reset_with_pf_id(key, scene_id)

                state = jax.vmap(reset, in_axes=[node._in_axes, 0, 0])(
                    node._mjx_model_v, rng, pf_id
                )
                break
            if hasattr(node, "reset_with_pf_id"):
                state = jax.vmap(node.reset_with_pf_id)(rng, pf_id)
                break
            node = node.env
        else:
            if not hasattr(node, "reset_with_pf_id"):
                return self.reset(rng)
            state = jax.vmap(node.reset_with_pf_id)(rng, pf_id)

        state.info["steps"] = jnp.zeros(rng.shape[:-1])
        state.info["truncation"] = jnp.zeros(rng.shape[:-1])
        state.info["episode_done"] = jnp.zeros(rng.shape[:-1])
        episode_metrics = {
            "sum_reward": jnp.zeros(rng.shape[:-1]),
            "length": jnp.zeros(rng.shape[:-1]),
        }
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info["episode_metrics"] = episode_metrics
        state.info["first_state"] = state.data
        state.info["first_obs"] = state.obs
        return state

    @staticmethod
    def _update_pf_sampling_info(state, done):
        if "pf_success_ema" not in state.info:
            return state, None

        done_bool = done.astype(jnp.bool_)
        truncation = state.info["truncation"].astype(jnp.float32)
        pf_id = state.info["pf_id"].astype(jnp.int32)
        num_pf = state.info["pf_success_ema"].shape[-1]
        one_hot = jax.nn.one_hot(pf_id, num_pf, dtype=jnp.float32)
        done_f = done_bool.astype(jnp.float32)
        episode_counts = jnp.sum(one_hot * done_f[:, None], axis=0)
        success_counts = jnp.sum(one_hot * (done_f * truncation)[:, None], axis=0)

        prev_episode_ema = jnp.mean(state.info["pf_episode_ema"], axis=0)
        prev_success_ema = jnp.mean(state.info["pf_success_ema"], axis=0)
        decay = jnp.mean(state.info["pf_sampling_ema_decay"])
        episode_ema = decay * prev_episode_ema + episode_counts
        success_ema = decay * prev_success_ema + success_counts
        success_rate = success_ema / (episode_ema + 1e-6)

        alpha = jnp.mean(state.info["pf_sampling_alpha"])
        weights = jnp.maximum((1.0 - success_rate) ** alpha, 1e-3)
        logits = jnp.log(weights / jnp.sum(weights) + 1e-8)
        logits_b = jnp.broadcast_to(logits, state.info["pf_sampling_logits"].shape)
        episode_ema_b = jnp.broadcast_to(episode_ema, state.info["pf_episode_ema"].shape)
        success_ema_b = jnp.broadcast_to(success_ema, state.info["pf_success_ema"].shape)
        state.info["pf_sampling_logits"] = logits_b
        state.info["pf_episode_ema"] = episode_ema_b
        state.info["pf_success_ema"] = success_ema_b
        return state, logits

    @staticmethod
    def _batch_size(state):
        try:
            return jax.tree_util.tree_leaves(state.obs)[0].shape[0]
        except Exception:
            return state.done.shape[0] if state.done.ndim else 1

    def reset(self, rng) -> mjx_env.State:
        state = self.env.reset(rng)
        return state

    def step(self, state: mjx_env.State, action) -> mjx_env.State:
        state = self.env.step(state, action)

        done = state.done
        if done.ndim == 0:
            done = done[None]

        state, pf_sampling_logits = self._update_pf_sampling_info(state, done)

        rng = state.info["rng"]
        if pf_sampling_logits is None:
            state_reset = self.reset(rng)
        else:
            reset_pf_id = jax.vmap(lambda key: jax.random.categorical(key, pf_sampling_logits))(rng)
            state_reset = self._reset_with_pf_id(rng, reset_pf_id.astype(jnp.int32))
            state_reset.info["pf_sampling_logits"] = state.info["pf_sampling_logits"]
            state_reset.info["pf_episode_ema"] = state.info["pf_episode_ema"]
            state_reset.info["pf_success_ema"] = state.info["pf_success_ema"]
        done_exp = done[:, None]

        def reset_obs_leaf(reset_leaf, leaf):
            done_shape = done.shape + (1,) * (leaf.ndim - done.ndim)
            return jnp.where(jnp.reshape(done, done_shape), reset_leaf, leaf)

        obs = jax.tree_util.tree_map(reset_obs_leaf, state_reset.obs, state.obs)
        state = state.replace(obs=obs)
        command = jnp.where(done_exp, state_reset.info["command"], state.info["command"])
        last_command = jnp.where(done_exp, state_reset.info["last_command"], state.info["last_command"])
        last_act = jnp.where(done_exp, state_reset.info["last_act"], state.info["last_act"])
        motor_targets = jnp.where(done_exp, state_reset.info["motor_targets"], state.info["motor_targets"])
        task_step = jnp.where(done, state_reset.info["step"], state.info["step"])
        stop_timestep = jnp.where(done, state_reset.info["stop_timestep"], state.info["stop_timestep"])
        phase = jnp.where(done_exp, state_reset.info["phase"], state.info["phase"])
        phase_dt = jnp.where(done, state_reset.info["phase_dt"], state.info["phase_dt"])
        gait_freq = jnp.where(done, state_reset.info["gait_freq"], state.info["gait_freq"])
        foot_height = jnp.where(done, state_reset.info["foot_height"], state.info["foot_height"])
        if "pf_id" in state.info:
            state.info["pf_id"] = jnp.where(done, state_reset.info["pf_id"], state.info["pf_id"])
        pf_sampling_updates = {}
        if "pf_sampling_logits" in state.info:
            pf_sampling_updates = {
                "pf_sampling_logits": state.info["pf_sampling_logits"],
                "pf_episode_ema": state.info["pf_episode_ema"],
                "pf_success_ema": state.info["pf_success_ema"],
                "pf_sampling_alpha": state.info["pf_sampling_alpha"],
                "pf_sampling_ema_decay": state.info["pf_sampling_ema_decay"],
            }
        state.info.update(
            {
                "rng": state_reset.info["rng"],
                "step": task_step,
                "command": command,
                "last_command": last_command,
                "last_act": last_act,
                "motor_targets": motor_targets,
                "stop_timestep": stop_timestep,
                "phase": phase,
                "phase_dt": phase_dt,
                "gait_freq": gait_freq,
                "foot_height": foot_height,
                **pf_sampling_updates,
            }
        )
        qpos = jnp.where(done_exp, state_reset.data.qpos, state.data.qpos)
        qvel = jnp.where(done_exp, state_reset.data.qvel, state.data.qvel)
        state = state.replace(
            data=state.data.replace(qpos=qpos, qvel=qvel),
        )
        reward = jnp.where(done, state_reset.reward, state.reward)
        state = state.replace(reward=reward)
        return state


def wrap_for_brax_training_reset(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[[mjx.Model], tuple[mjx.Model, mjx.Model]] | None = None,
) -> wrapper.Wrapper:
    if vision:
        env = wrapper.MadronaWrapper(env, num_vision_envs, randomization_fn)
    elif randomization_fn is None:
        env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = wrapper.BraxAutoResetWrapper(env)
    env = SamplePFWrapper(env)
    return env
