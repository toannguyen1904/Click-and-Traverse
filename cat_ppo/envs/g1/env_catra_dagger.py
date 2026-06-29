"""DAgger distillation task for G1CaTra.

Provides ``G1CaTraDagger``: a G1CaTra student rollout env with per-scene
potential-field routing so that one privileged specialist per scene can act as a
DAgger teacher. Teachers may be single-agent (``G1CaTraPri``) or two-agent
(``G1CaTra2APri``); a given run uses one kind, never a mix (selected via
``dagger_config.teacher_kind`` / ``train_ppo_dagger.py --teacher_kind``).

The teacher policy observation is exposed under ``obs["teacher_state"]``. Because
both privileged variants define their actor state as ``privileged_state[:-31]``
(see env_catra_pri.py), the teacher obs here is simply the student's noiseless
333-dim ``privileged_state`` with the 31 domain-randomization tail dims stripped —
no separate reconstruction is needed.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

import cat_ppo
from cat_ppo.envs.g1.env_catra import (
    G1CaTraEnv,
    g1_catra_task_config,
    command_to_reference,
)
from cat_ppo.envs.g1.env_catra_2a import G1CaTra2AEnv, g1_catra_2a_task_config
from cat_ppo.envs.g1.env_catra_pri import _DR_TAIL_DIMS


def _add_dagger_config(
    config: config_dict.ConfigDict, default_teacher_kind: str = "single"
) -> config_dict.ConfigDict:
    config.policy_config.dagger_config = config_dict.create(
        enable=True,
        loss="kl",
        # "single" -> G1CaTraPri teachers (one 20-dim policy net per teacher).
        # "2a"     -> G1CaTra2APri teachers (lower 12-dim + upper 8-dim nets).
        # A single-agent student requires single-agent teachers and vice versa
        # (enforced in train_ppo_dagger.py: single-single / two-two).
        teacher_kind=default_teacher_kind,
        teacher_restore_names=[],
        teacher_checkpoint_paths=[],
        # Hidden-layer sizes used to rebuild the two-agent teacher policy nets
        # (only consumed when teacher_kind == "2a"). Filled in from the teacher
        # config.json by train_ppo_dagger.py.
        teacher_policy_hidden_layer_sizes=[],
        kl_eps=1e-5,
        # Schedule: "two_phase" (DAgger KL until dagger_timesteps, then PPO) or
        # "blend" (every step: lambda_ppo*L_PPO + lambda_dagger*L_DAgger, with
        # lambda_dagger = max(blend_lambda_floor, 1 - env_steps/blend_anneal_timesteps)
        # so DAgger is never fully turned off). dagger_timesteps is only used by
        # two_phase; blend_anneal_timesteps (0 -> num_timesteps//2) by blend.
        dagger_mode="two_phase",
        dagger_timesteps=0,
        blend_lambda_floor=0.1,
        blend_anneal_timesteps=0,
        actor_loss_scale=1.0,
        value_loss_scale=1.0,
        # The student rollout env additionally exposes the privileged-teacher
        # policy observation (num_obs=302) under this key so G1CaTraPri /
        # G1CaTra2APri specialists can be used as DAgger teachers even though the
        # G1CaTra student policy observation ("state") is num_obs=239.
        teacher_obs_key="teacher_state",
    )
    config.env_config.pf_config.sampling_weights = []
    config.env_config.pf_config.sampling_alpha = 1.0
    config.env_config.pf_config.sampling_ema_decay = 0.95
    return config


def g1_catra_dagger_task_config() -> config_dict.ConfigDict:
    return _add_dagger_config(g1_catra_task_config(), default_teacher_kind="single")


def g1_catra_2a_dagger_task_config() -> config_dict.ConfigDict:
    return _add_dagger_config(g1_catra_2a_task_config(), default_teacher_kind="2a")


def _load_pf_fields(config: config_dict.ConfigDict):
    """Load per-scene sdf/bf/gf (+ box guidance field) for every DAgger scene.

    Returns stacked-shape-validated lists plus the (unnormalized) sampling
    weights. The box guidance field matches G1CaTraEnv._post_init_catra: the
    inflation field (gf_inflation.npy) when box_use_inflation, else the regular gf.
    """
    pf_paths = list(getattr(config.pf_config, "paths", []))
    if not pf_paths:
        pf_paths = [config.pf_config.path]

    use_inflation = bool(getattr(config, "box_use_inflation", True))
    sdf, bf, gf, gf_box = [], [], [], []
    for pf_path in pf_paths:
        sdf.append(np.load(f"{pf_path}/sdf.npy")[..., None])
        bf.append(np.load(f"{pf_path}/bf.npy"))
        gf_arr = np.load(f"{pf_path}/gf.npy")
        gf.append(gf_arr)
        gf_box.append(np.load(f"{pf_path}/gf_inflation.npy") if use_inflation else gf_arr)

    sdf_shape, bf_shape, gf_shape = sdf[0].shape, bf[0].shape, gf[0].shape
    for pf_path, sdf_arr, bf_arr, gf_arr, gfb_arr in zip(pf_paths, sdf, bf, gf, gf_box):
        if (
            sdf_arr.shape != sdf_shape
            or bf_arr.shape != bf_shape
            or gf_arr.shape != gf_shape
            or gfb_arr.shape != gf_shape
        ):
            raise ValueError(
                "All DAgger pf_config.paths must have matching sdf/bf/gf/gf_inflation "
                f"shapes; got mismatch at {pf_path}"
            )

    sampling_weights = list(getattr(config.pf_config, "sampling_weights", []))
    if not sampling_weights:
        sampling_weights = [1.0] * len(pf_paths)
    if len(sampling_weights) != len(pf_paths):
        raise ValueError(
            f"pf_config.sampling_weights length must match pf paths: "
            f"{len(sampling_weights)} != {len(pf_paths)}"
        )
    sampling_weights = np.asarray(sampling_weights, dtype=np.float32)
    if np.any(sampling_weights < 0.0) or not np.any(sampling_weights > 0.0):
        raise ValueError("pf_config.sampling_weights must be non-negative with at least one positive value")

    return pf_paths, sdf, bf, gf, gf_box, sampling_weights


cat_ppo.registry.register("G1CaTraDagger", "config")(g1_catra_dagger_task_config())
cat_ppo.registry.register("G1CaTra2ADagger", "config")(g1_catra_2a_dagger_task_config())


class _DaggerSceneMixin:
    """Per-scene potential-field routing for G1CaTra DAgger.

    Loads every teacher's scene fields stacked along a leading scene axis and
    samples a scene per episode. ``sample_field`` is overridden to index the
    active scene so the student rollout (and the teacher obs it exposes) always
    queries the scene that matches the selected teacher (``pf_id``).
    """

    def __init__(
        self,
        task_type: str = "flat_terrain_catra",
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task_type=task_type,
            config=config,
            config_overrides=config_overrides,
        )
        _, sdf, bf, gf, gf_box, sampling_weights = _load_pf_fields(config)

        self.num_pf_scenes = len(sdf)
        self.sdf = jp.array(np.stack(sdf, axis=0))
        self.bf = jp.array(np.stack(bf, axis=0))
        self.gf = jp.array(np.stack(gf, axis=0))
        self.gf_box = jp.array(np.stack(gf_box, axis=0))
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape[1:]
        self._field_pf_id = jp.array(0, dtype=jp.int32)
        sampling_weights = sampling_weights / sampling_weights.sum()
        self._pf_sampling_logits = jp.log(jp.array(sampling_weights) + 1e-8)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pf_key = jax.random.split(rng)
        self._field_pf_id = jax.random.categorical(pf_key, self._pf_sampling_logits).astype(jp.int32)
        state = super().reset(rng)
        state.info["pf_id"] = self._field_pf_id
        return state

    def reset_with_pf_id(self, rng: jax.Array, pf_id: jax.Array) -> mjx_env.State:
        self._field_pf_id = pf_id.astype(jp.int32)
        state = super().reset(rng)
        state.info["pf_id"] = self._field_pf_id
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        self._field_pf_id = state.info["pf_id"].astype(jp.int32)
        return super().step(state, action)

    def sample_field(self, field, pos):
        return super().sample_field(field[self._field_pf_id], pos)


def _teacher_state_obs(obs: mjx_env.Observation) -> jax.Array:
    """Privileged-teacher policy observation (302-dim) = noiseless privileged state
    minus the 31 domain-randomization tail dims. Matches the actor `state` of both
    G1CaTraPri and G1CaTra2APri."""
    return obs["privileged_state"][:-_DR_TAIL_DIMS]


@cat_ppo.registry.register("G1CaTraDagger", "train_env_class")
class G1CaTraDaggerEnv(_DaggerSceneMixin, G1CaTraEnv):
    """G1CaTra (single-agent) with per-scene PF routing for DAgger teachers."""

    def _get_obs(self, data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        obs = super()._get_obs(data, info, feet_contact)
        obs["teacher_state"] = _teacher_state_obs(obs)
        return obs


@cat_ppo.registry.register("G1CaTra2ADagger", "train_env_class")
class G1CaTra2ADaggerEnv(_DaggerSceneMixin, G1CaTra2AEnv):
    """G1CaTra2A (two-agent) with per-scene PF routing for DAgger teachers.

    Inherits the two-agent action split + per-agent reward streams from
    G1CaTra2AEnv, and adds the scene routing / teacher observation needed for
    distillation from G1CaTra2APri specialists.
    """

    def _get_obs(self, data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        obs = super()._get_obs(data, info, feet_contact)
        obs["teacher_state"] = _teacher_state_obs(obs)
        return obs


cat_ppo.registry.register("G1CaTraDagger", "command_to_reference_fn")(command_to_reference)
cat_ppo.registry.register("G1CaTra2ADagger", "command_to_reference_fn")(command_to_reference)
