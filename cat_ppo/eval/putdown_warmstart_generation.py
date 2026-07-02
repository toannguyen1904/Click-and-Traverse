"""Offline warm-start state generation for PUTDOWN training.

Rolls out a trained two-agent CaTra (box carry & traverse) policy on a single
obstacle scene and snapshots the state at the moment each env first reaches the
goal (base_x >= goal_x) while still holding the box.  Those "arrived at the
destination, still carrying the box" states are the natural precondition for a
PUTDOWN policy (place the box on a supporting surface).

Mirrors cat_ppo/eval/warmstart_generation.py (pickup -> CaTra), but:
  * the rollout policy is a two-agent CaTra policy (lower legs + upper arms),
    loaded from a Brax checkpoint and combined into one JAX inference fn;
  * the env is the real G1CaTra2A MJX training env with pf_config.path set to the
    chosen scene, warm-started from the CaTra init states (catra_pickup_states_*.npz);
  * a state is kept only if the env reaches the goal while alive (no fall / box
    drop / obstacle collision) AND the lowest box corner is still above the
    supporting-surface height (default 0.5 m), i.e. the box is still held.

Saved .npz matches the existing warm-start format (qpos, qvel, box_mass, box_size
+ metadata) so a PUTDOWN env can reuse the same warm-start loading machinery.

Public entry-points:
  load_catra_2a_inference_fn(...)  – build a combined 2A JAX inference fn
  generate_putdown_states(...)     – roll out + snapshot + save
"""

import functools
import os
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from tqdm import tqdm

from mujoco_playground import wrapper as mp_wrapper

from cat_ppo.envs.g1.env_catra import BOX_QPOS_START
from cat_ppo.eval.warmstart_generation import _quat_rotate  # (w,x,y,z) quat rotation of a vec

# Box corner sign pattern (matches env_catra._box_corners_from_pose corner_signs order).
_CORNER_SIGNS = np.array([
    [-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.], [-1.,  1.,  1.],
    [ 1., -1., -1.], [ 1., -1.,  1.], [ 1.,  1., -1.], [ 1.,  1.,  1.],
], dtype=np.float32)


def _lowest_corner_z(box_pos: np.ndarray, box_quat: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    """Lowest world-z of the 8 box corners, per env.

    box_pos (B,3), box_quat (B,4) in (w,x,y,z), box_size (B,3) half-extents. Mirrors
    env_catra._box_corners_from_pose: world_corner = box_pos + R(box_quat) @ (signs * size).
    """
    local = _CORNER_SIGNS[None, :, :] * box_size[:, None, :]      # (B, 8, 3)
    world = _quat_rotate(box_quat[:, None, :], local)             # (B, 8, 3), broadcasts quat over corners
    world_z = world[..., 2] + box_pos[:, None, 2]                 # (B, 8)
    return world_z.min(axis=1)                                    # (B,)


def load_catra_2a_inference_fn(exp_name: str, task: str = "G1CaTra2A"):
    """Load a deterministic two-agent CaTra policy as one JAX inference fn.

    Reconstructs the two actor networks from the task's policy_config.network_factory,
    loads the checkpoint's Brax params (5-tuple: normalizer, policy_lower, policy_upper,
    value_lower, value_upper), and returns fn(obs_dict, rng) -> (action, {}) with the
    lower- and upper-body actions concatenated [lower, upper] (matching the env action
    ordering and the two-agent ONNX export).
    """
    import cat_ppo
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from cat_ppo.learning.policy.ppo.networks_2a import make_ppo_networks_2a

    ckpt = cat_ppo.get_latest_ckpt(exp_name)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for exp_name={exp_name}")
    print(f"Loading two-agent CaTra policy from:\n  {ckpt}")
    params = ppo_checkpoint.load(str(ckpt))

    task_cfg = cat_ppo.registry.get(task, "config")
    env_cfg = task_cfg.env_config
    obs_size = {"state": (env_cfg.num_obs,), "privileged_state": (env_cfg.num_pri,)}
    net = make_ppo_networks_2a(
        obs_size,
        action_size_lower=env_cfg.num_act_lower,
        action_size_upper=env_cfg.num_act_upper,
        preprocess_observations_fn=lambda x, y: x,   # identity: matches training (no obs norm)
        **dict(task_cfg.policy_config.network_factory),
    )
    normalizer, pl, pu = params[0], params[1], params[2]

    def inference_fn(obs, rng):
        a_lower = net.parametric_action_distribution_lower.mode(
            net.policy_network_lower.apply(normalizer, pl, obs))
        a_upper = net.parametric_action_distribution_upper.mode(
            net.policy_network_upper.apply(normalizer, pu, obs))
        return jp.concatenate([a_lower, a_upper], axis=-1), {}

    return inference_fn


def _read_box_use_inflation(exp_name, fallback):
    """Read env_config.box_use_inflation from the run's checkpoints/config.json so the
    generated rollout matches training. Falls back to the given value if unavailable."""
    import cat_ppo
    if not exp_name:
        return fallback
    cfg_path = cat_ppo.get_path_log(exp_name) / "checkpoints" / "config.json"
    if not cfg_path.exists():
        print(f"[putdown_gen] config.json not found at {cfg_path}; using box_inflation={fallback}")
        return fallback
    try:
        import json
        saved = json.loads(cfg_path.read_text())
        return bool(saved["env_config"]["box_use_inflation"])
    except (KeyError, ValueError):
        print(f"[putdown_gen] box_use_inflation absent from {cfg_path}; using box_inflation={fallback}")
        return fallback


def _set_box_noise(env_cfg, enabled: bool):
    """Toggle box position/orientation observation noise (no-op on older configs)."""
    scales = getattr(getattr(env_cfg, "noise_config", None), "scales", None)
    if scales is None or not hasattr(scales, "box_pos"):
        return
    if not enabled:
        scales.box_pos = 0.0
        scales.box_ori = 0.0
        print("[putdown_gen] box tracking noise DISABLED (box_pos/box_ori scales set to 0)")


def generate_putdown_states(
    exp_name: str,
    obs_path: str,
    warmstart_states_path: str,
    num_states: int,
    goal_x: float,
    surface_z: float,
    output_path: str,
    task: str = "G1CaTra2A",
    episode_length: int = None,
    batch_size: int = None,
    max_batches: int = 50,
    seed: int = 0,
    box_inflation: bool = True,
    box_noise: bool = True,
) -> None:
    """Generate and save PUTDOWN warm-start states from a trained two-agent CaTra policy.

    Args:
        exp_name: CaTra run whose checkpoint holds the policy (resolved via get_latest_ckpt).
        obs_path: Scene / obstacle-field directory (e.g. data/assets/TypiObs/bar2/). Every
            env in every batch runs on this single scene.
        warmstart_states_path: CaTra init states (.npz) the rollout is warm-started from
            (robot already holding the box; e.g. data/warmstart/catra_pickup_states_9.0.npz).
        num_states: Number of valid states to save.
        goal_x: Base x (m) counted as reaching the goal; snapshot taken at first crossing.
        surface_z: Supporting-surface height (m). A state is valid only if the lowest box
            corner is above this at the snapshot (box still held, clears the surface).
        output_path: Where to write the .npz.
        task: Two-agent CaTra task name (env + network config).
        episode_length: Rollout length per batch (default: env_config.episode_length).
        batch_size: Envs per batch (default: num_states). Lower if the GPU OOMs.
        max_batches: Safety cap on batches before giving up.
        seed: PRNG seed.
        box_inflation: Fallback for box_use_inflation when config.json is unavailable.
        box_noise: Keep box tracking noise in the deployable obs (matches training).
    """
    import cat_ppo
    from cat_ppo.envs.g1.env_catra_2a import G1CaTra2AEnv
    from cat_ppo.envs.g1.env_catra import make_warmstart_domain_randomize_catra_indexed
    from cat_ppo.envs.g1 import constants as consts

    # --- Build env config for the chosen scene + warm-start init ---
    task_cfg = cat_ppo.registry.get(task, "config")
    env_cfg = task_cfg.env_config
    env_cfg.pf_config.path = obs_path
    env_cfg.warmstart_states_path = warmstart_states_path
    if hasattr(env_cfg, "box_use_inflation"):
        env_cfg.box_use_inflation = _read_box_use_inflation(exp_name, box_inflation)
        print(f"[putdown_gen] box_use_inflation = {env_cfg.box_use_inflation}")
    _set_box_noise(env_cfg, box_noise)

    if episode_length is None:
        episode_length = int(env_cfg.episode_length)
    if batch_size is None:
        batch_size = num_states
    batch_size = int(batch_size)

    env = G1CaTra2AEnv(task_type=env_cfg.task_type, config=env_cfg)
    # BraxDomainRandomizationVmapWrapper.reset/step mutate env.unwrapped._mjx_model with a
    # per-env vmap tracer that leaks onto the shared env; capture the pristine model so each
    # batch can restore it before re-wrapping (same fix as warmstart_generation).
    base_mjx_model = env.unwrapped._mjx_model

    # IDs for reading per-env box parameters after DR.
    _mj = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    box_body_id = _mj.body("carried_box").id
    box_geom_id = _mj.geom("box_geom").id
    del _mj

    # Number of warm-start init states available (to permute indices per batch).
    n_ws = int(np.load(warmstart_states_path)["qpos"].shape[0])

    inference_fn = load_catra_2a_inference_fn(exp_name, task)
    # Inference depends only on obs shape, so it compiles once and is reused across batches.
    act_fn = jax.jit(jax.vmap(inference_fn))

    def run_one_batch(batch_rng, index_offset):
        """Roll out one batch of `batch_size` envs; return valid-state arrays + stats."""
        env.unwrapped._mjx_model = base_mjx_model

        # Each batch draws a different slice of the warm-start pool for diversity.
        indices = (jp.arange(batch_size) + index_offset) % n_ws
        dr_fn = make_warmstart_domain_randomize_catra_indexed(warmstart_states_path, indices)
        batch_rng, dr_key = jax.random.split(batch_rng)
        dr_rngs = jax.random.split(dr_key, batch_size)
        v_dr_fn = functools.partial(dr_fn, rng=dr_rngs)
        wrapped_env = mp_wrapper.BraxDomainRandomizationVmapWrapper(env, v_dr_fn)

        # Per-env box parameters from the DR'd model (from the warm-start states).
        box_mass = np.array(wrapped_env._mjx_model_v.body_mass[:, box_body_id])   # (batch_size,)
        box_size = np.array(wrapped_env._mjx_model_v.geom_size[:, box_geom_id])   # (batch_size, 3)

        batch_rng, reset_key = jax.random.split(batch_rng)
        reset_rngs = jax.random.split(reset_key, batch_size)
        state = jax.jit(wrapped_env.reset)(reset_rngs)

        # Snapshot accumulators (updated the first step each env crosses the goal while alive).
        snap_qpos = state.data.qpos
        snap_qvel = state.data.qvel
        snap_box_xpos = state.data.xpos[:, box_body_id]
        snap_box_xquat = state.data.xquat[:, box_body_id]
        ever_reached = jp.zeros((batch_size,), dtype=bool)
        ever_done = jp.zeros((batch_size,), dtype=bool)

        step_fn = jax.jit(wrapped_env.step)
        for _ in tqdm(range(episode_length), leave=False):
            batch_rng, act_key = jax.random.split(batch_rng)
            act_rngs = jax.random.split(act_key, batch_size)
            action, _ = act_fn(state.obs, act_rngs)
            state = step_fn(state, action)

            alive = ~ever_done
            base_x = state.data.qpos[:, 0]
            reached_now = (base_x >= goal_x) & alive
            first_reach = reached_now & ~ever_reached

            snap_qpos = jp.where(first_reach[:, None], state.data.qpos, snap_qpos)
            snap_qvel = jp.where(first_reach[:, None], state.data.qvel, snap_qvel)
            snap_box_xpos = jp.where(first_reach[:, None], state.data.xpos[:, box_body_id], snap_box_xpos)
            snap_box_xquat = jp.where(first_reach[:, None], state.data.xquat[:, box_body_id], snap_box_xquat)

            ever_reached = ever_reached | reached_now
            ever_done = ever_done | (state.done > 0.5)

        snap_qpos_np = np.array(snap_qpos)
        snap_qvel_np = np.array(snap_qvel)
        reached_np = np.array(ever_reached)
        low_corner_z = _lowest_corner_z(np.array(snap_box_xpos), np.array(snap_box_xquat), box_size)

        box_held_mask = low_corner_z > surface_z
        nan_mask = np.any(np.isnan(snap_qpos_np), axis=1) | np.any(np.isnan(snap_qvel_np), axis=1)
        valid_mask = reached_np & box_held_mask & ~nan_mask

        stats = dict(
            n_reached=int(reached_np.sum()),
            n_not_reached=int((~reached_np).sum()),
            n_box_low=int((reached_np & ~box_held_mask & ~nan_mask).sum()),
            n_nan=int(nan_mask.sum()),
            n_valid=int(valid_mask.sum()),
        )
        batch_valid = dict(
            qpos=snap_qpos_np[valid_mask],
            qvel=snap_qvel_np[valid_mask],
            box_mass=box_mass[valid_mask],
            box_size=box_size[valid_mask],
            low_corner_z=low_corner_z[valid_mask],
        )
        return batch_valid, stats

    rng = jax.random.PRNGKey(seed)
    print(
        f"Collecting {num_states} valid PUTDOWN states in batches of {batch_size} "
        f"(up to {max_batches} batches); rolling out {episode_length} steps each on scene "
        f"'{obs_path}' (goal_x={goal_x}, surface_z={surface_z})."
    )

    collected = {k: [] for k in ("qpos", "qvel", "box_mass", "box_size", "low_corner_z")}
    n_collected = 0
    n_run = 0
    batch_idx = -1
    for batch_idx in range(max_batches):
        rng, batch_rng = jax.random.split(rng)
        batch_valid, stats = run_one_batch(batch_rng, index_offset=(batch_idx * batch_size) % n_ws)
        n_run += batch_size
        for k in collected:
            collected[k].append(batch_valid[k])
        n_collected += stats["n_valid"]

        print(
            f"[batch {batch_idx + 1}] valid {stats['n_valid']}/{batch_size} "
            f"(not-reached {stats['n_not_reached']}, box-low {stats['n_box_low']}, nan {stats['n_nan']}) "
            f"— total {n_collected}/{num_states}"
        )
        if n_collected >= num_states:
            break

    if n_collected < num_states:
        raise RuntimeError(
            f"Only {n_collected} valid states after {batch_idx + 1} batches ({n_run} envs), "
            f"but {num_states} requested. Increase --max_batches (currently {max_batches}), "
            f"lower --goal_x, or check policy quality on this scene."
        )

    qpos_all = np.concatenate(collected["qpos"], axis=0)[:num_states]
    qvel_all = np.concatenate(collected["qvel"], axis=0)[:num_states]
    box_mass_all = np.concatenate(collected["box_mass"], axis=0)[:num_states]
    box_size_all = np.concatenate(collected["box_size"], axis=0)[:num_states]
    low_corner_z_all = np.concatenate(collected["low_corner_z"], axis=0)[:num_states]

    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        qpos=qpos_all,
        qvel=qvel_all,
        box_mass=box_mass_all,
        box_size=box_size_all,
        # metadata (0-dim arrays)
        exp_name=np.array(exp_name),
        task=np.array(task),
        obs_path=np.array(obs_path),
        warmstart_states_path=np.array(warmstart_states_path),
        goal_x=np.array(goal_x),
        surface_z=np.array(surface_z),
        seed=np.array(seed),
        num_states=np.array(num_states),
    )
    print(f"Saved {num_states} states ({n_collected} valid from {n_run} envs over {batch_idx + 1} batches) → {output_path}")
    print(f"  qpos shape: {qpos_all.shape}  qvel shape: {qvel_all.shape}")
    print(f"  box_mass range: [{box_mass_all.min():.3f}, {box_mass_all.max():.3f}]")
    print(f"  box_size x range: [{box_size_all[:, 0].min():.3f}, {box_size_all[:, 0].max():.3f}]")
    print(f"  lowest box corner z range: [{low_corner_z_all.min():.3f}, {low_corner_z_all.max():.3f}] (surface_z={surface_z})")
