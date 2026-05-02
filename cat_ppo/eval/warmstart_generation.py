"""Offline warm-start state generation for CaTra training.

Rolls out a trained pickup policy with full domain randomization to produce
N diverse "robot holding box" terminal states.  Saves (qpos, qvel, box_mass,
box_size) per state so CaTra training can load them directly at reset —
no live rollout required.

Sampling strategy:
  - For each env, a snapshot step K is drawn uniformly from [sample_start, sample_end].
  - A state is kept only if, lookahead steps after K, the box z is still above
    the pillar top (0.6 m) + box half-z — i.e. the robot is still holding the box.
  - States that fail this check are discarded; the generator oversamples by `oversample`x
    and truncates to exactly `num_states` valid states.

Public entry-point: generate_warmstart_states(...)
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

# qpos layout: [0:7] root, [7:36] robot joints, [36:43] box freejoint
BOX_QPOS_START = 36
# pillar top z = surface_z (0.3) + support_half_z (0.3)
PILLAR_TOP_Z = 0.6


def generate_warmstart_states(
    pickup_checkpoint_path: str,
    num_states: int,
    sample_start: int,
    sample_end: int,
    lookahead: int,
    oversample: float,
    seed: int,
    output_path: str,
) -> None:
    """Generate and save warm-start states from the trained pickup policy.

    Args:
        pickup_checkpoint_path: Path to the Brax orbax checkpoint directory for
            the trained G1Pickup policy.
        num_states: Number of valid states to save (should match CaTra num_envs).
        sample_start: Earliest step from which a snapshot can be taken (inclusive).
        sample_end: Latest step from which a snapshot can be taken (inclusive).
        lookahead: Steps after the snapshot to check that the box has not dropped.
            A state is valid only if box_z > PILLAR_TOP_Z + box_half_z + 8cm at step K+lookahead.
        oversample: Multiplier on num_states for the number of envs to run (e.g. 1.5
            runs 1.5x envs so that after filtering there are enough valid states).
        seed: PRNG seed for reproducibility.
        output_path: Where to write the .npz file.
    """
    from cat_ppo.envs.g1.env_pickup import G1PickupEnv, g1_pickup_task_config, domain_randomize_pickup
    from cat_ppo.envs.g1.pickup_warmstart import load_pickup_inference_fn
    from cat_ppo.envs.g1 import constants as consts

    task_cfg = g1_pickup_task_config()
    env = G1PickupEnv(task_type=task_cfg.env_config.task_type, config=task_cfg.env_config)

    # IDs for reading per-env box parameters after DR
    _mj = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    box_body_id = _mj.body("carried_box").id
    box_geom_id = _mj.geom("box_geom").id
    del _mj

    print(f"Loading pickup inference fn from:\n  {pickup_checkpoint_path}")
    inference_fn = load_pickup_inference_fn(pickup_checkpoint_path)

    rng = jax.random.PRNGKey(seed)

    n_envs = int(np.ceil(num_states * oversample))
    print(f"Oversampling: running {n_envs} envs (x{oversample}) to collect {num_states} valid states.")

    # Apply pickup DR to get per-env model (vmapped)
    rng, dr_key = jax.random.split(rng)
    dr_rngs = jax.random.split(dr_key, n_envs)
    v_dr_fn = functools.partial(domain_randomize_pickup, rng=dr_rngs)
    wrapped_env = mp_wrapper.BraxDomainRandomizationVmapWrapper(env, v_dr_fn)

    # Extract per-env box parameters from the DR'd model
    # body_mass/geom_size shapes: (n_envs, nbody/ngeom) after vmap
    box_mass_per_env = np.array(wrapped_env._mjx_model_v.body_mass[:, box_body_id])   # (n_envs,)
    box_size_per_env = np.array(wrapped_env._mjx_model_v.geom_size[:, box_geom_id])   # (n_envs, 3)

    # Reset all envs
    rng, reset_key = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_key, n_envs)
    print(f"JIT-compiling reset and running {n_envs} envs ...")
    state = jax.jit(wrapped_env.reset)(reset_rngs)

    # Sample per-env snapshot step K ∈ [sample_start, sample_end]
    rng, k_key = jax.random.split(rng)
    k_rngs = jax.random.split(k_key, n_envs)
    K_per_env = jax.vmap(lambda k: jax.random.randint(k, (), sample_start, sample_end + 1))(k_rngs)

    # Initialize accumulators
    snap_qpos = state.data.qpos                        # (n_envs, nq) — overwritten at step K
    snap_qvel = state.data.qvel                        # (n_envs, nv) — overwritten at step K
    future_box_z = jp.full((n_envs,), -jp.inf)         # box z at step K+lookahead

    # JIT compile step and inference once
    step_fn = jax.jit(wrapped_env.step)
    act_fn = jax.jit(jax.vmap(inference_fn))

    total_steps = sample_end + lookahead
    print(
        f"JIT-compiling step+inference and rolling out {total_steps} steps "
        f"(sample [{sample_start}, {sample_end}], lookahead {lookahead}) ..."
    )
    for t in tqdm(range(total_steps)):
        rng, act_key = jax.random.split(rng)
        act_rngs = jax.random.split(act_key, n_envs)
        action, _ = act_fn(state.obs, act_rngs)
        state = step_fn(state, action)

        # Snapshot qpos/qvel for envs whose sample step ends here (0-indexed: K-1)
        is_snap = (t == K_per_env - 1)
        snap_qpos = jp.where(is_snap[:, None], state.data.qpos, snap_qpos)
        snap_qvel = jp.where(is_snap[:, None], state.data.qvel, snap_qvel)

        # Record box z at step K+lookahead to verify the box is still held
        is_future = (t == K_per_env + lookahead - 1)
        cur_box_z = state.data.qpos[:, BOX_QPOS_START + 2]
        future_box_z = jp.where(is_future, cur_box_z, future_box_z)

    snap_qpos_np = np.array(snap_qpos)
    snap_qvel_np = np.array(snap_qvel)
    future_box_z_np = np.array(future_box_z)

    # Validity: box must still be above pillar top + box half-z at K+lookahead
    box_half_z_per_env = box_size_per_env[:, 2]
    box_held_mask = future_box_z_np > (PILLAR_TOP_Z + box_half_z_per_env + 0.08)

    # NaN check
    nan_mask = np.any(np.isnan(snap_qpos_np), axis=1) | np.any(np.isnan(snap_qvel_np), axis=1)
    valid_mask = box_held_mask & ~nan_mask

    n_nan = int(nan_mask.sum())
    n_dropped = int((~box_held_mask & ~nan_mask).sum())
    n_valid = int(valid_mask.sum())

    if n_nan > 0:
        print(f"WARNING: {n_nan}/{n_envs} states contain NaNs — excluded.")
    if n_dropped > 0:
        print(f"WARNING: {n_dropped}/{n_envs} states failed the box-held check (dropped within {lookahead} steps) — excluded.")
    print(f"Valid states: {n_valid}/{n_envs} ({100.0 * n_valid / n_envs:.1f}%)")

    if n_valid < num_states:
        raise RuntimeError(
            f"Only {n_valid} valid states after filtering, but {num_states} requested. "
            f"Increase --oversample (currently {oversample}) or check policy quality."
        )

    # Filter then truncate to exactly num_states
    snap_qpos_np     = snap_qpos_np[valid_mask][:num_states]
    snap_qvel_np     = snap_qvel_np[valid_mask][:num_states]
    box_mass_per_env = box_mass_per_env[valid_mask][:num_states]
    box_size_per_env = box_size_per_env[valid_mask][:num_states]

    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        qpos=snap_qpos_np,
        qvel=snap_qvel_np,
        box_mass=box_mass_per_env,
        box_size=box_size_per_env,
        # metadata (stored as 0-dim arrays)
        pickup_checkpoint_path=np.array(pickup_checkpoint_path),
        seed=np.array(seed),
        sample_start=np.array(sample_start),
        sample_end=np.array(sample_end),
        lookahead=np.array(lookahead),
        num_states=np.array(num_states),
    )
    print(f"Saved {num_states} states ({n_valid} valid from {n_envs} envs) → {output_path}")
    print(f"  qpos shape: {snap_qpos_np.shape}  qvel shape: {snap_qvel_np.shape}")
    print(f"  box_mass range: [{box_mass_per_env.min():.3f}, {box_mass_per_env.max():.3f}]")
    print(f"  box_size x range: [{box_size_per_env[:,0].min():.3f}, {box_size_per_env[:,0].max():.3f}]")
    print(f"  future_box_z range (valid only): [{future_box_z_np[valid_mask].min():.3f}, {future_box_z_np[valid_mask].max():.3f}]")
