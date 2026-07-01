"""Offline warm-start state generation for CaTra training.

Rolls out a trained pickup policy with full domain randomization to produce
N diverse "robot holding box" terminal states.  Saves (qpos, qvel, box_mass,
box_size) per state so CaTra training can load them directly at reset —
no live rollout required.

Sampling strategy:
  - For each env, a snapshot step K is drawn uniformly from [sample_start, sample_end].
  - A state is kept only if, lookahead steps after K, the box z is still above
    the pillar top (0.6 m) + box half-z — i.e. the robot is still holding the box.
  - A state is also kept only if both palms, projected onto the box's side-face
    plane, lie within HAND_FACE_PROJ_TOL (5 cm) of the face centers — i.e. the
    hands are squarely on the box rather than gripping near an edge.
  - States that fail these checks are discarded. Rather than oversampling a single
    huge batch (which can OOM the GPU), the generator runs fixed-size batches of
    `batch_size` envs sequentially, accumulating valid states across batches until
    it has at least `num_states`, then truncates to exactly `num_states`.

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
# Max allowed in-plane offset of a hand from the center of its box face (m).
HAND_FACE_PROJ_TOL = 0.05


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec (..., 3) by MuJoCo quaternion quat (..., 4) in (w, x, y, z) order."""
    w = quat[..., 0:1]
    qvec = quat[..., 1:4]
    t = 2.0 * np.cross(qvec, vec)
    return vec + w * t + np.cross(qvec, t)


def generate_warmstart_states(
    pickup_checkpoint_path: str,
    num_states: int,
    sample_start: int,
    sample_end: int,
    lookahead: int,
    seed: int,
    output_path: str,
    batch_size: int = None,
    max_batches: int = 50,
) -> None:
    """Generate and save warm-start states from the trained pickup policy.

    Runs fixed-size batches of `batch_size` envs sequentially, keeping the valid
    states from each batch, until at least `num_states` have been collected (or
    `max_batches` is reached). This keeps GPU memory bounded by `batch_size`
    rather than requiring one large oversampled batch.

    Args:
        pickup_checkpoint_path: Path to the Brax orbax checkpoint directory for
            the trained G1Pickup policy.
        num_states: Number of valid states to save (should match CaTra num_envs).
        sample_start: Earliest step from which a snapshot can be taken (inclusive).
        sample_end: Latest step from which a snapshot can be taken (inclusive).
        lookahead: Steps after the snapshot to check that the box has not dropped.
            A state is valid only if box_z > PILLAR_TOP_Z + box_half_z + 8cm at step K+lookahead.
        seed: PRNG seed for reproducibility.
        output_path: Where to write the .npz file.
        batch_size: Number of envs to roll out per batch. Defaults to num_states
            (1x at a time). Lower this if the GPU runs out of memory.
        max_batches: Safety cap on the number of batches before giving up.
    """
    from cat_ppo.envs.g1.env_pickup import G1PickupEnv, g1_pickup_task_config, domain_randomize_pickup
    from cat_ppo.envs.g1.pickup_warmstart import load_pickup_inference_fn
    from cat_ppo.envs.g1 import constants as consts

    if batch_size is None:
        batch_size = num_states
    batch_size = int(batch_size)

    task_cfg = g1_pickup_task_config()
    env = G1PickupEnv(task_type=task_cfg.env_config.task_type, config=task_cfg.env_config)
    # BraxDomainRandomizationVmapWrapper.reset/step mutate env.unwrapped._mjx_model
    # with a per-env vmap tracer, which leaks onto the shared env after a rollout.
    # Capture the pristine model so each batch can restore it before re-wrapping.
    base_mjx_model = env.unwrapped._mjx_model

    # IDs for reading per-env box parameters after DR
    _mj = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    box_body_id = _mj.body("carried_box").id
    box_geom_id = _mj.geom("box_geom").id
    lhand_id = _mj.site("left_palm").id
    rhand_id = _mj.site("right_palm").id
    del _mj

    print(f"Loading pickup inference fn from:\n  {pickup_checkpoint_path}")
    inference_fn = load_pickup_inference_fn(pickup_checkpoint_path)

    # Inference depends only on obs shape (batch_size, obs_dim), not on the DR'd
    # model, so it compiles once and is reused across all batches.
    act_fn = jax.jit(jax.vmap(inference_fn))
    total_steps = sample_end + lookahead

    def run_one_batch(batch_rng):
        """Roll out one batch of `batch_size` envs; return valid-state arrays + stats."""
        # Restore the pristine model so the wrapper reads it (not a leaked tracer
        # left on the shared env by the previous batch's vmapped reset/step).
        env.unwrapped._mjx_model = base_mjx_model

        # Apply pickup DR to get per-env model (vmapped)
        batch_rng, dr_key = jax.random.split(batch_rng)
        dr_rngs = jax.random.split(dr_key, batch_size)
        v_dr_fn = functools.partial(domain_randomize_pickup, rng=dr_rngs)
        wrapped_env = mp_wrapper.BraxDomainRandomizationVmapWrapper(env, v_dr_fn)

        # Per-env box parameters from the DR'd model (shapes: (batch_size, ...))
        box_mass = np.array(wrapped_env._mjx_model_v.body_mass[:, box_body_id])    # (batch_size,)
        box_size = np.array(wrapped_env._mjx_model_v.geom_size[:, box_geom_id])    # (batch_size, 3)

        # Reset all envs. A fresh wrapped_env is built per batch, so step/reset are
        # re-jitted here (batch_size is constant, so the trace shapes are identical).
        batch_rng, reset_key = jax.random.split(batch_rng)
        reset_rngs = jax.random.split(reset_key, batch_size)
        state = jax.jit(wrapped_env.reset)(reset_rngs)

        # Sample per-env snapshot step K ∈ [sample_start, sample_end]
        batch_rng, k_key = jax.random.split(batch_rng)
        k_rngs = jax.random.split(k_key, batch_size)
        K_per_env = jax.vmap(lambda k: jax.random.randint(k, (), sample_start, sample_end + 1))(k_rngs)

        # Snapshot accumulators (overwritten at each env's step K)
        snap_qpos = state.data.qpos
        snap_qvel = state.data.qvel
        snap_box_xpos  = state.data.xpos[:, box_body_id]
        snap_box_xquat = state.data.xquat[:, box_body_id]
        snap_lhand = state.data.site_xpos[:, lhand_id]
        snap_rhand = state.data.site_xpos[:, rhand_id]
        future_box_z = jp.full((batch_size,), -jp.inf)

        step_fn = jax.jit(wrapped_env.step)
        for t in tqdm(range(total_steps), leave=False):
            batch_rng, act_key = jax.random.split(batch_rng)
            act_rngs = jax.random.split(act_key, batch_size)
            action, _ = act_fn(state.obs, act_rngs)
            state = step_fn(state, action)

            # Snapshot for envs whose sample step ends here (0-indexed: K-1)
            is_snap = (t == K_per_env - 1)
            snap_qpos = jp.where(is_snap[:, None], state.data.qpos, snap_qpos)
            snap_qvel = jp.where(is_snap[:, None], state.data.qvel, snap_qvel)
            snap_box_xpos  = jp.where(is_snap[:, None], state.data.xpos[:, box_body_id], snap_box_xpos)
            snap_box_xquat = jp.where(is_snap[:, None], state.data.xquat[:, box_body_id], snap_box_xquat)
            snap_lhand = jp.where(is_snap[:, None], state.data.site_xpos[:, lhand_id], snap_lhand)
            snap_rhand = jp.where(is_snap[:, None], state.data.site_xpos[:, rhand_id], snap_rhand)

            # Record box z at step K+lookahead to verify the box is still held
            is_future = (t == K_per_env + lookahead - 1)
            cur_box_z = state.data.qpos[:, BOX_QPOS_START + 2]
            future_box_z = jp.where(is_future, cur_box_z, future_box_z)

        snap_qpos_np = np.array(snap_qpos)
        snap_qvel_np = np.array(snap_qvel)
        future_box_z_np = np.array(future_box_z)
        snap_box_xpos_np  = np.array(snap_box_xpos)
        snap_box_xquat_np = np.array(snap_box_xquat)
        snap_lhand_np = np.array(snap_lhand)
        snap_rhand_np = np.array(snap_rhand)

        # Validity: box must still be above pillar top + box half-z at K+lookahead
        box_held_mask = future_box_z_np > (PILLAR_TOP_Z + box_size[:, 2] + 0.08)

        # Hand-centering: each palm, projected onto its box face plane, must lie within
        # HAND_FACE_PROJ_TOL of the face center. The two side faces share the box's local
        # +Y axis as normal, so the in-plane offset from the face center equals the
        # component of (hand - box_center) perpendicular to that axis.
        box_left_axis = _quat_rotate(snap_box_xquat_np, np.array([0.0, 1.0, 0.0]))  # (batch_size, 3)
        box_left_axis = box_left_axis / np.linalg.norm(box_left_axis, axis=1, keepdims=True)

        def _proj_offset(hand):
            vec = hand - snap_box_xpos_np
            normal_comp = np.sum(vec * box_left_axis, axis=1, keepdims=True) * box_left_axis
            return np.linalg.norm(vec - normal_comp, axis=1)

        hands_centered_mask = (
            (_proj_offset(snap_lhand_np) < HAND_FACE_PROJ_TOL)
            & (_proj_offset(snap_rhand_np) < HAND_FACE_PROJ_TOL)
        )

        nan_mask = np.any(np.isnan(snap_qpos_np), axis=1) | np.any(np.isnan(snap_qvel_np), axis=1)
        valid_mask = box_held_mask & hands_centered_mask & ~nan_mask

        stats = dict(
            n_nan=int(nan_mask.sum()),
            n_dropped=int((~box_held_mask & ~nan_mask).sum()),
            n_off_center=int((box_held_mask & ~hands_centered_mask & ~nan_mask).sum()),
            n_valid=int(valid_mask.sum()),
        )
        batch_valid = dict(
            qpos=snap_qpos_np[valid_mask],
            qvel=snap_qvel_np[valid_mask],
            box_mass=box_mass[valid_mask],
            box_size=box_size[valid_mask],
            future_box_z=future_box_z_np[valid_mask],
        )
        return batch_valid, stats

    rng = jax.random.PRNGKey(seed)
    print(
        f"Collecting {num_states} valid states in batches of {batch_size} "
        f"(up to {max_batches} batches); rolling out {total_steps} steps each "
        f"(sample [{sample_start}, {sample_end}], lookahead {lookahead})."
    )

    collected = {k: [] for k in ("qpos", "qvel", "box_mass", "box_size", "future_box_z")}
    n_collected = 0
    n_run = 0
    for batch_idx in range(max_batches):
        rng, batch_rng = jax.random.split(rng)
        batch_valid, stats = run_one_batch(batch_rng)
        n_run += batch_size
        for k in collected:
            collected[k].append(batch_valid[k])
        n_collected += stats["n_valid"]

        msg = (
            f"[batch {batch_idx + 1}] valid {stats['n_valid']}/{batch_size} "
            f"(dropped {stats['n_dropped']}, off-center {stats['n_off_center']}, nan {stats['n_nan']}) "
            f"— total {n_collected}/{num_states}"
        )
        print(msg)
        if n_collected >= num_states:
            break

    if n_collected < num_states:
        raise RuntimeError(
            f"Only {n_collected} valid states after {batch_idx + 1} batches "
            f"({n_run} envs), but {num_states} requested. Increase --max_batches "
            f"(currently {max_batches}) or check policy quality."
        )

    # Concatenate across batches, then truncate to exactly num_states
    qpos_all     = np.concatenate(collected["qpos"], axis=0)[:num_states]
    qvel_all     = np.concatenate(collected["qvel"], axis=0)[:num_states]
    box_mass_all = np.concatenate(collected["box_mass"], axis=0)[:num_states]
    box_size_all = np.concatenate(collected["box_size"], axis=0)[:num_states]
    future_box_z_all = np.concatenate(collected["future_box_z"], axis=0)[:num_states]

    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        qpos=qpos_all,
        qvel=qvel_all,
        box_mass=box_mass_all,
        box_size=box_size_all,
        # metadata (stored as 0-dim arrays)
        pickup_checkpoint_path=np.array(pickup_checkpoint_path),
        seed=np.array(seed),
        sample_start=np.array(sample_start),
        sample_end=np.array(sample_end),
        lookahead=np.array(lookahead),
        num_states=np.array(num_states),
    )
    print(f"Saved {num_states} states ({n_collected} valid from {n_run} envs over {batch_idx + 1} batches) → {output_path}")
    print(f"  qpos shape: {qpos_all.shape}  qvel shape: {qvel_all.shape}")
    print(f"  box_mass range: [{box_mass_all.min():.3f}, {box_mass_all.max():.3f}]")
    print(f"  box_size x range: [{box_size_all[:,0].min():.3f}, {box_size_all[:,0].max():.3f}]")
    print(f"  future_box_z range (valid only): [{future_box_z_all.min():.3f}, {future_box_z_all.max():.3f}]")
