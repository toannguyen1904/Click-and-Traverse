"""Sanity check for a pre-generated warm-start states .npz file.

Loads the file, picks a few random state indices, prints diagnostics to verify
the robot is in a valid "holding box" configuration.

Usage:
    python check_warmstart_states.py --states data/warmstart/catra_pickup_states.npz
    python check_warmstart_states.py --states ... --view   # opens MuJoCo viewer; press N/P to switch states
    python check_warmstart_states.py --states ... --indices 10,25,42 --view
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, help="Path to the .npz warm-start states file.")
    parser.add_argument("--num_check", type=int, default=4, help="Number of states to inspect.")
    parser.add_argument("--indices", default="", help="Comma-separated state indices to inspect instead of random samples.")
    parser.add_argument("--view", action="store_true", help="Open MuJoCo viewer for inspected states; press N/P to switch.")
    args = parser.parse_args()

    npz = np.load(args.states, allow_pickle=True)
    qpos_all  = npz["qpos"]    # (N, nq)
    qvel_all  = npz["qvel"]    # (N, nv)
    box_mass  = npz["box_mass"]  # (N,)
    box_size  = npz["box_size"]  # (N, 3)
    N = qpos_all.shape[0]

    print(f"File: {args.states}")
    print(f"  num_states = {N}")
    print(f"  qpos shape = {qpos_all.shape}   (expect ({N}, 50))")
    print(f"  qvel shape = {qvel_all.shape}   (expect ({N}, 47))")
    print(f"  box_mass range: [{box_mass.min():.3f}, {box_mass.max():.3f}] kg  (expect [1.0, 2.0])")
    print(f"  box_size x range: [{box_size[:,0].min():.3f}, {box_size[:,0].max():.3f}]  (expect [0.10, 0.15])")
    print(f"  box_size y range: [{box_size[:,1].min():.3f}, {box_size[:,1].max():.3f}]  (expect [0.10, 0.20])")
    print(f"  box_size z range: [{box_size[:,2].min():.3f}, {box_size[:,2].max():.3f}]  (expect [0.10, 0.15])")
    has_nan_by_state = np.any(np.isnan(qpos_all), axis=1) | np.any(np.isnan(qvel_all), axis=1)
    n_nan = int(has_nan_by_state.sum())
    print(f"  states with NaNs: {n_nan}/{N}")
    print()

    # Load the scene to get body/site IDs for diagnostics
    import mujoco
    import jax.numpy as jp
    from mujoco.mjx._src.math import rotate
    import cat_ppo
    from cat_ppo.envs.g1 import constants as consts

    mj_model = mujoco.MjModel.from_xml_path(str(consts.CATRA_FLAT_TERRAIN_XML))
    box_body_id  = mj_model.body("carried_box").id
    box_geom_id  = mj_model.geom("box_geom").id
    support_geom_id = mj_model.geom("box_support_col").id
    lhand_id = mj_model.site("left_palm").id
    rhand_id = mj_model.site("right_palm").id

    surface_z = 0.3
    support_half_z = mj_model.geom_size[support_geom_id][2]
    pillar_top_z = surface_z + support_half_z

    if args.indices:
        try:
            indices = np.array([int(idx.strip()) for idx in args.indices.split(",") if idx.strip()], dtype=np.int64)
        except ValueError as exc:
            raise SystemExit("--indices must be a comma-separated list of integers.") from exc

        if indices.size == 0:
            raise SystemExit("--indices did not contain any valid state indices.")
        bad_indices = indices[(indices < 0) | (indices >= N)]
        if bad_indices.size:
            raise SystemExit(f"--indices out of range [0, {N - 1}]: {bad_indices.tolist()}")
    else:
        if args.num_check <= 0:
            raise SystemExit("--num_check must be positive.")
        rng = np.random.default_rng(42)
        indices = rng.integers(0, N, size=args.num_check)

    for i, idx in enumerate(indices):
        qpos = qpos_all[idx]    # (50,)
        qvel = qvel_all[idx]    # (47,)

        has_nan = bool(np.any(np.isnan(qpos)) or np.any(np.isnan(qvel)))

        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = qvel
        mj_model.geom_size[box_geom_id] = box_size[idx]  # patch box size for this state
        mujoco.mj_forward(mj_model, mj_data)

        pelvis_z = float(qpos[2])
        box_pos  = np.array(mj_data.xpos[box_body_id])
        box_z    = float(box_pos[2])

        bsz   = box_size[idx]
        box_quat = np.array(mj_data.xquat[box_body_id])  # (4,)
        # Left/right face positions
        box_left_axis  = np.array(rotate(jp.array([0., 1., 0.]), jp.array(box_quat)))
        left_face  = box_pos + box_left_axis * bsz[1]
        right_face = box_pos - box_left_axis * bsz[1]

        left_hand  = np.array(mj_data.site_xpos[lhand_id])
        right_hand = np.array(mj_data.site_xpos[rhand_id])
        dist_left  = float(np.linalg.norm(left_hand  - left_face))
        dist_right = float(np.linalg.norm(right_hand - right_face))

        lift_height = box_z - pillar_top_z - bsz[2]

        print(f"State {idx}:")
        print(f"  qpos.shape     = {qpos.shape[0]}  (expect 50)")
        print(f"  pelvis_z       = {pelvis_z:.3f} m  (expect ~0.6–0.75 m, crouched)")
        print(f"  box_z          = {box_z:.3f} m  (expect >{pillar_top_z + bsz[2]:.2f} m lifted)")
        print(f"  lift_height    = {lift_height:.3f} m  (expect > 0 m)")
        print(f"  dist_left_hand = {dist_left:.3f} m  (< 0.08 m = touching)")
        print(f"  dist_right_hand= {dist_right:.3f} m  (< 0.08 m = touching)")
        print(f"  box_mass       = {box_mass[idx]:.3f} kg")
        print(f"  box_size       = {bsz}")
        print(f"  has_nan        = {has_nan}")
        print()

    if args.view:
        import time
        import mujoco.viewer

        view_pos = 0
        pending_pos = 0
        mj_data = mujoco.MjData(mj_model)

        def set_view_state(idx):
            mj_data.qpos[:] = qpos_all[idx]
            mj_data.qvel[:] = qvel_all[idx]
            mj_model.geom_size[box_geom_id] = box_size[idx]
            mujoco.mj_forward(mj_model, mj_data)

        def key_callback(keycode):
            nonlocal pending_pos
            if keycode == ord("N"):
                pending_pos = (pending_pos + 1) % len(indices)
            elif keycode == ord("P"):
                pending_pos = (pending_pos - 1) % len(indices)

        set_view_state(int(indices[view_pos]))
        with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as v:
            print("Viewer open. Press N/P to switch states. Close the viewer to exit.")
            while v.is_running():
                if pending_pos != view_pos:
                    view_pos = pending_pos
                    idx = int(indices[view_pos])
                    with v.lock():
                        set_view_state(idx)
                    print(f"Viewer state {idx} ({view_pos + 1}/{len(indices)})")
                v.sync()
                time.sleep(0.05)


if __name__ == "__main__":
    main()
