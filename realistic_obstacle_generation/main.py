"""
Generate one usable 5x5 m capture from a 3D-FRONT scene.

Output mirrors `procedural_obstacle_generation/main.py`: writes obs.npy, sdf.npy,
bf.npy, gf.npy, obs.obj into `data/assets/RealObs/<scene_uid>_S<seed>/`. The
returned metadata dict carries the per-capture origin_w / start_w / goal_w that
downstream training code needs to interpret the precomputed fields.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from procedural_obstacle_generation.pf_modular import (
    PFConfig,
    grad3,
    make_guidance_field_progressive,
    make_sdf,
    visualize_all,
)
from procedural_obstacle_generation.utills import marching_cubes_mesh

from .config import RealCfg
from .front_loader import list_scene_uids, load_scene
from .voxelize import voxelize_scene
from .walkable import build_walkable_mask, sample_start_goal


_log = logging.getLogger(__name__)
_FIG = Path(__file__).resolve().parents[1] / "fig"


def _better_mesh(spacing, obs_mask):
    """Marching-cubes obstacle skin (mirrors procedural's `better_mesh`).

    Clears boundary voxels to keep the surface closed, inverts the mask so
    `marching_cubes` traces the obstacle shell, and returns a trimesh.
    """
    obs_mask = obs_mask.copy()
    obs_mask[:, 0, :] = 0
    obs_mask[:, -1, :] = 0
    obs_mask[:, :, -1] = 0
    inverted = 1 - obs_mask.astype(np.uint8)
    return marching_cubes_mesh(inverted, spacing=spacing)


def generate_realistic_scene(
    scene_index: int,
    seed: int | None = None,
    cfg: RealCfg | None = None,
) -> dict | None:
    """Process one 3D-FRONT scene into a 5x5 m capture.

    Returns the per-capture metadata dict on success, or None when the scene
    yields no usable (start, 2 m-radius goal) pair within the retry budget.
    `IndexError` is raised for an out-of-range `scene_index`.
    """
    cfg = cfg or RealCfg()
    rng = np.random.default_rng(seed)

    uids = list_scene_uids(cfg.data_root)
    uid = uids[scene_index]
    scene = load_scene(uid, cfg.data_root)

    mask2d, x0, y0, vx = build_walkable_mask(scene, cfg.data_root, cfg)
    if not mask2d.any():
        _log.info("scene %s has no walkable area after erosion", uid)
        return None

    sampled = sample_start_goal(mask2d, x0, y0, vx, cfg, rng)
    if sampled is None:
        _log.info("scene %s: failed to sample (start, goal) within retry budget", uid)
        return None
    (sx, sy), (gx, gy) = sampled

    origin_w = np.array([sx - cfg.Lx / 2.0, sy - cfg.Ly / 2.0, 0.0], dtype=np.float32)
    start_w = np.array([sx, sy, cfg.robot_z], dtype=np.float32)
    goal_w = np.array([gx, gy, cfg.robot_z], dtype=np.float32)

    obs = voxelize_scene(scene, cfg.data_root, origin_w, cfg.Lx, cfg.Ly, cfg.Lz, cfg.voxel)

    Nx = int(round(cfg.Lx / cfg.voxel))
    Ny = int(round(cfg.Ly / cfg.voxel))
    Nz = int(round(cfg.Lz / cfg.voxel))
    xv = origin_w[0] + (np.arange(Nx) + 0.5) * cfg.voxel
    yv = origin_w[1] + (np.arange(Ny) + 0.5) * cfg.voxel
    zv = origin_w[2] + (np.arange(Nz) + 0.5) * cfg.voxel
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing="ij")

    pfc = PFConfig()
    pfc.voxel = cfg.voxel
    pfc.Lx, pfc.Ly, pfc.Lz = cfg.Lx, cfg.Ly, cfg.Lz
    pfc.origin_w = origin_w
    pfc.start_w = start_w
    pfc.goal_w = goal_w

    sdf = make_sdf(obs, cfg.voxel)
    bf = grad3(sdf, cfg.voxel)
    _T, gf = make_guidance_field_progressive(pfc, (X, Y, Z), obs, goal_w, bf, sdf)

    seed_tag = "R" if seed is None else str(seed)
    _FIG.mkdir(parents=True, exist_ok=True)
    visualize_all(xv, yv, zv, sdf, _T, gf, obs, start_w, goal_w, bf=bf,
                  title_prefix=str(_FIG / f"{uid}_S{seed_tag}"))
    out_dir = Path(cfg.out_root) / f"{uid}_S{seed_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if obs.any():
        mesh = _better_mesh((cfg.voxel,) * 3, obs)
        mesh.export(str(out_dir / "obs.obj"))
    np.save(out_dir / "obs.npy", obs.astype(np.uint8))
    np.save(out_dir / "sdf.npy", sdf)
    np.save(out_dir / "bf.npy", bf)
    np.save(out_dir / "gf.npy", gf)

    return {
        "scene_uid": uid,
        "scene_index": scene_index,
        "sample_seed": seed,
        "origin_w": origin_w,
        "voxel": cfg.voxel,
        "Lx": cfg.Lx,
        "Ly": cfg.Ly,
        "Lz": cfg.Lz,
        "start_w": start_w,
        "goal_w": goal_w,
        "out_dir": str(out_dir),
    }


def _main():
    parser = argparse.ArgumentParser(description="Generate one realistic CaTra scene from 3D-FRONT.")
    parser.add_argument("scene_index", type=int, help="Index into the sorted list of 3D-FRONT scene UIDs.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for start/goal sampling.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    meta = generate_realistic_scene(args.scene_index, seed=args.seed)
    if meta is None:
        print(f"scene_index={args.scene_index}: unusable (no walkable area or sampling failed)")
        raise SystemExit(1)
    print(meta)


if __name__ == "__main__":
    _main()
