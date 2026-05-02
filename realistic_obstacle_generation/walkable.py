"""
2D walkable-region construction and start/goal sampling.

The walkable mask is computed in world Z-up frame at the same voxel pitch as the
3D obstacle grid (`cfg.voxel`). It is the union of floor polygons minus the
ground projection of furniture whose AABB.zmin <= cfg.footprint_z_max, eroded
by `cfg.walk_erode_r` to reserve clearance for the robot.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.draw import polygon as sk_polygon

from .config import RealCfg
from .front_loader import (
    apply_furniture_transform,
    iter_floor_meshes,
    iter_furniture_instances,
    load_furniture_mesh,
)


def _disk(radius_vox: int) -> np.ndarray:
    r = max(1, int(radius_vox))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= r * r


def build_walkable_mask(scene: dict, data_root, cfg: RealCfg):
    """Returns (mask_2d, x0, y0, voxel) where mask_2d is in (i, j) = (x, y) order."""
    voxel = cfg.voxel

    floors = list(iter_floor_meshes(scene))
    if not floors:
        # Empty mask with arbitrary anchor; caller will treat as unusable.
        return np.zeros((1, 1), dtype=bool), 0.0, 0.0, voxel

    all_floor_xy = np.concatenate([f.vertices[:, :2] for f in floors], axis=0)
    pad = 1.0
    xmin = float(all_floor_xy[:, 0].min()) - pad
    ymin = float(all_floor_xy[:, 1].min()) - pad
    xmax = float(all_floor_xy[:, 0].max()) + pad
    ymax = float(all_floor_xy[:, 1].max()) + pad

    Nx = int(np.ceil((xmax - xmin) / voxel))
    Ny = int(np.ceil((ymax - ymin) / voxel))
    if Nx <= 0 or Ny <= 0:
        return np.zeros((1, 1), dtype=bool), 0.0, 0.0, voxel

    mask = np.zeros((Nx, Ny), dtype=bool)

    # Rasterize each floor triangle.
    for floor in floors:
        xy = floor.vertices[:, :2]
        for tri in floor.faces:
            p = xy[tri]
            i = (p[:, 0] - xmin) / voxel
            j = (p[:, 1] - ymin) / voxel
            rr, cc = sk_polygon(i, j, shape=mask.shape)
            mask[rr, cc] = True

    # Subtract ground projection of low furniture.
    for inst in iter_furniture_instances(scene, data_root):
        base = load_furniture_mesh(inst.jid, data_root)
        if base is None:
            continue
        world_mesh = apply_furniture_transform(
            base, inst.M_world_zup, inst.size, inst.extents_norm
        )
        bb_min = world_mesh.vertices.min(axis=0)
        bb_max = world_mesh.vertices.max(axis=0)
        if bb_min[2] > cfg.footprint_z_max:
            continue  # ceiling-mounted; preserve walkable area underneath
        i0 = int(np.floor((bb_min[0] - xmin) / voxel))
        i1 = int(np.ceil((bb_max[0] - xmin) / voxel))
        j0 = int(np.floor((bb_min[1] - ymin) / voxel))
        j1 = int(np.ceil((bb_max[1] - ymin) / voxel))
        i0 = max(0, i0); i1 = min(Nx, i1)
        j0 = max(0, j0); j1 = min(Ny, j1)
        if i0 < i1 and j0 < j1:
            mask[i0:i1, j0:j1] = False

    radius_vox = max(1, int(round(cfg.walk_erode_r / voxel)))
    if radius_vox > 0 and mask.any():
        mask = binary_erosion(mask, structure=_disk(radius_vox))

    return mask, xmin, ymin, voxel


def sample_start_goal(mask, x0, y0, voxel, cfg: RealCfg, rng):
    """Sample (start_xy, goal_xy) inside the walkable mask, with goal on a 2 m circle.

    Returns None if no valid pair is found within the retry budget.
    """
    walk_idx = np.argwhere(mask)
    if walk_idx.size == 0:
        return None
    Nx, Ny = mask.shape

    for _ in range(cfg.max_start_retries):
        pick = rng.integers(0, walk_idx.shape[0])
        i, j = walk_idx[pick]
        sx = x0 + (int(i) + 0.5) * voxel
        sy = y0 + (int(j) + 0.5) * voxel

        for _ in range(cfg.max_goal_retries):
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            gx = sx + cfg.goal_radius * np.cos(theta)
            gy = sy + cfg.goal_radius * np.sin(theta)
            gi = int((gx - x0) / voxel)
            gj = int((gy - y0) / voxel)
            if 0 <= gi < Nx and 0 <= gj < Ny and mask[gi, gj]:
                return (float(sx), float(sy)), (float(gx), float(gy))

    return None
