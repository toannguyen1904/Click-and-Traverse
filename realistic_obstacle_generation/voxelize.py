"""
3D voxelization of furniture meshes into a local crop grid.

`voxelize_scene` walks every furniture instance in the scene, transforms it to
world Z-up coordinates, voxelizes via trimesh, and ORs the result into a fixed
(Nx, Ny, Nz) bool grid anchored at `origin_w` with pitch `voxel`. Architectural
meshes (walls, ceilings, doors) are intentionally excluded.
"""

from __future__ import annotations

import logging

import numpy as np

from .front_loader import (
    apply_furniture_transform,
    iter_furniture_instances,
    load_furniture_mesh,
)


_log = logging.getLogger(__name__)


def voxelize_scene(scene, data_root, origin_w, Lx, Ly, Lz, voxel) -> np.ndarray:
    Nx = int(round(Lx / voxel))
    Ny = int(round(Ly / voxel))
    Nz = int(round(Lz / voxel))
    obs = np.zeros((Nx, Ny, Nz), dtype=bool)

    origin_w = np.asarray(origin_w, dtype=np.float64)
    crop_min = origin_w
    crop_max = origin_w + np.array([Lx, Ly, Lz], dtype=np.float64)

    for inst in iter_furniture_instances(scene, data_root):
        base = load_furniture_mesh(inst.jid, data_root)
        if base is None:
            continue
        try:
            world_mesh = apply_furniture_transform(
                base, inst.M_world_zup, inst.size, inst.extents_norm
            )
            bb_min = world_mesh.vertices.min(axis=0)
            bb_max = world_mesh.vertices.max(axis=0)
            if (bb_max < crop_min).any() or (bb_min > crop_max).any():
                continue  # AABB-disjoint with crop

            vox = world_mesh.voxelized(pitch=voxel).fill()
            points = np.asarray(vox.points, dtype=np.float64)
            if points.size == 0:
                continue

            idx = np.floor((points - origin_w) / voxel).astype(np.int64)
            keep = (
                (idx[:, 0] >= 0) & (idx[:, 0] < Nx)
                & (idx[:, 1] >= 0) & (idx[:, 1] < Ny)
                & (idx[:, 2] >= 0) & (idx[:, 2] < Nz)
            )
            idx = idx[keep]
            obs[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        except Exception as e:
            _log.warning("voxelize failed for jid=%s: %s", inst.jid, e)
            continue

    return obs
