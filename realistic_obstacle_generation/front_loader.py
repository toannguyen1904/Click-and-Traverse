"""
3D-FRONT scene + 3D-FUTURE mesh loading.

3D-FRONT uses a Y-up world frame in meters. The rest of this project uses Z-up,
so every world-space vertex is rotated through `YUP_TO_ZUP` before leaving this module.

Quaternion convention in 3D-FRONT JSONs is [x, y, z, w].
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import trimesh


# Y-up -> Z-up: (x, y, z)_zup = (x, -z, y).
# As a 4x4 transform matrix: applied to a column vector [x,y,z,1].
YUP_TO_ZUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


_OBJ_CACHE: dict[str, trimesh.Trimesh] = {}


class FurnitureInstance(NamedTuple):
    jid: str
    M_world_zup: np.ndarray  # 4x4 world transform after Y-up -> Z-up
    size: np.ndarray | None  # 3D-FRONT real-size (meters) in Y-up local mesh frame
    extents_norm: np.ndarray  # AABB extents of the normalized OBJ in Y-up frame


def list_scene_uids(data_root: os.PathLike | str) -> list[str]:
    folder = Path(data_root) / "3D-FRONT"
    return sorted(p.stem for p in folder.iterdir() if p.suffix == ".json")


def load_scene(uid: str, data_root: os.PathLike | str) -> dict:
    path = Path(data_root) / "3D-FRONT" / f"{uid}.json"
    with open(path) as f:
        return json.load(f)


def quat_to_matrix(q_xyzw) -> np.ndarray:
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q_xyzw
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
            [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compose_transform(pos, rot_xyzw, scale) -> np.ndarray:
    """Build a 4x4 TRS matrix in 3D-FRONT (Y-up) frame."""
    R = quat_to_matrix(rot_xyzw)
    S = np.diag(np.asarray(scale, dtype=np.float64))
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R @ S
    M[:3, 3] = np.asarray(pos, dtype=np.float64)
    return M


def load_furniture_mesh(jid: str, data_root: os.PathLike | str) -> trimesh.Trimesh | None:
    """Load `normalized_model.obj` for a 3D-FUTURE model id. Cached by jid.

    Returns None if the model file is absent (curtains/doors/windows often have
    placeholder furniture entries with no on-disk geometry).
    """
    if jid in _OBJ_CACHE:
        return _OBJ_CACHE[jid]
    path = Path(data_root) / "3D-FUTURE-model" / jid / "normalized_model.obj"
    if not path.exists():
        _OBJ_CACHE[jid] = None  # type: ignore[assignment]
        return None
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.shape[0] == 0:
        _OBJ_CACHE[jid] = None  # type: ignore[assignment]
        return None
    _OBJ_CACHE[jid] = mesh
    return mesh


def iter_furniture_instances(
    scene: dict, data_root: os.PathLike | str
) -> Iterator[FurnitureInstance]:
    """Walk every room.children whose ref resolves to a furniture[*] entry with an
    on-disk model. Yields the world-frame (Z-up) transform plus the metadata
    needed to rescale the normalized OBJ to its real size.
    """
    furniture_by_uid: dict[str, dict] = {f["uid"]: f for f in scene.get("furniture", [])}

    for room in scene.get("scene", {}).get("room", []):
        M_room = compose_transform(room.get("pos", [0, 0, 0]),
                                   room.get("rot", [0, 0, 0, 1]),
                                   room.get("scale", [1, 1, 1]))
        for child in room.get("children", []):
            ref = child.get("ref")
            if ref not in furniture_by_uid:
                continue  # mesh ref or orphan furniture ref
            fent = furniture_by_uid[ref]
            if fent.get("valid") is False:
                continue
            jid = fent.get("jid")
            if not jid:
                continue
            mesh = load_furniture_mesh(jid, data_root)
            if mesh is None:
                continue

            extents_norm = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
            # Avoid divide-by-zero on degenerate axes.
            extents_norm = np.where(extents_norm < 1e-9, 1.0, extents_norm)

            size = fent.get("size")
            size_arr = np.asarray(size, dtype=np.float64) if size is not None else None

            M_child = compose_transform(child.get("pos", [0, 0, 0]),
                                        child.get("rot", [0, 0, 0, 1]),
                                        child.get("scale", [1, 1, 1]))
            M_world = M_room @ M_child
            M_world_zup = YUP_TO_ZUP @ M_world

            yield FurnitureInstance(
                jid=jid,
                M_world_zup=M_world_zup,
                size=size_arr,
                extents_norm=extents_norm,
            )


def iter_floor_meshes(scene: dict) -> Iterator[trimesh.Trimesh]:
    """Yield each `mesh` of type 'Floor' as a trimesh in world Z-up frame.

    3D-FRONT floor meshes store vertices in world Y-up coords directly (room
    transforms are typically identity for arch meshes, but we still apply the
    room transform if present, matching the children-iteration semantics).
    """
    # Build a uid -> room transform map by walking room.children for floor refs.
    room_T_by_uid: dict[str, np.ndarray] = {}
    for room in scene.get("scene", {}).get("room", []):
        M_room = compose_transform(room.get("pos", [0, 0, 0]),
                                   room.get("rot", [0, 0, 0, 1]),
                                   room.get("scale", [1, 1, 1]))
        for child in room.get("children", []):
            ref = child.get("ref")
            if ref:
                # Only the most-recent assignment matters; mesh uids are unique per scene.
                M_child = compose_transform(child.get("pos", [0, 0, 0]),
                                            child.get("rot", [0, 0, 0, 1]),
                                            child.get("scale", [1, 1, 1]))
                room_T_by_uid[ref] = M_room @ M_child

    for m in scene.get("mesh", []):
        if m.get("type") != "Floor":
            continue
        xyz = np.asarray(m.get("xyz", []), dtype=np.float64).reshape(-1, 3)
        faces_flat = np.asarray(m.get("faces", []), dtype=np.int64)
        if xyz.size == 0 or faces_flat.size == 0:
            continue
        faces = faces_flat.reshape(-1, 3)

        M = room_T_by_uid.get(m["uid"], np.eye(4))
        # Apply room transform (Y-up), then Y-up -> Z-up.
        verts_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        verts_world = (YUP_TO_ZUP @ M @ verts_h.T).T[:, :3]
        yield trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)


def apply_furniture_transform(
    base_mesh: trimesh.Trimesh,
    M_world_zup: np.ndarray,
    size: np.ndarray | None,
    extents_norm: np.ndarray,
) -> trimesh.Trimesh:
    """Transform a normalized OBJ to its world-space Z-up pose.

    Steps: per-axis rescale (size / extents_norm) -> apply M_world_zup.
    Falls back to no-rescale when `size` is missing.
    """
    verts = np.asarray(base_mesh.vertices, dtype=np.float64)
    if size is not None:
        scale_per_axis = size / extents_norm
        verts = verts * scale_per_axis  # in Y-up local frame
    verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
    verts_world = (M_world_zup @ verts_h.T).T[:, :3]
    return trimesh.Trimesh(
        vertices=verts_world.astype(np.float32),
        faces=np.asarray(base_mesh.faces, dtype=np.int64),
        process=False,
    )
