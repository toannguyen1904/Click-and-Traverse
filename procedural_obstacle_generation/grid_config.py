from pathlib import Path

import numpy as np
import yaml


DEFAULT_GRID_CONFIG_PATH = Path(__file__).with_name("pf_grid_config.yaml")


def _as_array3(value, field_name: str, dtype=np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (3,):
        raise ValueError(f"{field_name} must be a 3-element list")
    return array


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Grid config must be a YAML mapping: {path}")
    return data


def normalize_grid_config(data: dict) -> dict:
    """Normalize flat export meta or nested grid YAML into one grid schema."""
    grid = data.get("grid", data)
    if not isinstance(grid, dict):
        raise ValueError("grid must be a YAML mapping")

    voxel = float(grid.get("voxel", grid.get("voxel_size", data.get("voxel_size", 0.04))))
    size_source = grid.get("size", grid.get("effective_size", data.get("size", data.get("effective_size"))))
    if size_source is None:
        raise ValueError("grid config must contain size or effective_size")

    size = _as_array3(size_source, "size", dtype=np.float64)
    shape = np.round(size / voxel).astype(np.int32)
    if "shape" in grid:
        declared_shape = _as_array3(grid["shape"], "shape", dtype=np.int32)
        if not np.array_equal(declared_shape, shape):
            raise ValueError(f"shape {declared_shape.tolist()} does not match computed shape {shape.tolist()}")

    axis_order = grid.get("axis_order", data.get("axis_order", "xyz"))
    if axis_order != "xyz":
        raise ValueError(f"Expected axis_order='xyz', got {axis_order!r}")

    return {
        "voxel": voxel,
        "size": size.astype(np.float32),
        "shape": shape,
        "origin_w": _as_array3(grid.get("origin_w", data.get("origin_w")), "origin_w"),
        "start_w": _as_array3(grid.get("start_w", data.get("start_w")), "start_w"),
        "goal_w": _as_array3(grid.get("goal_w", data.get("goal_w")), "goal_w"),
        "axis_order": axis_order,
    }


def load_grid_config(config_path: str | Path | None = None) -> dict:
    path = DEFAULT_GRID_CONFIG_PATH if config_path is None else Path(config_path)
    return normalize_grid_config(_load_yaml(path))


def apply_grid_config(cfg, config_path: str | Path | None = None) -> dict:
    grid = load_grid_config(config_path)
    cfg.voxel = grid["voxel"]
    cfg.Lx, cfg.Ly, cfg.Lz = [float(v) for v in grid["size"]]
    cfg.origin_w = grid["origin_w"].copy()
    cfg.start_w = grid["start_w"].copy()
    cfg.goal_w = grid["goal_w"].copy()
    cfg.axis_order = grid["axis_order"]
    return grid


def expected_shape_from_config(config_path: str | Path | None = None) -> tuple[int, int, int]:
    grid = load_grid_config(config_path)
    return tuple(int(v) for v in grid["shape"])
