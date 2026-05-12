#!/usr/bin/env python3
"""
Visualize the HumanoidPF guidance field (gf) and SDF for any saved scene under
data/assets/{TypiObs,RandObs,RealObs}/<scene>/.

For each scene, three slices are rendered (top / side / front) showing:
    - SDF as a coloured background (coolwarm)
    - obstacle contour in black
    - gf as white quiver arrows
    - start (white) and goal (red star) markers

Run from the project root:
    python tools/visualize_guidance_field.py --scene-type TypiObs --scene bend
    python tools/visualize_guidance_field.py --scene-type RandObs --scene D2G3L9O3S42
    python tools/visualize_guidance_field.py --scene-type RealObs \\
        --scene 01951c96-631e-4880-83ab-1d69835e635c_S0
    python tools/visualize_guidance_field.py --path data/assets/TypiObs/bend

Output files: <out-dir>/<scene>_{top,side,front}.png (default out-dir: fig/)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "procedural_obstacle_generation"))

from pf_modular import PFConfig, visualize_all  # noqa: E402


SCENE_TYPES = ("TypiObs", "RandObs", "RealObs")


def resolve_scene_dir(scene_type: str | None, scene: str | None, path: str | None) -> Path:
    if path is not None:
        scene_dir = Path(path)
        if not scene_dir.is_absolute():
            scene_dir = _REPO_ROOT / scene_dir
        return scene_dir
    if scene_type is None or scene is None:
        raise SystemExit("must provide either --path, or both --scene-type and --scene")
    if scene_type not in SCENE_TYPES:
        raise SystemExit(f"--scene-type must be one of {SCENE_TYPES}, got {scene_type!r}")
    return _REPO_ROOT / "data" / "assets" / scene_type / scene


def load_fields(scene_dir: Path):
    required = ["obs.npy", "sdf.npy", "gf.npy"]
    missing = [f for f in required if not (scene_dir / f).exists()]
    if missing:
        raise SystemExit(f"missing files in {scene_dir}: {missing}")
    obs = np.load(scene_dir / "obs.npy").astype(bool)
    sdf = np.load(scene_dir / "sdf.npy")
    gf = np.load(scene_dir / "gf.npy")
    return obs, sdf, gf


def make_axes_from_shape(shape, voxel: float, origin: np.ndarray):
    Nx, Ny, Nz = shape
    ox, oy, oz = origin
    xv = ox + (np.arange(Nx) + 0.5) * voxel
    yv = oy + (np.arange(Ny) + 0.5) * voxel
    zv = oz + (np.arange(Nz) + 0.5) * voxel
    return xv, yv, zv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-type", choices=SCENE_TYPES, help="Scene category under data/assets/.")
    parser.add_argument("--scene", help="Scene folder name (e.g. 'bend', 'D2G3L9O3S42').")
    parser.add_argument("--path", help="Direct path to a scene folder (overrides --scene-type / --scene).")
    parser.add_argument("--out-dir", default="fig", help="Output directory for PNGs (default: fig/).")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Override the output filename prefix (default: <scene-type>_<scene> or folder name).",
    )
    args = parser.parse_args()

    scene_dir = resolve_scene_dir(args.scene_type, args.scene, args.path)
    if not scene_dir.is_dir():
        raise SystemExit(f"scene directory does not exist: {scene_dir}")

    obs, sdf, gf = load_fields(scene_dir)

    cfg = PFConfig()
    xv, yv, zv = make_axes_from_shape(obs.shape, cfg.voxel, cfg.origin_w)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prefix is not None:
        prefix = args.prefix
    elif args.scene_type and args.scene:
        prefix = f"{args.scene_type}_{args.scene}"
    else:
        prefix = scene_dir.name
    title_prefix = str(out_dir / prefix)

    visualize_all(
        xv, yv, zv,
        sdf, None, gf, obs,
        cfg.start_w, cfg.goal_w,
        bf=None,
        title_prefix=title_prefix,
    )
    print(f"[OK] wrote {title_prefix}_{{top,side,front}}.png")


if __name__ == "__main__":
    main()
