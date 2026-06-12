#!/usr/bin/env python3
"""
Interactive 3D visualization of the HumanoidPF guidance field (gf) for any saved
scene under data/assets/{TypiObs,RandObs,RealObs}/<scene>/.

The gf array has shape (Nx, Ny, Nz, 3): a 3D vector at every voxel pointing along
the guidance direction. This tool renders, in a single 3D matplotlib axes:
    - gf as colored quiver arrows (color/length = vector magnitude), subsampled
    - obstacle voxels as a translucent grey volume
    - start (green) and goal (red) markers

Run from the project root:
    python tools/visualize_gf_3d.py --scene-type TypiObs --scene bar2
    python tools/visualize_gf_3d.py --path data/assets/TypiObs/bar2
    python tools/visualize_gf_3d.py --path data/assets/TypiObs/bar2 --stride 3 --save fig/bar2_gf3d.png

By default the figure opens in an interactive window (rotate/zoom). Use --save to
also write a PNG. Use --stride to control arrow density (higher = sparser).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "procedural_obstacle_generation"))

from pf_modular import PFConfig  # noqa: E402

SCENE_TYPES = ("TypiObs", "RandObs", "RealObs")


def resolve_scene_dir(scene_type, scene, path):
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


def load_fields(scene_dir):
    gf_path = scene_dir / "gf.npy"
    if not gf_path.exists():
        raise SystemExit(f"missing gf.npy in {scene_dir}")
    gf = np.load(gf_path)
    obs_path = scene_dir / "obs.npy"
    obs = np.load(obs_path).astype(bool) if obs_path.exists() else None
    return gf, obs


def make_axes_from_shape(shape, voxel, origin):
    Nx, Ny, Nz = shape[:3]
    ox, oy, oz = origin
    xv = ox + (np.arange(Nx) + 0.5) * voxel
    yv = oy + (np.arange(Ny) + 0.5) * voxel
    zv = oz + (np.arange(Nz) + 0.5) * voxel
    return xv, yv, zv


def set_equal_aspect(ax, xv, yv, zv):
    ranges = np.array([np.ptp(xv), np.ptp(yv), np.ptp(zv)])
    centers = np.array([xv.mean(), yv.mean(), zv.mean()])
    r = ranges.max() / 2.0
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene-type", choices=SCENE_TYPES, help="Scene category under data/assets/.")
    parser.add_argument("--scene", help="Scene folder name (e.g. 'bar2').")
    parser.add_argument("--path", help="Direct path to a scene folder (overrides --scene-type/--scene).")
    parser.add_argument("--stride", type=int, default=4, help="Subsample step for quiver arrows (default: 4).")
    parser.add_argument("--length", type=float, default=0.6,
                        help="Arrow length as a fraction of voxel*stride (default: 0.6).")
    parser.add_argument("--min-mag", type=float, default=1e-6,
                        help="Hide arrows with magnitude below this (default: 1e-6).")
    parser.add_argument("--no-obstacles", action="store_true", help="Do not draw obstacle voxels.")
    parser.add_argument("--obs-stride", type=int, default=1,
                        help="Subsample step for obstacle voxels (default: 1).")
    parser.add_argument("--cmap", default="viridis", help="Colormap for arrow magnitude (default: viridis).")
    parser.add_argument("--save", default=None, help="Optional PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    args = parser.parse_args()

    scene_dir = resolve_scene_dir(args.scene_type, args.scene, args.path)
    if not scene_dir.is_dir():
        raise SystemExit(f"scene directory does not exist: {scene_dir}")

    gf, obs = load_fields(scene_dir)
    if gf.ndim != 4 or gf.shape[-1] != 3:
        raise SystemExit(f"expected gf of shape (Nx,Ny,Nz,3), got {gf.shape}")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    cfg = PFConfig()
    voxel = cfg.voxel
    xv, yv, zv = make_axes_from_shape(gf.shape, voxel, cfg.origin_w)

    s = max(1, args.stride)
    Xc, Yc, Zc = np.meshgrid(xv[::s], yv[::s], zv[::s], indexing="ij")
    U = gf[::s, ::s, ::s, 0]
    V = gf[::s, ::s, ::s, 1]
    W = gf[::s, ::s, ::s, 2]
    mag = np.sqrt(U**2 + V**2 + W**2)

    keep = mag > args.min_mag
    Xc, Yc, Zc = Xc[keep], Yc[keep], Zc[keep]
    U, V, W, mag = U[keep], V[keep], W[keep], mag[keep]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    arrow_len = args.length * voxel * s
    cmap = plt.get_cmap(args.cmap)
    mmax = mag.max() if mag.size else 1.0
    norm = plt.Normalize(vmin=0.0, vmax=mmax if mmax > 0 else 1.0)
    colors = cmap(norm(mag))

    if mag.size:
        # normalize directions so color encodes magnitude and length is uniform
        eps = 1e-9
        Un, Vn, Wn = U / (mag + eps), V / (mag + eps), W / (mag + eps)
        ax.quiver(
            Xc, Yc, Zc, Un, Vn, Wn,
            length=arrow_len, normalize=False,
            colors=colors, linewidth=0.8, arrow_length_ratio=0.4,
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label("|gf|")

    if obs is not None and not args.no_obstacles:
        os_ = max(1, args.obs_stride)
        oi, oj, ok = np.where(obs[::os_, ::os_, ::os_])
        if oi.size:
            ax.scatter(
                xv[::os_][oi], yv[::os_][oj], zv[::os_][ok],
                c="grey", marker="s", s=6, alpha=0.12, depthshade=False,
            )

    sx, sy, sz = cfg.start_w
    gx, gy, gz = cfg.goal_w
    ax.scatter([sx], [sy], [sz], c="lime", marker="o", s=120, edgecolors="k", label="start")
    ax.scatter([gx], [gy], [gz], c="red", marker="*", s=240, edgecolors="k", label="goal")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Guidance field (gf) — {scene_dir.name}  |  stride={s}, arrows={mag.size}")
    ax.legend(loc="upper left")
    set_equal_aspect(ax, xv, yv, zv)

    if args.save:
        out = Path(args.save)
        if not out.is_absolute():
            out = _REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[OK] wrote {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
