#!/usr/bin/env python3
"""
Interactive 3D visualization of the HumanoidPF guidance field (gf) for any saved
scene under data/assets/{TypiObs,RandObs,RealObs}/<scene>/.

The gf array has shape (Nx, Ny, Nz, 3): a 3D vector at every voxel pointing along
the guidance direction. This tool renders an interactive Plotly scene (smooth
zoom / pan / rotate in the browser):
    - gf as cone glyphs, colored by vector magnitude |gf|
    - obstacle voxels as a translucent grey point cloud
    - start (green) and goal (red) markers

Run from the project root:
    python tools/visualize_gf_3d.py --scene-type TypiObs --scene bar2
    python tools/visualize_gf_3d.py --path data/assets/TypiObs/bar2
    python tools/visualize_gf_3d.py --path data/assets/TypiObs/bar2 --stride 3

The figure opens in your browser by default and is also written to an HTML file
(reusable / shareable). Use --out to set the path, --no-show to skip opening.
Use --stride to control cone density (higher = sparser, faster).
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


def load_fields(scene_dir, inflation=False):
    fname = "gf_inflation.npy" if inflation else "gf.npy"
    gf_path = scene_dir / fname
    if not gf_path.exists():
        raise SystemExit(f"missing {fname} in {scene_dir}")
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene-type", choices=SCENE_TYPES, help="Scene category under data/assets/.")
    parser.add_argument("--scene", help="Scene folder name (e.g. 'bar2').")
    parser.add_argument("--path", help="Direct path to a scene folder (overrides --scene-type/--scene).")
    parser.add_argument("--stride", type=int, default=3, help="Subsample step for cone glyphs (default: 3).")
    parser.add_argument("--min-mag", type=float, default=1e-6,
                        help="Hide arrows with magnitude below this (default: 1e-6).")
    parser.add_argument("--sizeref", type=float, default=0.5,
                        help="Cone size scale (larger = bigger cones, default: 0.5).")
    parser.add_argument("--inflation", action="store_true",
                        help="Load gf_inflation.npy (anticipatory box field) instead of gf.npy.")
    parser.add_argument("--no-obstacles", action="store_true", help="Do not draw obstacle voxels.")
    parser.add_argument("--obs-stride", type=int, default=1,
                        help="Subsample step for obstacle voxels (default: 1).")
    parser.add_argument("--cmap", default="Viridis", help="Plotly colorscale for |gf| (default: Viridis).")
    parser.add_argument("--out", default=None, help="HTML output path (default: fig/<scene>_gf3d.html).")
    parser.add_argument("--no-show", action="store_true", help="Write HTML without opening a browser.")
    args = parser.parse_args()

    scene_dir = resolve_scene_dir(args.scene_type, args.scene, args.path)
    if not scene_dir.is_dir():
        raise SystemExit(f"scene directory does not exist: {scene_dir}")

    gf, obs = load_fields(scene_dir, inflation=args.inflation)
    if gf.ndim != 4 or gf.shape[-1] != 3:
        raise SystemExit(f"expected gf of shape (Nx,Ny,Nz,3), got {gf.shape}")

    import plotly.graph_objects as go  # noqa: E402

    cfg = PFConfig()
    voxel = cfg.voxel
    xv, yv, zv = make_axes_from_shape(gf.shape, voxel, cfg.origin_w)

    s = max(1, args.stride)
    Xc, Yc, Zc = np.meshgrid(xv[::s], yv[::s], zv[::s], indexing="ij")
    U = gf[::s, ::s, ::s, 0]
    V = gf[::s, ::s, ::s, 1]
    W = gf[::s, ::s, ::s, 2]
    mag = np.sqrt(U**2 + V**2 + W**2)

    keep = (mag > args.min_mag).ravel()
    x, y, z = Xc.ravel()[keep], Yc.ravel()[keep], Zc.ravel()[keep]
    u, v, w = U.ravel()[keep], V.ravel()[keep], W.ravel()[keep]

    traces = []
    traces.append(
        go.Cone(
            x=x, y=y, z=z, u=u, v=v, w=w,
            colorscale=args.cmap,
            sizemode="scaled",
            sizeref=args.sizeref,
            anchor="tail",
            colorbar=dict(title="|gf|", len=0.6),
            name="gf",
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        )
    )

    if obs is not None and not args.no_obstacles:
        os_ = max(1, args.obs_stride)
        oi, oj, ok = np.where(obs[::os_, ::os_, ::os_])
        if oi.size:
            traces.append(
                go.Scatter3d(
                    x=xv[::os_][oi], y=yv[::os_][oj], z=zv[::os_][ok],
                    mode="markers",
                    marker=dict(size=3, color="red", opacity=0.15, symbol="square"),
                    name="obstacle",
                )
            )

    sx, sy, sz = cfg.start_w
    gx, gy, gz = cfg.goal_w
    traces.append(go.Scatter3d(
        x=[sx], y=[sy], z=[sz], mode="markers",
        marker=dict(size=8, color="lime", line=dict(color="black", width=1)), name="start"))
    traces.append(go.Scatter3d(
        x=[gx], y=[gy], z=[gz], mode="markers",
        marker=dict(size=10, color="red", symbol="diamond", line=dict(color="black", width=1)), name="goal"))

    fig = go.Figure(data=traces)
    n_arrows = int(keep.sum())
    field_name = "gf_inflation (box)" if args.inflation else "gf"
    fig.update_layout(
        title=f"Guidance field ({field_name}) — {scene_dir.name}  |  stride={s}, arrows={n_arrows}",
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98),
    )

    _suffix = "_inflation" if args.inflation else ""
    out = Path(args.out) if args.out else _REPO_ROOT / "fig" / f"{scene_dir.name}_gf3d{_suffix}.html"
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"[OK] wrote {out}  ({n_arrows} arrows)")

    if not args.no_show:
        try:
            fig.show()
        except Exception as e:
            print(f"[warn] could not open browser ({e}); open the HTML file manually.")


if __name__ == "__main__":
    main()
