# pf_modular.py
"""
This is the core HumanoidPF construction code, which takes an obstacle occupancy grid
and computes all the spatial fields the policy needs.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import skfmm
# @dataclass
class PFConfig:
    voxel: float = 0.04           # resolution (m)
    Lx: float = 3.0               # x-axis (m) forward
    Ly: float = 2.0               # y-axis (m) left
    Lz: float = 1.5               # z-axis (m) up
    origin_w: np.ndarray = np.array([-0.5, -1.0, 0.0], dtype=np.float32) # pre-defined origin in canonical frame (m)
    start_w:  np.ndarray = np.array([0.0,  0.0, 0.75], dtype=np.float32)    # start position in canonical frame (m)
    goal_w:   np.ndarray = np.array([2.0,  0.0, 0.75], dtype=np.float32)    # goal position in canonical frame (m)

    v_max: float = 0.6          # max velocity far from goal (m/s)
    k_decay: float = 0.6        # decay radius near goal (m), defined but not used
    goal_seed_r: float = 0.12   # goal negative region radius (m), for the fmm

    # Option-B (SDF-offset) obstacle inflation radius (m) for the *anticipatory*
    # box guidance field. Sized as: box circumscribed radius + braking/turning margin.
    # Larger -> earlier/farther standoff, but too large seals narrow corridors.
    box_inflation_margin: float = 0.06

def world_to_local(p_w, cfg: PFConfig):
    return p_w - cfg.origin_w

def make_sdf(obs_mask: np.ndarray, voxel: float) -> np.ndarray:
    # Returns (Nx, Ny, Nz): distance to nearest obstacle surface in meters,
    # positive outside obstacles, negative inside.
    phi_obs = np.ones(obs_mask.shape, dtype=float)
    phi_obs[obs_mask] = -1.0
    sdf = skfmm.distance(phi_obs, dx=voxel).astype(np.float32)  # signed distance (m)
    return sdf

def grad3(scalar_field: np.ndarray, voxel: float):
    # Returns (Nx, Ny, Nz, 3) boundary field: 3D gradient vector at each voxel [df/dx, df/dy, df/dz].
    # When applied to the SDF, gives the boundary field bf — outward normals at obstacle surfaces.
    dfx, dfy, dfz = np.gradient(scalar_field, voxel, voxel, voxel, edge_order=2)
    return np.stack([dfx, dfy, dfz], axis=-1).astype(np.float32)


def make_raw_guidance_field(cfg, grids, obs_mask, goal_local, r_proj=None): # not being used currently
    voxel = cfg.voxel
    eps = 1e-9
    if r_proj is None:
        r_proj = 5.0 * voxel  # can be adjusted according to robot size

    X, Y, Z = grids
    # goal negative region (small sphere)
    phi = np.ones(obs_mask.shape, dtype=float)
    goal_seed = ((X - goal_local[0])**2 + (Y - goal_local[1])**2 + (Z - goal_local[2])**2) <= cfg.goal_seed_r**2
    phi[goal_seed] = -1.0
    phi = np.ma.MaskedArray(phi, mask=obs_mask)

    # Fast Marching: T (geodesic distance)
    T_ma = skfmm.distance(phi, dx=cfg.voxel)
    T_free_max = np.max(T_ma[~T_ma.mask]) if np.any(~T_ma.mask) else 0.0
    T = T_ma.filled(T_free_max).astype(np.float32)
    return T

def make_guidance_field_progressive(cfg, grids, obs_mask, goal_local, bf, sdf, r_proj=None):
    """
    cfg       : PFConfig — uses voxel, v_max, goal_seed_r
    grids     : tuple (X, Y, Z), each (Nx,Ny,Nz), world-space coords of every voxel
                (from np.meshgrid(xv, yv, zv, indexing='ij'))
    obs_mask  : bool ndarray (Nx,Ny,Nz), True=inside obstacle
    goal_local: (3,) goal position in world/local coords; seeds the geodesic solve
    bf        : (Nx,Ny,Nz,3) normal field (recommended to use SDF outward normal)
    sdf       : (Nx,Ny,Nz) signed distance to obstacles (positive outside, negative inside)
    r_proj    : radius of influence for normal projection (m). If None, defaults to 5*voxel
    returns:
      T : (Nx,Ny,Nz) geodesic distance field to the goal
      gf: (Nx,Ny,Nz,3) HumanoidPF guidance field (unit direction * speed)
    """
    voxel = cfg.voxel
    eps = 1e-9
    if r_proj is None:
        r_proj = 5.0 * voxel  # can be adjusted according to robot size

    X, Y, Z = grids
    # goal negative region (small sphere)
    phi = np.ones(obs_mask.shape, dtype=float)
    goal_seed = ((X - goal_local[0])**2 + (Y - goal_local[1])**2 + (Z - goal_local[2])**2) <= cfg.goal_seed_r**2
    phi[goal_seed] = -1.0
    phi = np.ma.MaskedArray(phi, mask=obs_mask)

    # Fast Marching: T (geodesic distance)
    T_ma = skfmm.distance(phi, dx=cfg.voxel)
    T_free_max = np.max(T_ma[~T_ma.mask]) if np.any(~T_ma.mask) else 0.0
    T = T_ma.filled(T_free_max).astype(np.float32)
    # -∇T
    dTx, dTy, dTz = np.gradient(T, voxel, voxel, voxel, edge_order=2)
    g = np.stack([-dTx, -dTy, -dTz], axis=-1).astype(np.float32)     # (...,3), g is the negative gradient of U_att in the paper (without eta, or eta=1.0)


    # 3) b̂: normalize the SDF gradient to a unit outward-normal direction (which way is "straight out of the wall")
    bnorm = np.linalg.norm(bf, axis=-1, keepdims=True)
    bunit = np.zeros_like(bf, dtype=np.float32)
    valid_b = bnorm[..., 0] > eps
    bunit[valid_b] = bf[valid_b] / bnorm[valid_b]


    # 4) g_perp: strip the wall-normal part out of g, leaving only flow that slides *along* the surface
    proj = np.sum(g * bunit, axis=-1, keepdims=True)  # g·b̂: how much of g points into/out of the wall
    # proj = np.clip(proj, -1.0, 0.0)
    g_perp = g - proj * bunit  # remove that component → tangential (wall-following) direction


    # 5) distance weight w ∈ [0,1]: a per-voxel "how close am I to a wall?" knob.
    #    Based on d_out = distance outside the obstacle (0 inside). w=1 at the surface -> w=0 at/beyond r_proj.
    #    smoothstep gives a smooth (no-kink) ramp instead of a linear one.
    d_out = np.maximum(sdf, 0.0)
    t = np.clip(d_out / (r_proj + eps), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    w = (1.0 - smooth)[..., None]  # (...,1)


    # 6) blend the two directions using w: near a wall (w->1) use the wall-following g_perp;
    #    far from obstacles (w->0) use the straight-to-goal g. In between, smoothly mix the two.
    g_mix = (1.0 - w) * g + w * g_perp
    # g_mix[g_mix[...,0]<0] *= -1

    # 7) never forget inside obstacles: use normal (usually outward normal to push field away; use -bunit to point inward)
    # from scipy.ndimage import binary_dilation
    # obs_mask = binary_dilation(obs_mask, iterations=1)
    g_mix[obs_mask] = bunit[obs_mask]


    # 8) normalize (final step) + speed scalar
    mag = np.linalg.norm(g_mix, axis=-1, keepdims=True)
    dir_unit = np.zeros_like(g_mix, dtype=np.float32)
    nz = (mag[..., 0] > eps)
    dir_unit[nz] = g_mix[nz] / mag[nz] # Throws away g_mix's length, keeping only its direction as unit vectors dir_unit.


    # A simple speed schedule: move at full speed v_max everywhere,
    # except within 0.3 m of the goal (in xy),
    # where speed ramps smoothly down to 0 with a cubic falloff. 
    T_thresh = 0.3
    p = 3.0 
    goal_dist = (((X - goal_local[0])**2 + (Y - goal_local[1])**2))**0.5
    speed = np.where(
        goal_dist > T_thresh,
        cfg.v_max,
        cfg.v_max * (goal_dist / T_thresh) ** p
    )
    speed = speed.astype(np.float32)[..., None]


    gf = (dir_unit * speed).astype(np.float32)
    return T, gf    # gf is not a unit-vector field, its lengths are specified by speed

def make_inflated_guidance_field(cfg, grids, sdf, bf, goal_local, margin=None, r_proj=None):
    """
    Anticipatory guidance field via Option-B (SDF-offset) obstacle inflation — meant
    for a large/heavy *carried box*, not the robot's own body. Inflating obstacles by
    a Euclidean `margin` makes -∇T curve away earlier and with standoff, so when the
    field is queried at the box location it already reads an avoidance direction before
    the box reaches the true obstacle surface.

    Exact for a true signed distance field: subtracting a constant shifts the zero-level
    (the obstacle surface) outward by `margin`, and the gradient (surface normals `bf`)
    is invariant under a constant offset. So we only re-threshold the mask and re-run the
    geodesic solve on the fattened obstacles — bf is reused unchanged, nothing is recomputed.

    cfg, grids, goal_local : as in make_guidance_field_progressive
    sdf    : (Nx,Ny,Nz) signed distance of the ORIGINAL (un-inflated) obstacles
    bf     : (Nx,Ny,Nz,3) ∇sdf normals of the original obstacles (reused as-is)
    margin : inflation radius (m). If None, uses cfg.box_inflation_margin
    r_proj : wall-following blend radius (m) for the inflated surface; None -> 5*voxel
    returns:
      T_inf  : (Nx,Ny,Nz) geodesic distance over the inflated obstacles
      gf_inf : (Nx,Ny,Nz,3) anticipatory guidance field
    """
    if margin is None:
        margin = cfg.box_inflation_margin
    sdf_inf = (sdf - margin).astype(np.float32)   # surface moves outward by `margin`
    obs_inf = sdf_inf <= 0.0                        # == (sdf <= margin): Euclidean Minkowski grow
    # bf == ∇sdf, and ∇(sdf - const) == ∇sdf, so the original normals are still correct.
    return make_guidance_field_progressive(cfg, grids, obs_inf, goal_local, bf, sdf_inf, r_proj=r_proj)

def save_all(cfg: PFConfig, sdf, bf, gf, obs_mask, meta_extra=None):    # currently not being used
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "sdf.npy", sdf)
    np.save(outdir / "bf.npy",  bf)
    np.save(outdir / "gf.npy",  gf)
    np.save(outdir / "obs.npy", obs_mask.astype(np.uint8))
    meta = {
        "voxel": cfg.voxel,
        "origin": cfg.origin_w,
        "shape_xyz": np.array(sdf.shape, dtype=np.int32),
        "start_w": cfg.start_w,
        "goal_w": cfg.goal_w,
        "scene": cfg.scene
    }
    if meta_extra:
        meta.update(meta_extra)
    np.save(outdir / "meta.npy", meta)
    print(f"[OK] Saved to {outdir}")

def visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, start_l, goal_l, bf=None, title_prefix=""):
    step = 3

    # --- Top view (xy plane at z ≈ robot waist) ---
    kz = int(np.argmin(np.abs(zv - start_l[2])))
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[:, :, kz].T, origin='lower',
                    extent=[xv[0], xv[-1], yv[0], yv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_xy = obs_mask[:, :, kz].T
    plt.contour(obs_xy, levels=[0.5], colors='k',
                extent=[xv[0], xv[-1], yv[0], yv[-1]])
    X2, Y2 = np.meshgrid(xv[::step], yv[::step], indexing='ij')
    U = gf[::step, ::step, kz, 0]; V = gf[::step, ::step, kz, 1]
    plt.quiver(X2, Y2, U, V, pivot='mid', scale=30, color='w', label='gf')
    if bf is not None:
        Ubf = bf[::step, ::step, kz, 0]; Vbf = bf[::step, ::step, kz, 1]
        plt.quiver(X2, Y2, Ubf, Vbf, pivot='mid', scale=30, color='yellow', alpha=0.6, label='bf')
    plt.scatter([start_l[0]],[start_l[1]], c='w', s=50, edgecolors='k', label='start')
    plt.scatter([goal_l[0]],[goal_l[1]], c='r', s=60, edgecolors='k', marker='*', label='goal')
    plt.title(f"{title_prefix} Top view (z≈{zv[kz]:.2f} m)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{title_prefix}_top.png", dpi=300)
    plt.close()

    # --- Side view (xz plane at y ≈ robot centerline) ---
    ky = int(np.argmin(np.abs(yv - start_l[1])))
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[:, ky, :].T, origin='lower',
                    extent=[xv[0], xv[-1], zv[0], zv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_xz = obs_mask[:, ky, :].T
    plt.contour(obs_xz, levels=[0.5], colors='k',
                extent=[xv[0], xv[-1], zv[0], zv[-1]])
    X2, Z2 = np.meshgrid(xv[::step], zv[::step], indexing='ij')
    U = gf[::step, ky, ::step, 0]; W = gf[::step, ky, ::step, 2]
    plt.quiver(X2, Z2, U, W, pivot='mid', scale=30, color='w', label='gf')
    if bf is not None:
        Ubf = bf[::step, ky, ::step, 0]; Wbf = bf[::step, ky, ::step, 2]
        plt.quiver(X2, Z2, Ubf, Wbf, pivot='mid', scale=30, color='yellow', alpha=0.6, label='bf')
    plt.scatter([start_l[0]],[start_l[2]], c='w', s=50, edgecolors='k')
    plt.scatter([goal_l[0]],[goal_l[2]], c='r', s=60, edgecolors='k', marker='*')
    plt.title(f"{title_prefix} Side view (y≈{yv[ky]:.2f} m)")
    plt.xlabel("x (m)"); plt.ylabel("z (m)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{title_prefix}_side.png", dpi=300)
    plt.close()

    # --- Front view (yz plane at x ≈ mid-scene) ---
    kx = int(np.argmin(np.abs(xv - 1.)))
    plt.figure(figsize=(7,5))
    im = plt.imshow(sdf[kx, :, :].T, origin='lower',
                    extent=[yv[0], yv[-1], zv[0], zv[-1]],
                    aspect='equal', cmap='coolwarm')
    plt.colorbar(im, label="SDF (m)")
    obs_yz = obs_mask[kx, :, :].T
    plt.contour(obs_yz, levels=[0.5], colors='k',
                extent=[yv[0], yv[-1], zv[0], zv[-1]])
    Y2, Z2 = np.meshgrid(yv[::step], zv[::step], indexing='ij')
    V = gf[kx, ::step, ::step, 1]; W = gf[kx, ::step, ::step, 2]
    plt.quiver(Y2, Z2, V, W, pivot='mid', scale=30, color='w', label='gf')
    if bf is not None:
        Vbf = bf[kx, ::step, ::step, 1]; Wbf = bf[kx, ::step, ::step, 2]
        plt.quiver(Y2, Z2, Vbf, Wbf, pivot='mid', scale=30, color='yellow', alpha=0.6, label='bf')
    plt.scatter([start_l[1]],[start_l[2]], c='w', s=50, edgecolors='k')
    plt.scatter([goal_l[1]],[goal_l[2]], c='r', s=60, edgecolors='k', marker='*')
    plt.title(f"{title_prefix} Front view (x≈{xv[kx]:.2f} m)")
    plt.xlabel("y (m)"); plt.ylabel("z (m)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{title_prefix}_front.png", dpi=300)
    plt.close()


