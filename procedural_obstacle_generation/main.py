from random_obstacle import make_axes, generate_and_save, extract_surface_voxels, Cfg as ObsCfg, get_elevation
from typical_obstacle import build_obstacles
import os
import shutil
from pathlib import Path
from pf_modular import make_sdf, make_guidance_field_progressive, grad3, PFConfig, visualize_all
import numpy as np
from utills import marching_cubes_mesh, occupancy_to_points, preview_matplotlib, combine_meshes
import itertools

# Resolved once so all helpers write to the same place regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASSETS = _REPO_ROOT / "data" / "assets"
_FIG    = _REPO_ROOT / "fig"


# function fo generating random obstacles
def generate_random_obstacle(difficulty, seed, n_rect_R, n_rect_F, n_rect_C):
    # difficulty [0,1]: controls geometry of each block, NOT the count.
    #   - higher → floor/ceiling blocks span wider laterally (harder to go around)
    #   - higher → more frequent tight-gap segments in the passable corridor
    #   - n_rect_* and difficulty are orthogonal: difficulty shapes each block, n_rect_* sets how many
    # seed: numpy RNG seed (np.random.default_rng), fully deterministic — same seed → same scene
    # n_rect_R / n_rect_L: number of right/left wall blocks (L is always set equal to R below)
    # n_rect_F: number of floor blocks (low obstacles, max height 0.25m, robot must step over)
    # n_rect_C: number of ceiling blocks (hanging from above min z=1.0m, robot must duck under)
    # NOTE: pass -1 for any n_rect_* to auto-compute it from difficulty instead of specifying manually
    n_rect_L=n_rect_R
    prefix = f"D{int(difficulty*10)}G{n_rect_F:01d}L{n_rect_R:01d}O{n_rect_C:01d}S{seed}/"

    print(prefix)
    save = False    # set save to False
    if save:
        os.makedirs(prefix, exist_ok=True)
    obs_cfg = ObsCfg(difficulty=difficulty, seed=seed, n_rect_L=n_rect_L, n_rect_R=n_rect_R, n_rect_F=n_rect_F, n_rect_C=n_rect_C)
    obs_mask, xv, yv, zv = generate_and_save(obs_cfg, prefix=prefix, save=save) # obs_mask: (Nx, Ny, Nz), xv: (Nx,), yv: (Ny,), zv: (Nz,)
    if obs_mask.any() == False:
        shutil.rmtree(prefix)
        return

    out_dir = _ASSETS / "RandObs" / prefix.rstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    # pts = occupancy_to_points(obs_mask, voxel_size=obs_cfg.voxel)
    # preview_matplotlib(pts)
    spacing = (obs_cfg.voxel, obs_cfg.voxel, obs_cfg.voxel)
    mesh = better_mesh(spacing, obs_mask)
    mesh.export(str(out_dir / "obs.obj"))

    cfg = PFConfig()
    cfg.voxel = obs_cfg.voxel
    cfg.start_w = obs_cfg.start_w
    cfg.goal_w = obs_cfg.goal_w
    cfg.origin_w = obs_cfg.origin_w
    cfg.Lx = obs_cfg.Lx
    cfg.Ly = obs_cfg.Ly
    cfg.Lz = obs_cfg.Lz
    
    # SDF and gradient (boundary field)
    sdf = make_sdf(obs_mask, cfg.voxel) # shape (Nx, Ny, Nz)
    bf  = grad3(sdf, cfg.voxel) # shape (Nx, Ny, Nz, 3)

    # HumanoidPF and gradient
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij') # X, Y, Z have the same shape (Nx, Ny, Nz). X contains the x-coordinates of all voxels, etc.
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    np.save(out_dir / "sdf.npy", sdf)
    np.save(out_dir / "bf.npy",  bf)
    np.save(out_dir / "gf.npy",  gf)
    np.save(out_dir / "obs.npy", obs_mask.astype(np.uint8))
    # sur = extract_surface_voxels(obs_mask)
    # np.save(out_dir / "sur.npy", sur.astype(np.uint8))
    # pts = occupancy_to_points(sur, voxel_size=obstacle_cfg.voxel)
    # preview_matplotlib(pts)

    _FIG.mkdir(parents=True, exist_ok=True)
    visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w, bf=bf,
                  title_prefix=str(_FIG / prefix.rstrip("/")))

# function fo generating typical obstacles
def generate_typical_obstacle(scene_type):
    prefix = f"{scene_type}/"
    obs_cfg = ObsCfg()
    cfg = PFConfig()
    # Sanity check: PFConfig and ObsCfg must agree on all geometric parameters.
    # If they diverge (e.g. voxel size changed in one but not the other), the obstacle
    # occupancy grid and potential field would be built on incompatible grids.
    assert cfg.voxel == obs_cfg.voxel
    assert (cfg.start_w == obs_cfg.start_w).all()
    assert (cfg.goal_w == obs_cfg.goal_w).all()
    assert (cfg.origin_w == obs_cfg.origin_w).all()
    assert cfg.Lx == obs_cfg.Lx
    assert cfg.Ly == obs_cfg.Ly
    assert cfg.Lz == obs_cfg.Lz

    xv, yv, zv = make_axes(cfg)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')    # make a 3D grid of points in the world coordinate system, with shape (Nx, Ny, Nz)
    obs_mask = build_obstacles(scene_type, (X, Y, Z))   # build the obstacle occupancy grid for the given scene type, with shape (Nx, Ny, Nz) and dtype bool

    out_dir = _ASSETS / "TypiObs" / prefix.rstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    # pts = occupancy_to_points(obs_mask, voxel_size=cfg.voxel)
    # preview_matplotlib(pts)
    spacing = (cfg.voxel, cfg.voxel, cfg.voxel)
    mesh = better_mesh(spacing, obs_mask)
    mesh.export(str(out_dir / "obs.obj"))   # export the obstacle mesh as an OBJ file for visualization in MuJoCo or other 3D software 

    sdf = make_sdf(obs_mask, cfg.voxel) # compute sdf
    bf  = grad3(sdf, cfg.voxel) # compute boundary field (outward normals at obstacle surfaces) by taking the gradient of the SDF

    # T: geodesic distance to goal (Nx,Ny,Nz). gf: HumanoidPF vector field (Nx,Ny,Nz,3).
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    np.save(out_dir / "sdf.npy", sdf)   # shape (Nx, Ny, Nz)
    np.save(out_dir / "bf.npy",  bf)   # shape (Nx, Ny, Nz, 3)
    np.save(out_dir / "gf.npy",  gf)   # shape (Nx, Ny, Nz, 3)
    np.save(out_dir / "obs.npy", obs_mask.astype(np.uint8))  # shape (Nx, Ny, Nz), dtype uint8
    # sur = extract_surface_voxels(obs_mask)
    # np.save(out_dir / "sur.npy", sur.astype(np.uint8))
    # ground_idx, ceil_idx = get_elevation(obs_mask)
    # np.save(out_dir / "ground.npy", ground_idx)
    # np.save(out_dir / "ceil.npy", ceil_idx)
    # pts = occupancy_to_points(sur, voxel_size=obs_cfg.voxel)
    # preview_matplotlib(pts)

    # visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w)


def generate_pf(scene_type, pc_path):   # used in deployment to generate PF from real-world occupancy (point cloud) data
    prefix = f"{scene_type}/"
    obs_cfg = ObsCfg()
    cfg = PFConfig()
    # Sanity check: PFConfig and ObsCfg must agree on all geometric parameters.
    # If they diverge (e.g. voxel size changed in one but not the other), the obstacle
    # occupancy grid and potential field would be built on incompatible grids.
    assert cfg.voxel == obs_cfg.voxel
    assert (cfg.start_w == obs_cfg.start_w).all()
    assert (cfg.goal_w == obs_cfg.goal_w).all()
    assert (cfg.origin_w == obs_cfg.origin_w).all()
    assert cfg.Lx == obs_cfg.Lx
    assert cfg.Ly == obs_cfg.Ly
    assert cfg.Lz == obs_cfg.Lz

    xv, yv, zv = make_axes(cfg)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    # obs_mask = torch.load(pc_path).cpu().numpy()
    obs_mask = np.load(pc_path, allow_pickle=True) 
    if obs_mask.dtype != np.bool_:
        obs_mask = obs_mask.astype(np.uint8) > 0

    # some pre-processing for real-to-sim occupancy
    z_bar_thresh = 6
    mask_bar = obs_mask[:, :, :z_bar_thresh] == 1
    filled = np.cumsum(mask_bar[:, :, ::-1], axis=2)[:, :, ::-1] > 0
    obs_mask[:, :, :z_bar_thresh] = filled.astype(np.uint8)

    out_dir = _ASSETS / "R2SObs" / prefix.rstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = occupancy_to_points(obs_mask, voxel_size=cfg.voxel)
    preview_matplotlib(pts)
    spacing = (cfg.voxel, cfg.voxel, cfg.voxel)
    mesh = marching_cubes_mesh(obs_mask, spacing=spacing)
    mesh.export(str(out_dir / "obs.obj"))

    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)
    # Eikonal 
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    # 保存
    np.save(out_dir / "sdf.npy", sdf)   # shape (Nx, Ny, Nz)
    np.save(out_dir / "bf.npy",  bf)   # shape (Nx, Ny, Nz, 3)
    np.save(out_dir / "gf.npy",  gf)   # shape (Nx, Ny, Nz, 3)
    np.save(out_dir / "obs.npy", obs_mask.astype(np.uint8))  # shape (Nx, Ny, Nz), dtype uint8
    # sur = extract_surface_voxels(obs_mask)
    # np.save(out_dir / "sur.npy", sur.astype(np.uint8))
    # 可视化
    # visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w)


def better_mesh(spacing, obs_mask):
    # Clear boundary voxels to avoid open/broken geometry at grid edges.
    obs_mask[:,0,:] = 0
    obs_mask[:,-1,:] = 0
    obs_mask[:,:,-1] = 0
    # Invert so obstacles=0, free=1. Marching cubes then traces the 0.5 isosurface,
    # which is the outer skin of the obstacles — the geometry MuJoCo needs for rendering.
    obs_mask_erosion = 1-obs_mask
    mesh = marching_cubes_mesh(obs_mask_erosion, spacing=spacing)
    return mesh

if __name__ == "__main__":
    # difficulties = [0.8]
    # d_Ls = [1,3]
    # d_Gs = [0, 1, 2]
    # d_Os = [0, 1, 2]
    # seeds = [1,2]
    # combos = itertools.product(difficulties, d_Ls, d_Gs, d_Os, seeds)
    # for difficulty, dL, dG, dO, seed in combos:
    #     rng = (dL + 0.5 * seed) * (dG + seed) * (dO + 1.5 * seed) + 1
    #     generate_random_obstacle(difficulty, int(rng), dL, dG, dO)
    # difficulties = [0.7]
    # d_Ls = [4]
    # d_Gs = [0, 1, 2]
    # d_Os = [0, 1, 2]
    # seeds = [1,2]
    # combos = itertools.product(difficulties, d_Ls, d_Gs, d_Os, seeds)
    # for difficulty, dL, dG, dO, seed in combos:
    #     rng = (dL + 0.5 * seed) * (dG + seed) * (dO + 1.5 * seed)
    #     generate_random_obstacle(difficulty, int(rng), dL, dG, dO)
    # generate_typical_obstacle('ceil1')
    # generate_typical_obstacle('bar0')
    # generate_typical_obstacle('bar1')
    # generate_typical_obstacle('bar2')
    # generate_typical_obstacle('bar3')
    # generate_typical_obstacle('Mceil0')
    # generate_typical_obstacle('Mceil1')
    # generate_typical_obstacle('Mbar0')
    # generate_typical_obstacle('Mbar1')
    # generate_typical_obstacle('ceilbar0')
    # generate_typical_obstacle('ceilbar1')
    # generate_typical_obstacle('chest')
    # generate_typical_obstacle('Nbar0')
    # generate_typical_obstacle('Nbar1')
    # generate_typical_obstacle('doubar')
    # generate_typical_obstacle('lowcorner')
    # generate_typical_obstacle('hole')
    # generate_random_obstacle(0.8, 13, 1, 0, 0)
    # generate_random_obstacle(0.8, 4, 1, 0, 1)
    generate_random_obstacle(0.9, 42, 9, 5, 3)  # generate a random obstacle scene with difficulty 0.2, seed 42, 9 left-wall blocks, 3 front-wall blocks, 3 overhead blocks
