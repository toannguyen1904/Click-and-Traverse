# 3D-FRONT Scene Generation for CaTra Training

This document describes the `realistic_obstacle_generation/` pipeline, which converts scenes from the [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset into training-ready cluttered-scene captures for the CaTra robot.

## Motivation

The CaTra policy is trained on procedurally-generated obstacle scenes (`procedural_obstacle_generation/`). To improve generalization to real indoor environments, the paper additionally leverages 3D-FRONT — 6,813 professionally designed furnished apartments — as a source of realistic clutter. This pipeline processes those scenes into the same artifact format the training code already consumes.

## Dataset Setup

Download the 3D-FRONT and 3D-FUTURE datasets and place them at:

```
/data/tientoan/3DFRONT/
├── 3D-FRONT/              # 6,813 JSON scene files (one per apartment)
├── 3D-FUTURE-model/       # 16,563 furniture OBJ models (one subdirectory per model UUID)
│   ├── model_info.json
│   └── {model_uuid}/
│       └── normalized_model.obj
└── README.md
```

The path can be changed via `RealCfg.data_root` (see `realistic_obstacle_generation/config.py`).

## Pipeline Overview

For each scene, the pipeline:

1. **Identifies walkable area** — rasterizes floor mesh polygons onto a 2D grid, subtracts the ground footprint of all furniture whose AABB bottom ≤ 1.0 m (ceiling-mounted items such as chandeliers are skipped so they do not block the walkable region), then erodes by 0.10 m to reserve robot clearance.

2. **Samples a start position** — picks a random point inside the walkable mask.

3. **Samples a goal position** — samples uniformly on a 2 m radius circle around the start, rejecting angles whose landing point falls outside the walkable mask. Up to 8 × 64 retries are attempted before the scene is marked unusable.

4. **Crops a 5 × 5 m block** centered at the start position.

5. **Voxelizes furniture** — every 3D-FUTURE furniture mesh that intersects the 5 × 5 × 1.5 m crop box (including ceiling-mounted items — they appear in the 3D obstacle grid so the robot can learn to duck under them) is transformed to world Z-up frame and voxelized via `trimesh.voxelized().fill()` at 4 cm resolution.

6. **Computes spatial fields** — SDF, boundary field (SDF gradient), and HumanoidPF guidance field (toward the sampled goal) are computed by reusing `procedural_obstacle_generation/pf_modular.py`.

7. **Writes artifacts** — all outputs land in `data/assets/RealObs/{scene_uid}_S{seed}/`, in the same format the CaTra training environments read.

8. **Saves visualizations** — three PNG views (front, side, top) are written to `fig/{scene_uid}_S{seed}_front/side/top.png`.

## Coordinate Conventions

3D-FRONT uses a **Y-up**, meter-scale world frame. Quaternions are `[x, y, z, w]`. This pipeline converts all geometry to the project's **Z-up** frame via:

```
(x, y, z)_zup = (x_3df, -z_3df, y_3df)
```

Furniture meshes in 3D-FUTURE (`normalized_model.obj`) are not unit-cube; they are rescaled per-axis by `size / extents_norm` using the `size` field from the scene JSON, then rotated and translated by the instance transform.

Architectural meshes (walls, ceilings, doors, windows) are **excluded from the obstacle map** — only 3D-FUTURE furniture instances become voxels. Floor meshes are used solely to define the walkable region.

## Output Format

Each capture writes the following files to `data/assets/RealObs/{scene_uid}_S{seed}/`:

| File | Shape | Description |
|---|---|---|
| `obs.npy` | `(125, 125, 38)` uint8 | Binary occupancy voxel grid |
| `sdf.npy` | `(125, 125, 38)` float32 | Signed distance field (m) |
| `bf.npy` | `(125, 125, 38, 3)` float32 | Boundary field (SDF gradient) |
| `gf.npy` | `(125, 125, 38, 3)` float32 | HumanoidPF guidance field toward goal |
| `obs.obj` | — | Marching-cubes obstacle mesh (MuJoCo visualization) |

Grid parameters: `Lx = Ly = 5.0 m`, `Lz = 1.5 m`, `voxel = 0.04 m`, `origin_w = (start.x − 2.5, start.y − 2.5, 0)`.

These are identical in format to `RandObs/` and `TypiObs/` scenes — downstream training code requires no changes.

## Usage

### Generate one scene

```bash
source .venv/bin/activate

# Generate scene index 0 with seed 0
python -m realistic_obstacle_generation.main 0 --seed 0

# Output:
#   data/assets/RealObs/{uid}_S0/  (obs/sdf/bf/gf/obs.obj)
#   fig/{uid}_S0_front.png
#   fig/{uid}_S0_side.png
#   fig/{uid}_S0_top.png
```

### Generate scenes in a loop

```python
from realistic_obstacle_generation.main import generate_realistic_scene

for idx in range(100):
    meta = generate_realistic_scene(idx, seed=0)
    if meta is None:
        print(f"scene {idx}: unusable, skipping")
        continue
    print(meta["out_dir"])
```

`generate_realistic_scene` returns `None` (no exception, no partial writes) when a scene has no walkable area or the retry budget is exhausted. The caller should try the next index.

### Visualize an obstacle mesh

```bash
python procedural_obstacle_generation/render_obj.py \
    --mesh_file data/assets/RealObs/{uid}_S0/obs.obj
```

### Customize generation parameters

```python
from realistic_obstacle_generation.config import RealCfg
from realistic_obstacle_generation.main import generate_realistic_scene
from pathlib import Path

cfg = RealCfg(
    voxel=0.04,
    Lx=5.0, Ly=5.0, Lz=1.5,
    goal_radius=2.0,
    walk_erode_r=0.10,
    footprint_z_max=1.0,
    max_start_retries=8,
    max_goal_retries=64,
    data_root=Path("/data/tientoan/3DFRONT"),
    out_root=Path("data/assets/RealObs"),
)
meta = generate_realistic_scene(42, seed=7, cfg=cfg)
```

## Module Structure

```
realistic_obstacle_generation/
├── config.py        # RealCfg dataclass — all tuneable parameters
├── front_loader.py  # 3D-FRONT JSON + 3D-FUTURE OBJ loading, transforms, Y-up→Z-up
├── walkable.py      # 2D walkable mask construction and start/goal sampling
├── voxelize.py      # Per-furniture 3D voxelization into the local crop grid
└── main.py          # generate_realistic_scene() entry point + CLI
```

Reused from `procedural_obstacle_generation/` (read-only):
- `pf_modular.py` — `make_sdf`, `grad3`, `make_guidance_field_progressive`, `visualize_all`
- `utills.py` — `marching_cubes_mesh`

## Known Limitations

- **Walls excluded**: room walls are not voxelized. The robot must rely on the guidance field pointing away from the scene boundary rather than colliding with wall geometry.
- **~7 s per scene**: most of the time is `trimesh.voxelized().fill()` (~0.7 s per furniture item × ~30 items). For bulk offline generation this is fine; it is not suitable for on-the-fly episode resets.
- **Some scenes unusable**: scenes with very small rooms or extreme furniture density may fail the walkable-area check. In practice, fewer than 5% of scenes return `None` for default parameters.
- **No inter-room clutter**: each capture is a 5 × 5 m crop within one apartment. Furniture from adjacent rooms may appear at the crop boundary but the walkable mask is computed globally across all rooms.
