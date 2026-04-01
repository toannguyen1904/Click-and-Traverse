# CaTra: Carry and Traverse

Extension of **Click-and-Traverse (CAT)** where the Unitree G1 humanoid carries a box while navigating cluttered indoor scenes.

---

## Overview

CaTra adds a **23-DOF action space** (12 legs + 3 waist + 8 arms) and a carried box attached via MuJoCo weld equality constraints. The policy is guided by HumanoidPF fields sampled both at body sites (as in CAT) and at the box center, so the robot learns to move the box away from obstacles while traversing the scene.

### Key Differences from G1Cat

| Aspect | G1Cat | G1CaTra |
|--------|-------|---------|
| Action DOF | 12 (legs only) | 23 (legs + waist + arms) |
| Action scale | 0.5 | 0.3 |
| Default pose | Arms neutral | Arms in carrying pose (shoulder_pitch=1.5, elbow=1.5) |
| Observations | 162-dim | 191-dim (+7 box PF, +22 for extra actuators) |
| Privileged obs | 224-dim | 259-dim (+35 extra) |
| Box | None | Welded to both wrists, mass 1.5 kg |
| Scene XMLs | flat_terrain / mesh | flat_terrain_catra / mesh_catra |

---

## Installation

Same as the base CAT repo. No additional dependencies required.

```bash
source .venv/bin/activate && source .env
python -m cat_ppo.utils.mj_playground_init   # initialize MuJoCo assets
```

---

## Training

### Basic (no obstacle rewards)

Train with rigid weld, no box or body-collision reward — robot learns to walk while carrying:

```bash
python train_ppo.py \
    --task G1CaTra \
    --exp_name catra_v1 \
    --obs_path data/assets/TypiObs/empty
```

### With HumanoidPF obstacle avoidance

Add `--ground`, `--lateral`, `--overhead`, `--box` to activate collision and box guidance rewards:

```bash
python train_ppo.py \
    --task G1CaTra \
    --exp_name catra_v1 \
    --ground 1.0 \
    --lateral 1.0 \
    --overhead 1.0 \
    --box 1.0 \
    --obs_path data/assets/TypiObs/narrow1
```

### Debug run (fast iteration)

```bash
python train_ppo.py --task G1CaTra --exp_name debug --obs_path data/assets/TypiObs/empty
```

Debug mode is triggered automatically when `"debug"` appears in `--exp_name`. It uses fewer environments and timesteps for quick shape/reward sanity checks.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | — | Must be `G1CaTra` |
| `--exp_name` | `debug` | Experiment tag (appended to auto-generated name) |
| `--num_timesteps` | 400M | Total environment steps |
| `--ground` | 0 | Scale for feet guidance + SDF rewards (ground obstacles) |
| `--lateral` | 0 | Scale for hands/knees/shoulders SDF rewards (lateral obstacles) |
| `--overhead` | 0 | Scale for head guidance + SDF rewards (overhead obstacles) |
| `--box` | 0 | Scale for box guidance + box SDF rewards |
| `--obs_path` | `data/assets/TypiObs/empty` | Path to HumanoidPF fields (sdf.npy, bf.npy, gf.npy) |
| `--term_collision_threshold` | 0.04 | SDF value below which collision termination triggers |
| `--seed` | 42 | Random seed |
| `--restore_name` | `none` | Resume from checkpoint (experiment name) |
| `--convert_onnx` | True | Export ONNX policy after training |

---

## Reward Configuration

Default reward scales in `g1_catra_task_config()` (see [env_catra.py](cat_ppo/envs/g1/env_catra.py)):

| Reward | Scale | Description |
|--------|-------|-------------|
| `tracking_lin_vel` | 1.5 | Follow linear velocity command |
| `tracking_ang_vel` | 0.75 | Follow angular velocity command |
| `lin_vel_z` | -2.0 | Penalize vertical root velocity |
| `ang_vel_xy` | -0.05 | Penalize roll/pitch angular velocity |
| `orientation` | -5.0 | Penalize tilt from upright |
| `feet_air_time` | 0.5 | Encourage foot liftoff |
| `foot_slip` | -0.1 | Penalize foot sliding |
| `action_rate` | -0.01 | Penalize action changes |
| `energy` | -1e-4 | Penalize torque × velocity |
| `alive` | 1.0 | Per-step survival bonus |
| `arm_pose` | -0.5 | Penalize arm deviation from carrying pose |
| `arm_smoothness` | -1e-3 | Penalize arm action magnitude |
| `boxgf` | 0.0 | Box moves in guidance direction (activated via `--box`) |
| `boxdf` | 0.0 | Box avoids obstacles (activated via `--box`) |
| `feetgf` | 0.0 | Feet guidance (activated via `--ground`) |
| `feetdf` | 0.0 | Feet SDF (activated via `--ground`) |
| `handsgf` | 0.0 | Hands guidance (activated via `--lateral`) |
| `handsdf` | 0.0 | Hands SDF (activated via `--lateral`) |
| `headgf` | 0.0 | Head guidance (activated via `--overhead`) |
| `headdf` | 0.0 | Head SDF (activated via `--overhead`) |

---

## Evaluation

### Export to ONNX

After training completes, the ONNX policy is auto-exported. To export manually:

```bash
python -m cat_ppo.eval.brax2onnx --task G1CaTra --exp_name <full_exp_name>
```

### Play in MuJoCo viewer

```bash
python -m cat_ppo.eval.mj_onnx_play --task G1CaTra --exp_name <full_exp_name> --obs_path <scene_path>
```

Use `mesh_catra` scene XML for visualization (set in `constants.task_to_xml`).

---

## Scene Preparation

CaTra uses the same HumanoidPF scenes as CAT. The box PF fields (`boxgf`, `boxbf`, `boxdf`) are sampled from the same precomputed `sdf.npy / bf.npy / gf.npy` at the box center position.

```bash
cd procedural_obstacle_generation && python main.py
```

To use a specific scene:
```bash
--obs_path data/assets/TypiObs/narrow1
--obs_path data/assets/TypiObs/empty       # flat terrain for initial training
--obs_path data/assets/RandObs/<scene>
```

---

## Architecture

### MJCF Changes

- **[g1_mjx_feetonly_torque.xml](data/assets/unitree_g1/g1_mjx_feetonly_torque.xml)**: Hand collision replaced with multi-geom cupped hand (palm box + fingers box + thumb capsule) for more realistic contact area.
- **[scene_mjx_feetonly_flat_terrain_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml)**: Adds `carried_box` body with freejoint. No weld constraints (Phase 3). Box initialized at palm midpoint each reset via FK.
- **[scene_mjx_feetonly_mesh_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml)**: Mesh variant for visualization.

### Box Freejoint DOF Handling

Adding the box freejoint changes total qpos from 36 to 43 elements:
```
qpos: [0:7] root | [7:36] robot joints | [36:43] box freejoint
qvel: [0:6] root | [6:35] robot joints | [35:41] box vel
```

`torque_step_catra` and `domain_randomize_catra` explicitly slice `[7:36]` / `[6:35]` to avoid shape mismatches inherited from `G1LocoEnv`.

### Weld Curriculum

Currently implemented: **Phase 3 (contact-only)** directly.

MJX does not support `weld` or `connect` equality constraints when the model also contains `accelerometer` or `force` sensors (both present in the G1 robot XML). Phase 3 was adopted as the starting point since it avoids this limitation entirely and is the final training goal.

| Phase | Weld | Behavior |
|-------|------|----------|
| 1 (skipped) | Rigid weld | Box locked to wrists — not MJX-compatible with G1 sensor setup |
| 2 (skipped) | Soft weld | Box wobbles — same MJX incompatibility |
| 3 (current) | No weld | Box held by contact forces only; reset places box at palm midpoint via FK |

---

## Known Limitations and TODOs

1. **Box drop in early training**: With no weld, the box falls immediately until the policy learns to press with both arms. The `arm_pose=-0.5` reward provides a natural signal, but early episodes will be short if `boxdf` termination fires. Consider disabling `boxdf` termination initially.

2. **Carrying pose geometry**: The FK-based init places the box at the palm midpoint, but the default carrying pose (`shoulder_pitch=1.5, elbow=1.5`) may not produce enough contact force to hold the box. Verify with `check_catra.py` and tune `DEFAULT_QPOS_CATRA` if needed.

3. **Box size randomization**: Currently box size is fixed. Randomizing `model.geom_size` per-environment in MJX requires modifying the randomization pipeline. Only mass is randomized for now.

4. **num_pri count**: If ONNX export fails with a shape error, verify that `num_pri=259` matches the actual privileged observation dimension (observed at 285 during testing). Training infers network sizes from actual data, so only export is affected.

5. **Action scale**: Arms have 25 Nm torque limits vs 88–139 Nm for legs. The global `action_scale=0.3` is conservative. A per-joint scale vector may give better performance.
