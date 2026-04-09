# CaTra: Carry and Traverse

Extension of **Click-and-Traverse (CAT)** where the Unitree G1 humanoid carries a box while navigating cluttered indoor scenes.

---

## Overview

CaTra adds a **23-DOF action space** (12 legs + 3 waist + 8 arms) and a carried box that the robot holds entirely by contact forces. The box rests on a fixed support surface (repositioned each episode) so it does not fall at the start of training. The policy is guided by the same HumanoidPF fields as CAT, sampled at body sites and additionally at the box center as observations.

### Key Differences from G1Cat

| Aspect | G1Cat | G1CaTra |
|--------|-------|---------|
| Action DOF | 12 (legs only) | 23 (legs + waist + arms) |
| Wrists | Not actuated | Not actuated (passive at default) |
| Action scale | 0.5 | 0.5 (same) |
| Default pose | Arms neutral/hanging | Arms neutral/hanging (same) |
| Hand collision | Single capsule | Single capsule (same) |
| Rewards | Body PF + locomotion | Same as G1Cat (no box-specific rewards) |
| Box | None | Free body, resting on support surface at random height |
| Box initialization | — | Surface height `U[0.4, 0.8]` m; box placed at palm-midpoint XY |
| Observations | 162-dim | 191-dim (+11 last_act, +11 motor_targets, +7 box PF) |
| Privileged obs | 224-dim | 259-dim (+11, +11, +7 box PF, +6 box pos/vel) |
| Scene XMLs | flat_terrain / mesh | flat_terrain_catra / mesh_catra |
| qpos size | 36 | 43 (robot 36 + box freejoint 7) |

---

## Installation

Same as the base CAT repo. No additional dependencies required.

```bash
source .venv/bin/activate && source .env
python -m cat_ppo.utils.mj_playground_init   # initialize MuJoCo assets
```

---

## Training

```bash
python train_ppo.py \
    --task G1CaTra \
    --exp_name catra_v1 \
    --obs_path data/assets/TypiObs/empty
```

### CLI Arguments

Same as `train_ppo.py`. Key ones:

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | — | Must be `G1CaTra` |
| `--exp_name` | `debug` | Experiment tag |
| `--ground` | 0 | Scale for feet guidance + SDF rewards |
| `--lateral` | 0 | Scale for hands/knees/shoulders SDF rewards |
| `--overhead` | 0 | Scale for head guidance + SDF rewards |
| `--obs_path` | `data/assets/TypiObs/empty` | Path to HumanoidPF fields |

---

## Reward Configuration

Identical to G1Cat. Default scales in `g1_catra_task_config()`:

| Reward | Scale | Description |
|--------|-------|-------------|
| `tracking_orientation` | 2.0 | Track upright orientation + torso alignment |
| `tracking_root_field` | 1.0 | Root velocity follows HumanoidPF guidance direction |
| `body_motion` | -0.5 | Penalize unintended body linear/angular motion |
| `body_rotation` | 1.0 | Reward torso facing the travel direction |
| `foot_contact` | -1.0 | Penalize incorrect foot contact w.r.t. gait phase |
| `foot_clearance` | -15.0 | Penalize insufficient swing foot clearance |
| `foot_slip` | -0.5 | Penalize foot sliding on the ground |
| `foot_balance` | -30.0 | Penalize CoP outside support polygon |
| `straight_knee` | -30.0 | Penalize locked-straight knee pose |
| `smoothness_joint` | -1e-6 | Penalize joint velocity jerk |
| `smoothness_action` | -1e-3 | Penalize action change rate |
| `joint_limits` | -1.0 | Penalize joint position near soft limits |
| `joint_torque` | -1e-4 | Penalize torque magnitude (energy) |
| `headgf` | 0.0 | Head guidance — activated via `--overhead` |
| `headdf` | 0.0 | Head SDF — activated via `--overhead` |
| `handsgf` | 0.0 | Hands guidance — activated via `--lateral` |
| `handsdf` | 0.0 | Hands SDF — activated via `--lateral` |
| `feetgf` | 0.0 | Feet guidance — activated via `--ground` |
| `feetdf` | 0.0 | Feet SDF — activated via `--ground` |
| `kneesdf` | 0.0 | Knees SDF — activated via `--lateral` |
| `shldsdf` | 0.0 | Shoulders SDF — activated via `--lateral` |

---

## Evaluation

### Export to ONNX

```bash
python -m cat_ppo.eval.brax2onnx --task G1CaTra --exp_name <full_exp_name>
```

### Play in MuJoCo viewer

```bash
python -m cat_ppo.eval.mj_onnx_play --task G1CaTra --exp_name <full_exp_name> --obs_path <scene_path>
```

---

## Scene Preparation

CaTra uses the same HumanoidPF scenes as CAT. Box PF fields (`boxgf`, `boxbf`, `boxdf`) are sampled from the same precomputed grids at the box center — no extra precomputation needed.

---

## Architecture

### MJCF Changes

- **[g1_mjx_feetonly_torque_catra.xml](data/assets/unitree_g1/g1_mjx_feetonly_torque_catra.xml)**: Separate robot XML with arm + waist actuators enabled (vs 12-leg-only in the base XML).
- **[scene_mjx_feetonly_flat_terrain_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml)**: Adds `carried_box` (freejoint) and `box_support` (mocap body, thin flat platform). Box rests on the support surface; surface height is randomized each episode via `data.mocap_pos`.
- **[scene_mjx_feetonly_mesh_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml)**: Mesh variant for visualization.

### Box Freejoint DOF Handling

Adding the box freejoint changes total qpos from 36 to 43 elements:
```
qpos: [0:7] root | [7:36] robot joints | [36:43] box freejoint
qvel: [0:6] root | [6:35] robot joints | [35:41] box vel
```

`torque_step_catra` and `domain_randomize_catra` use explicit `[7:36]` / `[6:35]` slices. `reset`, `step`, `_get_obs`, and `_get_termination` are fully overridden in `G1CaTraEnv`.

### Box Reset

Each episode:
1. Robot qpos initialized (random xy spawn, random yaw, randomized joints)
2. `mjx.forward()` to compute FK
3. Palm midpoint XY computed from `site_xpos`
4. `surface_z` sampled from `U[0.4, 0.8]` m
5. `data.mocap_pos[box_support]` set to `[palm_mid_x, palm_mid_y, surface_z]`
6. Box qpos set to `[palm_mid_x, palm_mid_y, surface_z + 0.08]` (surface top + box half-height), identity quaternion
7. `mjx.forward()` to settle

### Box Termination

Two conditions terminate the episode:
- `boxdf < -0.04` — box center is inside an obstacle
- `box_pos[2] < 0.3` — box has fallen to near floor level
