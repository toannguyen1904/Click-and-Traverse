# CaTra: Carry and Traverse

Extension of **Click-and-Traverse (CAT)** where the Unitree G1 humanoid carries a box while navigating cluttered indoor scenes.

---

## Overview

CaTra adds a **23-DOF action space** (12 legs + 3 waist + 8 arms) and a carried box that the robot holds entirely by contact forces. The policy is guided by HumanoidPF fields sampled both at body sites (as in CAT) and at the box center, so the robot learns to keep the box away from obstacles while traversing the scene.

### Key Differences from G1Cat

| Aspect | G1Cat | G1CaTra |
|--------|-------|---------|
| Action DOF | 12 (legs only) | 23 (legs + waist + arms) |
| Wrists | Not actuated | Not actuated (passive at default) |
| Action scale | 0.5 | 0.3 (conservative; arm torque limits 25 Nm vs 88–139 Nm for legs) |
| Default pose | Arms neutral/hanging | Carrying pose: `shoulder_pitch=1.5 rad`, `elbow=1.5 rad` |
| Box | None | Free body, mass 1.5 kg (randomized 0.5–3.0 kg), held by contact |
| Box initialization | — | Reset at palm midpoint each episode via FK |
| Observations | 162-dim | 191-dim (+11 last_act, +11 motor_targets, +7 box PF) |
| Privileged obs | 224-dim | 259-dim (+11, +11, +7 box PF, +6 box pos/vel) |
| Hand collision | Single capsule | Multi-geom cupped hand (palm + fingers + thumb) |
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

### Basic (no obstacle rewards)

Robot learns to walk while carrying the box, with arm stabilization rewards only:

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
    --exp_name G1CaTra_empty \
    --ground 1.0 \
    --lateral 1.0 \
    --overhead 1.0 \
    --box 1.0 \
    --obs_path data/assets/TypiObs/empty
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
| `box_height` | -2.0 | Penalize box below carrying height (always active; 0 above target, proportional penalty below) |
| `boxgf` | 0.0 | Box moves in guidance direction — activated via `--box` |
| `boxdf` | 0.0 | Box SDF penalty (box near/in obstacles) — activated via `--box` |
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

CaTra uses the same HumanoidPF scenes as CAT. The box PF fields (`boxgf`, `boxbf`, `boxdf`) are sampled from the same precomputed `sdf.npy / bf.npy / gf.npy` grids at the box center position — no extra precomputation needed.

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

## Design Notes

### Action Scale and Motor Targets (PD Control)

The policy does **not** output torques directly. Instead it outputs a **delta in joint-position space**, applied to the running motor target each step:

```
motor_target[t] = motor_target[t-1] + action * action_scale
torque = Kp * (motor_target - current_qpos) + Kd * (0 - current_qvel)
```

`action_scale` limits how far the desired joint position can shift per policy step. The environment does **not** wait for joints to reach the target — each policy step is `ctrl_dt=0.02 s` and runs 10 physics substeps of `sim_dt=0.002 s` each, all applying PD torques toward the same fixed motor target.

`action_scale=0.3` for CaTra vs `0.5` for G1Cat: arm joints have much lower torque limits (25 Nm vs 88–139 Nm for legs), so a smaller scale prevents large target shifts that the actuators cannot follow.

### HumanoidPF Box Fields: `boxgf`, `boxbf`, `boxdf`

These are the same three HumanoidPF fields used for body sites (head, feet, hands, …), evaluated at the **box center** instead of a body site:

| Field | Meaning | Shape |
|-------|---------|-------|
| `boxgf` | **Guidance field** — 3D vector pointing toward the goal along an obstacle-free geodesic path | (3,) |
| `boxbf` | **Boundary field** — 3D vector pointing away from the nearest obstacle surface (SDF gradient) | (3,) |
| `boxdf` | **Distance field** — scalar SDF at the box center; positive = free space, negative = inside obstacle | (1,) |

In the deployable `state` observation they are transformed into the navigation frame (same delayed/noisy treatment as body PF fields). In `privileged_state` they are kept in world frame and supplemented with noiseless `box_pos` and `box_vel`.

### Box Holding: `box_height` Reward and Drop Termination

The robot is free to hold the box in any arm configuration — there is no fixed carrying pose enforced as a reward. Instead, the incentive to hold comes from two complementary signals:

**`box_height` reward** (always active, scale -2.0):
```python
reward = clip(box_pos[2] - box_height_target, -1.0, 0.0)
```
- Returns `0` when the box is at or above `box_height_target` (default 0.7 m)
- Returns a negative value proportional to how far below the target the box is, capped at -1
- Encourages the robot to keep the box at waist height or above, regardless of arm pose

**Drop termination** (hard stop):
```python
terminate = box_pos[2] < box_drop_threshold   # default 0.3 m
```
- Ends the episode immediately if the box falls near floor level
- Complementary to `box_height`: the reward gives a soft gradient, the termination gives a hard boundary

**Full holding incentive chain:**
1. `box_height` → soft gradient encouraging the box to stay at carrying height
2. Episode termination at `box_drop_threshold` → hard signal if box reaches the floor
3. `boxdf` → penalizes the box entering obstacles (activated via `--box`)
4. `boxgf` → rewards the box moving toward the goal (activated via `--box`)

### Hand Geometry: Cupped Hand vs. Inspire Hand

The base G1 MJCF has no finger geometry — only a single capsule collision stub at the wrist. CaTra replaces this with a three-geom cupped hand (palm box + fingers box + thumb capsule) to provide a contact surface for box holding.

The Unitree G1 EDU is optionally equipped with an **Inspire RH56DFX dexterous hand** (~12 DOF per hand). Adding it for collision-only purposes — finger joints fixed in a pre-grip pose, no extra action DOFs — would give more accurate contact geometry and is architecturally straightforward. The current blocker is that the Inspire hand MJCF is not publicly distributed. If the MJCF becomes available, the finger joints can be locked via `<equality>` constraints and included purely for collision.

---

## Architecture

### MJCF Changes

- **[g1_mjx_feetonly_torque.xml](data/assets/unitree_g1/g1_mjx_feetonly_torque.xml)**: Hand collision replaced with multi-geom cupped hand (palm box + fingers box + thumb capsule).
- **[scene_mjx_feetonly_flat_terrain_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml)**: Adds `carried_box` body with freejoint. Box initialized at palm midpoint each reset via FK.
- **[scene_mjx_feetonly_mesh_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml)**: Mesh variant for visualization.

### Box Freejoint DOF Handling

Adding the box freejoint changes total qpos from 36 to 43 elements:
```
qpos: [0:7] root | [7:36] robot joints | [36:43] box freejoint
qvel: [0:6] root | [6:35] robot joints | [35:41] box vel
```

All parent-class methods that use `data.qpos[7:]` or `data.qvel[6:]` assume 29-element robot-joint slices and break with the extra 7 box DOFs. `torque_step_catra` and `domain_randomize_catra` use explicit `[7:36]` / `[6:35]` slices; `reset`, `step`, `_get_obs`, `_get_reward`, and `_get_termination` are fully overridden in `G1CaTraEnv` for the same reason.

### Box Reset via Forward Kinematics

At each episode reset, the box is placed at the midpoint between the two palm sites:
1. Initialize robot qpos in the carrying pose
2. Run `mjx.forward()` to compute FK
3. Read `site_xpos` for left and right palms
4. Set box freejoint qpos to their midpoint (position) + identity quaternion
5. Re-run `mjx.forward()` to settle

This gives the policy a consistent starting configuration where the box is already in the hands.

---

## Known Limitations and TODOs

1. **Box drop in early training**: The box falls immediately until the policy discovers arm–box contact. The `box_height` reward provides a gradient, but if `box_drop_threshold` termination fires too quickly, episodes will be too short to learn from. Consider raising `box_drop_threshold` to 0.0 (disable it) or lowering it temporarily during the first phase of training.

2. **Initial arm pose**: `DEFAULT_QPOS_CATRA` (`shoulder_pitch=1.5, elbow=1.5`) initializes the reset pose. The robot is free to deviate from this during rollout — the `box_height` reward only cares about where the box is, not how the arms get there. Verify the initial contact geometry with `check_catra.py`.

3. **Box size randomization**: Box size is fixed. Randomizing `model.geom_size` per-environment in MJX requires modifying the randomization pipeline. Only mass is randomized (0.5–3.0 kg) for now.

4. **num_pri count**: If ONNX export fails with a shape error, verify that `num_pri=259` matches the actual privileged observation dimension at runtime (training infers correct network sizes from data; only export is affected).

5. **Per-joint action scale**: The global `action_scale=0.3` is a conservative compromise. A per-joint scale vector (e.g., 0.5 for legs, 0.3 for waist, 0.2 for arms) would allow better tracking across the mixed torque limits without slowing down leg response.

6. **Inspire hand**: The cupped hand is a simplified contact proxy. Adding the Inspire RH56DFX hand MJCF (fingers locked in pre-grip pose) would improve contact accuracy without adding action DOFs.
