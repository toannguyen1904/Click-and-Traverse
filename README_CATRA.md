# G1 Carry and Traverse Policy (CaTra)

Two-stage end-to-end task for the Unitree G1 humanoid: the robot first reaches for a box on a support pillar and lifts it (Stage 1), then walks through a cluttered obstacle course while carrying the box (Stage 2).

This is **Phase 2** of a two-phase curriculum. It combines the pickup policy (Phase 1) with the collision-aware traversal policy (CAT, [arXiv:2601.16035](https://arxiv.org/abs/2601.16035)) into a single end-to-end task.

---

## Overview

| Property | Value |
|----------|-------|
| Robot | Unitree G1 humanoid |
| Task | Pick up a box from a pillar, then carry it through obstacles |
| Action space | 23 DOF (12 legs + 3 waist + 8 arms) |
| Episode length | 1100 steps (22 s at 50 Hz) |
| Stage 1 | Steps 0–99 (2 s): stand-and-reach, pickup reward set |
| Stage 2 | Steps 100–1099 (20 s): PF-guided traversal + grasp-maintenance rewards |
| Box placement | 0.3 m in front of robot, on a support pillar (surface height = 0.3 m) |
| Success criterion | Box lifted ≥ 10 cm in Stage 1 AND box still held + robot reaches PF target by Stage 2 end |

---

## Action Space

Same 23-joint PD position target space as G1Pickup:

```
left_hip_pitch_joint     left_hip_roll_joint      left_hip_yaw_joint
left_knee_joint          left_ankle_pitch_joint   left_ankle_roll_joint
right_hip_pitch_joint    right_hip_roll_joint     right_hip_yaw_joint
right_knee_joint         right_ankle_pitch_joint  right_ankle_roll_joint
waist_yaw_joint          waist_roll_joint         waist_pitch_joint
left_shoulder_pitch_joint  left_shoulder_roll_joint  left_shoulder_yaw_joint
left_elbow_joint
right_shoulder_pitch_joint right_shoulder_roll_joint right_shoulder_yaw_joint
right_elbow_joint
```

Wrist joints are not actuated. Action scale: `0.5`.

---

## Stage Schedule

| Stage | Steps | Duration | Command | Reward focus |
|-------|-------|----------|---------|-------------|
| Stage 1 — Pickup | 0–99 | 2 s | `[0, 0, 0, 0]` (stationary) | All 22 G1Pickup reward terms |
| Stage 2 — Carry & Traverse | 100–1099 | 20 s | PF-derived `[move_flag, vx, vy, yaw]` | CAT navigation rewards + 6 `_carry` grasp-maintenance terms |

The stage transition is a hard cut: `info["step"] == 100`. The policy receives a **stage flag** (0.0 in Stage 1, 1.0 in Stage 2) as part of its observation so it can learn stage-specific behaviors from a single policy.

Command in Stage 1 is always `[0, 0, 0, 0]`. In Stage 2 the command is derived every step from PF (potential field) grid data via `compute_cmd_from_rtf`: a 4-dim vector `[move_flag, vx, vy, yaw]` in the navigation frame.

---

## Observation Space

### State (195-dim) — deployable on real robot

| Field | Dims | Notes |
|-------|------|-------|
| `gyro_pelvis` | 3 | Angular velocity from pelvis IMU `[+ noise]` |
| `gvec_pelvis` | 3 | Gravity direction in pelvis frame `[+ noise]` |
| `joint_angles` | 23 | Controlled joints, relative to default `[+ noise]` |
| `joint_vel` | 23 | Controlled joint velocities `[+ noise]` |
| `last_action` | 23 | Previous policy output |
| `motor_targets` | 23 | Current PD targets for controlled joints |
| `command` | 4 | Navigation command `[move_flag, vx, vy, yaw]`; zeros in Stage 1 |
| `foot_height` | 1 | Target foot height for gait |
| `gait_phase` | 4 | cos+sin of 2D gait clock (left + right) |
| `body_pf` | 77 | Navigation-frame PF fields for 11 body sites: head, pelvis, torso (×1 each), feet, hands, knees, shoulders (×2 each); gf(3)+bf(3)+df(1)=7 dims per site |
| `box_pos_local` | 3 | Box center position in pelvis frame |
| `box_quat_local` | 4 | Box orientation in pelvis frame (wxyz) |
| `box_size` | 3 | Box half-extents (l, w, h) |
| `stage_flag` | 1 | 0.0 in Stage 1, 1.0 in Stage 2 |
| **Total** | **195** | |

### Privileged State (289-dim) — critic only during training

Noiseless version of the state block (without `box_pos_local`/`box_quat_local`) plus privileged extras:

| Field | Dims | Notes |
|-------|------|-------|
| Noiseless state block | 188 | Same structure as state but noiseless; box_pos/quat_local omitted (world-frame used instead) |
| `linvel_pelvis` | 3 | Pelvis linear velocity (world frame) |
| `pelvis_pos` + `torso_pos` + `head_pos` | 9 | Absolute body positions |
| `shlds_pos` + `hands_pos` + `knees_pos` + `feet_pos` | 24 | 2 sites × 4 body groups × 3 |
| `head_vel` + `hands_vel` + `feet_vel` | 15 | Linear velocities for 5 sites (1+2+2) |
| `box_pos_world` | 3 | Box center in world frame |
| `box_quat_world` | 4 | Box orientation in world frame (wxyz) |
| `box_linvel_world` | 3 | Box linear velocity in world frame (finite-diff) |
| `box_angvel_world` | 3 | Box angular velocity in world frame (finite-diff) |
| `navi_torso_rpy[:2]` | 2 | Torso roll + pitch in navigation frame |
| `gait_mask` | 2 | Per-foot contact-based gait mask (left, right) |
| `feet_contact` | 2 | Binary foot contact flags: `[left_touching_floor, right_touching_floor]` |
| `rfi_lim_scale` | 29 | Per-joint random force injection scale |
| `kp_scale` | 1 | PD gain DR scalar |
| `kd_scale` | 1 | PD gain DR scalar |
| **Total** | **289** | |

---

## Reward Design

All rewards are multiplied by `dt`.

### Always-Active Terms (applied in both stages)

| Term | Scale | Purpose |
|------|-------|---------|
| `joint_torque` | -1e-4 | Penalize high motor torque |
| `smoothness_joint` | -1e-6 | Penalize jerky joint motion |
| `joint_limits` | -1.0 | Penalize joint limit violations |

### Stage 1 — Pickup Rewards (active when step < 100)

| Term | Scale | Formula |
|------|-------|---------|
| `reach` | 1.5 | Distance from each palm to its target box face |
| `lift` | 2.0 | Box height above pillar, capped at +10 cm |
| `hand_contact` | 2.0 | 0.5 per hand in contact with box |
| `box_pillar_contact` | -1.5 | Penalize box still resting on pillar |
| `grasp_symmetry` | -2.0 | Penalize height/depth asymmetry between two palms |
| `palm_orient` | 2.0 | Reward palm normals facing inward toward box |
| `hands_level` | -1.0 | Penalize hand-to-hand tilt out of horizontal plane |
| `hold_stable` | 0.0 | Box linvel + angvel penalty (scaled off) |
| `box_yaw_stable` | 0.0 | Box yaw drift penalty (scaled off) |
| `box_centering` | 0.0 | Lateral offset penalty (scaled off) |
| `box_vertical` | -0.5 | Penalize XY drift from pillar center during grasp |
| `box_upright` | 0.0 | Keep box vertical once lifted (scaled off) |
| `upright` | 3.0 | Asymmetric pitch/roll penalty on robot posture |
| `foot_contact` | -0.5 | Both feet must stay planted |
| `foot_slip` | -0.1 | Penalize foot sliding |
| `straight_knee` | -5.0 | Discourage locked knees |
| `smoothness` | 1e-3 | Smooth action transitions |
| `base_height` | 1.0 | Target pelvis height 0.75 m |
| `foot_balance` | -30.0 | Penalize COM centering and foot spread |

### Stage 2 — Traversal + Carry Rewards (active when step ≥ 100)

Navigation rewards (from G1CatEnv):

| Term | Scale | Purpose |
|------|-------|---------|
| `tracking_orientation` | 2.0 | Match commanded yaw orientation |
| `tracking_root_field` | 1.0 | Follow PF velocity command |
| `body_motion` | -0.5 | Penalize undesired body translation |
| `body_rotation` | 1.0 | Reward upright body alignment |
| `foot_contact_trav` | -1.0 | Gait-consistent foot contact |
| `foot_clearance` | -15.0 | Penalize foot scuffing |
| `foot_slip_trav` | -0.5 | Penalize stance foot slip |
| `foot_balance_trav` | -30.0 | Foot/COM balance |
| `foot_far` | 0.0 | Foot overstep penalty (scaled off) |
| `straight_knee_trav` | -30.0 | Discourage locked knees during locomotion |
| `smoothness_action` | -1e-3 | Smooth action transitions |
| `forward_progress` | 5.0 | Linear reward for velocity in the command direction; `clip(v·cmd_dir, 0, |cmd|)` — nonzero gradient from a dead stop, unlike the exp-based `tracking_root_field` |
| `headgf/handsgf/feetgf` | 0.0 | Body goal field tracking (scaled off) |
| `headdf/handsdf/feetdf/kneesdf/shldsdf` | 0.0 | Body distance field penalties (scaled off) |

Grasp-maintenance rewards (`_carry` suffix, same formulas as Stage 1):

| Term | Scale | Purpose |
|------|-------|---------|
| `reach_carry` | 0.75 | Keep palms close to box faces |
| `lift_carry` | 1.0 | Keep box elevated |
| `hand_contact_carry` | 1.0 | Maintain bilateral contact |
| `grasp_symmetry_carry` | -1.0 | Maintain symmetric grasp |
| `palm_orient_carry` | 1.0 | Maintain correct palm orientation |
| `hands_level_carry` | -0.5 | Maintain level hands |

---

## Termination Conditions

| Condition | Threshold |
|-----------|-----------|
| Robot fall (gravity vector) | `gvec_z < 0` |
| Robot fall (head height) | `head_z < 0.7 m` |
| Box dropped | `box_z < 0.3 m` — active throughout full episode |
| Body-obstacle SDF collision | any of head/torso/pelvis/feet/hands/knees/shoulders df < −4 cm (active after step 150) |
| NaN in qpos or qvel | any |
| Episode timeout | 1100 steps (22 s) |

---

## Randomization

### Per-episode (in `reset`)

| Property | Range | Notes |
|----------|-------|-------|
| Robot initial yaw | `U[−90°, 90°]` | Facing generally toward ±X |
| Robot joint init | `U[0.5, 1.5] × default`, clipped to soft limits | |
| Box XY position | 0.3 m forward from robot pelvis | Robot yaw/XY spawn provides variety |
| Box yaw offset | `U[−10°, 10°]` | Relative to robot forward direction |
| Support pillar height | Fixed at surface_z = 0.3 m | Box-center z at surface_z + box_half_z |

### Per-environment (domain randomization via `domain_randomize_catra`)

| Property | Range | Notes |
|----------|-------|-------|
| Joint frictionloss | `U[0.9, 1.1] × nominal` | Robot joints |
| Joint armature | `U[1.0, 1.05] × nominal` | Robot joints |
| Torso CoM offset | `U[−0.1, 0.1]` m per axis | |
| All body masses | `U[0.9, 1.1] × nominal` | |
| Torso mass perturbation | `U[−1, 1]` kg additive | |
| qpos0 perturbation | `U[−0.05, 0.05]` per joint | |
| Box half-size x | `U[0.10, 0.15]` m | Per-environment |
| Box half-size y | `U[0.10, 0.20]` m | Per-environment |
| Box half-size z | `U[0.10, 0.15]` m | Per-environment |
| Box mass | `U[1.0, 3.0]` kg | Per-environment (heavier than Pickup) |
| KP scale | `U[0.75, 1.25]` | |
| KD scale | `U[0.75, 1.25]` | |
| RFI (per-joint torque noise) | Enabled | Applied every substep in both stages; simulates actuator noise |
| Push impulses | Enabled | Random root velocity impulses; gated to step ≥ 100 (Stage 2 only) |

---

## Architecture

### Class Hierarchy

```
MjxEnv (mujoco_playground)
  └─ G1Env
       └─ G1LocoEnv
            └─ G1CatEnv
                 └─ G1CaTraEnv   ← this task
                      └─ G1PickupEnv
```

`G1CaTraEnv` overrides `reset`, `step`, `_get_obs`, `_get_reward`, and `_get_termination`. It inherits the HumanoidPF fields and navigation reward infrastructure from `G1CatEnv`, and incorporates the full G1Pickup reward set for Stage 1.

### Key Implementation Details

- **Stage flag in obs**: The scalar `0.0` (Stage 1) or `1.0` (Stage 2) appended as the last element of the state obs. Lets the policy learn stage-specific behavior from a single network.
- **JIT-friendly reward gating**: Both stage reward dicts are computed every step; `jp.where(step < 100, ...)` gates which set contributes to the return. Dict shape is static — no dynamic branching.
- **Push forces gated**: Random push perturbations (RFI) are enabled but suppressed during Stage 1 (`step < 100`). Full pushes activate in Stage 2 when locomotion is expected.
- **Box PF removed**: The old CaTra observation included box goal-field / boundary-field / distance-field (7 dims). These are removed; the policy instead gets `box_pos_local`, `box_quat_local`, and `box_size` directly.
- **Box drop threshold**: Termination fires when `box_z < 0.3 m` (at or below pillar surface), allowing the box to move freely at any height above that during carries.
- **SDF termination gating**: Body-obstacle collision termination is suppressed until step 150 (50 steps into Stage 2), giving the robot time to stabilize its carry before collision penalties apply.
- **qpos/qvel layout**:
  ```
  qpos: [0:7] root | [7:36] robot joints (29) | [36:43] box freejoint | [43:50] support freejoint
  qvel: [0:6] root | [6:35] robot joints (29) | [35:41] box vel       | [41:47] support vel
  ```

### Files

| File | Purpose |
|------|---------|
| [cat_ppo/envs/g1/env_catra.py](cat_ppo/envs/g1/env_catra.py) | Main training environment, config, DR function |
| [cat_ppo/envs/g1/play_catra.py](cat_ppo/envs/g1/play_catra.py) | CPU inference env for ONNX playback |
| [train_ppo_catra.py](train_ppo_catra.py) | Training entry point |
| [check_catra.py](check_catra.py) | CPU-based visualization (static initial state) |
| [data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml) | MJX training scene |
| [data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml) | CPU play/eval scene |

---

## Running

### Setup

```bash
source .venv/bin/activate && source .env
python -m cat_ppo.utils.mj_playground_init
```

### Visualize an Episode

```bash
# Static initial state (CPU, no physics stepping):
python check_catra.py
python check_catra.py --obs_path data/assets/TypiObs/narrow1
```

### Train

Without obstacles (flat empty scene, no body-PF reward/penalty):
```bash
python train_ppo_catra.py \
    --task G1CaTra \
    --exp_name catra_v1
```

With obstacles (enable body-PF guidance + SDF penalties, select a scene):
```bash
python train_ppo_catra.py \
    --task G1CaTra \
    --exp_name catra_v1 \
    --obs_path data/assets/TypiObs/bar0 \
    --ground 1.0 \
    --lateral 1.0 \
    --overhead 1.0
```

`--ground` scales `feetgf`/`feetdf`, `--lateral` scales `handsgf`/`handsdf`/`kneesdf`/`shldsdf`, `--overhead` scales `headgf`/`headdf`. All default to 0 (disabled). These are only meaningful in Stage 2.

### Export to ONNX

```bash
python -m cat_ppo.eval.brax2onnx --task G1CaTra --exp_name <full_exp_name>
```

### Play in MuJoCo Viewer

```bash
python -m cat_ppo.eval.mj_onnx_play --task G1CaTra --exp_name <full_exp_name>
```

### Smoke Test (stage transition verification)

```python
import jax, cat_ppo
env_cls = cat_ppo.registry.get("G1CaTra", "env_class")
cfg = cat_ppo.registry.get("G1CaTra", "config")
env = env_cls(cfg.env_config)
key = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(key)
step = jax.jit(env.step)
for i in range(110):
    state = step(state, jax.numpy.zeros(23))
    if i in (98, 99, 100, 101):
        print(f"step {i}: command={state.info['command']}, reach={state.metrics.get('reach', 0):.4f}, reach_carry={state.metrics.get('reach_carry', 0):.4f}")
```
