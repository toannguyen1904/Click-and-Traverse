# G1 Box Pickup Policy

Standalone manipulation policy for the Unitree G1 humanoid: the robot reaches for a box resting on a support surface, grasps it with both hands, and lifts it off.

This is **Phase 1** of a two-phase curriculum. The pickup policy produces diverse "robot holding box" initial states that serve as warm-start conditions for the traversal phase.

---

## Overview

| Property | Value |
|----------|-------|
| Robot | Unitree G1 humanoid |
| Task | Reach, grasp, and lift a box from a support surface |
| Action space | 11 DOF (waist × 3 + arms × 8) |
| Legs | Held at default pose (not actuated) |
| Episode length | 500 steps (10 s at 50 Hz) |
| Box placement | 0.4 m in front of robot, on a support surface at random height |
| Success criterion | Box lifted ≥ 10 cm above the support surface |

---

## Action Space

The robot controls **11 joints** via PD position targets (delta from current target):

```
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
```

Leg joints (12 DOF) and wrist joints (6 DOF) are **not actuated** — the PD controller holds legs at their default standing pose and wrists at their default neutral pose. Action scale: `0.5`.

---

## Observation Space

### State (61-dim) — deployable on real robot

All sensor readings include realistic noise to match real deployment conditions.

| Field | Dims | Notes |
|-------|------|-------|
| `gyro_pelvis` | 3 | Angular velocity from pelvis IMU `[+ noise]` |
| `gvec_pelvis` | 3 | Gravity direction in pelvis frame `[+ noise]` |
| `joint_angles` | 11 | Waist + arm joint positions (relative to default) `[+ noise]` |
| `joint_vel` | 11 | Waist + arm joint velocities `[+ noise]` |
| `last_action` | 11 | Previous policy output |
| `motor_targets` | 11 | Current PD targets for controlled joints |
| `box_pos_local` | 3 | Box center position in pelvis frame |
| `box_quat_local` | 4 | Box orientation in pelvis frame (wxyz) |
| `box_size` | 3 | Box half-extents (l, w, h) — pre-determined at deployment |
| `surface_z` | 1 | Support surface height — pre-determined at deployment |
| **Total** | **61** | |

### Privileged State (99-dim) — critic only during training

Built from scratch with **noiseless** sensor readings. The critic sees clean versions of all noisy state fields, plus additional privileged quantities not available at deployment.

| Field | Dims | Notes |
|-------|------|-------|
| `gyro_pelvis` | 3 | Noiseless |
| `gvec_pelvis` | 3 | Noiseless |
| `joint_angles` | 11 | Noiseless |
| `joint_vel` | 11 | Noiseless |
| `last_action` | 11 | Same as state |
| `motor_targets` | 11 | Same as state |
| `box_pos_local` | 3 | Same as state |
| `box_quat_local` | 4 | Same as state |
| `box_size` | 3 | Same as state |
| `surface_z` | 1 | Same as state |
| `box_vel_local` | 3 | Box linear velocity in pelvis frame |
| `box_angvel` | 3 | Box angular velocity (world frame) |
| `left_hand_pos` | 3 | Absolute left palm position |
| `right_hand_pos` | 3 | Absolute right palm position |
| `box_pos_world` | 3 | Absolute box center position |
| `pelvis_pos` | 3 | Absolute pelvis site position |
| `torso_pos` | 3 | Absolute torso site position |
| `left_shoulder_pos` | 3 | Absolute left shoulder site position |
| `right_shoulder_pos` | 3 | Absolute right shoulder site position |
| `head_pos` | 3 | Absolute head site position |
| `left_hand_vel` | 3 | Left palm linear velocity |
| `right_hand_vel` | 3 | Right palm linear velocity |
| `kp_scale` | 1 | PD gain DR scalar |
| `kd_scale` | 1 | PD gain DR scalar |
| **Total** | **99** | 61 state (noiseless) + 38 privileged-only |

---

## Reward Design

All rewards are multiplied by `dt` and clipped to `[0, 10000]`.

| Term | Formula | Scale | Purpose |
|------|---------|-------|---------|
| `reach` | `−‖left_palm − box‖ − ‖right_palm − box‖` | 1.0 | Pull hands toward box |
| `lift` | `clip(box_z − surface_z − 0.01 − half_z, 0, 0.1) / 0.1` | 5.0 | Reward liftoff; saturates at +10 cm |
| `table_force` | `box_z − (surface_z + 0.01 + half_z)` | 2.0 | Dense signal bridging reach → lift; negative when resting, 0 at liftoff |
| `hold_stable` | `−‖box_angvel‖` | 0.5 | Penalize box tumbling |
| `box_upright` | `exp(−θ²)` where θ = box tilt angle | 1.0 | Keep box vertical; ensures good handoff state for traversal phase |
| `upright` | `exp(−‖gvec_pelvis[:2]‖²)` | 1.0 | Keep robot upright |
| `energy` | `−‖actuator_force‖²` | 1e-4 | Minimize energy expenditure |
| `smoothness` | `−‖action − last_action‖²` | 1e-3 | Smooth action transitions |
| `joint_limits` | Soft joint limit penalty | 1.0 | Avoid joint limit violations |

**Notes:**
- `table_force` uses an unclipped height proxy rather than direct contact force reading. This provides a continuous gradient even before the box leaves the surface (hands gradually bear more weight).
- `box_upright`: θ is computed from box quaternion as `arccos(1 − 2(qx² + qy²))`, the angle between the box z-axis and world z-axis.

---

## Termination Conditions

| Condition | Threshold |
|-----------|-----------|
| Robot fall (gravity vector) | `gvec_z < 0` |
| Robot fall (head height) | `head_z < 0.5 m` |
| Box dropped to floor | `box_z < surface_z − 0.1 m` |
| NaN in qpos or qvel | any |
| Episode timeout | 500 steps (10 s) |

---

## Randomization

### Per-episode (in `reset`)

| Property | Range | Notes |
|----------|-------|-------|
| Robot spawn XY | `U[−1, 1]` m offset | Flat terrain |
| Robot initial yaw | `U[−90°, 90°]` | Facing generally toward ±X |
| Robot joint init | `U[0.5, 1.5] × default`, clipped to soft limits | Covers a range of initial arm poses |
| Box yaw offset | `U[−10°, 10°]` | Relative to robot forward direction |
| Support surface height | `U[0.4, 0.8]` m | Center z of the support platform |

### Per-environment (domain randomization via `domain_randomize_pickup`)

| Property | Range | Notes |
|----------|-------|-------|
| Joint frictionloss | `U[0.9, 1.1] × nominal` | Robot joints only |
| Joint armature | `U[1.0, 1.05] × nominal` | Robot joints only |
| Torso CoM offset | `U[−0.1, 0.1]` m per axis | Simulates mass distribution uncertainty |
| All body masses | `U[0.9, 1.1] × nominal` | Global scale |
| Torso mass perturbation | `U[−1, 1]` kg additive | Additional torso mass noise |
| qpos0 perturbation | `U[−0.05, 0.05]` per joint | Shifts nominal pose |
| Box half-size x | `U[0.10, 0.20]` m | Per-environment |
| Box half-size y | `U[0.10, 0.25]` m | Per-environment |
| Box half-size z | `U[0.10, 0.20]` m | Per-environment |
| Box mass | `U[0.5, 3.0]` kg | Per-environment |
| KP scale | `U[0.75, 1.25]` | PD gain randomization |
| KD scale | `U[0.75, 1.25]` | PD gain randomization |
| RFI | Disabled | Legs not actively controlled; perturbations risk toppling |

---

## Architecture

### Class Hierarchy

```
MjxEnv (mujoco_playground)
  └─ G1Env
       └─ G1LocoEnv
            └─ G1CatEnv
                 └─ G1CaTraEnv
                      └─ G1PickupEnv   ← this task
```

`G1PickupEnv` overrides `reset`, `step`, `_get_obs`, `_get_reward`, and `_get_termination`. It reuses the CaTra scene XML (box freejoint + mocap support surface) but disables gait clock, push forces, and command tracking.

### Key Implementation Details

- **Leg locking**: Legs are excluded from `action_joint_ids`. The PD controller still targets them at `_default_qpos` values, holding them stationary without active stepping.
- **Box freejoint**: Adding the box freejoint extends qpos from 36 → 43 and qvel from 35 → 41. `torque_step_catra` slices only `[7:36]` / `[6:35]` to avoid shape mismatch.
- **Box size in DR**: `domain_randomize_pickup` modifies `model.geom_size[box_geom_id]` per-environment via JAX vmap, so each parallel environment sees a different box.
- **Noisy vs noiseless obs**: State uses per-step noise injection on gyro, gravity, joint_angles, joint_vel. Privileged state is built independently with raw sensor values.

### Files

| File | Purpose |
|------|---------|
| [cat_ppo/envs/g1/env_pickup.py](cat_ppo/envs/g1/env_pickup.py) | Main environment, config, DR function |
| [cat_ppo/envs/g1/play_pickup.py](cat_ppo/envs/g1/play_pickup.py) | CPU inference env for ONNX playback |
| [train_ppo_pickup.py](train_ppo_pickup.py) | Training entry point |
| [check_pickup.py](check_pickup.py) | Episode visualization script |
| [data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml) | Scene XML (shared with CaTra) |

---

## Running

### Setup

```bash
source .venv/bin/activate && source .env
python -m cat_ppo.utils.mj_playground_init
```

### Visualize an Episode

```bash
python check_pickup.py                    # random surface height
python check_pickup.py --surface_z 0.6   # fixed surface height at 0.6 m
```

### Train

```bash
python train_ppo_pickup.py \
    --task G1Pickup \
    --exp_name pickup_v1
```

### Export to ONNX

```bash
python -m cat_ppo.eval.brax2onnx --task G1Pickup --exp_name <full_exp_name>
```

### Play in MuJoCo Viewer

```bash
python -m cat_ppo.eval.mj_onnx_play --task G1Pickup --exp_name <full_exp_name>
```
