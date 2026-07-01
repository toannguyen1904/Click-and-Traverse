# G1 Box Pickup Policy

Standalone manipulation policy for the Unitree G1 humanoid: the robot reaches for a box resting on a support surface, grasps it with both hands, and lifts it off.

This is **Phase 1** of a two-phase curriculum. The pickup policy produces diverse "robot holding box" initial states that serve as warm-start conditions for the traversal phase.

---

## Overview

| Property | Value |
|----------|-------|
| Robot | Unitree G1 humanoid |
| Task | Reach, grasp, and lift a box from a support surface |
| Action space | 20 DOF (all leg joints × 2 + arms × 8; waist joints held at default) |
| Legs | All 12 leg joints actuated (hip pitch/roll/yaw, knee, ankle pitch/roll × 2) |
| Episode length | 1000 steps (20 s at 50 Hz) |
| Box placement | 0.3 m in front of robot, on a support pillar (0.4 × 0.5 m top, 0.6 m tall) |
| Success criterion | Box lifted ≥ 10 cm above the support pillar top |

---

## Action Space

The robot controls **20 joints** via PD position targets (delta from current target):

```
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
```

Waist (yaw/roll/pitch) and wrist joints are **not actuated** — the PD controller holds them at their default pose. Action scale: `0.2`.

---

## Observation Space

### State (97-dim) — deployable on real robot

All sensor readings include realistic noise to match real deployment conditions.

| Field | Dims | Notes |
|-------|------|-------|
| `gyro_pelvis` | 3 | Angular velocity from pelvis IMU `[+ noise]` |
| `gvec_pelvis` | 3 | Gravity direction in pelvis frame `[+ noise]` |
| `joint_angles` | 20 | Controlled joint positions (legs + arms, relative to default) `[+ noise]` |
| `joint_vel` | 20 | Controlled joint velocities `[+ noise]` |
| `last_action` | 20 | Previous policy output |
| `motor_targets` | 20 | Current PD targets for controlled joints |
| `box_pos_local` | 3 | Box center position in pelvis frame `[+ noise: ±5 cm per axis]` |
| `box_quat_local` | 4 | Box orientation in pelvis frame (wxyz) `[+ noise: ±5° random axis-angle]` |
| `box_size` | 3 | Box half-extents (l, w, h) — pre-determined at deployment |
| `box_mass` | 1 | Box mass (DR'd per environment, U[0.5, 4.0] kg) — pre-determined at deployment |
| **Total** | **97** | |

### Privileged State (135-dim) — critic only during training

Built from scratch with **noiseless** sensor readings. The critic sees clean versions of all noisy state fields, plus additional privileged quantities not available at deployment.

| Field | Dims | Notes |
|-------|------|-------|
| `gyro_pelvis` | 3 | Noiseless |
| `gvec_pelvis` | 3 | Noiseless |
| `joint_angles` | 20 | Noiseless |
| `joint_vel` | 20 | Noiseless |
| `last_action` | 20 | Same as state |
| `motor_targets` | 20 | Same as state |
| `box_pos_local` | 3 | Same as state |
| `box_quat_local` | 4 | Same as state |
| `box_size` | 3 | Same as state |
| `box_mass` | 1 | Box mass (DR'd per environment, U[0.5, 4.0] kg); also in deployable state |
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
| **Total** | **135** | 97 state (noiseless) + 38 privileged-only |

---

## Reward Design

All rewards are multiplied by `dt` and clipped to `[0, 10000]`.

| Term | Formula | Scale | Purpose |
|------|---------|-------|---------|
| `reach` | `−‖left_palm − left_face‖ − ‖right_palm − right_face‖` | 3.0 | Pull each hand to its own box face; zero when both palms touch their respective sides |
| `lift` | `clip(h, 0, 0.10) / 0.10 − clip(h − 0.10, 0, ∞)` where `h = box_z − surface_z − support_half_z − half_z` | 2.0 | Ramps to 1.0 at +10 cm; penalizes lifting beyond 10 cm |
| `hand_contact` | `0.5 · (left_touching_box + right_touching_box)` | 2.0 | 0.5 per hand in contact with box; 1.0 for bilateral grasp |
| `box_pillar_contact` | `1` if box touching pillar, else `0` | -1.5 | Penalize box remaining on pillar; encourages liftoff |
| `grasp_symmetry` | `(Δheight)² + (Δdepth)²` along box local Z/X axes | -2.0 | Penalize height and front-back asymmetry between the two palms |
| `palm_orient` | `0.5 · (clip(left_normal · left_desired, 0, 1) + clip(right_normal · right_desired, 0, 1))` | 2.0 | Reward palm normals (site local +Y for left, -Y for right — mirrored frames) pointing toward their respective box face; 1.0 when both palms face squarely inward |
| `hands_level` | `(left_z - right_z)^2 / (‖left_palm - right_palm‖^2 + 1e-6)` | -1.0 | Penalize the line connecting the two palms tilting out of the world XY plane; 0 when both hands are at the same height, 1 when the hand-to-hand vector is vertical |
| `hold_stable` | `−‖box_linvel‖ − ‖box_angvel‖` | 0.5 | Penalize box translation and tumbling |
| `box_yaw_stable` | `wrap(yaw_now − yaw_init)²`, gated on both hands touching box | 0.0 | Penalize box yaw deviation from its initial orientation during active bilateral grasp; **currently disabled** |
| `box_centering` | `(dot(box_pos − torso_pos, torso_right))²` | 0.0 | Penalize box being laterally offset from the torso's forward axis; **currently disabled** |
| `box_vertical` | `‖box_xy − box_xy_init‖²`, gated on both hands touching box | -0.5 | Penalize XY drift from pillar center during active bilateral grasp; box should rise straight up |
| `box_upright` | `exp(−θ²)` where θ = box tilt angle, gated on `lift_height > 0` | 2.0 | Keep box vertical once lifted; zero while resting on pillar to avoid free reward |
| `upright` | `exp(−0.5(‖roll‖ + \|pitch_back\| + \|pitch\|)) − \|pitch_back\|` | 3.0 | Asymmetric: backward lean is double-penalised; matches CAT orientation reward |
| `base_height` | target pelvis height 0.75 m; capped when above target | 1.0 | Encourage upright stance height while reaching |
| `foot_contact` | stance contact mismatch cost | -0.5 | Encourage both feet to stay in contact with the floor |
| `foot_slip` | stance foot speed squared | -0.1 | Discourage shuffling/sliding while crouching and lifting |
| `foot_balance` | `‖(left_foot + right_foot − 2·pelvis_com)_xy‖² · (1 + spread_penalty)` | -30.0 | Penalise pelvis COM off-center and feet closer than 0.35 m |
| `straight_knee` | `sum(max(0, 0.1 − knee_angle))` | -5.0 | Discourage fully locked knees; allow a slight crouch |
| `joint_torque` | `sum(actuator_force²)` | -1e-4 | Penalize high motor torque, matching CAT |
| `smoothness_joint` | `sum(0.01·qvel² + qacc²)` | -1e-6 | Discourage jerky joint motion across control steps |
| `smoothness` | `−‖action − last_action‖²` | 1e-3 | Smooth action transitions |
| `joint_limits` | Soft joint limit penalty | -1.0 | Penalize joint limit violations |

**Notes:**
- `reach` targets each hand to its respective face: `left_face = box_pos + box_left_axis·half_y`, `right_face = box_pos − box_left_axis·half_y`, where `box_left_axis = rotate([0,1,0], box_quat)` is the box local +Y in world frame — which points to the robot's **left** in this setup. This prevents the crossed-hands failure mode.
- `hold_stable` scale is 0.5; contact forces naturally cause small box rotations even at rest, so this term is kept light.
- `box_upright`: θ is computed from box quaternion as `arccos(1 − 2(qx² + qy²))`, the angle between the box z-axis and world z-axis.
- `upright` uses CAT's asymmetric formula: `‖roll‖ = |roll_pelvis| + |roll_torso|`; `pitch_back = clip(torso_pitch, −π, 0)` (backward lean only); backward lean enters the exponential twice and also subtracts linearly, making the robot significantly more reluctant to fall backward.
- `base_height` calls `_reward_base_height(qpos[2], move_flag=0)`: rewards reaching the 0.75 m pelvis target; output is capped at 0.5 when above target (no bonus for extra height).
- `hands_level` is a normalized, scale-invariant tilt penalty on the hand-to-hand vector: it depends only on orientation in the world frame, not on how far apart the hands are.
- `foot_balance` works in world XY without a navigation frame: `foot_center = (left_foot + right_foot − 2·pelvis_com)` measures COM centering; `spread_penalty = max(0, (0.35 − foot_dist) · 10)` penalises feet closer than 0.35 m.
- `foot_contact` and `foot_slip` treat pickup as an always-stance task: both feet are expected to remain planted on the floor throughout the episode.
- `straight_knee` is borrowed from CAT and penalizes knee angles below `0.1 rad`, encouraging a slightly bent, compliant stance instead of locked knees.
- `smoothness_joint` is borrowed from CAT and penalizes joint velocity and finite-difference acceleration using the previous control step's joint velocities.

---

## Termination Conditions

| Condition | Threshold |
|-----------|-----------|
| Robot fall (gravity vector) | `gvec_z < 0` |
| Robot fall (head height) | `head_z < 0.5 m` |
| Box dropped | `box_z < box_z_init − 0.1 m` (0.1 m below its reset height) |
| NaN in qpos or qvel | any |
| Episode timeout | 1000 steps (20 s) |

---

## Randomization

### Per-episode (in `reset`)

| Property | Range | Notes |
|----------|-------|-------|
| Robot spawn XY | `U[−1, 1]` m offset | Flat terrain |
| Robot initial yaw | `U[−90°, 90°]` | Facing generally toward ±X |
| Robot joint init | `U[0.5, 1.5] × default`, clipped to soft limits | Covers a range of initial arm poses |
| Box XY position | 0.3 m in front of robot along its forward direction | Deterministic; robot XY/yaw spawn adds implicit variety |
| Box yaw offset | `U[−10°, 10°]` | Relative to robot forward direction |
| Support pillar top height | Fixed at 0.6 m (body-center z = 0.3 m) | Pillar extends from floor; height randomization planned for curriculum later |
| Support pillar yaw | Same as box yaw | Rectangular pillar (0.4 × 0.5 m) faces same direction as box |

### Per-environment (domain randomization via `domain_randomize_pickup`)

| Property | Range | Notes |
|----------|-------|-------|
| Joint frictionloss | `U[0.9, 1.1] × nominal` | Robot joints only |
| Joint armature | `U[1.0, 1.05] × nominal` | Robot joints only |
| Torso CoM offset | `U[−0.1, 0.1]` m per axis | Simulates mass distribution uncertainty |
| All body masses | `U[0.9, 1.1] × nominal` | Global scale |
| Torso mass perturbation | `U[−1, 1]` kg additive | Additional torso mass noise |
| qpos0 perturbation | `U[−0.05, 0.05]` per joint | Shifts nominal pose |
| Box half-size x | `U[0.10, 0.15]` m | Per-environment |
| Box half-size y | `U[0.10, 0.20]` m | Per-environment |
| Box half-size z | `U[0.10, 0.15]` m | Per-environment; reset places the box at `pillar_top + box_half_z` so it rests exactly on the pillar top |
| Box mass | `U[0.5, 3.0]` kg | Per-environment; included in the deployable obs |
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

`G1PickupEnv` overrides `reset`, `step`, `_get_obs`, `_get_reward`, and `_get_termination`. It reuses the CaTra scene XML (box freejoint + support pillar freejoint) but disables gait clock, push forces, and command tracking. The training scene uses `condim=3` contacts: `left_hand_collision ↔ box_geom` and `right_hand_collision ↔ box_geom` explicit pairs so hands physically interact with the box, and box↔pillar contact via `contype/conaffinity` bit masking so MJX broadphase handles it correctly (explicit pairs with mocap bodies are not processed by MJX).

### Key Implementation Details

- **Full leg control**: All 12 leg joints are included in `action_joint_ids`. Wrist joints still receive `_default_qpos` targets each step.
- **Wrist mass lumping**: `left/right_wrist_yaw_link` inertial mass is set to 0.8 kg (vs. 0.254 kg without hand). The rubber hand meshes are visual-only (no separate body/inertial), so the ~0.54 kg hand mass is lumped into the wrist link to keep contact dynamics physically grounded.
- **qpos/qvel layout**: Two freejoints extend the state beyond the 36-dim robot qpos:
  ```
  qpos: [0:7] root | [7:36] robot joints (29) | [36:43] box freejoint | [43:50] support freejoint
  qvel: [0:6] root | [6:35] robot joints (29) | [35:41] box vel       | [41:47] support vel
  ```
  `torque_step_catra` slices only `[7:36]` / `[6:35]` so the extra DOFs don't interfere.
- **Support pillar**: The support body has a freejoint (not mocap) so MJX contact detection works. Its position and yaw are set in `reset()` via `qpos[43:50]`; yaw matches the box so the rectangular face (0.4 × 0.5 m, half-extents 0.2 × 0.25) is aligned with the box sides. With `mass=1000 kg` it is effectively immovable. Contact bit scheme: pillar `contype=6/conaffinity=6` (bits 1+2), box `contype=3/conaffinity=3` (bits 0+1), floor `contype=5/conaffinity=5` (bits 0+2) — pillar sits on floor (bit 2) and supports the box (bit 1) without colliding with the robot (bit 0 only).
- **Box size in DR**: `domain_randomize_pickup` modifies `model.geom_size[box_geom_id]` per-environment via JAX vmap, so each parallel environment sees a different box.
- **Noisy vs noiseless obs**: State uses per-step noise injection on gyro, gravity, joint_angles, joint_vel, **and the box pose** (`box_pos_local` ±5 cm per axis, `box_quat_local` ±5° random axis-angle — mimicking imperfect box tracking, e.g. via vision, at deployment). Magnitudes are `noise_config.scales.box_pos` / `box_ori`, gated by `noise_config.level`. Privileged state is built independently with raw sensor values, so the critic keeps the clean ground-truth box pose. Applied in `pickup_obs_from_data` (the canonical obs used by training and warm-start generation) and mirrored in `play_pickup.py` for ONNX playback.

### Files

| File | Purpose |
|------|---------|
| [cat_ppo/envs/g1/env_pickup.py](cat_ppo/envs/g1/env_pickup.py) | Main environment, config, DR function |
| [cat_ppo/envs/g1/play_pickup.py](cat_ppo/envs/g1/play_pickup.py) | CPU inference env for ONNX playback |
| [train_ppo_pickup.py](train_ppo_pickup.py) | Training entry point |
| [check_pickup.py](check_pickup.py) | CPU-based visualization (static initial state via play env) |
| [check_pickup_mjx.py](check_pickup_mjx.py) | MJX-based live physics visualization (zero-action rollout) |
| [data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml) | Scene XML used by MJX training env |
| [data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml](data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml) | Scene XML used by CPU play/eval env |

---

## Running

### Setup

```bash
source .venv/bin/activate && source .env
python -m cat_ppo.utils.mj_playground_init
```

### Visualize an Episode

```bash
# CPU-based (static initial pose, no physics stepping):
python check_pickup.py

# MJX-based (live physics, zero-action rollout):
python check_pickup_mjx.py
python check_pickup_mjx.py --task G1Stand
python check_pickup_mjx.py --seed 5
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
