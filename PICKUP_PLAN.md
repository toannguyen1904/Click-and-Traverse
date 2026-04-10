# Plan: G1 Box Pickup Training Environment

## Context

The long-term goal is to train a CaTra (Carry and Traverse) policy where a G1 humanoid carries a box through cluttered scenes. Training this end-to-end is hard because the robot must simultaneously learn to pick up the box AND walk. The strategy is to decompose this into two phases:

1. **Pickup policy** (this task): Robot stands in place, reaches for a nearby box on a surface, grasps it, and lifts it off the surface. This produces diverse "robot holding box" states.
2. **Traversal policy** (future): Initialized from pickup policy terminal states, the robot walks through obstacles while carrying the box.

This plan covers Phase 1: a standalone pickup environment and training pipeline.

---

## Design

### Task Overview
- Robot stands stationary (no stepping), uses waist + arms to reach and lift a box from a support surface
- Robot yaw randomized in [-90°, 90°]; box placed 0.4m in front of robot
- Box orientation offset: [-10°, 10°] yaw relative to robot forward
- Box size, mass, and surface height randomized per episode
- Success = box lifted ≥10cm above the support surface
- Episode length: 500 steps (10s)

### Action Space: 11 DOF (waist + arms only)
```
waist_yaw_joint, waist_roll_joint, waist_pitch_joint,           # 3 waist
left_shoulder_pitch_joint, left_shoulder_roll_joint,             # 8 arm
left_shoulder_yaw_joint, left_elbow_joint,
right_shoulder_pitch_joint, right_shoulder_roll_joint,
right_shoulder_yaw_joint, right_elbow_joint
```
Legs are held at default pose (not actuated). This simplifies the learning problem since the robot doesn't need to learn balance-while-stepping — just upper body manipulation.

### Observation Space (state — deployable)
```
gyro_pelvis (3)           [+ noise]      # angular velocity
gvec_pelvis (3)           [+ noise]      # gravity direction
joint_angles (11)         [+ noise]      # waist+arms only
joint_vel (11)            [+ noise]      # waist+arms only
last_action (11)                         # previous action
motor_targets (11)                       # current PD targets
box_pos_local (3)                        # box position in robot frame
box_quat_local (4)                       # box orientation in robot frame
box_size (3)                             # half-extents (l, w, h) — known a priori in deployment
surface_z (1)                            # support surface height
───────────────────────────────────
Total: ~61 dims
```

### Observation Space (privileged_state — critic only)
```
gyro_pelvis (3)           [noiseless]    # replaces noisy version from state
gvec_pelvis (3)           [noiseless]
joint_angles (11)         [noiseless]
joint_vel (11)            [noiseless]
last_action (11)                         # same as state
motor_targets (11)                       # same as state
box_pos_local (3)                        # same as state
box_quat_local (4)                       # same as state
box_size (3)                             # same as state
surface_z (1)                            # same as state
+ box_vel_local (3)                      # box linear velocity (not deployable)
+ box_angvel (3)                         # box angular velocity
+ left_hand_pos (3)                      # absolute hand positions
+ right_hand_pos (3)
+ box_pos_world (3)                      # absolute box position
+ pelvis_pos (3)                         # absolute body positions (not deployable)
+ torso_pos (3)
+ left_shoulder_pos (3)
+ right_shoulder_pos (3)
+ head_pos (3)
+ left_hand_vel (3)                      # hand linear velocities (not deployable)
+ right_hand_vel (3)
+ kp_scale (1)                           # PD gain DR scalar
+ kd_scale (1)                           # PD gain DR scalar
───────────────────────────────────
Total: ~99 dims  (built from scratch, not from noisy state)
```

### Reward Design
| Term | Formula | Scale | Purpose |
|------|---------|-------|---------|
| **reach** | `-\|left_palm - box\| - \|right_palm - box\|` | 1.0 | Encourage hands to approach box |
| **lift** | `clip(box_z - surface_z - box_half_z, 0, 0.1) / 0.1` | 5.0 | Reward lifting box above surface (saturates at 10cm) |
| **table_force** | `1 - clip(F_support_z / (m * g), 0, 1)` | 2.0 | Reduce contact force on support surface; dense signal bridging reach→lift. Fallback: `box_z - (surface_z + box_half_z)` (no floor clip) if contact reading is too costly in JAX |
| **hold_stable** | `-\|box_angvel\|` | 0.5 | Penalize box tumbling |
| **box_upright** | `exp(-box_tilt_angle^2)` | 1.0 | Keep box upright; ensures good CaTra handoff state |
| **upright** | `exp(-\|pitch\|^2 - \|roll\|^2)` | 1.0 | Robot stays upright |
| **energy** | `-\|torque\|^2` | 1e-4 | Minimize energy |
| **smoothness** | `-\|action - last_action\|^2` | 1e-3 | Smooth actions |
| **joint_limits** | penalty near limits | 1.0 | Avoid joint limits |

### Termination Conditions
- Robot falls (head z < 0.5m, or gravity z < 0)
- Box drops to floor (box z < surface_z - 0.1)
- Any qpos/qvel NaN
- Episode timeout (500 steps = 10s)

### Randomization
| Property | Range | Notes |
|----------|-------|-------|
| Robot yaw | [-90°, 90°] | Uniform |
| Robot joint init | 0.5–1.5× default, clipped | Same as CaTra |
| Box yaw offset | [-10°, 10°] | Relative to robot forward |
| Box half-size x | [0.10, 0.20] | Meters (randomized in DR) |
| Box half-size y | [0.10, 0.25] | Meters (randomized in DR) |
| Box half-size z | [0.10, 0.20] | Meters (randomized in DR) |
| Box mass | [0.5, 3.0] | kg |
| Surface height | [0.4, 0.8] | Meters (center z) |
| KP/KD scale | [0.75, 1.25] | Domain randomization |
| Friction loss, armature, mass | Same as CaTra | Domain randomization |
| RFI | Disabled | Legs not actively controlled; upper-body perturbations risk toppling |

**Note:** Box size/mass randomization requires modifying MJX model fields per-environment in `domain_randomize`. The box geom size is in `model.geom_size[box_geom_id]` and mass in `model.body_mass[box_body_id]`. These can be vmapped like other DR fields.

---

## Files to Create

### 1. `cat_ppo/envs/g1/env_pickup.py` (NEW — main environment)
- **`g1_pickup_task_config()`**: Config function with pickup-specific parameters
- **`domain_randomize_pickup()`**: Extends `domain_randomize_catra` with box size/mass randomization
- **`G1PickupEnv(G1CaTraEnv)`**: New env class overriding:
  - `_post_init_pickup()`: Set 11-DOF action space, cache hand/box IDs
  - `reset()`: Robot yaw [-90°,90°], box with orientation offset, randomized placement
  - `step()`: No gait clock, no command tracking, no push force. Just PD control + reward
  - `_get_obs()`: Compact obs focused on hand-box spatial relationship
  - `_get_reward()`: reach + lift + hold + posture + energy
  - `_get_termination()`: Fall + box drop + NaN
- Register as `"G1Pickup"` task

### 2. `train_ppo_pickup.py` (NEW — training entry point)
- Thin wrapper like `train_ppo_catra.py`
- Imports and calls `train(tyro.cli(Args))`

### 3. `check_pickup.py` (NEW — visualization script)
- Similar to `check_catra.py` but for the pickup env
- Shows initial robot pose, box on surface, and can step with random actions

### 4. `cat_ppo/envs/g1/play_pickup.py` (NEW — CPU inference/visualization env)
- `PlayG1PickupEnv(PlayG1CaTraEnv)` for ONNX inference and visualization
- Handles 11-DOF action space, pickup-specific obs

### Files to Minimally Edit (registration only)
- `cat_ppo/envs/g1/__init__.py` — add `import cat_ppo.envs.g1.env_pickup` so the task registers

---

## Key Assumptions

1. **Legs locked at default**: The 11-DOF action space means legs stay at `DEFAULT_QPOS_CATRA` leg values. The PD controller will hold them there since they receive no action updates. This assumes the robot can balance without active leg control (it can, since it starts balanced and doesn't step).

2. **Reuse CaTra scene XML**: The existing `scene_mjx_feetonly_flat_terrain_catra.xml` already has the box + support surface setup. No new XML needed — box size/mass are randomized via `domain_randomize_pickup()` at the model level.

3. **No HumanoidPF fields**: The pickup task has no obstacles (`TypiObs/empty`), so PF fields are all zeros. We skip computing/observing them entirely to keep obs compact.

4. **No gait clock**: Since the robot doesn't step, gait phase/frequency are unnecessary. The env won't track or reward gait-related quantities.

5. **No command tracking**: No velocity commands. The only "goal" is implicit in the reward: lift the box.

6. **Contact detection via geometry**: We detect "hand touching box" by distance (palm site to box center < threshold) rather than reading MuJoCo contact arrays. MJX contact arrays can be tricky to query efficiently in JAX. Distance-based reach reward is simpler and sufficient.

---

## Verification
1. `python check_pickup.py` — visualize initial state, verify box/surface placement, robot yaw randomization
2. `python train_ppo_pickup.py --task G1Pickup --exp_name pickup_debug --num_timesteps 10_000_000` — short training run to verify no crashes, rewards increase
3. Check that box size/mass vary across environments in DR
4. Verify the robot doesn't step (leg joints stay near default throughout training)
