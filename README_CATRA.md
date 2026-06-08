# G1 Carry and Traverse Policy (CaTra)

Two-stage end-to-end task for the Unitree G1 humanoid: the robot first reaches for a box on a support pillar and lifts it (Stage 1), then walks through a cluttered obstacle course while carrying the box (Stage 2).

This is **Phase 2** of a two-phase curriculum. It combines the pickup policy (Phase 1) with the collision-aware traversal policy (CAT, [arXiv:2601.16035](https://arxiv.org/abs/2601.16035)) into a single end-to-end task.

Two task variants share the same environment dynamics, action space, rewards, and termination:

- **`G1CaTra`** — deployable policy. Actor sees noisy, delayed, navigation-frame observations.
- **`G1CaTraPri`** — privileged-actor variant (teacher for distillation). Actor sees noiseless world-frame observations directly. Domain randomization is disabled. Same critic obs as `G1CaTra`. This is the analogue of `G1CatPri` for the carry-and-traverse setting.

---

## Overview

| Property | Value |
|----------|-------|
| Robot | Unitree G1 humanoid |
| Task | Pick up a box from a pillar, then carry it through obstacles |
| Action space | 20 DOF (12 legs + 8 arms; TEMP: all 3 waist joints removed, held at default) |
| Episode length | 1100 steps (22 s at 50 Hz) |
| Stage 1 | Steps 0–(stage1_steps−1): stand-and-reach, pickup reward set. Set `stage1_steps=0` when using warm-start. |
| Stage 2 | Steps stage1_steps to stage1_steps+999: PF-guided traversal + grasp-maintenance rewards |
| Box placement | 0.3 m in front of robot on a support pillar (default), or loaded from warm-start file |
| Success criterion | Box lifted ≥ 10 cm in Stage 1 AND box still held + robot reaches PF target by Stage 2 end |

| Task | Actor obs | Critic obs | Domain rand | Deployable |
|---|---:|---:|---|---|
| `G1CaTra` | 239 | 333 | on (`domain_randomize_catra`) | yes |
| `G1CaTraPri` | 302 | 333 | **off** | no (uses ground-truth signals) |

---

## Action Space

20-joint PD position target space (TEMP: all 3 waist joints dropped from CaTra's 23-joint set; waist yaw/roll/pitch are held at their default PD targets each step):

```
left_hip_pitch_joint     left_hip_roll_joint      left_hip_yaw_joint
left_knee_joint          left_ankle_pitch_joint   left_ankle_roll_joint
right_hip_pitch_joint    right_hip_roll_joint     right_hip_yaw_joint
right_knee_joint         right_ankle_pitch_joint  right_ankle_roll_joint
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
| Stage 1 — Pickup | 0–(stage1_steps−1) | `stage1_steps × 0.02` s | `[0, 0, 0, 0]` (stationary) | All 22 G1Pickup reward terms |
| Stage 2 — Carry & Traverse | stage1_steps–(stage1_steps+999) | 20 s | PF-derived `[move_flag, vx, vy, yaw]` | CAT navigation rewards + 7 `_carry` grasp-maintenance terms + `feet_rotation` |

`stage1_steps` defaults to 100 (2 s pickup phase). When using warm-start (robot already holds the box at reset), set `stage1_steps=0` to skip Stage 1 entirely and start directly in Stage 2. This is done automatically by `train_ppo.py` when `--warmstart_states_path` is provided and `--stage1_steps` is not specified.

The stage transition is a hard cut at `info["step"] == stage1_steps`. The policy receives a **stage flag** (0.0 in Stage 1, 1.0 in Stage 2) as part of its observation so it can learn stage-specific behaviors from a single policy.

Command in Stage 1 is always `[0, 0, 0, 0]`. In Stage 2 the command is derived every step from PF (potential field) grid data via `compute_cmd_from_rtf`: a 4-dim vector `[move_flag, vx, vy, yaw]` in the navigation frame.

---

## Observation Spaces

Both tasks use an **asymmetric actor-critic** design: the policy sees `state` and the critic sees `privileged_state`. The two tasks differ in what goes into `state` — the `privileged_state` (critic input) is identical between them.

### G1CaTra (239 / 333) — deployable policy

#### `state` (239-dim) — actor input, deployable on real robot

The PF subblock is **not additively noised** — only `gyro`/`gvec`/`joint_angles`/`joint_vel` get noise injection. The body and box PF subblocks use 5-step-delayed samples transformed into the navigation frame (to simulate odometry latency at deployment).

| Field | Dims | Notes |
|-------|------|-------|
| `gyro_pelvis` | 3 | Angular velocity from pelvis IMU `[+ noise]` |
| `gvec_pelvis` | 3 | Gravity direction in pelvis frame `[+ noise]` |
| `joint_angles` | 20 | Controlled joints, relative to default `[+ noise]` |
| `joint_vel` | 20 | Controlled joint velocities `[+ noise]` |
| `last_action` | 20 | Previous policy output |
| `motor_targets` | 20 | Current PD targets for controlled joints |
| `command` | 4 | Navigation command `[move_flag, vx, vy, yaw]`; zeros in Stage 1 |
| `foot_height` | 1 | Target foot height for gait |
| `gait_phase` | 4 | cos+sin of 2D gait clock (left + right) |
| `body_pf` | 77 | Nav-frame PF fields (delayed) for 11 body sites: head, pelvis, torso (×1), feet, hands, knees, shoulders (×2); gf(3)+bf(3)+df(1)=7 per site |
| `box_pf` | 56 | Nav-frame PF fields (delayed) for the **8 corners** of the box; gf(3)+bf(3)+df(1)=7 per corner; bf zeroed and df clipped to [−1, 0.5] when df > 0.5 |
| `box_pos_local` | 3 | Box center position in pelvis frame |
| `box_quat_local` | 4 | Box orientation in pelvis frame (wxyz) |
| `box_size` | 3 | Box half-extents (hx, hy, hz) |
| `stage_flag` | 1 | 0.0 in Stage 1, 1.0 in Stage 2 |
| **Total** | **239** | |

#### `privileged_state` (333-dim) — critic input only during training

Noiseless, world-frame version of the state block (without `box_pos_local`/`box_quat_local`) plus privileged extras. The PF block here uses non-delayed, world-frame samples for both body and box corners.

| Field | Dims | Notes |
|-------|------|-------|
| Noiseless state block | 232 | Same structure as state but noiseless and world-frame; body_pf(77) + box_pf(56) = 133; box_pos/quat_local omitted |
| `linvel_pelvis` | 3 | Pelvis linear velocity (world frame) |
| `pelvis_pos` + `torso_pos` + `head_pos` | 9 | Absolute body positions |
| `shlds_pos` + `hands_pos` + `knees_pos` + `feet_pos` | 24 | 2 sites × 4 body groups × 3 |
| `head_vel` + `hands_vel` + `feet_vel` | 15 | Linear velocities for 5 sites (1+2+2) |
| `box_pos_world` | 3 | Box center in world frame |
| `box_quat_world` | 4 | Box orientation in world frame (wxyz) |
| `box_linvel_world` | 3 | Box linear velocity in world frame |
| `box_angvel_world` | 3 | Box angular velocity in world frame |
| `navi_torso_rpy[:2]` | 2 | Torso roll + pitch in navigation frame |
| `gait_mask` | 2 | Per-foot contact-based gait mask (left, right) |
| `feet_contact` | 2 | Binary foot contact flags: `[left_touching_floor, right_touching_floor]` |
| `rfi_lim_scale` | 29 | Per-joint random force injection scale |
| `kp_scale` | 1 | PD gain DR scalar |
| `kd_scale` | 1 | PD gain DR scalar |
| **Total** | **333** | |

### G1CaTraPri (302 / 333) — privileged-actor variant

Teacher policy for distillation. **Not deployable** (uses ground-truth signals unavailable on hardware).

#### `state` (302-dim) — actor input

The actor sees the **same content as the G1CaTra `privileged_state`, minus the trailing 31 DR dims** (`rfi_lim_scale` + `kp_scale` + `kd_scale`). Implemented as a `priv[:-31]` slice in `G1CaTraPriEnv._get_obs` — no noise, current world-frame PF, body positions / velocities, box world pose and velocities, pelvis linvel, `navi_torso_rpy[:2]`, `gait_mask`, `feet_contact`, `stage_flag`.

| Field block | Dims | Notes |
|-------|------|-------|
| Noiseless state block | 232 | Same as G1CaTra's privileged block |
| Privileged extras (no DR scales) | 70 | linvel_pelvis(3) + body positions(33) + body velocities(15) + box world(13) + nav/gait/contact(6) |
| **Total** | **302** | |

Key differences from G1CaTra `state`: no sensor noise, PF in world frame without 5-step delay (not nav frame), includes body positions/velocities and box world dynamics directly, no DR scales.

#### `privileged_state` (333-dim) — critic input

**Identical** to G1CaTra's `privileged_state`. Both tasks share the same critic input dimensions and signals.

---

## Reward Design

All rewards are multiplied by `dt`. The reward set is identical for `G1CaTra` and `G1CaTraPri` — both inherit `_get_reward` from `G1CaTraEnv` (since `G1CaTraPriEnv` only overrides `_get_obs`).

### Always-Active Terms (applied in both stages)

| Term | Scale | Purpose |
|------|-------|---------|
| `joint_torque` | -1e-4 | Penalize high motor torque |
| `smoothness_joint` | -1e-6 | Penalize jerky joint motion |
| `joint_limits` | -1.0 | Penalize joint limit violations |

### Stage 1 — Pickup Rewards (active when step < stage1_steps)

| Term | Scale | Formula |
|------|-------|---------|
| `reach` | 0.0 | Distance from each palm to its target box face |
| `lift` | 0.0 | Box height above pillar, capped at +10 cm |
| `hand_contact` | 0.0 | 0.5 per hand in contact with box |
| `box_pillar_contact` | 0.0 | Penalize box still resting on pillar |
| `grasp_symmetry` | 0.0 | Penalize height/depth asymmetry between two palms |
| `palm_orient` | 0.0 | Reward palm normals facing inward toward box |
| `hands_level` | 0.0 | Penalize hand-to-hand tilt out of horizontal plane |
| `hold_stable` | 0.0 | Box linvel + angvel penalty |
| `box_yaw_stable` | 0.0 | Box yaw drift penalty |
| `box_centering` | 0.0 | Lateral offset penalty |
| `box_vertical` | 0.0 | Penalize XY drift from pillar center |
| `box_upright` | 0.0 | Keep box vertical once lifted |
| `upright` | 3.0 | Asymmetric pitch/roll penalty on robot posture |
| `foot_contact` | -0.5 | Both feet must stay planted |
| `foot_slip` | -0.1 | Penalize foot sliding |
| `straight_knee` | -5.0 | Discourage locked knees |
| `smoothness` | 1e-3 | Smooth action transitions |
| `base_height` | 1.0 | Target pelvis height 0.75 m |
| `foot_balance` | -30.0 | Penalize COM centering and foot spread |

Most pickup terms are at scale 0.0 by default since Stage 1 is typically skipped in training (warm-start replaces it). Re-enable by editing the config if you want to train Stage 1 end-to-end.

### Stage 2 — Traversal + Carry Rewards (active when step ≥ stage1_steps)

Navigation rewards (from `G1CatEnv`):

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
| `feet_rotation` | 1.0 | Reward clean knee+ankle alignment with nav frame (knee roll/yaw, ankle roll/pitch/yaw → 0). Indirect anti-crouch; peaks at +1.0 in a tall, forward-aligned stance. Ported from `G1CatPri`. |
| `smoothness_action` | -1e-3 | Smooth action transitions |
| `forward_progress` | 20.0 | Linear reward for velocity in the command direction: `clip(v·cmd_dir, 0, ‖cmd‖)`. Gives nonzero gradient from a dead stop, unlike the exp-based `tracking_root_field`. |
| `upper_body_align` | -0.0 | Penalize XY drift of torso and head from pelvis (scaled off; previously −2.0) |
| `headgf/handsgf/feetgf` | 0.0 | Body goal field tracking; set via `--overheadgf` / `--lateralgf` / `--groundgf` |
| `headdf/handsdf/feetdf/kneesdf/shldsdf` | 0.0 | Body distance field penalties; set via `--overheaddf` / `--lateraldf` / `--grounddf` |
| `boxdf` | 0.0 | Box-corner SDF collision penalty: `mean(softplus((0.05 − sdf) / 0.02))` over 8 corners; enable with `--box <scale>` |

Grasp-maintenance rewards (`_carry` suffix):

| Term | Scale | Purpose |
|------|-------|---------|
| `reach_carry` | 3.0 | Keep palms close to box faces |
| `lift_carry` | 0.0 | Keep box elevated (scaled off) |
| `hand_contact_carry` | 2.0 | Maintain bilateral contact |
| `grasp_symmetry_carry` | -2.0 | Maintain symmetric grasp |
| `palm_orient_carry` | 2.0 | Maintain correct palm orientation |
| `hands_level_carry` | -1.0 | Maintain level hands |
| `box_upright_carry` | 2.0 | Reward small box tilt angle: `exp(−tilt²)` |

---

## Termination Conditions

| Condition | Threshold |
|-----------|-----------|
| Robot fall (gravity vector) | `gvec_z < 0` |
| Robot fall (head height) | `head_z < 0.7 m` |
| Box dropped | `box_z < 0.3 m` — active throughout full episode |
| Body-obstacle SDF collision | any of head/torso/pelvis/feet/hands/knees/shoulders df < −4 cm (active after step `stage1_steps + 50`) |
| Box-corner SDF collision | any of the 8 box corners df < −4 cm (same threshold; gated identically) |
| NaN in qpos or qvel | any |
| Episode timeout | 1100 steps (22 s) |

---

## Randomization

There are two reset modes (default / warm-start), and warm-start has two DR variants depending on the task.

### Default reset (no warm-start)

Per-episode randomization in `reset()`:

| Property | Range | Notes |
|----------|-------|-------|
| Robot initial yaw | `U[−90°, 90°]` | Facing generally toward ±X |
| Robot joint init | `U[0.5, 1.5] × default`, clipped to soft limits | |
| Box XY position | 0.3 m forward from robot pelvis | Robot yaw/XY spawn provides variety |
| Box yaw offset | `U[−10°, 10°]` | Relative to robot forward direction |
| Support pillar height | Fixed at surface_z = 0.3 m | Box-center z at surface_z + box_half_z |

Per-environment DR via `domain_randomize_catra` (applied once at training init, used by `G1CaTra` only):

| Property | Range | Notes |
|----------|-------|-------|
| Joint frictionloss | `U[0.9, 1.1] × nominal` | Robot joints |
| Joint armature | `U[1.0, 1.05] × nominal` | Robot joints |
| Torso CoM offset | `U[−0.1, 0.1]` m per axis | |
| All body masses | `U[0.9, 1.1] × nominal` | |
| Torso mass perturbation | `U[−1, 1]` kg additive | |
| qpos0 perturbation | `U[−0.05, 0.05]` per joint | |
| Box half-size (x, y, z) | Fixed at (0.15, 0.20, 0.15) m | Uniform size, no DR |
| Box mass | `U[1.0, 2.0]` kg | Per-environment |
| KP scale | `U[0.75, 1.25]` | Per-reset |
| KD scale | `U[0.75, 1.25]` | Per-reset |
| RFI (per-joint torque noise) | Enabled | Applied every substep in both stages |
| Push impulses | Enabled | Random root velocity impulses; gated to stage 2 only |

`G1CaTraPri` ships with `randomization_fn = None` — none of the per-environment DR above is applied. Per-reset KP/KD/RFI scalars are still drawn (they're set inside `reset()`, not in the DR function).

### Warm-start reset (`warmstart_states_path` set)

Per-episode reset loads `(qpos, qvel)` exactly as saved from the pickup policy rollout — robot pose, box position, and box orientation are **not changed**.

`train_ppo.py` dispatches to one of two DR factories based on the task config's `randomization_fn`:

| Task | DR factory used when warm-starting | Behavior |
|---|---|---|
| `G1CaTra` (config: `randomization_fn = domain_randomize_catra`) | `make_warmstart_domain_randomize_catra` | Full robot DR (frictionloss / armature / CoM / mass / qpos0 jitter) **plus** per-env box mass/size from the state file **plus** the `qpos0[0]` index dispatch needed by `reset()`. |
| `G1CaTraPri` (config: `randomization_fn = None`) | `make_warmstart_only_catra` | **DR-free**: only the box mass/size load and `qpos0[0]` index dispatch. Robot params stay at nominal — appropriate for a teacher policy. |

For both factories, state index `i` is encoded in `qpos0[0]` and decoded in `reset()` to look up env i's pre-saved `(qpos, qvel)` from the `.npz` file. The state file must contain exactly `num_envs` entries.

---

## Architecture

### Class Hierarchy

```
MjxEnv (mujoco_playground)
  └─ G1Env
       └─ G1LocoEnv
            └─ G1CatEnv
                 └─ G1CaTraEnv          ← deployable task
                      ├─ G1CaTraPriEnv  ← privileged-actor variant (overrides _get_obs only)
                      └─ G1PickupEnv
```

`G1CaTraEnv` overrides `reset`, `step`, `_get_obs`, `_get_reward`, and `_get_termination`. It inherits the HumanoidPF fields and navigation reward infrastructure from `G1CatEnv`, and incorporates the full G1Pickup reward set for Stage 1.

`G1CaTraPriEnv` is a thin subclass — it overrides only `_get_obs` to feed the actor a `priv[:-31]` slice of the privileged state (everything except the DR scales). All dynamics, rewards, terminations, and warm-start logic are inherited from `G1CaTraEnv`.

### Key Implementation Details

- **Stage flag in obs**: The scalar `0.0` (Stage 1) or `1.0` (Stage 2) appended as the last element of the proprioceptive block. Lets the policy learn stage-specific behavior from a single network. With warm-start (`stage1_steps=0`) the flag is always 1.0.
- **Warm-start**: When `warmstart_states_path` is set, `reset()` loads `(qpos, qvel)` directly from the pre-generated `.npz` (no physics rollout inside reset). Box mass/size are carried exactly from the pickup generation run.
- **G1CaTraPri obs construction**: `G1CaTraPriEnv._get_obs` calls `super()._get_obs()` and slices `priv[:-31]` to form the actor state. Because the slice is structural, the bit-exact identity `G1CaTra.privileged_state[:-31] == G1CaTraPri.state` holds for the same initial state.
- **JIT-friendly reward gating**: Both stage reward dicts are computed every step; `jp.where(step < stage1_steps, ...)` gates which set contributes to the return. Dict shape is static — no dynamic branching.
- **Push forces gated**: Random push perturbations are suppressed during Stage 1 (`step < stage1_steps`). Full pushes activate in Stage 2 when locomotion is expected.
- **Box corner PF**: The obstacle fields (sdf, bf, gf) are sampled at all **8 corners** of the box each step and included in both the deployable state and privileged state (56 dims each: 8 corners × (gf:3 + bf:3 + sdf:1)). The same 5-step delay + nav-frame transform applied to body PF is applied to box corner PF in the deployable state. The `boxdf` reward is scaled 0.0 by default; enable with `--box <scale>`.
- **Box drop threshold**: Termination fires when `box_z < 0.3 m` (at or below pillar surface), allowing the box to move freely above that during carries.
- **SDF termination gating**: Body/box-obstacle collision termination is suppressed until step `stage1_steps + 50`, giving the robot time to stabilize its carry before collision penalties apply.
- **Navigation command sites**: `compute_cmd_from_rtf` builds the Stage 2 PF command from the pelvis + head + feet goal/body fields only. Hands were removed from this aggregation — they still appear in the observation PF subblock and are still affected by `handsdf` / `handsgf` rewards, but no longer steer the navigation command.
- **Collision geometry updates**: `torso_collision` is a fatter capsule (`size=0.09`, shifted to `fromto="0.01 0 0.08 0.01 0 0.2"`) and `head_collision` is a larger sphere (`size=0.06`, `pos="0 0 0.43"`) — closer to the actual robot envelope so the box does not sink into the torso/head. A `pelvis_collision` ↔ `box_geom` contact pair is declared in both the flat-terrain training scene and the mesh play scene, letting the box physically rest against the pelvis during carries.
- **qpos/qvel layout**:
  ```
  qpos: [0:7] root | [7:36] robot joints (29) | [36:43] box freejoint | [43:50] support freejoint
  qvel: [0:6] root | [6:35] robot joints (29) | [35:41] box vel       | [41:47] support vel
  ```

### Files

| File | Purpose |
|------|---------|
| [cat_ppo/envs/g1/env_catra.py](cat_ppo/envs/g1/env_catra.py) | Main training environment, config, DR functions (default + warm-start + warm-start-only) |
| [cat_ppo/envs/g1/env_catra_pri.py](cat_ppo/envs/g1/env_catra_pri.py) | `G1CaTraPriEnv` subclass and `g1_catra_pri_task_config` |
| [cat_ppo/envs/g1/play_catra.py](cat_ppo/envs/g1/play_catra.py) | CPU inference env for ONNX playback. Dual-registered for both tasks; toggles between actor obs styles via `env.pri` (set by `--pri` in `mj_onnx_play`) |
| [cat_ppo/envs/g1/pickup_warmstart.py](cat_ppo/envs/g1/pickup_warmstart.py) | `load_pickup_inference_fn`, `pickup_obs_from_data` (used by warm-start generation) |
| [cat_ppo/eval/warmstart_generation.py](cat_ppo/eval/warmstart_generation.py) | Offline warm-start state generation implementation |
| [generate_warmstart_states.py](generate_warmstart_states.py) | CLI entry point for generating warm-start states |
| [check_warmstart_states.py](check_warmstart_states.py) | Sanity-check script for generated `.npz` state files |
| [train_ppo_catra.py](train_ppo_catra.py) | Training entry point (dispatches on `--task`) |
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

### Generate Warm-Start States (one-time, requires trained G1Pickup checkpoint)

```bash
python generate_warmstart_states.py \
    --pickup_checkpoint_path /abs/path/to/G1Pickup/checkpoints/000403046400 \
    --num_states 32768 \
    --output data/warmstart/catra_pickup_states.npz
```

Verify the generated file:
```bash
python check_warmstart_states.py --states data/warmstart/catra_pickup_states.npz
# --view  to open the MuJoCo viewer on one of the saved states
```

### Train G1CaTra (deployable policy)

Without obstacles, default reset (2 s Stage 1 pickup + Stage 2 traversal):
```bash
python train_ppo_catra.py \
    --task G1CaTra \
    --exp_name catra_v1
```

With warm-start (robot initializes already holding the box, jumps straight to traversal):
```bash
python train_ppo_catra.py \
    --task G1CaTra \
    --exp_name catra_v1 \
    --warmstart_states_path data/warmstart/catra_pickup_states.npz
```
This automatically sets `stage1_steps=0` and installs `make_warmstart_domain_randomize_catra` (warm-start with full robot DR).

With obstacles (enable body-PF guidance + SDF penalties, including box-corner collision penalty):
```bash
python train_ppo_catra.py \
    --task G1CaTra \
    --exp_name catra_v1 \
    --warmstart_states_path data/warmstart/catra_pickup_states.npz \
    --obs_path data/assets/TypiObs/bar0 \
    --groundgf 1.0 --grounddf 1.0 \
    --lateralgf 1.0 --lateraldf 0.4 \
    --overheadgf 1.0 --overheaddf 1.0 \
    --box 1.0
```

### Train G1CaTraPri (privileged-actor teacher)

Same CLI; the task name selects the privileged variant. DR is automatically off (`randomization_fn = None` in the config). When you pass `--warmstart_states_path`, `train_ppo.py` detects `randomization_fn is None` and installs `make_warmstart_only_catra` instead of the DR-on warm-start fn — preserving the warm-start state load (box mass/size + index dispatch) while keeping robot DR off.

```bash
python train_ppo_catra.py \
    --task G1CaTraPri \
    --exp_name catrapri_v1 \
    --warmstart_states_path data/warmstart/catra_pickup_states.npz \
    --obs_path data/assets/TypiObs/bar0 \
    --groundgf 1.0 --grounddf 1.0 \
    --lateralgf 1.0 --lateraldf 0.4 \
    --overheadgf 1.0 --overheaddf 1.0 \
    --box 1.0
```

### Per-body-group reward scales (apply to both tasks)

- `--groundgf` / `--grounddf` → `feetgf` / `feetdf`
- `--lateralgf` / `--lateraldf` → `handsgf` / `handsdf` / `kneesdf` / `shldsdf`
- `--overheadgf` / `--overheaddf` → `headgf` / `headdf`
- `--box` → `boxdf` (box-corner SDF penalty)

All default to 0 (disabled).

### Export to ONNX

```bash
python -m cat_ppo.eval.brax2onnx --task G1CaTra    --exp_name <full_exp_name>
python -m cat_ppo.eval.brax2onnx --task G1CaTraPri --exp_name <full_exp_name>
```

ONNX export auto-detects the actor obs size from the task config (`policy_obs_key="state"` → 239 for `G1CaTra`, 302 for `G1CaTraPri`).

### Play in MuJoCo Viewer

```bash
# G1CaTra: deployable policy, builds the 239-dim noisy/delayed state
python -m cat_ppo.eval.mj_onnx_play --task G1CaTra --exp_name <full_exp_name>

# G1CaTraPri: privileged-actor policy, --pri flips the play env to build the 302-dim state
python -m cat_ppo.eval.mj_onnx_play --task G1CaTraPri --pri --exp_name <full_exp_name>

# Warm-start (either task): robot starts already holding the box, Stage 2 only
python -m cat_ppo.eval.mj_onnx_play --task G1CaTra --exp_name <full_exp_name> \
    --warmstart_states_path data/warmstart/catra_pickup_states.npz
# Use --warmstart_idx <N> to load a specific state instead of random
```

The play env (`PlayG1CaTraEnv`) is dual-registered for both tasks. The `--pri` flag sets `env.pri = True`, which switches `get_obs` between the deployable (239-dim) and privileged (302-dim) state builders.

### Smoke Test (stage transition verification)

```python
import jax, jax.numpy as jp, cat_ppo, cat_ppo.envs.g1

env_cls = cat_ppo.registry.get("G1CaTra", "train_env_class")
cfg     = cat_ppo.registry.get("G1CaTra", "config").env_config
env     = env_cls(task_type=cfg.task_type, config=cfg)

state = jax.jit(env.reset)(jax.random.PRNGKey(0))
step  = jax.jit(env.step)
for i in range(110):
    state = step(state, jp.zeros(cfg.num_act))
    if i in (98, 99, 100, 101):
        print(f"step {i}: command={state.info['command']}, "
              f"reach={state.metrics.get('reward/reach', 0):.4f}, "
              f"reach_carry={state.metrics.get('reward/reach_carry', 0):.4f}")
```
