# CLAUDE.md

## Project Overview

**Click-and-Traverse (CAT)** — Collision-free humanoid traversal in cluttered indoor scenes.
Paper: [arXiv:2601.16035](https://arxiv.org/abs/2601.16035) (Xue et al., 2026).
Robot: Unitree G1 humanoid. Simulator: MuJoCo/MJX. RL framework: Brax PPO + JAX.

The core contribution is **HumanoidPF** (Humanoid Potential Field), a representation that encodes humanoid-obstacle spatial relationships as collision-free motion directions, used both as policy observation and for reward shaping.

## Tech Stack

- **Python 3.12.9**, CUDA 12.5
- **JAX 0.4.38** + **Brax 0.12.3** — RL training (PPO), vectorized environments
- **Flax 0.10.4** — neural network definitions
- **MuJoCo 3.3.1** / MJX — physics simulation (GPU-accelerated via JAX)
- **mujoco_playground** — environment base class (`MjxEnv`)
- **ml_collections** — config management (dictionary-like, supports attribute access and overrides)
- **tyro** — CLI argument parsing
- **TensorFlow 2.19 + tf2onnx** — JAX-to-ONNX model export
- **onnxruntime** — inference
- **scikit-fmm** — fast marching method (geodesic distances for HumanoidPF)
- **swanlab / wandb** — experiment tracking
- **Formatter**: ruff (line-length 120, target py312, select F, ignore F401)

## Key Commands

```bash
# Setup
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .venv/bin/activate && source .env

# Initialize MuJoCo assets
python -m cat_ppo.utils.mj_playground_init

# Train single experiment
python train_ppo.py --task G1Cat --exp_name <name> --ground 1.0 --lateral 1.0 --overhead 1.0 --obs_path data/assets/TypiObs/<scene>

# Batch training (multi-GPU)
python train_batch.py

# Export to ONNX
python -m cat_ppo.eval.brax2onnx --task G1Cat --exp_name <name>

# Evaluate
python -m cat_ppo.eval.mj_onnx_play --task G1Cat --exp_name <name> --obs_path <path>

# Generate obstacles
cd procedural_obstacle_generation && python main.py
```

## Architecture

### Environment Hierarchy
`G1Env` (base, MjxEnv wrapper) → `G1LocoEnv` (locomotion) → `G1CatEnv` / `G1CatPriEnv` (collision-aware traversal)

- **G1LocoEnv**: joystick-guided walking, 12-dim action (PD joint targets), gait clock, domain randomization base
- **G1CatEnv** (`G1Cat`): sim-to-real deployable specialist. 162-dim `state` (HumanoidPF fields only, no absolute positions), 250-dim `privileged_state` (critic). Domain randomization **enabled**. Terminates on SDF collision for all 7 body groups.
- **G1CatPriEnv** (`G1CatPri`): teacher policy for DAgger distillation, NOT deployable. 175-dim `state` adds ground-truth `linvel`, absolute body positions, feet_contact. 209-dim `privileged_state`. Domain randomization **disabled**. Terminates on SDF collision for head/feet/hands only.

### Registered Tasks
- `G1Loco` / `G1LocoDis` — baseline locomotion
- `G1Cat` — standard collision-aware (sim-to-real ready)
- `G1CatPri` — privileged observation (for distillation)

### Key Directories
```
cat_ppo/envs/g1/          # Environment definitions (env_cat.py is the main one)
cat_ppo/learning/          # PPO training (modified Brax trainer)
cat_ppo/eval/              # ONNX export + inference
cat_ppo/utils/             # Registry, logging, asset init
procedural_obstacle_generation/  # Obstacle & HumanoidPF construction
deploy/                    # Real-world deployment (Unitree SDK, ROS2, SLAM)
data/assets/               # Obstacle scenes (TypiObs/, RandObs/, R2SObs/) + robot MJCF
data/logs/origin/          # Pre-trained checkpoints
```

### HumanoidPF Pipeline
1. Occupancy grid (voxel 0.04m) → 2. SDF (fast marching) → 3. Boundary field (gradient of SDF) → 4. Guidance field (combined attractive + repulsive potential gradient)
- Attractive field: geodesic distance to goal
- Repulsive field: obstacle proximity penalty
- Priority weighting: root body part gets higher weight; collision-urgency weighting based on distance and velocity
- Sampled at K=13 body parts → forms `OBS_Field` observation
- Also used for vMF-distribution-based reward (`R_Field`)

### Training Pipeline
1. Hybrid scene generation: 3D-FRONT crops + procedurally synthesized obstacles
2. Parallel specialist training: one PPO policy per scene (32,768 envs, ~400M timesteps)
3. Specialist-to-generalist distillation via DAgger (TODO — not yet released)
4. Sim-to-real via domain randomization (PD gains, sensor noise, force perturbations)

---

## G1Cat — Observation & Reward Details

### Key env_config parameters
- `ctrl_dt=0.02`, `sim_dt=0.002` — policy runs at 50 Hz; physics integrates at 500 Hz (10 substeps per action)
- `episode_length=1000` — 1000 control steps = 20 seconds per episode
- `action_repeat=1` — policy called every control step (no action repetition)
- `action_scale=0.5` — policy output scaled by 0.5 before adding to default joint angles as PD target
- `history_len=15` — last 15 timesteps of joint state stacked into observation

### Gait clock
A phase oscillator that drives the walking rhythm. Phase advances each step by `2π × ctrl_dt × freq`.
- `freq_range=[1.3, 1.5]` Hz — stepping frequency, randomized per episode for behavioral diversity
- `foot_height_range=[0.07, 0.07]` m — target foot clearance during swing phase (incentivized via reward, not hard constraint)
- `gait_bound=0.6` — threshold on `cos(phase)` to determine swing/stance: `>0.6` = right swings, `<-0.6` = left swings, in between = transition
- `(cos, sin)` of each foot's phase is included in observation so the policy knows unambiguously where in the cycle each foot is

### Domain randomization (`dm_rand_config`)
Injected during training to reduce sim-to-real gap:
- `enable_pd=True`, `kp_range=[0.75, 1.25]`, `kd_range=[0.75, 1.25]` — PD gains scaled by random ±25% each episode, simulating motor uncertainty
- `enable_rfi=True`, `rfi_lim=0.1`, `rfi_lim_range=[0.5, 1.5]` — random torque noise up to 10% of torque limit (×random scale), simulating joint friction and unmodeled dynamics
- `enable_ctrl_delay=False` — control latency simulation (disabled)

### Sensor noise (`noise_config`)
Added to policy observations to match real IMU/encoder noise:
- `joint_pos=0.03` rad — encoder noise (±1.7°)
- `joint_vel=1.5` rad/s — velocity estimate noise (large, as velocity is derived from finite differences on hardware)
- `gravity=0.05` — IMU gravity vector noise
- `gyro=0.2` rad/s — gyroscope noise
- Set `level=0.0` to disable all noise at once

### Observations

**`state` (162-dim) — policy input, deployable on real hardware (noisy, no absolute positions)**
| Component | Dims | Description |
|---|---|---|
| gyro (pelvis) | 3 | Angular velocity from IMU |
| gravity vector (pelvis) | 3 | Tilt direction from IMU |
| joint angles (23 joints) | 23 | Relative to default pose, noisy |
| joint velocities (23 joints) | 23 | Noisy |
| last action | 12 | Previous policy output |
| motor targets | 12 | Previous PD targets |
| command | 4 | [move_flag, vx, vy, yaw] |
| foot height target | 1 | Sampled foot clearance for this episode |
| gait phase | 4 | [cos_L, cos_R, sin_L, sin_R] |
| HumanoidPF fields (`pf`) | 77 | See below |

**HumanoidPF `pf` block (77-dim):** 3 fields per body group (gf=guidance, bf=boundary, df=SDF distance):
- Single-point bodies (head, pelvis, torso): gf(3) + bf(3) + df(1) = 7 each × 3 = 21
- Paired bodies (feet, hands, knees, shoulders): gf(6) + bf(6) + df(2) = 14 each × 4 = 56

**`privileged_state` (224-dim) — critic input only, noiseless + ground-truth extras**

Superset of `state` but noiseless, plus: all absolute body positions/velocities (head, pelvis, torso, feet, hands, knees, shoulders), feet contact, gait mask, torso RPY, domain randomization scales (kp, kd, rfi).

### Rewards

**Behavior rewards** (scale in `reward_config.scales`):
- `tracking_orientation=2.0` — keep pelvis/torso upright and level
- `tracking_root_field=1.0` — match commanded velocity (exp(-4 × vel_error))
- `body_motion=-0.5` — penalize lateral drift and unwanted rotation
- `body_rotation=1.0` — reward facing the commanded direction
- `foot_contact=-1.0` — penalize wrong foot contacting ground (gait clock violation)
- `foot_clearance=-15.0` — penalize insufficient foot height during swing
- `foot_slip=-0.5` — penalize foot sliding during stance
- `foot_balance=-30` — penalize feet too far apart laterally
- `foot_far=-0` (inactive in G1Cat, -3.0 in G1Loco; misleading name) — penalizes feet being too CLOSE together (< 0.35m apart), not too far
- `straight_knee=-30` — penalize fully locked/hyperextended knees

**Energy/regularization rewards:**
- `smoothness_joint=-1e-6` — penalize joint jerk (change in velocity)
- `smoothness_action=-1e-3` — penalize sudden action changes
- `joint_limits=-1.0` — penalize exceeding soft joint range (95% of hardware limit)
- `joint_torque=-1e-4` — penalize high motor torques

**HumanoidPF field rewards** (all 0.0 by default, set via CLI `--ground/lateral/overhead`):
- `headgf`, `feetgf`, `handsgf` — vMF-based: reward body velocity aligning with guidance field direction when near obstacles (activated within SDF distance `tau`)
- `headdf`, `feetdf`, `handsdf`, `kneesdf`, `shldsdf` — SDF penalty: softplus penalty when body part is within 0.05m of obstacle surface

---

## G1CatPri — Differences from G1Cat

G1CatPri is the **teacher policy** for DAgger distillation. It has access to privileged ground-truth information not available on real hardware, so it is NOT deployable.

### Observation differences

**`state` (175-dim = 162 + 13)** — adds to G1Cat's state:
- `linvel_pelvis` (3) — ground-truth linear velocity (not available from IMU on real robot)
- `head_pos`, `feet_pos`, `hands_pos` (3+6+6=15? → net +13) — absolute body positions
- `feet_contact` (2), `gait_mask` (2), `torso_rpy[:2]` (2) — explicit contact/gait state

Also note: G1CatPri's `state` does **not** include domain randomization scales (kp, kd, rfi) — those are commented out, unlike G1Cat.

**`privileged_state` (209-dim)** — smaller than G1Cat's 224 because the policy already sees more, so the critic needs less extra information.

Also G1CatPri includes `rtf` (relative-to-frame) vector in both state and privileged_state — not present in G1Cat.

### Reward differences
- Adds `feet_rotation=1.0` — rewards feet pointing in the commanded direction (extra locomotion quality term for the teacher)
- `foot_balance=-10` (vs -30 in G1Cat) — less strict foot balance penalty
- No `foot_far` term

### Config differences
- `foot_height_range=[0.05, 0.05]` (vs 0.07 in G1Cat) — lower target foot clearance
- Domain randomization **disabled** (`randomization_fn=None`) — teacher trains in clean sim

---

## Environment Variables (.env)
- `GLI_PATH` — path to cat_ppo package
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` — experiment tracking
- `MUJOCO_GL=egl` — GPU rendering (set in train_ppo.py)

## Project Status (as of repo)
- Specialist training, obstacle generation, HumanoidPF, R2S, deployment: done
- Generalist distillation code, generalist models, expanded datasets: TODO
