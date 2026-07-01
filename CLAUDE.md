# CLAUDE.md

## Project Overview

This branch (`object_carry_exp`) implements **G1Pickup** — a manipulation policy where the Unitree G1 humanoid reaches for a box resting on a support surface, grasps it with both hands, and lifts it off.

**G1Pickup is Phase 1 of a two-phase curriculum:**
1. **Pickup (this branch)**: Robot stands in place, uses waist + arms to lift a box. Produces diverse "robot holding box" states.
2. **CaTra (Carry and Traverse, future)**: Robot initialized from pickup terminal states, walks through cluttered indoor scenes while carrying the box. Builds on the base CAT ([arXiv:2601.16035](https://arxiv.org/abs/2601.16035)) locomotion policy.

Robot: Unitree G1 humanoid. Simulator: MuJoCo/MJX. RL framework: Brax PPO + JAX.

## Tech Stack

- **Python 3.12.9**, CUDA 12.5
- **JAX 0.4.38** + **Brax 0.12.3** — RL training (PPO), vectorized environments
- **Flax 0.10.4** — neural network definitions
- **MuJoCo 3.3.1** / MJX — physics simulation (GPU-accelerated via JAX)
- **mujoco_playground** — environment base class (`MjxEnv`)
- **ml_collections** — config management
- **tyro** — CLI argument parsing
- **TensorFlow 2.19 + tf2onnx** — JAX-to-ONNX model export
- **onnxruntime** — inference
- **swanlab / wandb** — experiment tracking
- **Formatter**: ruff (line-length 120, target py312, select F, ignore F401)

## Key Commands

```bash
# Setup
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .venv/bin/activate && source .env

# Initialize MuJoCo assets
python -m cat_ppo.utils.mj_playground_init

# Visualize a pickup episode
python check_pickup.py
python check_pickup.py --surface_z 0.6

# Train pickup policy
python train_ppo_pickup.py --task G1Pickup --exp_name pickup_v1

# Export to ONNX
python -m cat_ppo.eval.brax2onnx --task G1Pickup --exp_name <full_exp_name>

# Play in MuJoCo viewer
python -m cat_ppo.eval.mj_onnx_play --task G1Pickup --exp_name <full_exp_name>
```

## Architecture

### Environment Hierarchy
```
MjxEnv → G1Env → G1LocoEnv → G1CatEnv → G1CaTraEnv → G1PickupEnv
```

- **G1LocoEnv**: joystick-guided walking, 12-DOF legs, gait clock, base domain randomization
- **G1CatEnv** (`G1Cat`): collision-aware traversal with HumanoidPF fields. 162-dim state, 250-dim privileged_state.
- **G1CaTraEnv** (`G1CaTra`): extends G1CatEnv with a carried box (freejoint) and box freejoint handling in qpos/qvel slicing. Single-stage box transport (500 steps): warm-started already holding the box, then PF-guided traversal + `_carry` grasp-maintenance rewards. Warm-start init is mandatory (no pickup stage). Action space is currently 20-DOF (12 legs + 8 arms; 3 waist joints TEMP held at default).
- **G1PickupEnv** (`G1Pickup`): overrides G1CaTraEnv to 11-DOF action (waist + arms only), legs held at default. Compact obs (61-dim state / 99-dim privileged), pickup-specific rewards.

### Registered Tasks
- `G1Loco` / `G1LocoDis` — baseline locomotion
- `G1Cat` / `G1CatPri` — collision-aware traversal
- `G1CaTra` — single-stage box carry & traverse (500 steps, warm-start only, box freejoint)
- `G1Pickup` — box pickup, Phase 1 of CaTra curriculum

### Key Files
```
cat_ppo/envs/g1/env_pickup.py     # G1PickupEnv, domain_randomize_pickup, config
cat_ppo/envs/g1/play_pickup.py    # CPU inference env for ONNX playback
cat_ppo/envs/g1/env_catra.py      # G1CaTraEnv (parent of G1PickupEnv)
cat_ppo/envs/g1/constants.py      # DEFAULT_QPOS_CATRA, CATRA_ACTION_JOINT_NAMES, etc.
train_ppo_pickup.py                # Training entry point
check_pickup.py                    # Episode visualization
README_PICKUP.md                   # Full task documentation
```

### Box Freejoint Layout
The CaTra scene XML adds a box body with a freejoint, extending qpos/qvel:
```
qpos: [0:7] root | [7:36] robot joints (29) | [36:43] box freejoint (7)
qvel: [0:6] root | [6:35] robot joints (29) | [35:41] box vel (6)
```
`torque_step_catra` explicitly slices `[7:36]` / `[6:35]` to avoid shape mismatch.

### Domain Randomization
`domain_randomize_pickup` = CAT's `domain_randomize` (frictionloss, armature, CoM, mass, qpos0) + box-specific terms (box half-size x/y/z, box mass). Robot DR terms are identical to CAT. RFI is disabled (legs not actively controlled; perturbations risk toppling).

## Environment Variables (.env)
- `GLI_PATH` — path to cat_ppo package
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` — experiment tracking
- `MUJOCO_GL=egl` — GPU rendering (set in train_ppo.py)

## Project Status
- G1Pickup environment, config, DR, obs, rewards: done
- Training: not yet started
- CaTra traversal phase (Phase 2): planned
