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
- **ml_collections** — config management
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

## Environment Variables (.env)
- `GLI_PATH` — path to cat_ppo package
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` — experiment tracking
- `MUJOCO_GL=egl` — GPU rendering (set in train_ppo.py)

## Project Status (as of repo)
- Specialist training, obstacle generation, HumanoidPF, R2S, deployment: done
- Generalist distillation code, generalist models, expanded datasets: TODO
