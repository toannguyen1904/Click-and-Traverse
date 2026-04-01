# CaTra (Carry and Traverse) — Implementation Plan

## Context

Extend Click-and-Traverse so the G1 humanoid carries a box while navigating cluttered obstacles. The robot starts in a holding pose with the box already in its hands. Active arm control from the start (arms in action space). Box size and mass are domain-randomized. Built as an extension within the current repo.

---

## Phase 1: MJCF Scene with Box + Carrying Pose

### 1.1 New scene XML
**Create**: `data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml`

Based on `scene_mjx_feetonly_flat_terrain.xml`. Additions:
- A `<body name="carried_box">` in `<worldbody>` with a `<freejoint>`
- `<geom type="box" size="0.1 0.075 0.075">` (20x15x15cm half-extents, randomizable)
- `<site name="box_center">` at box origin for PF sampling
- `<site name="box_left_grip">` and `<site name="box_right_grip">` on opposite faces
- Two `<equality><weld>` constraints: one from `left_wrist_yaw_link` to `carried_box`, one from `right_wrist_yaw_link` to `carried_box`. These keep the box attached to the hands so it can't be dropped. Stiff `solref`/`solimp` initially.
- Contact pairs: `box_geom` vs `floor`, exclude `box_geom` vs hand collision geoms (they're welded)

Also create a mesh variant `scene_mjx_feetonly_mesh_catra.xml` for visualization (same pattern as existing mesh scene).

### 1.2 Carrying default pose
**Modify**: `cat_ppo/envs/g1/constants.py`

Add `DEFAULT_QPOS_CATRA` — same as `DEFAULT_QPOS` but with arm joints adjusted to a carrying pose (arms forward, elbows bent, hands ~30cm in front of torso at chest height). The weld `relpose` in the XML must match this configuration.

Current arm defaults: `[0.2, 0.3, 0, 1.28, 0, 0, 0]` (arms at sides, elbows bent).
Carrying pose estimate: `[0.4, 0.2, 0, 1.0, 0, 0, 0]` (shoulders pitched forward, elbows less bent). Exact values need tuning in MuJoCo viewer.

Add constants:
```python
CATRA_FLAT_TERRAIN_XML = PATH_ASSET / "unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml"
CATRA_MESH_XML = PATH_ASSET / "unitree_g1/scene_mjx_feetonly_mesh_catra.xml"
BOX_SITE = "box_center"
BOX_GEOM = "carried_box_geom"
CATRA_ACTION_JOINT_NAMES = [
    # 12 leg joints (same)
    *ACTION_JOINT_NAMES,
    # 3 waist joints
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 8 arm joints (shoulder + elbow, no wrists)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
# num_act = 23 (12 leg + 3 waist + 8 arm)
```

Update `task_to_xml` dict for `"flat_terrain_catra"` and `"mesh_catra"`.

### 1.3 Verification
- Load the new XML in MuJoCo viewer (`python -m mujoco.viewer --mjcf=<path>`)
- Verify box is attached and stays when robot is posed
- Verify `DEFAULT_QPOS_CATRA` puts hands in a natural carrying position
- Verify weld constraints hold the box firmly

---

## Phase 2: Environment (`env_catra.py`)

### 2.1 Create environment file
**Create**: `cat_ppo/envs/g1/env_catra.py`

Subclass `G1CatEnv`. Override:

**`__init__`**:
- Call `super().__init__(task_type="flat_terrain_catra", ...)`
- Look up `self._box_site_id`, `self._box_geom_id`
- Store `self._catra_action_joint_ids` (indices for the 23 actuated joints)
- Override `self._default_qpos` with `DEFAULT_QPOS_CATRA`

**`reset`**:
- Call parent `reset()` (handles PF field loading, randomization)
- Randomize box mass: modify `self._mjx_model.body_mass[box_body_id]` via domain randomization
- Initialize box position between the palms (computed from the carrying pose)
- Add `info` fields: `"boxgf"`, `"boxbf"`, `"boxdf"`, `"box_pos"`, `"box_vel"`
- Sample box PF fields at `box_center` site position

**`step`**:
- Extend action processing: the 23-dim action updates `motor_targets` for legs + waist + arms (instead of just 12 legs)
- The existing `torque_step` already operates on all 29 joints via PD — only the `action_joint_ids` indexing changes
- Sample box PF fields at `data.site_xpos[self._box_site_id]`
- Track box velocity: `(box_pos - prev_box_pos) / dt`
- Update `info["boxgf"]`, `info["boxbf"]`, `info["boxdf"]`, `info["box_pos"]`, `info["box_vel"]`

**`_get_obs`**:
- Extend `state` with box PF fields in nav frame: `boxgf` (3), `boxbf` (3), `boxdf` (1) = +7 dims
- Extend `privileged_state` with: box PF fields + `box_pos` (3) + `box_vel` (3) = +13 dims
- Note: the arm joint observations are already in `OBS_JOINT_NAMES` (shoulder + elbow). Wrist joints are NOT observed (acceptable for now since wrists aren't actuated).
- New `num_obs` = 162 + 7 = 169 (approx, need exact count after also accounting for arm action history growing from 12 to 23)
- The `last_act` and `motor_targets[action_joint_ids]` in the observation will grow from 12-dim to 23-dim each → +22 more dims
- Revised: `num_obs` ≈ 162 + 22 (bigger action) + 7 (box PF) = 191
- `num_pri` needs similar recalculation

**`_get_reward`**:
- Inherit all existing rewards from G1CatEnv
- Add `"boxgf"`: guidance field alignment for box, using `_re_gf0(boxgf, box_vel, boxdf, ...)`
- Add `"boxdf"`: SDF penalty for box, using `_re_sdf(boxdf)`
- Add `"arm_pose"`: penalize arm deviation from carrying pose (keep arms stable)
- Add `"arm_smoothness"`: action smoothness for arm joints specifically

**`_get_termination`**:
- Inherit all existing termination from G1CatEnv
- Add: `boxdf < -threshold` (box penetrates obstacle)
- Consider: weld constraint violation check (if box detaches from hands)

### 2.2 Action space considerations

The `torque_step` function computes PD for all 29 joints — `motor_targets` is 29-dim where non-actuated joints stay at `_default_qpos`. Extending `action_joint_ids` from 12 to 23 means the action now updates 23 of the 29 targets. The remaining 6 (wrist joints) stay at defaults.

**Action scale**: arm motors have much lower torque limits (25 Nm shoulder, 5 Nm wrist) vs legs (88-139 Nm). Using a single `action_scale=0.5` may be too aggressive for arms. Options:
- (Simple) Use a smaller global `action_scale` like 0.3
- (Better) Use per-joint action scales stored in a vector

For the initial version, start with a single smaller `action_scale` (e.g., 0.3) and tune later.

---

## Phase 3: Config, Registration, and Training Integration

### 3.1 Config function
In `env_catra.py`, define `g1_catra_task_config()`:
- Based on `g1_loco_task_config()` from `env_cat.py`
- Override `num_obs`, `num_pri`, `num_act=23`
- Add reward scales: `boxgf=0.0`, `boxdf=0.0`, `arm_pose=-0.5`, `arm_smoothness=-1e-3`
- Add `box_config`: `mass_range=[0.5, 3.0]`, `size_range=[[0.08, 0.12], [0.06, 0.09], [0.06, 0.09]]` (half-extents)
- Set `task_type="flat_terrain_catra"`

### 3.2 Registration
In `env_catra.py`:
```python
@cat_ppo.registry.register("G1CaTra", "train_env_class")
class G1CaTraEnv(G1CatEnv): ...

cat_ppo.registry.register("G1CaTra", "config")(g1_catra_task_config())
```

### 3.3 Import
**Modify**: `cat_ppo/envs/g1/__init__.py`
Add: `from cat_ppo.envs.g1 import env_catra`

### 3.4 Training CLI extension
**Modify**: `train_ppo.py`
Add `--box` CLI arg to `Args` class, mapping to `boxgf` and `boxdf` reward scales in `_apply_args_to_config`.

### 3.5 Training command
```bash
python train_ppo.py --task G1CaTra --exp_name catra_v1 \
    --ground 1.0 --lateral 1.0 --overhead 1.0 --box 1.0 \
    --obs_path data/assets/TypiObs/<scene>
```

---

## Phase 4: Obstacle Generation (Minimal Changes)

The existing HumanoidPF pipeline generates `sdf.npy`, `bf.npy`, `gf.npy` based on obstacle geometry only — these fields are obstacle-agnostic to the robot/box. The box samples from the same fields as body parts. **No changes needed to obstacle generation for Phase 1.**

The only consideration: the box makes the robot's effective volume larger. Scenes with very tight passages may become impossible. Start training with easier scenes (fewer obstacles, wider passages).

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain_catra.xml` | **Create** | Scene with box body + weld constraints |
| `data/assets/unitree_g1/scene_mjx_feetonly_mesh_catra.xml` | **Create** | Mesh variant for visualization |
| `cat_ppo/envs/g1/constants.py` | **Modify** | Add CATRA XML paths, DEFAULT_QPOS_CATRA, CATRA_ACTION_JOINT_NAMES, BOX_SITE |
| `cat_ppo/envs/g1/env_catra.py` | **Create** | G1CaTraEnv class + config + registration |
| `cat_ppo/envs/g1/__init__.py` | **Modify** | Add `import env_catra` |
| `train_ppo.py` | **Modify** | Add `--box` CLI arg |

---

## Key Reference Files (Read These First)

- `cat_ppo/envs/g1/env_cat.py` — Primary reference; G1CaTraEnv subclasses G1CatEnv
- `cat_ppo/envs/g1/env_loco.py` — Parent class: gait clock, PD controller (`torque_step`), action processing
- `cat_ppo/envs/g1/constants.py` — Joint names, KPs/KDs, torque limits, XML paths, DEFAULT_QPOS
- `data/assets/unitree_g1/g1_mjx_feetonly_torque.xml` — Robot MJCF: palm sites at lines 377/449, hand collision at 378-380/450-452, arm kinematic chain
- `data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain.xml` — Base scene template
- `data/assets/unitree_g1/scene_mjx_feetonly_mesh.xml` — Mesh scene template (shows how obstacle mesh is loaded)
- `cat_ppo/utils/registry.py` — Registration pattern
- `train_ppo.py` — Training entry point, CLI args, `_apply_args_to_config`

---

## Verification Plan

1. **XML sanity check**: Load `scene_mjx_feetonly_flat_terrain_catra.xml` in MuJoCo viewer. Verify box is held, weld is firm, arms are in carrying pose.
2. **Import check**: `python -c "import cat_ppo; print(cat_ppo.registry.get('G1CaTra', 'train_env_class'))"` — should print the class.
3. **Debug training run**: `python train_ppo.py --task G1CaTra --exp_name debug --obs_path data/assets/TypiObs/empty` — verify obs dims match, no shape errors, rewards compute, episodes run.
4. **Full training**: Run with obstacles enabled and box reward scales > 0 on a single scene.

---

## Risks and Open Questions

1. **Weld constraint in MJX**: MJX supports equality constraints but behavior may differ from CPU MuJoCo. Test early.
2. **Carrying pose tuning**: The exact arm joint angles for a natural carrying pose need manual tuning in the viewer. The weld `relpose` must be computed to match.
3. **Action scale for arms**: A single `action_scale` may not work for both legs (high torque) and arms (low torque). May need per-joint scaling.
4. **Observation dim calculation**: The exact `num_obs` and `num_pri` must be carefully counted after all additions. Off-by-one errors cause silent training failures in JAX.
5. **Box mass randomization in MJX**: Modifying `model.body_mass` per-env in MJX requires the domain randomization pipeline. Check if the existing `randomize.py` pattern supports this.

---

## Weld Relaxation Curriculum (single training run)

One training run (~400M steps). Weld stiffness is treated as a domain randomization parameter that shifts over training time (same pattern as `kp_scale`/`kd_scale` in `randomize.py`). At each episode reset, `model.eq_solref[weld_eq_id]` is sampled from a range that shifts based on `training_step`:

| Steps | Weld stiffness | How box is held |
|-------|---------------|-----------------|
| 0–100M | Always rigid | Constraint only. Policy learns locomotion + carrying mass. |
| 100M–250M | Random [rigid, medium] | Box wobbles slightly. Policy learns to stabilize arms. |
| 250M+ | Random [soft, none] | Held by contact + active arm pressing. |

**Requires improved hand geometry** (cupped hand geoms) so that when the weld is soft, there is sufficient contact area to help hold the box.

## Future Improvements (Not in This Plan)

- Relax weld to soft constraint → robot must learn to actively maintain grasp
- Add wrist joint control for finer manipulation
- Box-specific obstacle scenes (e.g., shelves to place box on)
- Privileged variant (G1CaTraPri) for DAgger distillation
- Play/inference env (play_catra.py) for ONNX deployment
