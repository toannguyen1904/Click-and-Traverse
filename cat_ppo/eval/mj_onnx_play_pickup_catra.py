"""Chained PICKUP -> CATRA playback in a single continuous episode.

Runs a trained G1Pickup policy live (scene initialized exactly like a pickup eval:
box resting on the support pillar), then hands the resulting "robot holding box"
state off to a trained G1CaTra policy that traverses the obstacle field while
carrying the box.

The handoff is *gated* using the same "is the robot really holding the box"
conditions that cat_ppo.eval.warmstart_generation uses to accept a warm-start state,
but evaluated instantaneously (no lookahead) on the live rollout:
  1. box held:     box_z > pillar_top + box_half_z + box_hold_margin (0.08 m)
  2. hands centered: each palm, projected onto the box side-face plane, within
                     hand_face_tol (0.05 m) of the box center
  3. no NaN in qpos/qvel
On the first step past --min_pickup_steps where all three hold, the live pickup
terminal state (qpos/qvel/box_mass/box_size) is injected as a 1-element warm-start
array into the CATRA play env and its existing warm-start reset path takes over
(box already carried, single-stage transport) -- no new reset logic. If the gate is never reached
within --max_pickup_steps, the run reports failure and exits without running CATRA.

Example:
  python -m cat_ppo.eval.mj_onnx_play_pickup_catra \
    --pickup_exp_name 06302043_G1Pickup_pickup_v1xT004xdataassetsTypiObsempty \
    --catra_task G1CaTra2ADagger \
    --catra_exp_name 06301402_G1CaTra2ADagger_dagger2ablend_v1x...xdataassetsTypiObsempty \
    --obs_path data/assets/TypiObs/empty --record
"""
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as rt
import tyro

import cat_ppo
from cat_ppo.eval.mj_onnx_test import _episode_status, _set_box_noise
from cat_ppo.eval.mj_onnx_play import _read_box_use_inflation
from cat_ppo.envs.g1.env_catra import BOX_QPOS_START


@dataclass
class Args:
    # --- PICKUP (phase 1) ---
    pickup_exp_name: str                 # G1Pickup run whose checkpoint holds policy.onnx
    pickup_onnx_path: str = None         # explicit pickup policy.onnx (overrides checkpoint lookup)
    surface_z: float = None              # fixed support-surface height (m); None = sample from cfg range

    # --- CATRA (phase 2) ---
    catra_task: str = "G1CaTra2ADagger"  # registry task for the carry+traverse policy
    catra_exp_name: str = None           # CATRA run whose checkpoint holds the policy .onnx file(s)
    catra_pri: bool = False              # use the privileged actor obs branch (debug only)

    # --- shared / scene ---
    obs_path: str = "data/assets/TypiObs/empty"  # obstacle field dir (sdf/gf/bf .npy)
    goal_x: float = 1.6                  # base x (m) counted as a completed traversal (success)
    seed: int = 42                       # seeds np.random -> reproducible box DR + noise
    box_inflation: bool = True           # CATRA: observe gf_inflation.npy (read from ckpt config when available)
    box_noise: bool = True               # add box tracking noise to the deployable obs

    # --- handoff gate ---
    min_pickup_steps: int = 150          # earliest step the gate may fire
    max_pickup_steps: int = None         # gate budget; None = pickup episode_length
    hand_face_tol: float = 0.05          # max in-plane palm offset from box face center (m)
    box_hold_margin: float = 0.08        # required box clearance above pillar top + box half-z (m)

    # --- CATRA init-pose perturbation (optional) ---
    init_pos_offset: float = 0.0         # random xy offset (m) applied to robot+box at handoff
    init_ang_offset: float = 0.0         # random yaw offset (deg) applied to robot+box at handoff

    # --- recording / camera ---
    record: bool = False                 # save one continuous rollout video (headless)
    video_path: str = None               # output path; default <catra_ckpt>/rollout_pickup_catra.mp4
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 50                  # matches ctrl_dt=0.02 s (50 Hz)
    cam_distance: float = 5.0
    cam_azimuth: float = 135.0
    cam_elevation: float = -20.0
    cam_lookat: str = None               # "x,y,z"; default tracks the robot base


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec (3,) by wxyz MuJoCo quaternion quat (4,). Matches warmstart_generation."""
    w = quat[0]
    qvec = quat[1:4]
    t = 2.0 * np.cross(qvec, vec)
    return vec + w * t + np.cross(qvec, t)


def _pickup_gate(env, info, hand_face_tol: float, box_hold_margin: float):
    """Instantaneous 'robot is holding the box' check on the live pickup env.

    Mirrors the accept conditions in cat_ppo.eval.warmstart_generation (box held above the
    pillar with margin + both palms centered on the box side faces + no NaN), evaluated on the
    current mj_data instead of a future snapshot. Returns (ok, diagnostics dict)."""
    d = env.mj_data
    box_z = float(d.qpos[BOX_QPOS_START + 2])
    box_half_z = float(info["box_size"][2])
    pillar_top = float(info["surface_z"]) + float(info["support_half_z"])
    held = box_z > pillar_top + box_half_z + box_hold_margin

    box_pos = d.xpos[env._box_body_id]
    box_quat = d.xquat[env._box_body_id]  # wxyz
    y_axis = _quat_rotate(box_quat, np.array([0.0, 1.0, 0.0]))
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

    def _proj_offset(site_id):
        v = d.site_xpos[site_id] - box_pos
        return float(np.linalg.norm(v - np.dot(v, y_axis) * y_axis))

    lo = _proj_offset(env._hands_site_id[0])
    ro = _proj_offset(env._hands_site_id[1])
    centered = (lo < hand_face_tol) and (ro < hand_face_tol)

    no_nan = not (np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)))
    ok = held and centered and no_nan
    diag = dict(box_z=box_z, box_clear=box_z - (pillar_top + box_half_z),
                lhand_off=lo, rhand_off=ro, held=held, centered=centered, no_nan=no_nan)
    return ok, diag


def _make_cam(args: Args):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.cam_distance
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation
    if args.cam_lookat is not None:
        cam.lookat[:] = [float(v) for v in args.cam_lookat.split(",")]
    return cam


def _render(renderer, cam, writer, env, track_base: bool):
    if renderer is None:
        return
    if track_base:
        cam.lookat[:] = env.mj_data.qpos[:3]
    renderer.update_scene(env.mj_data, camera=cam)
    writer.append_data(renderer.render())


def play(args: Args):
    np.random.seed(args.seed)
    output_names = ["continuous_actions"]

    # ------------------------------------------------------------------ PICKUP env + policy
    pickup_cfg = cat_ppo.registry.get("G1Pickup", "config").env_config
    pickup_cfg.pf_config.path = args.obs_path
    _set_box_noise(pickup_cfg, args.box_noise)
    pickup_env_class = cat_ppo.registry.get("G1Pickup", "play_env_class")
    pickup_env = pickup_env_class(
        task_type=pickup_cfg.task_type, config=pickup_cfg,
        headless=args.record, surface_z=args.surface_z,
    )

    if args.pickup_onnx_path is not None:
        pickup_onnx = args.pickup_onnx_path
    else:
        pickup_onnx = cat_ppo.get_latest_ckpt(args.pickup_exp_name) / "policy.onnx"
    pickup_policy = rt.InferenceSession(str(pickup_onnx), providers=["CPUExecutionProvider"])

    # ------------------------------------------------------------------ CATRA env + policy(ies)
    catra_cfg = cat_ppo.registry.get(args.catra_task, "config").env_config
    catra_cfg.pf_config.path = args.obs_path
    if hasattr(catra_cfg, "box_use_inflation"):
        catra_cfg.box_use_inflation = _read_box_use_inflation(args.catra_exp_name, args.box_inflation)
        print(f"[pickup_catra] box_use_inflation = {catra_cfg.box_use_inflation}")
    # We hand off a state that is already holding the box; CATRA is single-stage transport.
    _set_box_noise(catra_cfg, args.box_noise)
    catra_env_class = cat_ppo.registry.get(args.catra_task, "play_env_class")
    catra_env = catra_env_class(task_type=catra_cfg.task_type, config=catra_cfg, headless=args.record)
    catra_env.pri = args.catra_pri

    is_2a = hasattr(catra_cfg, "num_act_lower")
    catra_ckpt = cat_ppo.get_latest_ckpt(args.catra_exp_name)
    if is_2a:
        catra_lower = rt.InferenceSession(str(catra_ckpt / "policy_lower.onnx"), providers=["CPUExecutionProvider"])
        catra_upper = rt.InferenceSession(str(catra_ckpt / "policy_upper.onnx"), providers=["CPUExecutionProvider"])
    else:
        catra_policy = rt.InferenceSession(str(catra_ckpt / "policy.onnx"), providers=["CPUExecutionProvider"])

    # ------------------------------------------------------------------ recorder
    renderer_pickup = renderer_catra = writer = cam = None
    if args.record:
        cam = _make_cam(args)
        renderer_pickup = mujoco.Renderer(pickup_env.mj_model, height=args.video_height, width=args.video_width)
        if args.video_path is not None:
            video_path = args.video_path
        else:
            video_path = str(catra_ckpt / "rollout_pickup_catra.mp4")
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=args.video_fps)
        print(f"[pickup_catra] recording video to: {video_path}")
    track_base = args.cam_lookat is None

    max_pickup_steps = args.max_pickup_steps
    if max_pickup_steps is None:
        max_pickup_steps = int(getattr(pickup_cfg, "episode_length", 1000))

    try:
        # ============================================================ PHASE 1: PICKUP
        state = pickup_env.reset(surface_z=args.surface_z)
        print(f"[pickup_catra] PICKUP start: surface_z={state.info['surface_z']:.3f} "
              f"box_half=({state.info['box_size'][0]:.3f},{state.info['box_size'][1]:.3f},"
              f"{state.info['box_size'][2]:.3f}) box_mass={state.info['box_mass']:.3f}")

        handoff_step = None
        while True:
            obs = state.obs["state"].reshape(1, -1).astype(np.float32)
            action = pickup_policy.run(output_names, {"obs": obs})[0][0]
            state = pickup_env.step(state, action)
            _render(renderer_pickup, cam, writer, pickup_env, track_base)

            step = int(state.info["step"])

            # Early failure: robot toppled or box fell off the pillar.
            if float(pickup_env.get_gravity("pelvis")[2]) < 0.0:
                print(f"[pickup_catra] PICKUP FAILED (fell) at step {step}")
                return
            box_z = float(pickup_env.mj_data.xpos[pickup_env._box_body_id][2])
            if box_z < float(pickup_cfg.box_drop_threshold):
                print(f"[pickup_catra] PICKUP FAILED (box dropped, box_z={box_z:.3f}) at step {step}")
                return

            # Instantaneous handoff gate.
            if step >= args.min_pickup_steps:
                ok, diag = _pickup_gate(pickup_env, state.info, args.hand_face_tol, args.box_hold_margin)
                if ok:
                    handoff_step = step
                    print(f"[pickup_catra] gate reached at step {step}: box_clear={diag['box_clear']:.3f} "
                          f"lhand_off={diag['lhand_off']:.3f} rhand_off={diag['rhand_off']:.3f}")
                    break

            if step >= max_pickup_steps:
                ok, diag = _pickup_gate(pickup_env, state.info, args.hand_face_tol, args.box_hold_margin)
                print(f"[pickup_catra] PICKUP FAILED (gate not reached in {max_pickup_steps} steps): "
                      f"held={diag['held']} centered={diag['centered']} "
                      f"box_clear={diag['box_clear']:.3f} "
                      f"lhand_off={diag['lhand_off']:.3f} rhand_off={diag['rhand_off']:.3f}")
                return

        # ============================================================ HANDOFF
        nq = catra_env.mj_model.nq
        nv = catra_env.mj_model.nv
        ho_qpos = pickup_env.mj_data.qpos[:nq].copy()
        ho_qvel = pickup_env.mj_data.qvel[:nv].copy()
        ho_box_size = pickup_env.mj_model.geom_size[pickup_env._box_geom_id].copy()
        ho_box_mass = float(pickup_env.mj_model.body_mass[pickup_env._box_body_id])

        # Inject a 1-element warm-start batch so PlayG1CaTraEnv.reset() takes its warm-start
        # branch (loads this exact holding-box state; single-stage transport with the box carried).
        catra_env._ws_qpos = ho_qpos[None]
        catra_env._ws_qvel = ho_qvel[None]
        catra_env._ws_box_size = ho_box_size[None]
        catra_env._ws_box_mass = np.array([ho_box_mass])

        # Avoid a second stale viewer window during phase 2 (interactive mode only).
        if not args.record and getattr(pickup_env, "viewer", None) is not None:
            pickup_env.viewer.close()

        state = catra_env.reset(
            warmstart_idx=0,
            pos_offset=args.init_pos_offset,
            ang_offset_deg=args.init_ang_offset,
        )
        if args.record:
            renderer_catra = mujoco.Renderer(catra_env.mj_model, height=args.video_height, width=args.video_width)
        print(f"[pickup_catra] handoff complete at pickup step {handoff_step}; running CATRA "
              f"({'2-agent' if is_2a else '1-agent'}, pri={args.catra_pri})")

        # ============================================================ PHASE 2: CATRA
        max_steps = int(getattr(catra_cfg, "episode_length", 10 ** 9))
        while True:
            obs = state.obs["state"].reshape(1, -1).astype(np.float32)
            if is_2a:
                a_lower = catra_lower.run(output_names, {"obs": obs})[0][0]
                a_upper = catra_upper.run(output_names, {"obs": obs})[0][0]
                action = np.concatenate([a_lower, a_upper])  # lower-first, matches action ordering
            else:
                action = catra_policy.run(output_names, {"obs": obs})[0][0]
            state = catra_env.step(state, action)
            _render(renderer_catra, cam, writer, catra_env, track_base)

            status, base_x = _episode_status(catra_env, state, catra_cfg, args.goal_x, max_steps)
            if status is not None:
                outcome = "SUCCESS" if status == "success" else f"FAIL ({status})"
                print(f"[pickup_catra] CATRA episode ended at step {int(state.info['step'])}: "
                      f"{outcome}  base_x={base_x:.3f}  goal_x={args.goal_x}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
            print(f"[pickup_catra] video saved to: {video_path}")
        for r in (renderer_pickup, renderer_catra):
            if r is not None:
                r.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
