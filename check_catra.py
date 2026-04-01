#!/usr/bin/env python3
"""Quick sanity-check for the CaTra initial state: robot, box, and weld constraints.

Loads the CaTra scene XML, poses the robot in DEFAULT_QPOS_CATRA, settles the weld
constraints for a short simulation, prints key state info, then opens the interactive
MuJoCo viewer so you can inspect the carrying pose visually.

Usage (from repo root):
    source .venv/bin/activate && source .env
    python check_catra.py
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cat_ppo.envs.g1.constants import (
    CATRA_FLAT_TERRAIN_XML,
    DEFAULT_QPOS_CATRA,
    BOX_SITE,
    BOX_GEOM,
    CATRA_ACTION_JOINT_NAMES,
)

NUM_ROBOT_JOINTS = 29
# Box freejoint initial pose: placed roughly between the hands; welds will pull it tight.
BOX_DEFAULT_QPOS = np.array([0.35, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
# Full 43-dim initial qpos: [0:36] robot | [36:43] box freejoint
INIT_QPOS = np.concatenate([DEFAULT_QPOS_CATRA, BOX_DEFAULT_QPOS])

W = 60
SEP = "─" * W


def header(title: str) -> None:
    pad = max(0, W - len(title) - 2)
    left = pad // 2
    right = pad - left
    print(f"\n{'─' * left} {title} {'─' * right}")


def main() -> None:
    print(SEP)
    print("  CaTra — Initial State Check (robot + box + weld constraints)")
    print(SEP)

    # ── Load model ──────────────────────────────────────────────────────────
    xml_path = str(CATRA_FLAT_TERRAIN_XML)
    print(f"\nXML: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # ── Model dimensions ────────────────────────────────────────────────────
    header("Model Dimensions")
    checks = [
        ("nq (qpos DOF)", model.nq, 43, "7 root + 29 robot + 7 box"),
        ("nv (qvel DOF)", model.nv, 41, "6 root + 29 robot + 6 box"),
        ("nu (actuators)", model.nu, 29, "29 robot joints"),
        ("neq (constraints)", model.neq, 2, "2 weld (left + right wrist)"),
    ]
    for label, val, expected, note in checks:
        ok = "✓" if val == expected else "✗ UNEXPECTED"
        print(f"  {label:<22} {val:>4}   {ok}  ({note})")
    print(f"  {'nbody':<22} {model.nbody:>4}")
    print(f"  {'nsite':<22} {model.nsite:>4}")

    # ── Set initial pose and run forward kinematics ─────────────────────────
    data.qpos[:] = INIT_QPOS
    data.ctrl[:] = DEFAULT_QPOS_CATRA[7:7 + NUM_ROBOT_JOINTS]
    mujoco.mj_forward(model, data)

    # ── Joint positions ─────────────────────────────────────────────────────
    header("Robot Joint Positions (DEFAULT_QPOS_CATRA)")
    joint_labels = [
        "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
        "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "L_shldr_pitch", "L_shldr_roll", "L_shldr_yaw", "L_elbow",
        "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
        "R_shldr_pitch", "R_shldr_roll", "R_shldr_yaw", "R_elbow",
        "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
    ]
    robot_qpos = data.qpos[7:7 + NUM_ROBOT_JOINTS]
    # Print in columns (two per row)
    for i in range(0, len(joint_labels), 2):
        left_s = f"  {joint_labels[i]:<20} {robot_qpos[i]:+.4f} rad"
        if i + 1 < len(joint_labels):
            right_s = f"    {joint_labels[i+1]:<20} {robot_qpos[i+1]:+.4f} rad"
        else:
            right_s = ""
        print(left_s + right_s)

    # ── Phase 3 holding strategy ─────────────────────────────────────────────
    header("Phase 3: Contact-Only Holding")
    if model.neq == 0:
        print("  neq=0  No equality constraints (weld removed)  ✓ MJX-compatible")
    else:
        print(f"  WARNING: neq={model.neq} — unexpected constraints present")
    print(f"  Box holding: contact forces + friction only (no weld)")
    print(f"  Reset strategy: box placed at palm midpoint via FK")
    # Compute expected box position from current data
    for site_name in ["left_palm", "right_palm"]:
        try:
            sid = model.site(site_name).id
            print(f"  {site_name:<14}: {data.site_xpos[sid]}")
        except Exception:
            print(f"  {site_name:<14}: (site not found)")
    try:
        lp = data.site_xpos[model.site("left_palm").id]
        rp = data.site_xpos[model.site("right_palm").id]
        midpoint = (lp + rp) / 2.0
        print(f"  Palm midpoint   : {midpoint}  ← box init target")
        print(f"  Palm separation : {np.linalg.norm(lp - rp):.4f} m  (box width=0.20 m)")
    except Exception:
        pass

    # ── Initial hand and box positions ───────────────────────────────────────
    header("Site Positions (before settling)")
    box_site_id = model.site(BOX_SITE).id
    print(f"  box_center  : {data.site_xpos[box_site_id]}")
    for site_name in ["left_palm", "right_palm"]:
        try:
            sid = model.site(site_name).id
            pos = data.site_xpos[sid]
            dist = np.linalg.norm(pos - data.site_xpos[box_site_id])
            print(f"  {site_name:<12}: {pos}  (dist to box: {dist:.4f} m)")
        except Exception:
            print(f"  {site_name:<12}: (site not found in XML)")

    # ── Settle weld constraints ──────────────────────────────────────────────
    n_settle = 500
    header(f"Settling ({n_settle} steps @ dt={model.opt.timestep:.4f}s = {n_settle * model.opt.timestep * 1000:.0f} ms sim)")
    for _ in range(n_settle):
        mujoco.mj_step(model, data)

    # ── Settled state ────────────────────────────────────────────────────────
    header("Settled State")
    root_pos = data.qpos[:3]
    root_z = root_pos[2]
    print(f"  Root (pelvis) height : {root_z:.4f} m   {'OK' if 0.6 < root_z < 1.1 else '⚠ unexpected'}")

    box_qpos_settled = data.qpos[36:43]
    box_site_pos = data.site_xpos[box_site_id]
    box_vel_lin = data.qvel[35:38]
    print(f"\n  Box position  (qpos) : {box_qpos_settled[:3]}")
    print(f"  Box quat (wxyz)      : {box_qpos_settled[3:]}")
    print(f"  Box site xpos        : {box_site_pos}")
    print(f"  Box linear vel       : {box_vel_lin}  (should be ~0 if settled)")

    header("Box-Hand Distance After Settling")
    print("  (Phase 3: box held by contact only — may drift if arms are not actively pressing)")
    for site_name in ["left_palm", "right_palm"]:
        try:
            sid = model.site(site_name).id
            dist = np.linalg.norm(data.site_xpos[sid] - box_site_pos)
            status = "held" if dist < 0.15 else "dropped (expected without trained policy)"
            print(f"  {site_name:<14} → box center : {dist:.4f} m   {status}")
        except Exception:
            pass

    # ── Action space summary ─────────────────────────────────────────────────
    header("Action Space (23-dim CATRA_ACTION_JOINT_NAMES)")
    for idx, name in enumerate(CATRA_ACTION_JOINT_NAMES):
        group = "leg  " if idx < 12 else ("waist" if idx < 15 else "arm  ")
        print(f"  [{idx:02d}] {group}  {name}")

    # ── Launch viewer ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Launching MuJoCo interactive viewer ...")
    print("  Controls: [Space] pause | [Ctrl+R] reset | [Scroll] zoom | [Q/Esc] quit")
    print(f"{SEP}\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 140
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            # Real-time pacing
            elapsed = time.perf_counter() - step_start
            time.sleep(max(0.0, model.opt.timestep - elapsed))


if __name__ == "__main__":
    main()
