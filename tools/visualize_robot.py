#!/usr/bin/env python3
"""
Interactive visualization of the G1 robot — joints, links, and key HumanoidPF body parts.

Run from the project root:
    python tools/visualize_robot.py

Controls:
    mouse drag  — rotate / pan / zoom (MuJoCo viewer defaults)
    [           — decrease marker size
    ]           — increase marker size
    L           — toggle labels
    Space       — pause / resume joint animation
    Q / Esc     — quit
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from cat_ppo.envs.g1.constants import FEET_ONLY_FLAT_TERRAIN_XML

# ---------------------------------------------------------------------------
# Body parts to highlight and their display colours (RGBA, 0-1)
# These match the 7 body groups used in HumanoidPF (see CLAUDE.md)
# ---------------------------------------------------------------------------
SITE_GROUPS: dict[str, dict] = {
    "head": {
        "sites": ["head"],
        "color": np.array([1.0, 0.2, 0.2, 1.0]),
        "radius": 0.045,
    },
    "feet": {
        "sites": ["left_foot", "right_foot"],
        "color": np.array([0.2, 0.9, 0.2, 1.0]),
        "radius": 0.04,
    },
    "hands": {
        "sites": ["left_palm", "right_palm"],
        "color": np.array([0.2, 0.5, 1.0, 1.0]),
        "radius": 0.04,
    },
    "knees": {
        "sites": ["left_knee", "right_knee"],
        "color": np.array([1.0, 0.55, 0.1, 1.0]),
        "radius": 0.04,
    },
    "shoulders": {
        "sites": ["left_shoulder", "right_shoulder"],
        "color": np.array([0.8, 0.2, 0.9, 1.0]),
        "radius": 0.04,
    },
}

# Key bodies shown as spheres at their frame origin position.
# pelvis gets a box (via two overlapping spheres + large radius) so it stands out.
BODY_GROUPS: dict[str, dict] = {
    "pelvis": {
        "color": np.array([1.0, 0.85, 0.0, 1.0]),   # bright gold, fully opaque
        "radius": 0.11,                               # noticeably large
        "label": "PELVIS (link/root)",
    },
    "torso_link": {
        "color": np.array([0.2, 1.0, 0.9, 0.3]),   # dimmed cyan
        "radius": 0.05,
        "label": "torso_link",
    },
    "waist_yaw_link": {
        "color": np.array([0.6, 0.6, 0.6, 0.25]),  # very faint
        "radius": 0.03,
        "label": "waist_yaw",
    },
}


def _add_sphere(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    radius: float,
    color: np.ndarray,
    label: str = "",
) -> None:
    """Add a labelled sphere to a user scene (no-op if scene is full)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        pos.copy(),
        np.eye(3).flatten(),
        color.astype(np.float32),
    )
    g.label = label
    scene.ngeom += 1


def main() -> None:
    # ------------------------------------------------------------------
    # Load model — use the flat terrain XML so the robot stands on a floor
    # ------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(FEET_ONLY_FLAT_TERRAIN_XML))
    data = mujoco.MjData(model)

    # Stand the robot in the default pose (no keyframes in this XML)
    mujoco.mj_resetData(model, data)
    # Lift the pelvis off the floor to the nominal standing height (set by the XML pos="0 0 0.793")
    data.qpos[2] = 0.793
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Resolve site / body IDs once (missing → -1, silently skipped)
    # ------------------------------------------------------------------
    site_ids: dict[str, dict] = {}
    for group, info in SITE_GROUPS.items():
        ids = []
        for name in info["sites"]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                print(f"[warn] site '{name}' not found in model")
            ids.append((name, sid))
        site_ids[group] = {**info, "ids": ids}

    body_ids: dict[str, dict] = {}
    for name, info in BODY_GROUPS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            print(f"[warn] body '{name}' not found in model")
        body_ids[name] = {**info, "id": bid}

    # ------------------------------------------------------------------
    # Mutable state shared with key_callback closure
    # ------------------------------------------------------------------
    state = {
        "show_labels": True,
        "marker_scale": 1.0,
        "paused": False,
        "anim_t": 0.0,
    }

    def key_callback(key: int) -> None:
        # ASCII: L=76, [=91, ]=93, space=32
        if key == 76:  # L — toggle labels
            state["show_labels"] = not state["show_labels"]
            print(f"Labels: {'on' if state['show_labels'] else 'off'}")
        elif key == 93:  # ] — bigger markers
            state["marker_scale"] = min(state["marker_scale"] * 1.25, 5.0)
        elif key == 91:  # [ — smaller markers
            state["marker_scale"] = max(state["marker_scale"] / 1.25, 0.2)
        elif key == 32:  # Space — pause / resume animation
            state["paused"] = not state["paused"]
            print(f"Animation: {'paused' if state['paused'] else 'running'}")

    # ------------------------------------------------------------------
    # Launch interactive viewer
    # ------------------------------------------------------------------
    print("\nG1 Joint Visualizer")
    print("===================")
    print("Colour legend:")
    for g, info in SITE_GROUPS.items():
        r, gr, b, _ = info["color"]
        names = ", ".join(info["sites"])
        print(f"  {g:12s}: ({r:.1f},{gr:.1f},{b:.1f})  →  {names}")
    for name, info in BODY_GROUPS.items():
        r, gr, b, _ = info["color"]
        print(f"  {name:12s}: ({r:.1f},{gr:.1f},{b:.1f})  [{info['label']}]")
    print()
    print("Keys: L=toggle labels  [/]=marker size  Space=pause  Q/Esc=quit\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Show all geom groups so meshes are visible
        for i in range(mujoco.mjtVisFlag.mjNVISFLAG):
            viewer.opt.flags[i] = 1
        viewer.opt.geomgroup[:] = 1

        while viewer.is_running():
            t0 = time.perf_counter()

            # Gentle idle animation: slowly rock the waist to make the robot
            # feel alive and help users spot the joints. Remove if not wanted.
            if not state["paused"]:
                state["anim_t"] += 0.01
                t = state["anim_t"]
                # Find actuatable DOFs and give them a tiny sinusoidal nudge
                waist_yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")
                if waist_yaw_id >= 0:
                    data.qpos[model.jnt_qposadr[waist_yaw_id]] = 0.15 * np.sin(t * 0.4)
                mujoco.mj_forward(model, data)

            # ----------------------------------------------------------
            # Draw markers into user_scn
            # ----------------------------------------------------------
            viewer.user_scn.ngeom = 0
            scale = state["marker_scale"]
            show = state["show_labels"]

            # Site spheres
            for group, info in site_ids.items():
                for site_name, sid in info["ids"]:
                    if sid < 0:
                        continue
                    pos = data.site_xpos[sid]
                    label = site_name if show else ""
                    _add_sphere(viewer.user_scn, pos, info["radius"] * scale, info["color"], label)

            # Body CoM spheres
            for name, info in body_ids.items():
                bid = info["id"]
                if bid < 0:
                    continue
                pos = data.xpos[bid]
                label = info["label"] if show else ""
                _add_sphere(viewer.user_scn, pos, info["radius"] * scale, info["color"], label)

            viewer.sync()

            # ~60 fps cap
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 1 / 60 - elapsed))


if __name__ == "__main__":
    main()
