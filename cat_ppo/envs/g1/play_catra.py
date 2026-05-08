"""CPU-based inference environment for the two-stage G1CaTra policy.

Stage 1 (steps 0 -> stage1_steps - 1):  zero command, robot picks up box.
Stage 2 (steps stage1_steps -> stage1_steps+999): PF-derived command, robot walks carrying box.

Mirrors G1CaTraEnv._get_obs (195-dim state) for ONNX policy playback.
"""
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

import cat_ppo
from cat_ppo.envs.g1 import constants as consts
from cat_ppo.envs.g1.env_catra import (
    BOX_QPOS_START,
    BOX_QVEL_START,
    SUPPORT_QPOS_START,
    NUM_ROBOT_JOINTS,
)
from cat_ppo.envs.g1.play_cat import (
    BaseEnv,
    State,
    set_scene_for_xml,
    base2navi_transform,
    world_to_navi_vel,
    delay_body_pos,
    noisy_rootpose,
    EPS,
)

_STOP_CMD = np.zeros(4)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


_BOX_CORNER_SIGNS = np.array([
    [-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.], [-1.,  1.,  1.],
    [ 1., -1., -1.], [ 1., -1.,  1.], [ 1.,  1., -1.], [ 1.,  1.,  1.],
], dtype=np.float32)


@cat_ppo.registry.register("G1CaTra", "play_env_class")
class PlayG1CaTraEnv(BaseEnv):
    """CPU inference env for two-stage G1CaTra (251-dim state)."""

    def __init__(
        self,
        task_type: str = "flat_terrain_catra",
        config=None,
        dt: float = 0.02,
        sim_dt: float = 0.002,
        headless: bool = False,
    ):
        xml_path = consts.CATRA_MESH_XML
        tmp_xml = set_scene_for_xml(xml_path, config.pf_config.path)
        self.mj_model = mujoco.MjModel.from_xml_path(tmp_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = sim_dt
        self.headless = headless
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self._config = config
        self.dt = dt
        self.sim_dt = sim_dt

        # PF fields
        pf_path = config.pf_config.path
        self.dx = config.pf_config.dx
        self.sdf = np.load(f"{pf_path}/sdf.npy")[..., None]
        self.bf  = np.load(f"{pf_path}/bf.npy")
        self.gf  = np.load(f"{pf_path}/gf.npy")
        self.pf_origin = np.array(config.pf_config.origin, dtype=np.float32)
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape

        self.gait_freq  = 1.5
        self.foot_height = 0.07
        self._post_init()

        ws_path = getattr(config, "warmstart_states_path", None)
        self._ws_qpos = None
        self._ws_qvel = None
        self._ws_box_mass = None
        self._ws_box_size = None
        if ws_path:
            npz = np.load(ws_path)
            self._ws_qpos      = npz["qpos"]       # (N, nq)
            self._ws_qvel      = npz["qvel"]        # (N, nv)
            self._ws_box_mass  = npz["box_mass"]    # (N,)
            self._ws_box_size  = npz["box_size"]    # (N, 3)
            print(f"[PlayG1CaTraEnv] Loaded {self._ws_qpos.shape[0]} warm-start states from {ws_path}")

    def _post_init(self):
        self._default_qpos = np.array(consts.DEFAULT_QPOS_CATRA[7:7 + NUM_ROBOT_JOINTS])

        self.action_joint_names = consts.CATRA_ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = np.array([
            self.mj_model.actuator(name).id for name in self.action_joint_names
        ])

        lowers, uppers = self.mj_model.jnt_range[1:1 + NUM_ROBOT_JOINTS].T
        c = (lowers + uppers) / 2
        r = uppers - lowers
        factor = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * factor
        self._soft_uppers = c + 0.5 * r * factor

        self._pelvis_imu_site_id = self.mj_model.site("imu_in_pelvis").id
        self._torso_imu_site_id  = self.mj_model.site("imu_in_torso").id
        self._head_site_id  = self.mj_model.site("head").id
        self._feet_site_id  = np.array([self.mj_model.site(s).id for s in consts.FEET_SITES])
        self._hands_site_id = np.array([self.mj_model.site(s).id for s in consts.HAND_SITES])
        self._knees_site_id = np.array([self.mj_model.site(s).id for s in consts.KNEE_SITES])
        self._shlds_site_id = np.array([self.mj_model.site(s).id for s in consts.SHOULDER_SITES])
        self._feet_geom_id  = np.array([self.mj_model.geom(g).id for g in consts.FEET_GEOMS])
        self._floor_geom_id = self.mj_model.geom("floor").id

        self._pelvis_body_id      = self.mj_model.body("pelvis").id
        self._box_body_id         = self.mj_model.body("carried_box").id
        self._box_geom_id         = self.mj_model.geom("box_geom").id
        self._box_support_geom_id = self.mj_model.geom("box_support_col").id

        self._init_phase   = np.array([0.0, np.pi])
        self._stance_phase = np.array([0.0, 0.0])

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)

    def reset(self, warmstart_idx: int = -1):
        """Reset to default pose (box on pillar) or to a saved warm-start state."""
        if self._ws_qpos is not None:
            # --- Warm-start path: load saved holding-box state ---
            N = self._ws_qpos.shape[0]
            idx = np.random.randint(0, N) if warmstart_idx < 0 else int(warmstart_idx) % N
            self.mj_model.geom_size[self._box_geom_id] = self._ws_box_size[idx]
            self.mj_model.body_mass[self._box_body_id] = self._ws_box_mass[idx]
            self.mj_data.qpos[:] = self._ws_qpos[idx]
            self.mj_data.qvel[:] = self._ws_qvel[idx]
        else:
            # --- Default path: robot standing, box on pillar ---
            qpos = np.array(consts.DEFAULT_QPOS_CATRA, dtype=np.float64)
            self.mj_data.qpos[:len(qpos)] = qpos
            self.mj_data.qvel[:] = 0.0

            surface_z = float(self._config.box_surface_height_range[0])
            box_half_z = float(self.mj_model.geom_size[self._box_geom_id][2])
            support_half_z = float(self.mj_model.geom_size[self._box_support_geom_id][2])
            box_z = surface_z + support_half_z + box_half_z

            w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
            forward_xy = np.array([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z)])
            box_xy = qpos[:2] + 0.3 * forward_xy
            box_quat = np.array([1.0, 0.0, 0.0, 0.0])

            self.mj_data.qpos[BOX_QPOS_START:BOX_QPOS_START + 3] = [box_xy[0], box_xy[1], box_z]
            self.mj_data.qpos[BOX_QPOS_START + 3:BOX_QPOS_START + 7] = box_quat
            self.mj_data.qpos[SUPPORT_QPOS_START:SUPPORT_QPOS_START + 3] = [box_xy[0], box_xy[1], surface_z]
            self.mj_data.qpos[SUPPORT_QPOS_START + 3:SUPPORT_QPOS_START + 7] = box_quat

        mujoco.mj_forward(self.mj_model, self.mj_data)
        if not self.headless:
            self.viewer.sync()

        phase_dt = 2 * np.pi * self.dt * self.gait_freq

        head_pos  = self.mj_data.site_xpos[self._head_site_id].copy()
        feet_pos  = self.mj_data.site_xpos[self._feet_site_id].copy()
        hands_pos = self.mj_data.site_xpos[self._hands_site_id].copy()
        knees_pos = self.mj_data.site_xpos[self._knees_site_id].copy()
        shlds_pos = self.mj_data.site_xpos[self._shlds_site_id].copy()
        pelv_pos  = self.mj_data.site_xpos[self._pelvis_imu_site_id].copy()
        tors_pos  = self.mj_data.site_xpos[self._torso_imu_site_id].copy()

        all_poses = np.concatenate([
            head_pos.reshape(1, -1), pelv_pos.reshape(1, -1), tors_pos.reshape(1, -1),
            feet_pos, hands_pos, knees_pos, shlds_pos,
        ], axis=0)
        all_gf = self.sample_field(self.gf, all_poses)
        all_bf = self.sample_field(self.bf, all_poses)
        all_df = self.sample_field(self.sdf, all_poses)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = np.split(all_df, [1, 2, 3, 5, 7, 9], axis=0)

        box_size = self.mj_model.geom_size[self._box_geom_id].copy()
        box_corners = self._box_corners_world(box_size)
        boxgf = self.sample_field(self.gf,  box_corners)
        boxbf = self.sample_field(self.bf,  box_corners)
        boxdf = self.sample_field(self.sdf, box_corners)

        if self._ws_qpos is not None:
            init_step = self._config.stage1_steps
            init_motor_targets = self.mj_data.qpos[7:7 + NUM_ROBOT_JOINTS].copy()
        else:
            init_step = 0
            init_motor_targets = self._default_qpos.copy()

        info = {
            "step": init_step,
            "command": np.zeros(4),
            "last_act": np.zeros(self.action_size),
            "motor_targets": init_motor_targets,
            "phase_dt": phase_dt,
            "phase": self._init_phase.copy(),
            "foot_height": self.foot_height,
            "gait_mask": np.zeros(2),
            "timestamp_move2stop": 100,
            "last_flags": np.zeros(2),
            "odom_delay": self.mj_data.qpos[:7].copy(),
            "box_size": box_size,
            "headgf": headgf, "headbf": headbf, "headdf": headdf,
            "pelvgf": pelvgf, "pelvbf": pelvbf, "pelvdf": pelvdf,
            "torsgf": torsgf, "torsbf": torsbf, "torsdf": torsdf,
            "feetgf": feetgf, "feetbf": feetbf, "feetdf": feetdf,
            "handsgf": handsgf, "handsbf": handsbf, "handsdf": handsdf,
            "kneesgf": kneesgf, "kneesbf": kneesbf, "kneesdf": kneesdf,
            "shldsgf": shldsgf, "shldsbf": shldsbf, "shldsdf": shldsdf,
            "boxgf": boxgf, "boxbf": boxbf, "boxdf": boxdf,
            "head_pos": head_pos, "head_vel": np.zeros(3),
            "feet_pos": feet_pos, "feet_vel": np.zeros_like(feet_pos),
            "hands_pos": hands_pos, "hands_vel": np.zeros_like(hands_pos),
        }

        obs = self.get_obs(info)
        return State(info, obs)

    def step(self, state: State, action: np.ndarray) -> State:
        """Apply PD control; gate command to zero in stage 1 (step < 100)."""
        lower_motor_targets = np.clip(
            state.info["motor_targets"][self.action_joint_ids] + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()
        motor_targets[self.action_joint_ids] = lower_motor_targets
        state.info["motor_targets"] = motor_targets.copy()

        for _ in range(int(self.dt / self.sim_dt)):
            torques = (
                consts.KPs * (motor_targets - self.mj_data.qpos[7:7 + NUM_ROBOT_JOINTS])
                + consts.KDs * (-self.mj_data.qvel[6:6 + NUM_ROBOT_JOINTS])
            )
            self.mj_data.ctrl[:] = torques
            mujoco.mj_step(self.mj_model, self.mj_data)

        if not self.headless:
            self.viewer.sync()
            time.sleep(self.dt)

        head_pos  = self.mj_data.site_xpos[self._head_site_id].copy()
        head_vel  = (head_pos - state.info["head_pos"]) / self.dt
        feet_pos  = self.mj_data.site_xpos[self._feet_site_id].copy()
        feet_vel  = (feet_pos - state.info["feet_pos"]) / self.dt
        hands_pos = self.mj_data.site_xpos[self._hands_site_id].copy()
        hands_vel = (hands_pos - state.info["hands_pos"]) / self.dt
        knees_pos = self.mj_data.site_xpos[self._knees_site_id].copy()
        shlds_pos = self.mj_data.site_xpos[self._shlds_site_id].copy()
        pelv_pos  = self.mj_data.site_xpos[self._pelvis_imu_site_id].copy()
        tors_pos  = self.mj_data.site_xpos[self._torso_imu_site_id].copy()

        all_poses = np.concatenate([
            head_pos.reshape(1, -1), pelv_pos.reshape(1, -1), tors_pos.reshape(1, -1),
            feet_pos, hands_pos, knees_pos, shlds_pos,
        ], axis=0)

        # Odometry delay simulation (every step)
        odom_delay = self.mj_data.qpos[:7].copy()
        p_gt, q_gt = self.mj_data.qpos[:3], self.mj_data.qpos[3:7]
        p_odom, q_odom = odom_delay[:3], odom_delay[3:7]
        all_poses_delay = delay_body_pos(p_gt, q_gt, p_odom, q_odom, all_poses)

        all_gf = self.sample_field(self.gf, all_poses_delay)
        all_bf = self.sample_field(self.bf, all_poses_delay)
        all_df = self.sample_field(self.sdf, all_poses_delay)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = np.split(all_df, [1, 2, 3, 5, 7, 9], axis=0)

        # PF-derived command (4-dim); gated to zero in stage 1
        cmd_pf = self._compute_cmd_4d(
            pelvgf.reshape(-1),
            np.concatenate([headgf, feetgf, handsgf], axis=0),
            np.concatenate([headbf, feetbf, handsbf], axis=0),
        )
        step = state.info["step"]
        command = np.zeros(4) if step < self._config.stage1_steps else cmd_pf

        # Gait update
        self._update_phase(state, command)

        move_flag_val = state.info["last_flags"][1]
        all_gf = all_gf * (move_flag_val > 0.5) / (np.linalg.norm(all_gf, axis=-1, keepdims=True) + EPS)
        all_bf = all_bf / (np.linalg.norm(all_bf, axis=-1, keepdims=True) + EPS)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1, 2, 3, 5, 7, 9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1, 2, 3, 5, 7, 9], axis=0)

        # Box corner PF (delayed, then move_flag-normalized)
        box_corners = self._box_corners_world(state.info["box_size"])
        box_corners_delay = delay_body_pos(p_gt, q_gt, p_odom, q_odom, box_corners)
        boxgf = self.sample_field(self.gf,  box_corners_delay)
        boxbf = self.sample_field(self.bf,  box_corners_delay)
        boxdf = self.sample_field(self.sdf, box_corners_delay)
        boxgf = boxgf * (move_flag_val > 0.5) / (np.linalg.norm(boxgf, axis=-1, keepdims=True) + EPS)
        boxbf = boxbf / (np.linalg.norm(boxbf, axis=-1, keepdims=True) + EPS)

        state.info.update({
            "step": step + 1,
            "command": command,
            "last_act": action.copy(),
            "odom_delay": odom_delay,
            "headgf": headgf, "headbf": headbf, "headdf": headdf,
            "pelvgf": pelvgf, "pelvbf": pelvbf, "pelvdf": pelvdf,
            "torsgf": torsgf, "torsbf": torsbf, "torsdf": torsdf,
            "feetgf": feetgf, "feetbf": feetbf, "feetdf": feetdf,
            "handsgf": handsgf, "handsbf": handsbf, "handsdf": handsdf,
            "kneesgf": kneesgf, "kneesbf": kneesbf, "kneesdf": kneesdf,
            "shldsgf": shldsgf, "shldsbf": shldsbf, "shldsdf": shldsdf,
            "boxgf": boxgf, "boxbf": boxbf, "boxdf": boxdf,
            "head_pos": head_pos, "head_vel": head_vel,
            "feet_pos": feet_pos, "feet_vel": feet_vel,
            "hands_pos": hands_pos, "hands_vel": hands_vel,
        })

        obs = self.get_obs(state.info)
        return State(state.info, obs)

    def get_obs(self, info: dict) -> dict:
        """Build 251-dim deployable state (matches G1CaTraEnv._get_obs)."""
        nl = self._config.noise_config.level
        ns = self._config.noise_config.scales

        gyro_pelvis = self.get_gyro("pelvis")
        pelvis_xmat = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        gvec_pelvis = pelvis_xmat.T @ np.array([0., 0., -1.])
        joint_angles = self.mj_data.qpos[7:7 + NUM_ROBOT_JOINTS]
        joint_vel    = self.mj_data.qvel[6:6 + NUM_ROBOT_JOINTS]

        noisy_gyro = gyro_pelvis + (2 * np.random.rand(3) - 1) * nl * ns.gyro
        noisy_gvec = gvec_pelvis + (2 * np.random.rand(3) - 1) * nl * ns.gravity
        noisy_ja   = joint_angles + (2 * np.random.rand(len(joint_angles)) - 1) * nl * ns.joint_pos
        noisy_jv   = joint_vel    + (2 * np.random.rand(len(joint_vel))    - 1) * nl * ns.joint_vel

        # Box pose in pelvis frame
        pelvis_pos   = self.mj_data.xpos[self._pelvis_body_id]
        pelvis_xquat = self.mj_data.xquat[self._pelvis_body_id]
        box_pos_world  = self.mj_data.xpos[self._box_body_id]
        box_quat_world = self.mj_data.xquat[self._box_body_id]
        box_pos_local  = pelvis_xmat.T @ (box_pos_world - pelvis_pos)
        pelvis_conj    = pelvis_xquat * np.array([1., -1., -1., -1.])
        box_quat_local = _quat_mul(pelvis_conj, box_quat_world)
        box_size = info["box_size"]

        stage_flag = np.array([1.0]) if info["step"] >= self._config.stage1_steps else np.array([0.0])

        # Nav-frame transform
        navi2world_rot  = base2navi_transform(pelvis_xmat)
        navi2world_pose = np.eye(4)
        navi2world_pose[:3, :3] = navi2world_rot
        navi2world_pose[:2, 3]  = self.mj_data.site_xpos[self._pelvis_imu_site_id][:2]
        navi2world_pose[2, 3]   = 0.75

        # Transform PF to nav frame
        headgf  = world_to_navi_vel(navi2world_pose, info["headgf"].reshape(-1, 3))
        headbf  = world_to_navi_vel(navi2world_pose, info["headbf"].reshape(-1, 3))
        pelvgf  = world_to_navi_vel(navi2world_pose, info["pelvgf"].reshape(-1, 3))
        pelvbf  = world_to_navi_vel(navi2world_pose, info["pelvbf"].reshape(-1, 3))
        torsgf  = world_to_navi_vel(navi2world_pose, info["torsgf"].reshape(-1, 3))
        torsbf  = world_to_navi_vel(navi2world_pose, info["torsbf"].reshape(-1, 3))
        feetgf  = world_to_navi_vel(navi2world_pose, info["feetgf"].reshape(-1, 3))
        feetbf  = world_to_navi_vel(navi2world_pose, info["feetbf"].reshape(-1, 3))
        handsgf = world_to_navi_vel(navi2world_pose, info["handsgf"].reshape(-1, 3))
        handsbf = world_to_navi_vel(navi2world_pose, info["handsbf"].reshape(-1, 3))
        kneesgf = world_to_navi_vel(navi2world_pose, info["kneesgf"].reshape(-1, 3))
        kneesbf = world_to_navi_vel(navi2world_pose, info["kneesbf"].reshape(-1, 3))
        shldsgf = world_to_navi_vel(navi2world_pose, info["shldsgf"].reshape(-1, 3))
        shldsbf = world_to_navi_vel(navi2world_pose, info["shldsbf"].reshape(-1, 3))

        # Command to nav frame
        cmd = info["command"].copy()
        cmd_vel_navi = world_to_navi_vel(navi2world_pose, cmd[-3:].reshape(-1, 3)).reshape(-1)
        cmd[-3:] = cmd_vel_navi
        cmd[-1] = 0.0

        headbf  = headbf  * (info["headdf"]  < 0.5); headdf  = np.clip(info["headdf"],  -1.0, 0.5)
        pelvbf  = pelvbf  * (info["pelvdf"]  < 0.5); pelvdf  = np.clip(info["pelvdf"],  -1.0, 0.5)
        torsbf  = torsbf  * (info["torsdf"]  < 0.5); torsdf  = np.clip(info["torsdf"],  -1.0, 0.5)
        feetbf  = feetbf  * (info["feetdf"]  < 0.5); feetdf  = np.clip(info["feetdf"],  -1.0, 0.5)
        handsbf = handsbf * (info["handsdf"] < 0.5); handsdf = np.clip(info["handsdf"], -1.0, 0.5)
        kneesbf = kneesbf * (info["kneesdf"] < 0.5); kneesdf = np.clip(info["kneesdf"], -1.0, 0.5)
        shldsbf = shldsbf * (info["shldsdf"] < 0.5); shldsdf = np.clip(info["shldsdf"], -1.0, 0.5)

        boxgf_n = world_to_navi_vel(navi2world_pose, info["boxgf"].reshape(-1, 3))
        boxbf_n = world_to_navi_vel(navi2world_pose, info["boxbf"].reshape(-1, 3))
        boxbf_n = boxbf_n * (info["boxdf"] < 0.5)
        boxdf_c = np.clip(info["boxdf"], -1.0, 0.5)

        gait_phase = np.hstack([np.cos(info["phase"]), np.sin(info["phase"])])
        ids = self.action_joint_ids

        pf = np.hstack([
            headgf.reshape(-1), headbf.reshape(-1), headdf.reshape(-1),
            pelvgf.reshape(-1), pelvbf.reshape(-1), pelvdf.reshape(-1),
            torsgf.reshape(-1), torsbf.reshape(-1), torsdf.reshape(-1),
            feetgf.reshape(-1), feetbf.reshape(-1), feetdf.reshape(-1),
            handsgf.reshape(-1), handsbf.reshape(-1), handsdf.reshape(-1),
            kneesgf.reshape(-1), kneesbf.reshape(-1), kneesdf.reshape(-1),
            shldsgf.reshape(-1), shldsbf.reshape(-1), shldsdf.reshape(-1),
            boxgf_n.reshape(-1), boxbf_n.reshape(-1), boxdf_c.reshape(-1),
        ])

        state = np.concatenate([
            noisy_gyro, noisy_gvec,
            (noisy_ja - self._default_qpos)[ids],
            noisy_jv[ids],
            info["last_act"],
            info["motor_targets"][ids],
            cmd, np.array([info["foot_height"]]), gait_phase,
            pf,
            box_pos_local, box_quat_local, box_size,
            stage_flag,
        ])

        return {"state": np.nan_to_num(state)}

    def get_gyro(self, frame: str) -> np.ndarray:
        sensor_id = self.mj_model.sensor(f"{consts.GYRO_SENSOR}_{frame}").id
        adr = self.mj_model.sensor_adr[sensor_id]
        dim = self.mj_model.sensor_dim[sensor_id]
        return self.mj_data.sensordata[adr:adr + dim].copy()

    def get_local_linvel(self, frame: str) -> np.ndarray:
        sensor_id = self.mj_model.sensor(f"{consts.LOCAL_LINVEL_SENSOR}_{frame}").id
        adr = self.mj_model.sensor_adr[sensor_id]
        dim = self.mj_model.sensor_dim[sensor_id]
        return self.mj_data.sensordata[adr:adr + dim].copy()

    def _compute_cmd_4d(self, rtf: np.ndarray, cgf: np.ndarray, cbf: np.ndarray) -> np.ndarray:
        """Returns [move_flag, vx, vy, yaw] command from PF fields (mirrors G1CatEnv.compute_cmd_from_rtf)."""
        v = rtf[:2] * 0.7
        b_hat = cbf[:, :2] / (np.linalg.norm(cbf[:, :2], axis=-1, keepdims=True) + 1e-9)
        Ls = np.sum(b_hat * cgf[:, :2], axis=-1)
        bv = np.sum(b_hat * v, axis=-1)
        diff = (Ls - bv)[:, None] / (np.sum(b_hat * b_hat, axis=-1, keepdims=True) + 1e-9)
        delta = diff * b_hat
        mask = (Ls > bv)[:, None]
        delta = np.where(mask, delta, 0.0)
        v_new = v + np.mean(delta, axis=0)
        command = np.array([1.0, v_new[0], v_new[1], 0.0]) * 0.75
        if np.linalg.norm(command[1:4]) < 0.2:
            command = np.zeros(4)
        return command

    def _update_phase(self, state: State, command: np.ndarray) -> None:
        """Advance gait phase clock; handle move↔stop transitions."""
        step = state.info["step"]
        last_flags = state.info["last_flags"]
        phase = state.info["phase"]
        phase_dt = state.info["phase_dt"]
        timestamp = state.info["timestamp_move2stop"]

        has_vel = np.linalg.norm(command) > 0.2
        had_vel = last_flags[0]

        move2stop = (had_vel == 1.0) & (has_vel == 0.0)
        stop2move = (had_vel == 0.0) & (has_vel == 1.0)

        state.info["timestamp_move2stop"] = np.where(move2stop, step + 50, timestamp)
        after_delay = step > state.info["timestamp_move2stop"]
        moving = np.where((has_vel == 0.0) & after_delay, 0.0, 1.0)

        new_phase = (phase + phase_dt + np.pi) % (2 * np.pi) - np.pi
        phase = np.where(moving, new_phase, self._stance_phase)
        phase = np.where(stop2move, np.array([0.0, np.pi]), phase)

        state.info["phase"] = phase
        state.info["last_flags"] = np.array([float(has_vel), float(moving)])
        gait_cycle = np.cos(phase)
        gait_mask = np.where(gait_cycle > 0.6, 1.0, 0.0)
        gait_mask = np.where(gait_cycle < -0.6, -1.0, gait_mask)
        state.info["gait_mask"] = np.float32(gait_mask)

    def _box_corners_world(self, box_size: np.ndarray) -> np.ndarray:
        # Returns (8, 3) world positions of box corners. box_size: (hx, hy, hz) half-extents.
        box_pos  = self.mj_data.xpos [self._box_body_id]
        box_quat = self.mj_data.xquat[self._box_body_id]  # wxyz
        R_mat = R.from_quat([box_quat[1], box_quat[2], box_quat[3], box_quat[0]]).as_matrix()
        return box_pos + (_BOX_CORNER_SIGNS * box_size) @ R_mat.T

    def sample_field(self, field: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of a 3D field at world-space positions (N,3) → (N,C)."""
        rel = pos - self.pf_origin
        idx = rel / self.dx
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
        x = np.clip(x, 0, self.Nx - 2)
        y = np.clip(y, 0, self.Ny - 2)
        z = np.clip(z, 0, self.Nz - 2)
        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)
        zi = np.floor(z).astype(np.int32)
        xd, yd, zd = x - xi, y - yi, z - zi
        offsets = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.int32)
        base = np.stack([xi, yi, zi], axis=1)
        corners = base[:, None, :] + offsets[None, :, :]
        vals = field[corners[..., 0], corners[..., 1], corners[..., 2], :]
        wx = np.stack([1.0 - xd, xd], axis=1)
        wy = np.stack([1.0 - yd, yd], axis=1)
        wz = np.stack([1.0 - zd, zd], axis=1)
        w = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)
        return np.einsum('ne,nec->nc', w, vals)
