"""CPU-based inference environments for G1Pickup and G1Stand policies.

Used by mj_onnx_play.py to run a trained ONNX policy in real-time MuJoCo.
Mirrors _get_obs of the respective training envs but runs on NumPy (no JAX).
"""
import time
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import cat_ppo
from cat_ppo.envs.g1 import constants as consts
from cat_ppo.envs.g1.env_pickup import (
    PICKUP_ACTION_JOINT_NAMES,
    STAND_ACTION_JOINT_NAMES,
    BOX_HALF_X_RANGE,
    BOX_HALF_Y_RANGE,
    BOX_HALF_Z_RANGE,
    BOX_MASS_RANGE,
)
from cat_ppo.envs.g1.env_catra import SUPPORT_QPOS_START
from cat_ppo.envs.g1.play_cat import BaseEnv, State, set_scene_for_xml

# qpos layout: [0:7] root | [7:36] robot | [36:43] box freejoint
BOX_QPOS_START = 36
BOX_QVEL_START = 35
NUM_ROBOT_JOINTS = 29


@cat_ppo.registry.register("G1Pickup", "play_env_class")
class PlayG1PickupEnv(BaseEnv):
    """CPU inference env for G1Pickup. 20-DOF action (legs + arms), compact obs."""

    def __init__(
        self,
        task_type: str = "flat_terrain_catra",
        config=None,
        dt: float = 0.02,
        sim_dt: float = 0.002,
        headless: bool = False,
        surface_z: Optional[float] = None,
    ):
        xml_path = consts.CATRA_MESH_XML
        tmp_xml = set_scene_for_xml(xml_path, config.pf_config.path)
        self.mj_model = mujoco.MjModel.from_xml_path(tmp_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = sim_dt
        self.headless = headless
        if not self.headless:
            import mujoco.viewer as _mv
            self.viewer = _mv.launch_passive(self.mj_model, self.mj_data)
        self._config = config
        self.dt = dt
        self.sim_dt = sim_dt
        self._surface_z_fixed = surface_z
        self._post_init()

    def _post_init(self):
        self._default_qpos = np.array(consts.DEFAULT_QPOS_CATRA[7:])  # 29-dim robot joints

        self.action_joint_names = PICKUP_ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = np.array([
            self.mj_model.actuator(name).id for name in self.action_joint_names
        ])

        self._pelvis_imu_site_id = self.mj_model.site("imu_in_pelvis").id
        self._torso_imu_site_id = self.mj_model.site("imu_in_torso").id
        self._head_site_id = self.mj_model.site("head").id
        self._hands_site_id = np.array([
            self.mj_model.site(name).id for name in consts.HAND_SITES
        ])
        self._shlds_site_id = np.array([
            self.mj_model.site(name).id for name in consts.SHOULDER_SITES
        ])
        self._pelvis_body_id = self.mj_model.body("pelvis").id
        self._box_body_id = self.mj_model.body("carried_box").id
        self._box_geom_id = self.mj_model.geom("box_geom").id
        self._box_support_geom_id = self.mj_model.geom("box_support_col").id

        # Nominal (XML) box half-extents — the DR maxima. reset() places the box
        # using the nominal half_z so DR'd (smaller) boxes are never embedded in
        # the pillar, matching training (env_pickup.domain_randomize_pickup).
        self._box_half_nominal = self.mj_model.geom_size[self._box_geom_id].copy()

        lowers, uppers = self.mj_model.jnt_range[1:1 + NUM_ROBOT_JOINTS].T
        c = (lowers + uppers) / 2
        r = uppers - lowers
        factor = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * factor
        self._soft_uppers = c + 0.5 * r * factor

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)

    def reset(self, surface_z: Optional[float] = None):
        """Reset robot to default pose, place box on support surface."""
        qpos = np.array(consts.DEFAULT_QPOS_CATRA, dtype=np.float64)
        self.mj_data.qpos[:len(qpos)] = qpos
        self.mj_data.qvel[:] = 0.0

        # Determine surface height
        if surface_z is None:
            surface_z = self._surface_z_fixed
        if surface_z is None:
            lo, hi = self._config.box_surface_height_range
            surface_z = float(np.random.uniform(lo, hi))

        # Randomize box half-extents and mass per reset, matching training DR
        # (env_pickup.domain_randomize_pickup). DR'd sizes are <= XML nominal.
        self.mj_model.geom_size[self._box_geom_id] = [
            np.random.uniform(*BOX_HALF_X_RANGE),
            np.random.uniform(*BOX_HALF_Y_RANGE),
            np.random.uniform(*BOX_HALF_Z_RANGE),
        ]
        self.mj_model.body_mass[self._box_body_id] = np.random.uniform(*BOX_MASS_RANGE)

        # Place box using the nominal (XML max) half_z so a smaller DR'd box rests
        # at or slightly above the pillar top — never embedded (matches training).
        support_half_z = float(self.mj_model.geom_size[self._box_support_geom_id][2])
        box_z = surface_z + support_half_z + float(self._box_half_nominal[2])

        # Place box 0.3 m in front of robot
        root_qpos = qpos[:7]
        w, x, y, z = root_qpos[3], root_qpos[4], root_qpos[5], root_qpos[6]
        forward_xy = np.array([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z)])
        box_xy = root_qpos[:2] + 0.3 * forward_xy

        # Box yaw = identity (no offset) for play env; pillar matches
        box_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Set box position and orientation
        self.mj_data.qpos[BOX_QPOS_START:BOX_QPOS_START + 3] = [box_xy[0], box_xy[1], box_z]
        self.mj_data.qpos[BOX_QPOS_START + 3:BOX_QPOS_START + 7] = box_quat

        # Position support pillar via qpos; yaw matches box
        self.mj_data.qpos[SUPPORT_QPOS_START:SUPPORT_QPOS_START + 3] = [box_xy[0], box_xy[1], surface_z]
        self.mj_data.qpos[SUPPORT_QPOS_START + 3:SUPPORT_QPOS_START + 7] = box_quat

        mujoco.mj_forward(self.mj_model, self.mj_data)
        if not self.headless:
            self.viewer.sync()

        left_hand_pos = self.mj_data.site_xpos[self._hands_site_id[0]].copy()
        right_hand_pos = self.mj_data.site_xpos[self._hands_site_id[1]].copy()
        head_pos = self.mj_data.site_xpos[self._head_site_id].copy()
        box_size = self.mj_model.geom_size[self._box_geom_id].copy()

        info = {
            "step": 0,
            "last_act": np.zeros(self.action_size),
            "motor_targets": self._default_qpos.copy(),
            "head_pos": head_pos,
            "last_left_hand_pos": left_hand_pos,
            "last_right_hand_pos": right_hand_pos,
            "surface_z": float(surface_z),
            "support_half_z": support_half_z,
            "box_size": box_size,
            "box_mass": float(self.mj_model.body_mass[self._box_body_id]),
            "box_z_init": float(box_z),
        }

        obs = self.get_obs(info)
        return State(info, obs)

    def step(self, state: State, action: np.ndarray) -> State:
        """Apply PD control for one policy step; uncontrolled joints stay at default."""
        lower_motor_targets = np.clip(
            state.info["motor_targets"][self.action_joint_ids] + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()
        motor_targets[self.action_joint_ids] = lower_motor_targets
        state.info["motor_targets"] = motor_targets.copy()

        for _ in range(int(self.dt / self.sim_dt)):
            torques = consts.KPs * (motor_targets - self.mj_data.qpos[7:7 + NUM_ROBOT_JOINTS]) \
                    + consts.KDs * (-self.mj_data.qvel[6:6 + NUM_ROBOT_JOINTS])
            self.mj_data.ctrl[:] = torques
            mujoco.mj_step(self.mj_model, self.mj_data)

        if not self.headless:
            self.viewer.sync()
            time.sleep(self.dt)

        head_pos = self.mj_data.site_xpos[self._head_site_id].copy()
        left_hand_pos = self.mj_data.site_xpos[self._hands_site_id[0]].copy()
        right_hand_pos = self.mj_data.site_xpos[self._hands_site_id[1]].copy()

        state.info["head_pos"] = head_pos
        obs = self.get_obs(state.info)

        state.info["last_left_hand_pos"] = left_hand_pos
        state.info["last_right_hand_pos"] = right_hand_pos
        state.info["last_act"] = action.copy()
        state.info["step"] += 1

        return State(state.info, obs)

    def get_obs(self, info: dict) -> dict:
        """Build 97-dim deployable state with sensor noise (matches G1PickupEnv._get_obs)."""
        nl = self._config.noise_config.level
        ns = self._config.noise_config.scales

        # Sensor readings
        gyro_pelvis = self.get_gyro("pelvis")
        pelvis_xmat = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        gvec_pelvis = pelvis_xmat.T @ np.array([0., 0., -1.])
        joint_angles = self.mj_data.qpos[7:7 + NUM_ROBOT_JOINTS]
        joint_vel = self.mj_data.qvel[6:6 + NUM_ROBOT_JOINTS]

        # Add sensor noise (same as training)
        noisy_gyro = gyro_pelvis + (2 * np.random.rand(3) - 1) * nl * ns.gyro
        noisy_gvec = gvec_pelvis + (2 * np.random.rand(3) - 1) * nl * ns.gravity
        noisy_ja = joint_angles + (2 * np.random.rand(len(joint_angles)) - 1) * nl * ns.joint_pos
        noisy_jv = joint_vel + (2 * np.random.rand(len(joint_vel)) - 1) * nl * ns.joint_vel

        # Box pose in pelvis frame
        pelvis_pos = self.mj_data.xpos[self._pelvis_body_id]
        box_pos_world = self.mj_data.xpos[self._box_body_id]
        box_pos_local = pelvis_xmat.T @ (box_pos_world - pelvis_pos)

        pelvis_xquat = self.mj_data.xquat[self._pelvis_body_id]  # wxyz
        box_xquat_world = self.mj_data.xquat[self._box_body_id]  # wxyz
        pelvis_conj = pelvis_xquat * np.array([1., -1., -1., -1.])
        box_quat_local = _quat_mul(pelvis_conj, box_xquat_world)

        # Box pose noise (imperfect box tracking): +/- box_pos per axis; +/- box_ori axis-angle.
        box_pos_local = box_pos_local + (2 * np.random.rand(3) - 1) * nl * ns.box_pos
        rand_axis = np.random.randn(3)
        rand_axis = rand_axis / (np.linalg.norm(rand_axis) + 1e-6)
        rand_angle = (2 * np.random.rand() - 1) * nl * ns.box_ori
        box_quat_local = _quat_mul(box_quat_local, _axis_angle_to_quat(rand_axis, rand_angle))

        box_size = info["box_size"]

        ids = self.action_joint_ids
        state = np.concatenate([
            noisy_gyro, noisy_gvec,
            (noisy_ja - self._default_qpos)[ids],
            noisy_jv[ids],
            info["last_act"],
            info["motor_targets"][ids],
            box_pos_local, box_quat_local, box_size,
            np.array([info["box_mass"]]),
        ])
        return {"state": np.nan_to_num(state)}

    def get_gyro(self, frame: str) -> np.ndarray:
        sensor_id = self.mj_model.sensor(f"{consts.GYRO_SENSOR}_{frame}").id
        adr = self.mj_model.sensor_adr[sensor_id]
        dim = self.mj_model.sensor_dim[sensor_id]
        return self.mj_data.sensordata[adr:adr + dim].copy()


@cat_ppo.registry.register("G1Stand", "play_env_class")
class PlayG1StandEnv(PlayG1PickupEnv):
    """CPU inference env for G1Stand. 12-DOF leg-only action, box-free obs (54-dim state)."""

    def _post_init(self):
        self._default_qpos = np.array(consts.DEFAULT_QPOS_CATRA[7:])  # 29-dim robot joints

        self.action_joint_names = STAND_ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = np.array([
            self.mj_model.actuator(name).id for name in self.action_joint_names
        ])

        self._pelvis_imu_site_id = self.mj_model.site("imu_in_pelvis").id
        self._torso_imu_site_id = self.mj_model.site("imu_in_torso").id
        self._head_site_id = self.mj_model.site("head").id
        self._box_geom_id = self.mj_model.geom("box_geom").id
        self._box_support_geom_id = self.mj_model.geom("box_support_col").id

        lowers, uppers = self.mj_model.jnt_range[1:1 + NUM_ROBOT_JOINTS].T
        c = (lowers + uppers) / 2
        r = uppers - lowers
        factor = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * factor
        self._soft_uppers = c + 0.5 * r * factor

    def reset(self, surface_z=None):
        """Reset robot to default pose. Box is present in scene but irrelevant to obs/policy."""
        qpos = np.array(consts.DEFAULT_QPOS_CATRA, dtype=np.float64)
        self.mj_data.qpos[:len(qpos)] = qpos
        self.mj_data.qvel[:] = 0.0

        if surface_z is None:
            lo, hi = self._config.box_surface_height_range
            surface_z = float(np.random.uniform(lo, hi))

        box_half_z = float(self.mj_model.geom_size[self._box_geom_id][2])
        support_half_z = float(self.mj_model.geom_size[self._box_support_geom_id][2])
        box_z = surface_z + support_half_z + box_half_z
        root_qpos = qpos[:7]
        w, x, y, z = root_qpos[3], root_qpos[4], root_qpos[5], root_qpos[6]
        forward_xy = np.array([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y + w * z)])
        box_xy = root_qpos[:2] + 0.3 * forward_xy
        box_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.mj_data.qpos[BOX_QPOS_START:BOX_QPOS_START + 3] = [box_xy[0], box_xy[1], box_z]
        self.mj_data.qpos[BOX_QPOS_START + 3:BOX_QPOS_START + 7] = box_quat
        self.mj_data.qpos[SUPPORT_QPOS_START:SUPPORT_QPOS_START + 3] = [box_xy[0], box_xy[1], surface_z]
        self.mj_data.qpos[SUPPORT_QPOS_START + 3:SUPPORT_QPOS_START + 7] = box_quat

        mujoco.mj_forward(self.mj_model, self.mj_data)
        if not self.headless:
            self.viewer.sync()

        info = {
            "step": 0,
            "last_act": np.zeros(self.action_size),
            "motor_targets": self._default_qpos.copy(),
        }
        obs = self.get_obs(info)
        return State(info, obs)

    def step(self, state: State, action: np.ndarray) -> State:
        """Apply PD control; uncontrolled joints stay at default."""
        lower_motor_targets = np.clip(
            state.info["motor_targets"][self.action_joint_ids] + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()
        motor_targets[self.action_joint_ids] = lower_motor_targets
        state.info["motor_targets"] = motor_targets.copy()

        for _ in range(int(self.dt / self.sim_dt)):
            torques = consts.KPs * (motor_targets - self.mj_data.qpos[7:7 + NUM_ROBOT_JOINTS]) \
                    + consts.KDs * (-self.mj_data.qvel[6:6 + NUM_ROBOT_JOINTS])
            self.mj_data.ctrl[:] = torques
            mujoco.mj_step(self.mj_model, self.mj_data)

        if not self.headless:
            self.viewer.sync()
            time.sleep(self.dt)

        state.info["last_act"] = action.copy()
        state.info["step"] += 1
        obs = self.get_obs(state.info)
        return State(state.info, obs)

    def get_obs(self, info: dict) -> dict:
        """Build 54-dim deployable state with sensor noise (matches G1StandEnv._get_obs)."""
        nl = self._config.noise_config.level
        ns = self._config.noise_config.scales

        gyro_pelvis = self.get_gyro("pelvis")
        pelvis_xmat = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        gvec_pelvis = pelvis_xmat.T @ np.array([0., 0., -1.])
        joint_angles = self.mj_data.qpos[7:7 + NUM_ROBOT_JOINTS]
        joint_vel = self.mj_data.qvel[6:6 + NUM_ROBOT_JOINTS]

        noisy_gyro = gyro_pelvis + (2 * np.random.rand(3) - 1) * nl * ns.gyro
        noisy_gvec = gvec_pelvis + (2 * np.random.rand(3) - 1) * nl * ns.gravity
        noisy_ja = joint_angles + (2 * np.random.rand(len(joint_angles)) - 1) * nl * ns.joint_pos
        noisy_jv = joint_vel + (2 * np.random.rand(len(joint_vel)) - 1) * nl * ns.joint_vel

        ids = self.action_joint_ids
        state = np.concatenate([
            noisy_gyro, noisy_gvec,
            (noisy_ja - self._default_qpos)[ids],
            noisy_jv[ids],
            info["last_act"],
            info["motor_targets"][ids],
        ])
        return {"state": np.nan_to_num(state)}


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


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """wxyz quaternion for a rotation of `angle` rad about unit `axis`."""
    s, c = np.sin(angle * 0.5), np.cos(angle * 0.5)
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])
