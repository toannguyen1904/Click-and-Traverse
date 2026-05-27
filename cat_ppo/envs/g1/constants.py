# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for G1."""

from pathlib import Path

import numpy as np

from cat_ppo.constant import PATH_ASSET

ROOT_PATH = PATH_ASSET / "unitree_g1"
# Lightweight scene XML used during training: feet-only collision geometry on flat ground,
# no visual meshes — faster to simulate under MJX.
# So during training, the robot physically walks on a flat floor (the XML),
# but the HumanoidPF fields encode the obstacle geometry and are used for observations and rewards.
# The collision avoidance is entirely driven by the precomputed fields — MuJoCo doesn't simulate the obstacle meshes during training.
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain.xml"
# Full-detail scene XML with a scene_mesh asset slot; used for evaluation.
# At eval time, set_scene_for_xml() patches this XML to point scene_mesh at the
# chosen obs.obj, so physical obstacle collisions are simulated. Any scene from
# TypiObs/, RandObs/, R2SObs/ etc. can be substituted via --obs_path.
MESH_XML = ROOT_PATH / "scene_mjx_feetonly_mesh.xml"

# CaTra (Carry and Traverse) scene XMLs
CATRA_FLAT_TERRAIN_XML = ROOT_PATH / "scene_mjx_feetonly_flat_terrain_catra.xml"
CATRA_MESH_XML = ROOT_PATH / "scene_mjx_feetonly_mesh_catra.xml"


def task_to_xml(task_name: str) -> Path:
    # Maps a task name string to its corresponding MuJoCo scene XML path.
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "mesh": MESH_XML,
        "flat_terrain_catra": CATRA_FLAT_TERRAIN_XML,
        "mesh_catra": CATRA_MESH_XML,
    }[task_name]


# MuJoCo site names for foot contact detection and HumanoidPF body-part sampling.
# Site is a Mujoco's concept - a coordinate frame you can place anywhere on the robot (tip of a foot, center of a palm, etc.)
# without adding any physics mass or collision geometry.
FEET_SITES = [
    "left_foot",
    "right_foot",
]

# MuJoCo site names for hand/palm positions used in HumanoidPF body-part sampling.
HAND_SITES = [
    "left_palm",
    "right_palm",
]

# MuJoCo site names for knee positions used in HumanoidPF body-part sampling.
KNEE_SITES = [
    "left_knee",
    "right_knee",
]

# MuJoCo site names for shoulder positions used in HumanoidPF body-part sampling.
SHOULDER_SITES = [
    "left_shoulder",
    "right_shoulder",
]

# MuJoCo geom names used for foot–ground contact detection.
LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

# Name of the root body (torso) in the MuJoCo model; used as the reference frame
# for local velocity, orientation, and HumanoidPF queries.
ROOT_BODY = "torso_link"

# MuJoCo sensor names — these map to named sensors in the MJCF XML.
GRAVITY_SENSOR = "upvector"          # unit vector pointing opposite to gravity, i.e. up in world frame
GLOBAL_LINVEL_SENSOR = "global_linvel"  # root linear velocity in world frame (m/s)
GLOBAL_ANGVEL_SENSOR = "global_angvel"  # root angular velocity in world frame (rad/s)
LOCAL_LINVEL_SENSOR = "local_linvel"    # root linear velocity projected into the robot's local frame (m/s)
ACCELEROMETER_SENSOR = "accelerometer" # IMU linear acceleration at the torso (m/s²)
GYRO_SENSOR = "gyro"                   # IMU angular velocity at the torso (rad/s)

# Conservative joint-range limits (radians) used during training to keep the robot
# within safe, physically realistic poses. Ordered to match OBS_JOINT_NAMES:
# 6 left-leg joints, 6 right-leg joints, 3 waist joints,
# 7 left-arm joints, 7 right-arm joints.
RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-1.57, 1.57),  # left_hip_pitch_joint
    (-0.5, 0.5),    # left_hip_roll_joint
    (-0.7, 0.7),    # left_hip_yaw_joint
    (0, 1.57),      # left_knee_joint (knee only bends forward)
    (-0.4, 0.4),    # left_ankle_pitch_joint
    (-0.2, 0.2),    # left_ankle_roll_joint
    # Right leg.
    (-1.57, 1.57),  # right_hip_pitch_joint
    (-0.5, 0.5),    # right_hip_roll_joint
    (-0.7, 0.7),    # right_hip_yaw_joint
    (0, 1.57),      # right_knee_joint
    (-0.4, 0.4),    # right_ankle_pitch_joint
    (-0.2, 0.2),    # right_ankle_roll_joint
    # Waist.
    (-2.618, 2.618),  # waist_yaw_joint   (±150°)
    (-0.52, 0.52),    # waist_roll_joint  (±30°)
    (-0.52, 0.52),    # waist_pitch_joint (±30°)
    # Left shoulder.
    (-3.0892, 2.6704),   # left_shoulder_pitch_joint
    (-1.5882, 2.2515),   # left_shoulder_roll_joint
    (-2.618, 2.618),     # left_shoulder_yaw_joint
    (-1.0472, 2.0944),   # left_elbow_joint
    (-1.97222, 1.97222), # left_wrist_roll_joint
    (-1.61443, 1.61443), # left_wrist_pitch_joint
    (-1.61443, 1.61443), # left_wrist_yaw_joint
    # Right shoulder.
    (-3.0892, 2.6704),   # right_shoulder_pitch_joint
    (-2.2515, 1.5882),   # right_shoulder_roll_joint (asymmetric vs left)
    (-2.618, 2.618),     # right_shoulder_yaw_joint
    (-1.0472, 2.0944),   # right_elbow_joint
    (-1.97222, 1.97222), # right_wrist_roll_joint
    (-1.61443, 1.61443), # right_wrist_pitch_joint
    (-1.61443, 1.61443), # right_wrist_yaw_joint
)

# Joints controlled by the policy action output (12-dim).
# Waist and arm joints are excluded: waist is locked for stability,
# arms follow a fixed default pose during locomotion.
ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    # "waist_yaw_joint",
    # "waist_roll_joint",
    # "waist_pitch_joint",
]

# Joints whose positions and velocities appear in the observation vector (20-dim).
# Superset of ACTION_JOINT_NAMES: includes waist and partial arm joints so the
# policy can observe full-body posture even though it only commands the legs.
OBS_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# CaTra box site/geom names in the scene XML
BOX_SITE = "box_center"
BOX_GEOM = "box_geom"

# TEMP: 20-joint action space for CaTra: 12 legs + 8 arms (all 3 waist joints dropped).
# Waist yaw/roll/pitch remain held at their default PD targets each step.
CATRA_ACTION_JOINT_NAMES = [
    # 12 leg joints (same as ACTION_JOINT_NAMES)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # 0 waist joints (TEMP: yaw + roll + pitch all removed)
    # 8 arm joints (left then right, shoulder + elbow only; wrists stay at default)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
# num_act = 20

# Same resting pose as DEFAULT_QPOS — arms hang naturally, not in a carrying pose.
# fmt: off
DEFAULT_QPOS_CATRA = np.float32([
    0, 0, 0.8,       # root xyz
    1, 0, 0, 0,      # root quat (identity)
    -0.1, 0, 0, 0.3, -0.2, 0,    # left leg (same as DEFAULT_QPOS)
    -0.1, 0, 0, 0.3, -0.2, 0,    # right leg
    0, 0, 0,                      # waist (neutral)
    0.2, 0.3, 0, 1.28, 0, 0, 0,  # left arm (same as DEFAULT_QPOS)
    0.2, -0.3, 0, 1.28, 0, 0, 0, # right arm (same as DEFAULT_QPOS)
])
# fmt: on

# Nominal chest height above ground (metres) used as a reference for height-reward
# and termination checks.
DEFAULT_CHEST_Z = 1.0

# fmt: off
# Peak torque limits (N·m) for each joint in OBS_JOINT_NAMES order.
# Used to normalise action outputs and clip commanded torques before applying PD control.
TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,   # left leg
    88., 139., 88., 139., 50., 50.,   # right leg
    88., 50., 50.,                    # waist
    25., 25., 25., 25., 25., 5., 5.,  # left arm
    25., 25., 25., 25., 25., 5., 5.,  # right arm
])

# Default joint configuration (qpos) used to initialise episodes and as the
# PD target reference for arm joints not commanded by the policy.
# Layout: [root_xyz(3), root_quat(4), left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
DEFAULT_QPOS = np.float32([
    0, 0, 0.8,          # root position: standing ~0.8 m above ground
    1, 0, 0, 0,         # root quaternion: upright (w=1, no rotation)
    -0.1, 0, 0, 0.3, -0.2, 0,   # left leg: slight hip pitch + knee bend for natural stance
    -0.1, 0, 0, 0.3, -0.2, 0,   # right leg: mirrored
    0, 0, 0,            # waist: neutral
    0.2, 0.3, 0, 1.28, 0, 0, 0,   # left arm: relaxed at side
    0.2, -0.3, 0, 1.28, 0, 0, 0,  # right arm: mirrored
])

# PD proportional gains (N·m/rad) per joint, tuned for the G1 hardware (version 2).
# Higher values on load-bearing joints (knee: 200, waist: 300) for stiffness.
KPs = np.float32([
    100, 100, 100, 200, 80, 20,   # left leg
    100, 100, 100, 200, 80, 20,   # right leg
    300, 300, 300,                # waist
    90, 60, 20, 60, 20, 20, 20,   # left arm
    90, 60, 20, 60, 20, 20, 20,   # right arm
])

# PD derivative gains (N·m·s/rad) per joint, providing damping to prevent oscillation.
KDs = np.float32([
    2, 2, 2, 4, 2, 1,   # left leg
    2, 2, 2, 4, 2, 1,   # right leg
    10, 10, 10,          # waist
    2, 2, 1, 1, 1, 1, 1,  # left arm
    2, 2, 1, 1, 1, 1, 1,  # right arm
])
# fmt: on
