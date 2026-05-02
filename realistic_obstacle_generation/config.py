from dataclasses import dataclass, field
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RealCfg:
    # Voxel grid geometry (parity with procedural PFConfig defaults).
    voxel: float = 0.04
    Lx: float = 5.0
    Ly: float = 5.0
    Lz: float = 1.5

    # Robot torso z used for start_w / goal_w (matches procedural).
    robot_z: float = 0.75

    # Goal sampled on a circle of this radius around the start.
    goal_radius: float = 2.0

    # Walkable-region erosion radius (m). 0.10 / 0.04 = 2.5 -> rounds to 3 voxels.
    walk_erode_r: float = 0.10

    # Furniture whose AABB.zmin is above this is treated as ceiling-mounted
    # for the walkable projection (so it does not block the floor).
    footprint_z_max: float = 1.0

    # Sampling retry budget. Total attempts <= max_start_retries * max_goal_retries.
    max_start_retries: int = 8
    max_goal_retries: int = 64

    # Dataset and output locations.
    data_root: Path = Path("/data/tientoan/3DFRONT")
    out_root: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "assets" / "RealObs")
