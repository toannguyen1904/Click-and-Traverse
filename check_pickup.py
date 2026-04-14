"""Visualize the start of a G1Pickup episode in the MuJoCo viewer.

Usage:
    python check_pickup.py
    python check_pickup.py --surface_z 0.6   # fix surface height instead of random
"""
import time
from dataclasses import dataclass
from typing import Optional

import tyro

import cat_ppo


@dataclass
class Args:
    surface_z: Optional[float] = None  # None = random from config range


def main(args: Args):
    env_class = cat_ppo.registry.get("G1Pickup", "play_env_class")
    cfg = cat_ppo.registry.get("G1Pickup", "config")

    env = env_class(config=cfg.env_config)
    state = env.reset(surface_z=args.surface_z)

    box_body_id     = env.mj_model.body("carried_box").id
    support_body_id = env.mj_model.body("box_support").id

    print(f"Support z    : {env.mj_data.xpos[support_body_id][2]:.3f} m")
    print(f"Box z        : {env.mj_data.xpos[box_body_id][2]:.3f} m")
    print(f"Box position : {env.mj_data.xpos[box_body_id]}")
    print(f"Box size (half-extents): {env.mj_model.geom_size[env._box_geom_id]}")
    print("\nViewer open — close the window to exit.")

    while env.viewer.is_running():
        env.viewer.sync()
        time.sleep(0.02)


if __name__ == "__main__":
    main(tyro.cli(Args))
