"""Training entry point for CaTra (Carry and Traverse) tasks.

Extends train_ppo.py with the --box reward scale argument used by G1CaTra.
"""
from dataclasses import dataclass

import tyro

from train_ppo import Args, train, _apply_args_to_config


@dataclass
class CatraArgs(Args):
    box: float = 0  # scale: box guidance-field and obstacle-distance rewards


def _apply_catra_args_to_config(args: CatraArgs, policy_cfg, env_config, debug: bool):
    _apply_args_to_config(args, policy_cfg, env_config, debug)
    if args.box != 0 and hasattr(env_config.reward_config.scales, "boxgf"):
        env_config.reward_config.scales.boxgf = args.box  # scale: box moves in guidance direction
        env_config.reward_config.scales.boxdf = args.box  # scale: box stays away from obstacles


def train_catra(args: CatraArgs):
    # Monkey-patch _apply_args_to_config so train() picks up the box scales.
    import train_ppo as _train_ppo_mod
    original = _train_ppo_mod._apply_args_to_config
    _train_ppo_mod._apply_args_to_config = _apply_catra_args_to_config
    try:
        train(args)
    finally:
        _train_ppo_mod._apply_args_to_config = original


if __name__ == "__main__":
    train_catra(tyro.cli(CatraArgs))
