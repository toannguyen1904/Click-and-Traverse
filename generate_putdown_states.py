"""Entry point: generate offline warm-start states for PUTDOWN training.

Rolls out a trained two-agent CaTra policy on a single scene and snapshots each
env at the first step it reaches the goal (base_x >= goal_x) while still holding
the box (lowest box corner above the supporting-surface height). These "arrived,
still carrying" states seed PUTDOWN training.

Usage:
    python generate_putdown_states.py \\
        --exp_name 07011515_G1CaTra2ADagger_..._Obsempty \\
        --task G1CaTra2A \\
        --obs_path data/assets/TypiObs/bar2/ \\
        --warmstart_states_path data/warmstart/catra_pickup_states_9.0.npz \\
        --num_states 32768 \\
        --output data/warmstart/putdown_states.npz

    # Optional flags
        --goal_x 1.6 --surface_z 0.6 --batch_size 8192 --seed 0
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate PUTDOWN warm-start states from a trained CaTra policy.")
    parser.add_argument("--exp_name", required=True,
                        help="CaTra run whose checkpoint holds the rollout policy (resolved via get_latest_ckpt).")
    parser.add_argument("--task", default="G1CaTra2A",
                        help="Two-agent CaTra task name (env + network config).")
    parser.add_argument("--obs_path", required=True,
                        help="Scene / obstacle-field directory, e.g. data/assets/TypiObs/bar2/. "
                             "Every env runs on this single scene.")
    parser.add_argument("--warmstart_states_path", required=True,
                        help="CaTra init states (.npz) the rollout is warm-started from "
                             "(robot already holding the box).")
    parser.add_argument("--num_states", type=int, default=32768,
                        help="Number of valid states to save.")
    parser.add_argument("--goal_x", type=float, default=1.6,
                        help="Base x (m) counted as reaching the goal; snapshot taken at first crossing.")
    parser.add_argument("--surface_z", type=float, default=0.5,
                        help="Supporting-surface height (m). A state is valid only if the lowest box "
                             "corner is above this at the snapshot (box still held).")
    parser.add_argument("--episode_length", type=int, default=None,
                        help="Rollout length per batch (default: env_config.episode_length).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Envs per batch (default: num_states). Lower this if the GPU OOMs.")
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Safety cap on number of batches before giving up.")
    parser.add_argument("--seed", type=int, default=0,
                        help="PRNG seed for reproducibility.")
    parser.add_argument("--output", default="data/warmstart/putdown_states.npz",
                        help="Output .npz file path.")
    parser.add_argument("--box_inflation", action="store_true", default=True,
                        help="Fallback for box_use_inflation when config.json is unavailable.")
    parser.add_argument("--no_box_noise", dest="box_noise", action="store_false", default=True,
                        help="Disable box tracking noise in the deployable obs (default: keep, matches training).")
    args = parser.parse_args()

    from cat_ppo.eval.putdown_warmstart_generation import generate_putdown_states
    generate_putdown_states(
        exp_name=args.exp_name,
        task=args.task,
        obs_path=args.obs_path,
        warmstart_states_path=args.warmstart_states_path,
        num_states=args.num_states,
        goal_x=args.goal_x,
        surface_z=args.surface_z,
        episode_length=args.episode_length,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
        output_path=args.output,
        box_inflation=args.box_inflation,
        box_noise=args.box_noise,
    )


if __name__ == "__main__":
    main()
