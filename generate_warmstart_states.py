"""Entry point: generate offline warm-start states for CaTra training.

Usage:
    python generate_warmstart_states.py \\
        --pickup_checkpoint_path /abs/path/to/checkpoints/000403046400 \\
        --num_states 32768 \\
        --output data/warmstart/catra_pickup_states.npz

    # Optional flags
        --sample_start 60 --sample_end 100 --lookahead 20 --seed 0
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate CaTra warm-start states from a trained pickup policy.")
    parser.add_argument("--pickup_checkpoint_path", required=True,
                        help="Absolute path to Brax orbax checkpoint directory.")
    parser.add_argument("--num_states", type=int, default=32768,
                        help="Number of states to generate (should match CaTra num_envs=32768).")
    parser.add_argument("--sample_start", type=int, default=60,
                        help="Earliest step from which a snapshot can be taken (inclusive).")
    parser.add_argument("--sample_end", type=int, default=100,
                        help="Latest step from which a snapshot can be taken (inclusive).")
    parser.add_argument("--lookahead", type=int, default=20,
                        help="Steps after snapshot to verify the box is still held.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Envs to roll out per batch (default: num_states). Lower this if the GPU OOMs; "
                             "batches run sequentially until num_states valid states are collected.")
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Safety cap on number of batches before giving up.")
    parser.add_argument("--seed", type=int, default=0,
                        help="PRNG seed for reproducibility.")
    parser.add_argument("--output", default="data/warmstart/catra_pickup_states.npz",
                        help="Output .npz file path.")
    args = parser.parse_args()

    from cat_ppo.eval.warmstart_generation import generate_warmstart_states
    generate_warmstart_states(
        pickup_checkpoint_path=args.pickup_checkpoint_path,
        num_states=args.num_states,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        lookahead=args.lookahead,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
