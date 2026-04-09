"""Training entry point for CaTra (Carry and Traverse) tasks."""
import tyro

from train_ppo import Args, train


if __name__ == "__main__":
    train(tyro.cli(Args))
