"""Training entry point for G1Pickup (box pickup task)."""
import tyro

from train_ppo import Args, train


if __name__ == "__main__":
    train(tyro.cli(Args))
