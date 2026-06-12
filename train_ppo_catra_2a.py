"""Training entry point for two-agent CaTra (G1CaTra2A / G1CaTra2APri)."""
import tyro

from train_ppo import Args, train


if __name__ == "__main__":
    train(tyro.cli(Args))
