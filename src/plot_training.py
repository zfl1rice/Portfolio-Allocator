# src/plot_training.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    path = Path("results/training_stats.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.train_dqn` first to generate it."
        )

    df = pd.read_csv(path)

    # Basic info
    print(df.head())

    # Plot training episode return and validation final value vs episode
    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["train_ep_return"], label="Train episode return")
    plt.plot(df["episode"], df["val_final_value"], label="Validation final value")
    plt.xlabel("Episode")
    plt.ylabel("Value / Return")
    plt.title("DQN Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
