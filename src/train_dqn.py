# src/train_dqn.py

from pathlib import Path
import numpy as np
import pandas as pd

from src.env_portfolio import PortfolioEnv, TICKERS
from src.dqn_agent import DQNAgent
from src.data_processing import train_val_test_split


def load_log_returns() -> pd.DataFrame:
    path = Path("data/processed/log_returns.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_date_split(name: str):
    path = Path(f"data/processed/{name}_dates.csv")
    # Read as a DataFrame (no header), then take the first column
    df = pd.read_csv(path, header=None)
    dates = pd.to_datetime(df.iloc[:, 0].values)
    return dates



def create_actions() -> np.ndarray:
    """
    Define a small discrete action set of portfolio weights.
    """
    n = len(TICKERS)
    actions = []

    # 100% in each single asset
    for i in range(n):
        w = np.zeros(n)
        w[i] = 1.0
        actions.append(w)

    # Equal weight across all
    actions.append(np.ones(n) / n)

    # 60/40 SPY/TLT if both exist
    if "SPY" in TICKERS and "TLT" in TICKERS:
        w = np.zeros(n)
        w[TICKERS.index("SPY")] = 0.6
        w[TICKERS.index("TLT")] = 0.4
        actions.append(w)

    return np.vstack(actions)


def evaluate_policy(env: PortfolioEnv, agent: DQNAgent) -> float:
    """
    Run one episode in eval mode and return final portfolio value.
    """
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        state = next_state if next_state is not None else state
    return env.value
def load_log_returns() -> pd.DataFrame:
    path = Path("data/processed/log_returns.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def train():
    log_rets = load_log_returns()

    # Compute chronological split directly from the index
    train_dates, val_dates, test_dates = train_val_test_split(log_rets.index)

    # Slice log returns by date
    log_rets_train = log_rets.loc[log_rets.index.isin(train_dates)].copy()
    log_rets_val = log_rets.loc[log_rets.index.isin(val_dates)].copy()

    actions = create_actions()

    # Create envs
    window_size = 20
    cost = 0.0005
    env_train = PortfolioEnv(log_rets_train, actions, window_size=window_size, cost=cost)
    env_val = PortfolioEnv(log_rets_val, actions, window_size=window_size, cost=cost)

    state_dim = env_train.state_dim
    n_actions = actions.shape[0]

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=50_000,
        min_buffer_size=1_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10_000,
        target_update_freq=1000,
    )

    num_episodes = 50

    # --- NEW: track history for plotting later ---
    history = []  # list of dicts: {"episode": ..., "train_ep_return": ..., "val_final_value": ...}

    for ep in range(1, num_episodes + 1):
        state = env_train.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, info = env_train.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss, batch_used = agent.train_step()
            ep_reward += reward
            state = next_state if next_state is not None else state
            step += 1

        # Evaluate on validation set (greedy policy, no exploration)
        val_final_value = evaluate_policy(env_val, agent)

        print(
            f"Episode {ep:03d} | train_ep_return={ep_reward:.4f} | "
            f"val_final_value={val_final_value:.4f} | steps={step}"
        )

        history.append(
            {
                "episode": ep,
                "train_ep_return": ep_reward,
                "val_final_value": val_final_value,
                "steps": step,
            }
        )

    # --- NEW: save history to CSV for plotting ---
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    hist_path = results_dir / "training_stats.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"Saved training history to {hist_path}")

    # Optional: save model
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    torch_path = out_dir / "dqn_portfolio.pt"
    import torch

    torch.save(agent.q_net.state_dict(), torch_path)
    print(f"Saved trained Q-network to {torch_path}")


if __name__ == "__main__":
    train()
