# src/env_portfolio.py

import numpy as np
import pandas as pd

TICKERS = ["SPY", "TLT", "GLD", "SHY"]


class PortfolioEnv:
    """
    A simple daily portfolio allocation environment.

    State: last `window_size` days of log returns (flattened) + current weights.
    Action: index into a fixed set of portfolio weight vectors.
    Reward: daily portfolio log return - cost * turnover.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        actions: np.ndarray,
        window_size: int = 20,
        cost: float = 0.0005,
    ):
        """
        Parameters
        ----------
        log_returns : DataFrame
            Daily log returns for TICKERS. Index: dates, Columns: TICKERS.
        actions : ndarray
            Shape (n_actions, n_assets). Each row is a weight vector (will be normalized).
        window_size : int
            Number of past days of returns in the state.
        cost : float
            Transaction cost per unit turnover.
        """
        assert all(t in log_returns.columns for t in TICKERS), "Missing tickers in log_returns."
        self.log_returns = log_returns[TICKERS].copy()
        self.dates = self.log_returns.index
        self.actions = actions.astype(float)
        self.window_size = window_size
        self.cost = cost

        self.n_assets = len(TICKERS)
        self.n_actions = self.actions.shape[0]

        self._check_lengths()
        self.reset()

    def _check_lengths(self):
        assert self.actions.shape[1] == self.n_assets, "actions must have n_assets columns"

    @property
    def state_dim(self) -> int:
        # window_size * n_assets returns + n_assets weights
        return self.window_size * self.n_assets + self.n_assets

    def reset(self):
        """
        Reset episode to the beginning and return initial state.
        """
        self.t = self.window_size  # index into self.dates
        self.value = 1.0
        self.prev_w = np.zeros(self.n_assets, dtype=float)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        window = self.log_returns.iloc[self.t - self.window_size : self.t].values
        state = np.concatenate([window.flatten(), self.prev_w])
        return state.astype(np.float32)

    def step(self, action_idx: int):
        """
        Take a step in the environment.

        Parameters
        ----------
        action_idx : int
            Index into actions array.

        Returns
        -------
        next_state : np.ndarray or None
        reward : float
        done : bool
        info : dict
        """
        # normalize action weights
        w = self.actions[action_idx].copy()
        if w.sum() == 0:
            w[:] = 1.0 / self.n_assets
        else:
            w /= w.sum()

        r_vec = self.log_returns.iloc[self.t].values
        port_ret = np.dot(w, r_vec)

        turnover = np.abs(w - self.prev_w).sum()
        reward = port_ret - self.cost * turnover

        self.value *= np.exp(reward)
        self.prev_w = w
        self.t += 1

        done = self.t >= len(self.dates)
        next_state = None if done else self._get_state()
        info = {"date": self.dates[self.t - 1], "portfolio_value": self.value}

        return next_state, reward, done, info
