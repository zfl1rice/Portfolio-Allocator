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

    Supports two modes:
      - Full-period episodes (for validation/test): random_start=False, episode_length=None.
      - Random rolling-window episodes (for training):
            random_start=True, episode_length=<number of steps>.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        actions: np.ndarray,
        window_size: int = 20,
        cost: float = 0.0005,
        episode_length: int | None = None,
        random_start: bool = False,
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
        episode_length : int or None
            If not None and random_start=True, each episode is a rolling window
            of this many steps.
        random_start : bool
            If True and episode_length is set, sample a random start index for each episode.
        """
        assert all(t in log_returns.columns for t in TICKERS), "Missing tickers in log_returns."
        self.log_returns = log_returns[TICKERS].copy()
        self.dates = self.log_returns.index.to_numpy()
        self.actions = actions.astype(float)
        self.window_size = window_size
        self.cost = cost
        self.episode_length = episode_length
        self.random_start = random_start

        self.n_assets = len(TICKERS)
        self.n_actions = self.actions.shape[0]

        self._check_lengths()

        # These will be set in reset()
        self.start_idx = None
        self.end_idx = None
        self.t = None
        self.value = None
        self.prev_w = None

    def _check_lengths(self):
        assert self.actions.shape[1] == self.n_assets, "actions must have n_assets columns"
        assert len(self.log_returns) > self.window_size, "Not enough data for the chosen window_size."

    @property
    def state_dim(self) -> int:
        # window_size * n_assets returns + n_assets weights
        return self.window_size * self.n_assets + self.n_assets

    def reset(self) -> np.ndarray:
        """
        Reset episode and return initial state.

        - If random_start and episode_length are set:
            sample a random starting index such that there is space for the
            return window and the full episode length.

        - Otherwise:
            start at window_size and run until the end of the series.
        """
        n_dates = len(self.log_returns)

        if self.random_start and self.episode_length is not None:
            # valid start indices must satisfy:
            #   start_idx >= window_size
            #   start_idx + episode_length <= n_dates
            max_start = n_dates - self.episode_length
            if max_start <= self.window_size:
                raise ValueError(
                    "Not enough data to support the requested episode_length and window_size."
                )
            start = np.random.randint(self.window_size, max_start)
            self.start_idx = start
            self.end_idx = start + self.episode_length
        else:
            # Full-period episode
            self.start_idx = self.window_size
            self.end_idx = n_dates

        self.t = self.start_idx
        self.value = 1.0
        self.prev_w = np.zeros(self.n_assets, dtype=float)

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        # window of past returns flattened + current weights
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

        done = self.t >= self.end_idx
        next_state = None if done else self._get_state()
        info = {
            "date": self.dates[self.t - 1],
            "portfolio_value": self.value,
        }

        return next_state, reward, done, info
