# src/baselines.py

import numpy as np
import pandas as pd

TICKERS = ["SPY", "TLT", "GLD", "SHY"]


def run_backtest(
    log_returns: pd.DataFrame,
    weights_fn,
    initial_value: float = 1.0,
    cost: float = 0.0005,
) -> pd.Series:
    """
    Run a simple backtest given a weight function.

    Parameters
    ----------
    log_returns : DataFrame
        Index: dates, Columns: tickers in TICKERS (same order).
    weights_fn : callable
        Function (date, prev_weights) -> weights (array-like of length len(TICKERS)).
    initial_value : float
        Starting portfolio value.
    cost : float
        Transaction cost per unit turnover (sum of abs weight changes).

    Returns
    -------
    Series
        Portfolio value over time (same index as log_returns).
    """
    dates = log_returns.index
    n_assets = len(TICKERS)

    value = initial_value
    values = []
    prev_w = np.zeros(n_assets)

    for date in dates:
        w = np.array(weights_fn(date, prev_w), dtype=float)
        if w.sum() == 0:
            # avoid division by zero
            w = np.ones(n_assets) / n_assets
        else:
            w = w / w.sum()

        r_vec = log_returns.loc[date, TICKERS].values
        port_ret = np.dot(w, r_vec)

        turnover = np.abs(w - prev_w).sum()
        port_ret_after_cost = port_ret - cost * turnover

        value *= np.exp(port_ret_after_cost)
        values.append(value)
        prev_w = w

    return pd.Series(values, index=dates, name="portfolio_value")


def buy_and_hold_spy_weights(date, prev_w):
    w = np.zeros(len(TICKERS))
    w[TICKERS.index("SPY")] = 1.0
    return w


def equal_weight_weights(date, prev_w):
    return np.ones(len(TICKERS)) / len(TICKERS)
