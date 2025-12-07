import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

TICKERS = ["SPY", "TLT", "GLD", "SHY"]

def load_data() -> pd.DataFrame:
    path = Path("data/raw/etf_prices.csv")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    adj_close = df["Adj Close"][TICKERS]
    log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
    return log_returns

def train_val_test_split(
    dates, train_frac=0.6, val_frac=0.2
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = sorted(dates)
    n = len(dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = dates[:n_train]
    val_idx = dates[n_train:n_train + n_val]
    test_idx = dates[n_train + n_val:]
    return train_idx, val_idx, test_idx

def main():
    df = load_data()
    log_rets = compute_log_returns(df)

    train_idx, val_idx, test_idx = train_val_test_split(log_rets.index)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_rets.to_csv(out_dir / "log_returns.csv")
    pd.Series(train_idx).to_csv(out_dir / "train_dates.csv", index=False)
    pd.Series(val_idx).to_csv(out_dir / "val_dates.csv", index=False)
    pd.Series(test_idx).to_csv(out_dir / "test_dates.csv", index=False)

    print("Saved processed returns and split indices.")

if __name__ == "__main__":
    main()
