# src/data_download.py

from pathlib import Path
import yfinance as yf
import pandas as pd

# Universe of ETFs (unchanged)
TICKERS = ["SPY", "TLT", "GLD", "SHY"]

# EXTENDED time span: go as far back as practical
START = "2005-01-01"   # previously 2010-01-01
END = "2024-12-31"     # or whatever end date you like


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {TICKERS} from {START} to {END}...")
    data = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=False,
        progress=True,
    )

    out_path = out_dir / "etf_prices.csv"
    data.to_csv(out_path)
    print(f"Saved raw data to {out_path}")


if __name__ == "__main__":
    main()
