import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["SPY", "TLT", "GLD", "SHY"]
START = "2010-01-01"
END = "2024-12-31"

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
    )

    out_path = out_dir / "etf_prices.csv"
    data.to_csv(out_path)
    print(f"Saved raw data to {out_path}")

if __name__ == "__main__":
    main()
