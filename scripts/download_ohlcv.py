import argparse
from pathlib import Path

import yfinance as yf


def download_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    out_path: str,
) -> None:
    """
    Download OHLCV data from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. 'SPY', '^GSPC', 'AAPL'.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    interval : str
        Data frequency, e.g. '1d', '1h', '1m' (check yfinance docs).
    out_path : str
        CSV path to save data.
    """
    print(f"[INFO] Downloading {ticker} from {start} to {end} with interval={interval}...")
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Downloaded DataFrame is empty. "
                           "Check ticker / date range / interval.")

    df = df.reset_index()  # ensure Date is a column, not index
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved OHLCV data to {out_path.resolve()}")
    print(f"[INFO] Data shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data via yfinance.")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Ticker symbol, e.g. 'SPY', '^GSPC', 'AAPL'.")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", type=str, required=True,
                        help="End date in YYYY-MM-DD.")
    parser.add_argument("--interval", type=str, default="1d",
                        help="Data frequency, e.g. '1d', '1h', '1m'. Default: 1d.")
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/raw/ohlcv.csv",
        help="Output CSV path. Default: data/raw/ohlcv.csv",
    )
    args = parser.parse_args()

    download_ohlcv(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
