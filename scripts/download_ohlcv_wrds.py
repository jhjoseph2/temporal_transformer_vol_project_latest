import argparse
from pathlib import Path

import pandas as pd
import wrds


def download_crsp_ohlcv_for_ticker(
    conn: wrds.Connection,
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download daily OHLCV for a single equity ticker from CRSP via WRDS.

    We use the CRSP Daily Stock File (crsp.dsf) joined with crsp.stocknames.

    Notes
    -----
    - Prices in CRSP can be negative to indicate bid/ask averages. We take
      absolute values for Open, High, Low, Close.
    - Volume is in shares.
    - Date filter is inclusive between `start` and `end`.
    """
    query = f"""
    select a.date,
           a.openprc as open,
           a.askhi    as high,
           a.bidlo     as low,
           a.prc     as close,
           a.vol     as volume
    from crsp.dsf as a
    join crsp.stocknames_v2 as b
      on a.permno = b.permno
    where b.ticker = '{ticker}'
      and a.date between '{start}' and '{end}'
      and b.namedt <= a.date
      and b.nameenddt >= a.date
    order by a.date;
    """

    df = conn.raw_sql(query)
    if df.empty:
        raise RuntimeError(
            "Empty result from CRSP; check ticker, date range, or your WRDS access."
        )

    # Take absolute values for prices (CRSP uses signed prices)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].abs()

    # Ensure we have a proper Date column with standard naming
    df.rename(columns={"date": "Date",
                       "open": "Open",
                       "high": "High",
                       "low": "Low",
                       "close": "Close",
                       "volume": "Volume"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV data from CRSP via WRDS."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol as in CRSP.stocknames, e.g. 'SPY'.",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/raw/spy_daily_wrds.csv",
        help="Output CSV path. Default: data/raw/spy_daily_wrds.csv",
    )
    args = parser.parse_args()

    print("[INFO] Connecting to WRDS...")
    conn = wrds.Connection(wrds_username='jj3476')
    print("[INFO] Connected. Querying CRSP...")

    df = download_crsp_ohlcv_for_ticker(
        conn=conn,
        ticker=args.ticker,
        start=args.start,
        end=args.end,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[INFO] Saved OHLCV data to {out_path.resolve()}")
    print(f"[INFO] Data shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
