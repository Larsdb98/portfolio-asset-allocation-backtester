from .markowitz_efficient_frontier import (
    MarkowitzEfficientFrontier,
)
from .monte_carlo import MonteCarloSim

import yfinance as yf
import datetime
import pandas as pd


class Model:
    def __init__(self):
        self.instrument_list = None
        self.instrument_df = pd.DataFrame(columns=["Ticker"])
        self.prices_df = pd.DataFrame()

    def add_instrument(self, ticker) -> None:
        if ticker and ticker not in self.instrument_df["Ticker"].values:
            new_row = pd.DataFrame([[ticker]], columns=["Ticker"])
            self.instrument_df = pd.concat(
                [self.instrument_df, new_row], ignore_index=True
            )

    def delete_selected_instrument(self, selected) -> None:
        if selected:
            # selection is a list of row indices
            self.instrument_df = self.instrument_df.drop(index=selected).reset_index(
                drop=True
            )

    # ---------------------------
    # Data fetching
    # ---------------------------
    def fetch_data(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        granularity: str = "1d",
    ):
        """
        Fetch closing prices for all tickers in instrument_df.
        granularity options: "1m", "2m", "5m", "15m", "30m", "60m",
                             "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
        """
        if self.instrument_df.empty:
            return pd.DataFrame()

        tickers = self.instrument_df["Ticker"].tolist()
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=granularity,
            progress=False,
            group_by="ticker",
            auto_adjust=True,
        )

        # yfinance returns multi-index if multiple tickers
        if len(tickers) == 1:
            prices = data["Close"].to_frame(name=tickers[0])
        else:
            prices = data["Close"]

        self.prices_df = prices
        return self.prices_df

    def _fetch_single_ticker(self, ticker: str, start_date, end_date, granularity="1d"):
        """Fetch a single ticker's closing prices and merge into prices_df."""
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=granularity,
            progress=False,
            auto_adjust=True,
        )

        if "Close" not in data.columns:
            return

        series = data["Close"].rename(ticker)

        if self.prices_df.empty:
            self.prices_df = series.to_frame()
        else:
            # Outer join to align dates
            self.prices_df = self.prices_df.join(series, how="outer")
