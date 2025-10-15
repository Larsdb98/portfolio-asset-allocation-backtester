from .markowitz_efficient_frontier import (
    MarkowitzEfficientFrontier,
)
from .yfinance_fetcher import YFinanceFetcher

import datetime
import pandas as pd
import numpy as np
from typing import Dict
import datetime


class Model:
    def __init__(self):
        self.instrument_list = None
        self.instrument_df = pd.DataFrame(columns=["Ticker"])
        self.prices_df = pd.DataFrame()
        self.closing_prices_df = pd.DataFrame()

        self.markow_frontier = None
        self.markowitz_plot_data = None

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

    def run_backtest(
        self, start_date: datetime.date, end_date: datetime.date, interval: str = "1d"
    ):
        ticker_list = self.instrument_df["Ticker"].to_list()
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        if len(ticker_list) < 2:
            raise ValueError(
                f"Warning: Not enough tickers have been provided ! Only {len(ticker_list)} have been given."
            )

        print("Model :: run_backtest: List of tickers given:")
        print(ticker_list)

        # Fetch yFinance prices
        try:
            yf_fetcher = YFinanceFetcher(
                start=start_date_str,
                end=end_date_str,
                tickers=ticker_list,
                interval=interval,
            )
            self.prices_df = yf_fetcher.get_raw_ohlc_data()
            self.closing_prices_df = yf_fetcher.get_price_data(price_type="Close")
        except Exception as e:
            raise Exception(
                f"Model :: run_backtest :: YFinanceFetcher: ran into the following error: {e}"
            )

        markow_frontier = MarkowitzEfficientFrontier(
            instrument_prices_df=self.closing_prices_df
        )
        markow_frontier.stochastic_optimisation_portfolio_allocation(
            portfolio_count=10000  # Maybe this will become an advanced feature input in the future ?
        )
        try:
            self.markowitz_plot_data = markow_frontier.get_plot_data
        except Exception as e:
            raise Exception(
                f"Model :: run_backtest: The following exception was caught: {e}"
            )

    @property
    def get_markowitz_plot_data(self) -> Dict[str, np.typing.NDArray]:
        if self.markowitz_plot_data is not None:
            return self.markowitz_plot_data
        else:
            raise Exception(
                "Model :: get_markowitz_plot_data: Markowitz data has not been generated."
            )
