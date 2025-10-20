from .markowitz_efficient_frontier import (
    MarkowitzEfficientFrontier,
)
from .yfinance_fetcher import YFinanceFetcher, yfinance_ticker_info

import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import datetime


class Model:
    def __init__(self):
        self.instrument_list = None
        self.instrument_df = pd.DataFrame(columns=["Ticker"])
        self.prices_df = pd.DataFrame()
        self.closing_prices_df = pd.DataFrame()

        self.markow_frontier = None
        self.markowitz_plot_data = None

        self.initial_portfolio_value = None
        self.final_portfolio_value = None
        self.portfolio_stats: Dict[str, Any] = None

        self.__optimal_weights = None
        self.__stoch_optimal_weights = None

        self.__yfinance_ticker_info = None

        self.allocation_df_raw = pd.DataFrame()
        self.allocation_df_display = pd.DataFrame()

    def update_initial_portfolio_value(self, portfolio_value: float) -> None:
        self.initial_portfolio_value = portfolio_value

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
            self.__yfinance_ticker_info = yfinance_ticker_info(ticker_list)
            self.prices_df = yf_fetcher.get_raw_ohlc_data()
            self.closing_prices_df = yf_fetcher.get_price_data(price_type="Close")
        except Exception as e:
            raise Exception(
                f"Model :: run_backtest :: YFinanceFetcher: ran into the following error: {e}"
            )

        markow_frontier = MarkowitzEfficientFrontier(
            instrument_prices_df=self.closing_prices_df
        )
        self.__optimal_weights = markow_frontier.optimise_portfolio_allocation()
        markow_frontier.stochastic_optimisation_portfolio_allocation(
            portfolio_count=10000  # Maybe this will become an advanced feature input in the future ?
        )
        self.__stoch_optimal_weights = markow_frontier.optimal_weights
        try:
            self.markowitz_plot_data = markow_frontier.get_plot_data
        except Exception as e:
            raise Exception(
                f"Model :: run_backtest: The following exception was caught: {e}"
            )

    def compute_portfolio_statistics(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> None:
        """Compute daily returns, annualized return, volatility, drawdown, and Sharpe ratio."""

        if self.closing_prices_df is None or self.closing_prices_df.empty:
            raise ValueError(
                "Model :: compute_portfolio_statistics: No historical price data available."
            )

        weights = np.array(self.get_optimal_weights)

        daily_returns = self.closing_prices_df.pct_change().dropna()

        portfolio_returns = (daily_returns * weights).sum(axis=1)

        portfolio_value = (1 + portfolio_returns).cumprod()

        trading_days = 252
        mean_daily_return = float(portfolio_returns.mean())
        std_daily = float(portfolio_returns.std())

        annualized_return = float((1 + mean_daily_return) ** trading_days - 1)
        annualized_volatility = float(std_daily * np.sqrt(trading_days))

        # --- Drawdown computation ---
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value / rolling_max) - 1
        max_drawdown = float(drawdown.min())

        # --- Sharpe ratio (risk-free = 0 assumed) ---
        sharpe_ratio = float(mean_daily_return / std_daily * np.sqrt(trading_days))

        # --- Store in model for controller access ---
        self.portfolio_stats = {
            "daily_returns": portfolio_returns,
            "portfolio_value": portfolio_value,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

    @property
    def get_initial_portfolio_value(self) -> int:
        if self.initial_portfolio_value is not None:
            return self.initial_portfolio_value
        else:
            raise ValueError(
                "Model :: get_initial_portfolio_value: initial portfolio value has not been properly registered."
            )

    @property
    def get_markowitz_plot_data(self) -> Dict[str, np.typing.NDArray]:
        if self.markowitz_plot_data is not None:
            return self.markowitz_plot_data
        else:
            raise Exception(
                "Model :: get_markowitz_plot_data: Markowitz data has not been generated."
            )

    @property
    def get_stochastic_optimal_weights(self) -> List[int]:
        if self.__stoch_optimal_weights is not None:
            return self.__stoch_optimal_weights
        else:
            raise ValueError(
                "Model :: Stochastically optimal weights have not been computed yet !"
            )

    @property
    def get_optimal_weights(self) -> List[int]:
        if self.__optimal_weights is not None:
            return self.__optimal_weights
        else:
            raise ValueError("Model :: Optimal weights have not been computed yet !")

    @property
    def get_ticker_long_names(self) -> List[str]:
        if self.__yfinance_ticker_info is not None:
            ret = [
                self.__yfinance_ticker_info[ticker]["longName"]
                for ticker in self.instrument_df["Ticker"].to_list()
            ]
            return ret
        else:
            raise ValueError(
                "Model :: Ticker information has not yet been retrieved ! Run the backtest first..."
            )
