from .markowitz_efficient_frontier import (
    MarkowitzEfficientFrontier,
)
from .yfinance_fetcher import YFinanceFetcher, yfinance_ticker_info

import datetime
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, linregress
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
        self.portfolio_risk_metrics_dict: Dict[str, Any] = None
        self.portfolio_risk_metrics_df: pd.DataFrame = None

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
                f"Model :: run_backtest: Warning: Not enough tickers have been provided ! Only {len(ticker_list)} have been given."
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

        # ==========================================================
        # EXTENDED METRICS SECTION
        # ==========================================================

        # Convert to monthly returns
        monthly_returns = portfolio_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )

        # Benchmark (optional)
        benchmark = getattr(self, "benchmark_returns", None)
        if benchmark is not None:
            benchmark = benchmark.loc[portfolio_returns.index]
            benchmark_monthly = benchmark.resample("M").apply(
                lambda x: (1 + x).prod() - 1
            )
        else:
            benchmark_monthly = pd.Series(index=monthly_returns.index, dtype=float)

        # --- Core stats ---
        mean_monthly = monthly_returns.mean()
        geo_mean_monthly = (np.prod(1 + monthly_returns)) ** (
            1 / len(monthly_returns)
        ) - 1
        std_monthly = monthly_returns.std()

        # Annualized conversions
        mean_annual = (1 + mean_monthly) ** 12 - 1
        geo_mean_annual = (1 + geo_mean_monthly) ** 12 - 1
        std_annual = std_monthly * np.sqrt(12)

        # Downside deviation (monthly)
        downside_returns = monthly_returns[monthly_returns < 0]
        downside_dev_monthly = np.sqrt(np.mean(downside_returns**2))

        # Skewness / Kurtosis
        skewness = skew(portfolio_returns)
        excess_kurtosis = kurtosis(portfolio_returns)

        # Value-at-Risk and CVaR
        var_hist = np.percentile(portfolio_returns, 5)
        var_analytic = mean_daily_return - 1.65 * std_daily
        cvar = portfolio_returns[portfolio_returns <= var_hist].mean()

        # Positive periods and gain/loss
        positive_periods = (monthly_returns > 0).sum()
        gain_loss_ratio = monthly_returns[monthly_returns > 0].mean() / abs(
            monthly_returns[monthly_returns < 0].mean()
        )

        # Benchmark correlation and regression (for Beta, Alpha, R^2)
        if benchmark is not None and benchmark.notna().any():
            aligned = pd.concat([portfolio_returns, benchmark], axis=1).dropna()
            r_port, r_bench = aligned.iloc[:, 0], aligned.iloc[:, 1]
            slope, intercept, r_value, _, _ = linregress(r_bench, r_port)
            beta = slope
            alpha = intercept * trading_days
            r2 = r_value**2
            corr = np.corrcoef(r_port, r_bench)[0, 1]
            active_return = mean_daily_return - r_bench.mean()
            tracking_error = np.std(r_port - r_bench)
            info_ratio = (
                active_return / tracking_error if tracking_error > 0 else np.nan
            )
            upside_capture = (
                100 * np.mean(r_port[r_bench > 0]) / np.mean(r_bench[r_bench > 0])
            )
            downside_capture = (
                100 * np.mean(r_port[r_bench < 0]) / np.mean(r_bench[r_bench < 0])
            )
        else:
            beta = alpha = r2 = corr = active_return = tracking_error = info_ratio = (
                np.nan
            )
            upside_capture = downside_capture = np.nan

        # Risk-adjusted measures
        sortino_ratio = (
            mean_monthly / downside_dev_monthly if downside_dev_monthly > 0 else np.nan
        )
        treynor_ratio = (
            (mean_daily_return / beta) * 100 if beta and not np.isnan(beta) else np.nan
        )
        calmar_ratio = (
            annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        )
        m2_ratio = sharpe_ratio * std_annual + 0  # Modigliani-Modigliani (risk-free=0)
        swr = annualized_return / (1 + annualized_volatility)
        pwr = annualized_return / (1 + 2 * annualized_volatility)

        # Store extended metrics
        self.portfolio_risk_metrics_dict = {
            "Arithmetic Mean (monthly)": mean_monthly,
            "Arithmetic Mean (annualized)": mean_annual,
            "Geometric Mean (monthly)": geo_mean_monthly,
            "Geometric Mean (annualized)": geo_mean_annual,
            "Standard Deviation (monthly)": std_monthly,
            "Standard Deviation (annualized)": std_annual,
            "Downside Deviation (monthly)": downside_dev_monthly,
            "Maximum Drawdown": max_drawdown,
            "Benchmark Correlation": corr,
            "Beta": beta,
            "Alpha (annualized)": alpha,
            "R^2": r2,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Treynor Ratio (%)": treynor_ratio,
            "Calmar Ratio": calmar_ratio,
            "Modigliani–Modigliani Measure": m2_ratio,
            "Active Return": active_return,
            "Tracking Error": tracking_error,
            "Information Ratio": info_ratio,
            "Skewness": skewness,
            "Excess Kurtosis": excess_kurtosis,
            "Historical VaR (5%)": var_hist,
            "Analytical VaR (5%)": var_analytic,
            "Conditional VaR (5%)": cvar,
            "Upside Capture Ratio (%)": upside_capture,
            "Downside Capture Ratio (%)": downside_capture,
            "Safe Withdrawal Rate": swr,
            "Perpetual Withdrawal Rate": pwr,
            "Positive Periods": int(positive_periods),
            "Gain/Loss Ratio": gain_loss_ratio,
        }

        self.portfolio_risk_metrics_df = pd.DataFrame.from_dict(
            self.portfolio_risk_metrics_dict,
            orient="index",
            columns=["Sample Portfolio"],
        )

        # Optional: round numeric values nicely
        self.portfolio_risk_metrics_df[
            "Sample Portfolio"
        ] = self.portfolio_risk_metrics_df["Sample Portfolio"].apply(
            lambda x: (
                f"{x:.2%}"
                if isinstance(x, (int, float)) and abs(x) < 10
                else f"{x:.2f}" if isinstance(x, (int, float)) else x
            )
        )

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
                "Model :: get_stochastic_optimal_weights: Stochastically optimal weights have not been computed yet !"
            )

    @property
    def get_optimal_weights(self) -> List[int]:
        if self.__optimal_weights is not None:
            return self.__optimal_weights
        else:
            raise ValueError(
                "Model :: get_optimal_weights: Optimal weights have not been computed yet !"
            )

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
                "Model :: get_ticker_long_names: Ticker information has not yet been retrieved ! Run the backtest first..."
            )

    @property
    def get_portfolio_risk_metrics_df(self):
        if self.portfolio_risk_metrics_df is not None:
            return self.portfolio_risk_metrics_df
        else:
            raise ValueError(
                "Model :: get_portfolio_risk_metrics_df: Ticker information has not yet been retrieved ! Run the backtest first.."
            )
