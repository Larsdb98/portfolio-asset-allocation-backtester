import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from typing import Tuple


class MarkowitzEfficientFrontier:
    def __init__(self, instrument_prices_df: pd.DataFrame):
        """
        Class to compute optimal portfolio allocation and compute the Markowitz Efficient Frontier.

        Parameters
        ----------
        instrument_prices_df: dataframe with prices of multiple instruments
        """
        self.instrument_prices_df = instrument_prices_df
        self.log_returns = (
            self.instrument_prices_df / self.instrument_prices_df.shift(1)
        ).apply(np.log)
        self.mean_log_return = self.log_returns.mean()
        self.sigma = self.log_returns.cov()
        self.number_of_instruments = len(self.instrument_prices_df.columns.to_list())

        self.__optimal_weights = None
        self.__optimal_sharpe = None
        self.__optimisation_method = None

        self.__all_weights = None
        self.__all_sharpe = None
        self.__all_expected_volatility = None
        self.__all_expected_log_returns = None
        self.__max_sharpe_index = None

        self.markowitz_optimal_volatility = None
        self.__linspaced_returns_for_efficient_frontier = None

    def stochastic_optimisation_portfolio_allocation(
        self, portfolio_count: int = 10000
    ) -> None:
        """
        Compute the stochastically optimal portfolio allocation.
        Works on a brute-force approach. Therefore the larger the "portfolio_count", the greater chance of approaching the true optimal weights for portfolio allocation.

        Parameters
        ----------
        portfolio_count: number of portfolios to simulate. Defaults at 10000
        """
        self.__optimisation_method = "stochastic"
        weight = np.zeros((portfolio_count, self.number_of_instruments))
        expected_log_returns = np.zeros(portfolio_count)
        expected_volatility = np.zeros(portfolio_count)
        sharpe_ratio = np.zeros(portfolio_count)

        for k in range(portfolio_count):
            w = np.random.random(self.number_of_instruments)
            w = w / np.sum(w)
            weight[k, :] = w

            expected_log_returns[k] = np.sum(self.mean_log_return * w)
            expected_volatility[k] = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))

            sharpe_ratio[k] = expected_log_returns[k] / expected_volatility[k]

        max_sharpe_id = sharpe_ratio.argmax()

        self.__optimal_weights = weight[max_sharpe_id, :]
        self.__optimal_sharpe = sharpe_ratio[max_sharpe_id]
        self.__all_sharpe = sharpe_ratio
        self.__all_weights = weight
        self.__all_expected_volatility = expected_volatility
        self.__all_expected_log_returns = expected_log_returns
        self.__max_sharpe_index = max_sharpe_id

    def optimise_portfolio_allocation(self):
        """
        Computes the optimal weights tu maximize returns of a portfolio in the Least Squares sense.
        """
        self.__optimisation_method = "SLSQP"

        w0 = 1 / self.number_of_instruments * np.ones(self.number_of_instruments)
        bounds = tuple([(0, 1)] * self.number_of_instruments)
        constraints = {"type": "eq", "fun": self.__check_normalized_weights}

        w_optimal = minimize(
            self.__negativeSR_cost_func,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return w_optimal

    def __check_normalized_weights(self, w):
        return np.sum(w) - 1

    def __volatility_cost_func(self, w):
        V = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))
        return V

    def __negativeSR_cost_func(self, w):
        R = np.sum(self.mean_log_return * w)
        V = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))
        SR = R / V
        return -1 * SR

    def __get_return_func(self, w):
        R = np.sum(self.mean_log_return * w)
        return R

    def compute_optimized_markowitz_frontier(self, max_log_returns: float = 0.05):
        """
        Computes the Markowitz Efficient Frontier

        Parameters
        ----------
        max_log_returns: for display purposes. Defaults at 0.05 but should be adjusted based on the preliminary graphs generated.
        """
        log_returns = np.linspace(0, max_log_returns, 50)
        w0 = 1 / self.number_of_instruments * np.ones(self.number_of_instruments)
        bounds = tuple([(0, 1)] * self.number_of_instruments)

        volatility_opt = []

        for return_ in log_returns:
            # compute best volatility
            constraints = (
                {
                    "type": "eq",
                    "fun": self.__check_normalized_weights,
                },
                {"type": "eq", "fun": lambda w: self.__get_return_func(w) - return_},
            )
            opt = minimize(
                self.__volatility_cost_func,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            volatility_opt.append(opt["fun"])

        self.markowitz_optimal_volatility = volatility_opt
        self.__linspaced_returns_for_efficient_frontier = log_returns

    def plot_stochastic_optimised_frontier(
        self,
        figsize: Tuple[int, int] = (16, 10),
        title: str = "Expected Volatility Against Expected Log Returns",
    ) -> None:
        """
        Computes and plots the Markowitz Efficient Frontier

        Parameters
        ----------
        figsize: matplotlib figure size. Defaults with (16, 10)
        title: graph title
        """
        if (
            self.__all_expected_volatility is not None
            and self.__optimisation_method == "stochastic"
        ):
            plt.figure(figsize=figsize)
            plt.scatter(
                self.__all_expected_volatility,
                self.__all_expected_log_returns,
                c=self.__all_sharpe,
            )
            plt.scatter(
                self.__all_expected_volatility[self.__max_sharpe_index],
                self.__all_expected_log_returns[self.__max_sharpe_index],
                c="red",
            )
            plt.xlabel("Expected Volatility")
            plt.ylabel("Expected Log Returns")
            plt.title(title)
            plt.colorbar(label="SR")
            plt.show()
        else:
            raise ValueError(
                "Stochastically optimal frontier has not been computed yet !"
            )

    def plot_optimal_markowitz_frontier(
        self,
        figsize: Tuple[int, int] = (16, 10),
        max_log_returns: float = 0.05,
        title: str = "Expected Volatility Against Expected Log Returns",
    ) -> None:
        """
        Plot the Markowitz Efficient Frontier with the stochastically computed portfolio returns.

        Parameters
        ----------
        figsize: matplotlib figure size. Defaults with (16, 10)
        max_log_returns: for display purposes. Defaults at 0.05 but should be adjusted based on the preliminary graphs generated.
        title: graph title
        """
        # Recompute efficient frontier
        self.compute_optimized_markowitz_frontier(max_log_returns=max_log_returns)

        if self.__all_weights is None:
            self.stochastic_optimisation_portfolio_allocation()

        plt.figure(figsize=figsize)
        plt.scatter(
            self.__all_expected_volatility,
            self.__all_expected_log_returns,
            c=self.__all_sharpe,
        )
        plt.scatter(
            self.__all_expected_volatility[self.__max_sharpe_index],
            self.__all_expected_log_returns[self.__max_sharpe_index],
            c="red",
        )

        plt.plot(
            self.markowitz_optimal_volatility,
            self.__linspaced_returns_for_efficient_frontier,
            "r",
        )
        plt.xlabel("Expected Volatility")
        plt.ylabel("Expected Log Returns")
        plt.title(title)
        plt.colorbar(label="Sharpe Ratio")
        plt.show()

    @property
    def optimisation_method(self):
        return self.__optimisation_method

    @property
    def optimal_weights(self):
        return self.__optimal_weights

    @property
    def optimal_sharpe_ratio(self):
        return self.__optimal_sharpe


def sharpe_ratio(instrument_prices_df: pd.DataFrame):
    # TODO: Implement this
    pass
