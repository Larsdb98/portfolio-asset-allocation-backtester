import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MonteCarloSim:
    def __init__(
        self, instrument_prices_df: pd.DataFrame, initial_portfolio_value: float = 1.0
    ):
        # Will need to heavily modify once instrument_prices_df's structure is defined
        self.instrument_prices_df = instrument_prices_df.dropna()
        self.initial_portfolio_value = initial_portfolio_value
        self.returns = instrument_prices_df.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_returns = self.returns.cov()
        self.weights = np.random.random(len(self.mean_returns))
        self.weights /= np.sum(self.weights)  # Normalize

        self.portfolio_returns = None

    def simulate(self, number_of_sim: int = 100, timeframe: int = 100) -> np.ndarray:
        portfolio_sims = np.full(shape=(timeframe, number_of_sim), fill_value=0.0)

        mean_M = np.full(
            shape=(timeframe, len(self.weights)), fill_value=self.mean_returns
        )
        mean_M = mean_M.T

        for m in range(0, number_of_sim):
            Z = np.random.normal(size=(timeframe, len(self.weights)))
            L = np.linalg.cholesky(self.cov_returns)
            daily_returns = mean_M + np.inner(L, Z)
            portfolio_sims[:, m] = (
                np.cumprod(np.inner(self.weights, daily_returns.T) + 1)
                * self.initial_portfolio_value
            )

        self.portfolio_returns = portfolio_sims
        return portfolio_sims

    def plot_sim_returns(self):
        if self.portfolio_returns is not None:
            plt.plot(self.portfolio_returns)
            plt.ylabel("Portfolio Value")
            plt.xlabel("Timeframe")
            plt.title("Monte Carlo Simulation of Stock Portfolio")
            plt.show()
        else:
            raise ValueError("Monte Carlo simulation has not yet been performed !")

    def value_at_risk(self, percentile: int = 5) -> np.ndarray:
        """Compute the Value at Risk (VaR) given a confidence value "percentile"
        Parameters
        ----------
        percentile: percentile for VaR
        """
        return np.percentile(self.portfolio_returns, percentile)

    def conditional_value_at_risk(self, percentile: int = 5):
        """Compute Conditional Value at Risk (CVaR) / Expected Shortfall given a confidence value "percentile"
        Parameters
        ----------
        percentile: percentile for VaR
        """
        final_values = self.portfolio_returns[-1, :]
        var_threshold = np.percentile(final_values, percentile)
        return final_values[final_values <= var_threshold].mean()

    def get_var_cvar_from_monte_carlo_sampling(self, percentile: int = 5):
        if self.portfolio_returns is None:
            raise ValueError(
                "You need to perform Monte Carlo sampling on the instruments first !"
            )
        else:
            VaR = self.initial_portfolio_value - self.value_at_risk(
                percentile=percentile
            )
            CVaR = self.initial_portfolio_value - self.conditional_value_at_risk(
                percentile=percentile
            )
            output_dict = {"VaR": VaR, "CVaR": CVaR}
            return output_dict
