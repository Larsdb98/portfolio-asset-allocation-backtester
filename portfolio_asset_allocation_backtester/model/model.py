from .markowitz_efficient_frontier import (
    MarkowitzEfficientFrontier,
)
from .monte_carlo import MonteCarloSim

import yfinance
import datetime
import pandas as pd


class Model:
    def __init__(self):
        self.securities_list = None
        self.securities_df = pd.DataFrame(columns=["Ticker"])

    def fetch_data(
        self, start_date: datetime, end_date: datetime, granularity: str = "1d"
    ):
        """granularity: Data interval, defaults to "1d" but can take: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo"""
