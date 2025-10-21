from ..model import Model

import plotly.graph_objects as go
import panel as pn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


class DetailedAnalyticsView:
    def __init__(self, model: Model, css_style_path: Path = None):
        self.model = model

        if css_style_path is not None:
            with open(css_style_path, "r") as f:
                self.css = f.read()
        else:
            self.css = ""

        pn.extension(
            "plotly", "mathjax", raw_css=[self.css], sizing_mode="stretch_width"
        )

        self.metrics_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["Metric", "Sample Portfolio"]),
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=40,
            sizing_mode="stretch_both",
            layout="fit_columns",  # key: stretch columns to fit width
            header_align="center",
            text_align="center",
            theme="simple",
        )
        self.metrics_table.columns = [
            {"field": "Metric", "title": "Metric", "widthGrow": 1},
            {"field": "Sample Portfolio", "title": "Sample Portfolio", "widthGrow": 3},
        ]

        self.performance_plot = pn.pane.Plotly(
            self._empty_plot(), sizing_mode="stretch_width"
        )

        self.detailed_analytics_tab = pn.Column(
            pn.pane.Markdown("## Detailed Portfolio Analytics", align="center"),
            self.metrics_table,
            # pn.layout.Spacer(height=15),
            pn.pane.Markdown(
                "## Monte Carlo Sampling Results of Portfolio Allocation",
                align="center",
            ),
            self.performance_plot,
            pn.layout.Spacer(height=15),
            sizing_mode="stretch_width",
        )

    def _empty_plot(self) -> go.Figure:
        """Return an empty placeholder plotly figure for initial state."""
        fig = go.Figure()
        fig.add_annotation(
            text="Run the backtest to see results",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            title=dict(text="Expected Volatility Against Expected Log Returns", x=0.5),
            xaxis_title="Expected Volatility",
            yaxis_title="Expected Log Returns",
            template="plotly_white",
            width=960,
            height=600,
        )
        return fig

    def loading_figure(self, text="Running backtest, please wait...") -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(
            height=400, margin=dict(l=0, r=0, t=40, b=0), template="plotly_white"
        )
        return fig

    def stochastic_optimised_frontier_plotly(
        self,
        all_expected_volatility: Optional[np.ndarray] = None,
        all_expected_log_returns: Optional[np.ndarray] = None,
        all_sharpe: Optional[np.ndarray] = None,
        max_sharpe_index: Optional[int] = None,
        optimisation_method: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        title: str = "Expected Volatility Against Expected Log Returns",
    ) -> go.Figure:
        """
        Generate or update the stochastic optimisation frontier plot.

        If parameters are None or optimisation method != 'stochastic',
        returns the placeholder figure.
        """
        # If data missing or wrong mode — return placeholder
        if (
            all_expected_volatility is None
            or all_expected_log_returns is None
            or all_sharpe is None
            or optimisation_method != "stochastic"
        ):
            return self._empty_plot()

        # Base scatter of all stochastic portfolios
        scatter = go.Scatter(
            x=all_expected_volatility,
            y=all_expected_log_returns,
            mode="markers",
            marker=dict(
                size=6,
                color=all_sharpe,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
            ),
            name="Stochastic Portfolios",
            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<br>"
                "Sharpe Ratio: %{marker.color:.2f}<extra></extra>"
            ),
        )

        # Highlight the maximum Sharpe portfolio
        max_sharpe = go.Scatter(
            x=[all_expected_volatility[max_sharpe_index]],
            y=[all_expected_log_returns[max_sharpe_index]],
            mode="markers",
            marker=dict(color="red", size=12, symbol="star"),
            name="Max Sharpe Portfolio",
            hovertemplate="Max Sharpe Portfolio<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )

        base_height = int(figsize[1] * 60)
        min_height = 900
        height = max(base_height, min_height)

        fig = go.Figure(data=[scatter, max_sharpe])
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="Expected Volatility",
            yaxis_title="Expected Log Returns",
            width=int(figsize[0] * 60),
            height=height,
            template="plotly_white",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.6)"),
        )

        return fig

    def update_metrics_table(self, metrics_df: pd.DataFrame) -> None:
        """Update the analytics table with a new metrics DataFrame."""
        formatted_df = metrics_df.copy()
        formatted_df.index.name = "Metric"
        formatted_df.reset_index(inplace=True)

        self.metrics_table.value = formatted_df
