from ..model import Model
from .instrument_table_widget import InstrumentTableWidget
from ..utils import GRANULARITY_DICT

import numpy as np
from pathlib import Path
import panel as pn
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


class View:
    def __init__(
        self,
        model: Model,
        css_style_path: Path = None,
        dashboard_title: str = "Portfolio Asset Allocation Backtester",
    ):
        self.model = model
        self.css_style_path = css_style_path
        self.dashboard_title = dashboard_title

        default_end_date = datetime.today()
        default_start_date = default_end_date - timedelta(days=365 * 3)

        self.instrument_table_widget = InstrumentTableWidget()

        self.title_widget = pn.pane.Markdown(
            "# Portfolio Asset Allocation", css_classes=["app-title"]
        )
        self.subtitle_widget = pn.pane.Markdown(
            "Compute call/put option prices using Black-Scholes or the Binomial pricing models.",
            css_classes=["app-sub"],
        )
        if css_style_path is not None:
            with open(css_style_path, "r") as f:
                self.css = f.read()
        else:
            self.css = ""

        pn.extension(
            "plotly", "mathjax", raw_css=[self.css], sizing_mode="stretch_width"
        )

        # Sidebar widgets
        self.start_date = pn.widgets.DatePicker(
            name="Start Date", value=default_start_date
        )
        self.end_date = pn.widgets.DatePicker(name="End Date", value=default_end_date)

        self.granularity_input = pn.widgets.Select(
            name="Granularity", options=GRANULARITY_DICT
        )

        self.initial_amount = pn.widgets.IntInput(
            name="Initial Investment ($)", value=100_000
        )

        self.run_button = pn.widgets.Button(name="Run Backtest", button_type="primary")

        # PlaceHolder for outputs
        self.performance_plot = pn.pane.Plotly(
            self._empty_plot(), sizing_mode="stretch_both"
        )

        dummy_df = pd.DataFrame(columns=["A", "B", "C"])
        self.stats_table = pn.pane.DataFrame(
            dummy_df,
            width=100,
            height=300,
        )

        # Tabs
        self.tabs = pn.Tabs(
            ("Overview", self.performance_plot),
            ("Statistics", self.stats_table),
            ("Allocations", pn.pane.Markdown("Allocations view placeholder")),
        )

        # Layout
        self.sidebar = pn.Column(
            "### Parameters",
            self.start_date,
            self.end_date,
            self.granularity_input,
            self.initial_amount,
            self.instrument_table_widget.panel(),
            self.run_button,
            width=300,
        )

        self.main_widget = pn.Column(
            f"# {self.dashboard_title}", self.tabs, sizing_mode="stretch_both"
        )

        self.template = pn.template.FastListTemplate(
            title=self.dashboard_title,
            sidebar=[self.sidebar],
            main=[self.main_widget],
            accent_base_color="#006699",
            header_background="#00334d",
        )

    def _empty_plot(self):
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

    def stochastic_optimised_frontier_plotly(
        self,
        all_expected_volatility: Optional[np.ndarray] = None,
        all_expected_log_returns: Optional[np.ndarray] = None,
        all_sharpe: Optional[np.ndarray] = None,
        max_sharpe_index: Optional[int] = None,
        optimisation_method: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        title: str = "Expected Volatility Against Expected Log Returns",
    ):
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

        # Build figure
        fig = go.Figure(data=[scatter, max_sharpe])
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="Expected Volatility",
            yaxis_title="Expected Log Returns",
            width=int(figsize[0] * 60),
            height=int(figsize[1] * 60),
            template="plotly_white",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.6)"),
        )

        return fig

    def loading_figure(self, text="Running backtest, please wait..."):
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

    def run(self, port=5006):
        self.template.servable()
        pn.serve(self.template, port=port, show=True)
