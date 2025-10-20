from ..model import Model
from .instrument_table_widget import InstrumentTableWidget
from .highlights_view import HighlightsView
from ..utils import GRANULARITY_DICT

import numpy as np
from pathlib import Path
import panel as pn
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict


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
        self.highlights_view = HighlightsView(model=self.model)

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

        # Allocation Table
        alloc_columns = [
            "Symbol",
            "Name",
            "Stochastic Weight Allocation (%)",
            "Optimal Weight Allocation (%)",
        ]
        self.allocation_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=alloc_columns),
            height=200,
            widths={"Symbol": 100, "Name": 120},
            disabled=True,
            pagination=None,
            theme="materialize",
        )

        # Pie chart and toggle
        self.weight_type_toggle = pn.widgets.RadioButtonGroup(
            name="Weight Type",
            options=["Optimal", "Stochastic"],
            button_type="primary",
            value="Optimal",
        )
        self.weight_pie_chart = pn.pane.Plotly(self._empty_pie(), height=400)

        dummy_df = pd.DataFrame(columns=["A", "B", "C"])
        self.stats_table = pn.pane.DataFrame(
            dummy_df,
            width=100,
            height=300,
        )

        # ------------------------------------------------------------------
        # LAYOUT SECTIONS
        # ------------------------------------------------------------------
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

        self.overview_tab = pn.Column(
            "## Portfolio Allocations",
            self.allocation_table,
            pn.layout.Spacer(height=15),
            pn.Row(
                pn.Column(
                    self.weight_type_toggle,
                    styles={
                        "display": "flex",
                        "align-items": "center",  # vertical centering
                        "justify-content": "center",  # horizontal centering
                        "height": "100%",
                    },
                    # width=200,
                ),
                self.weight_pie_chart,
            ),
            pn.layout.Spacer(height=15),
            self.highlights_view.highlights_section,
            pn.layout.Spacer(height=15),
            self.performance_plot,
        )

        # Tabs
        self.tabs = pn.Tabs(
            ("Overview", self.overview_tab),
            ("Statistics", pn.pane.Markdown("Statistics view placeholder")),
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

    # ------------------------------------------------------------------
    # FIGURE HELPERS
    # ------------------------------------------------------------------

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

    def _empty_pie(self):
        fig = go.Figure()
        fig.add_annotation(text="No weights yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", width=400, height=400)
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

    # ------------------------------------------------------------------
    # UPDATE METHODS (to be called from controller)
    # ------------------------------------------------------------------
    def update_allocation_table(self, data: pd.DataFrame):
        """Populate the allocation table."""
        self.allocation_table.value = data

    def update_pie_chart(
        self, weights: pd.Series, labels: list, title: str = "Portfolio Distribution"
    ):
        """Update pie chart with weight distribution."""
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=weights, textinfo="label+percent")]
        )
        fig.update_layout(
            title=dict(text=title, x=0.5), template="plotly_white", height=400
        )
        self.weight_pie_chart.object = fig

    def update_weight_pie_chart(self, tickers: list, weights: list, title: str):
        """Update the weight distribution pie chart."""
        # Ensure numeric weights
        weights = np.array(weights, dtype=float)

        # Normalize just to be safe
        total = weights.sum()
        if total > 0:
            weights = weights / total

        percent_labels = [f"{w * 100:.2f}%" for w in weights]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=tickers,
                    values=weights,
                    text=[f"{t} — {p}" for t, p in zip(tickers, percent_labels)],
                    hoverinfo="text",
                    textinfo="percent+label",
                    texttemplate="%{label}<br>%{value:.2%}",
                    insidetextorientation="radial",
                    hole=0.4,
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white",
        )

        self.weight_pie_chart.object = fig

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

        base_height = int(figsize[1] * 60)
        min_height = 900  # minimum in pixels
        height = max(base_height, min_height)

        # Build figure
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

    def run(self, port=5006):
        self.template.servable()
        pn.serve(self.template, port=port, show=True)
