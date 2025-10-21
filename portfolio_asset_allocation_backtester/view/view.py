from ..model import Model
from .instrument_table_widget import InstrumentTableWidget
from .highlights_view import HighlightsView
from .detailed_analytics_view import DetailedAnalyticsView
from ..utils import GRANULARITY_DICT

import numpy as np
from pathlib import Path
import panel as pn
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta


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
        self.highlights_view = HighlightsView(
            model=self.model, css_style_path=css_style_path
        )

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

        # Allocation Table
        alloc_columns = [
            "Symbol",
            "Name",
            "Stochastic Weight Allocation (%)",
            "Optimal Weight Allocation (%)",
        ]
        self.allocation_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=alloc_columns),
            height=250,
            widths={"Symbol": 100, "Name": 120},
            disabled=True,
            show_index=False,
            layout="fit_columns",
            pagination=None,
            theme="materialize",
        )

        self.allocation_table.columns = [
            {"field": "Name", "title": "Name", "widthGrow": 4},
            {
                "field": "Stochastic Weight Allocation (%)",
                "title": "Stochastic Weight Allocation (%)",
                "widthGrow": 1,
            },
            {
                "field": "Optimal Weight Allocation (%)",
                "title": "Optimal Weight Allocation (%)",
                "widthGrow": 1,
            },
        ]

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

        # Other tabs
        self.detailed_analytics_view = DetailedAnalyticsView(
            model=self.model, css_style_path=css_style_path
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
                    align="center",
                    sizing_mode="stretch_both",
                    styles={"margin": "auto 0"},
                    # width=200,
                ),
                self.weight_pie_chart,
                sizing_mode="stretch_width",
                align="center",
            ),
            pn.layout.Spacer(height=15),
            self.highlights_view.highlights_section,
            pn.layout.Spacer(height=15),
        )

        # Tabs
        self.tabs = pn.Tabs(
            ("Overview", self.overview_tab),
            (
                "Detailed Statistics",
                self.detailed_analytics_view.detailed_analytics_tab,
            ),
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

    def _empty_pie(self) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text="No weights yet", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", width=400, height=400)
        return fig

    # ------------------------------------------------------------------
    # UPDATE METHODS
    # ------------------------------------------------------------------
    def update_allocation_table(self, data: pd.DataFrame) -> None:
        """Populate the allocation table."""
        self.allocation_table.value = data

    def update_pie_chart(
        self, weights: pd.Series, labels: list, title: str = "Portfolio Distribution"
    ) -> None:
        """Update pie chart with weight distribution."""
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=weights, textinfo="label+percent")]
        )
        fig.update_layout(
            title=dict(text=title, x=0.5), template="plotly_white", height=400
        )
        self.weight_pie_chart.object = fig

    def update_weight_pie_chart(self, tickers: list, weights: list, title: str) -> None:
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

    def run(self, port=5006):
        self.template.servable()
        pn.serve(self.template, port=port, show=True)
