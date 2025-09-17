import os
from pathlib import Path
from .model import Model

import panel as pn
import plotly.graph_objects as go
import pandas as pd


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
        self.start_date = pn.widgets.DatePicker(name="Start Date")
        self.end_date = pn.widgets.DatePicker(name="End Date")
        self.initial_amount = pn.widgets.IntInput(
            name="Initial Investment ($)", value=100_000
        )

        self.asset_allocation = pn.widgets.TextAreaInput(
            name="Asset Allocation (JSON)",
            value='{"Stocks": 0.6, "Bonds": 0.4}',
            height=100,
        )
        self.run_button = pn.widgets.Button(name="Run Backtest", button_type="primary")

        # PlaceHolder for outputs
        self.performance_plot = pn.pane.Plotly(
            self._empty_plot(), sizing_mode="stretch_both"
        )
        self.stats_table = pn.pane.DataFrame(pd.DataFrame, width=100, height=300)

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
            self.initial_amount,
            self.asset_allocation,
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
        fig = go.Figure()
        fig.add_annotation(
            text="Run the backtext to see results",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
        )

    def run(self, port=5006):
        self.template.servable()
        pn.serve(self.template, port=port, show=True)
