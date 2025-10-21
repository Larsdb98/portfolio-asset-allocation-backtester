from ..model import Model

import panel as pn
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


class HighlightsView:
    def __init__(self, model=Model, css_style_path: Path = None):
        self.model = model

        if css_style_path is not None:
            with open(css_style_path, "r") as f:
                self.css = f.read()
        else:
            self.css = ""

        pn.extension(
            "plotly", "mathjax", raw_css=[self.css], sizing_mode="stretch_width"
        )

        self.valuation_plot = pn.pane.Plotly(
            self._empty_portfolio_valuation_plot(),
            sizing_mode="stretch_width",
            height=400,
        )

        self.return_card = pn.Column(
            pn.Row(
                pn.pane.Markdown("**Portfolio Return**", align="center"),
                pn.pane.Markdown("**Annualized Return**", align="center"),
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.pane.HTML(
                    "<div class='metric-box green'>0.0%</div>", align="center"
                ),
                pn.pane.HTML(
                    "<div class='metric-box green'>0.0%</div>", align="center"
                ),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            align="center",
        )

        self.risk_card = pn.Column(
            pn.Row(
                pn.pane.Markdown("**Volatility (Standard Deviation)**", align="center"),
                pn.pane.Markdown("**Maximum Drawdown**", align="center"),
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.pane.HTML(
                    "<div class='metric-box green'>0.0%</div>", align="center"
                ),
                pn.pane.HTML("<div class='metric-box gray'>0.0%</div>", align="center"),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            align="center",
        )

        self.interpretation_text = pn.pane.Markdown(
            "Run backtest first",
            align="center",
            margin=(10, 0, 0, 0),
        )

        self.highlights_section = pn.Column(
            pn.pane.Markdown("## Highlights", styles={"color": "blue"}, align="center"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Return", align="center"),
                    self.return_card,
                    sizing_mode="stretch_width",
                    align="center",
                ),
                pn.Column(
                    pn.pane.Markdown("### Risk", align="center"),
                    self.risk_card,
                    sizing_mode="stretch_width",
                    align="center",
                ),
                sizing_mode="stretch_width",
                align="center",
                margin=(0, 0, 10, 0),
            ),
            pn.layout.Divider(),
            self.interpretation_text,
            self.valuation_plot,
            sizing_mode="stretch_width",
            align="center",
            css_classes=["highlight-section"],
            margin=(10, 15, 25, 15),
        )

    def _empty_portfolio_valuation_plot(self) -> go.Figure:
        """
        Returns a placeholder Plotly figure for the portfolio valuation chart
        (displayed before any backtest is run).
        """
        fig = go.Figure()

        fig.add_annotation(
            text="Portfolio valuation over time will appear here after running the backtest.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
            align="center",
        )

        fig.update_layout(
            title=dict(
                text="Portfolio Valuation Over Time",
                x=0.5,
                font=dict(size=18, color="gray"),
            ),
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=20, t=60, b=40),
        )

        return fig

    def update_portfolio_highlights(self, initial_investment: int):
        try:
            annual_return = self.model.portfolio_stats.get("annualized_return", np.nan)
            volatility = self.model.portfolio_stats.get("annualized_volatility", np.nan)
            drawdown = self.model.portfolio_stats.get("max_drawdown", np.nan)
            sharpe = self.model.portfolio_stats.get("sharpe_ratio", np.nan)
            cumulative_returns = self.model.portfolio_stats.get("portfolio_value")
            start_date = self.model.portfolio_stats.get("start_date")
            end_date = self.model.portfolio_stats.get("end_date")

            final_returns = cumulative_returns.iloc[-1]
            final_value = final_returns * initial_investment

            def fmt_percent(x):
                return f"{x * 100:.2f}%" if pd.notna(x) else "–"

            # Color helper
            def color_box(value, inverse=False):
                if pd.isna(value):
                    return "gray"
                if inverse:
                    # For metrics where lower = better (drawdown)
                    return "green" if value < 0.1 else "red"
                else:
                    return "green" if value > 0.05 else "red"

            # Build updated metric boxes
            return_box = f"<div class='metric-box {color_box((final_returns - 1))}'>{fmt_percent((final_returns - 1))}</div>"
            annualized_return_box = f"<div class='metric-box {color_box(annual_return)}'>{fmt_percent(annual_return)}</div>"
            vol_box = f"<div class='metric-box gray'>{fmt_percent(volatility)}</div>"
            draw_box = f"<div class='metric-box {color_box(abs(drawdown), inverse=True)}'>{fmt_percent(abs(drawdown))}</div>"

            # Update UI elements
            self.return_card[1][0].object = return_box
            self.return_card[1][1].object = annualized_return_box

            self.risk_card[1][0].object = vol_box
            self.risk_card[1][1].object = draw_box

            interpretation_text = f"""
            **Portfolio Growth**

            With an initial investment of **${initial_investment:,.0f}** on **{start_date}**, the portfolio would have grown to **${final_value:,.0f}** by **{end_date}**, representing a **total cumulative return of {(final_returns - 1) * 100:.2f}%**. This performance reflects the realized market dynamics and the optimized asset allocation.

            **Return**

            During this backtested period, the portfolio achieved an **annualized return of {annual_return * 100:.2f}%**, with a **Sharpe ratio of {sharpe:.2f}**, suggesting {'strong' if sharpe > 1 else 'balanced' if sharpe > 0.5 else 'modest'} risk-adjusted performance.

            **Risk**

            The portfolio’s annualized volatility was **{volatility * 100:.2f}%**, and the **maximum drawdown** reached **{abs(drawdown) * 100:.2f}%** during the period. This level of drawdown corresponds to a {('resilient profile' if abs(drawdown) < 0.15 else 'moderate risk profile' if abs(drawdown) < 0.3 else 'high risk exposure')}.
            """

            # Update the text in the Highlights section
            self.interpretation_text.object = interpretation_text

            # Generate portolio valuation plot
            valuation_fig = self.portfolio_valuation_plotly(
                self.model.portfolio_stats["portfolio_value"],
                start_date_str=self.model.portfolio_stats["start_date"],
                end_date_str=self.model.portfolio_stats["end_date"],
                initial_investment=initial_investment,
            )

            self.valuation_plot.object = valuation_fig

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"View :: update_portfolio_highlights failed — {e}")

    def portfolio_valuation_plotly(
        self,
        portfolio_value_series: pd.Series,
        start_date_str: str,
        end_date_str: str,
        initial_investment: int = 100_000,
    ) -> go.Figure:

        portfolio_valuation = portfolio_value_series.astype(float) * initial_investment

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=portfolio_valuation.index,
                y=portfolio_valuation.values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#2E86AB", width=2),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[portfolio_valuation.index[0], portfolio_valuation.index[-1]],
                y=[portfolio_valuation.iloc[0], portfolio_valuation.iloc[-1]],
                mode="markers+text",
                text=["Start", "End"],
                textposition="top center",
                marker=dict(color="#1B9E77", size=10, symbol="circle"),
                showlegend=False,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Portfolio Valuation Over Time<br><sup>{start_date_str} → {end_date_str}</sup>",
                x=0.5,
                font=dict(size=18),
            ),
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=20, t=60, b=40),
            hovermode="x unified",
        )

        return fig
