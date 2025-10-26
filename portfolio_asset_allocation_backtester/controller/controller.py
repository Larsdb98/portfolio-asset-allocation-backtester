from ..model.model import Model
from ..view.view import View

# from .table_view_controller import TableViewController

import threading
from param.parameterized import Event
import panel as pn
import pandas as pd
import numpy as np
from typing import Dict


class Controller:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view

        self._bind()

        # update initial amount right away
        self.model.update_initial_portfolio_value(
            portfolio_value=self.view.initial_amount.value
        )

    def _bind(self):
        # Instrument table widget:
        self.view.instrument_table_widget.add_button.on_click(
            self.view.instrument_table_widget.show_form
        )
        self.view.instrument_table_widget.confirm_button.on_click(self.add_instrument)
        self.view.instrument_table_widget.cancel_button.on_click(
            self.view.instrument_table_widget.hide_form
        )
        self.view.instrument_table_widget.delete_button.on_click(self.delete_instrument)

        # Main components
        self.view.run_button.on_click(self.run_backtest)
        self.view.initial_amount.param.watch(self.update_initial_amount, "value")
        self.view.weight_type_toggle.param.watch(self.on_weight_toggle, "value")

        self.view.highlights_view.benchmark_selector.param.watch(
            self.on_benchmark_toggle, "value"
        )

    def on_benchmark_toggle(self, event: Event):
        """Update valuation plot when benchmark selection changes."""
        try:
            portfolio_series = self.model.portfolio_stats["portfolio_value"]
            benchmark_df = self.model.portfolio_stats["benchmark_value"]
            start = self.model.portfolio_stats["start_date"]
            end = self.model.portfolio_stats["end_date"]

            fig = self.view.highlights_view.portfolio_valuation_plotly(
                portfolio_series,
                benchmark_df,
                selected_benchmarks=event.new,
                start_date_str=start,
                end_date_str=end,
            )

            self.view.highlights_view.valuation_plot.object = fig

        except Exception as e:
            print(f"Controller :: Benchmark toggle update failed — {e}")

    def add_default_instrument(self, ticker: str) -> None:
        self.model.add_instrument(ticker=ticker)
        self.view.instrument_table_widget.refresh_table(
            instrument_df=self.model.instrument_df
        )

    def update_initial_amount(self, event: Event) -> None:
        self.model.update_initial_portfolio_value(self.view.initial_amount.value)

    def add_instrument(self, event: Event) -> None:
        ticker = self.view.instrument_table_widget.ticker_input.value.strip().upper()
        self.model.add_instrument(ticker=ticker)
        self.view.instrument_table_widget.refresh_table(
            instrument_df=self.model.instrument_df
        )

    def delete_instrument(self, event: Event) -> None:
        selected = self.view.instrument_table_widget.table.selection
        self.model.delete_selected_instrument(selected=selected)
        self.view.instrument_table_widget.refresh_table(
            instrument_df=self.model.instrument_df
        )

    def on_weight_toggle(self, event: Event):
        """Callback when user toggles between stochastic and optimal weights."""
        df = self.model.allocation_df_raw
        if df.empty:
            return

        tickers = df["Symbol"].tolist()

        if self.view.weight_type_toggle.value == "Stochastic":
            weights = df["Stochastic Weight Allocation (%)"].to_numpy() / 100
            title = "Stochastic Portfolio Weights"
        else:
            weights = df["Optimal Weight Allocation (%)"].to_numpy() / 100
            title = "Optimal Portfolio Weights"

        self.view.update_weight_pie_chart(tickers, weights, title)

    def run_backtest(self, event: Event):
        """Callback triggered when the Run Backtest button is clicked."""
        # Capture inputs before starting thread (thread-safe)
        start_date = self.view.start_date.value
        end_date = self.view.end_date.value
        interval = self.view.granularity_input.value

        # Optional immediate feedback
        self.view.detailed_analytics_view.performance_plot.object = (
            self.view.detailed_analytics_view.loading_figure()
        )

        def worker():
            """Runs heavy tasks in a separate thread."""
            try:
                self.model.run_backtest(
                    start_date=start_date, end_date=end_date, interval=interval
                )
                self.model.compute_portfolio_statistics(
                    start_date=start_date, end_date=end_date
                )

                figure_dict = self.model.markowitz_plot_data

                plotly_fig = self.view.detailed_analytics_view.stochastic_optimised_frontier_plotly(
                    all_expected_volatility=figure_dict["all_expected_volatility"],
                    all_expected_log_returns=figure_dict["all_expected_log_returns"],
                    all_sharpe=figure_dict["all_sharpe"],
                    max_sharpe_index=figure_dict["max_sharpe_index"],
                    optimisation_method=figure_dict["optimization_method"],
                )

                self.update_ui_after_backtest()
                # Schedule UI update back on main thread
                if pn.state.curdoc:
                    pn.state.curdoc.add_next_tick_callback(
                        lambda: setattr(
                            self.view.detailed_analytics_view.performance_plot,
                            "object",
                            plotly_fig,
                        )
                    )
                else:
                    # Fallback: directly update the UI
                    self.view.detailed_analytics_view.performance_plot.object = (
                        plotly_fig
                    )

            except Exception as e:
                if pn.state.curdoc:
                    pn.state.curdoc.add_next_tick_callback(
                        lambda: setattr(
                            self.view.detailed_analytics_view.performance_plot,
                            "object",
                            self.view.detailed_analytics_view.loading_figure(
                                text=f"Exception: {e}"
                            ),
                        )
                    )
                else:
                    self.view.detailed_analytics_view.performance_plot.object = (
                        self.view.detailed_analytics_view.loading_figure(
                            text=f"Exception: {e}"
                        )
                    )

        # Launch thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def update_ui_after_backtest(self) -> None:
        """Update the allocations table and pie chart after a backtest."""
        try:
            tickers = self.model.instrument_df["Ticker"].to_list()
            names = self.model.get_ticker_long_names
            stochastic_weights = self.model.get_stochastic_optimal_weights
            optimal_weights = self.model.get_optimal_weights

            raw_df = pd.DataFrame(
                {
                    "Symbol": tickers,
                    "Name": names,
                    "Stochastic Weight Allocation (%)": np.array(stochastic_weights)
                    * 100,
                    "Optimal Weight Allocation (%)": np.array(optimal_weights) * 100,
                }
            )
            self.model.allocation_df_raw = raw_df

            display_df = raw_df.copy()

            # Format percentage columns
            display_df["Stochastic Weight Allocation (%)"] = raw_df[
                "Stochastic Weight Allocation (%)"
            ].map(lambda x: f"{x:.2f}%")
            display_df["Optimal Weight Allocation (%)"] = raw_df[
                "Optimal Weight Allocation (%)"
            ].map(lambda x: f"{x:.2f}%")

            self.model.allocation_df_display = display_df

            # Update the view's table
            self.view.allocation_table.value = display_df

            # Update pie chart
            selected_mode = (
                self.view.weight_type_toggle.value
            )  # "Stochastic" or "Optimal"
            if selected_mode == "Stochastic":
                weights = stochastic_weights
                title = "Stochastic Portfolio Weights"
            else:
                weights = optimal_weights
                title = "Optimal Portfolio Weights"

            self.view.update_weight_pie_chart(tickers, weights, title)

            # Update highlights section
            self.view.highlights_view.update_portfolio_highlights(
                initial_investment=self.model.get_initial_portfolio_value
            )

            # Update statistics table under other tab
            metrics_df = self.model.get_portfolio_risk_metrics_df
            self.view.detailed_analytics_view.update_metrics_table(
                metrics_df=metrics_df
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Controller :: Error while updating UI: {e}")

        # Update the view's table
        self.view.allocation_table.value = display_df
