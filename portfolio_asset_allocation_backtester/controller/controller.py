from ..model.model import Model
from ..view.view import View

# from .table_view_controller import TableViewController

import threading
from param.parameterized import Event
import panel as pn
import pandas as pd
import numpy as np


class Controller:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view

        self._bind()

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

        self.view.weight_type_toggle.param.watch(self.on_weight_toggle, "value")

    def add_default_instrument(self, ticker: str) -> None:
        self.model.add_instrument(ticker=ticker)
        self.view.instrument_table_widget.refresh_table(
            instrument_df=self.model.instrument_df
        )

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
        # df = self.view.allocation_table.value
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
        self.view.performance_plot.object = self.view.loading_figure()

        def worker():
            """Runs heavy tasks in a separate thread."""
            try:
                print("DEBUG: Starting background backtest computation")
                self.model.run_backtest(
                    start_date=start_date, end_date=end_date, interval=interval
                )

                figure_dict = self.model.markowitz_plot_data

                plotly_fig = self.view.stochastic_optimised_frontier_plotly(
                    all_expected_volatility=figure_dict["all_expected_volatility"],
                    all_expected_log_returns=figure_dict["all_expected_log_returns"],
                    all_sharpe=figure_dict["all_sharpe"],
                    max_sharpe_index=figure_dict["max_sharpe_index"],
                    optimisation_method=figure_dict["optimization_method"],
                )

                print("DEBUG: Backtest done — updating UI")
                self.update_ui_after_backtest()
                # Schedule UI update back on main thread
                if pn.state.curdoc:
                    pn.state.curdoc.add_next_tick_callback(
                        lambda: setattr(
                            self.view.performance_plot, "object", plotly_fig
                        )
                    )
                else:
                    # Fallback: directly update the UI
                    self.view.performance_plot.object = plotly_fig

            except Exception as e:
                if pn.state.curdoc:
                    pn.state.curdoc.add_next_tick_callback(
                        lambda: setattr(
                            self.view.performance_plot,
                            "object",
                            self.view.loading_figure(text=f"Exception: {e}"),
                        )
                    )
                else:
                    self.view.performance_plot.object = self.view.loading_figure(
                        text=f"Exception: {e}"
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

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Controller :: Error while updating UI: {e}")

        # Update the view's table
        self.view.allocation_table.value = display_df

    def update_highlights_section(self, stats):
        """Update the metric boxes in the Highlights section."""
        ret = stats["annualized_return"] * 100
        vol = stats["annualized_volatility"] * 100
        dd = stats["max_drawdown"] * 100

        self.view.return_card[2][
            0
        ].object = f"<div class='metric-box green'>{ret:.1f}%</div>"
        self.view.risk_card[2][
            0
        ].object = f"<div class='metric-box green'>{vol:.1f}%</div>"
        self.view.risk_card[2][
            1
        ].object = f"<div class='metric-box gray'>{dd:.1f}%</div>"
