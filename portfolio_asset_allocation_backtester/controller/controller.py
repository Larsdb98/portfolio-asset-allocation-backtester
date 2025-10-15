from ..model.model import Model
from ..view.view import View

# from .table_view_controller import TableViewController

import threading
from param.parameterized import Event
import panel as pn


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
