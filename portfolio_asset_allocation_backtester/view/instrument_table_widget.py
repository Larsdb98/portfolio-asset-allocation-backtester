import panel as pn
import pandas as pd


class InstrumentTableWidget:
    def __init__(self, table_width: int = 400, table_height: int = 250):
        self.data = pd.DataFrame(columns=["Ticker"])

        self.table = pn.widgets.Tabulator(
            self.data,
            name="Portfolio",
            width=table_width,
            height=table_height,
            show_index=False,
            disabled=True,  # read-only
            selectable=True,  # allow row selection
        )

        # Add instrument form
        self.add_button = pn.widgets.Button(
            name="Add Instrument", button_type="primary"
        )
        self.ticker_input = pn.widgets.TextInput(name="Ticker", placeholder="e.g. AAPL")
        self.confirm_button = pn.widgets.Button(name="Confirm", button_type="success")
        self.cancel_button = pn.widgets.Button(name="Cancel", button_type="danger")

        self.form = pn.Column(
            pn.pane.Markdown("### Add New Instrument"),
            self.ticker_input,
            pn.Row(self.confirm_button, self.cancel_button),
            visible=False,
        )

        self.delete_button = pn.widgets.Button(
            name="Delete Selected", button_type="danger"
        )

        # Layout
        self.layout = pn.Column(
            "### Portfolio Instruments",
            self.table,
            pn.Row(self.add_button, self.delete_button),
            self.form,
        )

    # ----------------
    # View Handlers
    # ----------------
    def show_form(self, event):
        self.form.visible = True

    def hide_form(self, event=None):
        self.form.visible = False
        self.ticker_input.value = ""

    def refresh_table(self, instrument_df: pd.DataFrame):
        """Syncs Tabulator widget with internal DataFrame."""
        self.table.value = instrument_df

    def panel(self):
        """Return layout to embed in View."""
        return self.layout
