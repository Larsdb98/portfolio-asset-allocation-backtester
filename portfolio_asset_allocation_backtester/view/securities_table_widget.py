import panel as pn
import pandas as pd


class SecuritiesTableWidget:
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

        # Add security form
        self.add_button = pn.widgets.Button(name="Add Security", button_type="primary")
        self.ticker_input = pn.widgets.TextInput(name="Ticker", placeholder="e.g. AAPL")
        self.confirm_button = pn.widgets.Button(name="Confirm", button_type="success")
        self.cancel_button = pn.widgets.Button(name="Cancel", button_type="danger")

        self.form = pn.Column(
            pn.pane.Markdown("### Add New Security"),
            self.ticker_input,
            pn.Row(self.confirm_button, self.cancel_button),
            visible=False,
        )

        self.delete_button = pn.widgets.Button(
            name="Delete Selected", button_type="danger"
        )

        # Layout
        self.layout = pn.Column(
            "### Portfolio Securities",
            self.table,
            pn.Row(self.add_button, self.delete_button),
            self.form,
        )

        # Wire events
        self.add_button.on_click(self._show_form)
        self.confirm_button.on_click(self._add_instrument)
        self.cancel_button.on_click(self._hide_form)
        self.delete_button.on_click(self._delete_selected)

    # ----------------
    # Event handlers
    # ----------------
    def _show_form(self, event):
        self.form.visible = True

    def _hide_form(self, event=None):
        self.form.visible = False
        self.ticker_input.value = ""

    def _add_instrument(self, event):
        ticker = self.ticker_input.value.strip().upper()
        if ticker and ticker not in self.data["Ticker"].values:
            new_row = pd.DataFrame([[ticker]], columns=["Ticker"])
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            self._refresh_table()
        self._hide_form()

    def _delete_selected(self, event):
        selected = self.table.selection
        if selected:
            # selection is a list of row indices
            self.data = self.data.drop(index=selected).reset_index(drop=True)
            self._refresh_table()

    def _refresh_table(self):
        """Syncs Tabulator widget with internal DataFrame."""
        self.table.value = self.data

    # ----------------
    # Public
    # ----------------
    def get_instruments(self):
        """Return current instruments as a DataFrame."""
        return self.data.copy()

    def panel(self):
        """Return layout to embed in View."""
        return self.layout
