from ..model.model import Model
from ..view.instrument_table_widget import InstrumentTableWidget

from param.parameterized import Event


class TableViewController:
    def __init__(self, model: Model, instrument_view: InstrumentTableWidget):
        self.model = model
        self.instrument_view = instrument_view

        self._bind()

    def _bind(self):
        self.instrument_view.add_button.on_click(self.instrument_view.show_form)
        self.instrument_view.confirm_button.on_click(self.add_instrument)
        self.instrument_view.cancel_button.on_click(self.instrument_view.hide_form)
        self.instrument_view.delete_button.on_click(self.delete_instrument)

    def add_instrument(self, event: Event) -> None:
        ticker = self.instrument_view.ticker_input.value.strip().upper()
        self.model.add_instrument(ticker=ticker)
        self.instrument_view.refresh_table(instrument_df=self.model.instrument_df)

    def delete_instrument(self, event: Event) -> None:
        selected = self.instrument_view.table.selection
        self.model.delete_selected_instrument(selected=selected)
        self.instrument_view.refresh_table(instrument_df=self.model.instrument_df)
