from ..model.model import Model
from ..view.view import View
from .table_view_controller import TableViewController

from param.parameterized import Event


class Controller:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view

        self.table_view_controller = TableViewController(
            model=self.model, instrument_view=view.instrument_table_widget
        )

    def _bind(self):
        pass
