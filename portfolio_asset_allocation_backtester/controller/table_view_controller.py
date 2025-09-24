from ..model.model import Model
from ..view.securities_table_widget import SecuritiesTableWidget


class TableViewController:
    def __init__(self, model: Model, securities_view: SecuritiesTableWidget):
        self.model = model
        self.securities_view = securities_view


# TODO: port the control and bindings to here from the table view widget
# TODO: also connect the securities_df to the model instead of internal configs.
