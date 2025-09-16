from .model import Model
from .view import View

class Controller:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view