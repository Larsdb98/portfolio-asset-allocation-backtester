import os
from pathlib import Path
from .model import Model


class View:
    def __init__(
        self,
        model: Model,
        css_style_path: Path = None,
        dashboard_title: str = "Portfolio Asset Allocation Backtester",
    ):
        self.model = model
        self.css_style_path = css_style_path
        self.dashboard_title = dashboard_title

    def run(self):
        pass
