from .model.model import Model
from .view.view import View
from .controller.controller import Controller

import os


def main():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    model = Model()
    view = View(
        model=model,
        css_style_path=css_path,
        dashboard_title="Portfolio Asset Allocation Backtester",
    )

    controller = Controller(model=model, view=view)

    # Run the app
    view.run()


if __name__ == "__main__":
    main()
