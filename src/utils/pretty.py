from rich import print as rprint
from rich.console import RenderableType
from rich.panel import Panel

def log_title(renderable: RenderableType, title: str = ""):
    rprint(
        Panel(
            renderable,
            title=title,
        )
    )