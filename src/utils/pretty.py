from dataclasses import dataclass, field
from typing import Self
from rich.panel import Panel
from rich.layout import Layout


from rich.console import RenderableType
from rich.rule import Rule
from rich.text import Text
from rich import print as rprint


def log_title(text: str | Text, rule_style="bright_black"):
    rprint(Rule(title=text, style=rule_style))


@dataclass
class APanel:
    title: str = "-"
    renderables: list[RenderableType] = field(default_factory=list)

    def with_content(
        self,
        renderable: RenderableType = "",
        print_now: bool = False,
    ) -> Self:
        if isinstance(renderable, str):
            renderable = Text(renderable)
        if renderable:
            self.renderables.append(renderable)
        if print_now:
            rprint(renderable)
        return self

    def print(self):
        root = Layout()
        root.split_column(*[Layout(renderable) for renderable in self.renderables])
        rprint(
            Panel.fit(
                root,
                title=self.title,
            )
        )


if __name__ == "__main__":
    # test func
    log_title("Hello World!")

    # test APanel
    import rich.table

    panel = APanel(title="My Panel")
    panel.with_content("This is a simple text.")
    rich_text = Text("This is a rich text.", style="bold red")
    panel.with_content(rich_text, print_now=True)
    table = rich.table.Table(title="Sample Table")
    table.add_column("Header 1")
    table.add_column("Header 2")
    table.add_row("Cell 1", "Cell 2")
    panel.with_content(table)
    panel.print()
