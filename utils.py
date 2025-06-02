from pathlib import Path

import numpy as np

from rich.console import Console
from rich.text import Text
from typing import List

color_map = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "white",
    8: "bright_red",
    9: "bright_green",
}

console = Console()

def make_rich_lines(grid: List[List[int]]) -> List[Text]:
    lines = []
    for row in grid:
        visual = Text()
        for cell in row:
            color = color_map.get(cell, "white")
            visual.append("  ", style=f"on {color}")
        raw = Text("  " + str(row) + "\n")
        visual.append(raw)
        lines.append(visual)
    return lines

def render_grid(grid: List[List[int]]):
    lines = make_rich_lines(grid)
    text = Text()
    for line in lines:
        text.append(line)
    console.print(text)

def get_base_model(model_name):
    available_models = [
        "meta-llama/Llama-3.2-1B",
    ]
    assert model_name in available_models, f"{model_name} is not available."

