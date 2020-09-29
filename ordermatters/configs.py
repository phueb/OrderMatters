from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    words = root / 'words'
    corpora = root / 'corpora'


class Constants:
    num_ticks = 32


class Figs:
    ax_font_size = 12
    leg_font_size = 10
    title_font_size = 8
    tick_font_size = 8
