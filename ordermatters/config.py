from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    words = root / 'words'
    corpora = root / 'corpora'
