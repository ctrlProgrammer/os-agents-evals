"""Prompts live next to the dataset: prompts/<version>/<model type>.json."""

import json
from pathlib import Path


def load_prompt(dataset: Path, version: str, kind: str) -> tuple[dict, Path]:
    path = dataset / "prompts" / version / f"{kind}.json"
    if not path.is_file():
        raise SystemExit(f"prompt not found: {path}")
    return json.loads(path.read_text()), path
