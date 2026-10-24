"""Save one JSON of metrics per model and prompt version, next to the dataset."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

VERSION_LENGTH = 8


def content_hash(folder: Path, prompt_file: Path | None) -> str:
    """Short hash of the prompt file and the cases. It changes when either one changes."""
    digest = hashlib.sha256()
    for path in (prompt_file, folder / "dataset.json"):
        if path is not None:
            digest.update(path.read_bytes())
    return digest.hexdigest()[:VERSION_LENGTH]


def save_result(folder: Path, result, version: str, prompt_file: Path | None = None, limited: bool = False) -> Path:
    """Write result/<version>/<model>.json, where <version> is the prompt version.

    A rerun replaces it. A limited run (--max-evals) writes <model>-n<cases>.json
    so it never replaces a full result.
    """
    target = folder / "result" / version
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": folder.resolve().name,
        "version": version,
        "prompt": prompt_file.relative_to(folder).as_posix() if prompt_file else None,
        "hash": content_hash(folder, prompt_file),
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **asdict(result),
    }
    suffix = f"-n{result.cases}" if limited else ""
    path = target / f"{result.model}{suffix}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
