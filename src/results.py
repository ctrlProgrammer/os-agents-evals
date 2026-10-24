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


def require_same_hash(folder: Path, version: str, prompt_file: Path | None) -> None:
    """Stop when this version already holds a result for the same prompt with another hash.

    JEV and Gemini keep different prompt files, so their hashes are not compared.
    """
    directory = folder / "result" / version
    if not directory.is_dir():
        return
    new_hash = content_hash(folder, prompt_file)
    prompt = prompt_file.relative_to(folder).as_posix() if prompt_file else None
    for path in sorted(directory.glob("*.json")):
        saved = json.loads(path.read_text())
        if saved.get("prompt") != prompt:
            continue
        old = saved.get("hash")
        if old and old != new_hash:
            raise SystemExit(
                f"prompt version {version} already has results with hash {old}; "
                f"this run is {new_hash}. Bump --prompt to a new version."
            )


def save_result(
    folder: Path,
    result,
    version: str,
    prompt_file: Path | None = None,
    sample: dict | None = None,
    limited: bool = False,
) -> Path:
    """Write result/<version>/<model>.json, where <version> is the prompt version.

    A finished rerun replaces it. A limited run (--max-evals) writes <model>-n<cases>.json
    so it never replaces a full result. An incomplete run writes <model>.incomplete.json
    and leaves a finished result in place.
    """
    target = folder / "result" / version
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": folder.resolve().name,
        "version": version,
        "prompt": prompt_file.relative_to(folder).as_posix() if prompt_file else None,
        "hash": content_hash(folder, prompt_file),
        "sample": sample,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **asdict(result),
    }
    suffix = f"-n{result.cases}" if limited else ""
    # An incomplete run keeps the paid calls, but it must not replace a finished result.
    marker = "" if result.complete else ".incomplete"
    path = target / f"{result.model}{suffix}{marker}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
