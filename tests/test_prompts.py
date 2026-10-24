import json

import pytest

from prompts import load_prompt


def test_the_prompt_is_the_json_of_the_version_and_model_type(tmp_path):
    folder = tmp_path / "prompts" / "v2"
    folder.mkdir(parents=True)
    (folder / "jev.json").write_text(json.dumps({"criteria": {"ok": None}}))

    prompt, path = load_prompt(tmp_path, "v2", "jev")

    assert prompt == {"criteria": {"ok": None}}
    assert path == folder / "jev.json"


def test_a_missing_prompt_names_the_file_it_looked_for(tmp_path):
    with pytest.raises(SystemExit) as error:
        load_prompt(tmp_path, "v9", "gemini")

    assert str(tmp_path / "prompts" / "v9" / "gemini.json") in str(error.value)
