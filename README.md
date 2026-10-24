# os-agents-evals

CLI that runs the same set of prompts against several models and compares them on accuracy, cost and latency.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env            # fill in the keys
set -a; . ./.env; set +a        # export them to this shell
.venv/bin/os-evals run realslab --models <modelo> --config models.json
.venv/bin/os-evals run oxford-iiit-pet --models <modelo> --max-evals 20   # 20 cases spread over the dataset
.venv/bin/os-evals run freezer --models jev,gemini --prompt v2            # another prompt version
```

```text
<model>: 75% exactitud, $0.48/1k consultas, p95 1.20s
```

- **Dataset**: a folder under `datasets/<name>/` with `dataset.json` and `prompts/`. Each case is `{"type": "text", "text", "expected"}` or `{"type": "image", "path", "expected"}`. `path` is relative to that folder; several images are a list. Scoring is exact match, ignoring case and surrounding spaces.
- **Models** (`--config`, default `models.json`): name → spec with token prices per million tokens. `type` is `fake`, `jev`, or `gemini`.
  - `fake` returns canned answers. Image answers are keyed by the absolute image path.
  - `jev` calls TypeSafe System One (`POST /v1/systemone`) as a Choice. The choice key is the label, so it must equal `expected`. `model` defaults to `jev-latest`. Published price: $0.042 per million input tokens, output free ([TypeSafe models](https://docs.typesafe.ai/models)).
  - `gemini` calls the Gemini API from Google AI Studio (`generateContent` on `generativelanguage.googleapis.com`) and sends the key in `x-goog-api-key`, so no project or region is needed. `model` is required. Thinking tokens count as output. Gemini 2.5 Flash standard text price: $0.30 per million input tokens and $2.50 per million output tokens ([Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)).
  - `api_key_env` names the environment variable for that service. The key is read from the environment and is never stored in the config. Defaults are `TYPESAFE_API_KEY` and `GEMINI_API_KEY`; `.env.example` lists them. The CLI reads the process environment, so export `.env` first (see Usage).
  - Image cases send the file bytes, never the path or file name (it can carry the answer, like `Abyssinian_201.jpg`). JEV needs Pillow to fit its 280-1568 px limit: `pip install -e ".[images]"`. JEV image support is in the vendor's opt-in beta. Photo URLs already inside the case text are sent as text.
- **Prompts**: `--prompt <version>` (default `v1`) reads `datasets/<name>/prompts/<version>/<model type>.json`, one file per model `type`. A new version is a new folder, so old ones stay for comparison. Each model has its own format:
  - `gemini.json`: `{"system": "...", "instructions": "...", "temperature": 0}`. All keys are optional. The user message is `instructions`, a blank line, then the case text (and the images). `system` goes in `systemInstruction`.
  - `jev.json`: `{"state": "...", "instructions": "...", "criteria": {"label": "description or null"}}`. `criteria` is required and its keys are the only answers JEV can give. JEV is not an instruction-following chat model: it scores the `state` against a typed Choice question. The `state` sent is `state`, a blank line, the case text, then the image placeholders. `instructions` is the question and each `criteria` description is the definition of that label, so the label definitions go there and not in free text.
  - Both end in a label that is compared with `expected`: JEV returns the choice key, Gemini returns text that must equal the label.
  - `fake` needs no prompt.
- **`--max-evals N`** runs only N cases, evenly spaced over the dataset (a head cut would be one class when the cases are sorted). A limited run saves `result/<model>-<version>-n<N>.json` and never replaces a full result.
- **Confidence**: JEV returns a 0–1 `confidence` per choice. The row shows its mean when the answer is right and when it is wrong (`confianza 0.90 si acierta, 0.40 si falla`). Gemini reports none.
- **Cost** is the mean cost per query × 1000. **Latency** is p95 (nearest rank).
- **Results**: every run on a dataset folder writes `datasets/<name>/result/<version>/<model>.json`, where `<version>` is the prompt version of `--prompt`, so `result/v2/` is the output of `prompts/v2/`. The file has `dataset`, `version`, `prompt` (the prompt file used, e.g. `prompts/v2/jev.json`; `null` for `fake`), `hash`, `run_at`, `model`, `cases`, `accuracy`, `cost_per_1k`, `p95_latency_s`, `per_label` (`correct` / `total` per expected label), and `confidence_correct` / `confidence_wrong` (mean model confidence when it hits or misses; `null` if the model reports none). `hash` is the first 8 hex chars of the SHA-256 of the prompt file followed by `dataset.json`: if it differs between two runs of the same version, the prompt or the cases were edited in place, so change the version instead. A rerun of the same version and model replaces the file; a `--max-evals N` run writes `<model>-n<N>.json` next to it and never replaces the full one. Compare prompts by diffing `result/v1/<model>.json` and `result/v2/<model>.json`. `v0` holds the runs made before prompts were versioned (a single `prompt.txt`). `prompts/` and `result/` are tracked in git; the cases and images are not.

## Datasets

Cases and images are not in git (only `_template/` and each `prompts/` and `result/`).

- `freezer`: tipo y marca de equipos de frío, 6037 fotos. Etiquetas `tipo/marca`: `nevera/{cocacola,postobon,colombiana,gatorade,hit,redbull}` y `congelador/{cremhelado,colombina}`. Fuente: `gs://harmony-ml-training/freezer/datasets/cold-equipment` (Harmony, config de gcloud por defecto, solo lectura). Se usa solo la categoría y marca de origen del manifiesto (`source_category` y `source_brand` iguales a las de la carpeta). Quedan fuera `sinmarca` y `review` (los etiquetó JEV al reetiquetar `generico`, sería circular), un nombre de archivo con dos etiquetas y 41 filas del manifiesto que ya no están en el bucket. Las fotos van planas en `images/` y el orden de `dataset.json` es aleatorio con semilla fija, para que `--max-evals` muestree varias clases. 
- `oxford-iiit-pet`: 37 breeds, official test split. `mmmu`: validation split. `realslab`: listing audit, 4 labels.

## Tests

```bash
.venv/bin/pytest
```
