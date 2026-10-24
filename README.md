# os-agents-evals

CLI that runs one dataset against several models and compares accuracy, cost and latency. Each model gets its own JSON prompt, so JEV and a normal chat model are not forced through the same text.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env          # fill in the keys
set -a; . ./.env; set +a      # the CLI reads the environment, not the file
.venv/bin/os-evals run freezer --models jev,gemini --max-evals 100 --workers 8
```

A name like `freezer` is read from `datasets/freezer`. `--config` defaults to `models.json`.

```text
jev: 94% (IC 91-95%) exactitud, $0.17/1k consultas, $0.19/1k aciertos, p95 0.85s, F1 0.92, confianza 0.96 si acierta, 0.66 si falla
```

- **IC** is the 95% Wilson interval of the accuracy. With 40 cases one hit is worth 2.5 points, so the interval says when two runs are not really different.
- **$X/1k consultas** is the mean cost of a call. **$X/1k aciertos** divides the same total by the hits only, and is omitted when nothing is right.
- **F1** is the unweighted mean of the per-class F1.
- **confianza** is JEV only: its mean confidence when the answer is right and when it is wrong.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--models a,b` | required | Names from `models.json`. |
| `--prompt v1` | `v1` | Reads `prompts/<version>/<model type>.json`. |
| `--max-evals N` | all | About the same number of cases from each class. A class smaller than its share gives every case it has and the rest is shared again. |
| `--seed 7` | `7` | Repeats that pick. |
| `--no-stratified` | off | The older even spread, which can skip a rare class. |
| `--workers N` | `1` | Calls in flight at once. 500 `freezer` cases took 29 s on JEV and 156 s on Gemini with 8 workers. A 429, 500, 502, 503, 504 or a timeout is tried 3 times, one second apart. If it still fails, that case is an error and the run continues. |

## Dataset

`datasets/<name>/` holds `dataset.json`, `prompts/`, `result/` and, for photos, `images/`. Cases and images stay out of git. The template, each `prompts/` and each `result/` are tracked.

A case is `{"type": "text", "text", "expected"}` or `{"type": "image", "path", "expected"}`. `path` is relative to the dataset folder. Several images are a list and travel as bytes only: the file name is never sent, because it can carry the answer (`Abyssinian_201.jpg`).

`meta` is optional and does not change the score. `{"tipo": "nevera", "marca": "hit"}` is counted under `per_group`.

Scoring is an exact match after stripping spaces and lowercasing. `A` matches `a`. `A. because...` does not.

## Prompts

`--prompt v2` reads `prompts/v2/` and leaves `v1` in place, so two versions can be compared. `fake` needs no prompt.

`jev.json`:

```json
{"state": "Foto de un equipo de frío.", "instructions": "Qué mueble es?", "criteria": {"nevera/hit": "Nevera verde con el logo Hit."}}
```

JEV is not a chat model. It scores `state` (then the case text, then the image) against one Choice question. `criteria` is required. Its keys are the only answers JEV can give, and each value is that label's definition (`null` when the name is enough). The reply is the chosen key, plus a confidence from 0 to 1.

`gemini.json`:

```json
{"system": "Eres un clasificador.", "instructions": "Responde solo con la etiqueta.", "temperature": 0, "labels": ["nunca", "hit"]}
```

Every key is optional. `system` goes in `systemInstruction`. The user message is `instructions`, a blank line, then the case text and the images. The reply is text and must equal the label.

`labels` lists the answers that count as valid, for both files. JEV uses its `criteria` keys when `labels` is absent. With no list at all the invalid-answer fields stay `null`. `mmmu` leaves it off because some answers are not from a fixed list.

## Models

`models.json` maps a name to `{type, model, api_key_env, price}`. Prices are USD per million tokens. The key is read from the environment variable named by `api_key_env` and is never stored in the file. `.env.example` lists the names.

- **fake** returns canned answers. An image answer is keyed by the absolute path.
- **jev** calls TypeSafe System One, `POST https://api.typesafe.ai/v1/systemone`. `model` defaults to `jev-latest`. Price: $0.042 per million input tokens, output free ([TypeSafe models](https://docs.typesafe.ai/models)). Photos are fitted to JEV's 280–1568 px limit, which needs Pillow (`pip install -e ".[images]"`). Image support is the vendor's opt-in beta. Default key variable: `TYPESAFE_API_KEY`.
- **gemini** calls the Gemini API on Google AI Studio, `generateContent` at `generativelanguage.googleapis.com`, with the key in `x-goog-api-key`. No project or region. `model` is required. Thinking tokens count as output. Gemini 2.5 Flash text price: $0.30 input and $2.50 output per million ([Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)). Default key variable: `GEMINI_API_KEY`.

## Results

A run writes `datasets/<name>/result/<version>/<model>.json`. `result/v2/jev.json` is the output of `prompts/v2/jev.json`. A finished rerun of the same version and model replaces the file. An incomplete one writes `<model>.incomplete.json` and leaves the finished file in place. `--max-evals N` writes `<model>-n<N>.json` beside it and never replaces the full run. `v0/` holds the runs from before prompts were versioned.

`hash` is the first 8 hex characters of the SHA-256 of the prompt file followed by `dataset.json`. If a result for that same prompt is already in the version folder with a different hash, the run stops before any model call. Bump `--prompt` to a new version. JEV and Gemini use different prompt files, so their hashes are allowed to differ. The image bytes are not part of the hash.

The file also keeps:

- `engine`, the `model` string from `models.json` (`null` for `fake`), `price` as copied from that file, and `commit`, the git HEAD of the directory you ran from (`null` when git is not available)
- `accuracy`, `accuracy_ci95`, `f1_macro`, `cases`, `wall_s`, `p95_latency_s`
- `complete`, `errors`, `error_samples`. A case that still fails after the retries, a missing photo, or a Gemini reply with no text counts as an error, not as a wrong answer. Accuracy, cost and the interval are over the cases that answered. `cases` stays the number tried. An incomplete run is written to `<model>.incomplete.json` (or `<model>-n<N>.incomplete.json`) and does not replace a finished result. The printed row ends with `incompleta`.
- `cost_per_1k`, `cost_per_1k_correct` (`null` when nothing is right)
- `per_label`: `correct`, `total`, `precision`, `recall`, `f1` for each expected label
- `confusion`: for each expected label, the answers given. An answer that matches none of them is `(otra)`
- `invalid_count`, `invalid_rate`, `invalid_samples` (up to 3, clipped to 80 characters). An invalid answer still counts as a miss. `null` when the prompt has no label list
- `confidence_correct`, `confidence_wrong`, and `by_confidence` at 0.5, 0.7, 0.9, 0.95 and 0.99 (`coverage` and `accuracy`). `null` when the model reports no confidence
- `per_group`: for each `meta` key, `correct` and `total` per value. `{}` when no case has `meta`
- `sample`: `{strategy, seed}`. `seed` is `null` for `--no-stratified`

## Datasets

- **freezer**. 6675 photos, 10 labels `tipo/marca`: `nevera/{cocacola, postobon, colombiana, gatorade, hit, redbull, sinmarca}` and `congelador/{cremhelado, colombina, sinmarca}`. Each case has `meta.tipo` and `meta.marca`. Source: `gs://harmony-ml-training/freezer/datasets/cold-equipment`, read with the default gcloud config. Kept only rows whose source category and brand match the folder. Dropped the `review` folder, one file name that had two labels, and 41 manifest rows missing from the bucket. `sinmarca` was labeled by JEV while relabeling `generico`, so a JEV hit there partly grades JEV on its own labels. Photos are flat in `images/`.
- **oxford-iiit-pet**. 37 breeds, official test split.
- **mmmu**. Validation split. Answers are not a fixed list, so the prompt has no `labels`.
- **realslab**. Listing audit: `ok`, `fraude`, `error_tipeo`, `anomalia_datos`.

## Tests

```bash
.venv/bin/pytest
```
