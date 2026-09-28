# Saw It: CUHK-X Large Model Track

Infrared video question answering with Gemini, selective IMU emotion corrections, and exact-text consistency across questions.

**[Read the paper](paper/Saw_It_CUHKX_Solution_Paper.pdf)** · **[Editable LaTeX](paper/main.tex)** · **[Framework PDF](paper/Saw_It_Framework.pdf)**

Author: Hanif Noer Rofiq · Team: Saw It · CUHK-X Challenge 2026, Large Model Track.

## Results

| Evaluation | Result |
|---|---:|
| Local development holdout | 768/972 correct (79.01%) |
| Kaggle public leaderboard | 0.78654 |
| Kaggle private leaderboard | 0.81176 |

The leaderboard scores were confirmed by the participant. The local split holds out subjects user4, user9, user19 and user24, but was inspected during development. These evaluations use different partitions.

## Method

![Inference framework](paper/framework.png)

1. **IR video question answering.** Each IR video and its questions are sent to Gemini, configured as `gemini-3.8-flash`, with requested sampling at 4 fps and structured JSON output. Gemini is not fine-tuned. The parser checks question coverage, option letters and sequence ordering.
2. **Exact-text consistency.** If a clip has exactly one combination question, actions named in its predicted option are matched to multiple-answer options. Exact matches are added to those answer sets.
3. **Selective emotion replacement.** A 400-tree Random Forest uses 1,010 features from five IMU positions resampled to 10 Hz. It predicts emotion descriptions, maps them to the current question's options, and blends the probabilities equally with a training-derived option prior. It replaces an emotion answer only with at least two usable devices and a top-two score margin of at least 0.10.

On the same 972 development questions, IR alone answered 743 correctly, consistency alone raised this to 748, IMU replacement alone to 763, and the complete pipeline to 768. The two additions affect different categories. The IMU addition fixed 42 errors and introduced 22 regressions; its margin is not a calibrated guarantee of correctness.

## What is public

This repository contains the **paper and source code**, including fixed configurations, training and inference entry points, and offline tests. Trained checkpoints, learned priors, prediction CSVs, competition datasets and API response caches are not published here. The paper's verification-package discussion refers to the separate organizer delivery.

## Setup and training

Use **Python 3.13.9**, a CPU host and Bash. No local GPU or CUDA is required. Obtain the official competition data separately.

```bash
git clone https://github.com/hanifnoerr/cuhk-x-saw-it.git
cd cuhk-x-saw-it
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash train.sh TRAINING_QA_CSV OFFICIAL_SENSOR_ZIP artifacts
```

Supply the official `training_qa.csv` and `LMT_(IMU,Radar,Skeleton).zip`. Training uses only usable labelled training emotion records and the frozen settings in `configs/sensor_config.json`. It writes the RF, imputer, option prior, feature columns and settings under `artifacts/`, which Git ignores. The script does not use test labels or predictions, repeat model selection or fine-tune Gemini.

The historical final fit used 787 emotion records. Its local checkpoint was 3.82 MB compressed, approximately 27.5 MB as an uncompressed serialization. These sizes are not peak RAM. Gemini's hosted weights are not downloaded and their size is undisclosed.

## Inference

Inference requires the locally trained artifacts and a separately supplied, funded Gemini API key. Enter the key without saving its value in shell command history:

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY
printf '\n'
export GEMINI_API_KEY
bash inference.sh DATA_DIR OUTPUT_CSV
```

`DATA_DIR` may be the official Testing directory, its track parent, or a root containing `Large-Model-Track/Testing`. It must contain label-free `test_qa.csv` and the IR videos, either extracted in the published layout or in `large_model_track_test.zip`.

Place `LMT_(IMU,Radar,Skeleton).zip` alongside the track data, or set `CUHKX_SENSOR_PATH` to that archive or its extracted `LMT_(IMU,Radar,Skeleton)` directory. Missing or unusable IMU falls back to the IR answer. Missing IR raises an error. Omitting the entire sensor source changes the pipeline's behavior.

The output columns are `qa_id,prediction`. A `sample_submission.csv`, when present beside the QA file, controls output order; its prediction values are never used. New question IDs are supported. Missing local model artifacts stop execution before any API request.

| Environment variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required for uncached API inference; never commit its value. |
| `CUHKX_SENSOR_PATH` | Official nonvisual archive or extracted root. |
| `CUHKX_RUN_DIR` | Writable response-cache directory; defaults beside the output CSV. |
| `PYTHON` | Python executable used by the shell wrappers. |

### Internet and API behavior

The code uses HTTPS at `generativelanguage.googleapis.com`: the Files upload API, file-processing status queries and `/v1beta/models/gemini-3.8-flash:generateContent`. Requests contain IR video and associated questions/options. The credential is sent in a header, not the URL. No model weights or datasets are downloaded during inference. Installing Python dependencies may require internet.

The historical test run used Batch scheduling. The portable reconstruction uses sequential requests, so cost and runtime can differ. Responses and repair attempts are cached on the operator's machine. An uncertain request with no saved response blocks automatic retries to avoid duplicate billing; reconcile it before retrying.

## Tests and reproduction limits

```bash
python3 -B -m unittest discover -s tests -v
```

The tests mock network calls and use temporary synthetic fixtures. They require no actual key, trained checkpoint or competition data. GitHub Actions runs them on Ubuntu with pinned Python dependencies.

Before this public export, raw-data training fitted 787 official emotion records. Its option scores agreed with the supplied reconstructed checkpoint across all 144 test emotion records to floating-point precision, and all 81 eligible replacements matched the historical selected CSV. Those checks do not establish exact fresh reproduction of the full leaderboard score.

The original RF checkpoint was deleted and later reconstructed. Original Gemini responses and an immutable service revision were not recovered. Fresh API predictions may differ. The reported leaderboard scores describe the historical submission; **fresh end-to-end API reproduction remains unverified**. No test labels were used to compensate for these gaps.

Public-export changes are limited to moving fixed training settings into `configs/`, adding a missing-model check before API calls, and making offline tests independent of trained artifacts. The selected preprocessing, probability blend, prompt and consistency rule are unchanged.

## Paper and files

- `paper/`: PDF, standalone `main.tex`, and editable TikZ framework.
- `code/`, `cuhkx/`: inference, training and component functions.
- `configs/`: fixed training settings, API configuration and prompt.
- `tests/`: network-mocked tests.
- `requirements.txt`, `.python-version`: pinned environment.
- `MANIFEST.sha256`: hashes of all other tracked files.

Build the paper with `latexmk -pdf main.tex` from `paper/`, or run `pdflatex main.tex` twice. The source contains the figure and bibliography.

Credentials, account sessions, private computer paths and conversation history are excluded. The author name and contact email appear in the paper.
