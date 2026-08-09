# AivoraML

AivoraML is a local-first data science workspace for profiling tabular data,
comparing machine-learning models, making predictions, and exporting project
artifacts. A completed run can generate an executed Jupyter notebook,
browser-ready HTML report, presentation-ready PDF report, metrics CSV, run
summary, feature contract, and reloadable fitted pipeline from the same
training evidence.

The current local release includes isolated multi-project restore/delete,
reconnectable background training with real stage and candidate progress,
cooperative cancellation, immutable checksum-verified champion versions,
single-row and bounded batch inference, probability/error diagnostics, and
privacy-safe prediction history. Remote AI remains explicitly opt-in.

The authoritative product and technical requirements are defined in
[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). The complete generated
notebook and report quality standard is incorporated there by reference from
[`docs/ML_NOTEBOOK_OUTPUT_REQUIREMENTS.md`](docs/ML_NOTEBOOK_OUTPUT_REQUIREMENTS.md).

## Setup

From `D:\project`:

```powershell
py -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\backend\requirements.txt
```

Copy `.env.example` to `.env` only when you need to change the defaults.
Remote AI is optional. To enable the project assistant, keep secrets only in
`.env` and configure:

```dotenv
AIVORAML_ENABLE_REMOTE_AI=true
AIVORAML_AI_PROVIDER=auto
AIVORAML_AUTO_PROFILE_AI=false
GEMINI_API_KEY=replace-with-a-private-key
GEMINI_MODEL=gemini-3.6-flash
MISTRAL_API_KEY=replace-with-a-private-key
MISTRAL_MODEL=mistral-large-latest
```

In `auto` mode Gemini answers first and Mistral is the automatic fallback.
Only bounded project metadata, schema summaries, conversation turns, and model
metrics are sent. Core profiling, training, prediction, and export remain local
and do not require either provider.

## Run

Use the launcher:

```powershell
.\START_AIVORAML.bat
```

Or start the API directly:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The launcher intentionally does not use `--reload`: a development reload
restarts the process and interrupts any in-memory training job. Use reload mode
only while editing code, never while a model run is active.

Open:

- Landing page: <http://127.0.0.1:8000>
- ML workspace: <http://127.0.0.1:8000/app>
- API documentation: <http://127.0.0.1:8000/docs>

## Test

```powershell
python -m unittest discover -s tests -v
```

The release suite covers the full demo lifecycle, training-job state and
cancellation, project isolation and checksums, immutable model versions,
classification and regression diagnostics, batch inference, browser trust
headers, generated notebook cleanliness, and verified report downloads.

## Local privacy and trust boundary

- The launcher binds to `127.0.0.1`; non-local Host and browser Origin values
  are rejected unless an explicit unsafe override is configured.
- Uploaded data, snapshots, models, reports, and histories remain on this
  machine. The browser UI loads no third-party fonts or runtime assets.
- Raw prediction inputs are not retained by default. Prediction history can be
  cleared from the Predict page, and a project can be deleted with its
  project-scoped snapshots, model versions, and report artifacts.
- Joblib artifacts are loaded only from application-owned paths after SHA-256
  verification. They should still be treated as trusted local artifacts, not
  as a safe interchange format for untrusted files.

## Generate the notebook and reports

Train a project in the workspace, open **Artifacts**, and choose **Generate
report bundle**. Generation is complete only after every notebook code cell
executes without an error and the exported artifacts pass server-side checks.

The workspace uses the reconnectable job API:

- `POST /api/reports/jobs?project_id=<active-project-id>` — queue a verified
  bundle and receive a job ID
- `GET /api/jobs/<job-id>` — reconnect to real stage/progress/result state
- `DELETE /api/jobs/<job-id>` — request cooperative cancellation

The corresponding download API is:

- `POST /api/reports/generate?project_id=<active-project-id>` — generate or
  reuse the current bundle synchronously for API compatibility
- `GET /api/reports/current?project_id=<active-project-id>` — read the real
  generation state
- `GET /api/notebook/download?project_id=<active-project-id>` — executed
  `.ipynb`
- `GET /api/reports/html?project_id=<active-project-id>` and
  `GET /api/reports/pdf?project_id=<active-project-id>` — rendered reports
- `GET /api/reports/summary`, `/metrics`, `/feature-schema`, and `/pipeline`
  with the same `project_id` query — machine-readable and deployment artifacts

Training follows the same job pattern through `POST /api/training/jobs`. Each
completed run registers an immutable champion version under
`GET /api/models/versions`. Bounded batch inference is available through
`POST /api/predict/batch`; the browser Predict page validates and downloads its
result as CSV.

## Project layout

```text
assets/       Design and QA images
backend/      FastAPI application, schemas, and ML services
data/         Source datasets (local content is ignored)
docs/         Architecture and project planning
frontend/     Browser UI served by FastAPI
models/       Generated model artifacts
notebooks/    Working and exported notebooks
outputs/      Reports, charts, and other generated results
reports/      Verified HTML, PDF, metrics, and run-summary exports
scripts/      Local development and launch scripts
src/          Reserved for reusable package code
storage/      Private runtime workspace state
tests/        Automated tests
venv/         Local Python environment
```

The app intentionally binds to `127.0.0.1` and has no authentication. Notebook
execution also runs locally in a child kernel. Do not expose this local mode
directly to the public internet or execute untrusted datasets/code without the
hosted isolation controls specified in the project plan.
