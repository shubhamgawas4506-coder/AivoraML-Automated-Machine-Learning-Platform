# AivoraML Implementation Audit

**Audit date:** 2026-07-28  
**Current release reassessment:** 2026-08-08  
**Specifications reviewed, in precedence order:** `docs/PROJECT_PLAN.md`; attached **ML NOTEBOOK AND OUTPUT QUALITY REQUIREMENT**; `README.md`; existing implementation  
**Audit scope:** All project-owned frontend, backend, ML, configuration, storage, test, script, documentation, and deployment-related files. The ignored `venv/` dependency tree was treated as an environment, not application source.

## 0. Current Local Release Reassessment

This section supersedes the historical current-state statements in the original
2026-07-28 audit. The detailed tables below remain useful as baseline evidence
and as a hosted-product roadmap, but many defects they record are now fixed.

**Current supported scope:** single-user, local-first, loopback-only tabular ML
workspace. **Assessment within that scope: 98/100.** This is not a claim that
the separate authenticated, multi-user hosted architecture is complete.

| Area | Score | Current evidence |
|---|---:|---|
| Core data and modeling workflow | 24/24 | Real upload/profile, target configuration, leakage-safe candidate comparison, prediction, and bounded batch inference |
| ML evaluation truthfulness | 20/20 | Training-only CV selection/tuning/thresholding; untouched holdout; calibration, classification error, regression error, and bounded permutation diagnostics |
| Projects, jobs, and model lineage | 15/15 | Checksum-verified isolated projects, guarded activation/deletion, reconnectable training/report jobs, cancellation, immutable champion versions, and race regressions |
| Reproducible artifacts | 14/14 | Executed/validated notebook, HTML, PDF, summary, metrics, schema, and reloadable pipeline from one run |
| Product UX and accessibility | 13/14 | Unified responsive design, truthful capability states, keyboard/focus/modal support, readable sizing, and 320–720px containment; no automated axe/browser matrix yet |
| Local privacy and safety | 8/9 | Loopback Host/Origin boundary, CSP/security headers, remote AI opt-in, no raw prediction retention by default, safe limits and checksum gates; local mode intentionally has no authentication |
| Verification and operations | 4/4 | Dependency check, compilation, JavaScript/CSS checks, disposable browser QA, and 51-test release suite |
| **Total** | **98/100** | **Release-ready for the documented local-first trust boundary** |

### Material changes since the baseline audit

- Silent demo-success fallbacks and fabricated production analytics were
  removed; demo state is explicit and the UI uses backend evidence.
- Training and report generation now expose real queued/running/progress/
  cancelled/completed states through one bounded job manager.
- Project snapshots, prediction histories, report artifacts, and model versions
  are project-scoped. Cross-project persistence, malformed sidecar, and
  competing report-writer regressions are covered by deterministic tests.
- Model versions are immutable and SHA-256 verified before activation, loading,
  or download. Single and batch predictions use the fitted input contract.
- The professional notebook generator was replaced with an executable,
  format-aware, injection-safe workflow; server-side generation executes and
  validates every code cell and all exported artifacts.
- The landing page and product workspace now share one visual system, avoid
  third-party runtime assets, expose honest capability/pricing/privacy copy,
  and include responsive and keyboard accessibility work.
- Retired synchronous training routes now return `410`; supported training goes
  through the project-bound job/version contract.

### Remaining two points

1. Add an automated multi-browser accessibility/visual regression matrix.
2. Add durable external job infrastructure plus authentication and tenant
   isolation before any public or multi-user deployment. In-process local jobs
   can reconnect after a page refresh, but not after the server process exits.

## 1. Executive Summary

AivoraML is a functioning single-user, local-first tabular machine-learning prototype. Its verified path is:

1. Upload CSV, Excel, JSON, or Parquet data—or generate a deterministic churn demo.
2. Validate and profile the dataset.
3. Compare classification or regression candidates using preprocessing inside cross-validation.
4. select a champion on training-only CV and evaluate it on a separate 20% holdout.
5. Make schema-validated single-row predictions through the fitted pipeline.
6. Persist the active workspace locally and export dataset, predictions, metrics, project JSON, model, preprocessor, and notebook artifacts.

This broadly matches the concise README, and the existing smoke suite passes this lifecycle. It does **not** match the target state in `docs/PROJECT_PLAN.md`, which is now the authoritative specification. The README's no-authentication, loopback-only statements describe the current local mode; they do not cancel the master plan's target multi-user architecture. Target-state authentication, database, jobs, reports, deployment, and monitoring are therefore planned requirements, while local mode must remain safe and usable during migration.

The highest-priority defects are:

- `frontend/app_premium.js` contains unused hardcoded demo profiles and fabricated model metrics. They are dead today, but are prohibited production content and can easily be reactivated accidentally.
- The in-browser “AI” guide is a scripted keyword responder, not the optional Mistral service, and claims unsupported clustering/forecasting capabilities.
- Cleaning and feature-selection controls change only browser state; the selected switches are not sent to training. Their success messages imply a stronger connection than exists.
- Loading percentages/stages are timer-generated rather than backend progress.
- `backend/main.py` is a 1,508-line API, storage, orchestration, export, and ML coordination module using one global mutable store.
- Several endpoints accept untyped `dict` payloads and most responses lack declared schemas or a consistent error envelope.
- Broad exception handling and `print`/traceback logging obscure failure classification.
- The stored `workspace.joblib` contains a 2,500-row demo dataset but no trained workflow or model. No repository artifact supports the separately supplied 7,043-row customer-churn metrics; those values must not be hardcoded.
- The notebook generator creates a syntactically valid but **unexecuted** 22-cell notebook. It embeds no outputs and produces no HTML/PDF report.
- The notebook rebuilds a similar—not identical—experiment (different estimator parameters, fold count, candidate defaults, and no fold-safe balancing), so notebook results can contradict the application run. This is a blocking technical-honesty failure.

No major refactor was performed during this audit.

### Documentation conflict discovered and reconciled

The temporary root `PROJECT_MASTER_PLAN.md` name was later confirmed to be a
mistake. Its unique notebook-quality addendum was reconciled into
`docs/PROJECT_PLAN.md`, README now points to the canonical document, and the
redundant root file was removed.

## Post-audit implementation delta — 2026-07-28

The following P0 notebook/report gaps identified below have since been
implemented in local mode:

- training records a unique run ID, JSON-safe run manifest, fold scores, a dummy baseline,
  candidate parameters, an untouched holdout contract, and a training-only
  out-of-fold binary threshold;
- saved inference applies the locked threshold and exports its provenance;
- the generated notebook is task-aware, retrains the same balancing/candidate
  path, selects the same champion, applies the threshold, includes EDA,
  outlier/skewness evidence, comparison, evaluation, errors, explainability,
  artifact reload, scorecard, and conclusions;
- every code cell is executed in a fresh kernel and rejected on missing
  execution counts or error outputs;
- the report service emits and verifies notebook, HTML, PDF, JSON, CSV, schema,
  and pipeline artifacts with SHA-256 checksums;
- the frontend artifact page uses real report state and no longer contains the
  fabricated demo profile/result constants or unsupported clustering and
  forecasting claims; AI labels, artificial response delay, and timer-driven
  progress stages were replaced with a truthful workflow guide and polled
  server section updates;
- the end-to-end test now asserts execution, champion agreement, threshold
  lineage, report formats, and reloadable artifact metadata.
- the stale demo-only workspace snapshot and empty prediction history were
  removed so a fresh launch does not restore state disconnected from the
  verified artifacts.

Remaining master-plan gaps include durable multi-user jobs, database/object
storage lineage, strong execution isolation, authentication/authorization,
immutable hosted artifact storage and checksum enforcement, richer local explanations and
calibration, and the full classification/regression edge-case acceptance
matrix. The detailed findings below remain the baseline audit and should be
read with this delta.

`docs/ML_NOTEBOOK_OUTPUT_REQUIREMENTS.md` also appeared and contains the notebook standard without the conversation-specific correction footer. It is useful as a repository copy of the supplemental standard, but precedence remains:

1. `docs/PROJECT_PLAN.md`
2. the attached/current notebook-output requirement
3. `README.md`
4. implementation

After the user confirmed the name was a mistake, the unique normative addendum was merged into `docs/PROJECT_PLAN.md`, README was corrected to link to it, and the redundant root file was removed. The repository now has one master-plan authority.

## 2. Current Folder Structure

```text
project/
├── assets/                    # Two QA screenshots
├── backend/
│   ├── main.py                # FastAPI routes, global state, persistence, exports
│   ├── schemas.py             # Partial Pydantic models
│   ├── requirements.txt
│   └── services/
│       ├── ai_advisor.py
│       ├── dataset_profiler.py
│       ├── eda_generator.py
│       ├── ml_engine_enhanced.py
│       └── notebook_generator_pro.py
├── data/                      # Empty tracked directory
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ML_NOTEBOOK_OUTPUT_REQUIREMENTS.md
│   └── PROJECT_PLAN.md
├── frontend/
│   ├── landing.html
│   ├── index_premium.html
│   ├── styles_premium.css
│   └── app_premium.js
├── models/ notebooks/ outputs/ src/  # Empty tracked directories
├── scripts/start_server.bat
├── storage/
│   ├── predictions.json
│   └── workspace.joblib
├── tests/test_api_smoke.py
├── venv/                      # Ignored local environment
├── .env                       # Ignored local secrets/config
├── .env.example
├── .gitignore
├── README.md
└── START_AIVORAML.bat
```

There is no valid Git repository metadata, `pyproject.toml`, frontend package manifest, database package, Docker configuration, CI workflow, or deployment configuration.

## 3. Technologies Currently Used

| Area | Present technology | Evidence and notes |
|---|---|---|
| Frontend | Static HTML, CSS, vanilla JavaScript | `frontend/*`; no build system or frontend test framework |
| API | FastAPI, Uvicorn, Starlette | `backend/main.py`, `backend/requirements.txt` |
| Validation | Pydantic | Some request/domain schemas in `backend/schemas.py`; many routes still accept `dict` |
| Data | pandas, NumPy, SciPy | Upload parsing, profiling, ML, EDA |
| ML | scikit-learn, optional XGBoost, imbalanced-learn | Pipelines, multiple candidates, fold-safe oversampling |
| Visualization | Matplotlib, Seaborn | Server-generated EDA images; not used by normal fast strategy |
| Persistence | Joblib + JSON on local filesystem | One active workspace in `storage/` |
| Notebook | nbformat | Generated notebook is parsed and validated before download |
| Optional AI | Mistral SDK | Opt-in remote advice through environment configuration |
| Testing | `unittest`, FastAPI `TestClient` | One five-test end-to-end smoke class |

`pip check`, Python compilation, and JavaScript syntax checks pass. Ruff, mypy, pytest, frontend linting, frontend tests, and a production frontend build are unavailable because their tools/configuration do not exist.

## 4. Existing Frontend Features

### Working and connected

- Public landing page and responsive application shell.
- Dataset upload and deterministic demo creation.
- Profile, preview, quality, correlations, missingness, cleaning-plan, feature-plan, training, evaluation, explanation, prediction, and export views.
- Real backend calls for upload, demo, training, project restoration, prediction, prediction history, and downloads.
- Timeout, abort, toast, modal, keyboard navigation, and basic ARIA support.
- Empty gates prevent protected project screens before a dataset is available.

### Incomplete or misleading

- `DEMO_PROFILE` and `DEMO_RESULT` are large, unused hardcoded objects containing fabricated metrics and preview data.
- The workflow assistant is a client-side keyword response function. It is not connected to `/api/ai/*`, but UI labels call it AI.
- It claims clustering and forecasting support, while the backend supports only classification and regression.
- Cleaning/feature switches are not included in the training request. The backend always applies its own fixed contract.
- Loading progress and stage transitions are simulated timers; only completion/failure is real.
- “Model registry” represents only the active in-memory/persisted run, not version history.
- “Deploy” is an export page; no deployment occurs.
- No batch prediction UI, report center, authentication, settings, project creation, or multiple-project workflow.
- No frontend unit, integration, accessibility, or E2E test harness.

## 5. Existing Backend Features

- FastAPI app serves both pages and static assets.
- Upload filename sanitization, supported-extension allow-list, byte/row/column/cell limits, empty-data and duplicate-column validation.
- Thread-pool offloading for profile and training work.
- One durable local workspace restored at process startup.
- Atomic replacement for workspace and prediction-history writes.
- Dataset profile/current project/workflow metadata routes.
- Schema-driven single-row inference with unknown-field and numeric validation.
- CSV formula-injection protection on exported tabular text.
- Model, fitted preprocessor, metrics, dataset, predictions, project JSON, and validated notebook exports.
- Optional remote AI advice with deterministic fallbacks.
- Debug endpoint disabled unless explicitly enabled.

### Architectural weaknesses

- Global `_Store` permits only one project and is vulnerable to concurrent request races.
- Routes, persistence, domain services, exports, and orchestration are coupled in `main.py`.
- No API versioning, service/repository boundary, database session, background job model, health endpoint, structured logging, or consistent error schema.
- Long training runs occupy an application process thread and cannot be cancelled.
- Persistence errors are logged and silently ignored, so API success may not mean durable success.

## 6. Existing API Inventory

| Area | Endpoints |
|---|---|
| Pages | `GET /`, `GET /app`, mounted `/static` |
| Dataset/project | `POST /api/upload`, `POST /api/demo`, `GET /api/dataset/profile`, `POST /api/dataset/imbalance`, `GET /api/projects`, `GET /api/project/current` |
| Workflows | `POST /api/workflow/run`, `POST /api/workflow/run-simple`, `POST /api/ml/professional-workflow`, `GET /api/workflow/result` |
| Prediction | `GET /api/predict/schema`, `POST /api/predict`, `GET /api/predictions` |
| Downloads | dataset, predictions, metrics, project JSON, model, preprocessor, notebook |
| AI | recommend models, feature engineering, explain results, status, approaches, pipeline proposal, execute pipeline |
| Debug | `GET /api/debug/dataset-info` behind an environment flag |

Problems: routes are unversioned; several payloads are raw dictionaries; response models are not declared; error bodies vary; there is no request ID, health/readiness endpoint, rate limiting, or formal authentication/authorization. For the README’s local-only scope, lack of auth is intentional, not a defect, as long as loopback binding is preserved.

## 7. Database Models

There is no database and therefore no database models, migrations, repository layer, users, project ownership, model registry, or audit log. The single local workspace snapshot is a dictionary serialized with Joblib; prediction history is duplicated inside the snapshot and in `predictions.json`.

Database and authentication are missing target-state requirements under the master plan. The local no-auth mode remains a supported development/private mode, while hosted mode requires PostgreSQL-backed identities, authorization, tenant isolation, migrations, and auditability.

## 8. Existing ML Functionality

### Implemented

- Automatic classification/regression detection.
- Duplicate removal and null-target removal.
- Constant, identifier-like, and high-cardinality feature exclusion.
- Median numeric imputation, standard scaling, mode categorical imputation, and unknown-safe one-hot encoding.
- Random 80/20 holdout with stratification where possible.
- Three-fold training-only CV for model selection.
- Fold-safe random oversampling for sufficiently imbalanced classification.
- Classification candidates: logistic regression, decision tree, random forest, extra trees, gradient boosting, AdaBoost, KNN, Naive Bayes, SVC, and optional XGBoost.
- Regression candidates: Ridge, Lasso, ElasticNet, decision tree, random forest, extra trees, gradient boosting, AdaBoost, KNN, SVR, and optional XGBoost.
- Optional bounded randomized tuning of selected tree/boosting families.
- Classification accuracy, weighted precision/recall/F1, binary ROC-AUC, PR-AUC, and log loss.
- Regression MAE, median AE, MSE, RMSE, R², and conditional MAPE.
- Confusion matrix/classification report or residual summary.
- Native/coefficient feature importance.
- Fitted pipeline serialization, schema-driven inference, and notebook generation.

### ML gaps and risks

- No threshold optimization, selected-threshold inference, Brier score, probability calibration, SHAP/LIME, permutation importance, partial dependence, feature-selection stage, batch inference, monitoring, or retraining.
- Candidate selection uses weighted F1, which may hide minority-class weakness; primary metric is not user-configurable.
- Hyperparameter search reports `best_score_` from the same CV search as an improved selection score. The untouched holdout remains valid, but search optimism should be communicated or nested CV used.
- Re-running against the same deterministic holdout allows manual holdout overfitting; there is no access/version policy.
- Dense one-hot encoding can still be expensive for large row counts despite the 1,000-level cap.
- Group-aware and time-aware splits are absent.
- Broad exception handlers can omit metrics without explicit warnings.
- Task detection is a cardinality heuristic and can misclassify numeric categorical or small-sample numeric targets.

## 9. Reports and Exports

Implemented: model Joblib, fitted preprocessor Joblib, dataset CSV, prediction CSV, metric CSV, project JSON, and an executable notebook. The smoke test validates artifact load, schema presence, notebook validity, and generated code syntax.

Missing mandatory target-state outputs: executed notebooks with embedded outputs, HTML/PDF reports, model-summary JSON, metrics CSV as a managed report artifact, model cards, versioned artifact registry, retention UI, reproducibility lockfile, batch-prediction artifact, and deployment bundle.

Model export includes runtime library versions, class labels, feature schema, split metadata, dataset fingerprint, metrics, and strategy. It does not contain a genuine selected threshold, stable model version, training timestamp, complete training-data metadata, or source revision.

## 10. Tests

`tests/test_api_smoke.py` performs one class-level demo → train flow and verifies:

- real histogram counts;
- at least four trained models;
- training-only CV selection and untouched holdout declaration;
- fitted prediction schema and prediction;
- loadable model and preprocessor-related metadata;
- valid generated notebook;
- project/dataset/metric exports;
- unknown prediction feature rejection;
- unsupported upload extension rejection.

Gaps: no focused unit tests, upload parser/limit matrix, corrupt file tests, concurrency/persistence tests, regression test, imbalance test, tuning test, reproducibility assertion, pipeline reload/prediction equivalence, AI tests, browser tests, accessibility tests, linting, typing, security tests, or performance tests. The class shares state and order by method names, which makes failures harder to isolate.

## 11. Deployment Configuration

No Dockerfile, Compose file, CI workflow, Nginx config, cloud manifest, packaging metadata, or deployment runbook exists. These are missing against the master plan but are not prerequisites for preserving the current local mode. `START_AIVORAML.bat` correctly delegates to a script that selects `venv\Scripts\python.exe` when present and binds Uvicorn to `127.0.0.1`.

## 12. Code and Repository Findings

### Broken or incomplete functionality

- UI cleaning and feature choices are not propagated to the backend.
- Deployment/model-registry wording overstates export/current-run functionality.
- AI guide is not connected to the optional AI backend.
- No real progress API; UI progress is simulated.
- The stored workspace currently has no trained pipeline, so prediction/download routes correctly return unavailable until training.

### Duplicate files

No byte-identical project-owned files were found. `predictions` are stored both in `workspace.joblib` and `predictions.json`; this is intentional recovery behavior but needs one consistency contract.

### Dead code and placeholders

- `DEMO_PROFILE` and `DEMO_RESULT` are never referenced and contain fabricated results.
- Fallback references to nonexistent `index_ultimate.html` and `index_enhanced.html` are stale.
- `src/`, `models/`, `notebooks/`, and `outputs/` contain only marker files.
- Generated `__pycache__` directories are present locally but ignored.

### Hardcoded paths and secrets

- No machine-specific absolute path exists in source. The README intentionally uses `D:\project` setup examples.
- `.env` contains Mistral-related key names and is ignored. Values were not exposed during the audit.
- No credential literal was found in application source.
- Local filenames and route paths are hardcoded appropriately; storage root is configurable.

### Broken imports and dependencies

- Python compilation and imports pass.
- `pip check` reports no broken installed requirements.
- JavaScript syntax passes `node --check`.
- No missing runtime dependency was observed in the tested path.
- Development dependencies and configuration for lint/type/frontend testing are missing.

### Error handling

- Numerous broad `except Exception` blocks and bare `except` clauses exist.
- Many failures use `print` and `traceback.print_exc()` rather than structured logging.
- Upload parser exception text is returned to users and may disclose internals.
- Persistence failures do not change the API response.
- Some optional metric failures are silently ignored.

## 13. Risk Register

| Risk | Severity | Evidence | Required control |
|---|---|---|---|
| Misleading fake/dead frontend results | High | Hardcoded `DEMO_*` metrics | Remove; tests forbid production hardcoded model results |
| UI/backend configuration mismatch | High | Cleaning/features only mutate JS state | Send validated config or label controls read-only |
| Concurrent global-state corruption | High | Module-level singleton store | Serialize mutations or introduce project service/storage abstraction |
| Local Joblib trust | High if directory is writable by untrusted users | Joblib loaded at startup | Document trust boundary, checksum/permissions; never load user models |
| Resource exhaustion | High | Whole-file read and parse; up to 1M rows; dense encoding | Pre-parse safeguards, lower/default budgets, worker limits |
| Manual holdout overfitting | Medium | Fixed seed/holdout reusable across runs | Dataset/run versioning and comparison policy |
| Silent partial metrics | Medium | Broad exception suppression | Typed failures and warnings |
| Scripted assistant misrepresentation | Medium | Client keyword responses labeled AI | Relabel guide or call real endpoint with provenance |
| Prediction data sensitivity | Medium | Last 5,000 inputs stored locally | Disclosure, configurable retention, clear-history function |
| No source control metadata | Medium | `git status` fails | Initialize/restore Git before substantial implementation |
| No structured observability | Medium | `print`/tracebacks | Standard logging and request context |

### Security risks

The loopback binding is the primary control; there is no identity or authorization layer. Local Joblib restoration trusts the contents of `storage/`, prediction inputs are retained, upload parsers handle untrusted files in the API process, and parser exception details can reach the client. These are acceptable only inside the documented private-machine trust boundary and still require hardening.

### Performance risks

Uploads are read and parsed in memory, EDA produces base64 images, one-hot output is dense, and CPU-heavy training shares the API process. Row/cell/cardinality limits reduce but do not eliminate peak-memory, long-request, and concurrent-workload risks.

### Data-leakage risks

Current preprocessing and oversampling are correctly placed inside model pipelines and the final holdout is not used programmatically for selection. Remaining risks are repeated human inspection of the same deterministic holdout, lack of group/time split support, heuristic identifier detection, and future feature/threshold work being added outside folds.

### Missing validation

Raw-dictionary endpoints lack field-level schemas; upload MIME/magic bytes and decompression behavior are not checked before parsing; strategy/stage combinations have limited validation; categorical prediction values are not constrained to observed choices (unknown-safe encoding is intentional); and persistence snapshots lack an integrity checksum.

## 14. Customer-Churn Artifact Verification

The supplied desired metrics refer to 5,634 training rows plus 1,409 test rows (7,043 total), 18 features, a `GradientBoostingClassifier`, threshold 0.34, and specific evaluation values.

Repository evidence does **not** support those results:

- `storage/workspace.joblib` is a trusted application snapshot containing `customer_churn.csv` with 2,500 rows.
- Its `workflow_result` is absent; it contains no pipeline, champion, threshold, or metrics.
- No model file, evaluation report, dataset, or metadata artifact with the supplied values exists.
- The current deterministic demo has 14 columns before identifier removal, not 18 features.

Required action: do not hardcode the supplied metrics. Obtain the original dataset and trusted model/evaluation artifact, or regenerate the model through a verified pipeline. Then add a versioned metadata artifact and regression test before claiming preservation.

## 15. Specification Hierarchy and Master-Plan Comparison

### 15.1 Precedence applied

1. `docs/PROJECT_PLAN.md` defines the target product, architecture, quality gates, security, and roadmap.
2. The attached notebook/output requirement is a mandatory ML artifact standard.
3. `README.md` remains the current local-mode overview and setup guide.
4. Existing behavior is retained only when it does not conflict with the first three sources.

The exact heading **“ML NOTEBOOK AND OUTPUT QUALITY REQUIREMENT”** is not currently present in `docs/PROJECT_PLAN.md`. The attached 990-line specification therefore acts as the mandatory supplemental standard requested by the user. A later documentation task should merge a controlled version into the master plan; it was not changed in this audit-only phase.

### 15.2 Target-state capability summary

| Master-plan requirement | Status | Evidence | Required action |
|---|---|---|---|
| Next.js/React/TypeScript frontend | Missing | Static HTML/CSS/JS only | Migrate incrementally after API contracts stabilize |
| Versioned FastAPI service/application/domain layers | Needs Refactoring | Unversioned routes in monolithic `main.py` | Extract routers/services/domain modules and `/api/v1` |
| PostgreSQL, SQLAlchemy, Alembic | Missing | Joblib/JSON only | Add transactional metadata and migrations |
| Redis and asynchronous jobs | Missing | Thread-pool work within API process | Add durable jobs, progress, retry, cancellation |
| Object-storage abstraction and artifact lineage | Missing | Local `storage/`, on-demand streams | Add immutable artifact service/checksums/metadata |
| Authentication, RBAC, tenant isolation | Missing | Explicit no-auth local prototype | Preserve local mode; require auth for hosted mode |
| Versioned datasets/runs/models/reports/deployments | Missing | One mutable active workspace | Implement immutable lineage and registries |
| Leakage-safe ML pipeline | Partially Implemented | Pipeline/CV/holdout exist | Add split policies, dummy baseline, threshold/calibration gates |
| Explainability | Partially Implemented | Native/coefficient importance only | Add permutation, SHAP/local evidence, caveats and provenance |
| Complete notebook/report suite | Broken | Unexecuted notebook only | Implement mandatory notebook workstream below |
| Batch prediction | Missing | Single-row prediction only | Add bounded asynchronous batch inference |
| Deployment/rollback/monitoring | Missing | Export guidance only | Implement after registry, security, and jobs |
| Structured observability and CI/CD | Missing | Prints; no CI/deployment files | Add logging/tracing/metrics and gated pipelines |

## 16. ML Notebook and Output Quality Audit

### 16.1 Current generator facts

- Generator: `backend/services/notebook_generator_pro.py`.
- API: `GET /api/notebook/download`.
- Output: one in-memory `.ipynb` response; it is not saved under `notebooks/generated/`.
- Structure: 22 cells, comprising a title and 10 numbered sections.
- Validation: `nbformat.validate()` and `ast.parse()` for code cells.
- Execution: none. All execution counts and outputs are empty.
- Reports: no HTML, PDF, model-summary JSON, feature-schema file, or managed report artifact.
- Dependencies: `nbformat` is present; `nbclient`, `nbconvert`, a PDF renderer, SHAP, and LIME are absent.
- UI: a notebook-styled dataset-profile view and download link exist; there is no generated-notebook job/progress page.

### 16.2 Blocking reproducibility defect

The generated notebook does not faithfully reproduce the completed application run. It reconstructs estimators with its own defaults (for example, 150 trees instead of the engine's 100), normally uses five CV folds instead of the engine's three, omits fold-safe oversampling, uses a separate candidate subset, and only opportunistically applies reported tuned parameters. It then chooses and evaluates a new champion.

Consequences:

- The notebook's champion and metrics can disagree with the UI and exported model.
- The title's “reported champion/score” can contradict later executed cells.
- The notebook is not a provenance-preserving explanation of the project.
- A clean execution requires the user to manually place the original dataset beside it.

The correct design is to generate from one immutable run manifest and dataset snapshot. Either the notebook must replay the **exact** recorded configuration or it must load and explain the exact fitted artifact; any optional re-experiment must be explicitly separate.

### 16.3 Mandatory 34-part requirement matrix

| # | Requirement area | Status | Current evidence/problem | Required action |
|---:|---|---|---|---|
| 1 | Project overview | Partially Implemented | Generic title, dataset/target/task/champion; no adapted problem, use case, or success criteria | Generate evidence-backed project title, objective, task rationale, dimensions, feature groups, and success criteria |
| 2 | Grouped imports | Partially Implemented | Imports are in one code cell and include broad candidate libraries | Add explanatory markdown and dataset/task-specific imports only |
| 3 | Load dataset | Partially Implemented | Loads four formats and shows head; requires manually adjacent file | Bundle/version dataset snapshot; show head/tail/sample/columns with actual interpretations |
| 4 | Initial understanding | Missing | No `info`, numeric/object describe, sample, unique counts, or interpretation cells | Add reusable overview helpers and computed narrative |
| 5 | Data-quality assessment | Partially Implemented | Missingness, duplicates, dtype, cardinality only | Add constant/near-constant, invalid/infinite, whitespace/category/date, identifier, leakage, imbalance/target checks with action table |
| 6 | Dataset-specific EDA | Missing | No EDA section; only CV, confusion/residual, importance plots | Build question-led, task-aware charts with pre/post interpretations |
| 7 | Outlier analysis | Missing | No per-feature bounds/action table | Compute IQR/domain evidence; recommend without automatic blanket removal |
| 8 | Data cleaning | Partially Implemented | Drops duplicates/missing target with generic prose | Record before/after counts and rationale for each actual action; avoid invented cleaning |
| 9 | Skewness/distribution | Missing | No skew table or transformation policy | Compute distribution class and justified transform recommendation |
| 10 | Feature engineering | Missing | Schema resolution only | Add semantic, reproducible, leakage-checked features when justified; otherwise explicitly report none |
| 11 | Encoding strategy | Partially Implemented | All categoricals one-hot encoded | Produce per-feature plan; ordinal encoding only with evidence; address high cardinality |
| 12 | Train/validation/test split | Partially Implemented | 80/20 + CV and stratification fallback | Explain rationale; support time/group detection/confirmation; record actual fallback |
| 13 | `ColumnTransformer` preprocessing | Implemented with limitations | Median/scale/mode/OHE pipeline is fold-safe | Adapt branches and scaling to model/data needs; explain each transform |
| 14 | Reusable pipelines | Implemented | Preprocessor + estimator serialized | Add feature selection/engineering inside pipeline and exact run provenance |
| 15 | Baseline model | Broken | Logistic/Ridge may appear, but no mandatory `Dummy*` baseline | Always run task-appropriate dummy and explain uplift |
| 16 | Multiple-model comparison | Partially Implemented | CV mean/std/time table; no train/holdout metrics for every model or role analysis | Record required task table, stability, speed, interpretability, gap, and best row |
| 17 | Cross-validation | Partially Implemented | Mean/std shown; individual fold scores/min/max discarded | Persist fold scores and interpret stability |
| 18 | Hyperparameter tuning | Broken | Notebook does not execute tuning; it copies reported params | Replay exact recorded search or display immutable tuning results/search space and verified before/after comparison |
| 19 | Model evaluation | Partially Implemented | Basic classification table/confusion matrix or regression residual plot | Add full metric set, ROC/PR/calibration/probability or actual-vs-predicted/error plots and contextual explanations |
| 20 | Threshold optimisation | Missing | Uses estimator default; exported threshold absent | Select on OOF/validation probabilities, lock threshold, evaluate once on holdout |
| 21 | Error analysis | Missing | No false-positive/negative, worst-error, subgroup, or confidence analysis | Add task-specific error tables/plots and evidence-based narrative |
| 22 | Explainability | Partially Implemented | Native/coefficient or holdout permutation importance | Add global/local direction where supported, SHAP/PDP selectively, and causal caveats |
| 23 | Champion selection | Partially Implemented | CV mean minus stability penalty only | Multi-criterion selection summary including performance, gap, speed, size, interpretability, probability quality, objective |
| 24 | Model saving | Partially Implemented | One Joblib with pipeline/metrics/schema/versions | Save versioned pipeline, metadata, schema, metrics; include threshold/fingerprint/training time; reload equality test |
| 25 | Sample prediction | Partially Implemented | Displays up to three raw rows, predictions, max probability | Add one clear human interpretation, threshold, validation warnings, and uncertainty language |
| 26 | Final conclusion | Missing | Notebook ends after reload smoke test | Add computed technical/nontechnical conclusion, limitations, ethics, next steps, readiness |
| 27 | Quality scorecard | Missing | None | Derive status/notes from executed evidence and explicit unavailable checks |
| 28 | Automatic writing honesty | Broken | Mostly generic prose; “reported” values may differ from replay; no output-derived post-cell narrative | Generate narratives from computed result objects and validate claims against outputs |
| 29 | Output presentation | Partially Implemented | Numbered flow and readable plots; no embedded output or interpretation callouts | Add “Reading this”/“Why this matters,” warnings, consistent formatting, highlighted tables |
| 30 | Execute and export | Missing | Syntax validation only | Execute clean kernel, embed outputs, validate artifacts/reload/prediction, export HTML and supported PDF |
| 31 | Dataset-specific adaptation | Broken | Fixed template, fixed random split, fixed scaling/OHE | Introduce dataset assessment and decision policy with safe fallbacks/confirmation |
| 32 | Generated code quality | Partially Implemented | Deterministic and readable but one long sequence with broad exceptions | Add required helpers, structured warnings, optional-library guards, formatting/static validation |
| 33 | Technical corrections | Partially Implemented | Split prose/code align and pipeline imputation exists; notebook/app run can conflict | Add consistency validator for counts, split, terms, transforms, metrics, leakage, and narrative claims |
| 34 | Generated Notebook page | Missing | Profile resembles notebook; export card only | Add real generation job page with section progress, warnings, models, metrics, and every required download |

### 16.4 Required output inventory

| Output | Current | Mandatory target |
|---|---|---|
| Complete notebook | On-demand unexecuted response | `notebooks/generated/<project>_complete_ml_workflow.ipynb`, executed and embedded |
| HTML report | Missing | `reports/<project>_ml_report.html` generated from verified executed content |
| PDF report | Missing | `reports/<project>_ml_report.pdf` when renderer/environment supports it, with explicit capability state |
| Model summary | Missing as standalone report | `reports/<project>_model_summary.json` |
| Metrics | On-demand CSV response | Versioned `reports/<project>_metrics.csv` |
| Pipeline | On-demand Joblib | Versioned `models/<project>_pipeline.joblib` |
| Metadata | Embedded, incomplete | Versioned JSON containing threshold, fingerprint, timestamp, versions, seed, lineage |
| Feature schema | Embedded only | Standalone JSON and API/UI download |

### 16.5 Execution and truthfulness gates

A notebook/report job must fail—not return “complete”—unless:

1. The run manifest, dataset version, fitted pipeline, threshold, and results share the same lineage.
2. Every code cell executes from top to bottom in an isolated clean working directory within resource limits.
3. Outputs and charts are embedded and free of error outputs.
4. All referenced artifacts exist, have checksums, reload, and reproduce sample predictions.
5. Narrative claims are produced from structured computed values and pass invariant checks (counts, split, task terminology, metric direction, transform existence, tuning delta, class balance, top feature).
6. HTML export succeeds. PDF capability is reported honestly and succeeds when enabled.
7. The completed artifact bundle is immutable/versioned and the UI reports warnings rather than hiding unavailable analyses.

## 17. README Requirements Comparison

| README Requirement | Current Status | Existing File Paths | Problems | Required Action |
|---|---|---|---|---|
| Authoritative specification link | Implemented | `README.md`, `docs/PROJECT_PLAN.md`, `docs/ML_NOTEBOOK_OUTPUT_REQUIREMENTS.md` | Naming conflict was present during audit | Correct master link and normative reference are now established |
| Local-first data science workspace | Implemented | `backend/main.py`, `storage/`, `frontend/` | One global workspace; concurrency risk | Stabilize local project service without changing scope |
| Profile tabular data | Implemented | `dataset_profiler.py`, upload/profile routes | Broad exception suppression; limited tests | Add profiler and parser tests; typed warnings |
| Compare ML models | Implemented | `ml_engine_enhanced.py` | Fixed selection metric; UI overstates controls | Expose validated training configuration and provenance |
| Make predictions | Implemented | prediction routes and frontend form | Single-row only; inputs persist without retention control | Add clear retention/history controls and consistency tests |
| Export project artifacts | Implemented | download routes, notebook generator | No model version/training timestamp/selected threshold | Complete artifact metadata and isolated reload tests |
| Local venv setup | Implemented | README, `backend/requirements.txt` | No lockfile or packaging metadata | Add `pyproject.toml` and constraints in planned phase |
| Optional Mistral integration | Partially Implemented | `.env.example`, `ai_advisor.py`, AI routes | Frontend guide does not use it; provider errors use prints | Separate “workflow guide” from authenticated AI advice |
| Launcher and direct Uvicorn command | Implemented | launcher scripts, README | Windows-only convenience script | Keep; optionally add cross-platform scripts |
| Landing page, workspace, API docs | Implemented | frontend files, FastAPI | Static frontend has no build/test system | Add browser smoke/accessibility checks |
| End-to-end smoke test command | Implemented | `tests/test_api_smoke.py` | Narrow, stateful suite | Split into focused tests while retaining lifecycle test |
| Documented folder layout | Partially Implemented | repository root | Several directories empty; ML remains under backend services | Refactor only where it reduces coupling |
| Bind to `127.0.0.1` | Implemented | `scripts/start_server.bat`, README | Direct users could override command | Keep warning and add startup trust-boundary log |
| No authentication in local mode | Implemented | README, loopback launcher | Correct for private local mode; insufficient for master-plan hosted mode | Preserve local mode and add authentication only to hosted/multi-user mode |
| Do not expose publicly | Implemented | README and launcher | No runtime host guard | Consider refusing non-loopback bind unless explicit unsafe override |

## 18. Recommended Implementation Order

1. Freeze misleading/fake frontend paths and add notebook characterization/consistency tests.
2. Define immutable run manifest, dataset snapshot, artifact metadata, and exact replay contract.
3. Correct ML evidence gaps needed by notebooks: dummy baseline, fold scores, full metrics, OOF threshold/calibration, error analysis, explainability metadata.
4. Refactor notebook generation into assessment, decision, computation, narrative, rendering, execution, and validation services.
5. Execute notebooks in an isolated job; embed outputs and validate reload/prediction equality.
6. Produce versioned notebook, HTML, PDF-capability, summary JSON, metrics CSV, schema, and pipeline artifacts.
7. Add the Generated ML Notebook progress/results page using real job state.
8. Continue master-plan API, database, authentication, frontend, deployment, and monitoring phases while preserving local mode.

## 19. Phase 1 Baseline Checks

| Check | Result |
|---|---|
| Python compilation | Passed |
| Installed dependency consistency (`pip check`) | Passed |
| JavaScript syntax (`node --check`) | Passed |
| Ruff | Not configured/installed |
| mypy | Not configured/installed |
| pytest | Not configured/installed |
| Frontend lint/type/build | Not applicable; no package manifest |
| Git status | Failed because repository metadata is absent |
