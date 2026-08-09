# AivoraML Implementation Plan

**Planning basis, highest precedence first:** `docs/PROJECT_PLAN.md`; attached **ML NOTEBOOK AND OUTPUT QUALITY REQUIREMENT**; `README.md`; existing implementation.  
**Current-state evidence:** `IMPLEMENTATION_AUDIT.md`.  
**Rule:** The complete executed notebook/report bundle is a mandatory product capability, not an optional export. The current local mode remains supported while the master-plan hosted architecture is implemented incrementally.

## 2026-08-08 Local Release Checkpoint

The documented single-user loopback release is complete at the 98/100 gate in
`IMPLEMENTATION_AUDIT.md`: real background training/report jobs, isolated
projects, immutable model versions, batch inference, richer diagnostics,
verified executed artifacts, responsive/accessibility work, and 51 release
tests are integrated. Remaining roadmap work is explicitly hosted scope:
durable external workers, authentication/tenant isolation, and an automated
multi-browser accessibility/visual matrix.

## Status and Priority

- Status: `Complete`, `In progress`, `Ready`, `Planned`, `Blocked`, or `Deferred`.
- Priority: `P0` correctness/security, `P1` core product quality, `P2` enhancement, `P3` future scope.
- A task is complete only after implementation, integration, validation, tests, documentation, and acceptance criteria pass.

## Mandatory Notebook and Report Workstream

This workstream blocks any claim that generated ML projects are complete. It runs across the ML, report, API, job, artifact, frontend, and testing phases. Tasks are ordered and must not be reduced to a larger static notebook template.

### DOC-001 — Reconcile the canonical specification

| Field | Detail |
|---|---|
| Feature name | One unambiguous master-document hierarchy |
| Master requirement | `docs/PROJECT_PLAN.md` is primary; attached notebook standard is mandatory; README is overview/setup |
| Current state | Reconciled: README points to `docs/PROJECT_PLAN.md`; normative notebook standard is referenced there; redundant root file removed |
| Files to create | No new specification unless an ADR/change record is required |
| Files to modify | `README.md`, `docs/PROJECT_PLAN.md`; deprecate/remove root conflicting file only after ownership confirmation |
| Dependencies | User confirmation received |
| Risks | Losing user-authored material or maintaining two contradictory masters |
| Tests required | Link validation, single-authority text scan, table-of-contents/heading check |
| Acceptance criteria | README points to `docs/PROJECT_PLAN.md`; notebook standard is incorporated or normatively referenced there; no file claims higher authority |
| Priority / complexity / status | P0 / Low / Complete |

### NBQ-001 — Immutable run manifest and exact replay contract

| Field | Detail |
|---|---|
| Feature name | One source of truth for training, notebook, report, and inference |
| Master requirement | Reproducibility, training/inference parity, lineage, technically honest outputs |
| Current state | Local run manifest now records a unique run ID, dataset/schema, cleaning, split, balancing, candidates, fold scores, champion, threshold, seed, and artifact checksums; immutable hosted storage and checksum enforcement remain |
| Files to create | `src/aivoraml/ml/run_manifest.py`, manifest schemas, lineage tests |
| Files to modify | ML engine, workflow persistence, model/notebook exports |
| Dependencies | Versioned dataset snapshot and artifact interface |
| Risks | Migrating snapshot v1; non-JSON estimator/search values |
| Tests required | Manifest serialization, exact configuration replay, checksum/lineage mismatch rejection |
| Acceptance criteria | Dataset, split, preprocessing, balancing, candidates, folds, search, champion, threshold, metrics, seeds, versions, and artifacts share one immutable run ID |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-002 — Dataset assessment and decision engine

| Field | Detail |
|---|---|
| Feature name | Evidence-driven understanding, quality, EDA, outlier, cleaning, skewness, encoding, and feature decisions |
| Master requirement | Notebook sections 1–13 and dataset-specific adaptation |
| Current state | Generated notebook now contains measured profiling, task-aware EDA, missingness, duplicates, outliers, skewness, correlation caveats, and cleaning/preprocessing decisions; richer date/group/domain validation remains |
| Files to create | Assessment result schemas; quality/outlier/skewness/decision services; fixtures |
| Files to modify | profiler, EDA, preprocessing configuration |
| Dependencies | NBQ-001 dataset version and bounded sampling policy |
| Risks | Invented domain constraints; excessive charts; leakage from target-aware decisions |
| Tests required | Classification/regression/mixed/date/group/time/high-cardinality/invalid-value datasets |
| Acceptance criteria | Every check returns measured evidence, action, rationale, applicability, and warning; unavailable domain decisions require user confirmation rather than invention |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-003 — Complete model comparison evidence

| Field | Detail |
|---|---|
| Feature name | Baseline, candidates, fold scores, tuning, and transparent champion selection |
| Master requirement | Notebook sections 14–18 and 23 |
| Current state | Dummy baseline, fold-score history, candidate parameters, stability-aware selection, balancing parity, and champion agreement are implemented; full tuning replay/search evidence and operational-size criteria remain |
| Files to create | Model catalog, CV/tuning result schemas, selection policy |
| Files to modify | ML engine and stored workflow results |
| Dependencies | NBQ-001 and preprocessing contract |
| Risks | Search optimism, excessive compute, choosing one metric over business needs |
| Tests required | Dummy uplift, per-fold reproducibility, tuning before/after, candidate failure, stability/speed/size criteria |
| Acceptance criteria | Comparison identifies performance, stability, interpretability, speed, gap, probability quality, and selection rationale without holdout-driven tuning |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-004 — Evaluation, threshold, calibration, and error evidence

| Field | Detail |
|---|---|
| Feature name | Task-correct final evaluation and model failure analysis |
| Master requirement | Notebook sections 19–21 |
| Current state | Untouched holdout metrics, ROC/PR/log-loss/Brier/specificity, training-only OOF threshold locking, confusion/residual and worst-error evidence are implemented; calibration plots and subgroup matrix remain |
| Files to create | Evaluation, OOF threshold, calibration, and error-analysis services |
| Files to modify | ML result schema, prediction threshold behavior |
| Dependencies | OOF probabilities and immutable untouched holdout |
| Risks | Test-set threshold leakage, class-order inversion, invalid MAPE/adjusted R² |
| Tests required | Binary/multiclass/regression fixtures, threshold lock, Brier/calibration, worst errors, subgroup minimum sizes |
| Acceptance criteria | Required applicable metrics/plots/tables exist; selected threshold is learned without final test data and applied by saved inference |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-005 — Explainability and final selection evidence

| Field | Detail |
|---|---|
| Feature name | Global/local explanations and multi-criterion champion summary |
| Master requirement | Notebook sections 22–23 |
| Current state | Native/coefficient/permutation global explanation and a measured selection scorecard are generated; local explanation adapters and stronger source-feature aggregation remain |
| Files to create | Explainability adapters, capability registry, selection-summary schema |
| Files to modify | dependencies, ML result, artifact metadata |
| Dependencies | Stable transformed/source feature mapping |
| Risks | Causal overstatement, SHAP cost/incompatibility, sensitive attributes |
| Tests required | Linear/tree/model-agnostic paths, sampled limits, source-name mapping, unsupported method |
| Acceptance criteria | Explanations identify method/version/sample/caveats; final selection table uses measured performance and operational criteria |
| Priority / complexity / status | P1 / High / In progress |

### NBQ-006 — Truthful narrative and consistency validator

| Field | Detail |
|---|---|
| Feature name | Computed interpretations that cannot contradict outputs |
| Master requirement | Sections 28, 29, and 33 |
| Current state | Notebook replay now matches the application champion and threshold and reports computed facts; a formal claim graph/invariant validator for every narrative sentence remains |
| Files to create | Narrative fact model, templating layer, invariant/claim validator |
| Files to modify | notebook generator and report renderer |
| Dependencies | Structured outputs from NBQ-002 through NBQ-005 |
| Risks | Qualitative adjectives without thresholds; localization/format drift |
| Tests required | Duplicate/class-balance/top-feature/tuning/split/transform/metric terminology contradiction tests |
| Acceptance criteria | Every numeric or qualitative claim maps to a structured fact and validation rule; contradictions fail the job |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-007 — Professional adaptive notebook renderer

| Field | Detail |
|---|---|
| Feature name | Complete portfolio-ready notebook structure and code |
| Master requirement | Primary format and sections 1–29, 31–33 |
| Current state | 32-cell adaptive executed notebook with 15 numbered sections, measured outputs, charts, interpretations, scorecard, export/reload, and conclusions; modular builders and the complete applicability matrix remain |
| Files to create | Section builders, code helpers, notebook schema/version, templates |
| Files to modify | Replace/refactor `notebook_generator_pro.py` |
| Dependencies | NBQ-001 through NBQ-006 |
| Risks | Huge notebooks, duplicated work, optional-library failures |
| Tests required | Section applicability matrix, AST/nbformat, PEP 8 where practical, no hardcoded results, notebook size limits |
| Acceptance criteria | Numbered, task-adaptive cells include explanations, code, tables, relevant charts, “Reading this,” “Why this matters,” warnings, conclusions, and scorecard |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-008 — Isolated clean-kernel execution and artifact verification

| Field | Detail |
|---|---|
| Feature name | Execute every cell and prove reproducibility |
| Master requirement | Section 30 and final acceptance criteria |
| Current state | Local generation executes every code cell in a fresh kernel, rejects errors/missing counts, embeds outputs, and reload-checks the exported pipeline; hardened worker isolation/resource policy remains |
| Files to create | Notebook execution worker, sandbox workspace, timeout/resource policy, execution validator |
| Files to modify | dependencies and notebook service |
| Dependencies | Background jobs, dataset/artifact snapshot, NBQ-007 |
| Risks | Arbitrary code/resource use, long jobs, platform-specific kernels |
| Tests required | Clean-kernel classification/regression execution, timeout, missing optional library, chart/output/error detection, reload equality |
| Acceptance criteria | Completion requires all cells executed with embedded outputs, no error outputs, artifacts/checksums present, reload prediction exactly matching original |
| Priority / complexity / status | P0 / Very High / In progress |

### NBQ-009 — HTML, PDF, JSON, CSV, schema, and pipeline bundle

| Field | Detail |
|---|---|
| Feature name | Versioned presentation and machine-readable outputs |
| Master requirement | Primary output paths, reports/exports, model saving |
| Current state | One local service emits and verifies executed notebook, HTML, PDF, summary JSON, metrics CSV, feature schema, and pipeline under one run ID with SHA-256 checksums; versioned immutable hosted storage remains |
| Files to create | Report renderer/exporter, artifact bundle manifest, HTML template, PDF capability adapter |
| Files to modify | artifact/report services and dependencies |
| Dependencies | Executed notebook and immutable artifact store |
| Risks | PDF renderer availability, unsafe HTML, inconsistent duplicates of metrics |
| Tests required | Render verification, HTML sanitization, PDF capability states, JSON schema, CSV formula protection, checksum manifest |
| Acceptance criteria | Required named outputs share run lineage and metrics; HTML always succeeds; PDF succeeds when capability enabled and otherwise reports a truthful unavailable state |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-010 — Notebook/report job API and progress model

| Field | Detail |
|---|---|
| Feature name | Durable generation, progress, warnings, retry, and downloads |
| Master requirement | Target API/jobs architecture and section 34 UX |
| Current state | Local API exposes generate/current state, real completed-section updates, failure warnings, and seven bundle downloads; durable queued jobs, idempotency keys, retry/cancel, and authorization remain |
| Files to create | `/api/v1` notebook/report job routes, schemas, worker task, artifact endpoints |
| Files to modify | FastAPI app and storage/job services |
| Dependencies | Master-plan jobs/database/artifact infrastructure; NBQ-008/009 |
| Risks | Duplicate jobs, stale progress, unauthorized artifact access |
| Tests required | Idempotency, progress transitions, cancellation, retry, failure, authorization, signed downloads |
| Acceptance criteria | Long work returns a job; state is real and durable; completed outputs are immutable and downloadable |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-011 — Generated ML Notebook application page

| Field | Detail |
|---|---|
| Feature name | End-to-end notebook generation user experience |
| Master requirement | Section 34 |
| Current state | Artifact page exposes real generation state plus notebook/HTML/PDF/summary/metrics/schema/pipeline actions and failure feedback; detailed section-level durable progress page remains |
| Files to create | Next.js route/components/hooks or an interim accurate local-mode view |
| Files to modify | navigation, artifact views, API client |
| Dependencies | NBQ-010 API contracts |
| Risks | Simulated progress or stale result summaries |
| Tests required | Loading/empty/error/cancel/complete states, accessibility, API integration, responsive behavior |
| Acceptance criteria | Shows real current/completed sections, warnings, models, champion, metrics, and links for notebook/HTML/PDF/pipeline/metrics/schema |
| Priority / complexity / status | P0 / High / In progress |

### NBQ-012 — Notebook quality acceptance suite

| Field | Detail |
|---|---|
| Feature name | Automated portfolio-quality and technical-honesty gate |
| Master requirement | All 34 sections and final acceptance criteria |
| Current state | Lifecycle test now rejects unexecuted/error cells, champion mismatch, threshold leakage, missing baseline/manifest, malformed report formats, and invalid reload metadata; regression/edge fixtures and visual automation remain |
| Files to create | Golden property fixtures, notebook executor tests, narrative invariants, render/image QA tests |
| Files to modify | Existing smoke suite and CI configuration |
| Dependencies | Every NBQ implementation task |
| Risks | Brittle snapshot tests; excessive CI runtime |
| Tests required | Small classification/regression plus edge-case matrix; fast structural PR tests and scheduled full execution |
| Acceptance criteria | No release passes if outputs are fake, cells fail, narratives conflict, required applicable sections are absent, or artifact reload/prediction differs |
| Priority / complexity / status | P0 / Very High / In progress |

## Phase 1 — Repository Cleanup and Architecture

### RDM-001 — Evidence-based audit and plan

| Field | Detail |
|---|---|
| README requirement | Preserve a clean local-first project with documented layout |
| Current state | Audit and plan requested; no reliable Git metadata |
| Files to create | `IMPLEMENTATION_AUDIT.md`, `IMPLEMENTATION_PLAN.md` |
| Files to modify | None |
| Dependencies | Complete source/artifact inspection |
| Risks | Treating the broader target plan as implemented README scope |
| Tests required | Document structure/status validation; baseline code checks |
| Acceptance criteria | Audit contains every requested category, comparison table, evidence, and risks; plan covers 20 ordered phases |
| Priority / complexity / status | P0 / Medium / Complete |

### RDM-002 — Remove deceptive and dead frontend paths

| Field | Detail |
|---|---|
| README requirement | Real profiling, comparison, prediction, and exports; no fake outputs |
| Current state | Unused hardcoded demo profile/results and scripted “AI” claims exist |
| Files to create | Frontend regression test harness if selected |
| Files to modify | `frontend/app_premium.js`, relevant HTML/docs |
| Dependencies | Preserve real `/api/demo` behavior |
| Risks | Removing data accidentally used by a hidden path |
| Tests required | JS unit/static assertions, browser smoke, demo API integration |
| Acceptance criteria | No fabricated production metrics; unsupported claims removed; real demo still works |
| Priority / complexity / status | P0 / Medium / Ready |

### RDM-003 — Extract backend boundaries incrementally

| Field | Detail |
|---|---|
| README requirement | Maintain working local workspace |
| Current state | 1,508-line `main.py` mixes transport, storage, orchestration, exports |
| Files to create | `backend/api/`, `backend/core/`, `backend/services/project_service.py`, `backend/services/artifact_service.py` |
| Files to modify | `backend/main.py`, tests |
| Dependencies | Characterization tests for current routes |
| Risks | Breaking persisted workspace compatibility or imports |
| Tests required | Existing smoke suite plus route contract/restore tests |
| Acceptance criteria | Routes delegate to services; snapshot v1 restores; public behavior unchanged |
| Priority / complexity / status | P1 / High / Ready |

## Phase 2 — Configuration and Environment Setup

### RDM-010 — Typed configuration and developer tooling

| Field | Detail |
|---|---|
| README requirement | Reproducible setup and optional environment configuration |
| Current state | Requirements file and env example exist; integers parsed at import; no tooling config |
| Files to create | `pyproject.toml`, constraints/lock strategy, optional cross-platform script |
| Files to modify | `.env.example`, README, backend config imports |
| Dependencies | RDM-003 boundaries |
| Risks | Dependency changes alter model results |
| Tests required | Config valid/invalid tests, clean environment install, lint/type checks |
| Acceptance criteria | Typed bounds, safe defaults, actionable config errors, repeatable install |
| Priority / complexity / status | P1 / Medium / Ready |

## Phase 3 — Backend Foundation

### RDM-020 — Versioned/typed local API foundation

| Field | Detail |
|---|---|
| README requirement | FastAPI workspace and `/docs` |
| Current state | Working unversioned API; raw dict payloads and inconsistent errors |
| Files to create | Router modules, response/error schemas, logging and request-ID middleware, health route |
| Files to modify | `backend/main.py`, `backend/schemas.py`, frontend API paths, tests |
| Dependencies | RDM-003, RDM-010 |
| Risks | Compatibility break with current frontend |
| Tests required | OpenAPI contract, status/error, health, legacy compatibility tests |
| Acceptance criteria | Typed routes, consistent errors, request IDs, real health check, frontend connected |
| Priority / complexity / status | P1 / High / Ready |

## Phase 4 — Database and Authentication

### RDM-030 — Implement target-state metadata, identity, and authorization foundation

| Field | Detail |
|---|---|
| README requirement | Preserve the simple loopback local setup as one supported mode |
| Current state | No database/auth; master plan requires PostgreSQL, users, workspaces, RBAC, jobs, artifacts, and audit |
| Files to create | SQLAlchemy models/repositories, Alembic migrations, identity/RBAC services, local-mode adapter, ADRs |
| Files to modify | API dependencies, configuration, README mode documentation |
| Dependencies | Threat model, PostgreSQL, typed API, artifact/run schemas |
| Risks | Tenant data exposure, migration failure, breaking local mode |
| Tests required | Migration, authentication, token lifecycle, RBAC/ownership matrix, local-mode compatibility |
| Acceptance criteria | Hosted resources are tenant-scoped and audited; local loopback mode remains explicit and functional |
| Priority / complexity / status | P0 / Very High / Planned |

## Phase 5 — Dataset Upload and Management

### RDM-040 — Harden local ingestion and workspace lifecycle

| Field | Detail |
|---|---|
| README requirement | Profile uploaded tabular datasets and restore local state |
| Current state | Four formats and limits implemented; full parse precedes shape checks; one workspace |
| Files to create | Upload policy/service, parser tests, local project-clear/retention API |
| Files to modify | upload routes, frontend upload/history UI, env docs |
| Dependencies | Typed config and service layer |
| Risks | Zip/decompression/memory exhaustion; snapshot compatibility |
| Tests required | Valid/corrupt/empty/oversized/duplicate-column/formula/adversarial filename matrix |
| Acceptance criteria | Bounded secure parse, atomic state, clear user errors, no regression in supported formats |
| Priority / complexity / status | P0 / High / Ready |

## Phase 6 — Data Validation and Profiling

### RDM-050 — Make profiling deterministic and fully tested

| Field | Detail |
|---|---|
| README requirement | Profile tabular data |
| Current state | Rich profile exists with distributions, warnings, targets |
| Files to create | unit fixtures and response schemas |
| Files to modify | `dataset_profiler.py`, schemas, profile route |
| Dependencies | RDM-040 |
| Risks | Expensive profiling on quota-sized datasets; silent skipped stats |
| Tests required | Schema inference, histogram totals, missing/duplicate/outlier/target fixtures |
| Acceptance criteria | Deterministic JSON-safe profile; explicit warnings for omitted analyses; bounded runtime |
| Priority / complexity / status | P1 / Medium / Ready |

## Phase 7 — EDA Engine

### RDM-060 — Connect measured EDA to the normal workflow

| Field | Detail |
|---|---|
| README requirement | Master/attached standard requires task-adaptive, question-led EDA; README provides the local overview |
| Current state | Matplotlib/Seaborn service exists but is skipped by default and not exposed as a distinct contract |
| Files to create | EDA schemas/tests and optional endpoint |
| Files to modify | `eda_generator.py`, frontend explore view |
| Dependencies | Profiling and resource budgets |
| Risks | Base64 memory and slow synchronous rendering |
| Tests required | Figure lifecycle, numeric/categorical/empty cases, size/runtime bounds |
| Acceptance criteria | Every displayed chart is measured, failures are explicit, resources are released |
| Priority / complexity / status | P2 / Medium / Ready |

## Phase 8 — Data Cleaning and Preprocessing

### RDM-070 — Connect UI preprocessing choices to fitted pipelines

| Field | Detail |
|---|---|
| README requirement | Compare models with a valid fitted preprocessing pipeline |
| Current state | Fixed leakage-safe backend pipeline; UI switches are cosmetic |
| Files to create | preprocessing config schemas and contract tests |
| Files to modify | frontend clean/features views, training payload, ML engine |
| Dependencies | Typed API |
| Risks | User-selected transformations create leakage or invalid combinations |
| Tests required | imputation, encoding, scaling, unseen category, training/inference parity |
| Acceptance criteria | Only supported choices shown; submitted recipe equals serialized fitted pipeline |
| Priority / complexity / status | P0 / High / Ready |

## Phase 9 — Feature Engineering and Selection

### RDM-080 — Add explicit, fold-safe feature policy

| Field | Detail |
|---|---|
| README requirement | Master/attached standard requires evidence-based feature engineering and selection when justified |
| Current state | Constant/identifier/high-cardinality exclusions only |
| Files to create | feature-policy module and tests |
| Files to modify | ML config/UI and artifact lineage |
| Dependencies | RDM-070 |
| Risks | Target leakage and training/inference mismatch |
| Tests required | identifiers, dates, feature lineage, learned selection inside CV |
| Acceptance criteria | Every feature exists at inference; learned steps fit within folds; lineage exported |
| Priority / complexity / status | P2 / High / Ready |

## Phase 10 — Model Training and Comparison

### RDM-090 — Stabilize candidate comparison and provenance

| Field | Detail |
|---|---|
| README requirement | Compare machine-learning models |
| Current state | Diverse real candidates and holdout evaluation work |
| Files to create | model catalog/config/provenance tests |
| Files to modify | `ml_engine_enhanced.py`, training UI, export metadata |
| Dependencies | RDM-070 |
| Risks | Fixed weighted F1 hides minority performance; resource-heavy models |
| Tests required | classification/regression, candidate failures, seeds, timing, dummy baseline |
| Acceptance criteria | Declared metric, reproducible run, honest comparison, champion tied to config/data fingerprint |
| Priority / complexity / status | P1 / High / Ready |

## Phase 11 — Cross-Validation and Hyperparameter Tuning

### RDM-100 — Improve selection protocol

| Field | Detail |
|---|---|
| README requirement | Reliable model comparison |
| Current state | Fold-safe CV and randomized tuning exist; random split only |
| Files to create | split/tuning policy modules and leakage tests |
| Files to modify | ML engine, schemas, evaluation UI |
| Dependencies | RDM-090 |
| Risks | CV search optimism; repeated holdout inspection |
| Tests required | stratified, group/time policy, rare classes, tuning reproducibility, holdout sentinel |
| Acceptance criteria | Holdout cannot enter selection; tuning protocol disclosed; invalid split requests rejected |
| Priority / complexity / status | P1 / High / Ready |

## Phase 12 — Threshold Optimisation and Calibration

### RDM-110 — Version decision thresholds

| Field | Detail |
|---|---|
| README requirement | Master/attached standard requires OOF/validation threshold optimisation and calibration for applicable classification |
| Current state | Class `predict()` and default threshold 0.5 only |
| Files to create | threshold/calibration services, metadata schema, tests |
| Files to modify | prediction service, evaluation UI, model export |
| Dependencies | Reliable out-of-fold probabilities |
| Risks | Test-set threshold leakage and class-label inversion |
| Tests required | OOF threshold, class order, calibration, serialized threshold inference |
| Acceptance criteria | Threshold learned without holdout leakage and applied identically after reload |
| Priority / complexity / status | P2 / High / Ready |

## Phase 13 — Explainability

### RDM-120 — Replace explanation claims with measured evidence

| Field | Detail |
|---|---|
| README requirement | Master/attached standard requires global/local explainability with provenance and limitations |
| Current state | Native/coefficient importance and scripted prose only |
| Files to create | permutation/optional SHAP service and explanation schemas/tests |
| Files to modify | explain view, AI advisor, dependencies if approved |
| Dependencies | Stable pipeline/feature mapping |
| Risks | Causal misinterpretation, compute cost |
| Tests required | transformed feature names, unsupported model, deterministic sampling |
| Acceptance criteria | Method/provenance/caveats shown; no fabricated local explanation |
| Priority / complexity / status | P2 / High / Ready |

## Phase 14 — Prediction System

### RDM-130 — Harden inference and add batch prediction

| Field | Detail |
|---|---|
| README requirement | Make predictions |
| Current state | Real strict single-row prediction and history exist |
| Files to create | prediction service, batch schemas/tests |
| Files to modify | prediction routes/UI, persistence policy |
| Dependencies | Versioned artifact schema and threshold task |
| Risks | Sensitive input retention, schema drift, large batches |
| Tests required | coercion, missing/unknown/unseen values, reload consistency, batch bounds |
| Acceptance criteria | Single and approved batch inference use exact saved pipeline; retention is visible/configurable |
| Priority / complexity / status | P1 / High / Ready |

## Phase 15 — Reports and Exports

### RDM-140 — Deliver the complete executed notebook/report artifact suite

| Field | Detail |
|---|---|
| README requirement | Export project artifacts; master/attached standards require presentation-ready executed notebook and reports |
| Current state | Seven on-demand export types; notebook is unexecuted; HTML/PDF/summary/schema artifacts missing |
| Files to create | NBQ-007 through NBQ-010 render/execution/artifact services and manifest/model-card schemas |
| Files to modify | model/notebook/project export services and dependencies |
| Dependencies | NBQ-001 through NBQ-006, threshold, feature lineage, durable jobs/artifacts |
| Risks | Joblib trust/compatibility, notebook execution isolation, PDF availability, narrative contradictions |
| Tests required | clean-kernel execution, render verification, isolated load/predict equality, CSV injection, checksums, claim invariants |
| Acceptance criteria | Versioned notebook/HTML/PDF-capability/summary/metrics/schema/pipeline bundle passes all truthfulness and execution gates |
| Priority / complexity / status | P0 / Very High / Planned |

## Phase 16 — Frontend Implementation

### RDM-150 — Make static UI accurate, accessible, and testable

| Field | Detail |
|---|---|
| README requirement | Landing page and ML workspace |
| Current state | Responsive vanilla UI with no frontend test/build toolchain |
| Files to create | minimal test/accessibility configuration; component modules if refactored |
| Files to modify | HTML/JS/CSS |
| Dependencies | Stable API contracts |
| Risks | Large rewrite breaks working UX; unnecessary framework migration |
| Tests required | keyboard/modal/forms, responsive smoke, accessibility scan, JS lint |
| Acceptance criteria | No disconnected or overstated actions; loading/empty/error states verified |
| Priority / complexity / status | P1 / High / Ready |

## Phase 17 — Frontend–Backend Integration

### RDM-160 — Contract-test every user action

| Field | Detail |
|---|---|
| README requirement | End-to-end local workflow |
| Current state | Core actions connected; cleaning/features/progress/assistant are not fully connected |
| Files to create | API client abstraction and integration tests |
| Files to modify | frontend actions and backend routers |
| Dependencies | RDM-020, RDM-150 |
| Risks | UI and API drift |
| Tests required | action-to-endpoint matrix, timeout/error/cancel, restored-state E2E |
| Acceptance criteria | Each action calls a real endpoint, performs valid local UI behavior, or is clearly disabled |
| Priority / complexity / status | P0 / High / Ready |

## Phase 18 — Testing and Security

### RDM-170 — Establish layered quality gates

| Field | Detail |
|---|---|
| README requirement | Smoke test and local trust boundary |
| Current state | Five passing lifecycle tests; no lint/type/security suites |
| Files to create | unit/integration/security fixtures and CI-ready tooling config |
| Files to modify | smoke tests, README testing instructions |
| Dependencies | Incremental feature work |
| Risks | Slow ML tests and flaky shared state |
| Tests required | All categories listed in audit plus loopback/trust tests |
| Acceptance criteria | Fast isolated suite, retained E2E, lint/type checks, no critical security failures |
| Priority / complexity / status | P0 / High / Ready |

## Phase 19 — Docker and Deployment

### RDM-180 — Package local and hosted deployment safely

| Field | Detail |
|---|---|
| README requirement | Preserve local loopback run; master plan separately defines secured hosted deployment |
| Current state | Windows launcher only |
| Files to create | Dockerfiles, Compose, Nginx, deployment manifests/runbook, CI release workflow |
| Files to modify | README and deployment documentation |
| Dependencies | Auth/RBAC, jobs, database, artifact store, observability |
| Risks | Publishing unauthenticated local mode; migration/secret/supply-chain failures |
| Tests required | Image/SBOM scan, non-root health, migration, staging smoke, canary/rollback |
| Acceptance criteria | Local image defaults safely; hosted mode requires full security controls and verified rollback |
| Priority / complexity / status | P1 / High / Planned |

## Phase 20 — Monitoring and Production Readiness

### RDM-190 — Implement observability and production readiness

| Field | Detail |
|---|---|
| README requirement | Local overview; master plan requires logs, metrics, alerts, drift, SLOs, backup, and recovery |
| Current state | Console prints only; no model/service monitoring |
| Files to create | OpenTelemetry/logging config, dashboards, alerts, drift service, runbooks |
| Files to modify | services, workers, deployment, operations documentation |
| Dependencies | Auth/database/jobs/deployment and immutable model/run lineage |
| Risks | Sensitive data in telemetry, noisy drift alerts, false readiness claims |
| Tests required | Redaction, trace propagation, alert, backup/restore, drift, rollback, failure injection |
| Acceptance criteria | SLOs are measurable; alerts have runbooks; recovery and rollback are rehearsed; ML telemetry uses minimum sample safeguards |
| Priority / complexity / status | P1 / Very High / Planned |

## Proposed Execution Order

1. DOC-001 specification reconciliation, after confirming ownership of concurrent files.
2. RDM-002 plus NBQ characterization and contradiction tests.
3. NBQ-001 immutable run manifest and dataset/artifact lineage.
4. RDM-010, RDM-003, and RDM-020 foundations without breaking local mode.
5. RDM-040/RDM-050 and NBQ-002 measured assessment/decision evidence.
6. RDM-070/RDM-080/RDM-090/RDM-100 and NBQ-003 exact model evidence.
7. RDM-110/RDM-120/RDM-130 and NBQ-004/NBQ-005 evaluation, threshold, errors, explanations, and inference parity.
8. NBQ-006/NBQ-007 truthful narrative and adaptive notebook rendering.
9. RDM-030 job/artifact prerequisites followed by NBQ-008/NBQ-009 isolated execution and reports.
10. NBQ-010/NBQ-011 plus RDM-150/RDM-160 Generated ML Notebook API/page integration.
11. NBQ-012, RDM-140, and RDM-170 full acceptance gates.
12. Continue remaining master-plan deployment and monitoring work in RDM-180/RDM-190.

## Phase Update Record

| Date | Phase | Result | Tests | Remaining blockers |
|---|---|---|---|---|
| 2026-07-28 | Phase 1 audit/planning | Documents created; no major code refactor | Python compile, pip check, JS syntax, and all 5 smoke tests passed | No Git metadata; README describes only local mode while master plan defines target state |
| 2026-07-28 | Notebook/output audit revision | Master-plan precedence, 34-part notebook matrix, and 12-task NBQ workstream added; no application/PDF changes | Documentation structure and consistency checks | Current notebook is unexecuted/not exact replay; job/artifact infrastructure missing; concurrent README/root-master conflict needs ownership confirmation |
| 2026-07-28 | Local P0 notebook/report implementation | Canonical plan reconciled; run manifest, baseline/fold evidence, OOF threshold, executed adaptive notebook, verified HTML/PDF/JSON/CSV/schema/pipeline bundle, API routes, and live artifact UI implemented | Python compile, JavaScript syntax, strict five-test lifecycle suite, notebook clean-kernel execution, artifact reload, PDF text/page validation, and manual six-page visual QA passed | Durable hosted jobs, immutable checksummed storage, authentication/isolation, deeper explainability/calibration, and classification/regression edge-case matrix remain |
