# Architecture

## Runtime flow

1. `backend/main.py` exposes the FastAPI routes.
2. `backend/services/` profiles data, trains models, records a run manifest,
   generates EDA, executes notebooks, and renders report bundles.
3. `frontend/` is served by FastAPI at `/` and `/app`.
4. `storage/` holds the active local workspace and prediction history.
5. `backend/services/report_generator.py` creates one internally consistent
   bundle: executed notebook, HTML, PDF, summary JSON, metrics CSV, feature
   schema, and fitted pipeline.
6. Report routes expose real generation state and validated downloads. The
   current local implementation is synchronous; the hosted target architecture
   replaces it with durable background jobs and isolated execution.

## Directory responsibilities

- Put user-provided source datasets in `data/`.
- Put reusable Python package code in `src/` when it is shared beyond the API.
- Put trained model files in `models/`.
- Put temporary report work and intermediate visualizations in `outputs/`.
- Put verified HTML/PDF/metrics/summary exports in `reports/`.
- Put exploratory and exported notebooks in `notebooks/`.
- Keep private application state in `storage/`; its contents are ignored.

## Trust boundary

This is a single-user local application. Production deployment requires
authentication, per-user storage isolation, managed secrets, job orchestration,
monitoring, network hardening, and sandboxed notebook execution. The local
kernel execution path is intended only for trusted local projects.
