"""Execute and render complete AivoraML machine-learning report bundles."""

from __future__ import annotations

import base64
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# pyzmq cannot use the Windows Proactor loop's ``add_reader`` API.  Report
# kernels run on worker threads, so selecting the compatible policy here keeps
# their newly-created loops native instead of making Tornado add a helper
# selector thread for every export.
if os.name == "nt":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if selector_policy is not None:
            asyncio.set_event_loop_policy(selector_policy())

import joblib
import nbformat
import numpy as np
import pandas as pd
import pdfplumber
from jupyter_client.kernelspec import NoSuchKernel
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .notebook_generator_pro import generate_professional_notebook


logger = logging.getLogger("aivoraml.reports")


class ReportGenerationError(RuntimeError):
    """Known report-generation failure safe to expose as a structured error."""

    code = "REPORT_GENERATION_FAILED"


class ReportInputError(ReportGenerationError):
    code = "REPORT_INPUT_INVALID"


class ReportVerificationError(ReportGenerationError):
    code = "REPORT_VERIFICATION_FAILED"


class NotebookExecutionError(ReportGenerationError):
    code = "NOTEBOOK_EXECUTION_FAILED"


def _shutdown_notebook_kernel(executor: ExecutePreprocessor) -> None:
    """Best-effort immediate teardown, including interrupted Windows exports."""
    kernel_client = getattr(executor, "kc", None)
    kernel_manager = getattr(executor, "km", None)
    try:
        if kernel_manager is not None:
            kernel_manager.shutdown_kernel(now=True)
    except Exception:
        logger.warning("Could not immediately stop the report kernel.", exc_info=True)
    try:
        if kernel_client is not None:
            kernel_client.stop_channels()
    except Exception:
        logger.warning("Could not stop the report kernel channels.", exc_info=True)
    try:
        if kernel_manager is not None:
            kernel_manager.cleanup_resources()
    except Exception:
        logger.warning("Could not release all report kernel resources.", exc_info=True)
    executor.kc = None
    executor.km = None


def _execute_notebook(
    executor: ExecutePreprocessor,
    notebook: nbformat.NotebookNode,
    work_dir: Path,
) -> None:
    try:
        executor.preprocess(
            notebook,
            resources={"metadata": {"path": str(work_dir)}},
        )
    except (
        CellExecutionError,
        CellTimeoutError,
        DeadKernelError,
        NoSuchKernel,
        RuntimeError,
    ) as exc:
        raise NotebookExecutionError(
            "Notebook kernel startup or execution failed; inspect the server traceback."
        ) from exc
    finally:
        # NotebookClient normally performs graceful cleanup itself.  This
        # second, idempotent guard matters when Ctrl+C or a kernel failure
        # interrupts that context manager before Windows releases its files.
        _shutdown_notebook_kernel(executor)


def _cleanup_report_workdir(
    path: Path,
    *,
    attempts: int = 8,
    retry_delay_seconds: float = 0.25,
) -> bool:
    """Remove a kernel workspace without surfacing transient Windows locks."""
    for attempt in range(max(1, attempts)):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt + 1 < max(1, attempts):
                time.sleep(retry_delay_seconds * (attempt + 1))
    logger.warning(
        "Report work directory is still locked and will be reclaimed later: %s",
        path,
    )
    return False


def _cleanup_stale_report_workdirs(
    root: Path,
    *,
    minimum_age_seconds: float = 900,
) -> None:
    """Reclaim abandoned kernel directories from earlier interrupted servers."""
    cutoff = time.time() - minimum_age_seconds
    try:
        candidates = list(root.iterdir())
    except OSError:
        return
    for candidate in candidates:
        try:
            stale = candidate.is_dir() and candidate.stat().st_mtime <= cutoff
        except OSError:
            continue
        if stale:
            _cleanup_report_workdir(candidate, attempts=2)


@contextmanager
def _report_workdir(root: Path, *, prefix: str):
    """Yield an explicit workspace whose cleanup never runs again at exit."""
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    try:
        yield path
    finally:
        _cleanup_report_workdir(path)


def _safe_name(filename: str) -> str:
    stem = Path(str(filename).replace("\\", "/")).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug[:80] or "aivoraml_project"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _first_value(candidate: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = candidate.get(key)
        if value is not None:
            return value
    return ""


def _pdf_text_has_required_sections(text: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", text).strip()
    return (
        "Machine Learning Report" in normalized_text
        and "Final Evaluation" in normalized_text
    )


def _extract_png_outputs(notebook: nbformat.NotebookNode, directory: Path) -> list[Path]:
    images: list[Path] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            encoded = (output.get("data") or {}).get("image/png")
            if not encoded:
                continue
            image_path = directory / f"chart_{len(images) + 1:02d}.png"
            image_path.write_bytes(base64.b64decode(encoded))
            images.append(image_path)
    return images


def _validate_executed_notebook(notebook: nbformat.NotebookNode) -> None:
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise ReportVerificationError(
            "Generated notebook contains no code cells."
        )
    incomplete = [
        index
        for index, cell in enumerate(code_cells, start=1)
        if cell.get("execution_count") is None
    ]
    if incomplete:
        raise ReportVerificationError(
            f"Notebook code cells were not executed: {incomplete}"
        )
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise ReportVerificationError(
            "Notebook execution contains error outputs: "
            + "; ".join(str(error.get("evalue") or "unknown") for error in errors)
        )


def _page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 10 * mm, "AivoraML ML Report")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def _styled_table(rows: list[list[Any]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5AA8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _report_pdf(
    path: Path,
    *,
    project_name: str,
    profile: dict[str, Any],
    result: dict[str, Any],
    chart_paths: list[Path],
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        textColor=colors.HexColor("#172033"),
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#2F5AA8"),
        spaceBefore=12,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="SmallBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    ))
    styles.add(ParagraphStyle(
        name="TinyBody",
        parent=styles["SmallBody"],
        fontSize=6.5,
        leading=8,
        wordWrap="CJK",
    ))
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"{project_name} ML Report",
        author="AivoraML",
    )
    story: list[Any] = [
        Spacer(1, 14 * mm),
        Paragraph(f"{project_name} Machine Learning Report", styles["ReportTitle"]),
        Paragraph(
            "Executed, evidence-based project summary",
            ParagraphStyle(
                name="Subtitle",
                parent=styles["BodyText"],
                alignment=TA_CENTER,
                textColor=colors.HexColor("#64748B"),
                fontSize=11,
            ),
        ),
        Spacer(1, 8 * mm),
    ]
    rows = profile.get("n_rows") or (result.get("data_shape") or {}).get("rows")
    columns = profile.get("n_cols") or (result.get("data_shape") or {}).get("columns")
    overview = [
        ["Measure", "Value"],
        ["Task", str(result.get("task_type") or "unknown").title()],
        ["Target", str(result.get("target_column") or "unknown")],
        ["Dataset shape", f"{rows or 'unknown'} rows x {columns or 'unknown'} columns"],
        ["Champion", str(result.get("best_model") or "not available")],
        ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ]
    story.extend([_styled_table(overview, [42 * mm, 112 * mm]), Spacer(1, 6 * mm)])

    story.append(Paragraph("1. Data Quality", styles["Section"]))
    quality_data = [
        ["Measure", "Computed value"],
        ["Missing cells", str(profile.get("total_missing", "unknown"))],
        ["Duplicate rows", str(profile.get("n_duplicates", "unknown"))],
        ["Quality warnings", str(len(profile.get("warnings") or []))],
        [
            "Dataset fingerprint",
            Paragraph(
                str(profile.get("dataset_fingerprint") or "not recorded"),
                styles["TinyBody"],
            ),
        ],
    ]
    story.extend([_styled_table(quality_data, [55 * mm, 99 * mm]), Spacer(1, 5 * mm)])

    story.append(Paragraph("2. Model Comparison", styles["Section"]))
    candidates = result.get("models_trained") or result.get("models_compared") or []
    comparison = [["Model", "CV mean", "CV std", "Train score", "Best"]]
    for candidate in candidates[:15]:
        comparison.append([
            str(candidate.get("name") or ""),
            str(_first_value(candidate, "cv_score_mean", "selection_score")),
            str(_first_value(candidate, "cv_score_std")),
            str(_first_value(candidate, "training_score")),
            "Yes" if candidate.get("is_best") else "",
        ])
    story.extend([
        _styled_table(
            comparison,
            [58 * mm, 25 * mm, 23 * mm, 29 * mm, 19 * mm],
        ),
        Spacer(1, 5 * mm),
    ])

    metrics = (result.get("evaluation_details") or {}).get("metrics") or {}
    metric_rows = [["Metric", "Value"]] + [
        [
            str(key).replace("_", " ").title(),
            f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value),
        ]
        for key, value in metrics.items()
    ]
    evaluation_content = [
        Paragraph("3. Final Evaluation", styles["Section"]),
        _styled_table(metric_rows, [78 * mm, 76 * mm]),
        Spacer(1, 4 * mm),
    ]
    threshold = result.get("decision_threshold")
    if threshold is not None:
        evaluation_content.append(Paragraph(
            f"The binary decision threshold is {float(threshold):.2f}. It was "
            "selected from training-only out-of-fold predictions; the final holdout "
            "was not used to optimize it.",
            styles["SmallBody"],
        ))
    story.append(KeepTogether(evaluation_content))

    for index, chart_path in enumerate(chart_paths[:8], start=1):
        if index == 1:
            story.append(Paragraph("4. Executed Notebook Charts", styles["Section"]))
        chart = Image(str(chart_path))
        scale = min(
            (160 * mm) / chart.imageWidth,
            (92 * mm) / chart.imageHeight,
        )
        chart.drawWidth = chart.imageWidth * scale
        chart.drawHeight = chart.imageHeight * scale
        story.append(KeepTogether([
            Paragraph(f"Chart {index}", styles["SmallBody"]),
            chart,
            Spacer(1, 5 * mm),
        ]))

    story.extend([
        PageBreak(),
        Paragraph("5. Limitations and Next Steps", styles["Section"]),
        Paragraph(
            "Results depend on the uploaded data, target definition, split "
            "assumptions, and future inputs matching the recorded feature contract. "
            "Correlation and feature importance describe association with model "
            "behavior, not causation. Validate domain constraints, subgroup errors, "
            "privacy, and monitoring requirements before consequential use.",
            styles["SmallBody"],
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "Recommended next steps: review the executed notebook with a domain "
            "expert, validate the champion on newly collected data, define deployment "
            "and drift controls, and retrain through a versioned review process.",
            styles["SmallBody"],
        ),
    ])
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def generate_report_bundle(
    *,
    project_root: Path,
    result: dict[str, Any],
    profile: dict[str, Any],
    dataset: pd.DataFrame,
    filename: str,
    timeout_seconds: int = 900,
    kernel_startup_timeout_seconds: int = 180,
    progress_callback: Callable[[str, list[str]], None] | None = None,
) -> dict[str, Any]:
    """Generate, execute, render, save, and verify one project report bundle."""
    if dataset is None or dataset.empty:
        raise ReportInputError(
            "A non-empty dataset is required for report generation."
        )
    if not result or result.get("_pipeline") is None:
        raise ReportInputError(
            "A fitted workflow is required for report generation."
        )

    project_root = project_root.resolve()
    notebook_dir = project_root / "notebooks" / "generated"
    report_dir = project_root / "reports"
    model_dir = project_root / "models"
    temp_root = project_root / "outputs" / "report-work"
    for directory in (notebook_dir, report_dir, model_dir, temp_root):
        directory.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_report_workdirs(temp_root)

    slug = _safe_name(filename)
    notebook_path = notebook_dir / f"{slug}_complete_ml_workflow.ipynb"
    html_path = report_dir / f"{slug}_ml_report.html"
    pdf_path = report_dir / f"{slug}_ml_report.pdf"
    summary_path = report_dir / f"{slug}_model_summary.json"
    metrics_path = report_dir / f"{slug}_metrics.csv"
    schema_path = model_dir / f"{slug}_feature_schema.json"
    pipeline_path = model_dir / f"{slug}_pipeline.joblib"

    safe_result = _json_safe(result)
    safe_profile = _json_safe(profile)
    completed: list[str] = []

    def progress(section: str, *newly_completed: str) -> None:
        completed.extend(
            item for item in newly_completed if item not in completed
        )
        if progress_callback is not None:
            progress_callback(section, list(completed))

    with _report_workdir(temp_root, prefix=f"{slug}-") as work_dir:
        dataset_name = f"{slug}_source.csv"
        dataset.to_csv(work_dir / dataset_name, index=False)
        notebook = generate_professional_notebook(
            safe_result,
            filename=dataset_name,
            profile=safe_profile,
        )
        progress("Executing complete notebook", "Notebook generated")
        executor = ExecutePreprocessor(
            timeout=timeout_seconds,
            startup_timeout=kernel_startup_timeout_seconds,
            kernel_name="python3",
            allow_errors=False,
            shutdown_kernel="immediate",
            extra_arguments=["--Application.log_level=ERROR"],
        )
        _execute_notebook(executor, notebook, work_dir)
        _validate_executed_notebook(notebook)
        nbformat.write(notebook, notebook_path)
        progress(
            "Reloading exported pipeline",
            "All cells executed",
            "Outputs embedded",
        )

        generated_model = work_dir / "aivoraml_model.joblib"
        if not generated_model.exists():
            raise ReportVerificationError(
                "Executed notebook did not create its model artifact."
            )
        artifact = joblib.load(generated_model)
        loaded_pipeline = artifact.get("pipeline") if isinstance(artifact, dict) else artifact
        if loaded_pipeline is None or not hasattr(loaded_pipeline, "predict"):
            raise ReportVerificationError(
                "Reloaded notebook artifact has no prediction pipeline."
            )
        shutil.copy2(generated_model, pipeline_path)
        progress("Rendering HTML report", "Pipeline reloaded")

        exporter = HTMLExporter(template_name="lab")
        html, _ = exporter.from_notebook_node(notebook)
        html_path.write_text(html, encoding="utf-8")
        if "<html" not in html.lower() or len(html) < 10_000:
            raise ReportVerificationError("HTML report export is incomplete.")
        progress("Rendering PDF report", "HTML rendered")

        charts = _extract_png_outputs(notebook, work_dir)
        _report_pdf(
            pdf_path,
            project_name=slug.replace("_", " ").title(),
            profile=safe_profile,
            result=safe_result,
            chart_paths=charts,
        )
        if len(PdfReader(str(pdf_path)).pages) < 1:
            raise ReportVerificationError("PDF report has no pages.")
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if not _pdf_text_has_required_sections(text):
            raise ReportVerificationError(
                "PDF report failed content verification."
            )
        progress("Saving metadata and metrics", "PDF rendered and verified")

    candidates = result.get("models_trained") or result.get("models_compared") or []
    metric_rows = []
    for candidate in candidates:
        row = {
            "model": candidate.get("name"),
            "is_best": candidate.get("is_best", False),
            "cv_score_mean": candidate.get("cv_score_mean"),
            "cv_score_std": candidate.get("cv_score_std"),
            "cv_score_min": candidate.get("cv_score_min"),
            "cv_score_max": candidate.get("cv_score_max"),
            "training_score": candidate.get("training_score"),
            "training_time_seconds": candidate.get("training_time_seconds"),
        }
        row.update(candidate.get("metrics") or {})
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    features = result.get("features") or result.get("feature_info") or {}
    feature_schema = {
        "schema_version": 1,
        "target_column": result.get("target_column"),
        "task_type": result.get("task_type"),
        "numeric": features.get("numeric") or [],
        "categorical": features.get("categorical") or [],
        "dropped": features.get("dropped") or [],
        "ordered": [
            *(features.get("numeric") or []),
            *(features.get("categorical") or []),
        ],
    }
    schema_path.write_text(
        json.dumps(_json_safe(feature_schema), indent=2),
        encoding="utf-8",
    )
    checksum_paths = {
        "notebook": notebook_path,
        "html_report": html_path,
        "pdf_report": pdf_path,
        "metrics": metrics_path,
        "feature_schema": schema_path,
        "pipeline": pipeline_path,
    }
    checksums = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in checksum_paths.items()
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": slug,
        "source_filename": filename,
        "dataset_fingerprint": profile.get("dataset_fingerprint"),
        "run_id": (result.get("run_manifest") or {}).get("run_id"),
        "target_column": result.get("target_column"),
        "task_type": result.get("task_type"),
        "best_model": result.get("best_model"),
        "metrics": (result.get("evaluation_details") or {}).get("metrics"),
        "decision_threshold": result.get("decision_threshold"),
        "threshold_selection": result.get("threshold_selection"),
        "run_manifest": result.get("run_manifest"),
        "feature_schema": feature_schema,
        "checksums_sha256": checksums,
        "artifacts": {
            "notebook": notebook_path.name,
            "html_report": html_path.name,
            "pdf_report": pdf_path.name,
            "metrics": metrics_path.name,
            "feature_schema": schema_path.name,
            "pipeline": pipeline_path.name,
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    progress("Complete", "Metadata and metrics saved")

    def relative(path: Path) -> str:
        return str(path.relative_to(project_root)).replace("\\", "/")

    return {
        "status": "completed",
        "project_name": slug,
        "generated_at_utc": summary["generated_at_utc"],
        "run_id": summary["run_id"],
        "checksums_sha256": checksums,
        "artifacts": {
            "notebook": relative(notebook_path),
            "html_report": relative(html_path),
            "pdf_report": relative(pdf_path),
            "model_summary": relative(summary_path),
            "metrics": relative(metrics_path),
            "feature_schema": relative(schema_path),
            "pipeline": relative(pipeline_path),
        },
    }
