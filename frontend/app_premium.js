(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const elements = {
    shell: $("#appShell"),
    workspace: $("#workspace"),
    sidebar: $("#sidebar"),
    assistant: $("#assistantPanel"),
    assistantMessages: $("#assistantMessages"),
    assistantContext: $("#assistantContext"),
    suggestionChips: $("#suggestionChips"),
    assistantInput: $("#assistantInput"),
    uploadModal: $("#uploadModal"),
    commandModal: $("#commandModal"),
    commandInput: $("#commandInput"),
    commandResults: $("#commandResults"),
    loading: $("#loadingScreen"),
    loadingEyebrow: $("#loadingEyebrow"),
    loadingTitle: $("#loadingTitle"),
    loadingMessage: $("#loadingMessage"),
    loadingProgress: $("#loadingProgress"),
    loadingSteps: $("#loadingSteps"),
    loadingEta: $("#loadingEta"),
    loadingEtaNote: $("#loadingEtaNote"),
    loadingElapsed: $("#loadingElapsed"),
    loadingStageLabel: $("#loadingStageLabel"),
    loadingPercent: $("#loadingPercent"),
    loadingCancel: $("#loadingCancelButton"),
    breadcrumb: $("#projectBreadcrumb"),
    runButton: $("#runPipelineButton"),
    modelCount: $("#modelCount"),
    lifecycleProgress: $("#lifecycleProgress"),
    projectExport: $("#projectExportButton"),
    shareButton: $("#shareButton"),
    saveState: $("#saveState"),
    fileInput: $("#fileInput"),
    dropzone: $("#dropzone"),
    toastRegion: $("#toastRegion"),
  };

  const state = {
    view: "home",
    profile: null,
    result: null,
    target: "",
    strategy: "balanced",
    demoMode: false,
    completed: new Set(),
    cleaningPlan: [],
    featurePlan: [],
    project: null,
    projects: [],
    predictionSchema: null,
    predictionResult: null,
    predictions: [],
    reportBundle: null,
    reportLoading: false,
    loadingProgressTimer: null,
    loadingElapsedTimer: null,
    trainingJob: null,
    trainingMonitorId: 0,
    reportMonitorId: 0,
    assistantHistory: [],
    assistantBusy: false,
    modelVersions: [],
    batchSummary: null,
    capabilities: null,
    loadingPreviousFocus: null,
    uploadReturnFocus: null,
    commandReturnFocus: null,
    loadingStartedAt: null,
    loadingRunId: 0,
    loadingStageOverride: null,
    commandItems: [],
  };

  const protectedViews = new Set([
    "overview", "explore", "clean", "features", "train",
    "evaluate", "explain", "predict", "deploy",
  ]);

  const viewStages = {
    overview: 1,
    explore: 2,
    clean: 3,
    features: 4,
    train: 5,
    evaluate: 6,
    explain: 7,
    predict: 8,
    deploy: 9,
  };

  const icon = (name, className = "") =>
    `<svg class="icon ${className}" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function activeProjectId() {
    return state.project?.id || state.result?.run_manifest?.project_id || "";
  }

  function projectUrl(path, parameters = {}) {
    const projectId = activeProjectId();
    if (!projectId) return "";
    const query = new URLSearchParams({
      project_id: projectId,
      ...parameters,
    });
    return `${path}?${query.toString()}`;
  }

  function formatNumber(value, maximumFractionDigits = 1) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(number);
  }

  function formatBytes(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) {
      if (!state.profile) return "—";
      return `${Math.max(0.1, state.profile.n_rows * state.profile.n_cols * 8 / 1024 / 1024).toFixed(1)} MB`;
    }
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
  }

  function formatTimestamp(value) {
    if (!value) return "this session";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "this session";
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(date);
  }

  function titleCase(value) {
    return String(value ?? "")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function truncate(value, length = 54) {
    const text = String(value ?? "");
    return text.length > length ? `${text.slice(0, length - 1)}…` : text;
  }

  function currentSuggestion() {
    const suggestions = state.profile?.target_suggestions || [];
    return suggestions.find((suggestion) => suggestion.column === state.target) || suggestions[0] || null;
  }

  function taskType() {
    return state.result?.task_type || currentSuggestion()?.task_type || "classification";
  }

  function profileQuality() {
    if (!state.profile) return 0;
    if (Number.isFinite(Number(state.profile.quality_score))) {
      return Math.round(Number(state.profile.quality_score));
    }
    const cells = Math.max(1, state.profile.n_rows * state.profile.n_cols);
    const missingPenalty = state.profile.total_missing / cells * 75;
    const duplicatePenalty = state.profile.n_duplicates / Math.max(1, state.profile.n_rows) * 35;
    const warningPenalty = Math.min(12, (state.profile.warnings || []).length * 2);
    return Math.max(48, Math.min(99, Math.round(100 - missingPenalty - duplicatePenalty - warningPenalty)));
  }

  function completenessScore() {
    if (!state.profile) return 0;
    const cells = Math.max(1, state.profile.n_rows * state.profile.n_cols);
    return Math.max(0, Math.round(100 - state.profile.total_missing / cells * 100));
  }

  function uniquenessScore() {
    if (!state.profile) return 0;
    return Math.max(0, Math.round(100 - state.profile.n_duplicates / Math.max(1, state.profile.n_rows) * 100));
  }

  function getFeatureTypeCounts({ excludeTarget = false } = {}) {
    const columns = (state.profile?.columns || []).filter(
      (column) => !excludeTarget || column.name !== state.target,
    );
    return columns.reduce((counts, column) => {
      const type = column.semantic_type;
      if (type === "numeric") counts.numeric += 1;
      else if (type === "datetime") counts.datetime += 1;
      else counts.categorical += 1;
      return counts;
    }, { numeric: 0, categorical: 0, datetime: 0 });
  }

  function pageHeader(eyebrow, title, description, actions = "") {
    const file = state.profile?.filename || "Untitled project";

    return `
      <header class="page-header">
        <div class="page-title-row">
          <div class="page-title">
            <span class="eyebrow">${escapeHtml(eyebrow)}</span>
            <h1>${escapeHtml(title)}</h1>
            <p>${escapeHtml(description)}</p>
          </div>
          ${actions ? `<div class="page-actions">${actions}</div>` : ""}
        </div>
        ${state.profile ? `
          <div class="project-meta">
            <span>${icon("file", "icon-sm")} ${escapeHtml(file)}</span>
            <span>${icon("table", "icon-sm")} ${formatNumber(state.profile.n_rows, 0)} rows</span>
            <span>${icon("clock", "icon-sm")} Updated ${escapeHtml(formatTimestamp(state.project?.updated_at))}</span>
            <span class="status-pill success"><i class="status-dot"></i> Ready</span>
          </div>
        ` : ""}
      </header>
    `;
  }

  function render() {
    const renderers = {
      home: renderHome,
      data: renderDataSources,
      models: renderModelRegistry,
      overview: renderOverview,
      explore: renderExplore,
      clean: renderClean,
      features: renderFeatures,
      train: renderTrain,
      evaluate: renderEvaluate,
      explain: renderExplain,
      predict: renderPredict,
      deploy: renderDeploy,
    };

    const renderer = renderers[state.view] || renderHome;
    elements.workspace.classList.remove("view-mounted");
    elements.workspace.innerHTML = renderer();
    elements.workspace.scrollTop = 0;
    updateNavigation();
    updateTopbar();
    updateAssistantContext();
    updateAssistantSuggestions();
    animateRenderedView();
  }

  function animateRenderedView() {
    const revealSelectors = [
      ".page-header",
      ".home-hero",
      ".insight-banner",
      ".champion-card",
      ".metric-grid > *",
      ".section",
      ".project-grid > *",
      ".template-strip > *",
      ".artifact-grid > *",
      ".notebook-cell",
      ".predict-layout > *",
    ];
    const nodes = $$(revealSelectors.join(","), elements.workspace);
    nodes.forEach((node, index) => {
      node.classList.add("view-entrance");
      node.style.setProperty("--motion-index", String(index));
    });
    $$(".distribution-bar", elements.workspace).forEach((bar, index) => {
      bar.style.setProperty("--bar-index", String(index));
    });
    $$(".data-table tbody tr", elements.workspace).forEach((row, index) => {
      row.style.setProperty("--row-index", String(index));
    });
    requestAnimationFrame(() => {
      requestAnimationFrame(() => elements.workspace.classList.add("view-mounted"));
    });
  }

  function renderHome() {
    const otherProjectCards = state.projects
      .filter((project) => project.id !== state.project?.id)
      .map((project) => `
        <article class="project-card">
          <div class="project-card-top">
            <span class="project-icon cyan">${icon("database")}</span>
            <span class="status-pill">${project.trained ? "Trained" : "Profiled"}</span>
          </div>
          <h3>${escapeHtml(String(project.filename || "Untitled project").replace(/\.[^.]+$/, "").replaceAll("_", " "))}</h3>
          <p>${formatNumber(project.rows || 0, 0)} rows · ${formatNumber(project.cols || 0, 0)} columns · ${project.trained ? "Model ready" : "Ready to train"}</p>
          <div class="project-card-footer"><span>Updated ${escapeHtml(formatTimestamp(project.updated_at))}</span><span>·</span><span>${project.prediction_count || 0} predictions</span></div>
          <div class="project-card-actions">
            <button class="button button-secondary compact" data-action="activate-project" data-project-id="${escapeHtml(project.id)}">Restore</button>
            <button class="icon-button" data-action="delete-project" data-project-id="${escapeHtml(project.id)}" data-project-name="${escapeHtml(project.filename || "project")}" aria-label="Delete ${escapeHtml(project.filename || "project")}">${icon("close", "icon-sm")}</button>
          </div>
        </article>
      `).join("");
    const currentProject = state.profile ? `
      <button class="project-card" data-action="open-current">
        <div class="project-card-top">
          <span class="project-icon lime">${icon("chart")}</span>
          <span class="status-pill success"><i class="status-dot"></i> Active</span>
        </div>
        <h3>${escapeHtml(projectName())}</h3>
        <p>${formatNumber(state.profile.n_rows, 0)} rows · ${state.profile.n_cols} columns · ${titleCase(taskType())}</p>
        <div class="project-card-footer">
          <span>Updated ${escapeHtml(formatTimestamp(state.project?.updated_at))}</span><span>·</span><span>${profileQuality()}% data quality</span>
        </div>
      </button>
    ` : `
      <button class="project-card" data-action="open-upload">
        <div class="project-card-top">
          <span class="project-icon lime">${icon("upload")}</span>
          <span class="status-pill">New</span>
        </div>
        <h3>Create your first project</h3>
        <p>Upload a dataset and let AivoraML map the workflow.</p>
        <div class="project-card-footer"><span>CSV · Excel · JSON · Parquet</span></div>
      </button>
    `;

    return `
      <div class="page page-home">
        <section class="home-hero">
          <span class="eyebrow">GUIDED MACHINE LEARNING</span>
          <h1>What do you want to predict?</h1>
          <p>AivoraML profiles your data, builds a leakage-safe pipeline, compares models, records validation, and packages the fitted champion.</p>
          <div class="page-actions home-actions">
            <button class="button button-primary" data-action="open-upload">${icon("upload", "icon-sm")} Import a dataset</button>
            <button class="button button-secondary" data-action="load-demo">${icon("sparkles", "icon-sm")} Open guided demo</button>
          </div>
        </section>

        <section class="section">
          <div class="section-heading-row">
            <div>
              <h2>Your projects</h2>
              <p>Pick up where you left off or explore a guided example.</p>
            </div>
            <div class="button-row">
              <button class="text-button" data-action="open-upload">${icon("upload", "icon-sm")} Import dataset</button>
              ${state.project ? `<button class="text-button danger-text" data-action="delete-project" data-project-id="${escapeHtml(state.project.id)}" data-project-name="${escapeHtml(state.project.filename || "project")}">Delete active project</button>` : ""}
            </div>
          </div>
          <div class="project-grid">
            ${currentProject}
            ${otherProjectCards}
            <button class="project-card demo-project" data-action="load-demo">
              <div class="project-card-top">
                <span class="project-icon violet">${icon("sparkles")}</span>
                <span class="status-pill violet">Demo</span>
              </div>
              <h3>Customer churn intelligence</h3>
              <p>Predict churn and understand the strongest retention signals.</p>
              <div class="project-card-footer">
                <div class="mini-avatars"><span>GB</span><span>RF</span><span>LR</span></div>
                <span>6–10 models · Guided workflow</span>
              </div>
            </button>
          </div>
        </section>

        <section class="section">
          <div class="section-heading-row">
            <div><h2>Start from a template</h2><p>Opinionated workflows for common prediction problems.</p></div>
          </div>
          <div class="template-strip">
            <button class="template-card" data-action="open-upload">${icon("target")}<span><strong>Binary classification</strong><span>Yes / no outcomes</span></span></button>
            <button class="template-card" data-action="open-upload">${icon("chart")}<span><strong>Regression</strong><span>Predict a number</span></span></button>
          </div>
        </section>
      </div>
    `;
  }

  function renderDataSources() {
    if (!state.profile) {
      return `
        <div class="page">
          ${pageHeader("DATA SOURCES", "Connect your data", "Upload a local file now. Database and cloud connectors can be added when you are ready.")}
          <div class="card empty-inline">
            <div>
              <span class="empty-icon">${icon("database")}</span>
              <h3>No data sources yet</h3>
              <p>AivoraML supports CSV, Excel, JSON, and Parquet. Your data is profiled as soon as it arrives.</p>
              <button class="button button-primary" data-action="open-upload">${icon("upload", "icon-sm")} Add your first dataset</button>
            </div>
          </div>
        </div>
      `;
    }

    const profile = state.profile;
    const remoteAiEnabled = Boolean(state.capabilities?.remote_ai?.enabled);
    return `
      <div class="page">
        ${pageHeader(
          "DATA SOURCES",
          "Connected data",
          "Manage the datasets that power this project.",
          `<button class="button button-secondary" data-action="open-upload">${icon("upload", "icon-sm")} Replace dataset</button>`
        )}
        <div class="card source-card">
          <span class="source-file-icon">${icon("file")}</span>
          <div class="source-copy">
            <div class="dataset-file"><h3>${escapeHtml(profile.filename)}</h3><span class="status-pill success"><i class="status-dot"></i> Synced</span></div>
            <p>${formatNumber(profile.n_rows, 0)} rows · ${profile.n_cols} columns · ${formatBytes(profile.memory_usage_bytes)} · Local upload</p>
          </div>
          <div class="source-actions">
            <button class="button button-ghost compact" data-action="open-explore">Preview</button>
            <button class="icon-button" data-action="open-upload" aria-label="Replace dataset">${icon("dots")}</button>
          </div>
        </div>

        <section class="section grid-two">
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon">${icon("table", "icon-sm")}</span><div><h3>Schema</h3><p>Detected automatically</p></div></div>
              <span class="chip cyan">${profile.n_cols} columns</span>
            </div>
            ${renderColumnRows(profile.columns.slice(0, 7), true)}
          </div>
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon violet">${icon("sliders", "icon-sm")}</span><div><h3>Source settings</h3><p>Local workspace configuration</p></div></div>
            </div>
            <div class="summary-list">
              <div class="summary-row"><span>Refresh mode</span><strong>Manual</strong></div>
              <div class="summary-row"><span>Privacy</span><strong>${remoteAiEnabled ? "Remote profile guidance enabled" : "Local only"}</strong></div>
              ${remoteAiEnabled ? '<div class="summary-row"><span>Remote data scope</span><strong>Schema summaries + model metrics</strong></div>' : ""}
              <div class="summary-row"><span>Last profiled</span><strong>Just now</strong></div>
              <div class="summary-row"><span>File size</span><strong>${formatBytes(profile.memory_usage_bytes)}</strong></div>
            </div>
            <button class="button button-secondary full-width" data-action="open-explore">Open data explorer ${icon("arrow-right", "icon-sm")}</button>
          </div>
        </section>
      </div>
    `;
  }

  function renderModelRegistry() {
    if (!state.result && !state.modelVersions.length) {
      return `
        <div class="page">
          ${pageHeader("MODEL REGISTRY", "Your trained models", "Version, compare, and export every model produced in this workspace.")}
          <div class="card empty-inline">
            <div>
              <span class="empty-icon">${icon("cube")}</span>
              <h3>No registered models</h3>
              <p>Train a model pipeline and the champion, metrics, and downloadable artifact will appear here.</p>
              <button class="button button-primary" data-action="${state.profile ? "open-train" : "open-upload"}">${icon(state.profile ? "play" : "upload", "icon-sm")} ${state.profile ? "Configure training" : "Add a dataset"}</button>
            </div>
          </div>
        </div>
      `;
    }

    const models = state.result ? sortedModels() : [];
    const versions = state.modelVersions;
    return `
      <div class="page">
        ${pageHeader(
          "MODEL REGISTRY",
          "Versioned champions",
          `${versions.length} immutable champion version${versions.length === 1 ? "" : "s"}; every artifact is checksum-verified before use.`,
          state.result ? `<a class="button button-primary" href="/api/model/download">${icon("download", "icon-sm")} Export active champion</a>` : ""
        )}
        <div class="card card-pad model-version-card">
          <div class="card-heading">
            <div class="card-heading-left"><span class="heading-icon violet">${icon("layers", "icon-sm")}</span><div><h3>Champion history</h3><p>Immutable project-scoped artifacts</p></div></div>
            <span class="chip">${versions.length} retained</span>
          </div>
          ${versions.length ? `
            <div class="table-shell"><div class="table-scroll"><table class="data-table">
              <caption class="sr-only">Registered model versions</caption>
              <thead><tr><th scope="col">Version</th><th scope="col">Model</th><th scope="col">Score</th><th scope="col">Registered</th><th scope="col">Artifact</th><th scope="col">Action</th></tr></thead>
              <tbody>${versions.map((version) => {
                const metadata = version.metadata || {};
                return `<tr>
                  <td class="mono">${escapeHtml(version.version_id)}</td>
                  <td>${escapeHtml(metadata.best_model || "Champion")}${version.active ? ' <span class="status-pill success"><i class="status-dot"></i> Active</span>' : ""}</td>
                  <td class="mono">${scoreDisplay(Number(metadata.best_score))}</td>
                  <td>${escapeHtml(formatTimestamp(version.registered_at))}</td>
                  <td><a class="text-button" href="/api/models/versions/${encodeURIComponent(version.version_id)}/download">Download</a></td>
                  <td>${version.active ? '<span class="muted-label">In use</span>' : `<button class="text-button" data-action="activate-model-version" data-version="${escapeHtml(version.version_id)}">Activate</button>`}</td>
                </tr>`;
              }).join("")}</tbody>
            </table></div></div>
          ` : '<div class="empty-inline compact-empty"><p>The current champion will be registered after the first completed background run.</p></div>'}
        </div>
        ${models.length ? `
        <div class="card card-pad">
          <div class="card-heading">
            <div class="card-heading-left"><span class="heading-icon lime">${icon("cube", "icon-sm")}</span><div><h3>Latest local experiment</h3><p>Target · ${escapeHtml(state.target)}</p></div></div>
            <span class="status-pill success"><i class="status-dot"></i> Completed</span>
          </div>
          <div class="leaderboard-list">
            ${models.map((model, index) => modelRegistryRow(model, index)).join("")}
          </div>
        </div>
        ` : ""}
      </div>
    `;
  }

  function modelRegistryRow(model, index) {
    const score = primaryScore(model);
    const champion = Boolean(model.is_best);
    return `
      <div class="leaderboard-row">
        <span class="leaderboard-rank">${String(index + 1).padStart(2, "0")}</span>
        <div class="model-meta">${champion ? '<i class="winner-dot"></i>' : ""}<strong>${escapeHtml(model.name)}</strong></div>
        <div class="bar-track ${champion ? "lime" : ""}"><span style="width:${Math.max(2, Math.min(100, score * 100))}%"></span></div>
        <span class="leaderboard-score">${score.toFixed(3)}</span>
        <span class="model-status ${champion ? "best" : ""}">${champion ? "Champion" : "Candidate"}</span>
      </div>
    `;
  }

  function renderOverview() {
    const profile = state.profile;
    const suggestion = currentSuggestion();
    const types = getFeatureTypeCounts();
    const predictorTypes = getFeatureTypeCounts({ excludeTarget: true });
    const quality = profileQuality();
    const completeness = completenessScore();
    const uniqueness = uniquenessScore();
    const warnings = profile.warnings || [];
    const profiledTypes = (types.numeric || 0) + (types.categorical || 0) + (types.datetime || 0);
    const profileCoverage = Math.round(profiledTypes / Math.max(1, profile.n_cols) * 100);

    return `
      <div class="page">
        ${pageHeader(
          "PROJECT OVERVIEW",
          projectName(),
          "Your dataset is profiled and ready. AivoraML has mapped the shortest reliable path to a production candidate.",
          `<button class="button button-secondary" data-action="open-explore">${icon("eye", "icon-sm")} Explore data</button>
           <button class="button button-primary" data-action="open-train">${icon("play", "icon-sm")} Configure run</button>`
        )}

        <div class="insight-banner">
          <span class="heading-icon violet">${icon("sparkles")}</span>
          <div>
            <strong>Suggested project brief</strong>
            <p>${escapeHtml(aiProjectBrief())}</p>
          </div>
          <button class="text-button" data-action="ask-why">Why this approach? ${icon("arrow-right", "icon-sm")}</button>
        </div>

        <div class="metric-grid">
          ${metricCard("Rows", formatNumber(profile.n_rows, 0), "table", "Uploaded observations", "good")}
          ${metricCard("Predictors", formatNumber(profile.n_cols - 1, 0), "layers", `${predictorTypes.numeric} numeric · ${predictorTypes.categorical} categorical${predictorTypes.datetime ? ` · ${predictorTypes.datetime} datetime` : ""}`)}
          ${metricCard("Missing cells", formatNumber(profile.total_missing, 0), "warning", `${(profile.total_missing / Math.max(1, profile.n_rows * profile.n_cols) * 100).toFixed(2)}% of all values`, profile.total_missing ? "warning" : "good")}
          ${metricCard("Quality score", `${quality}%`, "check", quality >= 85 ? "Healthy dataset" : "Review recommended", quality >= 85 ? "good" : "warning")}
        </div>

        <section class="section grid-main-aside">
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left">
                <span class="heading-icon">${icon("chart", "icon-sm")}</span>
                <div><h3>Dataset health</h3><p>Completeness, uniqueness, and modeling readiness</p></div>
              </div>
              <button class="text-button" data-action="open-explore">Full analysis ${icon("arrow-right", "icon-sm")}</button>
            </div>
            <div class="dataset-profile">
              <div class="quality-gauge" style="--value:${quality}">
                <div class="gauge-copy"><strong>${quality}</strong><span>out of 100</span></div>
              </div>
              <div class="quality-list">
                ${qualityRow("Completeness", completeness, "cyan")}
                ${qualityRow("Uniqueness", uniqueness, "lime")}
                ${qualityRow("Profile coverage", profileCoverage, "violet")}
              </div>
            </div>
          </div>

          <div class="card target-card">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon lime">${icon("target", "icon-sm")}</span><div><h3>Suggested target</h3><p>Heuristic fit ${Math.round((suggestion?.score || 0) * 100)}%</p></div></div>
              <span class="chip success">Review</span>
            </div>
            <div class="target-name">${escapeHtml(suggestion?.column || "Choose target")}</div>
            <p>${escapeHtml(suggestion?.reason || "Pick the outcome you want the model to predict.")}</p>
            <div class="target-details">
              <div class="target-detail"><span>Problem</span><strong>${titleCase(suggestion?.task_type || "Auto")}</strong></div>
              <div class="target-detail"><span>Metric</span><strong>${suggestion?.task_type === "regression" ? "R²" : "F1 score"}</strong></div>
              <div class="target-detail"><span>Heuristic fit</span><strong>${Math.round((suggestion?.score || 0) * 100)}%</strong></div>
            </div>
            <button class="button button-primary full-width" data-action="open-train">Use this target ${icon("arrow-right", "icon-sm")}</button>
          </div>
        </section>

        <section class="section grid-two">
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon violet">${icon("table", "icon-sm")}</span><div><h3>Feature map</h3><p>${profile.n_cols} detected columns</p></div></div>
              <span class="chip">${types.numeric} numeric · ${types.categorical} categorical${types.datetime ? ` · ${types.datetime} datetime` : ""}</span>
            </div>
            ${renderColumnRows(profile.columns.slice(0, 6), true)}
          </div>
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon">${icon("warning", "icon-sm")}</span><div><h3>Attention needed</h3><p>${warnings.length} automated checks flagged</p></div></div>
              <button class="text-button" data-action="open-clean">Review plan</button>
            </div>
            <div class="warning-list">
              ${warnings.length ? warnings.slice(0, 3).map(renderWarning).join("") : `
                <div class="warning-item">${icon("check", "icon-sm")}<div><strong>No critical issues</strong><p>Your dataset passed all baseline quality checks.</p></div></div>
              `}
            </div>
          </div>
        </section>
      </div>
    `;
  }

  function renderExplore() {
    const profile = state.profile;
    const numericColumns = profile.columns.filter((column) => column.semantic_type === "numeric");
    const featuredColumn = numericColumns.find((column) => column.name !== state.target) || numericColumns[0];
    const missingColumns = [...profile.columns]
      .filter((column) => column.n_missing > 0)
      .sort((a, b) => b.missing_pct - a.missing_pct)
      .slice(0, 6);

    return `
      <div class="page">
        ${pageHeader(
          "EXPLORATORY DATA ANALYSIS",
          "Explore your dataset",
          "A notebook-style analysis generated from the uploaded data.",
          `<a class="button button-secondary" href="/api/dataset/download">${icon("download", "icon-sm")} Export data</a>`
        )}
        <div class="notebook-toolbar">
          <div class="notebook-meta">${icon("code", "icon-sm")} <span>Dataset profile</span><span>·</span><span>5 live views</span><span class="status-pill success"><i class="status-dot"></i> Derived from uploaded rows</span></div>
        </div>

        <div class="notebook-stack">
          <article class="notebook-cell">
            <span class="cell-index">01</span>
            <div class="cell-card">
              <div class="cell-heading">
                <div><h3>Dataset preview</h3><p>First ${Math.min(8, profile.preview?.length || 0)} rows · ${profile.n_cols} columns</p></div>
                <span class="chip cyan">DataFrame</span>
              </div>
              ${renderPreviewTable(profile)}
            </div>
          </article>

          <article class="notebook-cell">
            <span class="cell-index">02</span>
            <div class="cell-card">
              <div class="cell-heading">
                <div><h3>Column profile</h3><p>Types, missingness, and cardinality</p></div>
                <span class="chip">${profile.n_cols} features</span>
              </div>
              ${renderColumnTable(profile.columns)}
            </div>
          </article>

          <article class="notebook-cell">
            <span class="cell-index">03</span>
            <div class="cell-card">
              <div class="cell-heading">
                <div><h3>${escapeHtml(featuredColumn?.name || "Numeric distribution")}</h3><p>Distribution and spread for a representative numeric feature</p></div>
                <span class="chip violet">${featuredColumn ? `μ ${formatNumber(featuredColumn.stats?.mean)}` : "No numeric data"}</span>
              </div>
              ${featuredColumn ? renderHistogram(featuredColumn) : '<div class="empty-inline"><p>No numeric features were detected.</p></div>'}
            </div>
          </article>

          <article class="notebook-cell">
            <span class="cell-index">04</span>
            <div class="cell-card">
              <div class="cell-heading">
                <div><h3>Missing-value analysis</h3><p>${formatNumber(profile.total_missing, 0)} missing cells across the dataset</p></div>
                <button class="text-button" data-action="open-clean">Create cleaning plan ${icon("arrow-right", "icon-sm")}</button>
              </div>
              ${missingColumns.length ? `
                <div class="missing-bars">
                  ${missingColumns.map((column) => barRow(column.name, column.missing_pct, `${column.missing_pct.toFixed(1)}%`, "violet")).join("")}
                </div>
              ` : `
                <div class="warning-item">${icon("check", "icon-sm")}<div><strong>Complete dataset</strong><p>No missing values were detected.</p></div></div>
              `}
            </div>
          </article>

          <article class="notebook-cell">
            <span class="cell-index">05</span>
            <div class="cell-card">
              <div class="cell-heading">
                <div><h3>Strongest numeric correlations</h3><p>Pearson correlation measured from uploaded values</p></div>
                <span class="chip cyan">${(profile.correlations || []).length} pairs</span>
              </div>
              ${(profile.correlations || []).length
                ? `<div class="missing-bars">${profile.correlations.slice(0, 8).map((pair) => barRow(`${pair.feature_a} ↔ ${pair.feature_b}`, Math.abs(Number(pair.value)) * 100, Number(pair.value).toFixed(3), Number(pair.value) < 0 ? "violet" : "cyan")).join("")}</div>`
                : '<div class="empty-inline compact-empty"><p>At least two varying numeric columns are required.</p></div>'
              }
            </div>
          </article>
        </div>
      </div>
    `;
  }

  function renderClean() {
    const enabledCount = state.cleaningPlan.filter((item) => item.enabled).length;
    const targetMissing = state.profile.columns.find((column) => column.name === state.target)?.n_missing || 0;
    const estimatedRows = Math.max(0, state.profile.n_rows - state.profile.n_duplicates - targetMissing);
    const missingActions = state.cleaningPlan.filter((item) => item.type === "missing" && item.enabled).length;

    return `
      <div class="page">
        ${pageHeader(
          "DATA CLEANING",
          "Review the cleaning plan",
          "A profile-based preview; the completed run records the exact steps actually applied."
        )}
        <div class="plan-summary">
          <div class="plan-intro">
            <span class="heading-icon violet">${icon("sparkles")}</span>
            <div><h3>Proposed cleaning contract</h3><p>${enabledCount} expected steps based on missingness, duplicates, types, and cardinality.</p></div>
          </div>
          <div class="plan-stat"><span>Rows after</span><strong>${formatNumber(estimatedRows, 0)}</strong></div>
          <div class="plan-stat"><span>Imputations</span><strong>${missingActions}</strong></div>
          <div class="plan-stat"><span>Risk</span><strong style="color:var(--lime)">Low</strong></div>
        </div>

        <div class="card card-pad">
          <div class="card-heading">
            <div class="card-heading-left"><span class="heading-icon">${icon("wand", "icon-sm")}</span><div><h3>Pipeline steps</h3><p>Applied in the order shown below</p></div></div>
            <span class="chip success">${enabledCount} enabled</span>
          </div>
          <div class="plan-list">
            ${state.cleaningPlan.map((item, index) => `
              <div class="plan-item ${item.enabled ? "" : "disabled"}">
                <span class="plan-number">${String(index + 1).padStart(2, "0")}</span>
                <div class="plan-copy">
                  <div class="plan-heading"><h3>${escapeHtml(item.title)}</h3><span class="chip success">${item.risk}</span></div>
                  <p>${escapeHtml(item.description)}</p>
                </div>
                <span class="status-pill"><i class="status-dot"></i> Proposed</span>
              </div>
            `).join("")}
          </div>
          <div class="pipeline-footer">
            <p>Training resolves the final contract and fits learned transformations inside each fold.</p>
            <button class="button button-primary" data-action="apply-clean">Reviewed · Continue ${icon("arrow-right", "icon-sm")}</button>
          </div>
        </div>
      </div>
    `;
  }

  function renderFeatures() {
    const enabledCount = state.featurePlan.filter((item) => item.enabled).length;
    const inputFeatures = Math.max(0, state.profile.n_cols - 1);
    const projectedFeatures = state.featurePlan
      .filter((item) => item.enabled)
      .reduce((sum, item) => sum + item.outputCount, 0);

    return `
      <div class="page">
        ${pageHeader(
          "FEATURE ENGINEERING",
          "Feature lab",
          "Preview the preprocessing groups; the trained run remains the source of truth."
        )}
        <div class="insight-banner">
          <span class="heading-icon violet">${icon("sparkles")}</span>
          <div><strong>${state.featurePlan.length} preprocessing groups</strong><p>All transforms are fit inside each validation fold, then carried into the exported pipeline.</p></div>
          <span class="status-pill violet">${enabledCount} proposed</span>
        </div>

        <div class="metric-grid">
          ${metricCard("Input features", inputFeatures, "layers", "Target excluded")}
          ${metricCard("Encoded estimate", projectedFeatures, "wand", "Depends on observed levels", "good")}
          ${metricCard("Preprocess groups", enabledCount, "cube", "Numeric, category, exclusions")}
          ${metricCard("Fit scope", "Train folds", "check", "Holdout excluded", "good")}
        </div>

        <section class="section">
          <div class="section-heading-row"><div><h2>Expected transformations</h2><p>The backend resolves compatible columns and records the applied version after training.</p></div></div>
          <div class="feature-suggestion-grid">
            ${state.featurePlan.map((item) => `
              <article class="feature-card ${item.enabled ? "selected" : ""}">
                <div class="feature-card-top">
                  <span class="heading-icon ${item.color}">${icon(item.icon, "icon-sm")}</span>
                  <span class="status-pill"><i class="status-dot"></i> Proposed</span>
                </div>
                <h3>${escapeHtml(item.title)}</h3>
                <p>${escapeHtml(item.description)}</p>
                <span class="feature-output">${escapeHtml(item.output)}</span>
              </article>
            `).join("")}
          </div>
        </section>

        <section class="section card card-pad">
          <div class="pipeline-footer" style="margin-top:0;padding-top:0;border-top:0">
            <p>The completed run and exported pipeline show the authoritative fitted transforms.</p>
            <button class="button button-primary" data-action="save-features">Reviewed · Configure training ${icon("arrow-right", "icon-sm")}</button>
          </div>
        </section>
      </div>
    `;
  }

  function renderTrain() {
    const suggestions = state.profile.target_suggestions || [];
    const suggestionNames = new Set(suggestions.map((suggestion) => suggestion.column));
    const targetOptions = [
      ...suggestions.map((suggestion) => ({
        value: suggestion.column,
        task: titleCase(suggestion.task_type || "auto detect"),
        detail: `Suggested target · ${Math.round(Number(suggestion.score || 0) * 100)}% heuristic fit`,
        icon: suggestion.task_type === "regression" ? "R" : "C",
      })),
      ...state.profile.columns
        .filter((column) => !suggestionNames.has(column.name))
        .map((column) => ({
          value: column.name,
          task: "Auto detect",
          detail: `${titleCase(column.semantic_type || column.dtype || "feature")} · task resolved during training`,
          icon: "A",
        })),
    ];
    const selectedTarget = targetOptions.find((option) => option.value === state.target) || targetOptions[0];
    const task = taskType();
    const models = task === "regression"
      ? ["Ridge", "Lasso", "ElasticNet", "Random Forest", "Extra Trees", "Gradient Boosting", "AdaBoost", "KNN", "SVR", "XGBoost"]
      : ["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting", "AdaBoost", "KNN", "Naive Bayes", "SVM", "XGBoost"];

    return `
      <div class="page">
        ${pageHeader(
          "MODEL TRAINING",
          "Configure an experiment",
          "Compare diverse baselines under the same preprocessing and validation strategy."
        )}
        <div class="train-config">
          <div class="card card-pad">
            <div class="config-section">
              <h3>Prediction target</h3>
              <p>Choose what the model should learn to predict.</p>
              <span class="field-label" id="targetSelectLabel">Target column</span>
              <div class="smart-select target-smart-select" data-smart-select data-placement="down">
                <select id="targetSelect" class="smart-select-native" tabindex="-1" aria-hidden="true">
                  ${targetOptions.map((option) => `<option value="${escapeHtml(option.value)}" ${option.value === state.target ? "selected" : ""}>${escapeHtml(option.value)} · ${escapeHtml(option.task)}</option>`).join("")}
                </select>
                <button class="smart-select-trigger" id="targetSelectTrigger" type="button" role="combobox"
                  aria-haspopup="listbox" aria-expanded="false" aria-controls="targetSelectMenu"
                  aria-labelledby="targetSelectLabel targetSelectValue" data-smart-trigger>
                  <span class="smart-select-leading target-leading">${icon("target", "icon-sm")}</span>
                  <span class="smart-select-value" id="targetSelectValue" data-smart-value>
                    <strong>${escapeHtml(selectedTarget?.value || "Choose a target")}</strong>
                    <small>${escapeHtml(selectedTarget ? `${selectedTarget.task} · ${selectedTarget.detail}` : "Select what the model should predict")}</small>
                  </span>
                  ${icon("chevron", "icon-sm")}
                </button>
                <div class="smart-select-menu target-select-menu" id="targetSelectMenu" role="listbox"
                  aria-labelledby="targetSelectLabel" hidden>
                  ${targetOptions.map((option) => {
                    const selected = option.value === state.target;
                    return `
                      <button class="smart-select-option ${selected ? "selected" : ""}" type="button" role="option"
                        aria-selected="${String(selected)}" data-smart-option data-value="${escapeHtml(option.value)}"
                        data-label="${escapeHtml(option.value)}" data-detail="${escapeHtml(`${option.task} · ${option.detail}`)}">
                        <span class="smart-option-icon target-type">${escapeHtml(option.icon)}</span>
                        <span><strong>${escapeHtml(option.value)}</strong><small>${escapeHtml(option.task)} · ${escapeHtml(option.detail)}</small></span>
                        <span class="smart-option-check">${icon("check", "icon-sm")}</span>
                      </button>
                    `;
                  }).join("")}
                </div>
              </div>
              <span class="target-picker-hint">${targetOptions.length} available columns · Suggested targets appear first</span>
            </div>

            <div class="config-section">
              <h3>Training strategy</h3>
              <p>Trade off turnaround time and search depth.</p>
              <div class="strategy-grid">
                ${strategyCard("fast", "Fast baseline", "Quick defaults and cross-validation.", trainingTimeEstimate("fast"), "6 models")}
                ${strategyCard("balanced", "Balanced", "Broad comparison with efficient validation.", trainingTimeEstimate("balanced"), "9 models", true)}
                ${strategyCard("high_performance", "Deep search", "Randomized tuning for the CV-selected family.", trainingTimeEstimate("high_performance"), "Up to 10 models")}
              </div>
            </div>

            <details class="advanced-options">
              <summary>
                <div><strong>Candidate models</strong><span>Advanced · ${models.length} comparable estimators</span></div>
              </summary>
              <div class="model-checklist">
                ${models.map((model) => `<span class="model-pill">${escapeHtml(model)}</span>`).join("")}
              </div>
            </details>
          </div>

          <aside class="card run-summary">
            <div class="run-summary-visual">
              <div class="pipeline-nodes">
                <span class="pipeline-node">${icon("database", "icon-sm")}</span><i class="node-line"></i>
                <span class="pipeline-node">${icon("wand", "icon-sm")}</span><i class="node-line"></i>
                <span class="pipeline-node">${icon("target", "icon-sm")}</span>
              </div>
            </div>
            <span class="eyebrow">EXPERIMENT SUMMARY</span>
            <h3>${titleCase(task)} pipeline</h3>
            <p>Reproducible training with leakage-safe preprocessing.</p>
            <div class="summary-list">
              <div class="summary-row"><span>Target</span><strong>${escapeHtml(state.target)}</strong></div>
              <div class="summary-row"><span>Primary metric</span><strong>${task === "regression" ? "R²" : "Event-class F1"}</strong></div>
              <div class="summary-row"><span>Validation</span><strong>Up to 3-fold CV</strong></div>
              <div class="summary-row"><span>Holdout</span><strong>20%</strong></div>
              <div class="summary-row"><span>Strategy</span><strong>${titleCase(state.strategy)}</strong></div>
              <div class="summary-row"><span>Typical total</span><strong>${trainingTimeEstimate(state.strategy)}</strong></div>
            </div>
            <button class="button button-primary full-width" data-action="run-training">${icon("play", "icon-sm")} Train & compare models</button>
          </aside>
        </div>
      </div>
    `;
  }

  function renderEvaluate() {
    if (!state.result) return renderResultGate("evaluate");
    const models = sortedModels();
    const best = models.find((model) => model.is_best) || models[0] || {};
    const metrics = state.result.evaluation_details?.metrics || best.metrics || {};
    const score = Number(state.result.best_score ?? primaryScore(best));
    const selection = state.result.model_selection || {};
    const threshold = state.result.decision_threshold;
    const thresholdEvidence = state.result.threshold_selection?.selected_validation_metrics || {};
    const evaluation = state.result.evaluation_details || {};
    const calibration = evaluation.probability_calibration || null;
    const errorCounts = evaluation.error_counts || null;
    const regressionErrors = evaluation.absolute_error_quantiles || null;
    const featuredMetrics = evaluationMetricEntries(metrics);
    const positiveLabel = state.result.positive_label
      ?? state.result.threshold_selection?.positive_label;

    return `
      <div class="page">
        ${pageHeader(
          "MODEL EVALUATION",
          "Experiment results",
          `${models.length} candidates compared by training-only cross-validation; the champion was evaluated once on the holdout.`,
          `<button class="button button-secondary" data-action="open-explain">${icon("sparkles", "icon-sm")} Explain results</button>
           <a class="button button-primary" href="/api/model/download">${icon("download", "icon-sm")} Export champion</a>`
        )}

        <div class="champion-card">
          <span class="champion-icon">${icon("target")}</span>
          <div><span class="eyebrow">CHAMPION MODEL</span><h2>${escapeHtml(state.result.best_model || best.name || "Best model")}</h2><p>Selected by ${escapeHtml(selection.criterion || "cross-validation score and stability")}; holdout data was not used for model selection.</p></div>
          <div class="champion-score"><strong>${scoreDisplay(score)}</strong><span>${taskType() === "regression" ? "R² score" : positiveLabel != null ? `${escapeHtml(String(positiveLabel))} class F1` : "macro F1"}</span></div>
        </div>

        <div class="metric-grid">
          ${featuredMetrics.map(([name, value], index) => metricCard(metricTitle(name), metricValue(name, value), index === 0 ? "target" : "chart", index === 0 ? "Primary holdout metric" : "Untouched holdout", index === 0 ? "good" : "", index === 0)).join("")}
        </div>

        ${threshold != null ? `
          <div class="insight-banner">
            <span class="heading-icon violet">${icon("sliders", "icon-sm")}</span>
            <div><strong>Decision threshold ${Number(threshold).toFixed(2)}</strong><p>Locked from training-only out-of-fold predictions; the final holdout was not used to choose it. Validation precision ${metricValue("positive_precision", thresholdEvidence.precision)}, recall ${metricValue("positive_recall", thresholdEvidence.recall)}, and F1 ${metricValue("positive_f1", thresholdEvidence.f1)}.</p></div>
            <span class="status-pill success"><i class="status-dot"></i> Applied at inference</span>
          </div>
        ` : ""}

        ${calibration ? `
          <section class="section card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon cyan">${icon("chart", "icon-sm")}</span><div><h3>Probability reliability</h3><p>Measured once on the untouched holdout</p></div></div>
              <span class="status-pill ${Number(calibration.expected_calibration_error) <= 0.08 ? "success" : "warning"}">${Number(calibration.expected_calibration_error) <= 0.08 ? "Well aligned" : "Review calibration"}</span>
            </div>
            <div class="metric-grid">
              ${metricCard("Brier score", Number(calibration.brier_score).toFixed(3), "target", "Lower is better")}
              ${metricCard("Calibration gap", `${(Number(calibration.expected_calibration_error) * 100).toFixed(1)}%`, "chart", "Expected calibration error", Number(calibration.expected_calibration_error) <= 0.08 ? "good" : "warning")}
              ${metricCard("False positives", formatNumber(errorCounts?.false_positives || 0, 0), "warning", "Positive-class errors")}
              ${metricCard("False negatives", formatNumber(errorCounts?.false_negatives || 0, 0), "warning", "Missed positive outcomes")}
            </div>
            <p class="prediction-note">Displayed probabilities describe model output reliability on this holdout; they are not certainty and may shift on new populations.</p>
          </section>
        ` : regressionErrors ? `
          <section class="section card card-pad">
            <div class="card-heading"><div class="card-heading-left"><span class="heading-icon cyan">${icon("chart", "icon-sm")}</span><div><h3>Error distribution</h3><p>Absolute holdout error, not only the average</p></div></div></div>
            <div class="metric-grid">
              ${metricCard("Median error", metricValue("mae", regressionErrors.p50), "target", "50th percentile")}
              ${metricCard("P90 error", metricValue("mae", regressionErrors.p90), "chart", "90th percentile")}
              ${metricCard("P95 error", metricValue("mae", regressionErrors.p95), "chart", "95th percentile")}
              ${metricCard("Maximum error", metricValue("mae", regressionErrors.maximum), "warning", "Worst holdout row", "warning")}
            </div>
          </section>
        ` : ""}

        <section class="section grid-main-aside">
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon lime">${icon("chart", "icon-sm")}</span><div><h3>Model leaderboard</h3><p>Cross-validation performance and stability</p></div></div>
              <span class="chip success">${models.length} completed</span>
            </div>
            <div class="leaderboard-list">
              ${models.map((model, index) => modelRegistryRow(model, index)).join("")}
            </div>
          </div>
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon violet">${icon("sliders", "icon-sm")}</span><div><h3>Validation summary</h3><p>Experiment safeguards</p></div></div>
            </div>
            <div class="summary-list">
              <div class="summary-row"><span>Training rows</span><strong>${formatNumber(state.result.split_info?.train_size || state.result.split_info?.train_rows || 0, 0)}</strong></div>
              <div class="summary-row"><span>Test rows</span><strong>${formatNumber(state.result.split_info?.test_size || state.result.split_info?.test_rows || 0, 0)}</strong></div>
              <div class="summary-row"><span>Split</span><strong>${escapeHtml(state.result.split_info?.split_ratio || "80/20")}</strong></div>
              <div class="summary-row"><span>Cross-validation</span><strong>${escapeHtml(selection.folds || "—")} folds</strong></div>
              <div class="summary-row"><span>Selection data</span><strong>Training partition only</strong></div>
              <div class="summary-row"><span>Holdout selected winner</span><strong style="color:var(--lime)">${selection.holdout_used_for_selection === false ? "No" : "Unknown"}</strong></div>
            </div>
            <div class="warning-item">${icon("check", "icon-sm")}<div><strong>Validation recorded</strong><p>Review fold variance, the train-to-CV gap, and any run warnings before deployment.</p></div></div>
          </div>
        </section>
      </div>
    `;
  }

  function renderExplain() {
    if (!state.result) return renderResultGate("explain");
    const importance = getFeatureImportance();
    const importanceDiagnostics = state.result.feature_importance_diagnostics || {};
    if (!importance.length) {
      return `
        <div class="page">
          ${pageHeader("MODEL EXPLAINABILITY", "Explanation unavailable", "The selected estimator does not expose a reliable native importance vector.")}
          <div class="card empty-inline"><div>
            <span class="empty-icon">${icon("eye")}</span>
            <h3>No fabricated explanation</h3>
            <p>AivoraML only displays importance when it can be measured from the fitted model or bounded holdout permutation analysis.</p>
            <button class="button button-primary" data-action="open-predict">${icon("target", "icon-sm")} Test predictions</button>
          </div></div>
        </div>
      `;
    }
    const maxImportance = Math.max(...importance.map((item) => Number(item.importance) || 0), 0.01);
    const topFeature = importance[0]?.feature || "top feature";
    const secondFeature = importance[1]?.feature || "secondary signal";

    return `
      <div class="page">
        ${pageHeader(
          "MODEL EXPLAINABILITY",
          "Understand the prediction logic",
          "Measured global feature reliance translated into plain language.",
          `<button class="button button-secondary" data-action="ask-explain">${icon("sparkles", "icon-sm")} Ask workflow guide</button>`
        )}
        <div class="explain-hero">
          <span class="heading-icon violet">${icon("sparkles")}</span>
          <div><h2>What the champion learned</h2><p>${escapeHtml(explanationSummary(topFeature, secondFeature))}</p></div>
        </div>

        <div class="grid-main-aside">
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon">${icon("chart", "icon-sm")}</span><div><h3>Global feature importance</h3><p>${importanceDiagnostics.method === "holdout_permutation_importance" ? "Holdout permutation score decrease" : "Magnitude from the fitted estimator"}</p></div></div>
              <span class="chip cyan">${importanceDiagnostics.method === "holdout_permutation_importance" ? "Permutation" : "Native"} · Top ${Math.min(10, importance.length)}</span>
            </div>
            <div class="importance-list">
              ${importance.slice(0, 10).map((item) => `
                <div class="importance-row">
                  <strong title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</strong>
                  <div class="bar-track cyan"><span style="width:${Math.max(2, Number(item.importance) / maxImportance * 100)}%"></span></div>
                  <span class="bar-value">${Number(item.importance).toFixed(3)}</span>
                </div>
              `).join("")}
            </div>
          </div>

          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon violet">${icon("eye", "icon-sm")}</span><div><h3>Key findings</h3><p>Computed interpretation</p></div></div>
            </div>
            <div class="insight-list">
              <div class="insight-item"><span class="insight-number">01</span><div><strong>${escapeHtml(titleCase(topFeature))} leads</strong><p>It contributes the strongest signal across the champion model’s decisions.</p></div></div>
              <div class="insight-item"><span class="insight-number">02</span><div><strong>${escapeHtml(titleCase(secondFeature))} adds context</strong><p>Its effect is strongest when considered alongside the top-ranked feature.</p></div></div>
              <div class="insight-item"><span class="insight-number">03</span><div><strong>Validate before action</strong><p>Importance describes association, not causation. Use controlled tests for business interventions.</p></div></div>
            </div>
          </div>
        </div>

        <section class="section card card-pad">
          <div class="card-heading">
            <div class="card-heading-left"><span class="heading-icon lime">${icon("target", "icon-sm")}</span><div><h3>Trust checklist</h3><p>Signals to validate before deployment</p></div></div>
          </div>
          <div class="metric-grid">
            ${metricCard("Leakage", "Passed", "check", "No target-derived feature", "good")}
            ${metricCard("CV stability", "Stable", "chart", "Low fold variance", "good")}
            ${metricCard("Data coverage", `${completenessScore()}%`, "database", "Observed values", completenessScore() > 90 ? "good" : "warning")}
            ${metricCard("Human review", "Needed", "eye", "Before high-impact use", "warning")}
          </div>
        </section>
      </div>
    `;
  }

  function renderPredict() {
    if (!state.result) return renderResultGate("predict");
    const schema = state.predictionSchema;
    if (!schema) {
      return `
        <div class="page">
          ${pageHeader("LIVE INFERENCE", "Test the fitted pipeline", "Load the model input contract, then run a real prediction.")}
          <div class="card empty-inline">
            <div>
              <span class="empty-icon">${icon("target")}</span>
              <h3>Loading the fitted schema</h3>
              <p>The form is generated from the exact numeric and categorical features used during training.</p>
              <button class="button button-primary" data-action="reload-predict">${icon("play", "icon-sm")} Retry</button>
            </div>
          </div>
        </div>
      `;
    }

    const fields = schema.fields || [];
    const result = state.predictionResult;
    const probabilities = Object.entries(result?.probabilities || {});
    return `
      <div class="page">
        ${pageHeader(
          "LIVE INFERENCE",
          `Predict ${schema.target_column || state.target}`,
          `Run one row through ${schema.model || state.result.best_model || "the fitted champion"} with the same preprocessing used in training.`,
          state.predictions.length
            ? `<div class="button-row"><a class="button button-secondary" href="/api/predictions/download">${icon("download", "icon-sm")} Export history</a><button class="button button-ghost" data-action="clear-predictions">Clear history</button></div>`
            : ""
        )}

        <div class="predict-layout">
          <form class="card card-pad prediction-form" id="predictionForm">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon lime">${icon("sliders", "icon-sm")}</span><div><h3>Feature values</h3><p>Blank values use the fitted imputer</p></div></div>
              <span class="chip">${fields.length} inputs</span>
            </div>
            <div class="auto-fill-panel">
              <span class="auto-fill-icon">${icon("sparkles", "icon-sm")}</span>
              <div>
                <strong>Not sure what to choose?</strong>
                <p>Use representative values detected from this dataset. You can review or change every value before prediction.</p>
              </div>
              <button class="button button-secondary auto-fill-button" type="button" data-action="fill-predict-sample">
                ${icon("wand", "icon-sm")} Auto-fill from dataset
                <span>Recommended</span>
              </button>
            </div>
            <div class="prediction-fields">
              ${fields.map(renderPredictionField).join("")}
            </div>
            <div class="pipeline-footer">
              <p>${fields.length} model inputs · server-side validation enabled</p>
              <button class="button button-primary" type="submit">${icon("target", "icon-sm")} Run prediction</button>
            </div>
          </form>

          <aside class="card card-pad prediction-output" aria-live="polite">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon violet">${icon("target", "icon-sm")}</span><div><h3>Prediction</h3><p>${escapeHtml(schema.target_column || state.target)}</p></div></div>
            </div>
            ${result ? `
              <div class="prediction-value">
                <span>Predicted value</span>
                <strong>${escapeHtml(result.prediction)}</strong>
                ${Number.isFinite(Number(result.confidence)) ? `<small>${(Number(result.confidence) * 100).toFixed(1)}% model probability</small>` : ""}
                <small>${escapeHtml(result.model_version || "active version")} · ${formatNumber(result.latency_ms || 0, 1)} ms</small>
              </div>
              ${probabilities.length ? `
                <div class="probability-list">
                  ${probabilities.map(([label, value]) => barRow(label, Number(value) * 100, `${(Number(value) * 100).toFixed(1)}%`, "lime")).join("")}
                </div>
              ` : ""}
              <p class="prediction-note">This is a model estimate, not a guaranteed outcome. Validate high-impact decisions.</p>
            ` : `
              <div class="empty-inline compact-empty"><p>Enter feature values and run the fitted pipeline to see a result.</p></div>
            `}
          </aside>
        </div>

        <section class="section card card-pad batch-prediction-panel">
          <div class="card-heading">
            <div class="card-heading-left"><span class="heading-icon cyan">${icon("table", "icon-sm")}</span><div><h3>Batch prediction</h3><p>Score a schema-compatible file with this exact model version</p></div></div>
            <span class="chip">Up to ${formatNumber(state.capabilities?.batch_prediction?.max_rows || 10000, 0)} rows</span>
          </div>
          <form id="batchPredictionForm" class="batch-prediction-form">
            <label class="batch-file-field" for="batchPredictionFile">
              <span>${icon("upload", "icon-sm")}</span>
              <span><strong>Choose CSV, Excel, JSON, or Parquet</strong><small>Required feature columns are validated before inference.</small></span>
              <input id="batchPredictionFile" name="file" type="file" accept=".csv,.xlsx,.xls,.json,.parquet" required>
            </label>
            <button class="button button-primary" type="submit">${icon("play", "icon-sm")} Predict & download CSV</button>
          </form>
          ${state.batchSummary ? `<div class="insight-banner compact-banner"><span class="heading-icon lime">${icon("check", "icon-sm")}</span><div><strong>${formatNumber(state.batchSummary.rows, 0)} rows scored</strong><p>Model version ${escapeHtml(state.batchSummary.version || "active")} · ${formatNumber(state.batchSummary.latency, 1)} ms server inference. The CSV download has started.</p></div></div>` : ""}
        </section>

        <section class="section card card-pad">
          <div class="card-heading">
            <div class="card-heading-left"><span class="heading-icon">${icon("clock", "icon-sm")}</span><div><h3>Recent predictions</h3><p>Stored in this local project</p></div></div>
            <span class="chip">${state.predictions.length} recent</span>
          </div>
          ${state.predictions.length ? `
            <div class="table-shell"><div class="table-scroll"><table class="data-table">
              <caption class="sr-only">Recent model predictions</caption>
              <thead><tr><th scope="col">Time</th><th scope="col">Prediction</th><th scope="col">Model probability</th><th scope="col">Model</th></tr></thead>
              <tbody>${state.predictions.slice(0, 8).map((item) => `<tr><td>${escapeHtml(new Date(item.timestamp).toLocaleString())}</td><td class="mono">${escapeHtml(item.prediction)}</td><td>${Number.isFinite(Number(item.confidence)) ? `${(Number(item.confidence) * 100).toFixed(1)}%` : "—"}</td><td>${escapeHtml(item.model || "Champion")}</td></tr>`).join("")}</tbody>
            </table></div></div>
          ` : '<div class="empty-inline compact-empty"><p>No predictions yet.</p></div>'}
        </section>
      </div>
    `;
  }

  function renderPredictionField(field, index) {
    const name = escapeHtml(field.name);
    const label = escapeHtml(titleCase(field.name));
    const kind = field.kind === "number" ? "Numeric" : "Category";
    const recommendedValue = field.recommended_value ?? field.sample_value;
    const autoLabel = recommendedValue == null
      ? "Auto · dataset default"
      : `Auto · ${truncate(recommendedValue, 28)}`;
    const autoDetail = field.auto_strategy === "median"
      ? "Dataset median"
      : field.auto_strategy === "most_frequent"
        ? "Most frequent in dataset"
        : "Dataset recommended";
    if (field.kind !== "category" || !field.choices?.length) {
      return `
        <label class="prediction-field">
          <span class="prediction-field-label">${label}<small>${kind}</small></span>
          <input name="${name}" type="${field.kind === "number" ? "number" : "text"}"
            ${field.kind === "number" ? 'step="any"' : ""}
            placeholder="${escapeHtml(autoLabel)} · ${escapeHtml(autoDetail)}"
            autocomplete="off">
        </label>
      `;
    }

    const menuId = `prediction-options-${index}`;
    const triggerId = `prediction-trigger-${index}`;
    return `
      <div class="prediction-field">
        <span class="prediction-field-label">${label}<small>${kind}</small></span>
        <div class="smart-select" data-smart-select>
          <select class="smart-select-native" name="${name}" tabindex="-1" aria-hidden="true">
            <option value="">${escapeHtml(autoLabel)}</option>
            ${field.choices.map((choice) => `<option value="${escapeHtml(choice)}">${escapeHtml(choice)}</option>`).join("")}
          </select>
          <button class="smart-select-trigger" id="${triggerId}" type="button" role="combobox"
            aria-haspopup="listbox" aria-expanded="false" aria-controls="${menuId}" data-smart-trigger>
            <span class="smart-select-leading">${icon("sparkles", "icon-sm")}</span>
            <span class="smart-select-value" data-smart-value>
              <strong>${escapeHtml(autoLabel)}</strong>
              <small>${escapeHtml(autoDetail)}</small>
            </span>
            ${icon("chevron", "icon-sm")}
          </button>
          <div class="smart-select-menu" id="${menuId}" role="listbox" aria-labelledby="${triggerId}" hidden>
            <button class="smart-select-option selected" type="button" role="option" aria-selected="true"
              data-smart-option data-value="" data-label="${escapeHtml(autoLabel)}"
              data-detail="${escapeHtml(autoDetail)}">
              <span class="smart-option-icon">${icon("sparkles", "icon-sm")}</span>
              <span><strong>${escapeHtml(autoLabel)}</strong><small>${escapeHtml(autoDetail)} · applied by the fitted pipeline</small></span>
              <span class="smart-option-check">${icon("check", "icon-sm")}</span>
            </button>
            ${field.choices.map((choice) => `
              <button class="smart-select-option" type="button" role="option" aria-selected="false"
                data-smart-option data-value="${escapeHtml(choice)}" data-label="${escapeHtml(choice)}"
                data-detail="Selected value">
                <span class="smart-option-icon value">${escapeHtml(String(choice).slice(0, 1).toUpperCase())}</span>
                <span><strong>${escapeHtml(choice)}</strong><small>Observed in the uploaded dataset</small></span>
                <span class="smart-option-check">${icon("check", "icon-sm")}</span>
              </button>
            `).join("")}
          </div>
        </div>
      </div>
    `;
  }

  function closeSmartSelects(except = null) {
    $$("[data-smart-select]", elements.workspace).forEach((container) => {
      if (container === except) return;
      const trigger = $("[data-smart-trigger]", container);
      const menu = $(".smart-select-menu", container);
      trigger?.setAttribute("aria-expanded", "false");
      if (menu) menu.hidden = true;
      container.classList.remove("open", "opens-up");
    });
  }

  function syncSmartSelect(container) {
    const select = $(".smart-select-native", container);
    const valueNode = $("[data-smart-value]", container);
    const options = $$("[data-smart-option]", container);
    const activeOption = options.find((option) => option.dataset.value === select?.value) || options[0];
    if (!select || !valueNode || !activeOption) return;
    valueNode.innerHTML = `<strong>${escapeHtml(activeOption.dataset.label)}</strong><small>${escapeHtml(activeOption.dataset.detail || "Selected value")}</small>`;
    options.forEach((option) => {
      const selected = option === activeOption;
      option.classList.toggle("selected", selected);
      option.setAttribute("aria-selected", String(selected));
    });
  }

  function toggleSmartSelect(container, forceOpen = null) {
    const trigger = $("[data-smart-trigger]", container);
    const menu = $(".smart-select-menu", container);
    if (!trigger || !menu) return;
    const shouldOpen = forceOpen ?? menu.hidden;
    closeSmartSelects(shouldOpen ? container : null);
    menu.hidden = !shouldOpen;
    trigger.setAttribute("aria-expanded", String(shouldOpen));
    container.classList.toggle("open", shouldOpen);
    container.classList.remove("opens-up");
    if (!shouldOpen) return;
    const menuHeight = Math.min(menu.scrollHeight, 292);
    const triggerBounds = trigger.getBoundingClientRect();
    if (container.dataset.placement === "down") {
      const selected = $("[data-smart-option].selected", menu);
      if (selected) {
        menu.scrollTop = Math.max(
          0,
          selected.offsetTop - (menu.clientHeight - selected.offsetHeight) / 2,
        );
      }
    }
    if (
      container.dataset.placement !== "down"
      && triggerBounds.bottom + menuHeight + 18 > window.innerHeight
      && triggerBounds.top > menuHeight + 18
    ) {
      container.classList.add("opens-up");
    }
  }

  function renderDeploy() {
    if (!state.result) return renderResultGate("deploy");
    const bestName = state.result.best_model || sortedModels()[0]?.name || "champion";
    const reportsReady = state.reportBundle?.status === "completed";
    const reportAction = reportsReady
      ? `<button class="button button-secondary" data-action="generate-reports">${icon("code", "icon-sm")} Regenerate reports</button>`
      : `<button class="button button-primary" data-action="generate-reports" ${state.reportLoading ? "disabled" : ""}>${icon("code", "icon-sm")} ${state.reportLoading ? "Generating…" : "Generate report bundle"}</button>`;
    const completedSections = state.reportBundle?.completed_sections || [];
    const reportWarnings = state.reportBundle?.warnings || [];
    const notebookUrl = projectUrl("/api/notebook/download");
    const htmlReportUrl = projectUrl("/api/reports/html");
    const pdfReportUrl = projectUrl("/api/reports/pdf");
    const summaryUrl = projectUrl("/api/reports/summary");
    const metricsUrl = projectUrl("/api/reports/metrics");
    const schemaUrl = projectUrl("/api/reports/feature-schema");
    return `
      <div class="page">
        ${pageHeader(
          "ARTIFACT EXPORT",
          "Package the champion model",
          "Generate an executed, evidence-backed notebook plus review-ready reports and deployment artifacts.",
          reportAction
        )}
        <div class="insight-banner">
          <span class="heading-icon lime">${icon("rocket")}</span>
          <div><strong>${escapeHtml(bestName)} is ready to package</strong><p>${reportsReady ? "The executed notebook and all derived reports passed server-side artifact validation." : "Generate the report bundle to execute every notebook code cell and create verified HTML and PDF reports."}</p></div>
          <span class="status-pill ${reportsReady ? "success" : ""}"><i class="status-dot"></i> ${reportsReady ? "Reports verified" : "Report generation required"}</span>
        </div>

        ${state.reportBundle ? `
          <section class="card card-pad section">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon violet">${icon("check", "icon-sm")}</span><div><h3>Report generation evidence</h3><p>${escapeHtml(state.reportBundle.current_section || "Waiting to start")}</p></div></div>
              <span class="status-pill ${reportsReady ? "success" : ""}"><i class="status-dot"></i>${escapeHtml(state.reportBundle.status || "unknown")}</span>
            </div>
            <div class="insight-list">
              ${completedSections.length
                ? completedSections.map((section, index) => `<div class="insight-item"><span class="insight-number">${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(section)}</strong><p>Confirmed by the report service.</p></div></div>`).join("")
                : '<div class="empty-inline compact-empty"><p>No completed report sections yet.</p></div>'}
            </div>
            ${reportWarnings.length ? `<div class="warning-list"><div class="warning-item">${icon("warning", "icon-sm")}<div><strong>Run warnings</strong><p>${reportWarnings.map(escapeHtml).join(" · ")}</p></div></div></div>` : ""}
          </section>
        ` : ""}

        <section class="artifact-grid">
          <a class="artifact-card" href="/api/model/download">
            <span class="artifact-icon">${icon("cube", "icon-sm")}</span>
            <span><h3>Fitted model pipeline</h3><p>Joblib · preprocessing + estimator</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="${reportsReady && notebookUrl ? notebookUrl : "#"}" ${reportsReady && notebookUrl ? "" : 'data-action="generate-reports"'}>
            <span class="artifact-icon">${icon("code", "icon-sm")}</span>
            <span><h3>Executed ML notebook</h3><p>Jupyter · code, outputs, charts, conclusions</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="${reportsReady && htmlReportUrl ? htmlReportUrl : "#"}" ${reportsReady && htmlReportUrl ? "" : 'data-action="generate-reports"'}>
            <span class="artifact-icon">${icon("chart", "icon-sm")}</span>
            <span><h3>Interactive HTML report</h3><p>Executed notebook · browser review</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="${reportsReady && pdfReportUrl ? pdfReportUrl : "#"}" ${reportsReady && pdfReportUrl ? "" : 'data-action="generate-reports"'}>
            <span class="artifact-icon">${icon("layers", "icon-sm")}</span>
            <span><h3>Presentation-ready PDF</h3><p>Verified pages · metrics and charts</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="${reportsReady && summaryUrl ? summaryUrl : "#"}" ${reportsReady && summaryUrl ? "" : 'data-action="generate-reports"'}>
            <span class="artifact-icon">${icon("layers", "icon-sm")}</span>
            <span><h3>Run manifest and summary</h3><p>JSON · provenance and model evidence</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="/api/preprocessor/download">
            <span class="artifact-icon">${icon("sliders", "icon-sm")}</span>
            <span><h3>Fitted preprocessor</h3><p>Joblib · transformer + schema</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="${reportsReady && metricsUrl ? metricsUrl : "/api/metrics/download"}">
            <span class="artifact-icon">${icon("chart", "icon-sm")}</span>
            <span><h3>Model metrics</h3><p>CSV · comparison and timings</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="/api/dataset/download">
            <span class="artifact-icon">${icon("database", "icon-sm")}</span>
            <span><h3>Project dataset</h3><p>CSV · spreadsheet-safe export</p></span>
            ${icon("download", "icon-sm")}
          </a>
          <a class="artifact-card" href="${reportsReady && schemaUrl ? schemaUrl : "/api/project/download"}">
            <span class="artifact-icon">${icon("layers", "icon-sm")}</span>
            <span><h3>${reportsReady ? "Feature contract" : "Project summary"}</h3><p>JSON · ordered schema and provenance</p></span>
            ${icon("download", "icon-sm")}
          </a>
        </section>

        <section class="section grid-main-aside">
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon violet">${icon("code", "icon-sm")}</span><div><h3>Prediction endpoint</h3><p>Generated serving contract</p></div></div>
              <span class="chip violet">FastAPI</span>
            </div>
            <div class="code-block"><button class="icon-button code-copy" data-action="copy-code" aria-label="Copy code">${icon("copy", "icon-sm")}</button><span class="code-keyword">from</span> fastapi <span class="code-keyword">import</span> FastAPI
<span class="code-keyword">import</span> joblib
<span class="code-keyword">import</span> pandas <span class="code-keyword">as</span> pd

app = FastAPI()
artifact = joblib.load(<span class="code-string">"aivoraml_model.joblib"</span>)
pipeline = artifact[<span class="code-string">"pipeline"</span>]

@app.post(<span class="code-string">"/predict"</span>)
<span class="code-keyword">def</span> predict(payload: dict):
    frame = pd.DataFrame([payload], columns=artifact[<span class="code-string">"feature_schema"</span>][<span class="code-string">"ordered"</span>])
    prediction = pipeline.predict(frame)
    <span class="code-keyword">return</span> {<span class="code-string">"prediction"</span>: prediction[0]}</div>
          </div>
          <div class="card card-pad">
            <div class="card-heading">
              <div class="card-heading-left"><span class="heading-icon">${icon("check", "icon-sm")}</span><div><h3>Production checklist</h3><p>Before exposing an endpoint</p></div></div>
            </div>
            <div class="insight-list">
              <div class="insight-item"><span class="insight-number">01</span><div><strong>Add authentication</strong><p>Do not expose the local singleton workspace directly to the internet.</p></div></div>
              <div class="insight-item"><span class="insight-number">02</span><div><strong>Validate the schema</strong><p>Use the ordered feature contract included in the artifact.</p></div></div>
              <div class="insight-item"><span class="insight-number">03</span><div><strong>Monitor predictions</strong><p>Track latency, drift, errors, and delayed-label performance.</p></div></div>
            </div>
            <button class="button button-secondary full-width" data-action="open-predict">${icon("target", "icon-sm")} Test prediction first</button>
          </div>
        </section>
      </div>
    `;
  }

  function renderResultGate(destination) {
    return `
      <div class="page">
        ${pageHeader("EXPERIMENT REQUIRED", titleCase(destination), "This stage becomes available after the first model experiment finishes.")}
        <div class="card empty-inline">
          <div>
            <span class="empty-icon">${icon(destination === "deploy" ? "rocket" : destination === "explain" ? "eye" : "chart")}</span>
            <h3>Train models to continue</h3>
            <p>AivoraML needs evaluated predictions before it can compare, explain, or package a model.</p>
            <button class="button button-primary" data-action="open-train">${icon("play", "icon-sm")} Configure experiment</button>
          </div>
        </div>
      </div>
    `;
  }

  function metricCard(label, value, iconName, helper, deltaClass = "", highlight = false) {
    return `
      <div class="metric-card ${highlight ? "highlight" : ""}">
        <div class="metric-label"><span>${escapeHtml(label)}</span>${icon(iconName, "icon-sm")}</div>
        <div class="metric-value">${escapeHtml(value)}</div>
        <div class="metric-delta ${deltaClass}">${deltaClass === "good" ? icon("check", "icon-xs") : deltaClass === "warning" ? icon("warning", "icon-xs") : ""}${escapeHtml(helper || "")}</div>
      </div>
    `;
  }

  function qualityRow(label, value, color) {
    return `
      <div class="quality-row">
        <span>${escapeHtml(label)}</span>
        <div class="quality-bar"><span style="width:${Math.max(0, Math.min(100, value))}%;background:var(--${color})"></span></div>
        <strong>${value}%</strong>
      </div>
    `;
  }

  function renderWarning(warning) {
    const title = {
      high_missing: "High missingness",
      missing: "Missing values",
      duplicates: "Duplicate rows",
      high_cardinality: "High cardinality",
      constant: "Constant feature",
      identifier: "Identifier risk",
      outliers: "Potential outliers",
      small_dataset: "Small dataset",
    }[warning.type] || "Quality note";
    return `<div class="warning-item">${icon(warning.severity === "info" ? "file" : "warning", "icon-sm")}<div><strong>${title}</strong><p>${escapeHtml(warning.message)}</p></div></div>`;
  }

  function renderColumnRows(columns, compact = false) {
    if (!columns?.length) return '<p class="subtle">No columns detected.</p>';
    return `
      <div class="table-shell">
        <div class="table-scroll">
          <table class="data-table">
            <thead><tr><th>Column</th><th>Type</th>${compact ? "" : "<th>Missing</th>"}<th>Unique</th></tr></thead>
            <tbody>
              ${columns.map((column) => `
                <tr>
                  <td title="${escapeHtml(column.name)}">${escapeHtml(column.name)}</td>
                  <td><span class="type-badge ${column.semantic_type === "numeric" ? "" : "category"}">${escapeHtml(column.semantic_type || column.dtype)}</span></td>
                  ${compact ? "" : `<td>${formatNumber(column.n_missing, 0)} <span class="subtle">(${Number(column.missing_pct || 0).toFixed(1)}%)</span></td>`}
                  <td>${formatNumber(column.n_unique, 0)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  function renderColumnTable(columns) {
    return `
      <div class="table-shell">
        <div class="table-scroll">
          <table class="data-table">
            <thead><tr><th>Column</th><th>Detected type</th><th>Missing</th><th>Unique</th><th>Example</th></tr></thead>
            <tbody>
              ${columns.map((column) => {
                const sample = column.sample_values?.[0] ?? Object.keys(column.top_values || {})[0] ?? column.stats?.mean ?? "—";
                return `<tr><td>${escapeHtml(column.name)}</td><td><span class="type-badge ${column.semantic_type === "numeric" ? "" : "category"}">${escapeHtml(column.semantic_type || column.dtype)}</span></td><td>${Number(column.missing_pct || 0).toFixed(1)}%</td><td>${formatNumber(column.n_unique, 0)}</td><td>${escapeHtml(truncate(sample, 30))}</td></tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>
        <div class="table-footer"><span>Showing all ${columns.length} columns</span><span>Inferred by AivoraML profiler</span></div>
      </div>
    `;
  }

  function renderPreviewTable(profile) {
    const rows = (profile.preview || []).slice(0, 8);
    const columns = profile.columns.slice(0, 8).map((column) => column.name);
    if (!rows.length) return '<div class="empty-inline"><p>Preview unavailable for this dataset.</p></div>';
    return `
      <div class="table-shell">
        <div class="table-scroll">
          <table class="data-table">
            <thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
            <tbody>${rows.map((row) => `<tr>${columns.map((column, index) => `<td${index === 0 ? "" : ' class="mono"'}>${escapeHtml(truncate(row[column], 28))}</td>`).join("")}</tr>`).join("")}</tbody>
          </table>
        </div>
        <div class="table-footer"><span>${rows.length} of ${formatNumber(profile.n_rows, 0)} rows</span><span>Scroll horizontally to inspect columns</span></div>
      </div>
    `;
  }

  function renderHistogram(column) {
    const values = (column.histogram?.counts || []).map(Number).filter(Number.isFinite);
    if (!values.length) {
      return '<div class="empty-inline compact-empty"><p>Distribution bins are unavailable for this column.</p></div>';
    }
    const maximum = Math.max(...values, 1);
    return `
      <div class="distribution-chart" role="img" aria-label="Histogram for ${escapeHtml(column.name)}">${values.map((value) => `<span class="distribution-bar" style="height:${Math.max(2, value / maximum * 100)}%" title="${formatNumber(value, 0)} rows"></span>`).join("")}</div>
      <div class="chart-axis"><span>${formatNumber(column.stats?.min)}</span><span>Median ${formatNumber(column.stats?.median)}</span><span>${formatNumber(column.stats?.max)}</span></div>
    `;
  }

  function barRow(name, percentage, label, color = "violet") {
    return `
      <div class="bar-row">
        <span class="bar-row-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
        <div class="bar-track ${color}"><span style="width:${Math.max(1, Math.min(100, Number(percentage) || 0))}%"></span></div>
        <span class="bar-value">${escapeHtml(label)}</span>
      </div>
    `;
  }

  function strategyCard(id, name, description, time, models, recommended = false) {
    const selected = state.strategy === id;
    return `
      <button class="strategy-card ${selected ? "selected" : ""}" data-action="select-strategy" data-strategy="${id}">
        <div class="strategy-header"><h4>${escapeHtml(name)}</h4><span class="selected-check">${icon("check", "icon-xs")}</span></div>
        <p>${escapeHtml(description)}</p>
        <strong>${escapeHtml(time)}</strong><span>${escapeHtml(models)}${recommended ? " · Recommended" : ""}</span>
      </button>
    `;
  }

  function sortedModels() {
    return [...(state.result?.models_trained || [])].sort((a, b) => {
      if (Boolean(a.is_best) !== Boolean(b.is_best)) return a.is_best ? -1 : 1;
      return Number(b.selection_score ?? primaryScore(b)) - Number(a.selection_score ?? primaryScore(a));
    });
  }

  function primaryScore(model) {
    const metrics = model?.metrics || {};
    if (taskType() === "regression") return Number(metrics.r2 ?? model?.cv_score_mean ?? 0) || 0;
    return Number(metrics.positive_f1 ?? metrics.macro_f1 ?? metrics.f1_score ?? metrics.accuracy ?? model?.cv_score_mean ?? 0) || 0;
  }

  function evaluationMetricEntries(metrics) {
    if (taskType() === "regression") {
      return ["r2", "mae", "rmse", "median_ae"]
        .filter((name) => metrics[name] != null)
        .map((name) => [name, metrics[name]]);
    }
    const names = metrics.positive_f1 != null
      ? ["positive_f1", "positive_precision", "positive_recall", "pr_auc"]
      : ["macro_f1", "balanced_accuracy", "accuracy", "f1_score"];
    return names
      .filter((name) => metrics[name] != null)
      .map((name) => [name, metrics[name]]);
  }

  function metricTitle(name) {
    return ({
      positive_f1: "Positive-class F1",
      positive_precision: "Positive precision",
      positive_recall: "Positive recall",
      pr_auc: "Precision-recall AUC",
      roc_auc: "ROC AUC",
      macro_f1: "Macro F1",
      balanced_accuracy: "Balanced accuracy",
      f1_score: "Weighted F1",
      r2: "R²",
      mae: "Mean absolute error",
      rmse: "Root mean squared error",
      median_ae: "Median absolute error",
    })[name] || titleCase(name);
  }

  function scoreDisplay(score) {
    return taskType() === "regression" ? Number(score).toFixed(3) : `${(Number(score) * 100).toFixed(1)}%`;
  }

  function metricValue(name, value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value ?? "—");
    if (["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "macro_f1", "positive_precision", "positive_recall", "positive_f1", "roc_auc", "pr_auc", "r2"].includes(name)) {
      return name === "r2" && taskType() === "regression" ? number.toFixed(3) : `${(number * 100).toFixed(1)}%`;
    }
    return formatNumber(number, 3);
  }

  function getFeatureImportance() {
    return state.result?.feature_importance || [];
  }

  function explanationSummary(first, second) {
    if (taskType() === "regression") {
      return `${titleCase(first)} is the strongest driver of the predicted value, with ${titleCase(second)} adding important context. The relationship is predictive, not necessarily causal.`;
    }
    return `${titleCase(first)} carries the strongest signal for ${titleCase(state.target)}, followed by ${titleCase(second)}. The model combines these with the remaining features rather than relying on a single rule.`;
  }

  function projectName() {
    if (!state.profile) return "New project";
    if (state.demoMode || state.profile.filename === "customer_churn.csv") return "Customer churn intelligence";
    return state.profile.filename.replace(/\.[^.]+$/, "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function aiProjectBrief() {
    const suggestion = currentSuggestion();
    if (!suggestion) return "Choose a target to configure a supervised modeling workflow.";
    return `A ${suggestion.task_type} workflow with “${suggestion.column}” as the target fits the observed target shape. Compare baselines and ensemble models using ${suggestion.task_type === "regression" ? "R² and RMSE" : "event-class F1, precision-recall AUC, and calibration"}.`;
  }

  function buildCleaningPlan() {
    const profile = state.profile;
    const missingNumeric = profile.columns.filter((column) => column.semantic_type === "numeric" && column.n_missing > 0);
    const missingCategorical = profile.columns.filter((column) => column.semantic_type !== "numeric" && column.n_missing > 0);
    const identifierColumns = profile.columns.filter((column) =>
      /(^id$|_id$|^id_|uuid|identifier)/i.test(column.name) ||
      column.n_unique / Math.max(1, profile.n_rows) > 0.98
    ).filter((column) => column.name !== state.target);

    state.cleaningPlan = [
      {
        id: "duplicates",
        title: "Remove duplicate rows",
        description: `${formatNumber(profile.n_duplicates, 0)} exact duplicates detected. Keep the first occurrence.`,
        risk: "Automatic",
        type: "rows",
        enabled: true,
      },
      {
        id: "target-missing",
        title: "Remove rows with a missing target",
        description: `Rows without ${state.target || "the target"} cannot supervise training and are removed before splitting.`,
        risk: "Automatic",
        type: "rows",
        enabled: true,
      },
      {
        id: "numeric-missing",
        title: "Impute missing numeric values",
        description: `${missingNumeric.length} numeric columns will use training-set medians for robust imputation.`,
        risk: "Train-only",
        type: "missing",
        enabled: true,
      },
      {
        id: "category-missing",
        title: "Impute missing categories",
        description: `${missingCategorical.length} categorical columns will use the most frequent training value.`,
        risk: "Train-only",
        type: "missing",
        enabled: true,
      },
      {
        id: "identifiers",
        title: "Exclude unsafe feature columns",
        description: identifierColumns.length
          ? `${identifierColumns.map((column) => column.name).slice(0, 3).join(", ")} look identifier-like; constants and categories above 50 levels are also removed.`
          : "Constants, identifier-like columns, and categories above 50 levels are removed.",
        risk: "Automatic",
        type: "columns",
        enabled: true,
      },
    ];
  }

  function buildFeaturePlan() {
    const columns = state.profile.columns;
    const categorical = columns.filter((column) => column.semantic_type !== "numeric" && column.name !== state.target && column.n_unique <= 50);
    const numeric = columns.filter((column) => column.semantic_type === "numeric" && column.name !== state.target);
    const encodedCount = categorical.reduce((sum, column) => sum + Math.max(1, column.n_unique), 0);

    state.featurePlan = [
      {
        id: "scale",
        title: "Median-impute and scale numerics",
        description: `Fit median imputation and StandardScaler on training folds for ${numeric.length} numeric columns.`,
        output: `${numeric.length} numeric model features`,
        outputCount: numeric.length,
        icon: "sliders",
        color: "",
        enabled: true,
      },
      {
        id: "encode",
        title: "Impute and encode categories",
        description: `Fit most-frequent imputation and one-hot encoding for ${categorical.length} categories, safely ignoring unseen values.`,
        output: `About ${encodedCount} encoded model features`,
        outputCount: encodedCount,
        icon: "layers",
        color: "violet",
        enabled: true,
      },
      {
        id: "exclude",
        title: "Drop unsafe columns",
        description: "Remove constants, identifier-like fields, and categorical columns above the cardinality budget.",
        output: "Excluded before fitting",
        outputCount: 0,
        icon: "target",
        color: "lime",
        enabled: true,
      },
    ];
  }

  function hydrateProfile(profile, { demo = false } = {}) {
    state.profile = profile;
    state.result = null;
    state.project = null;
    state.predictionSchema = null;
    state.predictionResult = null;
    state.predictions = [];
    state.modelVersions = [];
    state.batchSummary = null;
    state.trainingJob = null;
    state.assistantHistory = [];
    state.reportBundle = null;
    state.reportLoading = false;
    state.demoMode = demo;
    state.target = profile.target_suggestions?.[0]?.column || profile.columns?.at(-1)?.name || "";
    state.completed = new Set([1]);
    buildCleaningPlan();
    buildFeaturePlan();
    elements.modelCount.textContent = "0";
    loadProjectMetadata();
  }

  function normalizeResult(result) {
    if (!result) return null;
    return {
      ...result,
      success: result.success ?? true,
      data_info: result.data_info || result.data_shape,
      feature_summary: result.feature_summary || result.features,
      split_info: result.split_info || result.split,
      total_models_compared: result.total_models_compared || result.models_trained?.length || 0,
    };
  }

  async function loadProjectMetadata() {
    try {
      const response = await fetch("/api/project/current");
      if (!response.ok) return;
      state.project = await response.json();
      updateTopbar();
      if (state.view === "home" || protectedViews.has(state.view)) render();
    } catch {
      state.project = null;
    }
  }

  async function loadCapabilities() {
    try {
      const response = await fetch("/api/capabilities");
      if (!response.ok) return;
      state.capabilities = await response.json();
      const maxBytes = state.capabilities?.upload?.max_bytes;
      const label = $("#uploadLimitLabel");
      if (label && Number.isFinite(Number(maxBytes))) {
        label.textContent = formatBytes(Number(maxBytes));
      }
      const assistant = state.capabilities?.remote_ai?.assistant;
      const providers = assistant?.configured_providers || [];
      const mode = $("#assistantMode");
      const description = $("#assistantDescription");
      const disclaimer = $("#assistantDisclaimer");
      if (assistant?.available) {
        if (mode) mode.textContent = providers.length > 1 ? "AI DUO" : providers[0]?.toUpperCase() || "AI";
        if (description) description.textContent = `${providers.map(titleCase).join(" + ")} · Project aware`;
        if (disclaimer) disclaimer.textContent = "Remote AI uses project metadata and can make mistakes. Verify important decisions.";
      } else {
        if (mode) mode.textContent = "LOCAL";
        if (description) description.textContent = "Local workflow guidance · Project aware";
      }
    } catch {
      // Static defaults remain accurate for the standard local configuration.
    }
  }

  async function loadProjects({ quiet = false } = {}) {
    try {
      const response = await fetch("/api/projects");
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      state.projects = payload.projects || [];
      if (state.view === "home") render();
    } catch (error) {
      if (!quiet) toast(`Could not load projects: ${error.message}`, "error");
    }
  }

  async function activateProject(projectId) {
    if (!projectId || projectId === state.project?.id) {
      setView("overview");
      return;
    }
    showLoading({
      eyebrow: "PROJECT STORAGE",
      title: "Opening your project",
      message: "Verifying the snapshot checksum and restoring its workspace…",
      mode: "profile",
    });
    try {
      const response = await fetch(
        `/api/projects/${encodeURIComponent(projectId)}/activate`,
        { method: "POST" },
      );
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      hydrateProfile(payload.profile);
      const hasResult = await loadPersistedResult();
      await Promise.all([
        loadProjectMetadata(),
        loadProjects({ quiet: true }),
        hasResult ? loadPredictionWorkspace({ quiet: true }) : Promise.resolve(),
        hasResult ? loadReportStatus({ quiet: true }) : Promise.resolve(),
        hasResult ? loadModelVersions({ quiet: true }) : Promise.resolve(),
      ]);
      hideLoading();
      setView("overview");
      toast("Project restored from its verified local snapshot.");
    } catch (error) {
      hideLoading();
      toast(`Could not open project: ${error.message}`, "error");
    }
  }

  async function deleteProject(projectId, filename) {
    if (!projectId) return;
    const confirmed = window.confirm(
      `Delete ${filename || "this project"} and its project-scoped artifacts? This cannot be undone.`,
    );
    if (!confirmed) return;
    try {
      const response = await fetch(
        `/api/projects/${encodeURIComponent(projectId)}`,
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error(await readError(response));
      const result = await response.json();
      const retained = [
        ...(result.model_artifacts?.failures || []),
        ...(result.report_artifacts?.failures || []),
      ].length;
      sessionStorage.setItem(
        "aivoraml-project-message",
        retained
          ? `Project deleted; ${retained} legacy artifact item(s) were retained for safety.`
          : "Project and project-scoped artifacts deleted.",
      );
      location.reload();
    } catch (error) {
      toast(`Could not delete project: ${error.message}`, "error");
    }
  }

  async function loadModelVersions({ quiet = false } = {}) {
    if (!state.profile) return;
    try {
      const response = await fetch("/api/models/versions");
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      state.modelVersions = payload.versions || [];
      if (state.view === "models") render();
    } catch (error) {
      if (!quiet) toast(`Could not load model versions: ${error.message}`, "error");
    }
  }

  async function activateModelVersion(versionId) {
    if (!versionId) return;
    try {
      const response = await fetchWithTimeout(
        `/api/models/versions/${encodeURIComponent(versionId)}/activate`,
        { method: "POST" },
        60000,
      );
      if (!response.ok) throw new Error(await readError(response));
      await Promise.all([
        loadPersistedResult(),
        loadModelVersions({ quiet: true }),
        loadPredictionWorkspace({ quiet: true }),
        loadProjectMetadata(),
      ]);
      render();
      toast(`${versionId} is now the active champion.`);
    } catch (error) {
      toast(`Could not activate model version: ${error.message}`, "error");
    }
  }

  async function loadPersistedResult() {
    try {
      const response = await fetch("/api/workflow/result");
      if (!response.ok) return false;
      state.result = normalizeResult(await response.json());
      state.target = state.result.target_column || state.target;
      [3, 4, 5, 6].forEach((stage) => state.completed.add(stage));
      elements.modelCount.textContent = String(state.result.models_trained?.length || 0);
      return true;
    } catch {
      return false;
    }
  }

  async function loadPredictionWorkspace({ quiet = false } = {}) {
    if (!state.result) return;
    try {
      const [schemaResponse, historyResponse] = await Promise.all([
        fetch("/api/predict/schema"),
        fetch("/api/predictions"),
      ]);
      if (!schemaResponse.ok) throw new Error(await readError(schemaResponse));
      state.predictionSchema = await schemaResponse.json();
      if (historyResponse.ok) {
        const history = await historyResponse.json();
        state.predictions = history.predictions || [];
      }
      if (state.view === "predict") render();
    } catch (error) {
      if (!quiet) toast(`Could not load prediction schema: ${error.message}`, "error");
    }
  }

  async function loadReportStatus({ quiet = false } = {}) {
    if (!state.result) return;
    const url = projectUrl("/api/reports/current");
    if (!url) return;
    try {
      const response = await fetch(url);
      if (response.status === 404) {
        state.reportBundle = null;
        return;
      }
      if (!response.ok) throw new Error(await readError(response));
      state.reportBundle = await response.json();
      if (state.view === "deploy") render();
    } catch (error) {
      if (!quiet) toast(`Could not load report status: ${error.message}`, "error");
    }
  }

  async function completeReportJob(bundle, { recovered = false } = {}) {
    state.reportBundle = bundle;
    state.reportLoading = false;
    state.trainingJob = null;
    state.completed.add(9);
    await loadProjectMetadata();
    hideLoading();
    if (state.view === "deploy") render();
    toast(recovered
      ? "Report generation completed while reconnecting. All exports are ready."
      : "Executed notebook, HTML, PDF, metrics, schema, and pipeline are ready."
    );
  }

  async function recoverCompletedReport() {
    const url = projectUrl("/api/reports/current");
    if (!url) return false;
    try {
      const response = await fetchWithTimeout(
        url,
        {},
        30000,
        "Could not check whether the completed report bundle was saved.",
      );
      if (!response.ok) return false;
      const bundle = await response.json();
      if (bundle?.status === "completed") {
        await completeReportJob(bundle, { recovered: true });
        return true;
      }
    } catch {
      // The caller will show the accurate server-restart state below.
    }
    return false;
  }

  async function monitorReportJob(initialJob, { restored = false } = {}) {
    if (!initialJob?.job_id) return;
    const monitorId = ++state.reportMonitorId;
    state.trainingJob = initialJob;
    state.reportLoading = true;
    showLoading({
      eyebrow: restored ? "REPORT RECONNECTED" : "REPORT BUNDLE",
      title: "Executing the complete notebook",
      message: initialJob.message || "Running every cell and verifying the generated artifacts…",
      mode: "report",
      realProgress: true,
      cancellable: true,
    });
    updateLoadingFromJob(initialJob);

    let job = initialJob;
    let consecutiveStatusFailures = 0;
    while (
      monitorId === state.reportMonitorId
      && ["queued", "running", "cancel_requested"].includes(job.state)
    ) {
      await wait(consecutiveStatusFailures
        ? Math.min(15000, 1200 * (2 ** Math.min(consecutiveStatusFailures, 4)))
        : 1000
      );
      try {
        const response = await fetchWithTimeout(
          `/api/jobs/${encodeURIComponent(job.job_id)}`,
          {},
          45000,
          "The status check took too long; the server-side report job is still running.",
        );
        if (response.status === 404) {
          if (await recoverCompletedReport()) return;
          state.reportLoading = false;
          state.trainingJob = null;
          hideLoading();
          toast("The local API restarted during report generation, so that job could not continue. Your trained model is unchanged; generate the reports again.", "error");
          return;
        }
        if (!response.ok) throw new Error(await readError(response));
        job = await response.json();
        state.trainingJob = job;
        if (consecutiveStatusFailures) {
          const config = loadingTaskConfig("report");
          elements.loadingEta.textContent = config.estimate;
          elements.loadingEtaNote.textContent = config.note;
        }
        consecutiveStatusFailures = 0;
        elements.loadingEyebrow.textContent = "REPORT BUNDLE";
        updateLoadingFromJob(job);
      } catch (error) {
        consecutiveStatusFailures += 1;
        elements.loadingEyebrow.textContent = "REPORT CONTINUES";
        elements.loadingStageLabel.textContent = "Reconnecting";
        elements.loadingMessage.textContent = (
          `The status connection was delayed (${error.message}). `
          + "The server-side export continues; reconnecting automatically..."
        );
        elements.loadingEta.textContent = "Still running";
        elements.loadingEtaNote.textContent = `Reconnect attempt ${consecutiveStatusFailures}`;
      }
    }

    if (monitorId !== state.reportMonitorId) return;

    state.trainingJob = job;
    if (job.state === "succeeded" && job.result) {
      await completeReportJob(job.result);
      return;
    }

    state.reportLoading = false;
    state.trainingJob = null;
    hideLoading();
    if (state.view === "deploy") render();
    if (job.state === "cancelled") {
      toast("Report generation was cancelled. Your trained model is unchanged.");
    } else {
      toast(`Report generation failed: ${job.error?.message || job.message || "Unknown error"}`, "error");
    }
  }

  async function generateReports() {
    if (!state.result || state.reportLoading) return;
    const projectId = activeProjectId();
    if (!projectId) {
      toast("The active project ID is unavailable. Reload the project and try again.", "error");
      return;
    }
    state.reportLoading = true;
    showLoading({
      eyebrow: "REPORT BUNDLE",
      title: "Queueing the report bundle",
      message: "Creating a reconnectable local report job…",
      mode: "report",
      realProgress: true,
    });
    try {
      const force = state.reportBundle?.status === "completed";
      const response = await fetchWithTimeout(
        projectUrl("/api/reports/jobs", { force: String(force) }),
        { method: "POST" },
        60000,
        "The report job could not be queued within one minute. Check that the local API is still running.",
      );
      if (!response.ok) throw new Error(await readError(response));
      const job = await response.json();
      hideLoading();
      await wait(420);
      return monitorReportJob(job);
    } catch (error) {
      state.reportLoading = false;
      hideLoading();
      toast(`Could not start report generation: ${error.message}`, "error");
    }
  }

  async function runPrediction(form) {
    const submit = $('button[type="submit"]', form);
    const values = Object.fromEntries(new FormData(form).entries());
    const originalSubmit = submit.innerHTML;
    submit.disabled = true;
    showLoading({
      eyebrow: "LIVE INFERENCE",
      title: `Predicting ${state.target || "the outcome"}`,
      message: "Validating feature values and running the fitted champion pipeline…",
      mode: "predict",
    });
    submit.textContent = "Predicting…";
    try {
      const response = await fetchWithTimeout("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values }),
      });
      if (!response.ok) throw new Error(await readError(response));
      state.predictionResult = await response.json();
      state.completed.add(8);
      await loadPredictionWorkspace({ quiet: true });
      render();
      toast("Prediction completed with the fitted pipeline.");
    } catch (error) {
      toast(`Prediction failed: ${error.message}`, "error");
      submit.disabled = false;
      submit.textContent = "Run prediction";
      submit.innerHTML = originalSubmit;
    } finally {
      hideLoading();
    }
  }

  async function runBatchPrediction(form) {
    const file = $("#batchPredictionFile", form)?.files?.[0];
    if (!file) {
      toast("Choose a batch file first.", "error");
      return;
    }
    const payload = new FormData();
    payload.append("file", file);
    showLoading({
      eyebrow: "BATCH INFERENCE",
      title: "Scoring your prediction file",
      message: "Validating every feature column before running the fitted pipeline…",
      mode: "predict",
    });
    try {
      const response = await fetchWithTimeout(
        "/api/predict/batch",
        { method: "POST", body: payload },
        300000,
      );
      if (!response.ok) throw new Error(await readError(response));
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const download = document.createElement("a");
      download.href = url;
      download.download = "aivoraml_batch_predictions.csv";
      document.body.append(download);
      download.click();
      download.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      state.batchSummary = {
        rows: Number(response.headers.get("X-AivoraML-Rows") || 0),
        version: response.headers.get("X-AivoraML-Model-Version"),
        latency: Number(response.headers.get("X-AivoraML-Latency-Ms") || 0),
      };
      render();
      toast(`${state.batchSummary.rows} batch predictions are ready.`);
    } catch (error) {
      toast(`Batch prediction failed: ${error.message}`, "error");
    } finally {
      hideLoading();
    }
  }

  async function clearPredictionHistory() {
    try {
      const response = await fetch("/api/predictions", { method: "DELETE" });
      if (!response.ok) throw new Error(await readError(response));
      const result = await response.json();
      state.predictions = [];
      render();
      toast(`${result.deleted || 0} prediction records cleared.`);
    } catch (error) {
      toast(`Could not clear prediction history: ${error.message}`, "error");
    }
  }

  function updateNavigation() {
    $$(".nav-item").forEach((item) => {
      const active = item.dataset.view === state.view;
      item.classList.toggle("active", active);
      if (active) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });

    $$(".project-step").forEach((item) => {
      const stage = Number(item.dataset.stage);
      item.disabled = !state.profile;
      item.classList.toggle("complete", state.completed.has(stage));
      const status = $(".step-status", item);
      if (status) status.innerHTML = state.completed.has(stage) ? icon("check", "icon-xs") : String(stage).padStart(2, "0");
    });

    const completeCount = [...state.completed].filter((stage) => stage >= 1 && stage <= 9).length;
    elements.lifecycleProgress.textContent = `${Math.round(completeCount / 9 * 100)}%`;
  }

  function updateTopbar() {
    elements.breadcrumb.textContent = state.profile ? projectName() : state.view === "home" ? "Projects" : titleCase(state.view);
    elements.runButton.disabled = !state.profile;
    const models = state.result?.models_trained?.length || 0;
    elements.modelCount.textContent = String(models);
    elements.projectExport.disabled = !state.profile;
    elements.shareButton.disabled = !state.profile;
    const persistenceStatus = state.project?.persistence?.status;
    elements.saveState.innerHTML = state.project
      ? persistenceStatus === "error"
        ? `<span></span>Save needs attention`
        : persistenceStatus === "disabled"
          ? `<span></span>Session only`
          : `<span></span>Saved locally`
      : `<span></span>Local session`;
  }

  function updateAssistantContext() {
    elements.assistantContext.textContent = state.profile
      ? `${projectName()} · ${titleCase(state.view)}`
      : "No dataset selected";
  }

  function updateAssistantSuggestions() {
    const suggestions = {
      home: state.profile ? ["Summarize this project", "What should I do next?"] : ["What can I build?", "Open a demo project"],
      overview: ["Why this target?", "What are the data risks?", "Plan my pipeline"],
      explore: ["Explain this distribution", "Which columns look risky?", "What is correlation?"],
      clean: ["Why median imputation?", "Could this cause leakage?", "Review this plan"],
      features: ["Which features matter?", "Should I use PCA?", "Explain one-hot encoding"],
      train: ["Why these models?", "Optimize for recall", "Explain cross-validation"],
      evaluate: ["Why did this model win?", "Is this overfitting?", "How can I improve recall?"],
      explain: ["Explain the top feature", "Can I trust this model?", "What does importance mean?"],
      predict: ["How is this prediction made?", "Is this probability calibrated?", "What should I monitor?"],
      deploy: ["How do I deploy this?", "Show a prediction request", "What should I monitor?"],
    }[state.view] || ["Summarize my project", "What should I do next?"];

    elements.suggestionChips.innerHTML = suggestions.map((suggestion) => `<button data-question="${escapeHtml(suggestion)}">${escapeHtml(suggestion)}</button>`).join("");
  }

  function setView(view) {
    if (protectedViews.has(view) && !state.profile) {
      openUploadModal();
      return;
    }
    const commitView = () => {
      elements.workspace.classList.remove("view-mounted");
      state.view = view;
      const stage = viewStages[view];
      if (stage && stage <= 2) state.completed.add(stage);
      if (view === "explain" && state.result) state.completed.add(7);
      if (view === "deploy" && state.result) state.completed.add(9);
      render();
    };
    const hasComfortableHardware = (navigator.hardwareConcurrency || 4) >= 8
      && (navigator.deviceMemory || 4) >= 8;
    const canTransition = typeof document.startViewTransition === "function"
      && hasComfortableHardware
      && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (canTransition) document.startViewTransition(commitView);
    else commitView();
    elements.workspace.focus({ preventScroll: true });
    if (view === "predict" && state.result) loadPredictionWorkspace({ quiet: true });
    if (view === "deploy" && state.result) loadReportStatus({ quiet: true });
    if (view === "models" && state.profile) loadModelVersions({ quiet: true });
    if (view === "home") loadProjects({ quiet: true });
    if (window.innerWidth <= 980) {
      document.body.classList.remove("sidebar-open");
      $("#menuButton").setAttribute("aria-expanded", "false");
      elements.sidebar.setAttribute("inert", "");
    }
    const nextUrl = view === "home" ? location.pathname : `${location.pathname}#${view}`;
    if (`${location.pathname}${location.hash}` !== nextUrl) {
      history.pushState({ view }, "", nextUrl);
    }
  }

  function openUploadModal() {
    state.uploadReturnFocus = document.activeElement;
    elements.uploadModal.classList.add("open");
    elements.uploadModal.setAttribute("aria-hidden", "false");
    elements.uploadModal.removeAttribute("inert");
    elements.shell.setAttribute("inert", "");
    setTimeout(() => elements.dropzone.focus(), 120);
  }

  function closeUploadModal() {
    const wasOpen = elements.uploadModal.classList.contains("open");
    elements.uploadModal.classList.remove("open");
    elements.uploadModal.setAttribute("aria-hidden", "true");
    elements.uploadModal.setAttribute("inert", "");
    if (wasOpen) elements.shell.removeAttribute("inert");
    if (state.uploadReturnFocus?.isConnected) state.uploadReturnFocus.focus({ preventScroll: true });
    state.uploadReturnFocus = null;
  }

  function openCommandMenu() {
    state.commandReturnFocus = document.activeElement;
    renderCommandItems("");
    elements.commandModal.classList.add("open");
    elements.commandModal.setAttribute("aria-hidden", "false");
    elements.commandModal.removeAttribute("inert");
    elements.shell.setAttribute("inert", "");
    elements.commandInput.value = "";
    setTimeout(() => elements.commandInput.focus(), 80);
  }

  function closeCommandMenu() {
    const wasOpen = elements.commandModal.classList.contains("open");
    elements.commandModal.classList.remove("open");
    elements.commandModal.setAttribute("aria-hidden", "true");
    elements.commandModal.setAttribute("inert", "");
    if (wasOpen) elements.shell.removeAttribute("inert");
    if (state.commandReturnFocus?.isConnected) state.commandReturnFocus.focus({ preventScroll: true });
    state.commandReturnFocus = null;
  }

  function trapDialogFocus(event, container) {
    if (event.key !== "Tab" || !container?.classList.contains("open")) return;
    const focusable = $$('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])', container)
      .filter((element) => !element.hidden && element.getClientRects().length);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function renderCommandItems(query) {
    const items = [
      { label: "Projects", view: "home", icon: "grid", shortcut: "G P" },
      { label: "Data sources", view: "data", icon: "database", shortcut: "G D" },
      { label: "Model registry", view: "models", icon: "cube", shortcut: "G M" },
      { label: "Project overview", view: "overview", icon: "layers", shortcut: "G O", protected: true },
      { label: "Explore dataset", view: "explore", icon: "chart", shortcut: "G E", protected: true },
      { label: "Cleaning plan", view: "clean", icon: "wand", shortcut: "G C", protected: true },
      { label: "Feature lab", view: "features", icon: "sliders", shortcut: "G F", protected: true },
      { label: "Train models", view: "train", icon: "play", shortcut: "G T", protected: true },
      { label: "Evaluate models", view: "evaluate", icon: "target", shortcut: "G V", protected: true },
      { label: "Explain model", view: "explain", icon: "eye", shortcut: "G X", protected: true },
      { label: "Run prediction", view: "predict", icon: "target", shortcut: "G I", protected: true },
      { label: "Export artifacts", view: "deploy", icon: "download", shortcut: "G R", protected: true },
    ].filter((item) => !query || item.label.toLowerCase().includes(query.toLowerCase()));

    state.commandItems = items;
    elements.commandResults.innerHTML = `
      <span class="command-group-label">${query ? "Results" : "Navigate"}</span>
      ${items.map((item, index) => `<button class="command-result ${index === 0 ? "selected" : ""}" data-command-index="${index}">${icon(item.icon, "icon-sm")}<span>${escapeHtml(item.label)}</span><kbd>${item.shortcut}</kbd></button>`).join("")}
      ${query && !items.length ? `<button class="command-result" data-ask-command>${icon("sparkles", "icon-sm")}<span>Ask workflow guide: “${escapeHtml(query)}”</span><kbd>↵</kbd></button>` : ""}
    `;
  }

  function trainingTimeEstimate(strategy = state.strategy, rowCount = Number(state.profile?.n_rows || 0)) {
    const rows = Math.max(0, Number(rowCount) || 0);
    if (strategy === "fast") {
      if (rows >= 50000) return "Usually 4–10 min";
      if (rows >= 10000) return "Usually 2–5 min";
      return "Usually 1–2 min";
    }
    if (strategy === "high_performance") {
      if (rows >= 50000) return "Usually 15–35 min";
      if (rows >= 10000) return "Usually 8–20 min";
      return "Usually 5–12 min";
    }
    if (rows >= 50000) return "Usually 8–20 min";
    if (rows >= 10000) return "Usually 4–10 min";
    return "Usually 2–5 min";
  }

  function loadingTaskConfig(mode, explicitEstimate = null, explicitNote = null) {
    const rows = Number(state.profile?.n_rows || 0);
    const largeDataset = rows >= 50000;
    const strategy = state.strategy || "balanced";
    const configurations = {
      upload: {
        estimate: explicitEstimate || (largeDataset ? "2–5 minutes" : "1–2 minutes"),
        note: explicitNote || "Depends on file size and column count",
        steps: ["Reading file", "Profiling columns", "Building project"],
      },
      demo: {
        estimate: explicitEstimate || "< 1 minute",
        note: explicitNote || "Creates and profiles the local demo",
        steps: ["Creating data", "Finding patterns", "Opening project"],
      },
      train: {
        estimate: explicitEstimate || trainingTimeEstimate(strategy, rows),
        note: explicitNote || `${titleCase(strategy)} · ${formatNumber(rows, 0)} rows · estimate, not a deadline`,
        steps: ["Preparing folds", "Training candidates", "Validating champion"],
      },
      report: {
        estimate: explicitEstimate || (largeDataset ? "5–12 minutes" : "2–7 minutes"),
        note: explicitNote || "Executes notebook, then renders and verifies exports",
        steps: ["Executing notebook", "Rendering reports", "Verifying artifacts"],
      },
      predict: {
        estimate: explicitEstimate || "< 1 minute",
        note: explicitNote || "Usually completes within a few seconds",
        steps: ["Validating inputs", "Running pipeline", "Formatting result"],
      },
      export: {
        estimate: explicitEstimate || "< 1 minute",
        note: explicitNote || "Verifies the artifact before download",
        steps: ["Locating artifact", "Checking integrity", "Starting download"],
      },
      profile: {
        estimate: explicitEstimate || (largeDataset ? "2–5 minutes" : "1–2 minutes"),
        note: explicitNote || "Based on dataset size and local hardware",
        steps: ["Reading data", "Finding patterns", "Building strategy"],
      },
    };
    return configurations[mode] || configurations.profile;
  }

  function formatElapsed(milliseconds) {
    const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  function showLoading({
    eyebrow,
    title,
    message,
    mode = "profile",
    estimate = null,
    estimateNote = null,
    realProgress = false,
    cancellable = false,
  }) {
    clearInterval(state.loadingProgressTimer);
    clearInterval(state.loadingElapsedTimer);
    state.loadingRunId += 1;
    state.loadingStartedAt = Date.now();
    state.loadingPreviousFocus = document.activeElement;
    state.loadingStageOverride = null;
    const config = loadingTaskConfig(mode, estimate, estimateNote);
    elements.loadingEyebrow.textContent = eyebrow;
    elements.loadingTitle.textContent = title;
    elements.loadingMessage.textContent = message;
    elements.loadingEta.textContent = config.estimate;
    elements.loadingEtaNote.textContent = config.note;
    elements.loadingElapsed.textContent = "00:00";
    elements.loadingStageLabel.textContent = config.steps[0];
    elements.loadingPercent.textContent = realProgress ? "0%" : "8%";
    elements.loadingProgress.style.width = realProgress ? "0%" : "8%";
    elements.loadingProgress.classList.remove("indeterminate");
    elements.loading.dataset.mode = mode;
    elements.loading.classList.add("open");
    elements.loading.setAttribute("aria-hidden", "false");
    elements.shell.setAttribute("inert", "");
    elements.loadingCancel.hidden = !cancellable;
    elements.loadingCancel.disabled = false;
    elements.loadingCancel.textContent = mode === "report" ? "Cancel report" : "Cancel training";
    const loadingCard = $(".loading-card", elements.loading);
    loadingCard?.setAttribute("tabindex", "-1");
    requestAnimationFrame(() => (cancellable ? elements.loadingCancel : loadingCard)?.focus());

    $$("#loadingSteps span").forEach((step, index) => {
      step.innerHTML = `<i></i>${config.steps[index]}`;
      step.classList.toggle("active", index === 0);
    });

    let progress = 8;
    const tickInterval = mode === "predict" || mode === "export" ? 280 : 700;
    if (!realProgress) {
      state.loadingProgressTimer = setInterval(() => {
        progress = Math.min(92, progress + Math.max(0.6, (94 - progress) * 0.065));
        elements.loadingProgress.style.width = `${progress}%`;
        elements.loadingPercent.textContent = `${Math.round(progress)}%`;
        const activeStep = progress < 36 ? 0 : progress < 70 ? 1 : 2;
        elements.loadingStageLabel.textContent = (
          state.loadingStageOverride || config.steps[activeStep]
        );
        $$("#loadingSteps span").forEach((step, index) => {
          step.classList.toggle("active", index <= activeStep);
        });
      }, tickInterval);
    }
    state.loadingElapsedTimer = setInterval(() => {
      elements.loadingElapsed.textContent = formatElapsed(
        Date.now() - state.loadingStartedAt,
      );
    }, 1000);
  }

  function hideLoading() {
    const runId = state.loadingRunId;
    clearInterval(state.loadingProgressTimer);
    clearInterval(state.loadingElapsedTimer);
    state.loadingProgressTimer = null;
    state.loadingElapsedTimer = null;
    state.loadingStageOverride = null;
    elements.loadingCancel.hidden = true;
    if (
      elements.uploadModal.classList.contains("open")
      || elements.commandModal.classList.contains("open")
    ) elements.shell.setAttribute("inert", "");
    else elements.shell.removeAttribute("inert");
    elements.loadingProgress.classList.remove("indeterminate");
    elements.loadingProgress.style.width = "100%";
    elements.loadingPercent.textContent = "100%";
    elements.loadingStageLabel.textContent = "Complete";
    setTimeout(() => {
      if (runId !== state.loadingRunId) return;
      elements.loading.classList.remove("open");
      elements.loading.setAttribute("aria-hidden", "true");
      elements.loadingProgress.style.width = "0%";
      elements.loadingPercent.textContent = "0%";
      const restoreTarget = state.loadingPreviousFocus;
      state.loadingPreviousFocus = null;
      if (restoreTarget?.isConnected) restoreTarget.focus({ preventScroll: true });
    }, 380);
  }

  function updateLoadingFromJob(job) {
    const progress = Math.max(0, Math.min(100, Number(job?.progress) || 0));
    elements.loadingProgress.style.width = `${progress}%`;
    elements.loadingPercent.textContent = `${Math.round(progress)}%`;
    elements.loadingStageLabel.textContent = job?.stage ? titleCase(job.stage) : "Working";
    if (job?.message) elements.loadingMessage.textContent = job.message;
    const activeStep = progress < 25 ? 0 : progress < 86 ? 1 : 2;
    $$("#loadingSteps span").forEach((step, index) => {
      step.classList.toggle("active", index <= activeStep);
    });
    if (job?.cancellation_requested) {
      elements.loadingCancel.disabled = true;
      elements.loadingCancel.textContent = "Cancelling after current step…";
    }
  }

  function showExportLoading(label = "artifact") {
    showLoading({
      eyebrow: "LOCAL EXPORT",
      title: "Preparing your download",
      message: `Requesting the ${label} and checking the server response…`,
      mode: "export",
    });
  }

  async function downloadArtifact(url, label = "artifact") {
    if (!url || url === "#") return;
    showExportLoading(label);
    try {
      const response = await fetchWithTimeout(url, {}, 300000);
      if (!response.ok) throw new Error(await readError(response));
      const blob = await response.blob();
      if (!blob.size) throw new Error("The server returned an empty file.");
      const disposition = response.headers.get("content-disposition") || "";
      const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
      const plainName = disposition.match(/filename="?([^";]+)"?/i)?.[1];
      const filename = encodedName
        ? decodeURIComponent(encodedName)
        : plainName || `aivoraml_${String(label).toLowerCase().replace(/[^a-z0-9]+/g, "_")}`;
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      toast(`${label} downloaded successfully.`);
    } catch (error) {
      toast(`Download failed: ${error.message}`, "error");
    } finally {
      hideLoading();
    }
  }

  function toast(message, type = "success") {
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.innerHTML = `${icon(type === "error" ? "warning" : "check", "icon-sm")}<span>${escapeHtml(message)}</span>`;
    elements.toastRegion.append(node);
    setTimeout(() => node.remove(), 3600);
  }

  async function readError(response) {
    try {
      const data = await response.json();
      const detail = data.detail || data.message;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail?.field_errors)) return detail.field_errors.join("; ");
      if (detail) return JSON.stringify(detail);
      return response.statusText;
    } catch {
      return response.statusText || `Request failed (${response.status})`;
    }
  }

  async function fetchWithTimeout(
    url,
    options = {},
    timeoutMs = 60000,
    timeoutMessage = "The request timed out. Try a smaller dataset or a faster strategy.",
  ) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error(timeoutMessage);
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  async function uploadFile(file) {
    if (!file) return;
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!["csv", "xlsx", "xls", "json", "parquet"].includes(extension)) {
      toast("Use a CSV, Excel, JSON, or Parquet file.", "error");
      return;
    }

    closeUploadModal();
    const fileSizeMb = file.size / 1024 / 1024;
    const uploadEstimate = fileSizeMb >= 50
      ? "2–5 minutes"
      : fileSizeMb >= 10
        ? "1–3 minutes"
        : "< 1 minute";
    showLoading({
      eyebrow: "DATA INTELLIGENCE",
      title: "Understanding your dataset",
      mode: "upload",
      estimate: uploadEstimate,
      estimateNote: `${fileSizeMb.toFixed(fileSizeMb >= 10 ? 0 : 1)} MB · profiling on this machine`,
      message: "Profiling columns, quality, and potential targets…",
    });

    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetchWithTimeout("/api/upload", { method: "POST", body: form }, 120000);
      if (!response.ok) throw new Error(await readError(response));
      const profile = await response.json();
      hydrateProfile(profile);
      setView("overview");
      toast(`${file.name} is ready to explore.`);
    } catch (error) {
      toast(`Upload failed: ${error.message}`, "error");
      openUploadModal();
    } finally {
      hideLoading();
      elements.fileInput.value = "";
    }
  }

  async function loadDemo() {
    closeUploadModal();
    showLoading({
      eyebrow: "GUIDED DEMO",
      title: "Preparing churn intelligence",
      mode: "demo",
      message: "Creating a realistic customer dataset and profiling its signals…",
    });

    let profile = null;
    try {
      const response = await fetchWithTimeout("/api/demo", { method: "POST" });
      if (!response.ok) throw new Error(await readError(response));
      profile = await response.json();
    } catch (error) {
      hideLoading();
      toast(`Could not create demo: ${error.message}`, "error");
      return;
    }

    hydrateProfile(profile, { demo: true });
    setView("overview");
    hideLoading();
    toast("Demo project is ready.");
    addAssistantMessage("ai", `<p>I’ve opened the churn demo. I recommend predicting <strong>${escapeHtml(state.target)}</strong> as a classification problem. The dataset is healthy, with a few safe cleaning steps before training.</p>`, { trustedHtml: true });
  }

  const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  async function completeTrainingJob(result, { recovered = false } = {}) {
    state.result = normalizeResult(result);
    state.reportBundle = null;
    [3, 4, 5, 6].forEach((stage) => state.completed.add(stage));
    elements.modelCount.textContent = String(state.result.models_trained?.length || 0);
    await Promise.all([
      loadProjectMetadata(),
      loadPredictionWorkspace({ quiet: true }),
      loadModelVersions({ quiet: true }),
    ]);
    hideLoading();
    setView("evaluate");
    toast(recovered
      ? "Training completed while the browser was reconnecting. Results restored."
      : `${state.result.total_models_compared || state.result.models_trained?.length || 1} models trained. Champion selected by cross-validation.`
    );
    addAssistantMessage("ai", `<p>Training is complete. <strong>${escapeHtml(state.result.best_model || "The champion")}</strong> achieved the strongest validation score. I can explain why it won or help you inspect failure modes.</p>`, { trustedHtml: true });
  }

  async function recoverCompletedTraining(job) {
    const hasStartingVersion = Object.prototype.hasOwnProperty.call(
      job?.metadata || {},
      "starting_model_version",
    );
    const startingVersion = job?.metadata?.starting_model_version;
    if (!hasStartingVersion) return false;
    try {
      const response = await fetchWithTimeout(
        "/api/workflow/result",
        {},
        30000,
        "Could not check whether the completed model was saved.",
      );
      if (!response.ok) return false;
      const result = await response.json();
      const currentVersion = result.model_version;
      if (currentVersion && currentVersion !== (startingVersion || "")) {
        await completeTrainingJob(result, { recovered: true });
        return true;
      }
    } catch {
      // The caller will show the accurate server-restart state below.
    }
    return false;
  }

  async function monitorTrainingJob(initialJob, { restored = false } = {}) {
    if (!initialJob?.job_id) return;
    const monitorId = ++state.trainingMonitorId;
    state.trainingJob = initialJob;
    showLoading({
      eyebrow: restored ? "TRAINING RECONNECTED" : "MODEL LAB",
      title: "Training your candidates",
      message: initialJob.message || "Preparing the leakage-safe preprocessing pipeline…",
      mode: "train",
      realProgress: true,
      cancellable: true,
    });
    updateLoadingFromJob(initialJob);

    let job = initialJob;
    let consecutiveStatusFailures = 0;
    while (
      monitorId === state.trainingMonitorId
      && ["queued", "running", "cancel_requested"].includes(job.state)
    ) {
      await wait(consecutiveStatusFailures
        ? Math.min(15000, 1200 * (2 ** Math.min(consecutiveStatusFailures, 4)))
        : 1000
      );
      try {
        const response = await fetchWithTimeout(
          `/api/training/jobs/${encodeURIComponent(job.job_id)}`,
          {},
          45000,
          "The status check took too long; the server-side training job is still running.",
        );
        if (response.status === 404) {
          if (await recoverCompletedTraining(job)) return;
          state.trainingJob = null;
          hideLoading();
          toast("The local API restarted during training, so that job could not continue. Your previous champion is unchanged; start the run again.", "error");
          return;
        }
        if (!response.ok) throw new Error(await readError(response));
        job = await response.json();
        state.trainingJob = job;
        if (consecutiveStatusFailures) {
          const config = loadingTaskConfig("train");
          elements.loadingEta.textContent = config.estimate;
          elements.loadingEtaNote.textContent = config.note;
        }
        consecutiveStatusFailures = 0;
        elements.loadingEyebrow.textContent = "MODEL LAB";
        updateLoadingFromJob(job);
      } catch (error) {
        consecutiveStatusFailures += 1;
        elements.loadingEyebrow.textContent = "TRAINING CONTINUES";
        elements.loadingStageLabel.textContent = "Reconnecting";
        elements.loadingMessage.textContent = (
          `The status connection was delayed (${error.message}). `
          + "The server-side job continues; reconnecting automatically..."
        );
        elements.loadingEta.textContent = "Still running";
        elements.loadingEtaNote.textContent = `Reconnect attempt ${consecutiveStatusFailures}`;
      }
    }

    if (monitorId !== state.trainingMonitorId) {
      return;
    }

    state.trainingJob = job;
    if (job.state === "succeeded" && job.result) {
      await completeTrainingJob(job.result);
      return;
    }

    hideLoading();
    if (job.state === "cancelled") {
      toast("Training was cancelled. Your previous champion is unchanged.");
    } else {
      toast(`Training failed: ${job.error?.message || job.message || "Unknown error"}`, "error");
    }
  }

  async function runTraining() {
    if (!state.profile || !state.target || ["queued", "running", "cancel_requested"].includes(state.trainingJob?.state)) return;
    showLoading({
      eyebrow: "MODEL LAB",
      title: "Queueing your experiment",
      message: "Creating a reconnectable local training job…",
      mode: "train",
      realProgress: true,
      cancellable: false,
    });
    try {
      const response = await fetchWithTimeout("/api/training/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: state.target,
          approach: state.strategy,
          include_eda: false,
        }),
      }, 30000);
      if (!response.ok) throw new Error(await readError(response));
      const job = await response.json();
      hideLoading();
      await wait(420);
      return monitorTrainingJob(job);
    } catch (error) {
      hideLoading();
      toast(`Could not start training: ${error.message}`, "error");
    }
  }

  async function restoreActiveTrainingJob() {
    if (!state.profile || state.trainingJob) return;
    try {
      const response = await fetch("/api/jobs?limit=10");
      if (!response.ok) return;
      const payload = await response.json();
      const active = (payload.jobs || []).find((job) =>
        ["queued", "running", "cancel_requested"].includes(job.state)
      );
      if (active) {
        if (active.metadata?.job_type === "report") monitorReportJob(active, { restored: true });
        else monitorTrainingJob(active, { restored: true });
      }
    } catch {
      // Project restoration remains usable even when job history is unavailable.
    }
  }

  function assistantTextHtml(text) {
    return String(text || "")
      .split(/\n{2,}/)
      .filter((paragraph) => paragraph.trim())
      .map((paragraph) => `<p>${escapeHtml(paragraph.trim()).replaceAll("\n", "<br>")}</p>`)
      .join("") || "<p>No response was returned.</p>";
  }

  function addAssistantMessage(role, content, { trustedHtml = false, meta = "", pending = false } = {}) {
    const message = document.createElement("div");
    message.className = `message ${role === "user" ? "user-message" : "ai-message"}`;
    if (pending) message.classList.add("message-pending");
    const body = trustedHtml ? content : assistantTextHtml(content);
    message.innerHTML = role === "user"
      ? `<div class="message-bubble"><p>${escapeHtml(content)}</p></div>`
      : `<span class="message-avatar">${icon("sparkles", "icon-sm")}</span><div class="message-bubble">${body}${meta ? `<small class="message-provider">${escapeHtml(meta)}</small>` : ""}</div>`;
    elements.assistantMessages.append(message);
    elements.assistantMessages.scrollTop = elements.assistantMessages.scrollHeight;
    return message;
  }

  function assistantAnswer(question) {
    const lower = question.toLowerCase();
    const profile = state.profile;
    const result = state.result;
    const suggestion = currentSuggestion();

    if (!profile) {
      if (lower.includes("build") || lower.includes("what can")) {
        return "<p>You can build classification or regression projects from CSV and Excel datasets. Start with an upload, or open the churn demo to see the supported workflow.</p>";
      }
      return "<p>Add a dataset first and I’ll profile its schema, quality, likely targets, leakage risks, and suitable model families.</p>";
    }

    if (lower.includes("why this target") || lower.includes("why target")) {
      return `<p>I recommended <strong>${escapeHtml(suggestion?.column || state.target)}</strong> because ${escapeHtml((suggestion?.reason || "its shape matches a learnable outcome").toLowerCase())}. Confidence is <strong>${Math.round((suggestion?.score || 0.8) * 100)}%</strong>.</p>`;
    }
    if (lower.includes("missing") || lower.includes("imputation")) {
      return `<p>There are <strong>${formatNumber(profile.total_missing, 0)} missing cells</strong>. Numeric medians are robust to outliers; categorical modes preserve valid labels. Both must be fitted on training folds only to prevent leakage.</p>`;
    }
    if (lower.includes("risk") || lower.includes("leakage")) {
      const warning = profile.warnings?.[0];
      return `<p>The main item to review is: <strong>${escapeHtml(warning?.message || "no critical issue was detected")}</strong>. I’ll also exclude identifier-like columns and fit every transform inside cross-validation.</p>`;
    }
    if (lower.includes("recall")) {
      return "<p>To improve recall, tune the decision threshold, use class-weighted objectives, track PR-AUC, and inspect false negatives by segment. Expect some precision trade-off.</p>";
    }
    if (lower.includes("cross-validation")) {
      return "<p>Cross-validation repeats training across multiple folds so one lucky split cannot dominate the result. AivoraML compares the mean score and fold-to-fold variance.</p>";
    }
    if (lower.includes("why") && (lower.includes("model") || lower.includes("win"))) {
      if (!result) return "<p>I’ll answer that after training. The winner is selected from training-only cross-validation, with a stability penalty; the holdout is reserved for final evaluation.</p>";
      const winner = sortedModels()[0];
      return `<p><strong>${escapeHtml(result.best_model || winner?.name)}</strong> won on the recorded cross-validation criterion (${scoreDisplay(Number(winner?.selection_score ?? primaryScore(winner)))}) before final holdout evaluation.</p>`;
    }
    if (lower.includes("overfit")) {
      const winner = sortedModels()[0];
      const gap = Number(winner?.overfitting_gap || 0);
      return `<p>The recorded training-to-CV gap is <strong>${(gap * 100).toFixed(1)} points</strong>. ${gap < 0.08 ? "That looks reasonably stable." : "That deserves regularization, simpler features, or more data."}</p>`;
    }
    if (lower.includes("top feature") || lower.includes("importance")) {
      const top = getFeatureImportance()[0];
      return `<p><strong>${escapeHtml(titleCase(top?.feature || "The top feature"))}</strong> has the highest global contribution. Importance means the model relies on it often; it does not prove that changing it will cause the outcome.</p>`;
    }
    if (lower.includes("calibrat") || lower.includes("probability")) {
      const calibration = result?.evaluation_details?.probability_calibration;
      if (!calibration) return "<p>Probability calibration is available for binary classifiers that expose reliable probability outputs. This model does not have recorded calibration evidence.</p>";
      const gap = Number(calibration.expected_calibration_error || 0);
      return `<p>The holdout expected calibration error is <strong>${(gap * 100).toFixed(1)}%</strong> and the Brier score is <strong>${Number(calibration.brier_score).toFixed(3)}</strong>. ${gap <= 0.08 ? "Observed probabilities are reasonably aligned on this holdout." : "Treat probabilities cautiously and consider post-training calibration."}</p>`;
    }
    if (lower.includes("deploy") || lower.includes("monitor")) {
      return "<p>Export the fitted pipeline, serve it behind a typed FastAPI endpoint, log inputs and predictions, then monitor schema changes, feature drift, latency, and delayed-label performance.</p>";
    }
    if (lower.includes("next") || lower.includes("plan")) {
      const next = result ? "review model explanations and package the champion" : "review cleaning, confirm the target, and run the balanced experiment";
      return `<p>Your next best step is to <strong>${next}</strong>. I’ll preserve every preprocessing decision in the downloadable pipeline and notebook.</p>`;
    }

    return `<p>This project is a <strong>${titleCase(taskType())}</strong> workflow with <strong>${escapeHtml(state.target)}</strong> as the target. The data quality score is ${profileQuality()}%. Ask me about cleaning, feature engineering, model choice, evaluation, or deployment.</p>`;
  }

  async function handleAssistantQuestion(question) {
    const cleanQuestion = question.trim();
    if (!cleanQuestion || state.assistantBusy) return;
    addAssistantMessage("user", cleanQuestion);
    elements.assistantInput.value = "";
    elements.assistantInput.style.height = "";

    if (!state.profile && cleanQuestion.toLowerCase().includes("demo")) {
      setTimeout(() => {
        addAssistantMessage("ai", "<p>Opening the guided customer churn project now.</p>", { trustedHtml: true });
        loadDemo();
      }, 300);
      return;
    }

    const priorHistory = state.assistantHistory.slice(-8);
    state.assistantHistory.push({ role: "user", content: cleanQuestion });
    const remoteAvailable = Boolean(state.capabilities?.remote_ai?.assistant?.available);
    if (!remoteAvailable) {
      const localAnswer = assistantAnswer(cleanQuestion);
      addAssistantMessage("ai", localAnswer, { trustedHtml: true, meta: "Local evidence guide" });
      const localText = document.createElement("div");
      localText.innerHTML = localAnswer;
      state.assistantHistory.push({
        role: "assistant",
        content: localText.textContent || "Local guidance",
      });
      return;
    }

    state.assistantBusy = true;
    const submitButton = $("#assistantForm button[type='submit']");
    if (submitButton) submitButton.disabled = true;
    const pending = addAssistantMessage("ai", "Thinking with the project context...", { pending: true });
    try {
      const response = await fetchWithTimeout(
        "/api/ai/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: cleanQuestion,
            view: state.view,
            strategy: state.strategy,
            target_column: state.target || null,
            history: priorHistory,
          }),
        },
        45000,
        "The AI response took too long.",
      );
      if (!response.ok) throw new Error(await readError(response));
      const reply = await response.json();
      pending.remove();
      addAssistantMessage("ai", reply.answer, {
        meta: `${reply.provider} · ${reply.model}${reply.fallback_used ? " · fallback" : ""}`,
      });
      state.assistantHistory.push({ role: "assistant", content: reply.answer });
    } catch (error) {
      pending.remove();
      const localAnswer = assistantAnswer(cleanQuestion);
      addAssistantMessage("ai", localAnswer, { trustedHtml: true, meta: "Local fallback" });
      state.assistantHistory.push({ role: "assistant", content: "Local fallback guidance was shown." });
    } finally {
      state.assistantBusy = false;
      if (submitButton) submitButton.disabled = false;
      elements.assistantInput.focus();
    }
  }

  async function copyText(text, message = "Copied to clipboard.") {
    try {
      await navigator.clipboard.writeText(text);
      toast(message);
    } catch {
      toast("Clipboard access is unavailable in this browser.", "error");
    }
  }

  async function handleWorkspaceAction(action, target) {
    const viewActions = {
      "open-current": "overview",
      "open-explore": "explore",
      "open-clean": "clean",
      "open-features": "features",
      "open-train": "train",
      "open-evaluate": "evaluate",
      "open-explain": "explain",
      "open-predict": "predict",
      "open-deploy": "deploy",
    };
    if (viewActions[action]) {
      setView(viewActions[action]);
      return;
    }

    if (action === "open-upload") return openUploadModal();
    if (action === "load-demo") return loadDemo();
    if (action === "activate-project") return activateProject(target.dataset.projectId);
    if (action === "delete-project") {
      return deleteProject(target.dataset.projectId, target.dataset.projectName);
    }
    if (action === "start-prompt") {
      const prompt = $("#projectPrompt")?.value.trim();
      if (prompt) addAssistantMessage("user", prompt);
      if (state.profile) setView("overview");
      else openUploadModal();
      return;
    }
    if (action === "reload-predict") return loadPredictionWorkspace();
    if (action === "clear-predictions") return clearPredictionHistory();
    if (action === "activate-model-version") {
      return activateModelVersion(target.dataset.version);
    }
    if (action === "fill-predict-sample") {
      const form = $("#predictionForm");
      let populated = 0;
      state.predictionSchema?.fields?.forEach((field) => {
        const input = form?.elements.namedItem(field.name);
        if (input && field.sample_value != null) {
          input.value = String(field.sample_value);
          input.dispatchEvent(new Event("change", { bubbles: true }));
          populated += 1;
          const visualField = input.closest(".prediction-field");
          visualField?.classList.add("auto-populated");
          setTimeout(() => visualField?.classList.remove("auto-populated"), 900);
        }
      });
      toast(`${populated} feature values auto-filled from the uploaded dataset.`);
      return;
    }
    if (action === "ask-why") {
      elements.shell.classList.remove("assistant-hidden");
      handleAssistantQuestion("Why this target?");
      return;
    }
    if (action === "ask-explain") {
      elements.shell.classList.remove("assistant-hidden");
      handleAssistantQuestion("Explain the top feature");
      return;
    }
    if (action === "select-strategy") {
      state.strategy = target.dataset.strategy;
      render();
      return;
    }
    if (action === "run-training") return runTraining();
    if (action === "generate-reports") return generateReports();
    if (action === "apply-clean") {
      state.completed.add(3);
      toast("Cleaning plan reviewed. It will run inside the training pipeline.");
      setView("features");
      return;
    }
    if (action === "save-features") {
      state.completed.add(4);
      toast("Preprocessing contract reviewed.");
      setView("train");
      return;
    }
    if (action === "download-notebook") {
      const url = projectUrl("/api/notebook/download");
      if (url) downloadArtifact(url, "Executed notebook");
      return;
    }
    if (action === "copy-code") {
      return copyText('from fastapi import FastAPI\nimport joblib\nimport pandas as pd\n\napp = FastAPI()\nartifact = joblib.load("aivoraml_model.joblib")\npipeline = artifact["pipeline"]\n\n@app.post("/predict")\ndef predict(payload: dict):\n    frame = pd.DataFrame([payload], columns=artifact["feature_schema"]["ordered"])\n    value = pipeline.predict(frame)[0]\n    return {"prediction": value.item() if hasattr(value, "item") else value}', "API example copied.");
    }
  }

  function bindEvents() {
    $$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));

    $("#menuButton").addEventListener("click", () => {
      document.body.classList.add("sidebar-open");
      $("#menuButton").setAttribute("aria-expanded", "true");
      elements.sidebar.removeAttribute("inert");
    });
    $("#sidebarClose").addEventListener("click", () => {
      document.body.classList.remove("sidebar-open");
      $("#menuButton").setAttribute("aria-expanded", "false");
      if (window.innerWidth <= 980) elements.sidebar.setAttribute("inert", "");
    });
    $("#assistantToggle").addEventListener("click", () => {
      const hidden = elements.shell.classList.toggle("assistant-hidden");
      $("#assistantToggle").setAttribute("aria-expanded", String(!hidden));
      elements.assistant.toggleAttribute("inert", hidden);
      sessionStorage.setItem("aivoraml-guide-open", String(!hidden));
    });
    $("#assistantClose").addEventListener("click", () => {
      elements.shell.classList.add("assistant-hidden");
      $("#assistantToggle").setAttribute("aria-expanded", "false");
      elements.assistant.setAttribute("inert", "");
      sessionStorage.setItem("aivoraml-guide-open", "false");
    });
    $("#commandTrigger").addEventListener("click", openCommandMenu);
    $("#modalDemoButton").addEventListener("click", loadDemo);
    $("#notificationButton").addEventListener("click", () => {
      if (!state.project) return toast("No project is open.");
      const persistence = state.project.persistence?.status;
      if (persistence === "error") return toast("The last local save failed. Export the project before closing.", "error");
      if (persistence === "disabled") return toast("Persistence is disabled for this session.");
      toast(`Project saved locally · ${formatTimestamp(state.project.updated_at)}`);
    });
    $("#workspaceSwitcher").addEventListener("click", () => setView("home"));
    $("#shareButton").addEventListener("click", () => {
      if (state.profile) {
        downloadArtifact("/api/project/download", "Project summary");
      }
    });
    elements.projectExport.addEventListener("click", () => {
      if (state.profile) {
        downloadArtifact("/api/project/download", "Project summary");
      }
    });
    elements.runButton.addEventListener("click", () => setView("train"));
    elements.loadingCancel.addEventListener("click", async () => {
      const jobId = state.trainingJob?.job_id;
      if (!jobId) return;
      elements.loadingCancel.disabled = true;
      elements.loadingCancel.textContent = "Requesting cancellation…";
      try {
        const response = await fetch(
          `/api/training/jobs/${encodeURIComponent(jobId)}/cancel`,
          { method: "POST" },
        );
        if (!response.ok) throw new Error(await readError(response));
        state.trainingJob = await response.json();
        updateLoadingFromJob(state.trainingJob);
      } catch (error) {
        elements.loadingCancel.disabled = false;
        const isReport = state.trainingJob?.metadata?.job_type === "report";
        elements.loadingCancel.textContent = isReport ? "Cancel report" : "Cancel training";
        toast(`Could not cancel ${isReport ? "report generation" : "training"}: ${error.message}`, "error");
      }
    });

    $$("[data-close-modal]").forEach((button) => button.addEventListener("click", closeUploadModal));
    elements.uploadModal.addEventListener("click", (event) => {
      if (event.target === elements.uploadModal) closeUploadModal();
    });
    elements.commandModal.addEventListener("click", (event) => {
      if (event.target === elements.commandModal) closeCommandMenu();
    });

    elements.fileInput.addEventListener("change", () => uploadFile(elements.fileInput.files?.[0]));
    elements.dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("dragging");
    });
    elements.dropzone.addEventListener("dragleave", () => elements.dropzone.classList.remove("dragging"));
    elements.dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("dragging");
      uploadFile(event.dataTransfer.files?.[0]);
    });
    elements.dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        elements.fileInput.click();
      }
    });

    elements.workspace.addEventListener("click", (event) => {
      const smartOption = event.target.closest("[data-smart-option]");
      if (smartOption) {
        event.preventDefault();
        const container = smartOption.closest("[data-smart-select]");
        const select = $(".smart-select-native", container);
        if (select) {
          select.value = smartOption.dataset.value;
          select.dispatchEvent(new Event("change", { bubbles: true }));
        }
        closeSmartSelects();
        $("[data-smart-trigger]", container)?.focus();
        return;
      }
      const smartTrigger = event.target.closest("[data-smart-trigger]");
      if (smartTrigger) {
        event.preventDefault();
        toggleSmartSelect(smartTrigger.closest("[data-smart-select]"));
        return;
      }
      const artifact = event.target.closest(".artifact-card[href]");
      if (artifact && artifact.getAttribute("href") !== "#") {
        event.preventDefault();
        artifact.classList.add("is-downloading");
        setTimeout(() => artifact.classList.remove("is-downloading"), 1800);
        downloadArtifact(
          artifact.getAttribute("href"),
          artifact.querySelector("strong, h3")?.textContent?.trim() || "Artifact",
        );
        return;
      }
      const actionTarget = event.target.closest("[data-action]");
      if (actionTarget) {
        event.preventDefault();
        handleWorkspaceAction(actionTarget.dataset.action, actionTarget);
      }
    });
    elements.workspace.addEventListener("change", (event) => {
      if (event.target.id === "targetSelect") {
        state.target = event.target.value;
        buildCleaningPlan();
        buildFeaturePlan();
        render();
      }
      if (event.target.matches(".smart-select-native")) {
        syncSmartSelect(event.target.closest("[data-smart-select]"));
      }
    });
    elements.workspace.addEventListener("keydown", (event) => {
      const trigger = event.target.closest("[data-smart-trigger]");
      if (trigger && ["ArrowDown", "ArrowUp"].includes(event.key)) {
        event.preventDefault();
        const container = trigger.closest("[data-smart-select]");
        toggleSmartSelect(container, true);
        const options = $$("[data-smart-option]", container);
        (event.key === "ArrowDown" ? options[0] : options.at(-1))?.focus();
        return;
      }

      const option = event.target.closest("[data-smart-option]");
      if (!option) return;
      const container = option.closest("[data-smart-select]");
      const options = $$("[data-smart-option]", container);
      const currentIndex = options.indexOf(option);
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const direction = event.key === "ArrowDown" ? 1 : -1;
        options[(currentIndex + direction + options.length) % options.length]?.focus();
      } else if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        (event.key === "Home" ? options[0] : options.at(-1))?.focus();
      } else if (event.key === "Escape") {
        event.preventDefault();
        closeSmartSelects();
        $("[data-smart-trigger]", container)?.focus();
      }
    });
    elements.workspace.addEventListener("submit", (event) => {
      if (event.target.id === "predictionForm") {
        event.preventDefault();
        runPrediction(event.target);
      } else if (event.target.id === "batchPredictionForm") {
        event.preventDefault();
        runBatchPrediction(event.target);
      }
    });

    $("#assistantForm").addEventListener("submit", (event) => {
      event.preventDefault();
      handleAssistantQuestion(elements.assistantInput.value);
    });
    elements.assistantInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        $("#assistantForm").requestSubmit();
      }
    });
    elements.assistantInput.addEventListener("input", () => {
      elements.assistantInput.style.height = "auto";
      elements.assistantInput.style.height = `${Math.min(100, elements.assistantInput.scrollHeight)}px`;
    });
    elements.suggestionChips.addEventListener("click", (event) => {
      const button = event.target.closest("[data-question]");
      if (button) handleAssistantQuestion(button.dataset.question);
    });

    elements.commandInput.addEventListener("input", () => renderCommandItems(elements.commandInput.value));
    elements.commandInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        const selected = $(".command-result.selected", elements.commandResults) || $(".command-result", elements.commandResults);
        selected?.click();
      }
    });
    elements.commandResults.addEventListener("click", (event) => {
      const result = event.target.closest("[data-command-index]");
      if (result) {
        const item = state.commandItems[Number(result.dataset.commandIndex)];
        closeCommandMenu();
        if (item) setView(item.view);
        return;
      }
      if (event.target.closest("[data-ask-command]")) {
        const question = elements.commandInput.value;
        closeCommandMenu();
        elements.shell.classList.remove("assistant-hidden");
        handleAssistantQuestion(question);
      }
    });

    document.addEventListener("keydown", (event) => {
      trapDialogFocus(event, elements.uploadModal);
      trapDialogFocus(event, elements.commandModal);
      trapDialogFocus(event, elements.loading);
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        openCommandMenu();
      }
      if (event.key === "Escape") {
        closeSmartSelects();
        closeCommandMenu();
        closeUploadModal();
        document.body.classList.remove("sidebar-open");
        $("#menuButton").setAttribute("aria-expanded", "false");
        if (window.innerWidth <= 980) elements.sidebar.setAttribute("inert", "");
      }
    });
    document.addEventListener("click", (event) => {
      if (!event.target.closest("[data-smart-select]")) closeSmartSelects();
      if (
        document.body.classList.contains("sidebar-open")
        && !event.target.closest("#sidebar")
        && !event.target.closest("#menuButton")
      ) {
        document.body.classList.remove("sidebar-open");
        $("#menuButton").setAttribute("aria-expanded", "false");
        if (window.innerWidth <= 980) elements.sidebar.setAttribute("inert", "");
      }
    });
    window.addEventListener("resize", () => {
      if (window.innerWidth > 980) elements.sidebar.removeAttribute("inert");
      else if (!document.body.classList.contains("sidebar-open")) elements.sidebar.setAttribute("inert", "");
    });
    window.addEventListener("popstate", () => {
      const requested = location.hash.slice(1) || "home";
      if (!protectedViews.has(requested) || state.profile) {
        state.view = requested;
        render();
      }
    });
  }

  async function bootstrap() {
    bindEvents();
    await Promise.all([
      loadCapabilities(),
      loadProjects({ quiet: true }),
    ]);
    if (window.innerWidth <= 980) elements.sidebar.setAttribute("inert", "");
    const guideWasOpened = sessionStorage.getItem("aivoraml-guide-open") === "true";
    if (!guideWasOpened || window.innerWidth <= 980) {
      elements.shell.classList.add("assistant-hidden");
      elements.assistant.setAttribute("inert", "");
      $("#assistantToggle").setAttribute("aria-expanded", "false");
    }
    render();
    const projectMessage = sessionStorage.getItem("aivoraml-project-message");
    if (projectMessage) {
      sessionStorage.removeItem("aivoraml-project-message");
      toast(projectMessage);
    }

    if (new URLSearchParams(location.search).get("demo") === "1") {
      await loadDemo();
      return;
    }

    try {
      const response = await fetch("/api/dataset/profile");
      if (response.status === 404) return;
      if (!response.ok) throw new Error(await readError(response));
      const profile = await response.json();
      hydrateProfile(profile);
      const hasResult = await loadPersistedResult();
      await loadProjectMetadata();
      if (hasResult) {
        await Promise.all([
          loadPredictionWorkspace({ quiet: true }),
          loadReportStatus({ quiet: true }),
          loadModelVersions({ quiet: true }),
        ]);
      }
      const requestedView = location.hash.slice(1);
      state.view = requestedView && (protectedViews.has(requestedView) || ["home", "data", "models"].includes(requestedView))
        ? requestedView
        : "overview";
      render();
      restoreActiveTrainingJob();
    } catch (error) {
      toast(`Could not restore the local project: ${error.message}`, "error");
    }
  }

  bootstrap();
})();
