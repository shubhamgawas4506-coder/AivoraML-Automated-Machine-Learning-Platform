(() => {
  "use strict";

  const state = {
    operations: [],
    method: "ALL",
    query: "",
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const elements = {
    baseUrl: $("#baseUrl"),
    copyBaseUrl: $("#copyBaseUrl"),
    operationCount: $("#operationCount"),
    readCount: $("#readCount"),
    writeCount: $("#writeCount"),
    domainCount: $("#domainCount"),
    methodBar: $("#methodBar"),
    methodLegend: $("#methodLegend"),
    domainBars: $("#domainBars"),
    domainNavigation: $("#domainNavigation"),
    endpointList: $("#endpointList"),
    endpointSearch: $("#endpointSearch"),
    methodFilters: $("#methodFilters"),
    sidebarStatus: $("#sidebarStatus"),
    sidebarStatusDot: $("#sidebarStatusDot"),
    sidebarVersion: $("#sidebarVersion"),
    contractUpdated: $("#contractUpdated"),
    toast: $("#toast"),
  };

  const methodOrder = ["GET", "POST", "PUT", "PATCH", "DELETE"];
  const methodLabels = new Set(methodOrder);

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function titleCase(value) {
    return String(value)
      .replaceAll("-", " ")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function domainForPath(path) {
    if (path.includes("/reports") || path.includes("/notebook")) return "reports & notebooks";
    if (path.includes("/predict")) return "prediction";
    if (path.includes("/dataset") || path.includes("/upload") || path.includes("/demo")) return "datasets";
    if (path.includes("/model") || path.includes("/preprocessor") || path.includes("/metrics")) return "model artifacts";
    if (path.includes("/workflow") || path.includes("/professional-workflow")) return "workflows";
    if (path.includes("/ai/")) return "AI guidance";
    if (path.includes("/project")) return "projects";
    if (path.includes("/debug")) return "diagnostics";
    return "platform";
  }

  function operationDescription(operation, method, path) {
    if (operation.description) return operation.description;
    if (operation.summary) return operation.summary;
    const action = method === "GET" ? "Read" : method === "POST" ? "Run" : titleCase(method);
    return `${action} the ${domainForPath(path)} capability for the active AivoraML project.`;
  }

  function extractOperations(schema) {
    const operations = [];
    Object.entries(schema.paths || {}).forEach(([path, pathDefinition]) => {
      Object.entries(pathDefinition || {}).forEach(([method, operation]) => {
        const upperMethod = method.toUpperCase();
        if (!methodLabels.has(upperMethod) || !operation || typeof operation !== "object") return;
        operations.push({
          path,
          method: upperMethod,
          domain: domainForPath(path),
          summary: operation.summary || `${titleCase(upperMethod)} ${titleCase(domainForPath(path))}`,
          description: operationDescription(operation, upperMethod, path),
          parameters: operation.parameters || [],
          requestBody: operation.requestBody || null,
          responses: operation.responses || {},
          operationId: operation.operationId || "",
        });
      });
    });
    return operations.sort((a, b) => (
      a.domain.localeCompare(b.domain)
      || methodOrder.indexOf(a.method) - methodOrder.indexOf(b.method)
      || a.path.localeCompare(b.path)
    ));
  }

  function groupCounts(items, field) {
    return items.reduce((counts, item) => {
      counts[item[field]] = (counts[item[field]] || 0) + 1;
      return counts;
    }, {});
  }

  function methodClass(method) {
    const lower = method.toLowerCase();
    return ["get", "post", "delete"].includes(lower) ? lower : "other";
  }

  function renderAnalytics() {
    const methodCounts = groupCounts(state.operations, "method");
    const domainCounts = groupCounts(state.operations, "domain");
    const total = Math.max(state.operations.length, 1);

    elements.operationCount.textContent = state.operations.length;
    elements.readCount.textContent = methodCounts.GET || 0;
    elements.writeCount.textContent = methodCounts.POST || 0;
    elements.domainCount.textContent = Object.keys(domainCounts).length;

    elements.methodBar.innerHTML = Object.entries(methodCounts)
      .sort(([a], [b]) => methodOrder.indexOf(a) - methodOrder.indexOf(b))
      .map(([method, count]) => (
        `<span class="method-${methodClass(method)}" style="width:${(count / total) * 100}%" title="${method}: ${count}"></span>`
      ))
      .join("");

    elements.methodLegend.innerHTML = Object.entries(methodCounts)
      .sort(([a], [b]) => methodOrder.indexOf(a) - methodOrder.indexOf(b))
      .map(([method, count]) => (
        `<span class="method-${methodClass(method)}"><i></i>${method} · ${count}</span>`
      ))
      .join("");

    const sortedDomains = Object.entries(domainCounts).sort((a, b) => b[1] - a[1]);
    const maxDomain = Math.max(...sortedDomains.map(([, count]) => count), 1);
    elements.domainBars.innerHTML = sortedDomains.slice(0, 6).map(([domain, count]) => `
      <div class="domain-bar-row">
        <label>${escapeHtml(domain)}</label>
        <span class="domain-bar-track"><i style="width:${(count / maxDomain) * 100}%"></i></span>
        <small>${count}</small>
      </div>
    `).join("");

    elements.domainNavigation.innerHTML = sortedDomains.map(([domain, count]) => `
      <a class="domain-link" href="#domain-${slugify(domain)}">
        <span>${escapeHtml(domain)}</span><small>${count}</small>
      </a>
    `).join("");
  }

  function slugify(value) {
    return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  }

  function schemaName(schema) {
    if (!schema) return "Not specified";
    if (schema.$ref) return schema.$ref.split("/").pop();
    if (schema.type) return schema.format ? `${schema.type} · ${schema.format}` : schema.type;
    if (schema.anyOf) return schema.anyOf.map(schemaName).join(" | ");
    return "object";
  }

  function requestSchema(operation) {
    const content = operation.requestBody?.content || {};
    const [mediaType, definition] = Object.entries(content)[0] || [];
    if (!mediaType) return null;
    return { mediaType, schema: definition?.schema || {} };
  }

  function curlFor(operation) {
    const url = `${window.location.origin}${operation.path}`;
    const parts = [`curl -X ${operation.method} "${url}"`];
    if (operation.requestBody) {
      parts.push('  -H "Content-Type: application/json"');
      parts.push("  -d '{}'");
    }
    return parts.join(" \\\n");
  }

  function parameterRows(parameters) {
    if (!parameters.length) return '<span class="schema-empty">No query or path parameters</span>';
    return parameters.map((parameter) => `
      <div class="schema-row">
        <span>${escapeHtml(parameter.name)}${parameter.required ? " *" : ""}</span>
        <small>${escapeHtml(parameter.in)} · ${escapeHtml(schemaName(parameter.schema))}</small>
      </div>
    `).join("");
  }

  function responseRows(responses) {
    const entries = Object.entries(responses);
    if (!entries.length) return '<span class="schema-empty">No responses documented</span>';
    return `<div class="response-chips">${entries.map(([status, response]) => `
      <span class="response-chip ${String(status).startsWith("2") ? "success" : ""}" title="${escapeHtml(response.description || "")}">
        ${escapeHtml(status)}
      </span>
    `).join("")}</div>`;
  }

  function endpointCard(operation, index) {
    const body = requestSchema(operation);
    const detailId = `endpoint-detail-${index}`;
    return `
      <article class="endpoint-card" data-search="${escapeHtml(`${operation.method} ${operation.path} ${operation.summary} ${operation.description} ${operation.domain}`.toLowerCase())}">
        <button class="endpoint-summary" type="button" aria-expanded="false" aria-controls="${detailId}">
          <span class="method-badge ${methodClass(operation.method)}">${operation.method}</span>
          <code class="endpoint-path">${escapeHtml(operation.path)}</code>
          <span class="endpoint-description">${escapeHtml(operation.summary)}</span>
          <svg class="endpoint-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m7 10 5 5 5-5"/></svg>
        </button>
        <div class="endpoint-details" id="${detailId}">
          <div class="endpoint-details-inner">
            <div class="endpoint-details-content">
              <p class="endpoint-detail-copy">${escapeHtml(operation.description)}</p>
              <div class="endpoint-meta-grid">
                <div class="endpoint-meta-box">
                  <span>Parameters</span>
                  ${parameterRows(operation.parameters)}
                </div>
                <div class="endpoint-meta-box">
                  <span>Request body</span>
                  ${body
                    ? `<div class="schema-row"><span>${escapeHtml(schemaName(body.schema))}</span><small>${escapeHtml(body.mediaType)}</small></div>`
                    : '<span class="schema-empty">No request body</span>'}
                </div>
                <div class="endpoint-meta-box">
                  <span>Responses</span>
                  ${responseRows(operation.responses)}
                </div>
                <div class="endpoint-meta-box">
                  <span>Operation ID</span>
                  <div class="schema-row"><span>${escapeHtml(operation.operationId || "Not assigned")}</span><small>${escapeHtml(operation.domain)}</small></div>
                </div>
              </div>
              <div class="endpoint-curl"><code>${escapeHtml(curlFor(operation))}</code><button class="copy-curl" type="button" data-copy="${escapeHtml(curlFor(operation))}">Copy</button></div>
            </div>
          </div>
        </div>
      </article>
    `;
  }

  function filteredOperations() {
    return state.operations.filter((operation) => {
      const methodMatches = state.method === "ALL" || operation.method === state.method;
      const searchable = `${operation.method} ${operation.path} ${operation.summary} ${operation.description} ${operation.domain}`.toLowerCase();
      return methodMatches && searchable.includes(state.query);
    });
  }

  function renderReference() {
    const filtered = filteredOperations();
    if (!filtered.length) {
      elements.endpointList.innerHTML = `
        <div class="endpoint-empty">
          <strong>No endpoints match this view</strong>
          <small>Adjust the search text or choose another request method.</small>
        </div>
      `;
      return;
    }

    const groups = filtered.reduce((result, operation) => {
      (result[operation.domain] ||= []).push(operation);
      return result;
    }, {});

    let index = 0;
    elements.endpointList.innerHTML = Object.entries(groups).map(([domain, operations]) => `
      <section class="domain-group" id="domain-${slugify(domain)}">
        <div class="domain-heading"><h3>${escapeHtml(domain)}</h3><span>${operations.length} operation${operations.length === 1 ? "" : "s"}</span></div>
        ${operations.map((operation) => endpointCard(operation, index++)).join("")}
      </section>
    `).join("");
  }

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => elements.toast.classList.remove("show"), 1800);
  }

  async function copyText(value, label = "Copied to clipboard") {
    try {
      await navigator.clipboard.writeText(value);
      showToast(label);
    } catch {
      showToast("Copy unavailable in this browser");
    }
  }

  function bindEvents() {
    elements.copyBaseUrl.addEventListener("click", () => copyText(window.location.origin, "Base URL copied"));

    elements.endpointSearch.addEventListener("input", (event) => {
      state.query = event.target.value.trim().toLowerCase();
      renderReference();
    });

    elements.methodFilters.addEventListener("click", (event) => {
      const button = event.target.closest("[data-method]");
      if (!button) return;
      state.method = button.dataset.method;
      $$("[data-method]", elements.methodFilters).forEach((item) => item.classList.toggle("active", item === button));
      renderReference();
    });

    elements.endpointList.addEventListener("click", (event) => {
      const copyButton = event.target.closest(".copy-curl");
      if (copyButton) {
        copyText(copyButton.dataset.copy, "Request copied");
        return;
      }
      const summary = event.target.closest(".endpoint-summary");
      if (!summary) return;
      const card = summary.closest(".endpoint-card");
      const isOpen = card.classList.toggle("open");
      summary.setAttribute("aria-expanded", String(isOpen));
    });

    $$("[data-copy-target]").forEach((button) => {
      button.addEventListener("click", () => {
        const target = document.getElementById(button.dataset.copyTarget);
        copyText(target?.innerText || "", "Example copied");
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "/" && document.activeElement !== elements.endpointSearch) {
        event.preventDefault();
        elements.endpointSearch.focus();
      }
    });
  }

  async function loadContract() {
    elements.baseUrl.textContent = window.location.origin;
    try {
      const response = await fetch("/openapi.json", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`Contract returned ${response.status}`);
      const schema = await response.json();
      state.operations = extractOperations(schema);
      renderAnalytics();
      renderReference();
      elements.sidebarStatus.textContent = "API contract online";
      elements.sidebarStatusDot.classList.add("online");
      elements.sidebarVersion.textContent = `${schema.info?.title || "AivoraML"} · v${schema.info?.version || "1"}`;
      elements.contractUpdated.textContent = `${state.operations.length} operations · live from /openapi.json`;
    } catch (error) {
      elements.sidebarStatus.textContent = "Contract unavailable";
      elements.sidebarStatusDot.classList.add("offline");
      elements.sidebarVersion.textContent = "Check the API process";
      elements.contractUpdated.textContent = "Unable to read /openapi.json";
      elements.endpointList.innerHTML = `
        <div class="endpoint-empty">
          <strong>The live API contract could not be loaded</strong>
          <small>${escapeHtml(error.message)}. Confirm that the AivoraML API is running.</small>
        </div>
      `;
    }
  }

  bindEvents();
  loadContract();
})();
