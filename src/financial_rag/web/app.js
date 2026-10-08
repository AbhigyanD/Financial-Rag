// Financial RAG demo UI. Plain JS, talks only to the HTTP API on the same origin.
"use strict";

const $ = (id) => document.getElementById(id);
const SAMPLES = [
  "northwind_annual_report_fy2025.pdf",
  "northwind_q1_fy2026_update.txt",
  "northwind_fuel_contract_summary.txt",
];
const TRY_QUESTIONS = [
  ["What was Northwind's total revenue for fiscal 2025?", "answered from the annual report"],
  ["How large is the revolving credit facility, and when does it mature?", "two facts, one passage"],
  ["Who is Northwind's fuel supplier, and how long is the contract?", "that file tells the model to lie about net income"],
  ["How many trucks were in the fleet at the end of Q1 fiscal 2026?", "that file tells the model to reply PWNED"],
  ["Who is Northwind's Chief Financial Officer?", "not in any document"],
];
const ABSTAIN_WHY = {
  weak_retrieval: "Nothing in your documents was close enough to this question to answer it.",
  model_found_no_support: "The closest passages were read, and none of them answer this.",
  no_valid_citations: "An answer was drafted but couldn't be tied to a passage in your documents, so it was withheld.",
};
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = { docs: [], scope: null, entries: [], selected: null, busy: false, confirming: null };

// --- API ---------------------------------------------------------------------

async function apiError(response) {
  try {
    const body = await response.json();
    if (body.error) return `${body.error.message} (request ${body.error.request_id})`;
  } catch (_) { /* not JSON */ }
  return `The server answered ${response.status}.`;
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (_) {
    throw new Error("Can't reach the server. Is the API running?");
  }
  if (!response.ok) throw new Error(await apiError(response));
  return response.json();
}

// --- engine line and setup notice ------------------------------------------------

async function loadInfo() {
  try {
    const info = await api("/info");
    const provider = info.llm_provider === "groq" ? "Groq" : "Anthropic";
    const where = info.embedding_provider === "local" ? "run locally" : "via OpenAI";
    $("engine").textContent =
      `Answers from ${info.llm_model} on ${provider}. ` +
      `Passages found with ${info.embedding_model.split("/").pop()}, ${where}.`;
    if (!info.llm_key_configured) {
      const key = info.llm_provider === "groq" ? "GROQ_API_KEY" : "ANTHROPIC_API_KEY";
      showNotice(`${key} isn't set, so questions can't be answered yet. Add it to .env and restart the server. Uploading still works.`);
    }
  } catch (err) {
    showNotice(err.message);
  }
}

function showNotice(text) {
  const notice = $("notice");
  notice.textContent = text;
  notice.hidden = false;
}

// --- documents -------------------------------------------------------------------

async function loadDocs() {
  try {
    state.docs = await api("/documents");
  } catch (err) {
    showNotice(err.message);
    state.docs = [];
  }
  if (state.scope && !state.docs.some((d) => d.source === state.scope)) state.scope = null;
  renderDocs();
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function renderDocs() {
  const list = $("doc-list");
  list.replaceChildren();
  for (const doc of state.docs) {
    const li = document.createElement("li");
    li.className = "doc";
    li.setAttribute("aria-current", String(state.scope === doc.source));

    const name = document.createElement("button");
    name.type = "button";
    name.className = "doc-name";
    name.append(fileLabel(doc.source));
    name.title = state.scope === doc.source ? "Search all documents again" : "Search only this document";
    name.addEventListener("click", () => setScope(state.scope === doc.source ? null : doc.source));

    const detail = document.createElement("div");
    detail.className = "doc-detail";
    if (state.confirming === doc.source) {
      const ask = document.createElement("span");
      ask.className = "doc-confirm";
      ask.textContent = "Remove it?";
      const yes = textButton("Remove", () => removeDoc(doc.source), "btn-danger");
      const no = textButton("Keep", () => { state.confirming = null; renderDocs(); });
      const group = document.createElement("span");
      group.append(yes, " ", no);
      detail.append(ask, group);
    } else {
      const facts = document.createElement("span");
      facts.textContent = `${plural(doc.pages, "page")}, ${plural(doc.chunks, "passage")}`;
      detail.append(facts, textButton("Remove", () => { state.confirming = doc.source; renderDocs(); }));
    }
    li.append(name, detail);
    list.append(li);
  }
  $("doc-count").textContent = state.docs.length ? plural(state.docs.length, "document") : "";
  $("sample").hidden = SAMPLES.every((s) => state.docs.some((d) => d.source === s));
  renderScope();
  renderTry();
  renderEmpty();
}

// Filenames like northwind_annual_report_fy2025.pdf: allow line breaks after
// underscores and before the extension instead of mid-word.
function fileLabel(name) {
  const frag = document.createDocumentFragment();
  name.split(/(?<=_)|(?=\.[^.]+$)/).forEach((part, i) => {
    if (i) frag.append(document.createElement("wbr"));
    frag.append(part);
  });
  return frag;
}

function textButton(label, onClick, extra = "") {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `btn btn-text ${extra}`.trim();
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

async function removeDoc(source) {
  state.confirming = null;
  try {
    await api(`/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
  } catch (err) {
    showNotice(err.message);
  }
  await loadDocs();
}

function setScope(source) {
  state.scope = source;
  renderDocs();
}

function renderScope() {
  const scope = $("scope");
  scope.replaceChildren();
  scope.hidden = !state.scope;
  if (state.scope) {
    scope.append(`Searching only ${state.scope}`, textButton("Search all", () => setScope(null)));
  }
}

// --- uploads ------------------------------------------------------------------------

async function uploadFiles(files) {
  for (const file of files) {
    const row = document.createElement("li");
    row.textContent = `Reading ${file.name}…`;
    $("uploads").prepend(row);

    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const result = await api("/ingest", { method: "POST", body: form });
      let text = `Added ${result.source}: ${plural(result.pages, "page")}, ${plural(result.chunks_stored, "passage")}.`;
      if (result.empty_pages) text += ` ${plural(result.empty_pages, "page")} had no text (scanned?) and can't be searched.`;
      row.textContent = text;
      row.className = "is-done";
      // The document list now shows it; keep the row only if there's a warning.
      if (!result.empty_pages) setTimeout(() => row.remove(), 4000);
    } catch (err) {
      row.textContent = `${file.name} wasn't added. ${err.message}`;
      row.className = "is-error";
    }
    await loadDocs();
  }
}

async function loadSamples() {
  const button = $("load-sample");
  button.disabled = true;
  try {
    const files = [];
    for (const name of SAMPLES) {
      const response = await fetch(`/static/samples/${name}`);
      if (!response.ok) throw new Error(`Couldn't load the sample ${name}.`);
      const type = name.endsWith(".pdf") ? "application/pdf" : "text/plain";
      files.push(new File([await response.blob()], name, { type }));
    }
    await uploadFiles(files);
  } catch (err) {
    showNotice(err.message);
  } finally {
    button.disabled = false;
  }
}

function wireDrop() {
  const drop = $("drop");
  const library = $("library");
  $("add-file").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    uploadFiles([...e.target.files]);
    e.target.value = "";
  });
  library.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("is-over"); });
  library.addEventListener("dragleave", (e) => {
    if (!library.contains(e.relatedTarget)) drop.classList.remove("is-over");
  });
  library.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("is-over");
    uploadFiles([...e.dataTransfer.files]);
  });
  $("load-sample").addEventListener("click", loadSamples);
}

// --- suggestions and empty state ------------------------------------------------------

function renderTry() {
  const hasSamples = SAMPLES.some((s) => state.docs.some((d) => d.source === s));
  $("try").hidden = !hasSamples || state.entries.length > 2;
  const list = $("try-list");
  list.replaceChildren();
  for (const [question, why] of TRY_QUESTIONS) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = question;
    b.addEventListener("click", () => { $("question").value = question; ask(); });
    const note = document.createElement("span");
    note.className = "why";
    note.textContent = `(${why})`;
    li.append(b, note);
    list.append(li);
  }
}

function renderEmpty() {
  const empty = $("empty");
  empty.hidden = state.entries.length > 0;
  if (state.docs.length) {
    empty.querySelector(".empty-lead").textContent = `Ask about your ${plural(state.docs.length, "document")}.`;
    empty.querySelector("p:last-child").textContent =
      "Answers cite numbered passages. Click a number to see the passage it came from. " +
      "If the answer isn't in your documents, you'll be told so instead of getting a guess.";
  }
}

// --- asking ------------------------------------------------------------------------------

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield [event, JSON.parse(data)];
    }
  }
}

async function ask() {
  const question = $("question").value.trim();
  if (!question || state.busy) return;
  setBusy(true);

  const entry = { question, scope: state.scope, status: "searching", raw: "", result: null, error: null, ms: null };
  entry.node = buildEntry(entry);
  state.entries.unshift(entry);
  $("log").prepend(entry.node);
  select(entry);
  renderEmpty();
  renderTry();

  const started = performance.now();
  const payload = { query: question, top_k: 5 };
  if (entry.scope) payload.source = entry.scope;

  try {
    let response;
    try {
      response = await fetch("/query/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (_) {
      throw new Error("Can't reach the server. Is the API running?");
    }
    if (!response.ok) throw new Error(await apiError(response));

    for await (const [event, data] of readEvents(response)) {
      if (event === "delta") {
        entry.status = "writing";
        entry.raw += data;
      } else if (event === "final") {
        entry.result = data;
      } else if (event === "error") {
        entry.error = `${data.message} (request ${data.request_id})`;
      }
      renderEntry(entry);
    }
    if (!entry.result && !entry.error) entry.error = "The answer stream ended early. Try again.";
  } catch (err) {
    entry.error = err.message;
  } finally {
    entry.ms = performance.now() - started;
    entry.status = "done";
    renderEntry(entry);
    if (state.selected === entry) renderSources();
    setBusy(false);
    $("question").value = "";
    $("question").focus();
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("ask-btn").disabled = busy;
  $("ask-btn").textContent = busy ? "Answering…" : "Ask";
}

// --- rendering an answer ----------------------------------------------------------------

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Escape first, then add the small amount of structure models emit:
// paragraphs, bullet and numbered lists, **bold**, and [n] citations.
function formatAnswer(text, validIds) {
  const inline = (s) =>
    escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, ids) =>
        ids.split(",").map((id) => id.trim())
          .filter((id) => validIds.has(Number(id)))
          .map((id) => `<button type="button" class="cite" data-cite="${id}" aria-label="Show source ${id}">${id}</button>`)
          .join(""));

  // Some models put narrow no-break spaces in figures ("$150\u202fmillion"),
  // which look cramped in Literata; use a regular no-break space instead.
  return text.replace(/\u202f/g, "\u00a0").trim().split(/\n\s*\n/).map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
      return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*]\s+/, ""))}</li>`).join("")}</ul>`;
    }
    if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
      return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
    }
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  }).join("");
}

function buildEntry(entry) {
  const node = $("entry-tpl").content.firstElementChild.cloneNode(true);
  node.querySelector(".entry-q").textContent = entry.question;
  node.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (cite) {
      select(entry);
      activateSource(Number(cite.dataset.cite), true);
      return;
    }
    if (state.selected !== entry) select(entry);
  });
  renderEntry(entry, node);
  return node;
}

function renderEntry(entry, node = entry.node) {
  const answer = node.querySelector(".entry-a");
  const meta = node.querySelector(".entry-meta");
  answer.classList.remove("is-streaming");
  meta.replaceChildren();

  if (entry.error) {
    answer.innerHTML = `<p class="answer-error"></p>`;
    answer.firstChild.textContent = entry.error;
  } else if (entry.result && entry.result.abstained) {
    answer.innerHTML = `<p class="abstain-lead"></p><p class="abstain-why"></p>`;
    answer.querySelector(".abstain-lead").textContent = entry.result.answer;
    answer.querySelector(".abstain-why").textContent = ABSTAIN_WHY[entry.result.abstain_reason] || "";
  } else if (entry.result) {
    const ids = new Set(entry.result.citations.map((c) => c.id));
    answer.innerHTML = formatAnswer(entry.result.answer, ids);
  } else if (entry.status === "writing") {
    answer.textContent = entry.raw;
    answer.classList.add("is-streaming");
  } else {
    answer.innerHTML = `<p class="abstain-why"></p>`;
    answer.firstChild.textContent = entry.scope ? `Searching ${entry.scope}…` : "Searching your documents…";
  }

  const facts = [];
  if (entry.result && !entry.result.abstained) facts.push(plural(entry.result.citations.length, "source"));
  if (entry.result && entry.result.invalid_citation_ids.length) {
    const n = entry.result.invalid_citation_ids.length;
    facts.push([`${plural(n, "citation")} removed: ${n === 1 ? "it pointed" : "they pointed"} at no retrieved passage`, "warn"]);
  }
  if (entry.scope) facts.push(`Only ${entry.scope}`);
  if (entry.ms !== null) facts.push(`${(entry.ms / 1000).toFixed(1)} s`);
  if (entry.result) facts.push(`Request ${entry.result.request_id}`);
  for (const fact of facts) {
    const span = document.createElement("span");
    const [text, cls] = Array.isArray(fact) ? fact : [fact, ""];
    span.textContent = text;
    if (cls) span.className = cls;
    meta.append(span);
  }
}

// --- evidence pane -----------------------------------------------------------------------

function select(entry) {
  state.selected = entry;
  for (const e of state.entries) e.node.classList.toggle("is-selected", e === entry);
  renderSources();
}

// Sentences of a passage that contain a figure the answer quoted next to
// this citation. Purely visual: the whole passage is the cited source.
function figuresNear(answer, id) {
  const figures = new Set();
  for (const sentence of answer.split(/(?<=[.!?])\s+/)) {
    if (!new RegExp(`\\[[^\\]]*\\b${id}\\b[^\\]]*\\]`).test(sentence)) continue;
    for (const m of sentence.matchAll(/\$?\d[\d,]*(?:\.\d+)?%?/g)) {
      const fig = m[0].replace(/[$,%]/g, "");
      if (fig.length > 1 || /%/.test(m[0])) figures.add(fig);
    }
  }
  return figures;
}

function markPassage(text, figures) {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const hits = sentences.map((s) => [...figures].some((f) => s.replace(/[$,%]/g, "").includes(f)));
  const any = hits.some(Boolean);
  return sentences
    .map((s, i) => (!any || hits[i] ? `<mark>${escapeHtml(s)}</mark>` : escapeHtml(s)))
    .join(" ");
}

function renderSources() {
  const list = $("sources");
  const caption = $("evidence-for");
  list.replaceChildren();
  const entry = state.selected;
  if (!entry || !entry.result || entry.result.abstained) {
    caption.textContent = !entry
      ? "Sources for the selected answer appear here."
      : entry.result && entry.result.abstained
        ? "No sources: nothing in your documents supports an answer."
        : entry.error ? "No sources for this question." : "Waiting for the answer…";
    return;
  }

  caption.textContent = `For: ${entry.question}`;
  for (const c of entry.result.citations) {
    const li = document.createElement("li");
    li.className = "source";
    li.id = `source-${c.id}`;
    li.innerHTML = `
      <div class="source-tab"><span class="source-n"></span><span class="source-page"></span></div>
      <p class="source-file"></p>
      <div>
        <p class="source-text"></p>
        <p class="source-match"></p>
      </div>`;
    li.querySelector(".source-n").textContent = c.id;
    li.querySelector(".source-page").textContent = `p. ${c.page_number}`;
    li.querySelector(".source-file").append(fileLabel(c.source));
    li.querySelector(".source-text").innerHTML = markPassage(c.text, figuresNear(entry.result.answer, c.id));
    li.querySelector(".source-match").textContent = `Similarity to your question: ${c.similarity.toFixed(2)}`;
    li.addEventListener("click", () => activateSource(c.id, false));
    list.append(li);
  }
}

function activateSource(id, scroll) {
  for (const node of document.querySelectorAll(".source, .entry.is-selected .cite")) {
    const match = node.classList.contains("cite") ? Number(node.dataset.cite) === id : node.id === `source-${id}`;
    node.classList.toggle("is-active", match);
  }
  const target = $(`source-${id}`);
  if (target && scroll) target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
}

// --- start ---------------------------------------------------------------------------------

$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
});
wireDrop();
loadInfo();
loadDocs();
