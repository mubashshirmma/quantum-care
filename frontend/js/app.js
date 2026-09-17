/* QuantumCare core: helpers, auth, router, dashboard, models & datasets, benchmark, quantum lab, settings.
   Patient analysis + records live in patient.js. */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, attrs = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) n.append(c);
  return n;
};
const svgEl = (tag, attrs = {}, ...children) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); for (const c of children.flat()) if (c != null) n.append(c); return n; };
const api = async (path, opts = {}) => {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  const isJson = r.headers.get("content-type")?.includes("json");
  const body = isJson ? await r.json() : await r.text();
  if (r.status === 401 && !path.startsWith("/api/auth/")) { Auth.showLogin(); throw new Error("Please sign in."); }
  if (!r.ok) { const d = typeof body === "object" ? body.detail : body; const err = new Error(typeof d === "string" ? d : (d && d.message) || JSON.stringify(d ?? body)); err.status = r.status; err.detail = d; throw err; }
  return body;
};
const postJson = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const patchJson = (path, data) => api(path, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const toast = (msg, ms = 3000) => { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._t); t._t = setTimeout(() => (t.hidden = true), ms); };
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const pct = x => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
const fmtTime = iso => { const d = new Date(iso); const today = new Date().toDateString() === d.toDateString(); return (today ? "Today " : d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " ") + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); };
const MODEL_LABELS = { logistic_regression: "Logistic Regression", random_forest: "Random Forest", svm_rbf: "SVM (RBF)", xgboost: "XGBoost", quantum_vqc: "Quantum VQC", quantum_qsvc: "Quantum QSVC" };
const MODEL_ORDER = ["logistic_regression", "random_forest", "svm_rbf", "xgboost", "quantum_vqc", "quantum_qsvc"];
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)"];
const mlabel = m => MODEL_LABELS[m] || m;
const mcolor = m => SERIES[Math.max(0, MODEL_ORDER.indexOf(m)) % SERIES.length];
const isQ = m => m.startsWith("quantum");
const METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"];
const METRIC_LABEL = { accuracy: "Accuracy", precision: "Precision", recall: "Recall / Sensitivity", specificity: "Specificity", f1: "F1", roc_auc: "ROC-AUC" };
const SOURCE_LABEL = { manual: "Manual entry", nvidia_ocr: "NVIDIA OCR", manual_transcription: "Manual transcription" };

const Settings = {
  key: "qc_settings", defaults: { criterion: "accuracy", preferProbability: false, compareAll: true },
  get() { try { return { ...this.defaults, ...(JSON.parse(localStorage.getItem(this.key)) || {}) }; } catch { return { ...this.defaults }; } },
  set(patch) { const v = { ...this.get(), ...patch }; try { localStorage.setItem(this.key, JSON.stringify(v)); } catch {} return v; },
};
const State = { categories: [], diseases: [], selected: null, pollers: {}, benchCache: {} };
async function refreshDiseases() {
  [State.categories, State.diseases] = await Promise.all([api("/api/categories"), api("/api/diseases")]);
  $("#env-pill").textContent = `${State.diseases.length} datasets · ${State.diseases.filter(d => d.trained).length} trained`;
  for (const d of State.diseases) if (d.training === "running") watchTraining(d.name);
  State.benchCache = {};
  return State.diseases;
}
const trainedDiseases = () => State.diseases.filter(d => d.trained);
async function benchmarkOf(name) { if (!State.benchCache[name]) State.benchCache[name] = await api(`/api/diseases/${name}`); return State.benchCache[name]; }

/* ---------- router ---------- */
const Pages = window.Pages = {};
const TITLES = { dashboard: "Dashboard", analysis: "Patient Analysis", records: "Patient Records", models: "Models & Datasets", benchmark: "Benchmark · Classical vs Quantum", quantum: "Quantum Lab", settings: "Settings" };
function route() {
  let hash = location.hash || "#/dashboard";
  if (hash === "#add") hash = "#/models?tab=upload&add=1";
  const [path, qs] = hash.slice(2).split("?");
  const name = TITLES[path] ? path : "dashboard";
  const params = Object.fromEntries(new URLSearchParams(qs || ""));
  $$(".page").forEach(p => (p.hidden = p.dataset.page !== name));
  $$(".nav a").forEach(a => { const h = a.getAttribute("href"); a.classList.toggle("active", a.dataset.route ? a.dataset.route === name : h === hash || (h.startsWith(`#/${name}?`) && Object.entries(Object.fromEntries(new URLSearchParams(h.split("?")[1]))).every(([k, v]) => params[k] === v))); });
  $("#page-title").textContent = TITLES[name]; $("#sidebar").classList.remove("open");
  Pages[name]?.render($(`.page[data-page="${name}"]`), params);
}
window.addEventListener("hashchange", route);
$("#menu-btn").onclick = () => $("#sidebar").classList.toggle("open");
const go = (path, params) => { location.hash = "#/" + path + (params ? "?" + new URLSearchParams(params) : ""); };
const segmented = (items, current, onpick) => el("div", { class: "seg" }, ...items.map(([k, l]) => el("button", { class: k === current ? "on" : "", onclick: () => onpick(k) }, l)));
const diseasePicker = (current, onpick, onlyTrained = true) => el("select", { onchange: e => onpick(e.target.value) }, ...(onlyTrained ? trainedDiseases() : State.diseases).map(d => el("option", { value: d.name, selected: d.name === current }, d.display_name)));

/* ==================================================================== AUTH */
const Auth = window.Auth = {
  user: null, env: {},
  async check() { const me = await api("/api/auth/me"); this.user = me.authenticated ? me.user : null; this.env = me.environment || {}; return this.user; },
  showLogin() { $("#shell").hidden = true; $("#login").hidden = false; $("#login-devnote").hidden = !this.env.default_credentials; setTimeout(() => $("#login-user").focus(), 50); },
  showApp() {
    $("#login").hidden = true; $("#shell").hidden = false; $("#user-name").innerHTML = ""; $("#user-name").append(icon("user", 16), " " + this.user.username);
    const notes = [];
    if (this.env.default_credentials) notes.push("Default operator credentials in use — set APP_ADMIN_USER / APP_ADMIN_PASSWORD.");
    if (this.env.dev_show_otp) notes.push("No mail server (SMTP_HOST): verification codes are shown on screen in DEV MODE.");
    $("#devbar").hidden = !notes.length; $("#devbar").textContent = notes.join("  ·  ");
    refreshDiseases().catch(() => {}); route();
  },
  async logout() { try { await api("/api/auth/logout", { method: "POST" }); } catch {} this.user = null; this.showLogin(); },
};
$("#login-form").addEventListener("submit", async e => {
  e.preventDefault(); const err = $("#login-err"); err.hidden = true;
  try { await postJson("/api/auth/login", { username: $("#login-user").value.trim(), password: $("#login-pass").value }); $("#login-pass").value = ""; await Auth.check(); Auth.showApp(); }
  catch (ex) { err.textContent = ex.message; err.hidden = false; }
});
$("#btn-logout").onclick = () => Auth.logout();
document.addEventListener("DOMContentLoaded", async () => {
  try { await Auth.check(); } catch (e) { $("#login").hidden = false; $("#login-err").textContent = "Cannot reach the server: " + e.message; $("#login-err").hidden = false; return; }
  if (Auth.user) Auth.showApp(); else Auth.showLogin();
});

/* ==================================================================== DASHBOARD */
const stat = (v, l) => el("div", { class: "stat" }, el("div", { class: "v" }, String(v)), el("div", { class: "l" }, l));
const quickAction = (ic, title, sub, href) => el("a", { class: "qa", href }, el("div", { class: "qi" }, icon(ic, 22)), el("div", {}, el("div", { class: "qt" }, title), el("div", { class: "qs" }, sub)), el("div", { class: "qarrow" }, icon("arrowright", 16)));
Pages.dashboard = {
  async render(c) {
    c.innerHTML = ""; await refreshDiseases();
    const trained = trainedDiseases();
    const [stats, recent, ocr] = await Promise.all([api("/api/patients-stats").catch(() => ({ patients: 0, analyses: 0 })), api("/api/analyses/recent?limit=4").catch(() => []), api("/api/ocr/status").catch(() => ({ available: false }))]);
    const nModels = trained.reduce((n, d) => n + (d.n_models || 6), 0);
    c.append(el("div", { class: "hero" },
      el("div", { class: "panel cta" }, el("h2", {}, "Hybrid Quantum Health Intelligence"),
        el("p", {}, "Early-disease decision support that runs classical and quantum machine-learning models side by side on the same patient data — from manual entry or NVIDIA-OCR'd medical reports — with per-patient explanations and honest benchmarking."),
        el("div", { class: "actions", style: "justify-content:flex-start" }, el("a", { class: "btn primary lg", href: "#/analysis" }, icon("plus", 18), "NEW ANALYSIS"), el("a", { class: "btn lg", href: "#/analysis?mode=ocr" }, icon("scan", 18), "NVIDIA OCR"))),
      el("div", { class: "orbits", "aria-hidden": "true" }, el("i", {}), el("i", {}), el("i", {})),
      el("div", { class: "stats" }, stat(stats.patients, "patients"), stat(stats.analyses, "analyses"), stat(nModels, "trained models"), stat(trained.length, "diseases"))));
    c.append(el("div", { class: "quickacts" },
      quickAction("stethoscope", "New Analysis", "manual entry, saved models", "#/analysis?mode=manual"),
      quickAction("scan", "NVIDIA OCR", ocr.available ? `${ocr.mode} · ${ocr.model}` : "not configured — manual transcription available", "#/analysis?mode=ocr"),
      quickAction("folder", "Patient Records", `${stats.patients} records · fast lookup by Patient ID`, "#/records"),
      quickAction("dna", "Models & Datasets", `${State.diseases.length} datasets · ${nModels} trained models`, "#/models"),
      quickAction("chart", "Benchmark", "classical vs quantum, same split", "#/benchmark"),
      quickAction("atom", "Quantum Lab", "circuits, encodings, ansatz", "#/quantum")));
    c.append(el("div", { class: "panel", style: "margin-bottom:18px" },
      el("div", { class: "panel-head" }, el("h2", {}, "Diseases with trained models"), el("span", { class: "muted" }, "click to start a patient analysis")),
      el("div", { class: "quick" }, ...State.diseases.map(d => el("div", { class: "qcard" + (d.trained ? "" : " disabled"), onclick: () => d.trained && go("analysis", { mode: "manual", disease: d.name }) },
        el("div", { class: "icon" }, dIcon(d)), el("div", {}, el("div", { class: "name" }, d.display_name), el("div", { class: "meta" }, d.trained ? `best: ${mlabel(d.best_model.name)} · ${pct(d.best_model.accuracy)} acc · ${d.n_features} features` : "not trained yet")))))));
    c.append(el("div", { class: "grid3" },
      el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, "Recent analyses"), el("a", { class: "btn ghost sm", href: "#/records?tab=history" }, "all →")),
        recent.length ? el("div", { class: "hlist" }, ...recent.map(h => window.historyItem(h, true))) : el("p", { class: "muted" }, "No analyses saved yet.")),
      el("div", { class: "panel" }, el("h2", {}, "Architecture"), el("div", { class: "flowsteps", style: "margin-top:10px;font-size:12.5px" },
        ...["Medical data (CSV · manual · document)", "NVIDIA OCR → extraction → human verification", "Preprocessing + feature selection", "Classical ML ‖ Quantum ML", "Prediction → explainability → benchmark", "Patient record + history"].map((t, i) => el("div", { class: "fstep", style: "padding:8px 10px" }, el("div", { class: "n" }, String(i + 1)), el("div", { class: "t", style: "font-weight:500" }, t), el("div", {}))))),
      el("div", { class: "panel" }, el("h2", {}, "Medical safety"), el("p", { class: "muted" }, "Patient information and model outputs are intended for authorized research and decision-support use. This system is not a substitute for professional medical diagnosis."),
        el("p", { class: "muted", style: "font-size:12.5px" }, "Outputs are model scores from a single hold-out evaluation, not calibrated clinical probabilities. Quantum models run on a noiseless statevector simulator."))));
  },
};

/* ==================================================================== MODELS & DATASETS */
Pages.models = {
  tab: "explorer", disease: null,
  async render(c, params) {
    await refreshDiseases();
    this.tab = params.tab || this.tab; this.disease = params.disease || this.disease || (trainedDiseases()[0] || State.diseases[0])?.name;
    $("#models-subnav").innerHTML = ""; $("#models-subnav").append(segmented([["upload", "CSV Upload"], ["explorer", "Dataset Explorer"], ["preprocessing", "Preprocessing"], ["models", "Models"]], this.tab, t => go("models", { tab: t, disease: this.disease })));
    const body = $("#models-body"); body.innerHTML = ""; $("#ds-layout").hidden = this.tab !== "upload";
    if (this.tab === "upload") { renderCategories(); if (State.selected) showDetail(State.selected); if (params.add) { openModal(); history.replaceState(null, "", "#/models?tab=upload"); } return; }
    if (!this.disease) { body.append(el("div", { class: "panel" }, el("p", { class: "muted" }, "No datasets registered."))); return; }
    const head = el("div", { class: "filters" }, el("label", {}, "Dataset", diseasePicker(this.disease, d => go("models", { tab: this.tab, disease: d }), false)));
    body.append(head);
    if (this.tab === "explorer") body.append(await datasetExplorer(this.disease));
    if (this.tab === "preprocessing") body.append(await preprocessingView(this.disease));
    if (this.tab === "models") body.append(await modelsView(this.disease));
  },
};
async function datasetExplorer(name) {
  const [d, form] = await Promise.all([api(`/api/diseases/${name}`), api(`/api/diseases/${name}/form`).catch(() => null)]);
  const s = d.spec, data = d.data; const wrap = el("div", { class: "grid3", style: "grid-template-columns:1fr" });
  const cls = data.class_counts || { "0": 0, "1": 0 }; const n = (cls["0"] || 0) + (cls["1"] || 0);
  wrap.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", { class: "with-icon" }, dIcon(s, 22), s.display_name), el("span", { class: s.target_documented ? "badge ok" : "badge warn" }, s.target_documented ? "target documented" : "target assumed")),
    el("div", { class: "stats", style: "margin-bottom:14px" }, stat(data.n_rows ?? "—", "rows"), stat(s.numeric_features.length + s.categorical_features.length, "features"), stat(s.numeric_features.length, "numeric"), stat(s.categorical_features.length, "categorical"),
      stat(Object.values(data.missing_per_feature || {}).reduce((a, b) => a + b, 0), "missing cells")),
    el("h3", {}, "Class distribution"), el("div", { class: "dist" }, el("i", { style: `width:${n ? cls["0"] / n * 100 : 0}%`, title: `0: ${cls["0"]}` }), el("i", { style: `width:${n ? cls["1"] / n * 100 : 0}%`, title: `1: ${cls["1"]}` })),
    el("div", { class: "legend" }, el("span", {}, el("i", { style: "background:var(--series-1)" }), `0 = ${s.target_meaning["0"]} (${cls["0"]}, ${n ? Math.round(cls["0"] / n * 100) : 0}%)`), el("span", {}, el("i", { style: "background:var(--series-2)" }), `1 = ${s.target_meaning["1"]} (${cls["1"]}, ${n ? Math.round(cls["1"] / n * 100) : 0}%)`)),
    el("dl", { class: "kv", style: "margin-top:14px" }, el("dt", {}, "Target rule"), el("dd", {}, `${s.target_column}: ${s.target_rule_text}`), el("dt", {}, "Source"), el("dd", {}, s.source || "—"), el("dt", {}, "Notes"), el("dd", {}, s.notes || "—"))));
  const rows = (form?.features || []).map(f => el("tr", {}, el("td", {}, f.label), el("td", { class: "mono muted" }, f.name), el("td", {}, f.type === "number" ? "numeric" : "categorical"),
    el("td", {}, f.type === "number" ? `${f.range.min} – ${f.range.max} (median ${f.range.median})${f.unit ? " " + f.unit : ""}` : f.options.map(o => `${o.label} (${o.count})`).join(", ")),
    el("td", { class: "num" }, String(data.missing_per_feature?.[f.name] || 0))));
  wrap.append(el("div", { class: "panel" }, el("h2", {}, "Features"), el("div", { class: "tablewrap", style: "margin-top:8px" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "feature"), el("th", {}, "column"), el("th", {}, "type"), el("th", {}, "range / categories (training data)"), el("th", { class: "num" }, "missing"))), el("tbody", {}, ...rows)))));
  return wrap;
}
async function preprocessingView(name) {
  const p = await api(`/api/diseases/${name}/pipeline`);
  const steps = el("div", { class: "flowsteps" });
  p.steps.forEach((s, i) => { steps.append(el("div", { class: "fstep" }, el("div", { class: "n" }, String(i + 1)), el("div", {}, el("div", { class: "t" }, s.step), el("div", { class: "d" }, s.detail)), s.leakage_safe === true ? el("span", { class: "badge ok" }, "fit on train only") : s.leakage_safe === false ? el("span", { class: "badge bad" }, "leak") : el("span", {}))); if (i < p.steps.length - 1) steps.append(el("div", { class: "farrow" }, "↓")); });
  return el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, `${p.icon} Hybrid preprocessing pipeline · ${p.display_name}`), el("span", { class: "muted" }, "every learned statistic comes from the training split; the same fitted pipeline is applied to new patients")), steps,
    el("p", { class: "disclaimer" }, "Raw data → missing values → encoding → scaling → feature selection/reduction → quantum feature encoding → ML/QML. Classical models receive all preprocessed features; the quantum branch is reduced to one feature per qubit."));
}
async function modelsView(name) {
  const d = await benchmarkOf(name); const meta = d.benchmark; const wrap = el("div", {});
  const running = d.card.training === "running";
  wrap.append(el("div", { class: "panel", style: "margin-bottom:16px" }, el("div", { class: "panel-head" }, el("h2", {}, `${d.spec.icon} ${d.spec.display_name} · trained models`),
    el("div", { class: "actions", style: "margin:0" }, meta ? el("a", { class: "btn sm", href: `#/benchmark?disease=${name}` }, "Benchmark →") : null, el("button", { class: "btn primary sm", disabled: running, onclick: () => startTraining(name) }, running ? "Training…" : meta ? "Re-train all models" : "Train all models"))),
    meta ? el("p", { class: "muted" }, `Trained ${meta.timestamp_utc} · hold-out test n=${meta.split.n_test} (seed ${meta.split.seed}) · ${meta.preprocessing.n_features_after_preprocessing} preprocessed features · Qiskit ${meta.environment.qiskit}, scikit-learn ${meta.environment.sklearn}`) : el("p", { class: "muted" }, "Not trained yet. Training fits 4 classical + 2 quantum models on the same split (≈ 1 min).")));
  if (!meta) return wrap;
  const cards = el("div", { class: "labcards" });
  for (const m of MODEL_ORDER.filter(m => meta.results[m])) {
    const r = meta.results[m], t = meta.training_info[m] || {};
    cards.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, (isQ(m) ? "⚛ " : "") + mlabel(m)), el("span", { class: isQ(m) ? "badge q" : "badge" }, isQ(m) ? "quantum" : "classical")),
      el("div", { class: "stats" }, stat(fmt(r.accuracy), "accuracy"), stat(fmt(r.roc_auc), "ROC-AUC"), stat(fmt(r.f1), "F1")),
      el("dl", { class: "kv", style: "margin-top:12px;font-size:12.5px" }, el("dt", {}, "Sensitivity"), el("dd", {}, fmt(r.recall)), el("dt", {}, "Specificity"), el("dd", {}, fmt(r.specificity)), el("dt", {}, "Train time"), el("dd", {}, `${fmt(t.train_time_s, 2)} s`), el("dt", {}, "Inference"), el("dd", {}, t.predict_ms_per_sample != null ? `${fmt(t.predict_ms_per_sample, 2)} ms / sample` : "—"), el("dt", {}, "Train acc"), el("dd", {}, fmt(t.train_accuracy)))));
  }
  wrap.append(cards); return wrap;
}
/* dataset cards + detail + wizard (unchanged behaviour) */
function renderCategories() {
  const root = $("#categories"); root.innerHTML = "";
  for (const cat of State.categories) {
    const items = State.diseases.filter(d => d.category === cat.key); if (!items.length && cat.key === "other") continue;
    const cards = el("div", { class: "cards" }); items.forEach(d => cards.append(dsCard(d)));
    if (!items.length) cards.append(el("div", { class: "card empty" }, "No dataset yet"));
    cards.append(el("div", { class: "card ds add", onclick: () => openModal(cat.key) }, el("div", { class: "icon" }, icon("plus", 22)), el("div", { class: "name" }, "Add New Dataset"), el("div", { class: "meta" }, `to ${cat.label}`)));
    root.append(el("section", { class: "category" }, el("h3", {}, dIcon({ category: cat.key }, 18), ` ${cat.label} `, el("span", { class: "count" }, `${items.length}`)), cards));
  }
}
function dsCard(d) {
  const status = d.training === "running" ? el("span", { class: "badge run" }, "training…") : d.trained ? el("span", { class: "badge ok" }, `best ${mlabel(d.best_model.name)} ${fmt(d.best_model.accuracy)}`) : el("span", { class: "badge" }, "not trained");
  return el("div", { class: "card ds" + (State.selected === d.name ? " selected" : ""), onclick: () => showDetail(d.name) }, el("div", { class: "icon" }, dIcon(d)), el("div", { class: "name" }, d.display_name),
    el("div", { class: "meta" }, `${d.n_features} features${d.builtin ? " · built-in" : " · user-added"}${d.target_documented ? "" : " · target assumed"}`), el("div", {}, status));
}
async function showDetail(name) {
  State.selected = name; renderCategories();
  const panel = $("#detail"); panel.hidden = false; $("#ds-layout").classList.add("with-detail"); $("#d-title").textContent = "Loading…"; $("#d-body").innerHTML = "";
  let d; try { d = await api(`/api/diseases/${name}`); } catch (e) { $("#d-body").textContent = e.message; return; }
  const s = d.spec, body = $("#d-body"); $("#d-title").textContent = `${s.icon} ${s.display_name}`; body.innerHTML = "";
  if (!s.target_documented) body.append(el("div", { class: "warnbox" }, "⚠ Target meaning is an assumption, not documented by the data source."));
  const data = d.data.error ? el("span", { class: "badge warn" }, d.data.error) : `${d.data.n_rows} rows · class 0: ${d.data.class_counts["0"]} · class 1: ${d.data.class_counts["1"]}` + (Object.keys(d.data.missing_per_feature || {}).length ? ` · missing: ${JSON.stringify(d.data.missing_per_feature)}` : "");
  body.append(el("dl", { class: "kv" }, el("dt", {}, "Category"), el("dd", {}, `${s.icon} ${s.category_label}`), el("dt", {}, "Data"), el("dd", {}, data), el("dt", {}, "Target"), el("dd", {}, `${s.target_column} — ${s.target_rule_text}`), el("dt", {}, "0 means"), el("dd", {}, s.target_meaning["0"]), el("dt", {}, "1 means"), el("dd", {}, s.target_meaning["1"]), el("dt", {}, "Source"), el("dd", {}, s.source || "—")));
  body.append(el("h3", {}, `Features (${s.numeric_features.length + s.categorical_features.length})`));
  const chips = el("div", { class: "chips" }); s.numeric_features.forEach(f => chips.append(el("span", { class: "chip" }, f))); s.categorical_features.forEach(f => chips.append(el("span", { class: "chip cat" }, f))); body.append(chips);
  body.append(el("h3", {}, "Trained models & benchmark")); if (d.benchmark) body.append(resultsTable(d.benchmark)); else body.append(el("p", { class: "muted" }, "Not trained yet."));
  const running = d.card.training === "running"; const actions = el("div", { class: "actions" });
  actions.append(el("a", { class: "btn sm", href: `#/models?tab=explorer&disease=${name}` }, "Explore"), el("a", { class: "btn sm", href: `#/models?tab=preprocessing&disease=${name}` }, "Preprocessing"));
  if (d.benchmark) actions.append(el("button", { class: "btn sm", onclick: () => go("analysis", { mode: "manual", disease: name }) }, "Analyse a patient →"));
  if (!s.builtin) actions.append(el("button", { class: "btn danger sm", onclick: () => removeDisease(name) }, "Remove"));
  actions.append(el("button", { class: "btn primary sm", disabled: running, onclick: () => startTraining(name) }, running ? "Training…" : d.benchmark ? "Re-train" : "Train & benchmark"));
  body.append(actions);
}
function resultsTable(meta, opts = {}) {
  const models = MODEL_ORDER.filter(m => meta.results[m]); const best = models.slice().sort((a, b) => meta.results[b].accuracy - meta.results[a].accuracy)[0];
  const cols = opts.full ? [...METRICS] : METRICS;
  const t = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "model"), ...cols.map(c => el("th", { class: "num" }, METRIC_LABEL[c] || c)), el("th", { class: "num" }, "train s"), opts.full ? el("th", { class: "num" }, "ms/sample") : null)));
  const tb = el("tbody", {});
  for (const m of models) { const r = meta.results[m], ti = meta.training_info[m] || {}; tb.append(el("tr", { class: m === best ? "best" : "" }, el("td", {}, el("i", { style: `display:inline-block;width:10px;height:10px;border-radius:3px;background:${mcolor(m)};margin-right:7px;vertical-align:-1px` }), (isQ(m) ? "⚛ " : "") + mlabel(m)), ...cols.map(c => el("td", { class: "num" }, fmt(r[c]))), el("td", { class: "num" }, fmt(ti.train_time_s, 2)), opts.full ? el("td", { class: "num" }, fmt(ti.predict_ms_per_sample, 2)) : null)); }
  t.append(tb);
  return el("div", {}, el("div", { class: "tablewrap" }, t), el("p", { class: "muted", style: "font-size:12px" }, `Split: ${meta.split.n_train} train / ${meta.split.n_test} test (seed ${meta.split.seed}) · ${meta.timestamp_utc}. Best row by accuracy; differences of a few points are within noise at this test size.`));
}
async function startTraining(name) {
  try { await api(`/api/train/${name}`, { method: "POST" }); toast("Training started (≈ 1 min incl. quantum models)"); } catch (e) { toast(e.message); return; }
  await refreshDiseases(); route(); watchTraining(name);
}
function watchTraining(name) {
  if (State.pollers[name]) return;
  State.pollers[name] = setInterval(async () => { const st = await api(`/api/train/${name}/status`); if (st.state !== "running") { clearInterval(State.pollers[name]); delete State.pollers[name]; toast(st.state === "done" ? `Training finished: ${name}` : `Training failed: ${st.error}`, 5000); await refreshDiseases(); route(); } }, 2500);
}
async function removeDisease(name) {
  if (!confirm(`Remove dataset '${name}', its uploaded file and any saved models?`)) return;
  try { await api(`/api/diseases/${name}`, { method: "DELETE" }); toast(`Removed ${name}`); } catch (e) { toast(e.message); return; }
  State.selected = null; $("#detail").hidden = true; $("#ds-layout").classList.remove("with-detail"); await refreshDiseases(); renderCategories();
}
$("#d-close").onclick = () => { State.selected = null; $("#detail").hidden = true; $("#ds-layout").classList.remove("with-detail"); renderCategories(); };
let upload = { file: null, profile: null };
const setStep = n => { $$(".steps li").forEach(li => { li.classList.toggle("active", +li.dataset.step === n); li.classList.toggle("done", +li.dataset.step < n); }); $$(".step").forEach(s => (s.hidden = +s.dataset.step !== n)); $("#m-error").hidden = true; };
const showErr = msg => { const e = $("#m-error"); e.textContent = msg; e.hidden = false; };
function openModal(category = "other") {
  upload = { file: null, profile: null }; $("#file").value = ""; $("#drop-text").textContent = "Click to choose a CSV file"; $("#s1-next").disabled = true;
  const sel = $("#f-category"); sel.innerHTML = ""; State.categories.forEach(c => sel.append(el("option", { value: c.key }, `${c.icon} ${c.label}`))); sel.value = category;
  ["#f-display", "#f-name", "#f-source", "#f-m0", "#f-m1", "#f-notes"].forEach(s => ($(s).value = "")); $("#f-documented").checked = false; $("#f-name")._touched = false; setStep(1); $("#modal").hidden = false;
}
$("#btn-add").onclick = () => openModal(); $("#m-close").onclick = () => ($("#modal").hidden = true);
$("#modal").addEventListener("click", e => { if (e.target === $("#modal")) $("#modal").hidden = true; });
$("#file").onchange = e => { upload.file = e.target.files[0] || null; $("#drop-text").textContent = upload.file ? `${upload.file.name} (${(upload.file.size / 1024).toFixed(1)} KB)` : "Click to choose a CSV file"; $("#s1-next").disabled = !upload.file; };
const drop = $("#drop"); drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); }); drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) { const dt = new DataTransfer(); dt.items.add(f); $("#file").files = dt.files; $("#file").dispatchEvent(new Event("change")); } });
$("#s1-next").onclick = async () => {
  const fd = new FormData(); fd.append("file", upload.file); $("#s1-next").disabled = true; $("#s1-next").textContent = "Inspecting…";
  try { upload.profile = await api("/api/datasets/inspect", { method: "POST", body: fd }); } catch (e) { showErr(e.message); return; } finally { $("#s1-next").disabled = false; $("#s1-next").textContent = "Inspect columns →"; }
  const base = upload.file.name.replace(/\.csv$/i, ""); if (!$("#f-display").value) $("#f-display").value = base.replace(/[_-]+/g, " "); if (!$("#f-name").value) $("#f-name").value = slug(base); renderColumns(); setStep(2);
};
const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").replace(/^[^a-z]+/, "").slice(0, 64) || "dataset";
$("#f-display").addEventListener("input", () => { if (!$("#f-name")._touched) $("#f-name").value = slug($("#f-display").value); }); $("#f-name").addEventListener("input", () => ($("#f-name")._touched = true));
function renderColumns() {
  const tb = $("#cols"); tb.innerHTML = ""; const p = upload.profile; $("#col-summary").textContent = `${p.n_rows} rows × ${p.n_columns} columns from ${p.filename}`;
  const targets = p.columns.filter(c => c.suggested_role === "target");
  p.columns.forEach(c => { let role = c.suggested_role; if (role === "target" && targets.length > 1 && c !== targets[targets.length - 1]) role = "categorical";
    const sel = el("select", { "data-col": c.name, onchange: renderTargetBox }, ...[["numeric", "Feature · numeric"], ["categorical", "Feature · categorical"], ["target", "Target"], ["ignore", "Ignore"]].map(([v, l]) => el("option", { value: v, selected: v === role }, l)));
    tb.append(el("tr", {}, el("td", {}, c.name), el("td", { class: "muted" }, c.dtype), el("td", { class: "num" }, c.n_unique), el("td", { class: "num" }, c.n_missing), el("td", { class: "muted" }, c.sample_values.slice(0, 6).join(", ")), el("td", {}, sel))); });
  renderTargetBox();
}
const roles = () => $$("#cols select").map(s => [s.dataset.col, s.value]); const targetCol = () => roles().filter(([, r]) => r === "target").map(([c]) => c);
function renderTargetBox() {
  const box = $("#target-box"); box.innerHTML = ""; const t = targetCol();
  if (t.length !== 1) { box.append(el("p", { class: "muted" }, t.length ? `Select exactly ONE target column (currently ${t.length}).` : "Select exactly one column with role Target above.")); return; }
  const col = upload.profile.columns.find(c => c.name === t[0]); const vals = col.sample_values.map(String); const isBinary01 = vals.length === 2 && vals.every(v => ["0", "1", "0.0", "1.0"].includes(v)); const numeric = /int|float/.test(col.dtype);
  box.append(el("p", {}, "Target column: ", el("b", {}, col.name), el("span", { class: "muted" }, ` · ${col.n_unique} distinct values: ${vals.join(", ")}${col.n_unique > vals.length ? ", …" : ""}`)));
  const rule = el("select", { id: "t-rule", onchange: renderRuleDetail }, el("option", { value: "binary", disabled: !isBinary01 }, "Already 0 / 1 (use as-is)"), el("option", { value: "positive_values" }, "1 if value is one of the selected values"), el("option", { value: "greater_than", disabled: !numeric }, "1 if value > threshold (numeric)"));
  rule.value = isBinary01 ? "binary" : (numeric && col.n_unique > 2 ? "greater_than" : "positive_values"); box.append(el("label", {}, "How does this column map to disease present (1)?", rule), el("div", { id: "t-detail" })); renderRuleDetail();
}
function renderRuleDetail() {
  const d = $("#t-detail"); d.innerHTML = ""; const col = upload.profile.columns.find(c => c.name === targetCol()[0]); const r = $("#t-rule").value;
  if (r === "greater_than") d.append(el("label", {}, "Threshold (label = 1 when value is strictly greater)", el("input", { id: "t-threshold", type: "number", step: "any", value: "0" })));
  if (r === "positive_values") { const vals = el("div", { class: "vals" }); col.sample_values.forEach(v => vals.append(el("label", {}, el("input", { type: "checkbox", class: "t-val", value: String(v) }), String(v)))); d.append(el("p", { class: "muted" }, "Tick every raw value that means disease PRESENT:"), vals); }
}
function buildSpec() {
  const t = targetCol(); if (t.length !== 1) throw new Error("select exactly one Target column");
  const numeric = roles().filter(([, r]) => r === "numeric").map(([c]) => c), categorical = roles().filter(([, r]) => r === "categorical").map(([c]) => c); if (!numeric.length && !categorical.length) throw new Error("at least one feature column is required");
  const ruleType = $("#t-rule").value; const rule = ruleType === "binary" ? { type: "binary" } : ruleType === "greater_than" ? { type: "greater_than", threshold: Number($("#t-threshold").value) } : { type: "positive_values", values: $$(".t-val:checked").map(i => i.value) };
  if (rule.type === "positive_values" && !rule.values.length) throw new Error("tick at least one value meaning disease present");
  const name = $("#f-name").value.trim(); if (!/^[a-z][a-z0-9_]{2,63}$/.test(name)) throw new Error("identifier must be lowercase letters/digits/underscores, 3–64 chars, start with a letter");
  return { name, display_name: $("#f-display").value.trim() || name, category: $("#f-category").value, numeric_features: numeric, categorical_features: categorical, target_column: t[0], target_rule: rule, target_meaning: { "0": $("#f-m0").value.trim() || "disease absent", "1": $("#f-m1").value.trim() || "disease present" }, target_documented: $("#f-documented").checked, source: $("#f-source").value.trim() || "user upload", notes: $("#f-notes").value.trim() };
}
$("#s2-back").onclick = () => setStep(1); $("#s2-next").onclick = () => { try { upload.spec = buildSpec(); $("#review").textContent = JSON.stringify(upload.spec, null, 2); setStep(3); } catch (e) { showErr(e.message); } }; $("#s3-back").onclick = () => setStep(2);
$("#s3-submit").onclick = async () => {
  const fd = new FormData(); fd.append("file", upload.file); fd.append("spec_json", JSON.stringify(upload.spec)); $("#s3-submit").disabled = true;
  try { const res = await api("/api/datasets", { method: "POST", body: fd }); $("#modal").hidden = true; toast(`Registered ${res.card.display_name}: ${res.data.n_rows} rows`, 5000); await refreshDiseases(); renderCategories(); showDetail(res.card.name); } catch (e) { showErr(e.message); } finally { $("#s3-submit").disabled = false; }
};

/* ==================================================================== BENCHMARK */
Pages.benchmark = {
  view: "table", metric: "accuracy", disease: null,
  async render(c, params) {
    c.innerHTML = ""; await refreshDiseases(); const trained = trainedDiseases();
    if (!trained.length) { c.append(el("div", { class: "panel" }, el("p", { class: "muted" }, "No trained models yet."))); return; }
    this.view = params.view || this.view; this.disease = params.disease || this.disease || trained[0].name;
    if (!trained.some(d => d.name === this.disease)) this.disease = trained[0].name;
    const d = await benchmarkOf(this.disease); const meta = d.benchmark;
    c.append(el("div", { class: "filters" }, segmented([["table", "Classical vs Quantum"], ["metrics", "Metrics"], ["roc", "ROC"], ["cm", "Confusion Matrix"]], this.view, v => go("benchmark", { view: v, disease: this.disease })),
      el("label", {}, "Disease", diseasePicker(this.disease, x => go("benchmark", { view: this.view, disease: x }))),
      this.view === "metrics" ? el("label", {}, "Metric", el("select", { onchange: e => { this.metric = e.target.value; route(); } }, ...METRICS.map(m => el("option", { value: m, selected: m === this.metric }, METRIC_LABEL[m])))) : null));
    const head = el("div", { class: "bench-head", style: "margin-bottom:12px" }, el("span", { class: "icon" }, dIcon(d.spec, 24)), el("h2", {}, d.spec.display_name), el("span", { class: "muted" }, `identical stratified split · test n=${meta.split.n_test} · seed ${meta.split.seed} · ${meta.timestamp_utc}`));
    const models = MODEL_ORDER.filter(m => meta.results[m]);
    if (this.view === "table") {
      const bestC = models.filter(m => !isQ(m)).sort((a, b) => meta.results[b].accuracy - meta.results[a].accuracy)[0], bestQ = models.filter(isQ).sort((a, b) => meta.results[b].accuracy - meta.results[a].accuracy)[0];
      const gap = meta.results[bestQ].accuracy - meta.results[bestC].accuracy;
      c.append(el("div", { class: "panel" }, head, el("div", { class: "stats", style: "margin-bottom:14px" }, stat(`${mlabel(bestC)} ${fmt(meta.results[bestC].accuracy)}`, "best classical (accuracy)"), stat(`${mlabel(bestQ)} ${fmt(meta.results[bestQ].accuracy)}`, "best quantum (accuracy)"), stat((gap >= 0 ? "+" : "") + fmt(gap, 3), "quantum − classical"), stat(`${meta.quantum_config.n_qubits} qubits`, meta.quantum_config.feature_reduction)),
        resultsTable(meta, { full: true }),
        el("div", { class: "infobox", style: "margin-top:12px" }, gap < 0 ? `On this split the best classical model outperforms the best quantum model by ${fmt(-gap, 3)} accuracy. The quantum models see only ${meta.quantum_config.n_qubits} PCA components of ${meta.preprocessing.n_features_after_preprocessing} preprocessed features and have ${meta.quantum_config.n_trainable_params} trainable parameters, so this is a capacity gap, not evidence against QML in general.` : `On this split the best quantum model matches or beats the best classical model by ${fmt(gap, 3)} accuracy; with a test set of ${meta.split.n_test} rows this is within noise until cross-validation confirms it.`)));
    }
    if (this.view === "metrics") {
      const rows = models.map(m => ({ m, v: meta.results[m][this.metric] })).sort((a, b) => (b.v ?? 0) - (a.v ?? 0));
      c.append(el("div", { class: "panel" }, head, el("h3", {}, `${METRIC_LABEL[this.metric]} by model`), el("div", { class: "bars" }, ...rows.map(r => barRow(r.m, r.v, r.m === rows[0].m, `${mlabel(r.m)}: ${fmt(r.v, 4)}`))),
        el("h3", {}, "All metrics"), el("div", { class: "cmgrid" }, ...METRICS.map(mt => el("div", { class: "cm" }, el("div", { class: "t" }, METRIC_LABEL[mt]), el("div", { class: "bars" }, ...models.map(m => el("div", { class: "brow", style: "grid-template-columns:90px 1fr 44px;font-size:11.5px" }, el("div", { class: "bl" }, (isQ(m) ? "⚛" : "") + mlabel(m).replace("Quantum ", "")), el("div", { class: "bt" }, el("i", { style: `width:${(meta.results[m][mt] || 0) * 100}%;background:${mcolor(m)}` })), el("div", { class: "bv" }, fmt(meta.results[m][mt], 2)))))))),
        el("h3", {}, "Computational cost"), el("div", { class: "bars" }, ...models.map(m => barRow(m, Math.min(1, (meta.training_info[m]?.train_time_s || 0) / Math.max(...models.map(x => meta.training_info[x]?.train_time_s || 0))), false, `${mlabel(m)}: ${fmt(meta.training_info[m]?.train_time_s, 2)} s training, ${fmt(meta.training_info[m]?.predict_ms_per_sample, 2)} ms/sample inference`, `${fmt(meta.training_info[m]?.train_time_s, 1)} s`))),
        el("p", { class: "muted", style: "font-size:12px" }, "Quantum timings are statevector simulation on CPU (noiseless), not hardware.")));
    }
    if (this.view === "roc") c.append(el("div", { class: "panel" }, head, rocChart(meta, models), el("p", { class: "muted", style: "font-size:12px" }, "ROC computed on the same hold-out test set for every model. Dashed lines are quantum models. AUC in the legend. SVM and QSVC use their decision margin as the score.")));
    if (this.view === "cm") c.append(el("div", { class: "panel" }, head, el("div", { class: "cmgrid" }, ...models.map(m => confusionCard(m, meta.results[m]))), el("p", { class: "muted", style: "font-size:12px" }, `Rows: actual class, columns: predicted class, on the ${meta.split.n_test}-row test set. Blue = correct, red = errors.`)));
    c.append(el("p", { class: "disclaimer" }, "Purpose: a fair scientific comparison of classical and quantum models on the same data, split and preprocessing. Neither is favoured. Single hold-out evaluation; stratified cross-validation is the next stage before conclusions."));
  },
};
function barRow(model, value, selected, tip, labelOverride) {
  const w = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return el("div", { class: "brow" + (selected ? " sel" : "") }, el("div", { class: "bl" }, isQ(model) ? el("span", { class: "q" }, "⚛") : null, mlabel(model)), el("div", { class: "bt", "data-tip": tip }, el("i", { style: `width:${w}%;background:${mcolor(model)}` })), el("div", { class: "bv" }, labelOverride ?? (value == null ? "—" : fmt(value, 3))));
}
function confusionCard(m, r) {
  const cm = r.confusion_matrix; const cell = (v, cls) => el("div", { class: "c " + cls }, String(v));
  return el("div", { class: "cm" }, el("div", { class: "t" }, el("span", {}, (isQ(m) ? "⚛ " : "") + mlabel(m)), el("span", { class: "muted" }, `acc ${fmt(r.accuracy, 2)}`)),
    el("div", { class: "g" }, el("div", {}), el("div", { class: "h" }, "pred 0"), el("div", { class: "h" }, "pred 1"), el("div", { class: "rl" }, "actual 0"), cell(cm.tn, "tn"), cell(cm.fp, "fp"), el("div", { class: "rl" }, "actual 1"), cell(cm.fn, "fn"), cell(cm.tp, "tp")),
    el("div", { class: "tiny", style: "margin-top:6px" }, `sens ${fmt(r.recall, 2)} · spec ${fmt(r.specificity, 2)} · prec ${fmt(r.precision, 2)}`));
}
function rocChart(meta, models) {
  const W = 520, H = 420, P = { l: 46, r: 16, t: 14, b: 40 }; const x = v => P.l + v * (W - P.l - P.r), y = v => H - P.b - v * (H - P.t - P.b);
  const svg = svgEl("svg", { class: "roc", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "ROC curves" });
  for (const t of [0, .2, .4, .6, .8, 1]) { svg.append(svgEl("line", { class: "grid", x1: x(t), x2: x(t), y1: y(0), y2: y(1) }), svgEl("line", { class: "grid", x1: x(0), x2: x(1), y1: y(t), y2: y(t) })); svg.append(svgEl("text", { x: x(t), y: y(0) + 16, "text-anchor": "middle" }, t.toFixed(1)), svgEl("text", { x: x(0) - 8, y: y(t) + 4, "text-anchor": "end" }, t.toFixed(1))); }
  svg.append(svgEl("line", { class: "diag", x1: x(0), y1: y(0), x2: x(1), y2: y(1) }));
  svg.append(svgEl("text", { x: (x(0) + x(1)) / 2, y: H - 6, "text-anchor": "middle" }, "False positive rate (1 − specificity)"), svgEl("text", { x: 12, y: (y(0) + y(1)) / 2, transform: `rotate(-90 12 ${(y(0) + y(1)) / 2})`, "text-anchor": "middle" }, "True positive rate (sensitivity)"));
  const legend = el("div", { class: "legend", style: "margin-top:10px" });
  for (const m of models) {
    const rc = meta.results[m].roc_curve; if (!rc) continue;
    const dpath = rc.fpr.map((f, i) => `${i ? "L" : "M"}${x(f).toFixed(1)},${y(rc.tpr[i]).toFixed(1)}`).join(" ");
    const p = svgEl("path", { class: "line" + (isQ(m) ? " q" : ""), d: dpath, stroke: mcolor(m) }); p.append(svgEl("title", {}, `${mlabel(m)} · AUC ${fmt(meta.results[m].roc_auc)}`)); svg.append(p);
    legend.append(el("span", {}, el("i", { style: `background:${mcolor(m)};${isQ(m) ? "border:1px dashed #333;background:repeating-linear-gradient(90deg," + mcolor(m) + " 0 4px,transparent 4px 7px)" : ""}` }), `${isQ(m) ? "⚛ " : ""}${mlabel(m)} · AUC ${fmt(meta.results[m].roc_auc)}`));
  }
  return el("div", { style: "max-width:560px" }, svg, legend);
}

/* ==================================================================== QUANTUM LAB */
Pages.quantum = {
  view: "models", disease: null,
  async render(c, params) {
    c.innerHTML = ""; await refreshDiseases(); const trained = trainedDiseases();
    if (!trained.length) { c.append(el("div", { class: "panel" }, el("p", { class: "muted" }, "Train a disease first."))); return; }
    this.view = params.view || this.view; this.disease = params.disease || this.disease || trained[0].name; if (!trained.some(d => d.name === this.disease)) this.disease = trained[0].name;
    const lab = await api(`/api/quantum/${this.disease}`);
    c.append(el("div", { class: "filters" }, segmented([["models", "Quantum Models"], ["circuit", "Circuit"], ["config", "Configuration"]], this.view, v => go("quantum", { view: v, disease: this.disease })), el("label", {}, "Disease", diseasePicker(this.disease, x => go("quantum", { view: this.view, disease: x })))));
    const models = Object.entries(lab.models);
    if (this.view === "models") {
      const cards = el("div", { class: "labcards" });
      for (const [m, q] of models) {
        cards.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, `⚛ ${mlabel(m)}`), el("span", { class: "badge q" }, q.type?.split(" (")[0] || "quantum")),
          el("p", { class: "muted" }, q.type),
          el("div", { class: "stats" }, stat(q.n_qubits, "qubits"), stat(q.n_trainable_params, "trainable params"), stat(q.circuit_depth, "circuit depth"), stat(fmt(q.results?.accuracy), "test accuracy")),
          el("dl", { class: "kv", style: "margin-top:12px;font-size:12.5px" }, el("dt", {}, "Feature encoding"), el("dd", {}, `${q.feature_map} feature map${q.entanglement ? " · " + q.entanglement + " entanglement" : ""}`), el("dt", {}, "Ansatz / kernel"), el("dd", {}, q.ansatz ? `${q.ansatz}, ${q.reps} rep(s)` : q.kernel), el("dt", {}, "Optimizer / classifier"), el("dd", {}, q.optimizer ? `${q.optimizer.toUpperCase()}, maxiter ${q.maxiter}` : q.classifier), el("dt", {}, "Measurement"), el("dd", {}, q.observable ? `⟨${q.observable}⟩ expectation, sign → class` : `fidelity kernel${q.n_support_vectors ? " · " + q.n_support_vectors + " support vectors" : ""}`), el("dt", {}, "Backend"), el("dd", {}, q.backend), el("dt", {}, "Feature reduction"), el("dd", {}, q.feature_reduction), el("dt", {}, "ROC-AUC / F1"), el("dd", {}, `${fmt(q.results?.roc_auc)} / ${fmt(q.results?.f1)}`), el("dt", {}, "Train time"), el("dd", {}, `${fmt(q.training_info?.train_time_s, 2)} s`)),
          q.loss_history?.length ? el("div", {}, el("h3", {}, "Training loss (COBYLA evaluations)"), sparkline(q.loss_history)) : null));
      }
      c.append(cards);
      const ref = Object.entries(lab.classical_reference).sort((a, b) => b[1].accuracy - a[1].accuracy)[0];
      c.append(el("div", { class: "panel", style: "margin-top:16px" }, el("h2", {}, "Reading these numbers"), el("p", { class: "muted" }, `Best classical reference on the same split: ${mlabel(ref[0])} at ${fmt(ref[1].accuracy)} accuracy. The quantum models operate on ${models[0]?.[1].n_qubits} PCA components of ${lab.n_features_after_preprocessing} preprocessed features. Simulation is exact and noiseless; on real hardware shot noise and gate errors would lower these figures. Qiskit ${lab.environment.qiskit}, qiskit-machine-learning ${lab.environment.qiskit_machine_learning}.`)));
    }
    if (this.view === "circuit") for (const [m, q] of models) c.append(el("div", { class: "panel", style: "margin-bottom:16px" }, el("div", { class: "panel-head" }, el("h2", {}, `⚛ ${mlabel(m)} — actual circuit`), el("span", { class: "muted" }, m === "quantum_vqc" ? "x[i] = encoded patient features (angles), θ[i] = trained parameters" : "feature-map circuit; kernel = |⟨φ(x)|φ(y)⟩|²")), el("pre", { class: "circuit" }, q.circuit_text || "circuit drawing unavailable")));
    if (this.view === "config") {
      const cfg = el("div", { class: "tablewrap" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "parameter"), ...models.map(([m]) => el("th", {}, mlabel(m))))), el("tbody", {}, ...[["type", "Type"], ["n_qubits", "Qubits"], ["feature_map", "Feature encoding"], ["entanglement", "Entanglement"], ["ansatz", "Ansatz"], ["reps", "Layers (reps)"], ["n_trainable_params", "Trainable parameters"], ["circuit_depth", "Circuit depth"], ["observable", "Observable"], ["optimizer", "Optimizer"], ["maxiter", "Max iterations"], ["kernel", "Kernel"], ["classifier", "Classifier"], ["backend", "Simulator / backend"], ["feature_reduction", "Feature reduction"]].map(([k, l]) => el("tr", {}, el("td", { class: "muted" }, l), ...models.map(([, q]) => el("td", {}, q[k] == null ? "—" : String(q[k]))))))));
      c.append(el("div", { class: "panel" }, el("h2", {}, "Configuration"), cfg, el("p", { class: "muted", style: "font-size:12px;margin-top:10px" }, "Every value is read from the saved model artefacts; nothing is inferred. Change them with scripts/train.py flags (--n-qubits, --reps, --feature-map, --entanglement, --optimizer, --maxiter) and re-train.")));
    }
  },
};
function sparkline(vals) {
  const W = 300, H = 60, mn = Math.min(...vals), mx = Math.max(...vals); const y = v => H - 4 - ((v - mn) / ((mx - mn) || 1)) * (H - 8);
  const d = vals.map((v, i) => `${i ? "L" : "M"}${(i / (vals.length - 1) * W).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return el("div", {}, svgEl("svg", { class: "spark", viewBox: `0 0 ${W} ${H}` }, svgEl("path", { d })), el("div", { class: "tiny" }, `${vals.length} evaluations · loss ${fmt(vals[0], 3)} → ${fmt(vals[vals.length - 1], 3)}`));
}

/* ==================================================================== SETTINGS */
Pages.settings = {
  async render(c) {
    c.innerHTML = ""; const s = Settings.get(); const ocr = await api("/api/ocr/status").catch(() => null);
    const crit = el("select", { onchange: e => { Settings.set({ criterion: e.target.value }); toast("Saved"); } }, ...["accuracy", "f1", "roc_auc"].map(m => el("option", { value: m, selected: m === s.criterion }, METRIC_LABEL[m])));
    c.append(el("div", { class: "panel", style: "max-width:820px" }, el("h2", {}, "Prediction defaults"),
      el("div", { class: "grid2" }, el("label", {}, "\"Use best performing model\" chooses by", crit),
        el("label", { class: "check" }, el("input", { type: "checkbox", checked: s.preferProbability, onchange: e => Settings.set({ preferProbability: e.target.checked }) }), "Prefer models with a probability-like score when choosing \"best\" (skips SVM / QSVC margins)"),
        el("label", { class: "check" }, el("input", { type: "checkbox", checked: s.compareAll, onchange: e => Settings.set({ compareAll: e.target.checked }) }), "Run all trained models for comparison by default")),
      el("h3", {}, "NVIDIA OCR"),
      ocr ? el("dl", { class: "kv" }, el("dt", {}, "Engine"), el("dd", {}, `${ocr.engine} · ${ocr.model}`), el("dt", {}, "Status"), el("dd", {}, ocr.available ? el("span", { class: "badge ok" }, `available · ${ocr.mode}`) : el("span", { class: "badge warn" }, "not configured")), el("dt", {}, "Local GPU"), el("dd", {}, ocr.gpu?.present ? `${ocr.gpu.name} · driver ${ocr.gpu.driver} · ${ocr.gpu.memory}${ocr.gpu.container_runtime ? " · nvidia container toolkit present" : " · no NVIDIA container toolkit (local NIM cannot use the GPU yet)"}` : "none detected"), el("dt", {}, "Hosted mode"), el("dd", {}, ocr.requirements.hosted), el("dt", {}, "Local mode"), el("dd", {}, ocr.requirements.local), ocr.reason ? el("dt", {}, "Note") : null, ocr.reason ? el("dd", {}, ocr.reason) : null) : el("p", { class: "muted" }, "status unavailable"),
      el("h3", {}, "Verification policy"),
      el("dl", { class: "kv" }, el("dt", {}, "New patient"), el("dd", {}, "email verified with a 6-digit code (10 min, single use, 5 attempts, resend cooldown 60 s) before the record exists; phone and email are unique per record"), el("dt", {}, "Routine access"), el("dd", {}, "search, open record, history, analyse, save — operator session only, no re-verification"), el("dt", {}, "Sensitive"), el("dd", {}, "changing a patient's email requires a code sent to the new address")),
      el("h3", {}, "Operator account"),
      (() => { const o = el("input", { type: "password", placeholder: "current password", autocomplete: "current-password" }), n = el("input", { type: "password", placeholder: "new password (min 8)", autocomplete: "new-password" });
        return el("div", { class: "grid2" }, el("label", {}, "Current password", o), el("label", {}, "New password", n), el("div", { style: "align-self:end" }, el("button", { class: "btn sm", onclick: async () => { try { const r = await postJson("/api/auth/password", { old_password: o.value, new_password: n.value }); toast(r.message, 5000); Auth.showLogin(); } catch (e) { toast(e.message); } } }, "Change password"))); })(),
      el("h3", {}, "About"), el("dl", { class: "kv" }, el("dt", {}, "Backend"), el("dd", {}, `${location.origin} · FastAPI · SQLite · API docs at `, el("a", { href: "/docs", target: "_blank" }, "/docs")), el("dt", {}, "Quantum"), el("dd", {}, "Qiskit 2.x — variational classifier (EstimatorQNN) and fidelity-kernel SVM, exact statevector simulation"), el("dt", {}, "Limitations"), el("dd", {}, "small datasets · single hold-out split · 4-qubit capacity · no hyper-parameter tuning · noiseless simulation · no clinical validation")),
      el("p", { class: "disclaimer" }, "Patient information and model outputs are intended for authorized research and decision-support use. This system is not a substitute for professional medical diagnosis.")));
  },
};
