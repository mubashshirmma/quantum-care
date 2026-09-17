/* Patient Analysis (manual + NVIDIA OCR) and Patient Records. Prediction always uses saved trained models. */
const categoryOf = name => (State.diseases.find(d => d.name === name) || {}).category || "other";
const RISK = { low: { icon: "✓", text: "Low risk", cls: "low" }, borderline: { icon: "◐", text: "Borderline risk", cls: "borderline" }, elevated: { icon: "⚠", text: "High risk", cls: "elevated" } };
const initials = n => (n || "?").split(/\s+/).map(w => w[0]).join("").slice(0, 2).toUpperCase();
const riskPill = h => el("span", { class: `risk-pill ${h.risk_level || "low"}` }, el("span", { style: "font-family:var(--emoji)" }, (RISK[h.risk_level] || RISK.low).icon), h.probability == null ? (RISK[h.risk_level] || RISK.low).text : `${pct(h.probability)} score`);
window.historyItem = (h, compact = false) => {
  const view = el("button", { class: "btn sm", onclick: () => go("analysis", { analysis: h.analysis_id }) }, "View Analysis");
  const src = el("span", { class: "srcchip" }, SOURCE_LABEL[h.input_source] || "Manual entry");
  const text = el("div", {}, el("div", { class: "hn" }, h.patient_name || h.patient_id, " ", el("span", { class: "pidsm" }, h.patient_id)), el("div", { class: "hm" }, `${h.display_name || h.disease} · ${mlabel(h.model)} · ${fmtTime(h.created_at)} · `, src));
  if (compact) { text.append(el("div", { style: "display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap" }, riskPill(h), view)); return el("div", { class: "card hitem compact" }, el("div", { class: "icon" }, dIcon({ category: h.category || categoryOf(h.disease) })), text); }
  return el("div", { class: "card hitem" }, el("div", { class: "icon" }, dIcon({ category: h.category || categoryOf(h.disease) })), text, riskPill(h), el("div", { class: "hact", style: "display:flex;gap:6px" }, view, el("button", { class: "btn ghost sm", onclick: () => go("records", { id: h.patient_id }) }, "Record →")));
};

const PA = { patient: null, record: null, disease: null, schema: null, models: null, values: {}, modelChoice: "best", compareAll: true, result: null, savedId: null, inputSource: "manual", ocr: null };
async function loadPatient(id) { const rec = await api(`/api/patients/${id}`); PA.patient = rec.patient; PA.record = rec; return rec; }

/* ==================================================================== ANALYSIS PAGE */
Pages.analysis = {
  async render(c, params) {
    c.innerHTML = ""; PA.compareAll = Settings.get().compareAll; await refreshDiseases();
    if (params.analysis) return renderSavedAnalysis(c, params.analysis);
    if (params.demo && params.disease) { PA.patient = null; PA.disease = params.disease; PA.schema = null; PA.values = {}; PA.result = null; PA.inputSource = "manual"; await loadDisease(); fillTypical(); if (params.model) PA.modelChoice = params.model; renderFlow(c); const err = el("div", { class: "errbox", hidden: true }); c.append(err); return analyze(c, err); }
    if (params.id) { try { if (!PA.patient || PA.patient.patient_id !== params.id.toUpperCase()) { PA.result = null; PA.values = {}; } await loadPatient(params.id); } catch (e) { c.append(el("div", { class: "panel" }, el("div", { class: "errbox" }, e.message))); return; } }
    if (params.disease && params.disease !== PA.disease) { PA.disease = params.disease; PA.schema = null; PA.values = {}; PA.result = null; }
    const mode = params.mode;
    if (!mode && !params.id) return renderHub(c);
    if (mode === "new") return renderPanel(c, "New patient", box => renderRegisterForm(box, params));
    if (mode === "existing") return renderFind(c, params, "analysis");
    if (mode === "ocr") { PA.inputSource = "nvidia_ocr"; return renderOCR(c, params); }
    /* manual (default when a patient id is present) */
    PA.inputSource = params.source || "manual";
    if (PA.result && PA.result.patient_id === (PA.patient?.patient_id || null) && !params.disease) return renderResult(c, PA.result);
    renderFlow(c);
  },
};
function renderPanel(c, title, fill) { const p = el("div", { class: "panel", style: "max-width:720px" }); fill(p); c.append(p); }
function renderHub(c) {
  const modes = [["new", "userplus", "New Patient", "register with email verification (once)"], ["existing", "search", "Existing Patient", "find by Patient ID · instant"], ["manual", "keyboard", "Manual Entry", "type the disease-specific features"], ["ocr", "scan", "NVIDIA OCR", "upload a medical report (JPG · PNG · PDF)"]];
  c.append(el("div", { class: "panel", style: "margin-bottom:16px" }, el("div", { class: "panel-head" }, el("div", {}, el("h2", {}, "Patient Analysis"), el("p", { class: "muted" }, "Patient → disease → disease-specific features → saved trained model → prediction → explanation → saved to the patient record.")), PA.patient ? el("span", { class: "badge ok" }, `current patient: ${PA.patient.patient_id}`) : null),
    el("div", { class: "modes" }, ...modes.map(([m, i, t, d]) => el("div", { class: "mode", onclick: () => go("analysis", { mode: m, ...(PA.patient ? { id: PA.patient.patient_id } : {}) }) }, el("div", { class: "mi" }, icon(i, 24)), el("div", { class: "mt" }, t), el("div", { class: "md" }, d))))));
  const trained = trainedDiseases();
  c.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, "Diseases with trained models"), el("span", { class: "muted" }, "each has its own feature form")), el("div", { class: "quick" }, ...trained.map(d => el("div", { class: "qcard", onclick: () => go("analysis", { mode: "manual", disease: d.name, ...(PA.patient ? { id: PA.patient.patient_id } : {}) }) }, el("div", { class: "icon" }, dIcon(d)), el("div", {}, el("div", { class: "name" }, d.display_name), el("div", { class: "meta" }, `${d.n_features} features · best ${mlabel(d.best_model.name)} ${pct(d.best_model.accuracy)}`)))))));
}

/* ==================================================================== RECORDS PAGE */
Pages.records = {
  async render(c, params) {
    c.innerHTML = ""; await refreshDiseases();
    if (params.id) { try { await loadPatient(params.id); } catch (e) { c.append(el("div", { class: "panel" }, el("div", { class: "errbox" }, e.message), el("a", { class: "btn sm", href: "#/records" }, "← Search"))); return; } return renderRecord(c); }
    const tab = params.tab || "search";
    c.append(el("div", { class: "subnav" }, segmented([["search", "Search"], ["list", "Patient List"], ["history", "Analysis History"]], tab, t => go("records", { tab: t }))));
    if (tab === "search") return renderFind(c, params, "records");
    if (tab === "list") { const list = await api("/api/patients?q=&limit=50"); return c.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, `Patients (${list.length})`), el("a", { class: "btn primary sm", href: "#/analysis?mode=new" }, icon("plus", 15), "New Patient")), list.length ? el("div", { class: "plist" }, ...list.map(patientRow)) : el("p", { class: "muted" }, "No patients registered yet."))); }
    const items = await api("/api/analyses/recent?limit=100");
    c.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("div", {}, el("h2", {}, "Analysis History"), el("p", { class: "muted" }, "Every analysis is stored separately on its patient record.")), el("a", { class: "btn primary sm", href: "#/analysis" }, icon("plus", 15), "New Analysis")), items.length ? el("div", { class: "hlist" }, ...items.map(h => historyItem(h))) : el("p", { class: "muted" }, "No analyses saved yet.")));
  },
};

/* ---------- find / register / verify ---------- */
function renderFind(c, params, context) {
  const grid = el("div", { class: "find-grid" }); c.append(grid);
  const results = el("div", { class: "plist" }); const input = el("input", { placeholder: "Patient ID (PAT-2026-000001), name, phone or email", autofocus: true });
  let t; const open = p => context === "analysis" ? go("analysis", { id: p.patient_id, mode: "manual", ...(params.disease ? { disease: params.disease } : {}) }) : go("records", { id: p.patient_id });
  const search = async () => { clearTimeout(t); t = setTimeout(async () => { const q = input.value.trim(); results.innerHTML = "";
    try { const list = await api(`/api/patients?q=${encodeURIComponent(q)}&limit=12`);
      if (!list.length) results.append(el("p", { class: "muted" }, q ? "No patient found. Register a new patient on the right." : "No patients registered yet."));
      if (list.length === 1 && q && list[0].patient_id.toLowerCase() === q.toLowerCase()) { results.append(el("div", { class: "badge ok", style: "margin-bottom:8px" }, "Patient Found ✓")); }
      list.forEach(p => results.append(patientRow(p, open))); } catch (e) { results.append(el("div", { class: "errbox" }, e.message)); } }, 160); };
  input.addEventListener("input", search); input.addEventListener("keydown", e => { if (e.key === "Enter") { clearTimeout(t); const q = input.value.trim(); if (/^pat-\d{4}-\d{6}$/i.test(q)) open({ patient_id: q.toUpperCase() }); else search(); } });
  grid.append(el("div", { class: "panel" }, el("h2", {}, "Returning patient"), el("p", { class: "muted" }, "Patient ID is the fastest lookup: type it and press Enter to open the record instantly. No re-verification for routine access."),
    el("div", { class: "searchbar" }, input, el("button", { class: "btn primary", onclick: () => { clearTimeout(t); search(); } }, "Search")), results,
    el("p", { class: "tiny", style: "margin-top:14px" }, "Access is protected by your operator session. A Patient ID identifies a record; it is not a credential.")));
  search();
  const reg = el("div", { class: "panel" }); grid.append(reg); renderRegisterForm(reg, params);
}
const patientRow = (p, open) => el("div", { class: "card pitem", onclick: () => (open || (x => go("records", { id: x.patient_id })))(p) },
  el("div", {}, el("div", {}, el("span", { class: "pid" }, p.patient_id), " ", el("b", {}, p.name)), el("div", { class: "pm" }, `${p.email} · ${p.phone} · ${p.email_verified ? "email verified ✓" : "email not verified"}`)), el("button", { class: "btn sm" }, "Open →"));

function renderRegisterForm(box, params = {}, prefill = {}) {
  box.innerHTML = "";
  const name = el("input", { placeholder: "Full name", value: prefill.name || "", autocomplete: "off" }), phone = el("input", { placeholder: "+91 98765 43210", value: prefill.phone || "", inputmode: "tel", autocomplete: "off" }), email = el("input", { placeholder: "name@example.com", type: "email", value: prefill.email || "", autocomplete: "off" });
  const err = el("div", { class: "errbox", hidden: true }), extra = el("div", {}); const showE = m => { err.textContent = m; err.hidden = false; };
  const openExisting = p => go("analysis", { id: p.patient_id, mode: "manual", ...(params.disease ? { disease: params.disease } : {}) });
  const submit = async () => {
    err.hidden = true; extra.innerHTML = "";
    if (name.value.trim().length < 2) return showE("Please enter the patient's name."); if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email.value.trim())) return showE("Please enter a valid email address."); if (phone.value.replace(/\D/g, "").length < 6) return showE("Please enter a valid phone number.");
    btn.disabled = true; btn.textContent = "Checking…";
    try {
      const r = await postJson("/api/patients/register", { name: name.value.trim(), phone: phone.value.trim(), email: email.value.trim() });
      if (r.status === "exists") { const p = r.patient;
        extra.append(el("div", { class: "infobox" }, el("b", {}, "Patient Found ✓"), ` — matched by ${r.matched_by.join(" and ")}.${r.mismatch ? " Note: " + r.mismatch + "." : ""} No new record was created.`),
          el("div", { class: "card", style: "margin-top:10px" }, el("dl", { class: "kv" }, el("dt", {}, "Name"), el("dd", {}, p.name), el("dt", {}, "Patient ID"), el("dd", { class: "pid" }, p.patient_id), el("dt", {}, "Phone"), el("dd", {}, p.phone), el("dt", {}, "Email"), el("dd", {}, p.email)),
            el("div", { class: "actions" }, el("button", { class: "btn sm", onclick: () => go("records", { id: p.patient_id }) }, "Open Patient"), el("button", { class: "btn primary sm", onclick: () => openExisting(p) }, icon("plus", 15), "Add New Analysis")))); }
      else if (r.status === "conflict") { extra.append(el("div", { class: "errbox" }, el("b", {}, "Conflict: "), r.message), el("div", { class: "plist" }, el("div", { class: "muted tiny" }, "Phone belongs to:"), patientRow(r.phone_patient), el("div", { class: "muted tiny" }, "Email belongs to:"), patientRow(r.email_patient))); }
      else renderVerify(box, r, { name: name.value.trim(), phone: phone.value.trim(), email: email.value.trim() }, params);
    } catch (e) { showE(e.message); } finally { btn.disabled = false; btn.textContent = "Continue → verify email"; }
  };
  const btn = el("button", { class: "btn primary", onclick: submit }, "Continue → verify email");
  box.append(el("h2", {}, "New patient"), el("p", { class: "muted" }, "Phone and email must be unique. The record is created only after the email is verified with a one-time code — once, not on later visits."),
    el("div", { class: "grid2" }, el("label", {}, "Patient name *", name), el("label", {}, "Phone *", phone), el("label", {}, "Email * (verification code is sent here)", email)), err, extra, el("div", { class: "actions" }, btn));
}
function renderVerify(box, pending, form, params = {}) {
  box.innerHTML = "";
  const code = el("input", { placeholder: "000000", inputmode: "numeric", maxlength: "6", autocomplete: "one-time-code" }); const err = el("div", { class: "errbox", hidden: true }); const state = el("span", { class: "vstate pending" }, "⏳ verification pending");
  const info = el("p", { class: "muted" }, `A 6-digit code was sent to ${pending.email_masked}. Expires ${new Date(pending.expires_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}, single use. ${pending.attempts_left} attempts left.`);
  const dev = pending.dev_code ? el("div", { class: "devcode" }, "DEV MODE — no mail server configured, so the code is shown here (and logged on the server): ", el("b", {}, pending.dev_code)) : null;
  const resendBtn = el("button", { class: "btn sm", disabled: true }, `Resend code (${pending.resend_after_s}s)`); let left = pending.resend_after_s;
  const tick = setInterval(() => { left--; if (left <= 0) { clearInterval(tick); resendBtn.disabled = false; resendBtn.textContent = "Resend code"; } else resendBtn.textContent = `Resend code (${left}s)`; }, 1000);
  resendBtn.onclick = async () => { try { const r = await postJson(`/api/patients/register/${pending.pending_id}/resend`, {}); toast("New code sent"); renderVerify(box, r, form, params); } catch (e) { err.textContent = e.message; err.hidden = false; } };
  const verify = async () => {
    err.hidden = true; if (!/^\d{6}$/.test(code.value.trim())) { err.textContent = "Enter the 6-digit code."; err.hidden = false; return; } vbtn.disabled = true;
    try { const r = await postJson(`/api/patients/register/${pending.pending_id}/verify`, { code: code.value.trim() }); clearInterval(tick); state.className = "vstate ok"; state.textContent = "✓ email verified";
      if (r.created) { toast(`Patient record created: ${r.patient.patient_id}`, 5000); go("analysis", { id: r.patient.patient_id, mode: params.mode === "ocr" ? "ocr" : "manual", ...(params.disease ? { disease: params.disease } : {}) }); }
      else { toast("Email updated"); go("records", { id: r.patient.patient_id }); } }
    catch (e) { err.textContent = e.message; err.hidden = false; code.value = ""; code.focus(); if (e.detail && e.detail.attempts_left != null) info.textContent = `${e.detail.attempts_left} attempts left.`; if ((e.detail && e.detail.expired) || e.status === 404 || e.status === 410 || e.status === 409) setTimeout(() => renderRegisterForm(box, params, form), 2000); }
    finally { vbtn.disabled = false; }
  };
  const vbtn = el("button", { class: "btn primary", onclick: verify }, pending.purpose === "email_change" ? "Verify & update email" : "Verify & create record");
  code.addEventListener("keydown", e => { if (e.key === "Enter") verify(); }); code.addEventListener("input", () => { if (code.value.length === 6) verify(); });
  const fix = el("div", { hidden: true }); const newEmail = el("input", { type: "email", placeholder: "corrected email", value: form.email });
  fix.append(el("div", { class: "grid2", style: "margin-top:8px" }, el("label", {}, "Corrected email", newEmail), el("div", { style: "align-self:end" }, el("button", { class: "btn sm", onclick: async () => { try { const r = await patchJson(`/api/patients/register/${pending.pending_id}/email`, { email: newEmail.value.trim() }); toast("Code sent to the corrected address"); renderVerify(box, r, { ...form, email: newEmail.value.trim() }, params); } catch (e) { err.textContent = e.message; err.hidden = false; } } }, "Send code to this address"))));
  box.append(el("div", { class: "panel-head" }, el("h2", {}, "Verify email"), state), el("p", {}, el("b", {}, form.name), form.phone ? ` · ${form.phone}` : ""), info, dev, el("div", { class: "otp", style: "margin:12px 0" }, code, vbtn), err,
    el("div", { class: "actions", style: "justify-content:flex-start" }, resendBtn, el("button", { class: "btn sm", onclick: () => (fix.hidden = !fix.hidden) }, "Wrong email? Correct it"), el("button", { class: "btn ghost sm", onclick: async () => { clearInterval(tick); try { await api(`/api/patients/register/${pending.pending_id}`, { method: "DELETE" }); } catch {} renderRegisterForm(box, params, form); } }, "Cancel")), fix,
    el("p", { class: "tiny", style: "margin-top:12px" }, "No record exists until the code is verified. Codes are stored hashed, expire after 10 minutes and are single use."));
  setTimeout(() => code.focus(), 50);
}

/* ---------- patient record / profile ---------- */
function renderRecord(c) {
  c.innerHTML = ""; const p = PA.patient, rec = PA.record; const wrap = el("div", { class: "flow", style: "max-width:1100px" }); c.append(wrap); const emailBox = el("div", { hidden: true });
  wrap.append(el("div", { class: "panel" }, el("div", { class: "record-head" }, el("div", { class: "avatar" }, initials(p.name)),
    el("div", {}, el("h2", {}, p.name), el("div", {}, el("span", { class: "pid" }, p.patient_id), "  ", el("span", { class: p.email_verified ? "verified" : "unverified" }, p.email_verified ? "✓ email verified" : "⚠ email not verified")), el("div", { class: "muted", style: "font-size:12.5px" }, `${p.email} · ${p.phone} · registered ${fmtTime(p.created_at)}${p.created_by ? " by " + p.created_by : ""} · ${rec.analyses.length} analyses`)),
    el("div", { class: "actions", style: "margin:0" }, el("a", { class: "btn sm", href: `/api/patients/${p.patient_id}/export`, download: `${p.patient_id}.json` }, icon("download", 16), "Export Record"), el("button", { class: "btn sm", onclick: () => (emailBox.hidden = !emailBox.hidden) }, "Change email"), el("button", { class: "btn primary lg", onclick: () => go("analysis", { id: p.patient_id, mode: "manual" }) }, icon("plus", 18), "New Analysis"))), emailBox));
  const ne = el("input", { type: "email", placeholder: "new email address" }), eerr = el("div", { class: "errbox", hidden: true });
  emailBox.append(el("div", { class: "infobox", style: "margin-top:14px" }, "Sensitive action: a verification code is sent to the new address; the record changes only after it is confirmed."), el("div", { class: "grid2" }, el("label", {}, "New email", ne), el("div", { style: "align-self:end" }, el("button", { class: "btn sm", onclick: async () => { eerr.hidden = true; try { const r = await postJson(`/api/patients/${p.patient_id}/email`, { email: ne.value.trim() }); emailBox.innerHTML = ""; renderVerify(emailBox, r, { name: p.name, phone: p.phone, email: ne.value.trim() }); } catch (e) { eerr.textContent = e.message; eerr.hidden = false; } } }, "Send verification code"))), eerr);
  wrap.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, "Start a new analysis"), el("div", { class: "actions", style: "margin:0" }, el("a", { class: "btn sm", href: `#/analysis?id=${p.patient_id}&mode=ocr` }, icon("scan", 16), "From medical report (NVIDIA OCR)"))),
    el("div", { class: "quick" }, ...trainedDiseases().map(d => el("div", { class: "qcard", onclick: () => go("analysis", { id: p.patient_id, mode: "manual", disease: d.name }) }, el("div", { class: "icon" }, dIcon(d)), el("div", {}, el("div", { class: "name" }, d.display_name), el("div", { class: "meta" }, `${d.n_features} features · best ${mlabel(d.best_model.name)} ${pct(d.best_model.accuracy)}`)))))));
  const tl = el("div", { class: "timeline" });
  rec.analyses.forEach(a => tl.append(el("div", { class: "tl" }, historyItem({ ...a, patient_name: p.name }))));
  wrap.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, `Analysis history (${rec.analyses.length})`), el("span", { class: "muted" }, "one record, many analyses")), rec.analyses.length ? tl : el("p", { class: "muted" }, "No analyses yet for this patient.")));
}

/* ==================================================================== MANUAL FLOW */
function renderFlow(c) {
  c.innerHTML = ""; const flow = el("div", { class: "flow" }); c.append(flow); const p = PA.patient;
  flow.append(el("div", { class: "panel stepcard" }, el("div", { class: "panel-head" }, el("h2", {}, el("span", { class: "num" }, "1"), p ? "Patient & disease" : "Disease (quick check — no patient record, nothing will be saved)"),
    p ? el("a", { class: "btn ghost sm", href: `#/records?id=${p.patient_id}` }, "← Record") : el("a", { class: "btn ghost sm", href: "#/analysis?mode=existing" }, "Select a patient")),
    p ? el("div", { class: "record-head", style: "margin-bottom:10px" }, el("div", { class: "avatar" }, initials(p.name)), el("div", {}, el("b", {}, p.name), " ", el("span", { class: "pid" }, p.patient_id), el("div", { class: "muted", style: "font-size:12px" }, `${p.email_verified ? "✓ verified email" : "⚠ unverified email"} · ${PA.record?.analyses.length ?? 0} previous analyses`)), el("span", { class: "srcchip" }, SOURCE_LABEL[PA.inputSource])) : null,
    el("label", {}, "Disease", el("select", { onchange: async e => { PA.disease = e.target.value || null; PA.schema = null; PA.values = {}; await loadDisease(); renderFlow(c); } }, el("option", { value: "" }, "Select disease ▾"), ...State.diseases.map(d => el("option", { value: d.name, disabled: !d.trained, selected: d.name === PA.disease }, `${d.icon} ${d.display_name}${d.trained ? "" : " (not trained)"}`))))));
  if (!PA.disease) return;
  if (!PA.schema) { flow.append(el("div", { class: "panel stepcard" }, el("p", { class: "muted" }, "Loading form…"))); loadDisease().then(() => renderFlow(c)).catch(e => toast(e.message)); return; }
  const s = PA.schema; const grid = el("div", { class: "field-grid" }); for (const f of s.features) grid.append(fieldFor(f));
  flow.append(el("div", { class: "panel stepcard" }, el("div", { class: "panel-head" }, el("h2", {}, el("span", { class: "num" }, "2"), `Patient information · ${s.display_name}`), el("span", { class: "muted" }, `${s.features.length} fields required by this trained model`)),
    s.target_documented ? null : el("div", { class: "warnbox" }, "⚠ For this dataset the meaning of the label is an assumption, not documented by its source."), grid,
    el("div", { class: "actions", style: "justify-content:flex-start" }, el("button", { class: "btn sm", onclick: () => { fillTypical(); renderFlow(c); } }, "Fill with typical values (median)"), el("span", { class: "tiny" }, "for demos; ranges under each field come from the training data"))));
  const m = PA.models; const mbox = el("div", { class: "panel stepcard" }, el("h2", {}, el("span", { class: "num" }, "3"), "Model"));
  if (m?.trained) mbox.append(el("label", { class: "toggle" }, el("input", { type: "radio", name: "mchoice", checked: PA.modelChoice === "best", onchange: () => { PA.modelChoice = "best"; renderFlow(c); } }), el("div", {}, el("div", { class: "mn" }, `Use best performing model — ${mlabel(m.best)}`), el("div", { class: "mm" }, `best ${m.criterion} on ${m.evaluated_on}`))),
    el("p", { class: "muted", style: "margin:10px 0 6px" }, "or pick a specific trained model:"), el("div", { class: "models" }, ...m.models.map(x => el("label", { class: "mopt" + (PA.modelChoice === x.name ? " sel" : "") }, el("input", { type: "radio", name: "mchoice", checked: PA.modelChoice === x.name, onchange: () => { PA.modelChoice = x.name; renderFlow(c); } }), el("div", {}, el("div", { class: "mn" }, (x.is_quantum ? "⚛ " : "") + x.label, x.is_best ? el("span", { class: "badge ok", style: "margin-left:6px" }, "best") : null), el("div", { class: "mm" }, `acc ${fmt(x.metrics.accuracy)} · AUC ${fmt(x.metrics.roc_auc)}${x.has_probability ? "" : " · margin score"}${x.explainable ? " · explainable" : ""}`))))),
    el("label", { class: "check" }, el("input", { type: "checkbox", checked: PA.compareAll, onchange: e => (PA.compareAll = e.target.checked) }), "Compare all trained models in the result"));
  else mbox.append(el("p", { class: "muted" }, "No trained models for this disease.")); flow.append(mbox);
  const errBox = el("div", { class: "errbox", hidden: true });
  flow.append(el("div", { class: "panel stepcard", style: "text-align:center" }, el("button", { class: "btn primary lg", onclick: () => analyze(c, errBox) }, icon("activity", 18), "Analyze Patient"), el("p", { class: "tiny", style: "margin-top:10px" }, "Uses the saved, already-trained model. Nothing is retrained. The result is stored on the record only when you choose Save."), errBox));
}
async function loadDisease() { if (!PA.disease) return; const st = Settings.get(); [PA.schema, PA.models] = await Promise.all([api(`/api/diseases/${PA.disease}/form`), api(`/api/diseases/${PA.disease}/models?criterion=${st.criterion}&prefer_probability=${st.preferProbability}`)]); if (PA.modelChoice !== "best" && !PA.models.models.some(x => x.name === PA.modelChoice)) PA.modelChoice = "best"; }
function fillTypical() { for (const f of PA.schema.features) PA.values[f.name] = f.type === "number" ? f.range.median : f.options.slice().sort((a, b) => b.count - a.count)[0].value; }
function fieldFor(f) {
  const v = PA.values[f.name] ?? ""; let input;
  if (f.type === "select") input = el("select", { onchange: e => { PA.values[f.name] = e.target.value; } }, el("option", { value: "" }, "Select…"), ...f.options.map(o => el("option", { value: o.value, selected: String(v) === o.value }, o.label)));
  else input = el("input", { type: "number", step: f.step, value: v, placeholder: `${f.range.min}–${f.range.max}`, oninput: e => { PA.values[f.name] = e.target.value; validateField(f, e.target); } });
  const err = el("div", { class: "ferr" }); input._err = err;
  return el("label", { class: "field" }, el("span", {}, f.label, f.unit ? el("span", { class: "unit" }, ` (${f.unit})`) : null, " *"), input, err, el("span", { class: "fhint" }, f.hint));
}
function validateField(f, input) { const v = input.value; let msg = ""; if (v !== "" && f.type === "number") { const n = Number(v); if (Number.isNaN(n)) msg = "must be a number"; else if (n < f.range.min || n > f.range.max) msg = `outside training range ${f.range.min}–${f.range.max} (allowed, but less reliable)`; } input.classList.toggle("invalid", !!msg && !msg.startsWith("outside")); input._err.textContent = msg; return msg; }
function clientValidate() { const errors = []; for (const f of PA.schema.features) { const v = PA.values[f.name]; if (v === undefined || v === "" || v === null) errors.push(`${f.label} is required`); else if (f.type === "number" && Number.isNaN(Number(v))) errors.push(`${f.label} must be a number`); } return errors; }
async function analyze(c, errBox) {
  errBox.hidden = true; const errors = clientValidate(); if (errors.length) { errBox.textContent = "Please fix before analysing:\n• " + errors.join("\n• "); errBox.hidden = false; return; }
  const overlay = $("#analyzing"), steps = $$("#an-steps li"); steps.forEach(li => (li.className = "")); overlay.hidden = false; const t0 = Date.now(); const tick = i => steps.forEach((li, j) => (li.className = j < i ? "done" : j === i ? "doing" : "")); tick(0); const timers = [setTimeout(() => tick(1), 450), setTimeout(() => tick(2), 950)];
  try { const st = Settings.get(); const body = { data: PA.values, model: PA.compareAll ? "all" : PA.modelChoice, criterion: st.criterion, prefer_probability: st.preferProbability, explain: true, patient_name: PA.patient?.name || null };
    let res = await postJson(`/api/predict/${PA.disease}`, body);
    if (PA.compareAll && PA.modelChoice !== "best" && res.models[PA.modelChoice]) { const single = await postJson(`/api/predict/${PA.disease}`, { ...body, model: PA.modelChoice }); res = { ...single, models: res.models }; }
    res.patient_id = PA.patient?.patient_id || null; res.input_source = PA.inputSource; res.ocr_summary = PA.ocr ? { engine: PA.ocr.engine, mean_confidence: PA.ocr.mean_confidence ?? null, filename: PA.ocr.filename ?? null } : null;
    await new Promise(r => setTimeout(r, Math.max(0, 1400 - (Date.now() - t0)))); tick(3); PA.result = res; PA.savedId = null; renderResult(c, res);
  } catch (e) { errBox.textContent = "Analysis failed: " + e.message; errBox.hidden = false; } finally { timers.forEach(clearTimeout); overlay.hidden = true; }
}

/* ==================================================================== NVIDIA OCR FLOW */
async function renderOCR(c, params) {
  const status = await api("/api/ocr/status");
  const wrap = el("div", { class: "flow", style: "max-width:1200px" }); c.append(wrap); const p = PA.patient;
  wrap.append(el("div", { class: "panel stepcard" }, el("div", { class: "panel-head" }, el("h2", {}, el("span", { class: "num" }, "1"), "Patient & disease"), status.available ? el("span", { class: "badge ok" }, `NVIDIA OCR · ${status.mode} · ${status.model}`) : el("span", { class: "badge warn" }, "NVIDIA OCR not configured")),
    p ? el("div", { class: "record-head", style: "margin-bottom:10px" }, el("div", { class: "avatar" }, initials(p.name)), el("div", {}, el("b", {}, p.name), " ", el("span", { class: "pid" }, p.patient_id)), el("a", { class: "btn sm", href: "#/analysis?mode=existing" }, "Change")) : el("div", { class: "infobox", style: "margin-bottom:10px" }, "No patient selected — the result can be viewed but not saved. ", el("a", { href: "#/analysis?mode=existing" }, "Select a patient"), " or ", el("a", { href: "#/analysis?mode=new" }, "register a new one"), "."),
    el("label", {}, "Disease model to map the report onto", el("select", { onchange: e => { PA.disease = e.target.value || null; PA.schema = null; PA.values = {}; loadDisease().then(() => renderOCR(c, params)); } }, el("option", { value: "" }, "Select disease ▾"), ...State.diseases.map(d => el("option", { value: d.name, disabled: !d.trained, selected: d.name === PA.disease }, `${d.icon} ${d.display_name}${d.trained ? "" : " (not trained)"}`))))));
  if (!PA.disease) return; if (!PA.schema) await loadDisease();
  const stage = el("div", {}); wrap.append(stage);
  const up = el("div", { class: "panel stepcard" }, el("h2", {}, el("span", { class: "num" }, "2"), "Medical report"),
    el("p", { class: "muted" }, "JPG, PNG or PDF (scanned reports, lab printouts). The document goes to NVIDIA OCR; nothing reaches the disease model until you confirm the extracted values."),
    status.available ? null : el("div", { class: "warnbox" }, el("b", {}, "NVIDIA OCR is not running. "), status.reason, el("div", { class: "tiny", style: "margin-top:6px" }, `Hosted: ${status.requirements.hosted}. Local: ${status.requirements.local}.`, status.gpu?.present ? ` Detected GPU: ${status.gpu.name}.` : "")));
  const fileIn = el("input", { type: "file", accept: ".jpg,.jpeg,.png,.pdf,image/*,application/pdf", hidden: true }); const dropZ = el("label", { class: "drop" }, fileIn, el("span", {}, status.available ? "Drop a medical report here or click to choose" : "NVIDIA OCR unavailable — use manual transcription below"));
  const err = el("div", { class: "errbox", hidden: true });
  const runFile = async file => { if (!status.available) { err.textContent = status.reason; err.hidden = false; return; } err.hidden = true; dropZ.querySelector("span").textContent = `Sending ${file.name} to NVIDIA OCR…`;
    try { const fd = new FormData(); fd.append("file", file); const r = await api(`/api/ocr/document/${PA.disease}`, { method: "POST", body: fd }); PA.ocr = r.ocr; PA.inputSource = "nvidia_ocr"; renderVerification(stage, r.mapping, r.ocr, r.findings); }
    catch (e) { err.textContent = e.message; err.hidden = false; dropZ.querySelector("span").textContent = "Drop a medical report here or click to choose"; } };
  fileIn.onchange = e => e.target.files[0] && runFile(e.target.files[0]);
  dropZ.addEventListener("dragover", e => { e.preventDefault(); dropZ.classList.add("over"); }); dropZ.addEventListener("dragleave", () => dropZ.classList.remove("over")); dropZ.addEventListener("drop", e => { e.preventDefault(); dropZ.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) runFile(f); });
  const ta = el("textarea", { rows: 6, placeholder: "Paste or type the report text, one item per line, e.g.\nAge: 54\nGender: Male\nBP: 140/90\nCholesterol: 242 mg/dl\nMax heart rate: 130" });
  up.append(dropZ, err, el("h3", {}, "Manual transcription (not OCR)"), el("p", { class: "tiny" }, "If the document cannot be processed, type its contents. This path is labelled 'manual transcription' on the saved analysis and carries no OCR confidence values."), ta,
    el("div", { class: "actions" }, el("button", { class: "btn", onclick: async () => { err.hidden = true; try { const r = await postJson(`/api/ocr/transcribe/${PA.disease}`, { text: ta.value }); PA.ocr = { engine: r.engine, mean_confidence: null, filename: null }; PA.inputSource = "manual_transcription"; renderVerification(stage, r.mapping, null, r.findings); } catch (e) { err.textContent = e.message; err.hidden = false; } } }, "Extract from text →")));
  stage.append(up);
}
function renderVerification(stage, mapping, ocr, findings) {
  stage.innerHTML = ""; const s = PA.schema; const grid = el("div", { class: "ocr-grid" });
  const left = el("div", { class: "panel" }, el("h2", {}, ocr ? "NVIDIA OCR output" : "Transcribed text"));
  if (ocr) { left.append(el("p", { class: "muted" }, `${ocr.engine} · ${ocr.model} · ${ocr.n_pages} page(s) · ${ocr.lines.length} text lines${ocr.mean_confidence != null ? ` · mean confidence ${pct(ocr.mean_confidence)}` : ""}`)); ocr.pages.forEach(pg => left.append(el("img", { class: "thumb", src: pg.thumbnail, alt: `page ${pg.page}` }))); }
  const lines = el("div", { class: "ocrlines" }); (ocr ? ocr.lines : findings.map(f => ({ text: f.raw, confidence: null }))).forEach(l => lines.append(el("div", {}, el("span", {}, l.text), el("span", { class: "c" }, l.confidence == null ? "" : pct(l.confidence))))); left.append(lines);
  if (mapping.unmapped_findings?.length) left.append(el("p", { class: "tiny", style: "margin-top:8px" }, `Also found but not used by this model: ${mapping.unmapped_findings.map(f => f.key).join(", ")}`));
  const right = el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, "Verify extracted information"), el("span", { class: "muted" }, `${mapping.summary.ok} ok · ${mapping.summary.low_confidence} low confidence · ${mapping.summary.needs_review} review · ${mapping.summary.missing} missing`)),
    el("p", { class: "muted" }, "Check every value against the document, edit where needed, fill the missing ones, then confirm. Only confirmed data is sent to the trained model."),
    mapping.confidence_available ? null : el("div", { class: "infobox" }, "No confidence values: this input is a manual transcription, not OCR."));
  const tb = el("tbody", {}); const inputs = {};
  for (const f of mapping.features) {
    const field = s.features.find(x => x.name === f.feature); let inp;
    if (field.type === "select") inp = el("select", {}, el("option", { value: "" }, "Select…"), ...field.options.map(o => el("option", { value: o.value, selected: f.value != null && String(f.value) === o.value }, o.label)));
    else inp = el("input", { type: "number", step: field.step, value: f.value ?? "", placeholder: `${field.range.min}–${field.range.max}` });
    inputs[f.feature] = inp; const stBadge = el("span", { class: `st ${f.status}` }, f.status.replace("_", " "));
    inp.addEventListener("input", () => { stBadge.className = "st confirmed"; stBadge.textContent = "edited"; });
    tb.append(el("tr", {}, el("td", {}, field.label, field.unit ? el("span", { class: "unit muted" }, ` (${field.unit})`) : null), el("td", { class: "muted", style: "white-space:normal;max-width:220px" }, f.source_text || "—"), el("td", { class: "num" }, f.confidence == null ? "—" : pct(f.confidence)), el("td", {}, stBadge, f.note ? el("div", { class: "tiny", style: "white-space:normal;max-width:200px" }, f.status === "low_confidence" ? "⚠ " + f.note : f.note) : null), el("td", {}, inp)));
  }
  right.append(el("div", { class: "tablewrap" }, el("table", { class: "verify" }, el("thead", {}, el("tr", {}, el("th", {}, "model feature"), el("th", {}, "extracted from"), el("th", { class: "num" }, "OCR conf."), el("th", {}, "status"), el("th", {}, "value (editable)"))), tb)));
  const verr = el("div", { class: "errbox", hidden: true });
  right.append(verr, el("div", { class: "actions" }, el("button", { class: "btn", onclick: () => renderOCR(stage.parentElement.parentElement, {}) }, "← Another document"), el("button", { class: "btn primary lg", onclick: async () => {
    PA.values = {}; for (const [k, inp] of Object.entries(inputs)) PA.values[k] = inp.value; const errors = clientValidate(); if (errors.length) { verr.textContent = "Please complete before confirming:\n• " + errors.join("\n• "); verr.hidden = false; return; }
    verr.hidden = true; const c = stage.parentElement.parentElement; await analyze(c, verr); } }, icon("check", 18), "Confirm Data & Analyze")));
  grid.append(left, right); stage.append(el("div", { class: "panel stepcard" }, el("h2", {}, el("span", { class: "num" }, "3"), "Human verification"), el("p", { class: "muted" }, `Source: ${ocr ? "NVIDIA OCR" : "manual transcription"} → medical information extraction → mapping onto the ${s.display_name} feature schema.`)), grid);
}

/* ==================================================================== RESULT */
async function renderSavedAnalysis(c, analysisId) {
  try { const a = await api(`/api/analyses/${analysisId}`); const r = { ...a.result, patient_id: a.patient_id, patient_name: a.patient_name, analysed_at: a.result.analysed_at || a.created_at, input_source: a.input_source }; PA.patient = PA.patient?.patient_id === a.patient_id ? PA.patient : { patient_id: a.patient_id, name: a.patient_name }; if (PA.disease !== r.disease || !PA.schema) { PA.disease = r.disease; await loadDisease().catch(() => {}); } renderResult(c, r, { saved: true }); }
  catch (e) { c.append(el("div", { class: "panel" }, el("div", { class: "errbox" }, e.message))); }
}
function renderResult(c, r, opts = {}) {
  c.innerHTML = ""; const p = r.prediction, risk = RISK[p.risk_level]; const hasScore = p.probability != null; const saved = opts.saved || (PA.savedId && PA.result === r);
  const wrap = el("div", { class: "flow", style: "max-width:1100px" }); c.append(wrap);
  const saveBtn = r.patient_id ? el("button", { class: "btn sm", disabled: !!saved, onclick: async e => { try { const a = await postJson(`/api/patients/${r.patient_id}/analyses`, { result: r, input_source: r.input_source || "manual" }); PA.savedId = a.analysis_id; e.target.disabled = true; e.target.textContent = "Saved ✓"; toast("Analysis saved to the patient record"); } catch (ex) { toast(ex.message); } } }, saved ? "Saved ✓" : [icon("check", 15), "Save Analysis"]) : el("span", { class: "badge warn" }, "quick check — not saved (no patient record)");
  wrap.append(el("div", { class: "panel" }, el("div", { class: "result-head" }, el("div", {}, el("div", { class: "tiny", style: "letter-spacing:.1em;text-transform:uppercase" }, "Analysis complete"), el("h2", { class: "with-icon" }, dIcon({ category: categoryOf(r.disease) }, 22), r.display_name), el("p", { class: "muted" }, el("b", { style: "color:var(--ink)" }, r.patient_name || "Quick check"), r.patient_id ? el("span", { class: "pid" }, ` ${r.patient_id}`) : null, ` · ${fmtTime(r.analysed_at)} · input: `, el("span", { class: "srcchip" }, SOURCE_LABEL[r.input_source] || "Manual entry"))),
    el("div", { class: "actions", style: "margin:0" }, opts.saved ? null : el("button", { class: "btn sm", onclick: () => { PA.result = null; if (r.input_source === "manual" || !r.input_source) renderFlow(c); else go("analysis", { mode: "ocr", ...(r.patient_id ? { id: r.patient_id } : {}), disease: r.disease }); } }, "← Edit inputs"), saveBtn, r.patient_id ? el("a", { class: "btn sm", href: `#/records?id=${r.patient_id}` }, "Patient record") : null,
      el("button", { class: "btn primary sm", onclick: () => { PA.result = null; PA.values = {}; go("analysis", r.patient_id ? { id: r.patient_id, mode: "manual" } : {}); } }, r.patient_id ? [icon("plus", 15), "New Analysis"] : [icon("plus", 15), "New Patient"])))));
  const grid = el("div", { class: "result-grid" }); wrap.append(grid);
  grid.append(el("div", { class: `panel riskcard ${risk.cls}` }, el("div", { class: "tiny", style: "letter-spacing:.1em;text-transform:uppercase" }, "Model prediction"),
    hasScore ? el("div", { class: "prob" }, pct(p.probability), el("small", {}, r.models[r.selected_model]?.score_kind === "model probability" ? "model score" : "risk score")) : el("div", { class: "prob", style: "font-size:30px;margin-top:14px" }, p.label === 1 ? "Positive class" : "Negative class", el("small", {}, `decision margin ${fmt(r.models[r.selected_model].score, 2)} · no probability output`)),
    el("div", { class: "risk-label" }, el("span", { class: "ri" }, risk.icon), risk.text.toUpperCase()), hasScore ? el("div", { class: "meter" }, el("i", { style: `width:${Math.round(p.probability * 100)}%` })) : null,
    el("div", { class: "kpis" }, kpi(p.label === 1 ? "Positive" : "Negative", "prediction"), kpi(mlabel(r.selected_model), r.selected_model === r.best_model ? `model (best by ${r.best_criterion})` : "model"), kpi(SOURCE_LABEL[r.input_source] || "Manual entry", "input"), kpi(p.confidence, "model agreement")),
    el("p", { class: "muted", style: "font-size:12px;margin-top:12px" }, `The model predicts the ${p.label === 1 ? "positive / high-risk" : "negative / low-risk"} class (“${p.meaning}”). ${hasScore ? "The score is the model's output, not a calibrated clinical probability." : ""} ${p.confidence_basis}`),
    r.warnings?.length ? el("div", { class: "warnbox", style: "text-align:left" }, ...r.warnings.map(w => el("div", {}, "⚠ " + w))) : null));
  const models = Object.entries(r.models).sort((a, b) => (b[1].probability ?? -1) - (a[1].probability ?? -1));
  grid.append(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, "Model results"), el("span", { class: "muted" }, models.length > 1 ? "model score per trained model" : "single model")),
    el("div", { class: "bars" }, ...models.map(([m, v]) => { const row = barRow(m, v.probability, m === r.selected_model, `${mlabel(m)}: ${v.probability != null ? pct(v.probability) : "margin " + fmt(v.score, 2)} · predicts ${v.prediction} · validated acc ${fmt(v.validated_accuracy)}`); if (v.probability == null) row.querySelector(".bv").innerHTML = `<small>margin ${fmt(v.score, 2)}</small>`; return row; })),
    el("p", { class: "muted", style: "font-size:12px;margin-top:10px" }, `${models.filter(([, v]) => v.prediction === 1).length} of ${models.length} models predict the positive class. Highlighted = model used for the headline. Validated accuracy is from the hold-out split, not this patient.`),
    el("div", { class: "tablewrap", style: "margin-top:10px" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "model"), el("th", { class: "num" }, "class"), el("th", { class: "num" }, "score"), el("th", { class: "num" }, "val. acc"), el("th", { class: "num" }, "val. AUC"))), el("tbody", {}, ...models.map(([m, v]) => el("tr", { class: m === r.selected_model ? "sel" : "" }, el("td", {}, (v.is_quantum ? "⚛ " : "") + mlabel(m)), el("td", { class: "num" }, String(v.prediction)), el("td", { class: "num" }, v.probability != null ? pct(v.probability) : `margin ${fmt(v.score, 2)}`), el("td", { class: "num" }, fmt(v.validated_accuracy)), el("td", { class: "num" }, fmt(v.validated_roc_auc)))))))));
  wrap.append(explanationPanel(r));
  const s = PA.schema && PA.schema.disease === r.disease ? PA.schema : null;
  wrap.append(el("div", { class: "panel" }, el("h2", {}, "Inputs used"), el("div", { class: "chips", style: "margin-top:8px" }, ...Object.entries(r.inputs).map(([k, v]) => { const f = s?.features.find(x => x.name === k); const label = f ? f.label : k; const disp = f?.type === "select" ? (f.options.find(o => o.value === String(v))?.label ?? v) : v; return el("span", { class: "chip" }, `${label}: ${disp}${f?.unit ? " " + f.unit : ""}`); }))));
  wrap.append(el("p", { class: "disclaimer" }, "Patient information and model outputs are intended for authorized research and decision-support use. This system is not a substitute for professional medical diagnosis."));
}
const kpi = (v, l) => el("div", { class: "kpi" }, el("div", { class: "v" }, String(v)), el("div", { class: "l" }, l));
function explanationPanel(r) {
  const e = r.explanation; const panel = el("div", { class: "panel" });
  panel.append(el("div", { class: "panel-head" }, el("h2", {}, "Why this prediction?"), el("span", { class: "muted" }, e?.available ? `${mlabel(e.model)} · ${e.method}` : mlabel(r.selected_model))));
  if (!e || !e.available) { panel.append(el("div", { class: "placeholder" }, el("p", {}, el("b", {}, "No explanation available for this model yet.")), el("p", { class: "muted" }, e?.note || "No explanation method for this model type."), el("p", { class: "muted", style: "font-size:12px" }, "Logistic Regression and XGBoost provide exact per-patient contributions; choose one of them in the model step to see them."))); return panel; }
  const feats = e.features.slice(0, 8), maxAbs = Math.max(...feats.map(f => Math.abs(f.contribution)), 1e-9), specific = e.patient_specific;
  panel.append(el("p", { class: "muted", style: "font-size:12.5px" }, e.note));
  panel.append(specific ? el("div", { class: "legend" }, el("span", {}, el("i", { style: "background:var(--div-pos)" }), `pushes toward “${r.target_meaning["1"]}”`), el("span", {}, el("i", { style: "background:var(--div-neg)" }), `pushes toward “${r.target_meaning["0"]}”`), el("span", { class: "faint" }, `units: ${e.unit}`)) : el("div", { class: "legend" }, el("span", {}, el("i", { style: "background:var(--series-1)" }), "global importance (not patient-specific)")));
  panel.append(el("div", { class: "bars" }, ...feats.map(f => { const w = Math.abs(f.contribution) / maxAbs * (specific ? 50 : 100); return el("div", { class: "crow" }, el("div", { class: "bl", title: f.feature }, f.label), el("div", { class: "ct", title: `${f.label}: ${f.contribution > 0 ? "+" : ""}${fmt(f.contribution, 3)} ${e.unit}` }, el("i", { class: specific ? (f.contribution >= 0 ? "pos" : "neg") : "imp", style: `width:${w}%` })), el("div", { class: "bv" }, (specific && f.contribution > 0 ? "+" : "") + fmt(f.contribution, 2))); })));
  const top = feats[0];
  panel.append(el("p", { style: "margin-top:12px" }, specific ? `${top.label} was the strongest contributing feature for this patient, pushing the prediction toward “${r.target_meaning[top.contribution > 0 ? "1" : "0"]}”.` + (feats[1] ? ` ${feats[1].label} was the next most influential.` : "") : `Across the whole training set, ${top.label} is the feature the ${mlabel(e.model)} relies on most. This ranking is the same for every patient.`));
  if (specific && e.baseline != null) panel.append(el("p", { class: "tiny" }, `Baseline ${fmt(e.baseline, 3)} ${e.unit}; contributions plus baseline give the model's raw output before the score transform.`));
  return panel;
}
