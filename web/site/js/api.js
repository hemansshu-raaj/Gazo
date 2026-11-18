// Tiny wrappers around the backend (web/api/main.py). Every call throws an Error with a readable message.

async function asJson(res) {
  if (res.ok) return res.json();
  let msg = `Server error (${res.status}).`;
  try { msg = (await res.json()).detail || msg; } catch { /* keep the generic message */ }
  throw new Error(msg);
}

export const api = {
  examples: () => fetch("/api/examples").then(asJson),
  thumb: (id) => `/api/examples/${encodeURIComponent(id)}/thumb.jpg`,
  upload(file) {
    const fd = new FormData();
    fd.append("file", file, file.name);
    return fetch("/api/upload", { method: "POST", body: fd }).then(asJson);
  },
  process: (method, source, options) =>
    fetch("/api/process", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method, ...source, options }),
    }).then(asJson),
  compare: (exampleId) => fetch(`/api/compare/${encodeURIComponent(exampleId)}`).then(asJson),
};

// What the visitor picked on the front page, carried to the method pages.
// {kind: "upload", upload_id, name} or {kind: "example", example, name}
export const selection = {
  save(sel) { try { sessionStorage.setItem("tap.selection", JSON.stringify(sel)); } catch { /* private mode */ } },
  load() { try { return JSON.parse(sessionStorage.getItem("tap.selection")); } catch { return null; } },
  toQuery(sel) {
    return sel.kind === "example" ? `example=${encodeURIComponent(sel.example)}` : `upload=${encodeURIComponent(sel.upload_id)}`;
  },
};

// Advanced settings from the phone's Advanced screen (kept for this browser tab).
// {model: "student"|"teacher", auto_exposure: bool, sharpen: "auto"|true|false, full_res: bool}
const DEFAULTS = { model: "student", auto_exposure: false, sharpen: "auto", full_res: false };
export const settings = {
  load() {
    try { return { ...DEFAULTS, ...JSON.parse(sessionStorage.getItem("tap.settings") || "{}") }; } catch { return { ...DEFAULTS }; }
  },
  save(s) { try { sessionStorage.setItem("tap.settings", JSON.stringify(s)); } catch { /* private mode */ } },
  // only non-default values go in the URL, e.g. "&model=teacher&ae=1"
  toQuery(s) {
    const q = [];
    if (s.model !== DEFAULTS.model) q.push(`model=${s.model}`);
    if (s.auto_exposure) q.push("ae=1");
    if (s.sharpen !== "auto") q.push(`sharpen=${s.sharpen ? 1 : 0}`);
    if (s.full_res) q.push("full=1");
    return q.length ? "&" + q.join("&") : "";
  },
  fromQuery(params) {
    const s = { ...DEFAULTS };
    if (params.get("model") === "teacher") s.model = "teacher";
    if (params.get("ae") === "1") s.auto_exposure = true;
    if (params.has("sharpen")) s.sharpen = params.get("sharpen") === "1";
    if (params.get("full") === "1") s.full_res = true;
    return s;
  },
};
