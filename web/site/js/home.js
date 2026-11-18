// Front page behaviour: the title animation, the phone, drag-and-drop, the screen menu, example photos.
import { createPhone } from "./phone3d.js";
import { drawScreen, drawPopRow, TEXT, SETTINGS, menuItems, valueLabel, cycleSetting } from "./screen-ui.js";
import { api, selection, settings } from "./api.js";
import { scribble } from "./scribble.js";

const $ = (s) => document.querySelector(s);
const stage = $("#stage"), statusEl = $("#status"), fileInput = $("#file");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const touch = matchMedia("(pointer: coarse)").matches;

// ------------------------------------------------------------------ "Take a Pic" lettering
async function loadLettering() {
  const svg = await fetch("assets/lettering/take-a-pic.svg").then((r) => r.text());
  $("#title").insertAdjacentHTML("beforeend", svg);
  // each path needs its own length for the "written by hand" stroke animation
  $("#title").querySelectorAll("path").forEach((p) => p.style.setProperty("--len", Math.ceil(p.getTotalLength())));
}

// ------------------------------------------------------------------ screen state machine
// idle -> (hover) -> busy -> menu <-> advanced ; error goes back to idle
let state = "idle", data = { settings: settings.load() }, targets = [], phone = null, errorTimer = 0;
const LISTS = new Set(["menu", "advanced"]);

scribble($("#examples-title"), { color: "#fff", text: $(".ex-scribble"), seed: 23 });
let hintText = "";

function setState(next, extra = {}) {
  state = next; data = { ...data, ...extra };
  document.body.classList.toggle("zoomed", LISTS.has(state));   // hint moves onto the zoomed phone
  const hint = state === "menu" ? "Pick an ISP on the screen, or press 1, 2 or 3. Press 4 for Advanced."
    : state === "advanced" ? "Change a setting on the screen, or press 1 to 4. Esc goes back."
    : touch ? "Tap the screen to choose a RAW photo (.dng), or tap an example below."
            : "Drag a RAW photo (.dng) onto the screen. Drag the phone to spin it.";
  if (hint !== hintText) { hintText = hint; $("#hint").textContent = hint; }
  statusEl.textContent = {
    idle: "", busy: `Reading ${data.name || "the file"}.`,
    error: `${TEXT.errorTitle}. ${data.message || ""}`,
    menu: `${data.name} is ready. Press 1 for Classic, 2 for Learned, 3 for Hybrid or 4 for Advanced settings.`,
    advanced: "Advanced settings. " + SETTINGS.map((s, i) => `${i + 1}: ${s.label}, ${valueLabel(data.settings, s)}`).join(". "),
  }[state] ?? "";
  render();
}

// the row that is selected in the current list, as {num, label, value} for the pop-out key
function selectedRow() {
  if (state === "menu") {
    const i = data.selected || 0, it = menuItems()[i];
    return { i, row: { num: String(i + 1), label: it.label, value: it.id === "advanced" ? "\u2026" : "\u203a" } };
  }
  const i = data.selectedSetting || 0, set = SETTINGS[i];
  return { i, row: { num: String(i + 1), label: set.label, value: valueLabel(data.settings, set) } };
}

function render() {
  if (!phone) return;
  data.popped = LISTS.has(state);
  targets = drawScreen(phone.screenCanvas, state, data, performance.now());
  phone.refreshScreen();
  // the selected row rises out of the list as a 3D key
  if (LISTS.has(state)) {
    const { i, row } = selectedRow();
    phone.setPop(targets.filter((t) => t.id !== "back")[i], (g, w, h) => drawPopRow(g, w, h, row));
  } else phone.setPop(null);
}

// hover and busy are animated, so keep redrawing while they show
(function loop() {
  if (state === "hover" || state === "busy") render();
  requestAnimationFrame(loop);
})();

// ------------------------------------------------------------------ picking a file or example
function popSfx() {
  const s = $("#sfx");
  s.classList.remove("pop"); void s.offsetWidth; s.classList.add("pop");
}

function ready(sel) {
  selection.save(sel);
  data.selected = 0;
  phone.faceFront();
  phone.slide(true);
  phone.zoom(true);
  if (window.scrollY > 40) window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
  if (!reduceMotion) popSfx();
  setState("menu", { name: sel.name });
  phone.canvas.focus({ preventScroll: true, focusVisible: false });   // keyboard users land on the phone's menu
}

function fail(message) {
  setState("error", { message });
  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => state === "error" && setState("idle"), 7000);
}

async function useFile(file) {
  if (!file) return;
  phone.zoom(false);
  phone.slide(false);
  setState("busy", { name: file.name });
  try {
    const r = await api.upload(file);
    ready({ kind: "upload", upload_id: r.upload_id, name: file.name });
  } catch (e) {
    fail(e.message);
  }
}

function useExample(ex) {
  ready({ kind: "example", example: ex.id, name: ex.title });
}

// ------------------------------------------------------------------ choosing a method
let leaving = false;
async function go(method) {
  const sel = selection.load();
  if (!sel || leaving) return sel ? undefined : setState("idle");
  leaving = true;
  const url = `${method}.html?${selection.toQuery(sel)}${settings.toQuery(data.settings)}`;
  statusEl.textContent = `Opening the ${method} ISP.`;
  if (reduceMotion) { window.location.href = url; return; }
  // turn the phone round, fly into its camera lens, then open an iris onto the next page
  stage.classList.add("leaving");
  // everything else fades away while the phone takes over the window
  document.body.classList.add("flying");
  const lens = await phone.flyIntoLens({ turnMs: 750, pushMs: 1000 });
  const iris = $("#iris");
  iris.style.setProperty("--x", lens.x + "px");
  iris.style.setProperty("--y", lens.y + "px");
  iris.classList.add("open");
  try { sessionStorage.setItem("tap.entry", "lens"); } catch { /* private mode */ }
  setTimeout(() => { window.location.href = url; }, 520);
}
// coming back with the browser's Back button: reset everything
window.addEventListener("pageshow", (e) => { if (e.persisted) window.location.reload(); });

function back() {
  if (state === "advanced") { data.selected = 3; return setState("menu"); }   // back to the ISP list
  phone.zoom(false);
  phone.slide(false);
  setState("idle");
}

// what a row does when it is picked (tap, Enter or its number key)
function activate(i) {
  if (state === "menu") {
    const it = menuItems()[i];
    if (!it) return;
    data.selected = i;
    if (it.id === "advanced") { data.selectedSetting = 0; return setState("advanced"); }
    render();
    return go(it.id);
  }
  if (state === "advanced" && SETTINGS[i]) {
    data.selectedSetting = i;
    data.settings = cycleSetting(data.settings, i);
    settings.save(data.settings);
    setState("advanced");                      // redraw + announce the new value
  }
}

const listLength = () => (state === "menu" ? menuItems().length : SETTINGS.length);
const selKey = () => (state === "menu" ? "selected" : "selectedSetting");

function rowAt(uv) {
  if (!uv) return -1;
  const y = uv.v * phone.screenCanvas.height;
  return targets.filter((t) => t.id !== "back").findIndex((t) => y >= t.top && y <= t.bottom);
}

function onScreenHover(uv) {
  if (!LISTS.has(state)) { phone.canvas.style.cursor = uv && state === "idle" ? "pointer" : "grab"; return; }
  const i = rowAt(uv);
  phone.canvas.style.cursor = i >= 0 ? "pointer" : "grab";
  if (i >= 0 && i !== data[selKey()]) { data[selKey()] = i; render(); }
}

function onScreenTap(uv) {
  if (state === "idle") return fileInput.click();
  if (state === "error") return setState("idle");
  if (!LISTS.has(state)) return;
  const y = uv.v * phone.screenCanvas.height;
  if (targets.some((t) => t.id === "back" && y >= t.top)) return back();
  const i = rowAt(uv);
  if (i >= 0) activate(i);
}

// keyboard: number keys pick, arrows move, Enter picks, Escape goes back
document.addEventListener("keydown", (e) => {
  if (!LISTS.has(state) || e.target.closest("input, textarea")) return;
  const n = listLength(), k = selKey();
  const onControl = e.target.closest("button, a");          // let focused buttons/links handle Enter
  if (/^[1-9]$/.test(e.key) && +e.key <= n) activate(+e.key - 1);
  else if (e.key === "ArrowDown") { e.preventDefault(); data[k] = ((data[k] || 0) + 1) % n; render(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); data[k] = ((data[k] || 0) + n - 1) % n; render(); }
  else if (e.key === "Enter" && !onControl) activate(data[k] || 0);
  else if (e.key === "Escape" || e.key === "Backspace") { e.preventDefault(); back(); }
});
fileInput.addEventListener("change", () => useFile(fileInput.files[0]));

// ------------------------------------------------------------------ drag and drop onto the screen
const EX_TYPE = "text/x-isp-example";
let examples = [];
const overScreen = (e) => phone && phone.screenUV(e.clientX, e.clientY) !== null;
const isOurDrag = (e) => [...e.dataTransfer.types].some((t) => t === "Files" || t === EX_TYPE);

window.addEventListener("dragover", (e) => {
  if (!isOurDrag(e)) return;
  e.preventDefault();                      // otherwise the browser opens the file
  const over = overScreen(e);
  e.dataTransfer.dropEffect = over ? "copy" : "none";
  stage.classList.add("dragging");
  if (state === "busy") return;
  if (over && state !== "hover") { phone.faceFront(); setState("hover"); }
  else if (!over && state === "hover") setState("idle");
});
window.addEventListener("dragleave", (e) => {
  if (e.relatedTarget === null && state === "hover") { setState("idle"); stage.classList.remove("dragging"); }
});
window.addEventListener("drop", (e) => {
  if (!isOurDrag(e)) return;
  e.preventDefault();
  stage.classList.remove("dragging");
  if (!overScreen(e)) { if (state === "hover") setState("idle"); return; }
  const exId = e.dataTransfer.getData(EX_TYPE);
  if (exId) return useExample(examples.find((x) => x.id === exId));
  useFile(e.dataTransfer.files[0]);
});

// ------------------------------------------------------------------ example cards
async function loadExamples() {
  try {
    examples = await api.examples();
  } catch {
    return;
  }
  const list = $("#example-list");
  for (const ex of examples) {
    const li = document.createElement("li");
    li.innerHTML = `<button type="button" class="example" draggable="true">
        <img src="${api.thumb(ex.id)}" alt="" width="64" height="64" loading="lazy">
        <span><strong></strong></span></button>`;
    li.querySelector("strong").textContent = ex.title;
    const btn = li.querySelector("button");
    btn.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData(EX_TYPE, ex.id);
      e.dataTransfer.effectAllowed = "copy";
      e.dataTransfer.setDragImage(btn.querySelector("img"), 32, 32);
    });
    btn.addEventListener("click", () => useExample(ex));   // tap works too (phones can't drag)
    list.appendChild(li);
  }
}

// ------------------------------------------------------------------ start
loadLettering();
loadExamples();
phone = await createPhone(stage, { model: "assets/models/phone.glb", onScreenTap, onScreenHover, frameEl: $("#stage-slot") });
document.fonts.ready.then(render);
setState("idle");
window.__home = { phone, setState, get state() { return state; } };   // handy in the console and in tests
