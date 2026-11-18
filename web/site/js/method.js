// Method page behaviour: run the chosen ISP on the visitor's photo, show it coming out of the vortex,
// then one section per stage, the finished photo, and the comparison of all three ISPs.
import { createVortex } from "./vortex.js";
import { api, selection, settings } from "./api.js";
import { scribble } from "./scribble.js";

const $ = (s) => document.querySelector(s);
const METHOD = document.body.dataset.method;              // "classic" | "learned" | "hybrid"
const NAMES = { classic: "Classic", learned: "Learned", hybrid: "Hybrid" };
const DEFAULT_EXAMPLE = "sidd_normal_nexus6";             // used when someone opens the page directly
const isLocal = ["localhost", "127.0.0.1"].includes(location.hostname);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// small numbers shown under each stage (from the backend's "info"); add a line per stage id
const CHIPS = {
  photosites: (i) => [`${i.width.toLocaleString()} × ${i.height.toLocaleString()} photosites`, "one number per photosite"],
  bayer: (i) => [`${i.pattern} colour filter pattern`],
  white_balance: (i) => [`R × ${i.gains_rgb[0]}`, `G × ${i.gains_rgb[1]}`, `B × ${i.gains_rgb[2]}`, `from ${i.source}`,
    ...(i.exposure_gain > 1 ? [`exposure × ${i.exposure_gain}`] : [])],
  demosaic: (i) => [i.method],
  denoise: (i) => [`${i.method} filter`, `noise σ ${i.sigma}`],
  // (white balance also shows the auto-exposure gain when it is on)
  sharpen: (i) => [`amount ${i.amount}`],
  noise_map: (i) => [`noise model: ${i.source}`, `mean σ ${i.mean_sigma}`],
  pack: () => ["4 planes: R, G, G, B", "half width, half height"],
};

// ------------------------------------------------------------------ which photo?
function currentSource() {
  const q = new URLSearchParams(location.search);
  if (q.get("example")) return { example: q.get("example") };
  if (q.get("upload")) return { upload_id: q.get("upload") };
  const sel = selection.load();
  if (sel) return sel.kind === "example" ? { example: sel.example } : { upload_id: sel.upload_id };
  return { example: DEFAULT_EXAMPLE };
}
const source = currentSource();
const options = settings.fromQuery(new URLSearchParams(location.search));   // from the phone's Advanced screen
const query = (source.example ? `example=${encodeURIComponent(source.example)}` : `upload=${encodeURIComponent(source.upload_id)}`)
  + settings.toQuery(options);

// one line under the title listing the Advanced settings that are on (nothing when all are default)
function settingsLine(applied) {
  const on = [];
  if (METHOD === "hybrid" && applied.model === "teacher") on.push("teacher model");
  if (applied.auto_exposure && METHOD !== "learned") on.push("auto exposure");
  if (applied.sharpen === true && METHOD !== "learned") on.push("sharpening on");
  if (applied.sharpen === false && METHOD === "classic") on.push("sharpening off");
  if (applied.full_res) on.push("full resolution");
  return on.length ? "Settings: " + on.join(" · ") : "";
}

// ------------------------------------------------------------------ page chrome
document.title = `${NAMES[METHOD]} ISP: Take a Pic`;
document.querySelectorAll("[data-nav]").forEach((a) => {
  a.href = `${a.dataset.nav}.html?${query}`;
  if (a.dataset.nav === METHOD) a.setAttribute("aria-current", "page");
});
// top bar: "Take a Pic" on a yellow scribble, the current method on a red scribble with a white edge
{
  const mark = $(".topbar .mark");
  mark.innerHTML = `<span class="sc-text">Take a Pic</span>`;
  scribble(mark, { color: "#FFD60A", text: mark.firstElementChild, seed: 5, drawOnView: false });
  const cur = document.querySelector(".method-nav a[aria-current=page]");
  cur.innerHTML = `<span class="sc-text">${cur.textContent}</span>`;
  scribble(cur, { color: "#E8332A", text: cur.firstElementChild, seed: 11, drawOnView: false });
}
$("#other-methods").innerHTML = Object.keys(NAMES).filter((m) => m !== METHOD)
  .map((m) => `<a class="pill" href="${m}.html?${query}">See the ${NAMES[m]} ISP</a>`).join("");

const fx = createVortex($("#vortex"), { flare: "assets/fx/flare.jpg" });
let entry = null;
try { entry = sessionStorage.getItem("tap.entry"); sessionStorage.removeItem("tap.entry"); } catch { /* private mode */ }
if (entry === "lens" && !reduceMotion) fx.openIris();

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ------------------------------------------------------------------ stages
function stageSection(st, idx, total, content) {
  const c = content.stages[st.id] || { title: st.id, text: "" };
  // explanation: "points" (bullet list) and/or "text" (paragraphs, blank line between them)
  const points = (c.points || []).length ? `<ul class="points">${c.points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : "";
  const paras = (c.text || "").split(/\n\s*\n/).filter(Boolean).map((p) => `<p>${esc(p)}</p>`).join("") + points;
  const todo = !paras && isLocal
    ? `<p class="todo">Explanation for “${esc(st.id)}” goes in content/stages.json</p>` : "";
  const chips = (CHIPS[st.id]?.(st.info) || []).map((x) => `<li>${esc(x)}</li>`).join("");
  const sec = document.createElement("section");
  sec.className = "stage-sec";
  sec.id = `stage-${st.id}`;
  sec.setAttribute("aria-labelledby", `h-${st.id}`);
  sec.innerHTML = `
    <figure class="frame">
      <img src="${st.image.src}" width="${st.image.w}" height="${st.image.h}" alt="The photo after the ${esc(c.title)} stage" loading="lazy">
      ${st.zoom ? `<div class="loupe"><img src="${st.zoom}" width="256" height="256" loading="lazy" alt="Close-up of a 40 by 40 photosite patch at this stage"></div>` : ""}
    </figure>
    <div class="stage-text">
      <p class="stage-num">${String(idx + 1).padStart(2, "0")}<span class="sr-only"> of ${total}</span></p>
      <h2 id="h-${st.id}"><span class="h-scribble">${esc(c.title)}</span></h2>
      ${paras}${todo}
      ${chips ? `<ul class="chips">${chips}</ul>` : ""}
    </div>`;
  return sec;
}

function buildRail(stages, content) {
  const rail = $("#rail");
  rail.innerHTML = stages.map((st) =>
    `<li><a href="#stage-${st.id}">${esc((content.stages[st.id] || {}).title || st.id)}</a></li>`).join("")
    + `<li><a href="#result">Result</a></li><li><a href="#compare">Compare</a></li>`;
  const links = [...rail.querySelectorAll("a")];
  const secs = links.map((a) => document.querySelector(a.getAttribute("href")));
  // the active stage is the section crossing the middle of the screen
  const update = () => {
    const mid = window.innerHeight / 2;
    const i = secs.findIndex((s) => { const r = s.getBoundingClientRect(); return r.top <= mid && r.bottom > mid; });
    links.forEach((a, j) => (j === i ? a.setAttribute("aria-current", "step") : a.removeAttribute("aria-current")));
  };
  addEventListener("scroll", () => requestAnimationFrame(update), { passive: true });
  addEventListener("resize", update);
  update();
}

// the vortex calms down and the big photo fades once you scroll past the intro
function linkScrollToVortex() {
  let ticking = false;
  const update = () => {
    ticking = false;
    const k = Math.min(1, window.scrollY / (window.innerHeight * 0.7));
    fx.set({ glow: 1 - 0.78 * k, photoMix: 1 - k, swirl: 0.55 - 0.3 * k });
    const end = document.querySelector("#result:not([hidden])");
    const pastStages = end && end.getBoundingClientRect().top < window.innerHeight * 0.5;
    $("#rail").classList.toggle("show", window.scrollY > window.innerHeight * 0.6 && !pastStages);
  };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
}

// ------------------------------------------------------------------ run
async function run() {
  const content = await fetch("content/stages.json").then((r) => r.json());
  const m = content.methods[METHOD];
  $("#method-title").textContent = m.title;
  $("#method-sub").textContent = m.subtitle;

  const loading = $("#loading");
  if (options.full_res) loading.textContent = "Developing your photo at full resolution (slower)";
  const t0 = performance.now();
  let result;
  try {
    result = await api.process(METHOD, source, options);
  } catch (e) {
    loading.classList.add("error");
    loading.innerHTML = `${esc(e.message)} <a href="./" style="color:inherit">Back to the phone</a>`;
    fx.set({ swirl: 0.3, glow: 0.6 });
    return;
  }
  // let the vortex spin for at least a moment, even when the answer is instant (cached)
  const minSpin = reduceMotion ? 0 : 1400 - (performance.now() - t0);
  if (minSpin > 0) await new Promise((r) => setTimeout(r, minSpin));

  loading.remove();
  await fx.showPhoto(result.final.src);
  $("#scroll-cue").hidden = false;
  $("#method-settings").textContent = settingsLine({ ...result.options, sharpen: options.sharpen });

  const holder = $("#stages");
  const order = m.order.filter((id) => result.stages.some((s) => s.id === id));
  const byId = Object.fromEntries(result.stages.map((s) => [s.id, s]));
  order.forEach((id, i) => holder.appendChild(stageSection(byId[id], i, order.length, content)));
  // marker scribble behind the words of each stage heading (line by line), drawn as it comes into view
  holder.querySelectorAll(".stage-sec h2").forEach((h2, i) =>
    scribble(h2, { color: "#FFD60A", text: h2.querySelector(".h-scribble"), seed: 101 + i * 37 }));

  $("#result").hidden = false;
  Object.assign($("#result-img"), { src: result.final.src, width: result.final.w, height: result.final.h });
  $("#result-img").alt = `The finished photo from the ${NAMES[METHOD]} ISP`;
  $("#result-meta").textContent = `${result.width.toLocaleString()} × ${result.height.toLocaleString()} px`
    + `${result.cropped ? " (centre crop)" : ""} · processed in ${(result.time_ms / 1000).toFixed(1)} s on a CPU`;

  buildRail(order.map((id) => byId[id]), content);
  linkScrollToVortex();
}

// ------------------------------------------------------------------ comparison (bundled examples only)
const LABEL = { classic: "Classic", learned: "Learned", hybrid: "Hybrid" };
async function setupCompare() {
  let examples = [];
  try { examples = await api.examples(); } catch { return; }
  const picker = $("#compare-picker");
  const start = examples.find((e) => e.id === source.example) ? source.example
    : (examples.find((e) => e.has_ground_truth) || examples[0]).id;
  picker.innerHTML = examples.map((e) =>
    `<button type="button" class="pill" data-ex="${esc(e.id)}" aria-pressed="${e.id === start}">${esc(e.title)}</button>`).join("");
  picker.addEventListener("click", (ev) => {
    const b = ev.target.closest("button[data-ex]");
    if (!b) return;
    picker.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    showCompare(b.dataset.ex);
  });
  // load the comparison only when it is close to the screen (it runs all three ISPs)
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); showCompare(start); }
  }, { rootMargin: "600px 0px" });
  io.observe($("#compare"));
}

let compareToken = 0;
async function showCompare(exId) {
  const grid = $("#compare-grid"), token = ++compareToken;
  grid.className = "compare-grid";
  grid.innerHTML = `<p class="compare-wait">Running all three ISPs on this photo…</p>`;
  let c;
  try { c = await api.compare(exId); } catch (e) { grid.innerHTML = `<p class="compare-wait">${esc(e.message)}</p>`; return; }
  if (token !== compareToken) return;
  const best = {};
  if (c.ground_truth) {
    const vals = (k) => c.results.map((r) => r.metrics[k]);
    best.psnr = Math.max(...vals("psnr")); best.ssim = Math.max(...vals("ssim")); best.delta_e = Math.min(...vals("delta_e"));
  }
  const cell = (r) => `
    <div class="cmp${r.method === METHOD ? " current" : ""}">
      <figure class="frame"><img src="${r.image.src}" width="${r.image.w}" height="${r.image.h}" loading="lazy" alt="${LABEL[r.method]} ISP result"></figure>
      <h3>${LABEL[r.method]}${r.method === METHOD ? " <span class='sr-only'>(this page)</span>" : ""}</h3>
      ${r.metrics ? `<ul class="metrics">
        <li class="${r.metrics.psnr === best.psnr ? "best" : ""}"><b>PSNR</b> ${r.metrics.psnr.toFixed(2)} dB</li>
        <li class="${r.metrics.ssim === best.ssim ? "best" : ""}"><b>SSIM</b> ${r.metrics.ssim.toFixed(3)}</li>
        <li class="${r.metrics.delta_e === best.delta_e ? "best" : ""}"><b>ΔE</b> ${r.metrics.delta_e.toFixed(2)}</li>
      </ul>` : `<ul class="metrics"><li>No ground truth for this phone</li></ul>`}
    </div>`;
  grid.innerHTML = c.results.map(cell).join("") + (c.ground_truth ? `
    <div class="cmp gt">
      <figure class="frame"><img src="${c.ground_truth.src}" width="${c.ground_truth.w}" height="${c.ground_truth.h}" loading="lazy" alt="Ground truth"></figure>
      <h3>Ground truth</h3><ul class="metrics"><li>150-frame average</li></ul>
    </div>` : "");
  if (!c.ground_truth) grid.classList.add("three");
}

run();
setupCompare();
