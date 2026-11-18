// A loose, hand-drawn marker scribble behind text, one scribble per line of text, with strokes that
// overshoot well past the words (it is decoration first; it does not need to cover the text exactly).
//
//   import { scribble } from "./scribble.js";
//   const s = scribble(box, { color: "#fff" });          // box: the element the SVG sits in
//   const s = scribble(h2, { text: h2.firstElementChild }) // follow the line boxes of an inline <span>
//   s.redraw();                                            // scribble again (e.g. after the text changed)
//
// Looks are set in CSS (.scribbled / .scribble-bg in base.css); shape knobs in SHAPE below.

const SVGNS = "http://www.w3.org/2000/svg";
const SHAPE = {
  stroke: 0.34,     // marker width, as a fraction of the line height
  slant: 0.55,      // how diagonal the fill strokes are (0 = vertical)
  reachY: 0.16,     // how far the fill spills above / below a line (x line height)
  fly: 4,           // long loose flourishes per line, flying far out to the sides
  flyReach: 3.2,    // how far those flourishes go past the words (x line height)
};

function rng(seed) {                       // small seeded random, so a heading keeps its own scribble
  let s = seed >>> 0 || 1;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// one line of text -> a diagonal marker fill under the words + long flourishes flying out sideways
function lineScribble(r, rand) {
  const h = r.height, sw = Math.max(7, h * SHAPE.stroke), slant = h * SHAPE.slant;
  const top = r.top - h * SHAPE.reachY, bottom = r.bottom + h * SHAPE.reachY * 0.6;
  let x = r.left - h * (0.25 + rand() * 0.3), up = false;
  const end = r.right + h * (0.2 + rand() * 0.4);
  let d = `M${x.toFixed(1)},${bottom.toFixed(1)}`;
  while (x < end) {                        // diagonal back-and-forth, like colouring in fast
    const nx = x + (up ? -slant * 0.35 : slant) + sw * (0.55 + rand() * 0.25);
    const y = up ? bottom - rand() * h * 0.12 : top + rand() * h * 0.12;
    d += ` L${nx.toFixed(1)},${y.toFixed(1)}`;
    x = nx; up = !up;
  }
  let fly = "";
  for (let i = 0; i < SHAPE.fly; i++) {    // flourishes: start inside the line, sweep far out
    const left = i % 2 === 0;
    const y0 = r.top + h * (0.2 + rand() * 0.6);
    const x0 = left ? r.left + rand() * h : r.right - rand() * h;
    const x1 = left ? r.left - h * SHAPE.flyReach * (0.5 + rand() * 0.5) : r.right + h * SHAPE.flyReach * (0.5 + rand() * 0.6);
    const y1 = y0 + (rand() - 0.5) * h * 0.9;
    const cy = y0 + (rand() - 0.5) * h * 1.2;
    fly += ` M${x0.toFixed(1)},${y0.toFixed(1)} Q${((x0 + x1) / 2).toFixed(1)},${cy.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
  }
  return { d, fly, sw };
}

export function scribble(box, { color = "#fff", text = null, seed = Math.floor(Math.random() * 1e9), drawOnView = true } = {}) {
  box.classList.add("scribbled");
  box.style.setProperty("--scribble", color);
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "scribble-bg");
  svg.setAttribute("aria-hidden", "true");
  box.prepend(svg);

  let lastKey = "", runs = 0;
  function lines() {                        // line boxes, relative to the box
    const b = box.getBoundingClientRect();
    const src = text ? [...text.getClientRects()] : [box.getBoundingClientRect()];
    return src.filter((r) => r.width > 2).map((r) => ({
      left: r.left - b.left, right: r.right - b.left, top: r.top - b.top, bottom: r.bottom - b.top, height: r.height,
    }));
  }
  function build(animate) {
    const ls = lines();
    if (!ls.length) return;
    lastKey = ls.map((r) => `${r.left | 0},${r.top | 0},${r.right | 0}`).join(";");
    const rand = rng(seed + runs * 7919);
    svg.replaceChildren();
    ls.forEach((r, li) => {
      const { d, fly, sw } = lineScribble(r, rand);
      [[d, sw, 0], [fly, sw * 0.32, 0.45]].forEach(([dd, w, delay]) => {
        const p = document.createElementNS(SVGNS, "path");
        p.setAttribute("d", dd);
        p.setAttribute("stroke-width", w.toFixed(1));
        svg.appendChild(p);
        p.style.setProperty("--len", Math.ceil(p.getTotalLength()));
        p.style.animationDelay = `${(li * 0.18 + delay).toFixed(2)}s`;
      });
    });
    if (animate) { box.classList.remove("drawn"); void svg.getBoundingClientRect(); box.classList.add("drawn"); }
  }

  new ResizeObserver(() => {
    const key = lines().map((r) => `${r.left | 0},${r.top | 0},${r.right | 0}`).join(";");
    if (key !== lastKey) build(box.classList.contains("drawn"));
  }).observe(box);
  if (drawOnView) {
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { io.disconnect(); build(true); }
    }, { threshold: 0.4 });
    io.observe(box);
  } else build(true);

  return { redraw() { runs++; build(true); } };
}
