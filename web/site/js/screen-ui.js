// Everything drawn on the phone's screen (a 480x640 canvas), one function per state.
//
//   drawScreen(canvas, state, data, timeMs) -> returns tap targets [{id, top, bottom}] in canvas pixels
//
// States: idle | hover | busy | error | menu | advanced.  Text lives in TEXT and SETTINGS below.

export const TEXT = {
  idleTitle: "Drop a RAW photo",
  idleSub: ["onto this screen", "or tap to choose a .dng"],
  hover: "Upload here",
  busy: "Reading RAW",
  errorTitle: "Needs a RAW file",
  errorHint: "Try an example below",
  menuTitle: "Choose an ISP",
  methods: [["classic", "Classic"], ["learned", "Learned"], ["hybrid", "Hybrid"]],
  advanced: "Advanced",
  softLeft: "Select",
  softRight: "Back",
  advancedTitle: "Advanced",
  advancedSub: "Used for the next run",
  softChange: "Change",
};

// The Advanced screen: each setting cycles through its values. The first value is the default.
// key = what is sent to the backend (web/api/service.py clean_options).
export const SETTINGS = [
  { key: "model", label: "Hybrid model", values: [["student", "0.7 MB"], ["teacher", "Teacher"]] },
  { key: "auto_exposure", label: "Auto exposure", values: [[false, "Off"], [true, "On"]] },
  { key: "sharpen", label: "Sharpening", values: [["auto", "Auto"], [true, "On"], [false, "Off"]] },
  { key: "full_res", label: "Full resolution", values: [[false, "Off"], [true, "On (slow)"]] },
];
export const defaultSettings = () => Object.fromEntries(SETTINGS.map((s) => [s.key, s.values[0][0]]));
export const valueLabel = (settings, s) => (s.values.find(([v]) => v === settings[s.key]) || s.values[0])[1];
export function cycleSetting(settings, i) {
  const s = SETTINGS[i], k = s.values.findIndex(([v]) => v === settings[s.key]);
  return { ...settings, [s.key]: s.values[(k + 1) % s.values.length][0] };
}

// menu rows: the three ISPs, then Advanced
export const menuItems = () => [...TEXT.methods.map(([id, label]) => ({ id, label })), { id: "advanced", label: TEXT.advanced }];

const C = {
  bg: "#0c1a22", bar: "#132a35", text: "#f4f1e8", dim: "#8fa6b0",
  yellow: "#FFD60A", ink: "#111014", red: "#E8332A",
};
const FONT = (w, px) => `${w} ${px}px Archivo, system-ui, sans-serif`;

function statusBar(g, W) {
  g.fillStyle = C.bar; g.fillRect(0, 0, W, 54);
  g.fillStyle = C.text;
  for (let i = 0; i < 4; i++) g.fillRect(22 + i * 12, 36 - i * 7, 8, 8 + i * 7);   // signal bars
  g.strokeStyle = C.text; g.lineWidth = 3; g.strokeRect(W - 70, 18, 42, 20);           // battery
  g.fillRect(W - 28, 24, 5, 8); g.fillRect(W - 66, 22, 30, 12);
  g.font = FONT(700, 24); g.textAlign = "center"; g.fillText("12:08", W / 2, 37);
}

function softKeys(g, W, H, left, right) {
  g.fillStyle = C.bar; g.fillRect(0, H - 62, W, 62);
  g.fillStyle = C.text; g.font = FONT(700, 26);
  g.textAlign = "left"; g.fillText(left, 24, H - 22);
  g.textAlign = "right"; g.fillText(right, W - 24, H - 22);
}

function wrap(g, text, maxW) {
  const words = text.split(" "), lines = [];
  let line = "";
  for (const w of words) {
    const t = line ? line + " " + w : w;
    if (g.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t;
  }
  if (line) lines.push(line);
  return lines;
}

function viewfinder(g, cx, cy, s, color) {   // four corner brackets + a lens dot
  g.strokeStyle = color; g.lineWidth = 7; g.lineCap = "round";
  for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    g.beginPath();
    g.moveTo(cx + dx * s, cy + dy * s * 0.55); g.lineTo(cx + dx * s, cy + dy * s); g.lineTo(cx + dx * s * 0.55, cy + dy * s);
    g.stroke();
  }
  g.beginPath(); g.arc(cx, cy, s * 0.32, 0, Math.PI * 2); g.stroke();
}

export function drawScreen(canvas, state, data = {}, t = 0) {
  const g = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.textBaseline = "alphabetic";
  const targets = [];

  if (state === "hover") {
    g.fillStyle = C.yellow; g.fillRect(0, 0, W, H);
    g.strokeStyle = C.ink; g.lineWidth = 8; g.setLineDash([26, 16]); g.lineDashOffset = -t / 25;
    g.strokeRect(26, 26, W - 52, H - 52); g.setLineDash([]);
    const bob = Math.sin(t / 180) * 10;
    g.fillStyle = C.ink;                                   // big arrow
    g.beginPath();
    g.moveTo(W / 2 - 34, 170 + bob); g.lineTo(W / 2 + 34, 170 + bob); g.lineTo(W / 2 + 34, 260 + bob);
    g.lineTo(W / 2 + 80, 260 + bob); g.lineTo(W / 2, 340 + bob); g.lineTo(W / 2 - 80, 260 + bob);
    g.lineTo(W / 2 - 34, 260 + bob); g.closePath(); g.fill();
    g.font = FONT(900, 62); g.textAlign = "center"; g.fillText(TEXT.hover, W / 2, 450);
    return targets;
  }

  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  statusBar(g, W);

  if (state === "idle") {
    viewfinder(g, W / 2, 230, 92, C.yellow);
    g.fillStyle = C.text; g.textAlign = "center";
    g.font = FONT(800, 40); g.fillText(TEXT.idleTitle, W / 2, 400);
    g.font = FONT(500, 27); g.fillStyle = C.dim;
    TEXT.idleSub.forEach((l, i) => g.fillText(l, W / 2, 448 + i * 38));
    softKeys(g, W, H, "", "Pick");
    targets.push({ id: "pick", top: 0, bottom: H });
  } else if (state === "busy") {
    g.fillStyle = C.text; g.textAlign = "center"; g.font = FONT(800, 40);
    g.fillText(TEXT.busy, W / 2, 290);
    g.font = FONT(500, 26); g.fillStyle = C.dim; g.fillText(data.name || "", W / 2, 334);
    const x = 70, w = W - 140, y = 380;                      // indeterminate progress bar
    g.strokeStyle = C.text; g.lineWidth = 4; g.strokeRect(x, y, w, 34);
    g.fillStyle = C.yellow;
    for (let i = 0; i < 6; i++) {
      const bx = x + 6 + ((t / 6 + i * 60) % (w - 40));
      g.fillRect(bx, y + 7, 26, 20);
    }
  } else if (state === "error") {
    g.fillStyle = C.red; g.fillRect(0, 54, W, 10);
    g.fillStyle = C.text; g.textAlign = "center"; g.font = FONT(800, 40);
    g.fillText(TEXT.errorTitle, W / 2, 170);
    g.font = FONT(500, 26); g.fillStyle = C.dim;
    wrap(g, data.message || "", W - 80).slice(0, 7).forEach((l, i) => g.fillText(l, W / 2, 230 + i * 36));
    g.fillStyle = C.yellow; g.font = FONT(800, 28); g.fillText(TEXT.errorHint + " ↓", W / 2, H - 100);
    softKeys(g, W, H, "", "OK");
    targets.push({ id: "dismiss", top: 0, bottom: H });
  } else if (state === "menu") {
    g.fillStyle = C.text; g.textAlign = "left"; g.font = FONT(800, 38);
    g.fillText(TEXT.menuTitle, 30, 112);
    g.font = FONT(500, 23); g.fillStyle = C.dim;
    const name = data.name || "";
    g.fillText(name.length > 30 ? name.slice(0, 29) + "\u2026" : name, 30, 146);
    menuItems().forEach((it, i) => {
      const top = 170 + i * 100, sel = i === (data.selected || 0);
      const row = { num: String(i + 1), label: it.label, value: it.id === "advanced" ? "\u2026" : "\u203a" };
      drawRow(g, 22, top, W - 44, ROW_H, row, sel, sel && data.popped, it.id === "advanced");
      targets.push({ id: it.id, top, bottom: top + ROW_H, left: 22, right: W - 22 });
    });
    softKeys(g, W, H, TEXT.softLeft, TEXT.softRight);
    targets.push({ id: "back", top: H - 62, bottom: H, right: true });
  } else if (state === "advanced") {
    g.fillStyle = C.text; g.textAlign = "left"; g.font = FONT(800, 38);
    g.fillText(TEXT.advancedTitle, 30, 112);
    g.font = FONT(500, 23); g.fillStyle = C.dim; g.fillText(TEXT.advancedSub, 30, 146);
    const st = data.settings || defaultSettings();
    SETTINGS.forEach((set, i) => {
      const top = 170 + i * 100, sel = i === (data.selectedSetting || 0);
      const row = { num: String(i + 1), label: set.label, value: valueLabel(st, set) };
      drawRow(g, 22, top, W - 44, ROW_H, row, sel, sel && data.popped, false);
      targets.push({ id: set.key, top, bottom: top + ROW_H, left: 22, right: W - 22 });
    });
    softKeys(g, W, H, TEXT.softChange, TEXT.softRight);
    targets.push({ id: "back", top: H - 62, bottom: H, right: true });
  }
  return targets;
}

const ROW_H = 88;

// One list row: number, label and a value (or arrow) on the right. A popped row leaves an empty slot.
function drawRow(g, x, top, w, h, row, sel, popped, quiet) {
  if (popped) {
    // the selected key has risen out of the list in 3D: leave its empty slot behind
    g.fillStyle = "#050b0f"; g.fillRect(x, top, w, h);
    g.strokeStyle = C.yellow; g.lineWidth = 3; g.setLineDash([12, 10]);
    g.strokeRect(x + 2, top + 2, w - 4, h - 4); g.setLineDash([]);
    return;
  }
  g.fillStyle = sel ? C.yellow : C.bar; g.fillRect(x, top, w, h);
  if (quiet && !sel) { g.strokeStyle = C.dim; g.lineWidth = 2; g.setLineDash([8, 8]); g.strokeRect(x + 1, top + 1, w - 2, h - 2); g.setLineDash([]); }
  g.fillStyle = sel ? C.ink : C.text; g.textAlign = "left";
  g.font = FONT(900, 30); g.fillText(row.num, x + 22, top + h / 2 + 11);
  g.font = FONT(800, 26);
  const valueW = g.measureText(row.value).width;
  fitFont(g, row.label, 800, 36, w - 66 - 22 - valueW - 16);
  g.fillText(row.label, x + 66, top + h / 2 + 12);
  g.textAlign = "right"; g.font = FONT(800, 26);
  g.fillStyle = sel ? C.ink : C.yellow; g.fillText(row.value, x + w - 22, top + h / 2 + 10);
  g.textAlign = "left";
}

// largest font size (down to 60%) at which text fits in maxW pixels; leaves it set on g
function fitFont(g, text, weight, size, maxW) {
  let px = size;
  g.font = FONT(weight, px);
  while (px > size * 0.6 && g.measureText(text).width > maxW) { px -= 1; g.font = FONT(weight, px); }
}

// The face of the 3D key that pops out of a list (drawn on its own small canvas).
export function drawPopRow(g, W, H, row) {
  g.fillStyle = C.yellow; g.fillRect(0, 0, W, H);
  g.strokeStyle = C.ink; g.lineWidth = 8; g.strokeRect(4, 4, W - 8, H - 8);
  g.fillStyle = C.ink; g.textBaseline = "middle"; g.textAlign = "left";
  g.font = FONT(900, 40); g.fillText(row.num, 30, H / 2 + 3);
  g.font = FONT(900, row.value.length > 3 ? 32 : 52);
  const valueW = g.measureText(row.value).width;
  fitFont(g, row.label, 900, 48, W - 80 - 30 - valueW - 20);
  g.fillText(row.label, 80, H / 2 + 3);
  g.textAlign = "right"; g.font = FONT(900, row.value.length > 3 ? 32 : 52); g.fillText(row.value, W - 30, H / 2 + 3);
  g.textBaseline = "alphabetic"; g.textAlign = "left";
}
