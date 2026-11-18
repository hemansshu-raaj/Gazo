// The 3D phone, drawn like a comic (flat colour bands, halftone dots, rough "boiling" ink lines).
//
//   import { createPhone } from "./phone3d.js";
//   const phone = await createPhone(document.querySelector("#stage"), { model: "assets/models/phone.glb" });
//   phone.screenCanvas      // 240x320-ratio <canvas>: draw the phone's screen UI on it, then
//   phone.refreshScreen()   // ...call this to show it
//   phone.screenRect()      // where the screen is on the page (for drag-and-drop hit tests)
//   phone.faceFront()       // turn the phone back towards the viewer
//   phone.slide(true)       // slide the phone open (false closes it)
//   phone.zoom(true)        // camera moves in to frame the screen (false: back out)
//   phone.setPop(rect, draw) // a 3D key rises out of the screen over rect (screen-canvas px); null hides it
//   await phone.flyIntoLens() // turn round and push into the back camera; resolves with the lens {x, y} on the page
//   onScreenTap({u, v})     // option: called when the screen is tapped/clicked (u, v in 0..1 from top-left)
//
// Look and feel knobs are in STYLE and PALETTE below. The model comes from web/blender/build_phone.py.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export const STYLE = {
  inkColor: [0.06, 0.05, 0.07],
  inkWidth: 2.4,        // outline thickness in CSS pixels
  inkWobble: 2.6,       // how far the lines wander, in CSS pixels
  boilFps: 12,          // lines redraw 12 times a second ("on twos"), like the film
  halftoneCell: 6.0,    // halftone dot spacing in CSS pixels
  misprint: 1.4,        // red/cyan colour offset at the edges, in CSS pixels
  floatAmount: 0.18,    // hover bob height (cm)
  returnAfterMs: 2500,  // idle time before the phone turns back to face you
};

// [lit colour, shadow colour, highlight strength 0..1, halftone/hatching colour] per material name
export const PALETTE = {
  body:         [[0.20, 0.20, 0.24], [0.06, 0.06, 0.08], 1.0, [0.16, 0.22, 0.55]],
  body_back:    [[0.24, 0.24, 0.28], [0.07, 0.07, 0.09], 0.6, [0.16, 0.22, 0.55]],
  trim:         [[0.88, 0.89, 0.92], [0.42, 0.44, 0.50], 1.0, [0.55, 0.62, 0.85]],
  metal:        [[0.70, 0.72, 0.76], [0.30, 0.31, 0.35], 1.0],
  glass:        [[0.04, 0.04, 0.05], [0.01, 0.01, 0.02], 1.0],
  key:          [[0.36, 0.36, 0.40], [0.12, 0.12, 0.14], 0.5, [0.18, 0.24, 0.58]],
  key_dark:     [[0.20, 0.20, 0.23], [0.07, 0.07, 0.09], 0.4],
  key_label:    [[0.95, 0.95, 0.92], [0.70, 0.70, 0.68], 0.0],
  icon:         [[0.90, 0.91, 0.95], [0.60, 0.62, 0.68], 0.0],
  led_blue:     [[0.30, 0.68, 1.00], [0.22, 0.55, 0.95], 0.0],
  accent_red:   [[0.93, 0.16, 0.13], [0.52, 0.04, 0.06], 0.6],
  accent_green: [[0.20, 0.78, 0.33], [0.05, 0.38, 0.14], 0.6],
  lens:         [[0.10, 0.16, 0.34], [0.02, 0.03, 0.09], 0.12],
  flash:        [[1.00, 0.97, 0.86], [0.75, 0.70, 0.55], 0.3],
};

// ------------------------------------------------------------------ shaders
const toonVert = /* glsl */ `
  varying vec3 vNormal; varying vec3 vView; varying vec2 vUv;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = -mv.xyz; vUv = uv;
    gl_Position = projectionMatrix * mv;
  }`;

const toonFrag = /* glsl */ `
  uniform vec3 uLit; uniform vec3 uShadow; uniform float uSpec; uniform vec3 uDotColor;
  uniform vec3 uLightDir; uniform float uCell;
  uniform sampler2D uMap; uniform float uHasMap;
  varying vec3 vNormal; varying vec3 vView; varying vec2 vUv;

  float halftone(float amount) {               // 1 inside a dot, 0 outside; dots grow with amount
    vec2 p = gl_FragCoord.xy * mat2(0.7071, -0.7071, 0.7071, 0.7071) / uCell;
    float r = length(fract(p) - 0.5);
    return 1.0 - step(sqrt(amount) * 0.62, r);
  }
  void main() {
    if (uHasMap > 0.5) { gl_FragColor = vec4(texture2D(uMap, vUv).rgb, 1.0); return; }
    vec3 n = normalize(vNormal), l = normalize(uLightDir), v = normalize(vView);
    if (!gl_FrontFacing) n = -n;
    float ndl = dot(n, l);
    vec3 col;
    vec2 hp = gl_FragCoord.xy * mat2(0.7071, 0.7071, -0.7071, 0.7071);
    float hatch = step(0.62, fract(hp.x / (uCell * 0.9)));                       // diagonal pen strokes
    if (ndl > 0.45)       col = uLit;                                            // lit band
    else if (ndl > -0.05) col = mix(uLit, uDotColor, halftone(0.5));            // mid band: coloured dots
    else                  col = mix(uShadow, uDotColor * 0.6, hatch * 0.8);     // shadow band: hatching
    float spec = pow(max(dot(n, normalize(l + v)), 0.0), 60.0);
    col = mix(col, vec3(1.0), step(0.6, spec) * uSpec);                         // hard comic highlight
    float rim = 1.0 - max(dot(n, v), 0.0);
    col += step(0.82, rim) * 0.10 * uSpec;                                       // thin rim light
    gl_FragColor = vec4(col, 1.0);
  }`;

const quadVert = /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const inkFrag = /* glsl */ `
  uniform sampler2D tColor; uniform sampler2D tNormal; uniform sampler2D tDepth;
  uniform vec2 uRes; uniform float uSeed; uniform float uWidth; uniform float uWobble; uniform float uMis;
  uniform vec3 uInk; uniform float uNear; uniform float uFar;
  varying vec2 vUv;

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
  }
  float linDepth(vec2 uv) {
    float z = texture2D(tDepth, uv).x * 2.0 - 1.0;
    return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear));
  }
  float edge(vec2 uv, float w) {               // normal + depth discontinuities around uv
    vec2 px = w / uRes;
    vec3 n0 = texture2D(tNormal, uv).rgb;
    float d0 = linDepth(uv), e = 0.0;
    for (int i = 0; i < 4; i++) {
      vec2 o = (i == 0) ? vec2(px.x, 0) : (i == 1) ? vec2(-px.x, 0) : (i == 2) ? vec2(0, px.y) : vec2(0, -px.y);
      vec3 n1 = texture2D(tNormal, uv + o).rgb;
      e = max(e, length(n1 - n0) * 1.6);
      e = max(e, abs(linDepth(uv + o) - d0) / max(d0, 1e-3) * 22.0);
    }
    return clamp(e, 0.0, 1.0);
  }
  void main() {
    vec2 px = 1.0 / uRes;
    // lines wander: offset the lookup by a noise field that changes every "drawing"
    vec2 j = vec2(noise(vUv * 9.0 + uSeed), noise(vUv * 9.0 - uSeed + 17.0)) - 0.5;
    vec2 j2 = vec2(noise(vUv * 23.0 + uSeed * 1.7), noise(vUv * 23.0 + 5.0 - uSeed)) - 0.5;
    float e1 = edge(vUv + j * uWobble * px * 2.0, uWidth);
    float e2 = edge(vUv + j2 * uWobble * px * 3.0, uWidth * 0.7);      // second, lighter pencil pass
    float grain = 0.55 + 0.45 * noise(gl_FragCoord.xy * 0.35 + uSeed * 3.0); // dry-brush texture
    float gaps = smoothstep(0.18, 0.4, noise(vUv * vec2(60.0, 60.0) + uSeed * 5.0));  // broken, sketchy strokes
    float ink = max(smoothstep(0.2, 0.55, e1) * (0.8 + 0.2 * grain), smoothstep(0.3, 0.7, e2) * 0.75 * grain * gaps);

    // colour pass with a slight print misregistration: red one way, blue the other
    vec4 c = texture2D(tColor, vUv);
    vec4 cr = texture2D(tColor, vUv + vec2(uMis, 0.0) * px);
    vec4 cb = texture2D(tColor, vUv - vec2(uMis, uMis * 0.5) * px);
    vec4 col = vec4(cr.r, c.g, cb.b, max(c.a, max(cr.a, cb.a)));
    gl_FragColor = mix(col, vec4(uInk, 1.0), ink);   // premultiplied: background stays transparent
  }`;

// ------------------------------------------------------------------ the phone
export async function createPhone(container, { model = "assets/models/phone.glb", screenSize = [480, 640],
                                               onScreenTap = null, onScreenHover = null, frameEl = null } = {}) {
  // The canvas may cover the whole window (behind the page). frameEl is the box on the page the phone
  // should appear in when it is not zoomed; zooming / flying blend that framing into the full window.
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, premultipliedAlpha: true });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;   // colours above are used exactly as written
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);
  const canvas = renderer.domElement;
  canvas.style.touchAction = frameEl ? "pan-y" : "none";   // full-window canvas: let the page scroll

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(26, 1, 1, 200);
  camera.position.set(0, 0, 27);
  const lightDir = new THREE.Vector3(-0.55, 0.65, 0.75).normalize();   // fixed relative to the viewer

  // screen UI canvas
  const screenCanvas = document.createElement("canvas");
  [screenCanvas.width, screenCanvas.height] = screenSize;
  const screenTex = new THREE.CanvasTexture(screenCanvas);
  screenTex.colorSpace = THREE.NoColorSpace;
  screenTex.flipY = false;   // glTF UV convention
  screenTex.anisotropy = 4;

  const gltf = await new GLTFLoader().loadAsync(model);
  const phone = gltf.scene;
  const parts = {};
  phone.traverse((o) => {
    parts[o.name] = o;
    if (!o.isMesh) return;
    const name = o.material.name;
    const isScreen = o.name === "screen";
    const [lit, shadow, spec, dot] = PALETTE[name] || [[0.5, 0.5, 0.5], [0.2, 0.2, 0.2], 0.5];
    o.material = new THREE.ShaderMaterial({
      vertexShader: toonVert, fragmentShader: toonFrag,
      uniforms: {
        uLit: { value: new THREE.Vector3(...lit) }, uShadow: { value: new THREE.Vector3(...shadow) },
        uSpec: { value: spec }, uDotColor: { value: new THREE.Vector3(...(dot || shadow)) }, uLightDir: { value: lightDir }, uCell: { value: STYLE.halftoneCell },
        uMap: { value: isScreen ? screenTex : null }, uHasMap: { value: isScreen ? 1 : 0 },
      },
    });
  });
  // centre the model and hang it in a "floater" group that we rotate
  const box = new THREE.Box3().setFromObject(phone);
  phone.position.sub(box.getCenter(new THREE.Vector3()));
  const phoneHomeY = phone.position.y;
  let baseZ = 27;
  const spinner = new THREE.Group();   // user rotation
  const floater = new THREE.Group();   // hover bob
  spinner.add(phone); floater.add(spinner); scene.add(floater);
  spinner.rotation.set(-0.12, 0.42, 0.04);

  // ---- render targets for the ink pass
  const mk = () => new THREE.WebGLRenderTarget(1, 1, { samples: 0 });
  const rtColor = mk();
  const rtNormal = mk();
  rtNormal.depthTexture = new THREE.DepthTexture(1, 1);
  const normalMat = new THREE.MeshNormalMaterial();
  const inkMat = new THREE.ShaderMaterial({
    vertexShader: quadVert, fragmentShader: inkFrag, transparent: true, depthTest: false,
    uniforms: {
      tColor: { value: rtColor.texture }, tNormal: { value: rtNormal.texture }, tDepth: { value: rtNormal.depthTexture },
      uRes: { value: new THREE.Vector2() }, uSeed: { value: 0 }, uWidth: { value: 1 }, uWobble: { value: 1 },
      uMis: { value: 1 }, uInk: { value: new THREE.Vector3(...STYLE.inkColor) },
      uNear: { value: camera.near }, uFar: { value: camera.far },
    },
  });
  const quadScene = new THREE.Scene();
  quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), inkMat));
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    camera.aspect = w / h;
    applyFraming();
    const W = Math.round(w * dpr), H = Math.round(h * dpr);
    rtColor.setSize(W, H); rtNormal.setSize(W, H);
    inkMat.uniforms.uRes.value.set(W, H);
    inkMat.uniforms.uWidth.value = STYLE.inkWidth * dpr;
    inkMat.uniforms.uWobble.value = STYLE.inkWobble * dpr;
    inkMat.uniforms.uMis.value = STYLE.misprint * dpr;
    scene.traverse((o) => o.material?.uniforms?.uCell && (o.material.uniforms.uCell.value = STYLE.halftoneCell * dpr));
    if (ready) draw();   // resizing wipes the canvas: redraw now, or it shows blank until the next frame
  }
  let ready = false;

  // ---- framing: where on the canvas the phone is drawn, and how big
  const BASE_FOV = 26, BASE_TAN = Math.tan(THREE.MathUtils.degToRad(BASE_FOV / 2));
  let framingP = 0;                         // 0 = inside frameEl, 1 = the whole window (zoom / flight)
  function frameRect() {
    const W = canvas.clientWidth || 1, H = canvas.clientHeight || 1;
    if (!frameEl) return { cx: W / 2, cy: H / 2, w: W, h: H, W, H };
    const r = frameEl.getBoundingClientRect(), c = canvas.getBoundingClientRect();
    return { cx: r.left - c.left + r.width / 2, cy: r.top - c.top + r.height / 2, w: r.width || 1, h: r.height || 1, W, H };
  }
  function applyFraming() {
    const f = frameRect(), p = framingP;
    // the un-zoomed camera distance fits the whole phone into the frame box (narrow boxes: further away)
    baseZ = 27 * Math.max(1, 0.62 / (f.w / f.h));
    const scale = (f.H / f.h) * (1 - p) + p;                       // a smaller box = a wider lens
    camera.fov = 2 * THREE.MathUtils.radToDeg(Math.atan(BASE_TAN * scale));
    const cx = f.cx * (1 - p) + (f.W / 2) * p, cy = f.cy * (1 - p) + (f.H / 2) * p;
    camera.setViewOffset(f.W, f.H, f.W / 2 - cx, f.H / 2 - cy, f.W, f.H);   // shift the picture onto the box
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  // ---- drag to rotate (any direction, all the way round), with inertia
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let dragging = false, last = null, vel = { x: 0, y: 0 }, lastInput = 0;
  const FACING = { x: -0.12, y: 0.42 };        // resting angle
  const ZOOM_FACING = { x: -0.16, y: 0.34 };    // flatter angle while the screen is zoomed in
  let down = null;
  canvas.addEventListener("pointerdown", (e) => {
    if (frameEl && framingP < 0.5) {
      const r = frameEl.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
    }
    dragging = true; last = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId);
    down = { x: e.clientX, y: e.clientY, t: performance.now() };
    canvas.style.cursor = "grabbing";
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) {
      if (onScreenHover) onScreenHover(screenUV(e.clientX, e.clientY), e);
      return;
    }
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    vel = { x: dy * 0.008, y: dx * 0.008 };
    spinner.rotation.y += vel.y; spinner.rotation.x += vel.x;
    lastInput = performance.now();
  });
  const end = () => { dragging = false; canvas.style.cursor = "grab"; lastInput = performance.now(); };
  canvas.addEventListener("pointerup", (e) => {
    end();
    // a short press that barely moved is a tap, not a drag
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6 && performance.now() - down.t < 500) {
      vel = { x: 0, y: 0 };
      const uv = screenUV(e.clientX, e.clientY);
      if (uv && onScreenTap) onScreenTap(uv);
    }
    down = null;
  });
  canvas.addEventListener("pointercancel", end);
  canvas.style.cursor = "grab";

  // keyboard: arrows rotate, so the phone is usable without a mouse
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "3D phone. Drag or use the arrow keys to turn it.");
  canvas.addEventListener("keydown", (e) => {
    if (zoomTo && (e.key === "ArrowUp" || e.key === "ArrowDown")) return;   // zoomed in: they move the menu
    const k = { ArrowLeft: [0, -0.3], ArrowRight: [0, 0.3], ArrowUp: [-0.3, 0], ArrowDown: [0.3, 0] }[e.key];
    if (!k) return;
    e.preventDefault();
    spinner.rotation.x += k[0]; spinner.rotation.y += k[1]; lastInput = performance.now();
  });

  function faceFront() { lastInput = 0; }

  // ---- the slider: open() pushes the screen half up to show the keypad
  const slideHome = parts.slide.position.y;
  const SLIDE_TRAVEL = 4.6;                       // cm
  let slideFrom = 0, slideTo = 0, slideStart = 0, slideP = 0;
  const easeOutBack = (x) => 1 + 2.2 * Math.pow(x - 1, 3) + 1.2 * Math.pow(x - 1, 2);
  function slide(open) {
    slideFrom = slideP; slideTo = open ? 1 : 0; slideStart = performance.now();
  }
  function updateSlide() {
    const k = Math.min(1, (performance.now() - slideStart) / (reduceMotion ? 1 : 420));
    slideP = slideFrom + (slideTo - slideFrom) * (slideTo > slideFrom ? easeOutBack(k) : k * k);
    parts.slide.position.y = slideHome + slideP * SLIDE_TRAVEL;
    phone.position.y = phoneHomeY - slideP * SLIDE_TRAVEL / 2;   // keep the taller phone centred
    // zoom: frame the screen so the menu is big and easy to pick from
    const kz = Math.min(1, Math.max(0, (performance.now() - zoomStart) / (reduceMotion ? 1 : 750)));
    zoomP = zoomFrom + (zoomTo - zoomFrom) * ease(kz);
    const far = new THREE.Vector3(0, 0, baseZ * (1 + 0.28 * Math.min(1, Math.max(0, slideP))));
    if (zoomP > 0.0001) {
      const sc = parts.screen.localToWorld(screenCentre.clone());   // middle of the screen glass
      const t = 2 * BASE_TAN;
      const d = Math.max(5.1 / (0.6 * t), 5.6 / (0.94 * t * camera.aspect));    // screen height / phone width
      // aim a little above the screen so it sits slightly low in the view, clear of the page title
      const near = new THREE.Vector3(sc.x, sc.y + d * t * 0.06, sc.z + d);
      camera.position.lerpVectors(far, near, zoomP);
    } else camera.position.copy(far);
  }
  let zoomFrom = 0, zoomTo = 0, zoomStart = 0, zoomP = 0;
  parts.screen.geometry.computeBoundingBox();
  const screenCentre = parts.screen.geometry.boundingBox.getCenter(new THREE.Vector3());
  function zoom(on) { zoomFrom = zoomP; zoomTo = on ? 1 : 0; zoomStart = performance.now() + (on ? 220 : 0); lastInput = 0; }

  // ---- the pop-out button: a real 3D key that rises out of the screen's menu
  const popCanvas = document.createElement("canvas");
  popCanvas.width = 512; popCanvas.height = 104;   // same shape as a screen list row
  const popTex = new THREE.CanvasTexture(popCanvas);
  popTex.colorSpace = THREE.NoColorSpace; popTex.anisotropy = 4;
  const side = (lit, shadow) => new THREE.ShaderMaterial({
    vertexShader: toonVert, fragmentShader: toonFrag,
    uniforms: { uLit: { value: new THREE.Vector3(...lit) }, uShadow: { value: new THREE.Vector3(...shadow) },
      uSpec: { value: 0.4 }, uDotColor: { value: new THREE.Vector3(0.75, 0.42, 0.0) }, uLightDir: { value: lightDir },
      uCell: { value: STYLE.halftoneCell }, uMap: { value: null }, uHasMap: { value: 0 } },
  });
  const sideMat = side([1.0, 0.84, 0.04], [0.78, 0.55, 0.0]);
  const faceMat = side([1, 1, 1], [1, 1, 1]);
  faceMat.uniforms.uMap.value = popTex; faceMat.uniforms.uHasMap.value = 1;
  const pop = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
    [sideMat, sideMat, sideMat, sideMat, faceMat, sideMat]);   // +z face shows the label
  pop.visible = false;
  parts.screen.add(pop);
  // screen-local frame from the screen mesh's own UVs (so it follows any model change)
  const uvFrame = (() => {
    const g = parts.screen.geometry, P = g.attributes.position, U = g.attributes.uv;
    let p00, p10, p01;
    for (let i = 0; i < U.count; i++) {
      const u = Math.round(U.getX(i)), v = Math.round(U.getY(i));
      const pt = new THREE.Vector3().fromBufferAttribute(P, i);
      if (u === 0 && v === 0) p00 = pt; if (u === 1 && v === 0) p10 = pt; if (u === 0 && v === 1) p01 = pt;
    }
    const ex = p10.clone().sub(p00), ey = p01.clone().sub(p00);   // along u, along v (v runs top -> bottom)
    const x = ex.clone().normalize(), y = ey.clone().negate().normalize(), z = new THREE.Vector3().crossVectors(x, y);
    return { at: (u, v) => p00.clone().addScaledVector(ex, u).addScaledVector(ey, v), w: ex.length(), h: ey.length(),
             basis: new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z)), z };
  })();
  pop.quaternion.copy(uvFrame.basis);
  const popState = { on: false, z: 0, vz: 0, pos: null, size: null };
  const POP_HEIGHT = 0.95, POP_DEPTH = 0.32;   // cm above the glass, key thickness
  // rect in screen-canvas pixels {left, top, right, bottom}; draw(ctx, w, h) paints the key's face
  function setPop(rect, draw, canvasSize = [480, 640]) {
    if (!rect) { popState.on = false; return; }
    const [W, H] = canvasSize;
    const u = (rect.left + rect.right) / 2 / W, v = (rect.top + rect.bottom) / 2 / H;
    const centre = uvFrame.at(u, v);
    const size = [(rect.right - rect.left) / W * uvFrame.w * 1.06, (rect.bottom - rect.top) / H * uvFrame.h * 1.12];
    const moved = !popState.pos || popState.pos.distanceTo(centre) > 1e-3;
    popState.pos = centre; popState.size = size;
    if (draw) { const g = popCanvas.getContext("2d"); g.setTransform(1, 0, 0, 1, 0, 0); draw(g, popCanvas.width, popCanvas.height); popTex.needsUpdate = true; }
    if (moved || !popState.on) { popState.z = reduceMotion ? POP_HEIGHT : 0; popState.vz = 0; }   // rises again
    popState.on = true;
  }
  function updatePop() {
    pop.visible = popState.on && zoomP > 0.6;
    if (!pop.visible) return;
    // springy rise with a little overshoot
    popState.vz += (POP_HEIGHT - popState.z) * 0.16; popState.vz *= 0.74; popState.z += popState.vz;
    pop.position.copy(popState.pos).addScaledVector(uvFrame.z, popState.z + POP_DEPTH / 2);
    // a little wobble while it springs up, settling flat
    pop.quaternion.copy(uvFrame.basis).multiply(new THREE.Quaternion().setFromEuler(
      new THREE.Euler(-popState.vz * 0.9, 0, popState.vz * 0.35)));
    pop.scale.set(popState.size[0], popState.size[1], POP_DEPTH);
  }

  // ---- fly into the back camera: turn the phone round, then push the camera into the lens
  let flight = null;
  parts.camera_lens.geometry.computeBoundingBox();
  const lensCentre = parts.camera_lens.geometry.boundingBox.getCenter(new THREE.Vector3());
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  function flyIntoLens({ turnMs = 650, pushMs = 900 } = {}) {
    slide(false);
    popState.on = false; pop.visible = false; zoomTo = 0;
    return new Promise((resolve) => {
      const twoPi = 2 * Math.PI;
      // nearest "back facing the viewer" angle, so it never spins the long way round
      const backY = Math.PI + Math.round((spinner.rotation.y - Math.PI) / twoPi) * twoPi;
      const backX = Math.round(spinner.rotation.x / twoPi) * twoPi;
      flight = {
        t0: performance.now(), turnMs, pushMs, resolve,
        from: { ry: spinner.rotation.y, rx: spinner.rotation.x, rz: spinner.rotation.z, fy: floater.position.y,
                cam: camera.position.clone(), fov: camera.fov },
        to: { ry: backY, rx: backX },
      };
    });
  }
  function updateFlight() {
    const f = flight, el = performance.now() - f.t0;
    const a = ease(Math.min(1, el / f.turnMs));
    spinner.rotation.y = f.from.ry + (f.to.ry - f.from.ry) * a;
    spinner.rotation.x = f.from.rx + (f.to.rx - f.from.rx) * a;
    spinner.rotation.z = f.from.rz * (1 - a);
    floater.position.y = f.from.fy * (1 - a); floater.rotation.z *= 1 - a;
    if (el > f.turnMs * 0.8) {
      const b = ease(Math.min(1, (el - f.turnMs * 0.8) / f.pushMs));
      scene.updateMatrixWorld(true);
      const lens = parts.camera_lens.localToWorld(lensCentre.clone());   // middle of the lens glass
      // stop 1.3 cm out: closer than the camera's 1 cm near plane the glass would be clipped away;
      // at this distance the lens still fills the whole window
      const end = lens.clone().add(new THREE.Vector3(0, 0, 1.3));
      // straight dolly: slide sideways and forward onto the lens, never turning the camera
      camera.position.lerpVectors(f.from.cam, end, b);
      camera.quaternion.identity();
      if (b >= 1 && f.resolve) { const r = f.resolve; f.resolve = null; r(lensOnPage()); }
    }
  }
  // the lens centre in page pixels (where the iris transition starts)
  function lensOnPage() {
    const v = parts.camera_lens.localToWorld(lensCentre.clone()).project(camera);
    const r = canvas.getBoundingClientRect();
    return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height };
  }

  // ---- which point of the screen (0..1 uv, origin top-left) is under a page position?
  const ray = new THREE.Raycaster();
  function screenUV(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((clientX - r.left) / r.width * 2 - 1, -(clientY - r.top) / r.height * 2 + 1), camera);
    const hit = ray.intersectObject(parts.screen, false)[0];
    return hit && hit.uv ? { u: hit.uv.x, v: hit.uv.y } : null;   // flipY=false: v already runs top->bottom
  }

  // ---- loop
  const clock = new THREE.Clock();
  let seed = 0, lastBoil = 0;
  function frame() {
    const t = clock.getElapsedTime();
    if (flight) {
      framingP = Math.min(1, framingP + 0.04);                     // the phone takes over the window
      applyFraming();
      updateFlight();
      if (t - lastBoil > 1 / STYLE.boilFps) { lastBoil = t; seed = (seed + 1) % 97; }
      inkMat.uniforms.uSeed.value = seed * 3.17;
      const k = Math.min(1, (performance.now() - slideStart) / 300);   // slider closes while turning
      slideP = slideFrom * (1 - k);
      parts.slide.position.y = slideHome + slideP * SLIDE_TRAVEL;
      phone.position.y = phoneHomeY - slideP * SLIDE_TRAVEL / 2;
      draw();
      requestAnimationFrame(frame);
      return;
    }
    if (!dragging) {
      spinner.rotation.y += vel.y; spinner.rotation.x += vel.x;
      vel.x *= 0.92; vel.y *= 0.92;
      const target = zoomTo ? ZOOM_FACING : FACING;
      if (performance.now() - lastInput > (zoomTo ? 700 : STYLE.returnAfterMs)) {
        // ease back to the nearest front-facing angle (whole turns are kept, so no rewinding)
        const ty = target.y + Math.round((spinner.rotation.y - target.y) / (2 * Math.PI)) * 2 * Math.PI;
        const tx = target.x + Math.round((spinner.rotation.x - target.x) / (2 * Math.PI)) * 2 * Math.PI;
        spinner.rotation.y += (ty - spinner.rotation.y) * 0.04;
        spinner.rotation.x += (tx - spinner.rotation.x) * 0.04;
      }
    }
    if (!reduceMotion) {
      floater.position.y = Math.sin(t * 1.3) * STYLE.floatAmount;
      floater.rotation.z = Math.sin(t * 0.9) * 0.025;
      if (t - lastBoil > 1 / STYLE.boilFps) { lastBoil = t; seed = (seed + 1) % 97; }
    }
    inkMat.uniforms.uSeed.value = seed * 3.17;
    framingP = zoomP;
    applyFraming();
    updateSlide();
    updatePop();

    draw();
    requestAnimationFrame(frame);
  }
  function draw() {
    renderer.setRenderTarget(rtColor); renderer.clear(); renderer.render(scene, camera);
    scene.overrideMaterial = normalMat;
    renderer.setRenderTarget(rtNormal); renderer.clear(); renderer.render(scene, camera);
    scene.overrideMaterial = null;
    renderer.setRenderTarget(null); renderer.clear(); renderer.render(quadScene, quadCam);
  }
  ready = true;
  requestAnimationFrame(frame);

  // where the screen is on the page, for drop-zone hit tests
  function screenRect() {
    const s = parts.screen;
    const g = s.geometry; g.computeBoundingBox();
    const b = g.boundingBox, r = canvas.getBoundingClientRect();
    const pts = [];
    for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) {
      const v = new THREE.Vector3(x, y, z).applyMatrix4(s.matrixWorld).project(camera);
      pts.push([r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]);
    }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    return { left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys) };
  }

  return {
    canvas, scene, camera, parts, spinner, floater, screenCanvas,
    refreshScreen: () => { screenTex.needsUpdate = true; },
    screenRect, screenUV, faceFront, slide, flyIntoLens, zoom, setPop,
  };
}
