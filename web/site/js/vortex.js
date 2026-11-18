// Full-screen WebGL backdrop for the method pages: the lens-flare image spinning in a swirl ("vortex"),
// with the processed photo twisting out of it.
//
//   const fx = createVortex(canvas, { flare: "assets/fx/flare.jpg" });
//   fx.set({ glow: 1, swirl: 1 })            // tweak any uniform below (eased smoothly)
//   await fx.showPhoto(url)                  // photo un-twists out of the vortex
//   fx.openIris()                            // circular reveal from black (after the lens transition)
import * as THREE from "three";

const frag = /* glsl */ `
  uniform sampler2D uFlare; uniform sampler2D uPhoto;
  uniform vec2 uRes; uniform float uTime; uniform float uFlareAspect; uniform float uPhotoAspect;
  uniform float uSwirl; uniform float uGlow; uniform float uPhotoMix; uniform float uTwist; uniform float uIris;
  uniform vec2 uPhotoCentre; uniform float uPhotoHeight;
  varying vec2 vUv;

  vec3 flareAt(vec2 q) {                       // q: centred, 1 unit = screen height
    vec2 uv = vec2(q.x / uFlareAspect, q.y) + 0.5;
    return texture2D(uFlare, clamp(uv, 0.001, 0.999)).rgb;
  }
  void main() {
    vec2 p = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
    float r = length(p), a = atan(p.y, p.x);

    // vortex: the flare turns, twists more near the centre, and is smeared along the turn (swirl blur)
    vec3 col = vec3(0.0);
    for (int i = 0; i < 14; i++) {
      float o = (float(i) / 13.0 - 0.5) * 0.42 * uSwirl;
      float aa = a + uTime * 0.10 + uSwirl * 1.4 * exp(-r * 2.2) + o * (1.0 - min(r, 1.0) * 0.5);
      float rr = r * (1.0 + 0.04 * sin(uTime * 0.6)) * (1.0 - 0.06 * uSwirl * float(i) / 13.0);
      col += flareAt(vec2(cos(aa), sin(aa)) * rr);
    }
    col = col / 14.0 * uGlow;

    // the photo, twisted by uTwist (radians at its centre), fading in with uPhotoMix
    if (uPhotoMix > 0.001) {
      vec2 d = p - uPhotoCentre;
      float dr = length(d) / uPhotoHeight;
      float tw = uTwist * (1.0 - smoothstep(0.0, 0.75, dr));
      float c = cos(tw), s = sin(tw);
      vec2 dd = mat2(c, -s, s, c) * d;
      vec2 uv = vec2(dd.x / (uPhotoHeight * uPhotoAspect), dd.y / uPhotoHeight) + 0.5;
      if (all(greaterThan(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)))) {
        vec3 ph = texture2D(uPhoto, uv).rgb;
        col = mix(col, ph, uPhotoMix);
      }
    }

    col *= 1.0 - smoothstep(uIris - 0.03, uIris, r);   // iris (lens opening)
    gl_FragColor = vec4(col, 1.0);
  }`;

export function createVortex(canvas, { flare = "assets/fx/flare.jpg" } = {}) {
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const loader = new THREE.TextureLoader();
  const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  blank.needsUpdate = true;

  const u = {
    uFlare: { value: blank }, uPhoto: { value: blank },
    uRes: { value: new THREE.Vector2() }, uTime: { value: 0 },
    uFlareAspect: { value: 1.79 }, uPhotoAspect: { value: 1.5 },
    uSwirl: { value: 1 }, uGlow: { value: 1 }, uPhotoMix: { value: 0 }, uTwist: { value: 0 }, uIris: { value: 3 },
    uPhotoCentre: { value: new THREE.Vector2(0, -0.04) }, uPhotoHeight: { value: 0.6 },
  };
  loader.load(flare, (t) => {
    t.colorSpace = THREE.NoColorSpace; t.minFilter = THREE.LinearFilter;
    u.uFlare.value = t; u.uFlareAspect.value = t.image.width / t.image.height;
  });
  const mat = new THREE.ShaderMaterial({
    uniforms: u, fragmentShader: frag,
    vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
  });
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  // targets: every frame each uniform eases towards its target
  const target = { uSwirl: 1, uGlow: 1, uPhotoMix: 0, uTwist: 0, uIris: 3 };
  const rate = { uSwirl: 0.04, uGlow: 0.05, uPhotoMix: 0.08, uTwist: 0.045, uIris: 0.06 };

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setPixelRatio(dpr); renderer.setSize(w, h, false);
    u.uRes.value.set(w * dpr, h * dpr);
    fitPhoto();
  }
  // keep the photo inside the screen with room for the title and the scroll cue
  function fitPhoto() {
    const ar = u.uRes.value.x / u.uRes.value.y;
    let hgt = 0.6;
    if (u.uPhotoAspect.value * hgt > ar * 0.9) hgt = (ar * 0.9) / u.uPhotoAspect.value;
    u.uPhotoHeight.value = hgt;
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  const clock = new THREE.Clock();
  (function frame() {
    const dt = Math.min(clock.getDelta(), 0.05) * 60;    // in 60 fps frames
    if (!reduceMotion) u.uTime.value += dt / 60;
    for (const k in target) {
      u[k].value += (target[k] - u[k].value) * (reduceMotion ? 1 : Math.min(1, rate[k] * dt));
    }
    renderer.render(scene, cam);
    requestAnimationFrame(frame);
  })();

  return {
    set(vals) { for (const k in vals) target["u" + k[0].toUpperCase() + k.slice(1)] = vals[k]; },
    openIris() { u.uIris.value = 0; target.uIris = 3; },
    showPhoto(url) {
      return new Promise((resolve, reject) => {
        loader.load(url, (t) => {
          t.colorSpace = THREE.NoColorSpace; t.minFilter = THREE.LinearFilter;
          u.uPhoto.value = t; u.uPhotoAspect.value = t.image.width / t.image.height;
          fitPhoto();
          u.uTwist.value = reduceMotion ? 0 : 9;          // start wound up, then unwind
          target.uTwist = 0; target.uPhotoMix = 1; target.uSwirl = 0.55;
          resolve();
        }, undefined, reject);
      });
    },
  };
}
