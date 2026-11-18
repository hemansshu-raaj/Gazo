// End-to-end browser test of the whole website, for playwright-cli (https://playwright.dev/agent-cli).
//   1) start the site:   make web            (http://localhost:8000)
//   2) in another shell: make web-e2e
// It throws (and playwright-cli prints the error) at the first step that fails.
async (page) => {
  const BASE = "http://localhost:8000";
  const log = [];
  const ok = (cond, msg) => { if (!cond) throw new Error("E2E FAIL: " + msg); log.push("ok  " + msg); };
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(BASE + "/");
  await page.waitForFunction(() => window.__home && window.__home.phone, null, { timeout: 30000 });
  ok(await page.locator("#title svg path.letter").count() >= 7, "title lettering loaded");
  await page.waitForSelector(".example", { timeout: 30000 });
  ok(await page.locator(".example").count() === 5, "5 example cards");

  // dragging a file over the screen shows "Upload here"
  const centre = await page.evaluate(() => {
    const s = window.__home.phone.screenRect();
    return { x: (s.left + s.right) / 2, y: (s.top + s.bottom) / 2 };
  });
  const state = await page.evaluate(({ x, y }) => {
    const dt = new DataTransfer();
    dt.items.add(new File([new Uint8Array([255, 216, 255])], "holiday.jpg", { type: "image/jpeg" }));
    const el = document.elementFromPoint(x, y);
    el.dispatchEvent(new DragEvent("dragover", { bubbles: true, cancelable: true, clientX: x, clientY: y, dataTransfer: dt }));
    const hover = window.__home.state;
    el.dispatchEvent(new DragEvent("drop", { bubbles: true, cancelable: true, clientX: x, clientY: y, dataTransfer: dt }));
    return hover;
  }, centre);
  ok(state === "hover", "drag over the screen shows Upload here");
  await page.waitForFunction(() => window.__home.state === "error", null, { timeout: 15000 });
  ok(true, "a JPG is refused with the needs-RAW message");

  // a real RAW upload opens the menu
  await page.setInputFiles("#file", "examples/sidd_normal_nexus6.npz");
  await page.waitForFunction(() => window.__home.state === "menu", null, { timeout: 30000 });
  ok(true, "RAW upload opens the Classic / Learned / Hybrid menu");

  // Advanced screen: 4 opens it, 2 turns auto exposure on, Escape goes back to the ISP list
  await page.keyboard.press("4");
  ok(await page.evaluate(() => window.__home.state) === "advanced", "4 opens the Advanced screen");
  await page.keyboard.press("2");
  await page.keyboard.press("Escape");
  ok(await page.evaluate(() => window.__home.state) === "menu", "Escape returns to the ISP list");

  // choose Hybrid -> transition -> method page (with the setting carried over)
  await page.keyboard.press("3");
  await page.waitForURL(/hybrid\.html\?upload=.*ae=1/, { timeout: 15000 });
  ok(true, "the Advanced setting travels to the method page");
  ok(true, "pressing 3 flies into the lens and opens hybrid.html");
  await page.waitForSelector("#scroll-cue:not([hidden])", { timeout: 90000 });
  const stages = await page.locator(".stage-sec").count();
  ok(stages === 7, `hybrid page shows 7 stages (got ${stages})`);
  ok(await page.locator(".stage-sec .loupe img").count() >= 5, "stages have photosite close-ups");

  // comparison at the bottom
  await page.evaluate(() => document.querySelector("#compare").scrollIntoView());
  await page.waitForSelector(".cmp", { timeout: 120000 });
  ok(await page.locator(".cmp").count() === 4, "comparison shows 3 ISPs + ground truth");
  ok(await page.locator(".metrics .best").count() === 3, "best metric badges shown");

  // the other method pages work on their own (no selection needed)
  for (const [m, n] of [["classic", 8], ["learned", 4]]) {
    await page.goto(`${BASE}/${m}.html?example=sidd_iso10000_pixel`);
    await page.waitForSelector("#scroll-cue:not([hidden])", { timeout: 90000 });
    ok(await page.locator(".stage-sec").count() === n, `${m} page shows ${n} stages`);
  }

  // phone-sized screen: nothing sticks out sideways
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(BASE + "/");
  await page.waitForTimeout(1500);
  ok(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), "no sideways scroll at 390 px");

  ok(errors.length === 0, "no JavaScript errors" + (errors.length ? ": " + errors.join(" | ") : ""));
  return log.join("\n");
}
