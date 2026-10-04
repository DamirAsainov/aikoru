// AIKORU — локальный веб-интерфейс
const $ = (s) => document.querySelector(s);

const QUICK = {
  ru: ["что вокруг", "кто рядом", "найди дверь", "где светофор", "найди стул"],
  kk: ["айналада не бар", "бұл кім", "есікті тап", "бағдаршам қайда", "орындықты тап"],
};
const PLACEHOLDER = {
  ru: "что вокруг · найди дверь · кто это",
  kk: "айналада не бар · есікті тап · бұл кім",
};
const INTENT = {
  describe: "описание", find: "поиск", who: "кто рядом", remember: "запомнить",
  forget: "забыть", pause: "пауза", resume: "продолжить", ask: "вопрос", unknown: "?",
};

// ?theme=light|dark — принудительная тема (по умолчанию — как в системе)
const theme = new URLSearchParams(location.search).get("theme");
if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;

let lang = "ru";
try { lang = localStorage.getItem("aikoru.lang") || "ru"; } catch {}
let tracks = [];

// --- язык -------------------------------------------------------------------
function setLang(l) {
  lang = l;
  try { localStorage.setItem("aikoru.lang", l); } catch {}
  document.querySelectorAll(".seg button").forEach((b) =>
    b.setAttribute("aria-checked", String(b.dataset.lang === l)));
  $("#cmd").placeholder = PLACEHOLDER[l];
  const q = $("#quick");
  q.replaceChildren(...QUICK[l].map((text) => {
    const b = document.createElement("button");
    b.className = "btn"; b.type = "button"; b.textContent = text;
    b.onclick = () => send(text);
    return b;
  }));
}
document.querySelectorAll(".seg button").forEach((b) => (b.onclick = () => setLang(b.dataset.lang)));

// --- API --------------------------------------------------------------------
async function post(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  return r.json();
}
const send = (text) => post("/api/command", { text, lang });

$("#cmd-form").onsubmit = (e) => {
  e.preventDefault();
  const v = $("#cmd").value.trim();
  if (v) { send(v); $("#cmd").value = ""; }
};
$("#remember-form").onsubmit = (e) => {
  e.preventDefault();
  const name = $("#remember-name").value.trim();
  if (!name) return;
  send(lang === "kk" ? `${name} деп есте сақта` : `запомни как ${name}`);
  $("#remember-name").value = "";
  setTimeout(refreshState, 2500);
};
$("#pause").onclick = async () => {
  const paused = $("#pause").getAttribute("aria-pressed") !== "true";
  renderState(await post("/api/pause", { paused }));
};

// --- состояние --------------------------------------------------------------
function renderState(s) {
  $("#device").textContent = s.device === "cuda" ? "GPU · CUDA" : "CPU";
  const p = $("#pause");
  p.setAttribute("aria-pressed", String(s.paused));
  p.textContent = s.paused ? "Подсказки: пауза" : "Подсказки: вкл";

  const list = $("#faces");
  if (s.faces === null) {
    list.innerHTML = '<li class="empty">Распознавание лиц выключено</li>';
    $("#faces-count").textContent = "—";
    return;
  }
  $("#faces-count").textContent = `${s.faces.length} в памяти`;
  if (!s.faces.length) {
    list.innerHTML = '<li class="empty">Пока никого. Скажите «Айкору, запомни как …» или введите имя ниже</li>';
    return;
  }
  list.replaceChildren(...s.faces.map((name) => {
    const li = document.createElement("li");
    li.append(name);
    const del = document.createElement("button");
    del.textContent = "×";
    del.title = `Забыть ${name}`;
    del.setAttribute("aria-label", `Забыть ${name}`);
    del.onclick = async () => { await post("/api/faces/forget", { name }); refreshState(); };
    li.append(del);
    return li;
  }));
}
async function refreshState() {
  try { renderState(await (await fetch("/api/state")).json()); } catch {}
}

// --- эфир -------------------------------------------------------------------
function addFeed(e) {
  const feed = $("#feed");
  feed.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = new Date(e.ts * 1000).toLocaleTimeString("ru-RU");
  const tag = document.createElement("span");
  tag.className = "tag";
  const text = document.createElement("span");
  text.className = "text";
  text.textContent = e.text;
  if (e.kind === "command") {
    li.className = "command";
    tag.textContent = `${e.lang} · ${INTENT[e.intent] || e.intent}`;
    if (["remember", "forget"].includes(e.intent)) setTimeout(refreshState, 3000);
  } else {
    li.className = ["danger", "answer", "info"][e.priority] || "info";
    tag.textContent = ["опасность", "ответ", "событие"][e.priority] || "";
  }
  li.append(time, tag, text);
  feed.prepend(li);
  while (feed.children.length > 60) feed.lastChild.remove();
}

// --- треки и рамки ----------------------------------------------------------
function renderTracks() {
  $("#track-count").textContent = `${tracks.length} в кадре`;
  const ul = $("#tracks");
  if (!tracks.length) { ul.innerHTML = '<li class="empty">В кадре пусто</li>'; }
  else ul.replaceChildren(...tracks.map((t) => {
    const li = document.createElement("li");
    if (t.danger) li.className = "danger"; else if (t.identity) li.className = "known";
    const b = document.createElement("b");
    b.textContent = t.label;
    li.append(`#${t.id} `, b, ` · ${t.where}`);
    return li;
  }));
  drawOverlay();
}

function drawOverlay() {
  const img = $("#stream"), cv = $("#overlay");
  const W = cv.clientWidth, H = cv.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const iw = img.naturalWidth, ih = img.naturalHeight;
  if (!iw || !ih) return;
  // object-fit: contain
  const k = Math.min(W / iw, H / ih), ox = (W - iw * k) / 2, oy = (H - ih * k) / 2;
  ctx.font = "600 13px ui-monospace, Consolas, monospace";
  ctx.textBaseline = "top";
  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.box;
    const x = ox + x1 * k, y = oy + y1 * k, w = (x2 - x1) * k, h = (y2 - y1) * k;
    const color = t.danger ? "#d7263d" : t.identity ? "#ffffff" : "#f2e94e";
    ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.strokeRect(x, y, w, h);
    // пиксельные уголки
    ctx.fillStyle = color;
    for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) ctx.fillRect(cx - 4, cy - 4, 8, 8);
    const label = `#${t.id} ${t.label}`;
    const tw = ctx.measureText(label).width + 10;
    const ly = y - 20 < 0 ? y + 4 : y - 20;
    ctx.fillRect(x, ly, tw, 18);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 5, ly + 3);
  }
}
window.addEventListener("resize", drawOverlay);

// --- видео: подгружаем кадры по одному, следующий — после загрузки предыдущего ---
(function video() {
  const img = $("#stream"), empty = $("#video-empty");
  let fails = 0;
  const next = (delay) => setTimeout(() => { img.src = "frame.jpg?" + Date.now(); }, delay);
  img.onload = () => { fails = 0; empty.hidden = true; drawOverlay(); next(100); };
  img.onerror = () => { if (++fails > 2) empty.hidden = false; next(1000); };
  next(0);
})();

// --- события (SSE) ------------------------------------------------------------
function connect() {
  const conn = $("#conn");
  const es = new EventSource("/events");
  es.onopen = () => { conn.className = "chip live"; conn.lastChild.textContent = "в эфире"; refreshState(); };
  es.onerror = () => { conn.className = "chip down"; conn.lastChild.textContent = "нет связи"; };
  es.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.kind === "tracks") { tracks = e.tracks; renderTracks(); }
    else addFeed(e);
  };
}

$("#feed").innerHTML = '<li class="empty">Пока тихо: AIKORU говорит только при изменениях и опасности</li>';
setLang(lang);
renderTracks();
refreshState();
connect();
