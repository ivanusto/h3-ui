#!/usr/bin/env python3
"""Local frontend for the MiniMax H3 video service.

Serves a browser UI and proxies generation requests to vLLM-Omni so the API key
never reaches the browser and long generations survive a dropped connection.
The supported task list is read from the mounted checkpoint's model_index.json,
so switching between the FL2VA and Ref2VA partitions changes the UI without a
code change. Standard library only.
"""

import base64
import binascii
import json
import mimetypes
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "media"

# Settings come from a .env file so the API key never lives in a shell history
# or a systemd unit. A local .env wins; otherwise fall back to the deployment
# repo's, which is where the key already lives on a Spark install.
DEFAULT_ENV_FILES = (ROOT / ".env", Path.home() / "MiniMax-H3-DGX-Spark" / ".env")
ENV_FILE = next(
    (path for path in
     ([Path(os.environ["H3_UI_ENV_FILE"])] if os.environ.get("H3_UI_ENV_FILE")
      else DEFAULT_ENV_FILES)
     if path.is_file()),
    DEFAULT_ENV_FILES[0],
)

REQUEST_TIMEOUT = 7200
MAX_BODY = 512 * 1024 * 1024

# Measured here: 768x448, 20 steps, 2.0 s took ~84 s on the Cache-DiT 0.10
# profile. Used only for the UI's rough time estimate.
REF_COST = 768 * 448 * 20 * 2.0
REF_SECONDS = 84.0

# torch seeds are unsigned 64-bit, but keeping this inside int32 avoids any
# rounding surprise in the browser (JS numbers lose integer precision above
# 2**53) and stays reproducible when pasted back into the seed box.
SEED_MAX = 2**31 - 1

TASK_LABELS = {
    "t2va": "文字 → 影片＋音訊",
    "fl2va": "首格圖片 → 影片＋音訊",
    "ref2va": "參考條件 → 影片＋音訊",
}


def load_env():
    values = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


ENV = load_env()


def setting(key, default):
    """Process environment first, then the .env file, then the default."""
    return os.environ.get(key) or ENV.get(key) or default


# Loopback by default: serving the UI on a LAN address also exposes whatever
# the browser can reach through it. Set H3_UI_HOST to bind wider on purpose.
UI_HOST = setting("H3_UI_HOST", "127.0.0.1")
UI_PORT = int(setting("H3_UI_PORT", "8080"))

API_BASE = setting("H3_API_BASE", "http://127.0.0.1:8000").rstrip("/")
if API_BASE.endswith("/v1"):
    API_BASE = API_BASE[: -len("/v1")]
API_KEY = ENV.get("H3_API_KEY", "")


def load_partition():
    """Read the served checkpoint's partition and task list."""
    model_dir = ENV.get("MINIMAX_H3_MODEL_DIR", "")
    index = Path(model_dir) / "model_index.json" if model_dir else None
    if not index or not index.is_file():
        return {"partition": "unknown", "tasks": ["t2va"]}
    meta = json.loads(index.read_text(encoding="utf-8")).get("_minimax_h3", {})
    return {"partition": meta.get("partition", "unknown"),
            "tasks": list(meta.get("tasks") or ["t2va"])}


PARTITION = load_partition()

JOBS = {}
JOBS_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()


def auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def split_data_url(value):
    """Return (mime, raw_bytes) from a data: URL."""
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", value or "", re.S)
    if not match:
        raise ValueError("附件必須是 base64 data URL")
    try:
        return match.group(1), base64.b64decode(match.group(2), validate=True)
    except binascii.Error as exc:
        raise ValueError(f"附件 base64 解碼失敗: {exc}") from exc


def encode_multipart(fields, files):
    """Build a multipart/form-data body.

    fields: {name: text}. files: [(name, filename, mime, bytes)].
    """
    boundary = "----h3ui" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    for name, filename, mime, blob in files:
        body += f"--{boundary}\r\n".encode()
        body += (f'Content-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\n').encode()
        body += f"Content-Type: {mime}\r\n\r\n".encode()
        body += blob + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def build_request(params, attachments):
    """Translate UI parameters into the vLLM-Omni video request."""
    extra = {"task": params["task"], "duration": params["duration"],
             "audio_flow_shift": params["audio_flow_shift"]}
    fields = {
        "prompt": params["prompt"],
        "num_inference_steps": params["steps"],
        "flow_shift": params["flow_shift"],
        "seed": params["seed"],
        "fps": params["fps"],
        "extra_params": json.dumps(extra),
    }
    # fl2va derives the canvas from the reference image when width/height are
    # omitted; the UI exposes that as "follow the image".
    if params.get("width") and params.get("height"):
        fields["width"] = params["width"]
        fields["height"] = params["height"]

    files = []
    image = attachments.get("image")
    audio = attachments.get("audio")
    videos = attachments.get("videos") or []

    if image:
        mime, blob = split_data_url(image)
        # input_reference is sniffed server-side into an image or video.
        files.append(("input_reference", "reference" +
                      (mimetypes.guess_extension(mime) or ".png"), mime, blob))
    if videos:
        name = "input_references" if len(videos) > 1 else "input_reference"
        for index, item in enumerate(videos):
            mime, blob = split_data_url(item)
            files.append((name, f"reference-{index}" +
                          (mimetypes.guess_extension(mime) or ".mp4"), mime, blob))
    if audio:
        mime, _ = split_data_url(audio)
        fields["audio_reference"] = json.dumps({"audio_url": audio})

    return fields, files


def run_job(job_id, params, attachments):
    def touch(**kw):
        with JOBS_LOCK:
            JOBS[job_id].update(kw)

    touch(state="queued")
    with GPU_LOCK:
        started = time.time()
        touch(state="running", started=started)
        try:
            fields, files = build_request(params, attachments)
        except ValueError as exc:
            touch(state="failed", error=str(exc), elapsed=0)
            return
        body, content_type = encode_multipart(fields, files)
        request = urllib.request.Request(
            f"{API_BASE}/v1/videos/sync", data=body,
            headers={"Content-Type": content_type, **auth_headers()},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = response.read()
                kind = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1200]
            touch(state="failed", error=f"HTTP {exc.code}: {detail}",
                  elapsed=time.time() - started)
            return
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
            touch(state="failed", error=f"{type(exc).__name__}: {exc}",
                  elapsed=time.time() - started)
            return

        elapsed = time.time() - started
        if "video" not in kind and not payload.startswith(b"\x00\x00\x00"):
            touch(state="failed", elapsed=elapsed,
                  error=f"non-video response ({kind}): "
                        f"{payload[:1200].decode('utf-8', 'replace')}")
            return

        MEDIA.mkdir(exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id[:8]}.mp4"
        (MEDIA / name).write_bytes(payload)
        (MEDIA / (name + ".json")).write_text(
            json.dumps({**params, "elapsed": elapsed, "file": name,
                        "attached": sorted(k for k, v in attachments.items() if v)},
                       indent=2, ensure_ascii=False))
        touch(state="done", elapsed=elapsed, file=name, size=len(payload))


def resolve_seed(value):
    """Return the seed to run with, drawing a fresh one when asked.

    A missing seed, null, or any negative number means "surprise me". The drawn
    value is stored in the job params and written to the sidecar JSON, so a
    result that came out well can always be reproduced by pasting its seed back.
    """
    if value is None or value == "":
        return random.randrange(SEED_MAX + 1)
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return random.randrange(SEED_MAX + 1)
    if seed < 0:
        return random.randrange(SEED_MAX + 1)
    return min(seed, SEED_MAX)


def estimate_seconds(params):
    width = params.get("width") or 1344
    height = params.get("height") or 768
    cost = width * height * params["steps"] * params["duration"]
    return round(REF_SECONDS * cost / REF_COST)


def service_status():
    request = urllib.request.Request(f"{API_BASE}/v1/models", headers=auth_headers())
    base = {"api_base": API_BASE, "partition": PARTITION["partition"],
            "tasks": PARTITION["tasks"], "labels": TASK_LABELS}
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.load(response)
        base["model"] = data["data"][0]["id"]
    except Exception as exc:  # noqa: BLE001
        return {**base, "online": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {**base, "online": True, "busy": GPU_LOCK.locked(),
            "profile": ENV.get("H3_CACHE_BACKEND", "none"),
            "attention": ENV.get("H3_DIFFUSION_ATTENTION_BACKEND", ""),
            "execution": ENV.get("H3_EXECUTION_MODE", "")}


def history():
    if not MEDIA.is_dir():
        return []
    items = []
    for meta in sorted(MEDIA.glob("*.mp4.json"), reverse=True)[:40]:
        try:
            items.append(json.loads(meta.read_text()))
        except json.JSONDecodeError:
            continue
    return items


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, content_type="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(200, json.dumps(service_status()))
        elif path.startswith("/api/job/"):
            with JOBS_LOCK:
                job = dict(JOBS.get(path.rsplit("/", 1)[-1], {}))
            if not job:
                self._send(404, json.dumps({"error": "unknown job"}))
                return
            if job.get("state") == "running":
                job["elapsed"] = time.time() - job["started"]
            self._send(200, json.dumps(job))
        elif path == "/api/history":
            self._send(200, json.dumps(history()))
        elif path.startswith("/media/"):
            self.serve_media(path[len("/media/"):])
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def serve_media(self, name):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            self._send(400, json.dumps({"error": "bad name"}))
            return
        target = MEDIA / name
        if not target.is_file():
            self._send(404, json.dumps({"error": "not found"}))
            return
        kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), kind, {"Accept-Ranges": "none"})

    def do_POST(self):
        if self.path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            self._send(413, json.dumps({"error": "附件過大（上限 512 MB）"}))
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "invalid JSON"}))
            return

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send(400, json.dumps({"error": "prompt 不可為空"}))
            return
        task = str(payload.get("task", "t2va"))
        if task not in PARTITION["tasks"]:
            self._send(400, json.dumps(
                {"error": f"目前 checkpoint（{PARTITION['partition']}）"
                          f"只支援 {PARTITION['tasks']}"}))
            return

        attachments = payload.get("attachments") or {}
        image = attachments.get("image")
        audio = attachments.get("audio")
        videos = attachments.get("videos") or []

        # Mirror the pipeline's own constraints so the UI fails fast instead of
        # spending a cold request on a rejected combination.
        problem = None
        if task == "t2va" and (image or audio or videos):
            problem = "t2va 不接受任何附件"
        elif task == "fl2va":
            if not image:
                problem = "fl2va 必須提供一張首格圖片"
            elif audio or videos:
                problem = "fl2va 只接受圖片，不接受音訊或影片"
        elif task == "ref2va":
            if videos and (image or audio):
                problem = "ref2va 的參考影片模式沿用影片原聲，不可再附圖片或音訊"
            elif not videos and not (image and audio):
                problem = "ref2va 需要「圖片＋音訊」成對，或一支以上參考影片"
        if problem:
            self._send(400, json.dumps({"error": problem}))
            return

        params = {
            "task": task,
            "prompt": prompt,
            "width": int(payload["width"]) if payload.get("width") else None,
            "height": int(payload["height"]) if payload.get("height") else None,
            "steps": int(payload.get("steps", 20)),
            "duration": float(payload.get("duration", 2.0)),
            "fps": int(payload.get("fps", 24)),
            "flow_shift": float(payload.get("flow_shift", 12)),
            "audio_flow_shift": float(payload.get("audio_flow_shift", 3.0)),
            "seed": resolve_seed(payload.get("seed")),
        }
        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {"id": job_id, "state": "queued", "params": params,
                            "estimate": estimate_seconds(params)}
        threading.Thread(target=run_job, args=(job_id, params, attachments),
                         daemon=True).start()
        self._send(200, json.dumps({"id": job_id}))


INDEX_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniMax H3</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --line: #262b36; --text: #e6e9ef;
    --muted: #8b94a7; --accent: #76b900; --danger: #e5534b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.55 ui-sans-serif, system-ui, "Noto Sans TC", sans-serif; }
  header { padding: 18px 24px; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  h1 { font-size: 17px; margin: 0; font-weight: 600; letter-spacing: .02em; }
  .pill { font-size: 12px; padding: 3px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted); }
  .pill.ok { color: var(--accent); border-color: #2f4a10; }
  .pill.bad { color: var(--danger); border-color: #4a2320; }
  main { display: grid; grid-template-columns: minmax(340px, 430px) 1fr;
    gap: 24px; padding: 24px; align-items: start; }
  @media (max-width: 940px) { main { grid-template-columns: 1fr; } }
  .panel { background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px; }
  label { display: block; font-size: 12px; color: var(--muted);
    margin: 14px 0 5px; letter-spacing: .03em; }
  label:first-child { margin-top: 0; }
  textarea, input, select { width: 100%; background: #0d0f14; color: var(--text);
    border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px;
    font: inherit; font-size: 14px; }
  textarea { min-height: 100px; resize: vertical; }
  input[type=file] { padding: 7px; font-size: 12px; }
  input[type=checkbox] { width: auto; margin-right: 7px; vertical-align: -2px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .seed { display: flex; gap: 6px; }
  .seed input { min-width: 0; }
  .seed input[readonly] { color: var(--muted); }
  .dice { width: auto; flex: none; margin: 0; padding: 0 11px; font-size: 15px;
    background: #0d0f14; color: var(--text); border: 1px solid var(--line); }
  .dice:hover:not(:disabled) { border-color: var(--accent); }
  .toggle { display: block; font-size: 12px; color: var(--muted);
    margin-top: 10px; cursor: pointer; letter-spacing: .03em; }
  button { width: 100%; margin-top: 18px; padding: 11px; font: inherit;
    font-weight: 600; background: var(--accent); color: #0b0d10; border: 0;
    border-radius: 8px; cursor: pointer; }
  button:disabled { background: #2a2f3a; color: var(--muted); cursor: not-allowed; }
  .hint { font-size: 12px; color: var(--muted); margin-top: 10px; }
  .status { font-size: 13px; padding: 12px 14px; border-radius: 8px;
    border: 1px solid var(--line); background: #0d0f14; margin-bottom: 16px; }
  .status.err { border-color: #4a2320; color: #f0a6a1; white-space: pre-wrap;
    font-family: ui-monospace, monospace; font-size: 12px; }
  video, .thumb { width: 100%; border-radius: 10px; background: #000; display: block; }
  .thumb { max-height: 150px; object-fit: contain; }
  .meta { font-size: 12px; color: var(--muted); margin-top: 10px;
    font-family: ui-monospace, monospace; }
  .hist { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 14px; margin-top: 14px; }
  .card { border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
    background: #0d0f14; cursor: pointer; }
  .card video { border-radius: 0; }
  .card div { padding: 8px 10px; font-size: 11px; color: var(--muted);
    font-family: ui-monospace, monospace; }
  h2 { font-size: 13px; color: var(--muted); font-weight: 600; margin: 0;
    letter-spacing: .06em; text-transform: uppercase; }
  a { color: var(--accent); }
  fieldset { border: 1px solid var(--line); border-radius: 10px; padding: 14px;
    margin: 16px 0 0; }
  legend { font-size: 12px; color: var(--muted); padding: 0 6px; }
</style>
</head>
<body>
<header>
  <h1>MiniMax H3</h1>
  <span class="pill" id="svc">連線中…</span>
  <span class="pill" id="part"></span>
  <span class="pill" id="prof"></span>
</header>
<main>
  <section class="panel">
    <label for="task">生成模式</label>
    <select id="task"></select>

    <label for="prompt">Prompt</label>
    <textarea id="prompt" placeholder="Macro soldering a PCB under warm bench light, soft room tone."></textarea>

    <fieldset id="att" style="display:none">
      <legend>參考輸入</legend>
      <div id="att-image" style="display:none">
        <label for="f-image">首格圖片 / 參考圖片</label>
        <input id="f-image" type="file" accept="image/*">
        <img id="pv-image" class="thumb" style="display:none; margin-top:10px">
      </div>
      <div id="att-audio" style="display:none">
        <label for="f-audio">參考音訊</label>
        <input id="f-audio" type="file" accept="audio/*">
      </div>
      <div id="att-video" style="display:none">
        <label for="f-video">參考影片（可多選，沿用其原聲）</label>
        <input id="f-video" type="file" accept="video/*" multiple>
      </div>
      <p class="hint" id="att-rule"></p>
    </fieldset>

    <label for="preset">解析度</label>
    <select id="preset">
      <option value="768x448">768 × 448 — 已驗證，快速</option>
      <option value="1344x768">1344 × 768 — 已驗證，品質</option>
      <option value="custom">自訂</option>
      <option value="auto">依參考圖片比例自動</option>
    </select>
    <div class="row" id="wh" style="margin-top:12px; display:none">
      <div><label for="width">寬</label><input id="width" type="number" value="768" step="32"></div>
      <div><label for="height">高</label><input id="height" type="number" value="448" step="32"></div>
    </div>

    <div class="row3">
      <div><label for="steps">Steps</label><input id="steps" type="number" value="20" min="1" max="200"></div>
      <div><label for="duration">秒數</label><input id="duration" type="number" value="2.0" step="0.5" min="0.5"></div>
      <div><label for="fps">FPS</label><input id="fps" type="number" value="24" readonly></div>
    </div>
    <div class="row3">
      <div><label for="flow">Flow shift</label><input id="flow" type="number" value="12" step="0.5"></div>
      <div><label for="aflow">Audio shift</label><input id="aflow" type="number" value="3.0" step="0.5"></div>
      <div><label for="seed">Seed</label>
        <div class="seed">
          <input id="seed" type="number" value="42" min="0" max="2147483647">
          <button type="button" class="dice" id="dice" title="換一個隨機 seed">🎲</button>
        </div>
      </div>
    </div>
    <label class="toggle"><input id="rand" type="checkbox">每次生成都用新的隨機 seed</label>

    <button id="go">生成</button>
    <p class="hint" id="est"></p>
  </section>

  <section>
    <div class="panel">
      <div class="status" id="stat">尚未送出請求。</div>
      <div id="out"></div>
    </div>
    <div class="panel" style="margin-top:24px">
      <h2>歷史紀錄</h2>
      <div class="hist" id="hist"></div>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const fmt = s => s < 90 ? s.toFixed(0) + " 秒"
  : Math.floor(s / 60) + " 分 " + (s % 60).toFixed(0) + " 秒";
const RULES = {
  t2va: "純文字生成，不接受任何附件。",
  fl2va: "必須提供一張圖片作為第一幀，不接受音訊或影片。",
  ref2va: "「圖片＋音訊」成對，或一支以上參考影片（影片模式沿用原聲，不可再附音訊）。"
};
let TASKS = ["t2va"];

function dims() {
  const p = $("preset").value;
  if (p === "auto") return [null, null];
  if (p === "custom") return [+$("width").value, +$("height").value];
  const [w, h] = p.split("x");
  return [+w, +h];
}
function estimate() {
  const [w, h] = dims();
  const cost = (w || 1344) * (h || 768) * (+$("steps").value) * (+$("duration").value);
  $("est").textContent = "預估耗時約 " + fmt(84 * cost / (768 * 448 * 20 * 2.0)) +
    "（依本機實測的 Cache-DiT profile 推算）";
}
function syncTask() {
  const t = $("task").value;
  $("att").style.display = t === "t2va" ? "none" : "block";
  $("att-image").style.display = (t === "fl2va" || t === "ref2va") ? "block" : "none";
  $("att-audio").style.display = t === "ref2va" ? "block" : "none";
  $("att-video").style.display = t === "ref2va" ? "block" : "none";
  $("att-rule").textContent = RULES[t] || "";
  estimate();
}
$("task").onchange = syncTask;
$("preset").onchange = () => {
  $("wh").style.display = $("preset").value === "custom" ? "grid" : "none";
  if ($("preset").value === "1344x768") { $("steps").value = 50; $("duration").value = 4.0; }
  if ($("preset").value === "768x448") { $("steps").value = 20; $("duration").value = 2.0; }
  estimate();
};
["steps", "duration", "width", "height"].forEach(id => $(id).oninput = estimate);

$("f-image").onchange = async () => {
  const f = $("f-image").files[0];
  if (!f) { $("pv-image").style.display = "none"; return; }
  $("pv-image").src = await toDataUrl(f);
  $("pv-image").style.display = "block";
};

const toDataUrl = file => new Promise((res, rej) => {
  const r = new FileReader();
  r.onload = () => res(r.result);
  r.onerror = rej;
  r.readAsDataURL(file);
});

async function poll() {
  try {
    const s = await (await fetch("/api/status")).json();
    $("part").textContent = "partition: " + s.partition;
    if (JSON.stringify(s.tasks) !== JSON.stringify(TASKS)) {
      TASKS = s.tasks;
      $("task").innerHTML = TASKS.map(t =>
        `<option value="${t}">${t} — ${(s.labels || {})[t] || ""}</option>`).join("");
      syncTask();
    }
    if (s.online) {
      $("svc").className = "pill ok";
      $("svc").textContent = s.busy ? "服務中 · 生成中" : "服務中 · 閒置";
      $("prof").textContent = s.attention + " / " + s.execution + " / cache: " + s.profile;
    } else {
      $("svc").className = "pill bad";
      $("svc").textContent = "無法連線";
      $("prof").textContent = s.detail || "";
    }
  } catch (e) { $("svc").className = "pill bad"; $("svc").textContent = "前端錯誤"; }
}
poll(); setInterval(poll, 5000);

async function loadHistory() {
  const items = await (await fetch("/api/history")).json();
  $("hist").innerHTML = items.map(i => `
    <div class="card" onclick="show('${i.file}')">
      <video src="/media/${i.file}" muted preload="metadata"></video>
      <div>${i.task || "t2va"} · ${i.width || "auto"}×${i.height || "auto"} ·
        ${i.steps} steps · seed ${i.seed}<br>${fmt(i.elapsed)}${
        (i.attached || []).length ? " · 附件: " + i.attached.join(",") : ""}</div>
    </div>`).join("") || '<p class="hint">尚無紀錄。</p>';
}
loadHistory();

function show(file) {
  $("out").innerHTML = `<video src="/media/${file}" controls autoplay></video>
    <p class="meta">${file} · <a href="/media/${file}" download>下載</a></p>`;
}

const SEED_MAX = 2147483647;
const rollSeed = () => { $("seed").value = Math.floor(Math.random() * (SEED_MAX + 1)); };

$("dice").onclick = rollSeed;
// Randomising keeps writing the drawn seed into the box rather than hiding it,
// so a good result stays reproducible: untick and the value is already there.
$("rand").onchange = () => {
  $("seed").readOnly = $("rand").checked;
  if ($("rand").checked) rollSeed();
};

$("go").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { $("stat").textContent = "請先輸入 prompt。"; return; }
  const [width, height] = dims();
  $("go").disabled = true;
  $("stat").className = "status";
  $("stat").textContent = "讀取附件…";
  $("out").innerHTML = "";

  const attachments = {};
  if ($("f-image").files[0] && $("att-image").style.display !== "none")
    attachments.image = await toDataUrl($("f-image").files[0]);
  if ($("f-audio").files[0] && $("att-audio").style.display !== "none")
    attachments.audio = await toDataUrl($("f-audio").files[0]);
  if ($("f-video").files.length && $("att-video").style.display !== "none")
    attachments.videos = await Promise.all([...$("f-video").files].map(toDataUrl));

  if ($("rand").checked) rollSeed();

  $("stat").textContent = "已送出，排隊中…";
  const res = await fetch("/api/generate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      task: $("task").value, prompt, width, height,
      steps: +$("steps").value, duration: +$("duration").value,
      fps: +$("fps").value, flow_shift: +$("flow").value,
      audio_flow_shift: +$("aflow").value, seed: +$("seed").value, attachments
    })
  });
  const {id, error} = await res.json();
  if (error) {
    $("stat").className = "status err"; $("stat").textContent = error;
    $("go").disabled = false; return;
  }

  const timer = setInterval(async () => {
    const j = await (await fetch("/api/job/" + id)).json();
    if (j.state === "queued") {
      $("stat").textContent = "排隊中（GPU 正在處理其他請求）…";
    } else if (j.state === "running") {
      $("stat").textContent = "生成中… 已經過 " + fmt(j.elapsed) +
        "，預估約 " + fmt(j.estimate);
    } else if (j.state === "done") {
      clearInterval(timer); $("go").disabled = false;
      $("stat").className = "status";
      $("stat").textContent = "完成，耗時 " + fmt(j.elapsed) +
        "（" + (j.size / 1048576).toFixed(2) + " MB）· seed " + j.params.seed;
      show(j.file); loadHistory();
    } else if (j.state === "failed") {
      clearInterval(timer); $("go").disabled = false;
      $("stat").className = "status err"; $("stat").textContent = j.error;
    }
  }, 2000);
};
</script>
</body>
</html>
"""


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    print(f"H3 UI  ->  http://{UI_HOST}:{UI_PORT}")
    print(f"upstream: {API_BASE}  auth: {'on' if API_KEY else 'off'}")
    print(f"partition: {PARTITION['partition']}  tasks: {PARTITION['tasks']}")
    ThreadingHTTPServer((UI_HOST, UI_PORT), Handler).serve_forever()
