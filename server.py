#!/usr/bin/env python3
# Copyright 2026 ivanusto
# SPDX-License-Identifier: Apache-2.0
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
import queue
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
    "en": {
        "t2va": "text → video + audio",
        "fl2va": "first frame → video + audio",
        "ref2va": "reference → video + audio",
    },
    "zh": {
        "t2va": "文字 → 影片＋音訊",
        "fl2va": "首格圖片 → 影片＋音訊",
        "ref2va": "參考條件 → 影片＋音訊",
    },
}

# Only Traditional Chinese gets the Chinese page; zh-CN and everything else
# fall through to English. Matched against X-Lang first (the page's own
# override) and then the browser's Accept-Language.
ZH_HANT = re.compile(r"\bzh[-_](hant|tw|hk|mo)", re.I)

MESSAGES = {
    "en": {
        "attachment_data_url": "Attachments must be base64 data URLs",
        "attachment_decode": "Attachment base64 decode failed: {error}",
        "body_too_large": "Attachments too large (512 MB limit)",
        "duration_number": "duration must be a number",
        "fl2va_image_only": "fl2va takes an image only, not audio or video",
        "fl2va_needs_image": "fl2va needs one first-frame image",
        "forget_state": "Still {state}; cannot remove it from the list",
        "cancel_state": "Already {state}; cannot cancel",
        "prompt_required": "prompt cannot be empty",
        "ref2va_pair_or_video":
            "ref2va needs an image + audio pair, or one or more reference videos",
        "ref2va_video_exclusive":
            "ref2va's reference-video mode keeps the video's own audio; "
            "don't attach an image or audio as well",
        "t2va_no_attachments": "t2va takes no attachments",
        "task_unsupported":
            "The loaded checkpoint ({partition}) only supports {tasks}",
    },
    "zh": {
        "attachment_data_url": "附件必須是 base64 data URL",
        "attachment_decode": "附件 base64 解碼失敗: {error}",
        "body_too_large": "附件過大（上限 512 MB）",
        "duration_number": "duration 必須是數字",
        "fl2va_image_only": "fl2va 只接受圖片，不接受音訊或影片",
        "fl2va_needs_image": "fl2va 必須提供一張首格圖片",
        "forget_state": "還在 {state}，無法從清單移除",
        "cancel_state": "已經在 {state}，無法取消",
        "prompt_required": "prompt 不可為空",
        "ref2va_pair_or_video": "ref2va 需要「圖片＋音訊」成對，或一支以上參考影片",
        "ref2va_video_exclusive": "ref2va 的參考影片模式沿用影片原聲，不可再附圖片或音訊",
        "t2va_no_attachments": "t2va 不接受任何附件",
        "task_unsupported": "目前 checkpoint（{partition}）只支援 {tasks}",
    },
}


def pick_lang(explicit, accept=""):
    """The page's own choice wins; otherwise sniff the browser's locales.

    `explicit` is X-Lang, which the page sends as a bare "en"/"zh" — it is the
    toggle's answer, not a locale, so it is matched before the regex that only
    Traditional Chinese locale tags satisfy.
    """
    if explicit in MESSAGES:
        return explicit
    return "zh" if ZH_HANT.search(f"{explicit or ''} {accept or ''}") else "en"


def t(lang, key, **kw):
    table = MESSAGES.get(lang) or MESSAGES["en"]
    return table[key].format(**kw)


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

# One worker draining a FIFO queue, rather than a thread per job contending on
# a lock: threading.Lock has no ordering guarantee, so queued work used to
# start in an arbitrary order. The GPU serialises the real work regardless —
# this makes the order you submitted the order you get.
JOB_QUEUE = queue.Queue()

# Finished jobs stay listed so the queue view keeps its history across a page
# reload, but not without bound.
MAX_FINISHED = 60
FINAL_STATES = ("done", "failed", "cancelled")


def auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def split_data_url(value, lang="en"):
    """Return (mime, raw_bytes) from a data: URL."""
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", value or "", re.S)
    if not match:
        raise ValueError(t(lang, "attachment_data_url"))
    try:
        return match.group(1), base64.b64decode(match.group(2), validate=True)
    except binascii.Error as exc:
        raise ValueError(t(lang, "attachment_decode", error=exc)) from exc


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


def build_request(params, attachments, lang="en"):
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
        mime, blob = split_data_url(image, lang)
        # input_reference is sniffed server-side into an image or video.
        files.append(("input_reference", "reference" +
                      (mimetypes.guess_extension(mime) or ".png"), mime, blob))
    if videos:
        name = "input_references" if len(videos) > 1 else "input_reference"
        for index, item in enumerate(videos):
            mime, blob = split_data_url(item, lang)
            files.append((name, f"reference-{index}" +
                          (mimetypes.guess_extension(mime) or ".mp4"), mime, blob))
    if audio:
        mime, _ = split_data_url(audio, lang)
        fields["audio_reference"] = json.dumps({"audio_url": audio})

    return fields, files


def run_job(job_id, params, attachments, lang="en"):
    """Run one generation to completion. Called only by the queue worker.

    `lang` is the submitting page's language: a job can fail long after the
    request that queued it, so the language travels with the job rather than
    being read off whichever request happens to collect the error.
    """
    def touch(**kw):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(kw)

    started = time.time()
    touch(state="running", started=started)
    try:
        fields, files = build_request(params, attachments, lang)
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


def worker_loop():
    """Drain the queue forever, one job at a time, in submission order."""
    while True:
        job_id, params, attachments, lang = JOB_QUEUE.get()
        try:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                skip = job is None or job.get("state") == "cancelled"
            if not skip:
                run_job(job_id, params, attachments, lang)
        except Exception as exc:  # noqa: BLE001 - a bad job must not end the worker
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id].update(
                        state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            JOB_QUEUE.task_done()
            prune_jobs()


def prune_jobs():
    """Drop the oldest finished jobs once the history grows past the cap."""
    with JOBS_LOCK:
        finished = [j for j in JOBS.values() if j.get("state") in FINAL_STATES]
        for job in sorted(finished, key=lambda j: j.get("created", 0))[:-MAX_FINISHED]:
            JOBS.pop(job["id"], None)


def job_list():
    """Every job the queue view needs, newest first, without the attachments."""
    now = time.time()
    with JOBS_LOCK:
        jobs = [dict(job) for job in JOBS.values()]
    for job in jobs:
        if job.get("state") == "running" and job.get("started"):
            job["elapsed"] = now - job["started"]
    jobs.sort(key=lambda j: j.get("created", 0), reverse=True)
    waiting = [j["id"] for j in sorted(
        (j for j in jobs if j.get("state") == "queued"),
        key=lambda j: j.get("created", 0))]
    for job in jobs:
        if job.get("state") == "queued":
            job["position"] = waiting.index(job["id"]) + 1
    return jobs


def forget_job(job_id):
    """Drop a finished job from the queue list. The video is untouched."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        if job.get("state") not in FINAL_STATES:
            return job.get("state")
        del JOBS[job_id]
        return "forgotten"


def forget_finished_jobs():
    """Clear every finished entry at once. Returns how many went."""
    with JOBS_LOCK:
        gone = [i for i, j in JOBS.items() if j.get("state") in FINAL_STATES]
        for job_id in gone:
            del JOBS[job_id]
    return len(gone)


def forget_jobs_for_file(name):
    """Drop queue entries pointing at a file that no longer exists."""
    with JOBS_LOCK:
        for job_id in [i for i, j in JOBS.items() if j.get("file") == name]:
            del JOBS[job_id]


def cancel_job(job_id):
    """Cancel a job that has not started. Returns the resulting state."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        if job.get("state") != "queued":
            # Running work is already on the GPU and the upstream call is
            # synchronous, so there is nothing safe to interrupt.
            return job.get("state")
        job["state"] = "cancelled"
        return "cancelled"


def queue_position(job_id):
    """1-based place in line among jobs still waiting, or None if not waiting."""
    with JOBS_LOCK:
        waiting = sorted(
            (j for j in JOBS.values() if j.get("state") == "queued"),
            key=lambda j: j.get("created", 0))
    for index, job in enumerate(waiting, 1):
        if job["id"] == job_id:
            return index
    return None


def compose_prompt(payload, task, duration):
    """Assemble H3's structured prompt, or pass a plain one straight through.

    H3 expects three named sections rather than free text — see the official
    prompt guide in MiniMax-AI/MiniMax-H3 under skills/h3-prompt-writing. A
    plain `prompt` still works and is sent unchanged; supplying `description`
    switches to the structured form.

    For FL2VA the guide also wants a leading line stating where each reference
    picture lands on the timeline. Its second timestamp is the video duration,
    which this process already knows, so it is filled in rather than left for
    the caller to keep in sync by hand.
    """
    description = str(payload.get("description") or "").strip()
    if not description:
        return str(payload.get("prompt", "")).strip()

    parts = []
    if task == "fl2va":
        # The guide prefers a single shot for FL2VA so the model interpolates
        # continuously, but honour a multi-shot description if one was written.
        shots = re.findall(r"\[Shot (\d+)\]", description)
        last_shot = shots[-1] if shots else "1"
        parts.append(
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
            f"target video; Picture 2 (from Shot {last_shot}) aligns with the "
            f"{duration:.2f}-second mark of the target video.")

    parts.append(f"integrated_multimodal_description: {description}")
    for key, label in (("soundscape", "overall_soundscape"),
                       ("music", "non_diegetic_music")):
        value = str(payload.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    return "\n\n".join(parts)


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


def service_status(lang="en"):
    request = urllib.request.Request(f"{API_BASE}/v1/models", headers=auth_headers())
    base = {"api_base": API_BASE, "partition": PARTITION["partition"],
            "tasks": PARTITION["tasks"],
            "labels": TASK_LABELS.get(lang, TASK_LABELS["en"])}
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.load(response)
        base["model"] = data["data"][0]["id"]
    except Exception as exc:  # noqa: BLE001
        return {**base, "online": False, "detail": f"{type(exc).__name__}: {exc}"}
    with JOBS_LOCK:
        busy = any(j.get("state") == "running" for j in JOBS.values())
        waiting = sum(1 for j in JOBS.values() if j.get("state") == "queued")
    return {**base, "online": True, "busy": busy, "waiting": waiting,
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

    @property
    def lang(self):
        """X-Lang is the page's explicit choice; Accept-Language is the default."""
        return pick_lang(self.headers.get("X-Lang"),
                         self.headers.get("Accept-Language"))

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
            self._send(200, json.dumps(service_status(self.lang)))
        elif path.startswith("/api/job/"):
            with JOBS_LOCK:
                job = dict(JOBS.get(path.rsplit("/", 1)[-1], {}))
            if not job:
                self._send(404, json.dumps({"error": "unknown job"}))
                return
            if job.get("state") == "running":
                job["elapsed"] = time.time() - job["started"]
            self._send(200, json.dumps(job))
        elif path == "/api/jobs":
            self._send(200, json.dumps(job_list()))
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

    def do_DELETE(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/history/"):
            self.delete_media(path[len("/api/history/"):])
        elif path == "/api/jobs/finished":
            self._send(200, json.dumps({"forgotten": forget_finished_jobs()}))
        elif path.startswith("/api/job/"):
            job_id = path[len("/api/job/"):]
            state = forget_job(job_id)
            if state is None:
                self._send(404, json.dumps({"error": "unknown job"}))
            elif state == "forgotten":
                self._send(200, json.dumps({"id": job_id, "state": state}))
            else:
                self._send(409, json.dumps(
                    {"error": t(self.lang, "forget_state", state=state),
                     "state": state}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def delete_media(self, name):
        """Delete one result and its sidecar. Unlinking, so no undo."""
        # Only ever a generated filename: no separators, and it must be the
        # .mp4 — the sidecar goes with it rather than being deletable alone.
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.mp4", name) or ".." in name:
            self._send(400, json.dumps({"error": "bad name"}))
            return
        root = MEDIA.resolve()
        target = (MEDIA / name).resolve()
        # Belt and braces: the pattern already excludes separators, but a
        # symlinked media dir could still land the resolved path elsewhere.
        if target.parent != root:
            self._send(400, json.dumps({"error": "bad name"}))
            return
        if not target.is_file():
            self._send(404, json.dumps({"error": "not found"}))
            return
        removed = []
        for path in (target, root / (name + ".json")):
            try:
                path.unlink()
                removed.append(path.name)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
                return
        # Otherwise the queue keeps listing a result whose file is gone, and
        # clicking it plays nothing.
        forget_jobs_for_file(name)
        self._send(200, json.dumps({"deleted": removed}))

    def do_POST(self):
        if self.path.startswith("/api/job/") and self.path.endswith("/cancel"):
            job_id = self.path[len("/api/job/"):-len("/cancel")]
            state = cancel_job(job_id)
            if state is None:
                self._send(404, json.dumps({"error": "unknown job"}))
            elif state == "cancelled":
                self._send(200, json.dumps({"id": job_id, "state": state}))
            else:
                self._send(409, json.dumps(
                    {"error": t(self.lang, "cancel_state", state=state),
                     "state": state}))
            return
        if self.path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"}))
            return
        lang = self.lang
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            self._send(413, json.dumps({"error": t(lang, "body_too_large")}))
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "invalid JSON"}))
            return

        task = str(payload.get("task", "t2va"))
        if task not in PARTITION["tasks"]:
            self._send(400, json.dumps(
                {"error": t(lang, "task_unsupported",
                            partition=PARTITION["partition"],
                            tasks=PARTITION["tasks"])}))
            return

        try:
            duration = float(payload.get("duration", 2.0))
        except (TypeError, ValueError):
            self._send(400, json.dumps({"error": t(lang, "duration_number")}))
            return

        prompt = compose_prompt(payload, task, duration)
        if not prompt:
            self._send(400, json.dumps({"error": t(lang, "prompt_required")}))
            return

        attachments = payload.get("attachments") or {}
        image = attachments.get("image")
        audio = attachments.get("audio")
        videos = attachments.get("videos") or []

        # Mirror the pipeline's own constraints so the UI fails fast instead of
        # spending a cold request on a rejected combination.
        problem = None
        if task == "t2va" and (image or audio or videos):
            problem = "t2va_no_attachments"
        elif task == "fl2va":
            if not image:
                problem = "fl2va_needs_image"
            elif audio or videos:
                problem = "fl2va_image_only"
        elif task == "ref2va":
            if videos and (image or audio):
                problem = "ref2va_video_exclusive"
            elif not videos and not (image and audio):
                problem = "ref2va_pair_or_video"
        if problem:
            self._send(400, json.dumps({"error": t(lang, problem)}))
            return

        params = {
            "task": task,
            "prompt": prompt,
            "width": int(payload["width"]) if payload.get("width") else None,
            "height": int(payload["height"]) if payload.get("height") else None,
            "steps": int(payload.get("steps", 20)),
            "duration": duration,
            "fps": int(payload.get("fps", 24)),
            "flow_shift": float(payload.get("flow_shift", 12)),
            "audio_flow_shift": float(payload.get("audio_flow_shift", 3.0)),
            "seed": resolve_seed(payload.get("seed")),
        }
        # Keep the sections as written when the structured form was used, so a
        # result can be reopened and edited rather than only re-run verbatim.
        for key in ("description", "soundscape", "music"):
            value = str(payload.get(key) or "").strip()
            if value:
                params[key] = value
        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {"id": job_id, "state": "queued", "params": params,
                            "estimate": estimate_seconds(params),
                            "created": time.time()}
        JOB_QUEUE.put((job_id, params, attachments, lang))
        self._send(200, json.dumps(
            {"id": job_id, "position": queue_position(job_id)}))


INDEX_HTML = r"""<!doctype html>
<html lang="en">
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
  .pill.link { margin-left: auto; cursor: pointer; text-decoration: none; }
  .pill.link:hover { color: var(--text); border-color: var(--accent); }
  main { display: grid; grid-template-columns: minmax(340px, 430px) 1fr;
    gap: 24px; padding: 24px; align-items: start; }
  /* Grid items default to min-width:auto, so one long unbroken line — a
     structured prompt, say — widens the track and scrolls the whole page
     sideways. Let them shrink and clip instead. */
  main > section { min-width: 0; }
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
  .q { display: flex; align-items: center; gap: 10px; padding: 9px 0;
    border-bottom: 1px solid var(--line); font-size: 13px; }
  .q:last-child { border-bottom: 0; }
  .q .txt { flex: 1; min-width: 0; }
  /* text-overflow only ellipsises a single line, so each row clips on its
     own rather than relying on a <br> inside one clipped box. */
  .q .line { display: block; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  .q .sub { font-size: 11px; color: var(--muted);
    font-family: ui-monospace, monospace; }
  .q .tag { font-size: 11px; padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted); flex: none; }
  .q .tag.run { color: var(--accent); border-color: #2f4a10; }
  .q .tag.bad { color: var(--danger); border-color: #4a2320; }
  .q .x { width: auto; flex: none; margin: 0; padding: 3px 9px; font-size: 12px;
    background: transparent; color: var(--muted); border: 1px solid var(--line); }
  .q .x:hover { color: var(--danger); border-color: #4a2320; }
  .q.click { cursor: pointer; }
  .note { text-transform: none; letter-spacing: 0; opacity: .72; font-weight: 400; }
  label .note::before { content: " · "; }
  textarea.short { min-height: 58px; }
  .tips { margin-top: 14px; border: 1px solid var(--line); border-radius: 8px;
    padding: 10px 12px; font-size: 12px; color: var(--muted); }
  .tips summary { cursor: pointer; letter-spacing: .03em; }
  .tips ul { margin: 10px 0 0; padding-left: 18px; }
  .tips li { margin-bottom: 6px; line-height: 1.5; }
  .tips code { font-family: ui-monospace, monospace; font-size: 11px;
    background: #0d0f14; padding: 1px 5px; border-radius: 4px; color: var(--text); }
  .tips p { margin: 10px 0 0; }
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
  /* Portrait clips are the reason for the height cap: at width:100% a 9:16
     video is taller than the viewport and pushes the history below the fold.
     Capping height and letting width follow keeps landscape unchanged. */
  #out video { max-height: 58vh; width: auto; max-width: 100%; margin: 0 auto; }
  .card video { max-height: 170px; object-fit: contain; }
  .meta { font-size: 12px; color: var(--muted); margin-top: 10px;
    font-family: ui-monospace, monospace; }
  .hist { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 14px; margin-top: 14px; }
  .card { border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
    background: #0d0f14; cursor: pointer; position: relative; }
  .card .del { position: absolute; top: 6px; right: 6px; width: auto; margin: 0;
    padding: 2px 7px; font-size: 12px; line-height: 1.4; border-radius: 6px;
    background: rgba(8,10,14,.78); color: var(--muted);
    border: 1px solid var(--line); opacity: 0; transition: opacity .12s; }
  .card:hover .del, .card .del:focus { opacity: 1; }
  .card .del:hover { color: var(--danger); border-color: #4a2320; }
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
  <span class="pill" id="svc" data-i18n="hdr.connecting"></span>
  <span class="pill" id="part"></span>
  <span class="pill" id="prof"></span>
  <a class="pill link" id="lang" href="#" data-i18n="lang.other"></a>
</header>
<main>
  <section class="panel">
    <label for="task" data-i18n="label.mode"></label>
    <select id="task"></select>

    <label class="toggle" style="margin:14px 0 0"><input id="structured" type="checkbox" checked>
      <span data-i18n="opt.structured"></span></label>

    <div id="plain-wrap">
      <label for="prompt">Prompt</label>
      <textarea id="prompt" placeholder="Macro soldering a PCB under warm bench light, soft room tone."></textarea>
    </div>

    <div id="struct-wrap">
      <label for="description">integrated_multimodal_description
        <span class="note" data-i18n="note.description"></span></label>
      <textarea id="description" placeholder="[Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as she places a fresh loaf on the wooden counter."></textarea>
      <p class="hint" id="align-hint" style="display:none"></p>

      <label for="soundscape">overall_soundscape
        <span class="note" data-i18n="note.soundscape"></span></label>
      <textarea id="soundscape" class="short" placeholder="Wooden shutters scrape open over a quiet street as trays clink softly inside."></textarea>

      <label for="music">non_diegetic_music
        <span class="note" data-i18n="note.music"></span></label>
      <textarea id="music" class="short" placeholder="A soft acoustic-guitar pattern at a moderate tempo."></textarea>

      <details class="tips">
        <summary data-i18n="tips.summary"></summary>
        <ul data-i18n-html="tips.list"></ul>
        <p data-i18n-html="tips.source"></p>
      </details>
    </div>

    <fieldset id="att" style="display:none">
      <legend data-i18n="att.legend"></legend>
      <div id="att-image" style="display:none">
        <label for="f-image" data-i18n="att.image"></label>
        <input id="f-image" type="file" accept="image/*">
        <img id="pv-image" class="thumb" style="display:none; margin-top:10px">
      </div>
      <div id="att-audio" style="display:none">
        <label for="f-audio" data-i18n="att.audio"></label>
        <input id="f-audio" type="file" accept="audio/*">
      </div>
      <div id="att-video" style="display:none">
        <label for="f-video" data-i18n="att.video"></label>
        <input id="f-video" type="file" accept="video/*" multiple>
      </div>
      <p class="hint" id="att-rule"></p>
    </fieldset>

    <label for="preset" data-i18n="label.resolution"></label>
    <select id="preset">
      <option value="768x448" data-i18n="preset.fast"></option>
      <option value="1344x768" data-i18n="preset.quality"></option>
      <option value="custom" data-i18n="preset.custom"></option>
      <option value="auto" data-i18n="preset.auto"></option>
    </select>
    <div class="row" id="wh" style="margin-top:12px; display:none">
      <div><label for="width" data-i18n="label.width"></label><input id="width" type="number" value="768" step="32"></div>
      <div><label for="height" data-i18n="label.height"></label><input id="height" type="number" value="448" step="32"></div>
    </div>

    <div class="row3">
      <div><label for="steps">Steps</label><input id="steps" type="number" value="20" min="1" max="200"></div>
      <div><label for="duration" data-i18n="label.duration"></label><input id="duration" type="number" value="2.0" step="0.5" min="0.5"></div>
      <div><label for="fps">FPS</label><input id="fps" type="number" value="24" readonly></div>
    </div>
    <div class="row3">
      <div><label for="flow">Flow shift</label><input id="flow" type="number" value="12" step="0.5"></div>
      <div><label for="aflow">Audio shift</label><input id="aflow" type="number" value="3.0" step="0.5"></div>
      <div><label for="seed">Seed</label>
        <div class="seed">
          <input id="seed" type="number" value="42" min="0" max="2147483647">
          <button type="button" class="dice" id="dice" data-i18n-title="dice.title">🎲</button>
        </div>
      </div>
    </div>
    <label class="toggle"><input id="rand" type="checkbox"><span data-i18n="opt.random"></span></label>

    <button id="go" data-i18n="btn.generate"></button>
    <p class="hint" id="est"></p>
  </section>

  <section>
    <div class="panel">
      <div class="status" id="stat" data-i18n="stat.idle"></div>
      <div id="out"></div>
    </div>
    <div class="panel" style="margin-top:24px">
      <h2 data-i18n="h2.queue"></h2>
      <div id="queue"></div>
    </div>
    <div class="panel" style="margin-top:24px">
      <h2 data-i18n="h2.history"></h2>
      <div class="hist" id="hist"></div>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);

// Two languages, one page. Traditional Chinese browsers land on Chinese and
// everyone else — including zh-CN — on English; the header link overrides that
// choice and remembers it, so a shared machine isn't stuck with one locale.
const STRINGS = {
  en: {
    "lang.other": "繁體中文",
    "hdr.connecting": "connecting…",
    "hdr.busy": "online · generating",
    "hdr.idle": "online · idle",
    "hdr.queue": " · queue {n}",
    "hdr.offline": "unreachable",
    "hdr.uierror": "frontend error",
    "label.mode": "Mode",
    "opt.structured": "Structured prompt (H3's own format)",
    "note.description": "Framing, action, camera, dialogue, on-scene sound",
    "note.soundscape": "Ambience and action sound — what the characters hear",
    "note.music": "Score only the audience hears; N/A for none",
    "tips.summary": "Format crib sheet",
    "tips.list":
      "<li><b>Shots</b>: open with <code>[Shot 1]</code> and state style and framing; later shots as <code>[Shot 2] At 00:03.500, the camera cuts to…</code> (no timestamp on the first)</li>" +
      "<li><b>Camera</b>: <code>Push In / Pull Out / Truck Left / Pan Right / Tilt Up / Arc Shot / Tracking Shot / Static Shot / POV</code>, optionally <code>with small amplitude</code> or <code>at slow speed</code></li>" +
      "<li><b>Dialogue</b>: <code>&lt;d&gt;[English] First batch of the morning.&lt;/d&gt;</code>, speaker tagged <code>(S1)</code></li>" +
      "<li><b>Style</b>: <code>Cinematic / live-action / 2D-animated / 3D CG / claymation / watercolor / vintage film</code></li>" +
      "<li><b>FL2VA</b> prefers a single shot so the model interpolates continuously; the alignment line is filled in from the duration</li>",
    "tips.source": 'Follows the official <a href="https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt" target="_blank" rel="noreferrer">h3-prompt-writing</a> guide.',
    "att.legend": "Reference input",
    "att.image": "First frame / reference image",
    "att.audio": "Reference audio",
    "att.video": "Reference video (multiple allowed; their own audio is kept)",
    "label.resolution": "Resolution",
    "preset.fast": "768 × 448 — verified, fast",
    "preset.quality": "1344 × 768 — verified, quality",
    "preset.custom": "Custom",
    "preset.auto": "Auto — follow the reference image",
    "label.width": "Width",
    "label.height": "Height",
    "label.duration": "Duration (s)",
    "dice.title": "Draw another random seed",
    "opt.random": "Draw a fresh seed for every generation",
    "btn.generate": "Generate",
    "stat.idle": "Nothing submitted yet.",
    "h2.queue": "Queue",
    "h2.history": "History",
    "rule.t2va": "Text only — no attachments.",
    "rule.fl2va": "Needs one image as the first frame; no audio or video.",
    "rule.ref2va": "An image + audio pair, or one or more reference videos (video mode keeps their own audio, so no separate audio).",
    "fmt.sec": "{n}s",
    "fmt.min": "{m}m {s}s",
    "est": "Roughly {t} (extrapolated from a measured run on this box's Cache-DiT profile)",
    "hist.empty": "Nothing yet.",
    "hist.attached": "attached: ",
    "queue.empty": "The queue is empty.",
    "del.title": "Delete this video and its parameter file",
    "del.confirm": "Delete {file}?\nThe video and its parameter file go together, and there is no undo.",
    "del.failed": "Delete failed",
    "link.download": "Download",
    "link.delete": "Delete",
    "state.queued": "queued",
    "state.running": "running",
    "state.done": "done",
    "state.failed": "failed",
    "state.cancelled": "cancelled",
    "job.done": "Done in {t} ({mb} MB) · seed {seed}",
    "job.progress": " · {done} of ~{est}",
    "job.position": " · #{n} in line",
    "job.noprompt": "(no prompt)",
    "btn.cancel": "Cancel",
    "forget.title": "Remove from the queue list (the video stays)",
    "forget.all": "Clear all finished ({n})",
    "need.description": "Fill in integrated_multimodal_description first.",
    "need.prompt": "Enter a prompt first.",
    "stat.reading": "Reading attachments…",
    "stat.sending": "Submitted, queueing…",
    "stat.queued": "Queued — {n} job(s) ahead of it.",
    "stat.next": "Queued — starting shortly.",
    "align.hint": "Prepended automatically: Picture 1 (from Shot 1) → 0.00 s; Picture 2 (from Shot {shot}) → {secs} s"
  },
  zh: {
    "lang.other": "English",
    "hdr.connecting": "連線中…",
    "hdr.busy": "服務中 · 生成中",
    "hdr.idle": "服務中 · 閒置",
    "hdr.queue": " · 佇列 {n}",
    "hdr.offline": "無法連線",
    "hdr.uierror": "前端錯誤",
    "label.mode": "生成模式",
    "opt.structured": "結構化 prompt（H3 官方格式）",
    "note.description": "畫面、動作、運鏡、對白、場景內聲音",
    "note.soundscape": "環境音與動作聲，角色聽得到的",
    "note.music": "配樂，只有觀眾聽得到；不要配樂就填 N/A",
    "tips.summary": "格式速查",
    "tips.list":
      "<li><b>鏡頭</b>：<code>[Shot 1]</code> 起頭並註明風格與構圖；後續鏡頭 <code>[Shot 2] At 00:03.500, the camera cuts to…</code>（首個鏡頭不加時間）</li>" +
      "<li><b>運鏡</b>：<code>Push In / Pull Out / Truck Left / Pan Right / Tilt Up / Arc Shot / Tracking Shot / Static Shot / POV</code>，可加 <code>with small amplitude</code>、<code>at slow speed</code></li>" +
      "<li><b>對白</b>：<code>&lt;d&gt;[English] First batch of the morning.&lt;/d&gt;</code>，說話者標 <code>(S1)</code></li>" +
      "<li><b>風格</b>：<code>Cinematic / live-action / 2D-animated / 3D CG / claymation / watercolor / vintage film</code></li>" +
      "<li><b>FL2VA</b> 偏好單一鏡頭，讓模型連續內插；對齊指令會依秒數自動帶入</li>",
    "tips.source": '依據官方 <a href="https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt" target="_blank" rel="noreferrer">h3-prompt-writing</a> 指引。',
    "att.legend": "參考輸入",
    "att.image": "首格圖片 / 參考圖片",
    "att.audio": "參考音訊",
    "att.video": "參考影片（可多選，沿用其原聲）",
    "label.resolution": "解析度",
    "preset.fast": "768 × 448 — 已驗證，快速",
    "preset.quality": "1344 × 768 — 已驗證，品質",
    "preset.custom": "自訂",
    "preset.auto": "依參考圖片比例自動",
    "label.width": "寬",
    "label.height": "高",
    "label.duration": "秒數",
    "dice.title": "換一個隨機 seed",
    "opt.random": "每次生成都用新的隨機 seed",
    "btn.generate": "生成",
    "stat.idle": "尚未送出請求。",
    "h2.queue": "佇列",
    "h2.history": "歷史紀錄",
    "rule.t2va": "純文字生成，不接受任何附件。",
    "rule.fl2va": "必須提供一張圖片作為第一幀，不接受音訊或影片。",
    "rule.ref2va": "「圖片＋音訊」成對，或一支以上參考影片（影片模式沿用原聲，不可再附音訊）。",
    "fmt.sec": "{n} 秒",
    "fmt.min": "{m} 分 {s} 秒",
    "est": "預估耗時約 {t}（依本機實測的 Cache-DiT profile 推算）",
    "hist.empty": "尚無紀錄。",
    "hist.attached": "附件: ",
    "queue.empty": "佇列是空的。",
    "del.title": "刪除這支影片與它的參數檔",
    "del.confirm": "刪除 {file} ？\n影片與參數檔會一起移除，無法復原。",
    "del.failed": "刪除失敗",
    "link.download": "下載",
    "link.delete": "刪除",
    "state.queued": "排隊中",
    "state.running": "生成中",
    "state.done": "完成",
    "state.failed": "失敗",
    "state.cancelled": "已取消",
    "job.done": "完成，耗時 {t}（{mb} MB）· seed {seed}",
    "job.progress": " · 已 {done} / 約 {est}",
    "job.position": " · 第 {n} 順位",
    "job.noprompt": "(無 prompt)",
    "btn.cancel": "取消",
    "forget.title": "從佇列清單移除（不影響影片檔）",
    "forget.all": "清除全部已完成（{n}）",
    "need.description": "請先填寫 integrated_multimodal_description。",
    "need.prompt": "請先輸入 prompt。",
    "stat.reading": "讀取附件…",
    "stat.sending": "已送出，排隊中…",
    "stat.queued": "已加入佇列，前面還有 {n} 個任務。",
    "stat.next": "已加入佇列，即將開始。",
    "align.hint": "會自動加在最前面：Picture 1 (from Shot 1) → 0.00 秒；Picture 2 (from Shot {shot}) → {secs} 秒"
  }
};

const detectLang = () =>
  (navigator.languages && navigator.languages.length
    ? navigator.languages : [navigator.language || "en"])
    .some(l => /^zh[-_](hant|tw|hk|mo)/i.test(l)) ? "zh" : "en";

let LANG = localStorage.getItem("h3-lang");
if (LANG !== "en" && LANG !== "zh") LANG = detectLang();

const tr = (key, vars) => (STRINGS[LANG][key] ?? STRINGS.en[key] ?? key)
  .replace(/\{(\w+)\}/g, (m, name) =>
    vars && name in vars ? vars[name] : m);

// The server localises its own errors and task labels, so it needs to know
// which way the page went — Accept-Language alone would ignore the toggle.
const api = (path, opts = {}) => fetch(path, {
  ...opts, headers: {...(opts.headers || {}), "X-Lang": LANG}
});

function applyI18n() {
  document.documentElement.lang = LANG === "zh" ? "zh-Hant" : "en";
  const set = (attr, apply) =>
    document.querySelectorAll("[" + attr + "]").forEach(
      el => apply(el, tr(el.getAttribute(attr))));
  set("data-i18n", (el, v) => el.textContent = v);
  set("data-i18n-html", (el, v) => el.innerHTML = v);
  set("data-i18n-title", (el, v) => el.title = v);
}
applyI18n();

$("lang").onclick = e => {
  e.preventDefault();
  localStorage.setItem("h3-lang", LANG === "zh" ? "en" : "zh");
  location.reload();
};

const fmt = s => s < 90
  ? tr("fmt.sec", {n: s.toFixed(0)})
  : tr("fmt.min", {m: Math.floor(s / 60), s: (s % 60).toFixed(0)});
const RULES = {
  t2va: "rule.t2va", fl2va: "rule.fl2va", ref2va: "rule.ref2va"
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
  $("est").textContent = tr("est",
    {t: fmt(84 * cost / (768 * 448 * 20 * 2.0))});
}
function syncTask() {
  const t = $("task").value;
  $("att").style.display = t === "t2va" ? "none" : "block";
  $("att-image").style.display = (t === "fl2va" || t === "ref2va") ? "block" : "none";
  $("att-audio").style.display = t === "ref2va" ? "block" : "none";
  $("att-video").style.display = t === "ref2va" ? "block" : "none";
  $("att-rule").textContent = RULES[t] ? tr(RULES[t]) : "";
  estimate();
  syncAlignHint();
}
$("task").onchange = syncTask;
$("preset").onchange = () => {
  $("wh").style.display = $("preset").value === "custom" ? "grid" : "none";
  if ($("preset").value === "1344x768") { $("steps").value = 50; $("duration").value = 4.0; }
  if ($("preset").value === "768x448") { $("steps").value = 20; $("duration").value = 2.0; }
  estimate();
};
["steps", "duration", "width", "height"].forEach(id => $(id).oninput = estimate);
$("duration").addEventListener("input", syncAlignHint);
syncStructured();

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
    const s = await (await api("/api/status")).json();
    $("part").textContent = "partition: " + s.partition;
    if (JSON.stringify(s.tasks) !== JSON.stringify(TASKS)) {
      TASKS = s.tasks;
      $("task").innerHTML = TASKS.map(t =>
        `<option value="${t}">${t} — ${(s.labels || {})[t] || ""}</option>`).join("");
      syncTask();
    }
    if (s.online) {
      $("svc").className = "pill ok";
      $("svc").textContent = tr(s.busy ? "hdr.busy" : "hdr.idle") +
        (s.waiting ? tr("hdr.queue", {n: s.waiting}) : "");
      $("prof").textContent = s.attention + " / " + s.execution + " / cache: " + s.profile;
    } else {
      $("svc").className = "pill bad";
      $("svc").textContent = tr("hdr.offline");
      $("prof").textContent = s.detail || "";
    }
  } catch (e) {
    $("svc").className = "pill bad"; $("svc").textContent = tr("hdr.uierror");
  }
}
poll(); setInterval(poll, 5000);

async function loadHistory() {
  const items = await (await api("/api/history")).json();
  $("hist").innerHTML = items.map(i => `
    <div class="card">
      <button class="del" title="${tr("del.title")}"
        onclick="delMedia('${i.file}')">✕</button>
      <video src="/media/${i.file}" muted preload="metadata"
        onclick="show('${i.file}')"></video>
      <div onclick="show('${i.file}')">${i.task || "t2va"} ·
        ${i.width || "auto"}×${i.height || "auto"} ·
        ${i.steps} steps · seed ${i.seed}<br>${fmt(i.elapsed)}${
        (i.attached || []).length
          ? " · " + tr("hist.attached") + i.attached.join(",") : ""}</div>
    </div>`).join("") || `<p class="hint">${tr("hist.empty")}</p>`;
}

// Deletion unlinks the file — there is no trash to recover it from, so the
// confirm carries the filename rather than a generic "are you sure".
let shownFile = null;
async function delMedia(file) {
  if (!confirm(tr("del.confirm", {file}))) return;
  const r = await api("/api/history/" + encodeURIComponent(file), {method: "DELETE"});
  if (!r.ok) {
    const {error} = await r.json().catch(() => ({error: tr("del.failed")}));
    $("stat").className = "status err"; $("stat").textContent = error;
    return;
  }
  if (shownFile === file) { $("out").innerHTML = ""; shownFile = null; }
  loadHistory();
}
loadHistory();

function show(file) {
  shownFile = file;
  $("out").innerHTML = `<video src="/media/${file}" controls autoplay></video>
    <p class="meta">${file} · <a href="/media/${file}" download>${tr("link.download")}</a> ·
      <a href="#" onclick="delMedia('${file}');return false">${tr("link.delete")}</a></p>`;
}

const esc = s => (s || "").replace(/[&<>"]/g, c =>
  ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

const STATE_TAG = {
  queued: ["", "state.queued"], running: ["run", "state.running"],
  done: ["", "state.done"], failed: ["bad", "state.failed"],
  cancelled: ["", "state.cancelled"]
};

// null until the first poll: without it, every job already finished before the
// page opened would be treated as newly done and yank the player around.
let seenDone = null;

async function loadQueue() {
  let jobs;
  try { jobs = await (await api("/api/jobs")).json(); } catch (e) { return; }

  const done = jobs.filter(j => j.state === "done");
  if (seenDone === null) {
    seenDone = new Set(done.map(j => j.id));
  } else {
    const fresh = done.filter(j => !seenDone.has(j.id));
    fresh.forEach(j => seenDone.add(j.id));
    if (fresh.length) {
      const j = fresh[0];   // newest first from the server
      $("stat").className = "status";
      $("stat").textContent = tr("job.done", {
        t: fmt(j.elapsed), mb: (j.size / 1048576).toFixed(2),
        seed: j.params.seed});
      show(j.file);
      loadHistory();
    }
  }

  const pending = jobs.filter(j => j.state === "queued" || j.state === "running");
  const shown = pending.length ? pending
    : jobs.slice(0, 3);   // nothing waiting: keep the last few for context
  $("queue").innerHTML = shown.map(j => {
    const [cls, key] = STATE_TAG[j.state] || ["", null];
    const text = key ? tr(key) : j.state;
    const p = j.params || {};
    let sub = `${p.task || "t2va"} · ${p.steps} steps · ${p.duration}s · seed ${p.seed}`;
    if (j.state === "running")
      sub += tr("job.progress",
                {done: fmt(j.elapsed || 0), est: fmt(j.estimate || 0)});
    else if (j.state === "queued") sub += tr("job.position", {n: j.position});
    else if (j.state === "done") sub += ` · ${fmt(j.elapsed || 0)}`;
    else if (j.state === "failed") sub = esc((j.error || "").split("\n")[0]).slice(0, 120);
    const done = j.state === "done";
    const final = done || j.state === "failed" || j.state === "cancelled";
    return `<div class="q${done ? " click" : ""}"${
        done ? ` onclick="show('${j.file}')"` : ""}>
      <span class="tag ${cls}">${text}</span>
      <span class="txt">
        <span class="line">${esc(p.description || p.prompt) || tr("job.noprompt")}</span>
        <span class="line sub">${sub}</span></span>
      ${j.state === "queued"
        ? `<button class="x" onclick="event.stopPropagation();cancelJob('${j.id}')">${
             tr("btn.cancel")}</button>`
        : ""}
      ${final
        ? `<button class="x" title="${tr("forget.title")}"
             onclick="event.stopPropagation();forgetJob('${j.id}')">✕</button>`
        : ""}
    </div>`;
  }).join("") || `<p class="hint">${tr("queue.empty")}</p>`;
  const finished = jobs.filter(j => ["done","failed","cancelled"].includes(j.state));
  if (finished.length > 1) {
    $("queue").innerHTML += `<p class="hint" style="margin-top:12px">
      <a href="#" onclick="forgetFinished();return false">${
        tr("forget.all", {n: finished.length})}</a></p>`;
  }
}

// Forgetting only drops the queue entry. The video stays on disk and in the
// history, where deleting is a separate, louder action.
async function forgetJob(id) {
  await api("/api/job/" + id, {method: "DELETE"});
  loadQueue();
}

async function forgetFinished() {
  await api("/api/jobs/finished", {method: "DELETE"});
  loadQueue();
}

async function cancelJob(id) {
  const r = await api("/api/job/" + id + "/cancel", {method: "POST"});
  if (!r.ok) {
    const {error} = await r.json();
    $("stat").className = "status err"; $("stat").textContent = error;
  }
  loadQueue();
}

loadQueue(); setInterval(loadQueue, 2000);

const SEED_MAX = 2147483647;
const rollSeed = () => { $("seed").value = Math.floor(Math.random() * (SEED_MAX + 1)); };

function syncStructured() {
  const on = $("structured").checked;
  $("struct-wrap").style.display = on ? "block" : "none";
  $("plain-wrap").style.display = on ? "none" : "block";
  syncAlignHint();
}

// FL2VA wants a leading line naming where each reference picture lands on the
// timeline. The server fills it in from the duration; showing it here means the
// number in the prompt can't silently disagree with the form.
function syncAlignHint() {
  const el = $("align-hint");
  if (!$("structured").checked || $("task").value !== "fl2va") {
    el.style.display = "none"; return;
  }
  const secs = (+$("duration").value || 0).toFixed(2);
  const shots = ($("description").value.match(/\[Shot (\d+)\]/g) || []);
  const last = shots.length ? shots[shots.length - 1].match(/\d+/)[0] : "1";
  el.style.display = "block";
  el.textContent = tr("align.hint", {shot: last, secs});
}

$("structured").onchange = syncStructured;
$("description").oninput = syncAlignHint;

$("dice").onclick = rollSeed;
// Randomising keeps writing the drawn seed into the box rather than hiding it,
// so a good result stays reproducible: untick and the value is already there.
$("rand").onchange = () => {
  $("seed").readOnly = $("rand").checked;
  if ($("rand").checked) rollSeed();
};

$("go").onclick = async () => {
  const structured = $("structured").checked;
  const text = structured
    ? {description: $("description").value.trim(),
       soundscape: $("soundscape").value.trim(),
       music: $("music").value.trim()}
    : {prompt: $("prompt").value.trim()};
  if (structured ? !text.description : !text.prompt) {
    $("stat").className = "status";
    $("stat").textContent = tr(structured ? "need.description" : "need.prompt");
    return;
  }
  const [width, height] = dims();
  $("go").disabled = true;
  $("stat").className = "status";
  $("stat").textContent = tr("stat.reading");

  const attachments = {};
  if ($("f-image").files[0] && $("att-image").style.display !== "none")
    attachments.image = await toDataUrl($("f-image").files[0]);
  if ($("f-audio").files[0] && $("att-audio").style.display !== "none")
    attachments.audio = await toDataUrl($("f-audio").files[0]);
  if ($("f-video").files.length && $("att-video").style.display !== "none")
    attachments.videos = await Promise.all([...$("f-video").files].map(toDataUrl));

  if ($("rand").checked) rollSeed();

  $("stat").textContent = tr("stat.sending");
  const res = await api("/api/generate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      task: $("task").value, ...text, width, height,
      steps: +$("steps").value, duration: +$("duration").value,
      fps: +$("fps").value, flow_shift: +$("flow").value,
      audio_flow_shift: +$("aflow").value, seed: +$("seed").value, attachments
    })
  });
  const {position, error} = await res.json();
  $("go").disabled = false;
  if (error) {
    $("stat").className = "status err"; $("stat").textContent = error;
    return;
  }
  $("stat").className = "status";
  $("stat").textContent = position > 1
    ? tr("stat.queued", {n: position - 1})
    : tr("stat.next");
  loadQueue();
};
</script>
</body>
</html>
"""


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    threading.Thread(target=worker_loop, daemon=True, name="JobQueue").start()
    print(f"H3 UI  ->  http://{UI_HOST}:{UI_PORT}")
    print(f"upstream: {API_BASE}  auth: {'on' if API_KEY else 'off'}")
    print(f"partition: {PARTITION['partition']}  tasks: {PARTITION['tasks']}")
    ThreadingHTTPServer((UI_HOST, UI_PORT), Handler).serve_forever()
