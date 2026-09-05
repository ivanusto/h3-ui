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

# Cost model for the UI's time estimate, in terms of X = megapixels * seconds
# of output:
#
#     t = FIXED + PER_MPXS*X + PER_MPXS_STEP*X*steps + PER_MPXS2_STEP*X^2*steps
#
# The first term is what runs once per request and does not scale at all. The
# second scales with output pixels but not with the step count: VAE decode and
# muxing. The third is the part of a denoise step that is linear in tokens, and
# the fourth is attention, which is quadratic in them. H3's token count is
# proportional to pixels/32^2 times frames/4, which is why X appears squared
# rather than pixels or duration separately.
#
# Defaults fit on 11 measured renders on one DGX Spark (GB10), online FP8,
# cuDNN attention, regional compile, no Cache-DiT, spanning 768x448 to
# 1344x768, 4.4 s to 12 s, and 4 to 50 steps. Worst leave-one-out error 8.4%,
# mean 3.8%. They are tied to that profile: a different accelerator, or turning
# Cache-DiT back on, needs a re-fit. See "Recalibrating" in the README.

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
        "fastvideo_no_reference": "This backend takes text only. Remove the attachment or send it to a vLLM-Omni box.",
        "attachment_decode": "Attachment base64 decode failed: {error}",
        "body_too_large": "Attachments too large (512 MB limit)",
        "duration_number": "duration must be a number",
        "duration_range": "duration must be between {lo} and {hi} seconds",
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
        "steps_number": "steps must be a whole number",
        "turbo_unavailable": "This server has no request-switchable adapter",
        "turbo_fused":
            "{adapter} is fused into this server's checkpoint, which refuses a "
            "request that also names a LoRA",
        "turbo_task": "{name} does not serve {task}; it serves {tasks}",
        "steps_pinned":
            "This server has {adapter} fused: num_inference_steps must be "
            "exactly {steps}",
        "shifts_locked":
            "flow_shift and audio_flow_shift are owned by the fused {adapter} "
            "schedule, so h3-ui does not send them",
        "t2va_no_attachments": "t2va takes no attachments",
        "task_adapter":
            "{adapter} is fused on this server, which only serves {tasks}",
        "task_unsupported":
            "The loaded checkpoint ({partition}) only supports {tasks}",
    },
    "zh": {
        "attachment_data_url": "附件必須是 base64 data URL",
        "fastvideo_no_reference": "這個後端只吃文字，請移除附件，或改送到 vLLM-Omni 的機器。",
        "attachment_decode": "附件 base64 解碼失敗: {error}",
        "body_too_large": "附件過大（上限 512 MB）",
        "duration_number": "duration 必須是數字",
        "duration_range": "duration 必須介於 {lo} 到 {hi} 秒",
        "fl2va_image_only": "fl2va 只接受圖片，不接受音訊或影片",
        "fl2va_needs_image": "fl2va 必須提供一張首格圖片",
        "forget_state": "還在 {state}，無法從清單移除",
        "cancel_state": "已經在 {state}，無法取消",
        "prompt_required": "prompt 不可為空",
        "ref2va_pair_or_video": "ref2va 需要「圖片＋音訊」成對，或一支以上參考影片",
        "ref2va_video_exclusive": "ref2va 的參考影片模式沿用影片原聲，不可再附圖片或音訊",
        "steps_number": "steps 必須是整數",
        "turbo_unavailable": "這台伺服器沒有可逐請求切換的 adapter",
        "turbo_fused": "這台伺服器已把 {adapter} 融進 checkpoint，會拒收同時指名 LoRA 的請求",
        "turbo_task": "{name} 不提供 {task}，只提供 {tasks}",
        "steps_pinned": "這台伺服器已融合 {adapter}，num_inference_steps 必須剛好是 {steps}",
        "shifts_locked": "flow_shift 與 audio_flow_shift 由融合的 {adapter} 排程決定，h3-ui 不送這兩個欄位",
        "t2va_no_attachments": "t2va 不接受任何附件",
        "task_adapter": "這台伺服器已融合 {adapter}，只提供 {tasks}",
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

def normalise_base(value):
    """One upstream URL, without a trailing slash or /v1."""
    base = value.strip().rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


# Several upstreams, so two GB10 boxes can each serve their own vLLM-Omni. A
# model cannot span them: the diffusion executor only ever spawns local
# processes, so there is no cross-machine tensor or sequence parallelism to be
# had. The win is one job per box at a time, not one job finishing twice as
# fast.
#
# H3_API_BASES is the one to set when this .env is shared with the deployment
# repo, whose scripts read H3_API_BASE as a single URL and build health check
# addresses out of it. A comma separated H3_API_BASE still works when nothing
# else reads the file.
API_BASES = [normalise_base(part)
             for part in setting("H3_API_BASES",
                                 setting("H3_API_BASE",
                                         "http://127.0.0.1:8000")).split(",")
             if part.strip()]
API_BASE = API_BASES[0]
API_KEY = ENV.get("H3_API_KEY", "")

# Which server speaks at each upstream. Two exist now and they are not the same
# contract, only the same shape: vLLM-Omni takes a multipart body and answers
# POST /v1/videos/sync with the MP4, and FastVideo's OpenAI compatible server
# takes JSON at the same path and answers the same way. The queue, the seeds
# and the sidecar do not care which is which, so the difference is confined to
# building the body and reading the reply.
#
# One value applies to every upstream; a comma separated list assigns them in
# the order H3_API_BASES lists them, so a box running each is expressible.
BACKEND_KINDS_VALID = ("vllm-omni", "fastvideo")


def backend_kinds():
    raw = [part.strip().lower() for part in
           setting("H3_BACKEND_KIND", "vllm-omni").split(",") if part.strip()]
    if len(raw) == 1:
        raw *= len(API_BASES)
    kinds = {}
    for index, base in enumerate(API_BASES):
        kind = raw[index] if index < len(raw) else "vllm-omni"
        kinds[base] = kind if kind in BACKEND_KINDS_VALID else "vllm-omni"
    return kinds


BACKEND_KIND = backend_kinds()

# The model alias FastVideo advertises. Its server rejects a request naming any
# other model, and the alias is an operator choice rather than a checkpoint
# property, so it has to be configurable.
FASTVIDEO_MODEL = setting("H3_FASTVIDEO_MODEL", "fasth3")

# Which request contract the upstream speaks. "current" is what vLLM-Omni has
# wanted since the 2026-08-22 nightly: t2va needs a named aspect ratio, and the
# clip has to be at least four seconds. "legacy" is the minimax-h3 image
# published on 2026-08-02, where the canvas alone was enough and nothing
# bounded the duration from below.
# Anything that is not the word "legacy" is treated as current, because that is
# the safe direction to guess in: sending a named ratio to the old image is
# ignored, while omitting one on a current server fails the request.
SERVER_CONTRACT = "legacy" if setting(
    "H3_SERVER_CONTRACT", "current").strip().lower() == "legacy" else "current"
DURATION_MIN, DURATION_MAX = (0.5, 60.0) if SERVER_CONTRACT == "legacy" else (4.0, 15.0)

# The named ratios t2va accepts. width and height still set the canvas; the
# ratio is a separate named field the server refuses to infer, so the nearest
# name is what it wants, not an exact match: the upstream recipe itself pairs
# 960x576 with 16:9.
NAMED_RATIOS = (("21:9", 21 / 9), ("16:9", 16 / 9), ("4:3", 4 / 3),
                ("1:1", 1.0), ("3:4", 3 / 4), ("9:16", 9 / 16))

# FastH3's five sigma points bound four denoiser evaluations. Only used as the
# default when H3_FASTH3 is a bare switch; an integer there wins.
FASTH3_STEPS = 4

# The fit found no constant term worth keeping: forcing it to zero was better
# held out than fitting it, so this ships as 0 and is here for a deployment
# whose own measurements disagree.
EST_FIXED = float(setting("H3_EST_FIXED_SECONDS", "0"))
EST_PER_MPXS = float(setting("H3_EST_PER_MPXS", "11.88"))
EST_PER_MPXS_STEP = float(setting("H3_EST_PER_MPXS_STEP", "6.05"))
EST_PER_MPXS2_STEP = float(setting("H3_EST_PER_MPXS2_STEP", "0.88"))


def request_lora():
    """A LoRA the server preloaded and each request may switch on.

    This is the other kind of adapter, and it is the opposite of a fused one.
    vLLM-Omni's --lora-path with --lora-backend peft keeps the adapter resident
    but inactive; a request carrying a `lora` field activates it, and one
    without renders on the base checkpoint. So it is a per-request choice
    rather than a property of the server, which is what makes a checkbox
    honest here and dishonest for FastH3.

    The adapter dictates its own sampling settings. The Turbo release wants
    five sigma points, which bound its four denoiser evaluations, and a video
    shift of 6 rather than the checkpoint's 12; sending anything else samples
    the student where it was never distilled.
    """
    path = setting("H3_REQUEST_LORA_PATH", "")
    if not path:
        return None
    tasks = [t.strip() for t in setting("H3_REQUEST_LORA_TASKS", "t2va,fl2va").split(",")
             if t.strip()]
    return {
        "name": setting("H3_REQUEST_LORA_NAME", "turbo"),
        "path": path,
        "scale": float(setting("H3_REQUEST_LORA_SCALE", "1.0")),
        "steps": int(setting("H3_REQUEST_LORA_STEPS", "5")),
        "flow_shift": float(setting("H3_REQUEST_LORA_FLOW_SHIFT", "6")),
        "audio_flow_shift": float(setting("H3_REQUEST_LORA_AUDIO_SHIFT", "3.0")),
        "tasks": tasks,
        "label": setting("H3_REQUEST_LORA_LABEL", "Turbo, 4 denoiser steps"),
    }


REQUEST_LORA = request_lora()


def named_ratio(width, height):
    """The named aspect ratio closest to a canvas."""
    if not width or not height:
        return "16:9"
    target = width / height
    return min(NAMED_RATIOS, key=lambda item: abs(item[1] - target))[0]


def fasth3_steps(value):
    """H3_FASTH3 read as a switch or as a step count.

    None when unset, 0 when explicitly off, otherwise the step count the fused
    adapter pins. The integer form is what saves this if FastVideo ships a two
    or eight step student later.
    """
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in ("0", "false", "off", "no"):
        return 0
    if text in ("1", "true", "on", "yes"):
        return FASTH3_STEPS
    try:
        return max(int(text), 0)
    except ValueError:
        return FASTH3_STEPS


def schedule_steps(meta):
    """The step count a distilled checkpoint pins, or None.

    Never a bare len(): a rectified flow schedule closes on a terminal 0, so
    the point count is one more than the number of denoiser evaluations.
    Anything unrecognised returns None rather than a guess.
    """
    explicit = meta.get("num_inference_steps")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    schedule = meta.get("base_schedule")
    if isinstance(schedule, dict):
        schedule = schedule.get("sigmas") or schedule.get("positions")
    if not isinstance(schedule, (list, tuple)) or len(schedule) < 2:
        return None
    try:
        tail = float(schedule[-1])
    except (TypeError, ValueError):
        return None
    return len(schedule) - 1 if tail == 0 else len(schedule)


def adapter_label(path):
    """Name the fused adapter by the variant directory it came from."""
    if not path:
        return "fasth3"
    variant = Path(path).parent.name
    return f"fasth3 ({variant})" if variant else "fasth3"


def load_partition():
    """The served checkpoint's partition, task list and pinned schedule.

    Two different things can pin the step count. A distilled checkpoint says so
    in its own metadata and the server pins the count from there. A FastH3
    adapter fused with --lora-path says nothing at all: /v1/models is unchanged
    and so is model_index.json, so that one can only come from configuration.
    """
    info = {"partition": "unknown", "tasks": ["t2va"],
            "pinned_steps": None, "locked_shifts": False, "adapter": None}
    model_dir = ENV.get("MINIMAX_H3_MODEL_DIR", "")
    index = Path(model_dir) / "model_index.json" if model_dir else None
    if index and index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8")).get("_minimax_h3", {})
        info["partition"] = meta.get("partition", "unknown")
        info["tasks"] = list(meta.get("tasks") or ["t2va"])
        pinned = schedule_steps(meta)
        if pinned:
            info.update(pinned_steps=pinned, locked_shifts=True,
                        adapter="a distilled checkpoint")
    # The adapter is fused over whatever the checkpoint says, so it wins. A
    # FastH3 box may not set MINIMAX_H3_MODEL_DIR at all, which is why this
    # sits outside the block above.
    lora_path = setting("H3_LORA_PATH", "")
    override = fasth3_steps(setting("H3_FASTH3", ""))
    fused = override if override is not None else (FASTH3_STEPS if lora_path else 0)
    if fused:
        info.update(pinned_steps=fused, locked_shifts=True, tasks=["t2va"],
                    adapter=adapter_label(lora_path))
    return info


PARTITION = load_partition()

JOBS = {}
JOBS_LOCK = threading.Lock()

# One worker per upstream draining a shared FIFO, rather than a thread per job
# contending on a lock: threading.Lock has no ordering guarantee, so queued
# work used to start in an arbitrary order. Each GPU serialises its own work,
# so this makes the order you submitted the order work starts, across however
# many boxes are configured.
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


def build_fastvideo_request(params, attachments, lang="en"):
    """Translate UI parameters into FastVideo's OpenAI compatible video request.

    Four differences from the vLLM-Omni body matter. The transport is JSON
    rather than multipart. Frames are named directly instead of being derived
    from a duration, because FastH3 only accepts counts on its causal VAE grid
    and rounding a duration into one behind the caller's back would silently
    change what was measured. Neither shift is ever sent: the released
    checkpoint owns its five point schedule, and a request naming a shift is
    refused rather than ignored.

    And num_inference_steps counts something else here. Both servers run the
    same distilled schedule, five sigma points bounding four denoiser
    evaluations, but vLLM-Omni names the four and FastVideo names the five. So
    the same render is steps=4 on one and steps=5 on the other, and neither is
    wrong. The number is passed through as given rather than translated, so a
    sidecar always records what was actually sent.

    Reference media would ride along as image_reference here. Nothing needs it
    yet, so an attachment is refused rather than dropped quietly.
    """
    if any(attachments.get(key) for key in ("image", "audio", "videos")):
        raise ValueError(t(lang, "fastvideo_no_reference"))
    body = {
        "model": FASTVIDEO_MODEL,
        "prompt": params["prompt"],
        "seed": params["seed"],
        "num_inference_steps": params["steps"],
        "guidance_scale": 1.0,
        "video_params": {"fps": params["fps"],
                         "num_frames": frame_count(params["duration"],
                                                   params["fps"])},
    }
    if params.get("width") and params.get("height"):
        body["size"] = f"{params['width']}x{params['height']}"
    return json.dumps(body).encode(), "application/json"


def frame_count(duration, fps):
    """Frames for a duration, snapped up to MiniMax H3's 17n+5 grid.

    Both servers land on that grid. vLLM-Omni snaps for you and reports the
    count it used; FastVideo rejects anything off grid. Snapping here means the
    number in the sidecar is the number that was rendered either way.
    """
    raw = max(int(round(float(duration) * float(fps))), 5)
    return raw + (5 - raw) % 17


def build_request(params, attachments, lang="en"):
    """Translate UI parameters into the vLLM-Omni video request."""
    extra = {"task": params["task"], "duration": params["duration"]}
    fields = {
        "prompt": params["prompt"],
        "num_inference_steps": params["steps"],
        "seed": params["seed"],
        "fps": params["fps"],
    }
    # An activated request LoRA brings its own schedule, so it decides the
    # step count and both shifts rather than the form.
    if params.get("lora") and REQUEST_LORA:
        fields["num_inference_steps"] = REQUEST_LORA["steps"]
        fields["flow_shift"] = REQUEST_LORA["flow_shift"]
        extra["audio_flow_shift"] = REQUEST_LORA["audio_flow_shift"]
        fields["lora"] = json.dumps({"name": REQUEST_LORA["name"],
                                     "path": REQUEST_LORA["path"],
                                     "scale": REQUEST_LORA["scale"]})
    # A fused schedule owns both shifts: measured against the 2026-08-31 nightly,
    # the server accepts a value only when it equals the checkpoint's own (12 and
    # 3 for FastH3) and answers "FastH3 requires flow_shift=12, got 8" otherwise.
    # Omitting them is always accepted, and it is the only option that cannot
    # disagree with a schedule this process never sees.
    elif not PARTITION["locked_shifts"]:
        fields["flow_shift"] = params["flow_shift"]
        extra["audio_flow_shift"] = params["audio_flow_shift"]
    # t2va on a current build wants a named ratio and refuses to infer one from
    # the canvas. fl2va takes its ratio from the input image and ref2va
    # defaults to 16:9, so neither is sent one.
    if SERVER_CONTRACT == "current" and params["task"] == "t2va":
        fields["aspect_ratio"] = named_ratio(params.get("width"),
                                             params.get("height"))
    fields["extra_params"] = json.dumps(extra)
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


def server_metrics(headers):
    """What the server measured about its own run, from the reply headers.

    Both servers send the same three: inference time, a per stage breakdown, and
    the peak memory the worker saw. FastVideo copied vLLM-Omni's contract here,
    header names included, which is what makes a stage by stage comparison
    across the two possible at all.

    The peak memory one is the only trustworthy memory figure on this hardware.
    Unified memory makes nvidia-smi report N/A, and the system used figure
    counts the page cache, so a number measured inside the worker is the only
    one that means what it says.

    An older build may send none of them. An empty result is that, not a
    failure, so it is recorded as absent rather than as zero.
    """
    found = {}
    for header, key in (("X-Inference-Time-S", "inference_seconds"),
                        ("X-Peak-Memory-MB", "peak_memory_mb")):
        raw = headers.get(header)
        if raw:
            try:
                found[key] = float(raw)
            except ValueError:
                found[key] = raw
    stages = headers.get("X-Stage-Durations")
    if stages:
        try:
            found["stages"] = json.loads(stages)
        except json.JSONDecodeError:
            found["stages"] = stages
    return found


def run_job(job_id, params, attachments, lang="en", api_base=None):
    """Run one generation to completion. Called only by a queue worker.

    `lang` is the submitting page's language: a job can fail long after the
    request that queued it, so the language travels with the job rather than
    being read off whichever request happens to collect the error.

    `api_base` is the upstream this worker owns. It is recorded on the job so
    the queue view can say which box a render is on.
    """
    api_base = api_base or API_BASE
    def touch(**kw):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(kw)

    started = time.time()
    kind = BACKEND_KIND.get(api_base, "vllm-omni")
    touch(state="running", started=started, backend=api_base, backend_kind=kind)
    try:
        if kind == "fastvideo":
            body, content_type = build_fastvideo_request(params, attachments, lang)
        else:
            fields, files = build_request(params, attachments, lang)
            body, content_type = encode_multipart(fields, files)
    except ValueError as exc:
        touch(state="failed", error=str(exc), elapsed=0)
        return
    request = urllib.request.Request(
        f"{api_base}/v1/videos/sync", data=body,
        headers={"Content-Type": content_type, **auth_headers()},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = response.read()
            content_kind = response.headers.get("Content-Type", "")
            metrics = server_metrics(response.headers)
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
    if "video" not in content_kind and not payload.startswith(b"\x00\x00\x00"):
        touch(state="failed", elapsed=elapsed,
              error=f"non-video response ({content_kind}): "
                    f"{payload[:1200].decode('utf-8', 'replace')}")
        return

    MEDIA.mkdir(exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id[:8]}.mp4"
    (MEDIA / name).write_bytes(payload)
    (MEDIA / (name + ".json")).write_text(
        json.dumps({**params, "elapsed": elapsed, "file": name,
                    "backend": api_base, "backend_kind": kind,
                    **({"server_metrics": metrics} if metrics else {}),
                    "attached": sorted(k for k, v in attachments.items() if v)},
                   indent=2, ensure_ascii=False))
    touch(state="done", elapsed=elapsed, file=name, size=len(payload))


def worker_loop(api_base):
    """Drain the shared queue forever, one job at a time on this upstream.

    One worker per upstream, all pulling from the same FIFO: whichever box
    frees up first takes the next job, and submission order is still the order
    work starts.
    """
    while True:
        # Do not draw work while this box is unreachable. Leaving the job in
        # the queue lets a healthy box take it, and if every box is down the
        # work waits rather than failing one request at a time.
        with STATUS_LOCK:
            usable = STATUS_CACHE[api_base]["online"] is True
        if not usable:
            time.sleep(PROBE_INTERVAL)
            continue
        try:
            job_id, params, attachments, lang = JOB_QUEUE.get(
                timeout=PROBE_INTERVAL)
        except queue.Empty:
            continue          # recheck this box's health, then wait again
        try:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                skip = job is None or job.get("state") == "cancelled"
            if not skip:
                run_job(job_id, params, attachments, lang, api_base)
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
    """Rough wall clock for one render. See the cost model above."""
    width = params.get("width") or 1344
    height = params.get("height") or 768
    steps = params["steps"]
    x = width * height / 1e6 * params["duration"]
    seconds = (EST_FIXED + EST_PER_MPXS * x + EST_PER_MPXS_STEP * x * steps
               + EST_PER_MPXS2_STEP * x * x * steps)
    return max(round(seconds), 1)


# The upstream probe runs on its own thread, never inside a request. A
# synchronous generation blocks vLLM-Omni's event loop for as long as the MP4
# takes to encode: measured here at 0.37 s for 768x448/2 s, 2.14 s for
# 1344x768/2 s and 11.65 s for 1344x768/4 s. Probing from the request path
# turned that into "unreachable" in the header, so a finished render looked
# like a dead server until the file landed.
PROBE_INTERVAL = 5.0
PROBE_TIMEOUT = 5

# One missed probe means the upstream is busy, not gone. Only a run of them
# says the service has actually died.
PROBE_FAILURES_BEFORE_OFFLINE = 3

STATUS_LOCK = threading.Lock()
STATUS_CACHE = {base: {"online": None, "model": None, "detail": None,
                       "checked": 0.0, "failures": 0}
                for base in API_BASES}


def probe_upstream(api_base):
    """Ask one upstream whether it is usable. Returns (ok, model, detail).

    /v1/models is the portable question and the only one vLLM-Omni answers
    usefully, but it only proves the HTTP layer is up. That distinction is not
    academic here: a vLLM-Omni whose result pump thread has died keeps
    answering both /health and /v1/models with a 200 while no generation ever
    returns again, which is the failure this frontend cannot see and cannot
    work around.

    FastVideo's /health is the better question, because it verifies the engine
    is open and every worker is alive and answers 503 when the pool is not
    usable. Where it is on offer, ask it first, so a dead engine reads as
    offline instead of as a box that is merely slow.
    """
    if BACKEND_KIND.get(api_base) == "fastvideo":
        request = urllib.request.Request(f"{api_base}/health",
                                         headers=auth_headers())
        try:
            with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as response:
                response.read()
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
            return False, None, f"{type(exc).__name__}: {exc}"
    request = urllib.request.Request(f"{api_base}/v1/models", headers=auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as response:
            data = json.load(response)
        return True, data["data"][0]["id"], None
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
        return False, None, f"{type(exc).__name__}: {exc}"


def status_prober_loop(api_base):
    """Keep one upstream's cached status fresh, off the request path."""
    entry = STATUS_CACHE[api_base]
    while True:
        ok, model, detail = probe_upstream(api_base)
        with STATUS_LOCK:
            entry["checked"] = time.time()
            if ok:
                entry.update(online=True, model=model, detail=None, failures=0)
            else:
                entry["detail"] = detail
                entry["failures"] += 1
                if entry["failures"] >= PROBE_FAILURES_BEFORE_OFFLINE:
                    entry["online"] = False
        time.sleep(PROBE_INTERVAL)


def backend_view(api_base, busy_bases):
    """One upstream as the queue view sees it."""
    with STATUS_LOCK:
        snapshot = dict(STATUS_CACHE[api_base])
    state = ("starting" if snapshot["online"] is None
             else "online" if snapshot["online"] else "offline")
    return {"base": api_base, "state": state,
            "busy": api_base in busy_bases,
            # A probe or two can be swallowed by a busy upstream. Say so
            # rather than presenting the reading as current.
            "stale": state == "online" and snapshot["failures"] > 0,
            "probe_age": (round(time.time() - snapshot["checked"], 1)
                          if snapshot["checked"] else None),
            "model": snapshot["model"],
            "detail": snapshot["detail"] or ""}


def service_status(lang="en"):
    """Answer from the cache. This function never touches the network."""
    base = {"api_bases": API_BASES, "partition": PARTITION["partition"],
            "tasks": PARTITION["tasks"],
            "pinned_steps": PARTITION["pinned_steps"],
            "locked_shifts": PARTITION["locked_shifts"],
            "adapter": PARTITION["adapter"],
            "contract": SERVER_CONTRACT,
            "request_lora": ({k: REQUEST_LORA[k] for k in
                              ("name", "label", "steps", "tasks")}
                             if REQUEST_LORA else None),
            "duration_min": DURATION_MIN, "duration_max": DURATION_MAX,
            "est": {"fixed": EST_FIXED, "per_mpxs": EST_PER_MPXS,
                    "per_mpxs_step": EST_PER_MPXS_STEP,
                    "per_mpxs2_step": EST_PER_MPXS2_STEP},
            "labels": TASK_LABELS.get(lang, TASK_LABELS["en"])}
    with JOBS_LOCK:
        busy_bases = {j.get("backend") for j in JOBS.values()
                      if j.get("state") == "running"}
        waiting = sum(1 for j in JOBS.values() if j.get("state") == "queued")
    backends = [backend_view(api_base, busy_bases) for api_base in API_BASES]
    live = [b for b in backends if b["state"] == "online"]

    common = {**base, "backends": backends, "waiting": waiting,
              "backend_count": len(backends),
              "busy_count": sum(1 for b in backends if b["busy"]),
              "offline_count": sum(1 for b in backends if b["state"] == "offline")}
    if not live:
        # Starting beats unreachable while any upstream is still unproven.
        starting = any(b["state"] == "starting" for b in backends)
        return {**common, "online": False, "starting": starting or None,
                "detail": next((b["detail"] for b in backends if b["detail"]), "")}
    return {**common, "online": True,
            "busy": bool(busy_bases),
            "stale": any(b["stale"] for b in live),
            "model": live[0]["model"],
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

    @staticmethod
    def parse_range(header, size):
        """(start, end) for a single `bytes=` range, or None for the whole file.

        Only the one-range form browsers send for media is handled; anything
        else falls back to the full body rather than erroring.
        """
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
        if not match or size == 0:
            return None
        first, last = match.group(1), match.group(2)
        if first:
            start = int(first)
            end = int(last) if last else size - 1
        elif last:                      # bytes=-500 is the final 500 bytes
            start, end = max(0, size - int(last)), size - 1
        else:
            return None
        if start >= size or start > end:
            return None
        return start, min(end, size - 1)

    def serve_media(self, name):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            self._send(400, json.dumps({"error": "bad name"}))
            return
        target = MEDIA / name
        if not target.is_file():
            self._send(404, json.dumps({"error": "not found"}))
            return
        kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
        size = target.stat().st_size
        span = self.parse_range(self.headers.get("Range"), size)
        start, end = span if span else (0, size - 1)
        length = end - start + 1

        self.send_response(206 if span else 200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if span:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with target.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Routine: the gallery re-renders and the browser drops the
            # transfers it no longer needs.
            pass

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
            # Blaming the checkpoint would be a lie when it is the fused
            # adapter that narrowed the task list.
            error = (t(lang, "task_adapter", adapter=PARTITION["adapter"],
                       tasks=PARTITION["tasks"])
                     if PARTITION["adapter"] else
                     t(lang, "task_unsupported",
                       partition=PARTITION["partition"],
                       tasks=PARTITION["tasks"]))
            self._send(400, json.dumps({"error": error}))
            return

        try:
            duration = float(payload.get("duration", 2.0))
        except (TypeError, ValueError):
            self._send(400, json.dumps({"error": t(lang, "duration_number")}))
            return
        if not DURATION_MIN <= duration <= DURATION_MAX:
            self._send(400, json.dumps(
                {"error": t(lang, "duration_range", lo=DURATION_MIN,
                            hi=DURATION_MAX)}))
            return

        turbo = bool(payload.get("turbo"))
        if turbo:
            if not REQUEST_LORA:
                self._send(400, json.dumps({"error": t(lang, "turbo_unavailable")}))
                return
            if PARTITION["adapter"]:
                # A fused checkpoint refuses a request that also names a LoRA,
                # so the two adapters are mutually exclusive by construction.
                self._send(400, json.dumps(
                    {"error": t(lang, "turbo_fused", adapter=PARTITION["adapter"])}))
                return
            if task not in REQUEST_LORA["tasks"]:
                self._send(400, json.dumps(
                    {"error": t(lang, "turbo_task", name=REQUEST_LORA["name"],
                                task=task, tasks=REQUEST_LORA["tasks"])}))
                return

        try:
            steps = int(payload.get("steps", 20))
        except (TypeError, ValueError):
            self._send(400, json.dumps({"error": t(lang, "steps_number")}))
            return
        # Reject rather than quietly force. A request that asked for 20 steps
        # and got 4 would leave a sidecar that says something the render never
        # did.
        pinned = PARTITION["pinned_steps"]
        if pinned and steps != pinned:
            self._send(400, json.dumps(
                {"error": t(lang, "steps_pinned",
                            adapter=PARTITION["adapter"], steps=pinned)}))
            return
        if PARTITION["locked_shifts"] and (
                "flow_shift" in payload or "audio_flow_shift" in payload):
            self._send(400, json.dumps(
                {"error": t(lang, "shifts_locked",
                            adapter=PARTITION["adapter"])}))
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

        if turbo:
            steps = REQUEST_LORA["steps"]

        params = {
            "task": task,
            "prompt": prompt,
            "width": int(payload["width"]) if payload.get("width") else None,
            "height": int(payload["height"]) if payload.get("height") else None,
            "steps": steps,
            "duration": duration,
            "fps": int(payload.get("fps", 24)),
            "seed": resolve_seed(payload.get("seed")),
        }
        # The sidecar is {**params, ...}, so recording the adapter and dropping
        # the shifts that were never sent costs one branch each.
        if turbo:
            params["lora"] = {"name": REQUEST_LORA["name"],
                              "scale": REQUEST_LORA["scale"],
                              "flow_shift": REQUEST_LORA["flow_shift"],
                              "audio_flow_shift": REQUEST_LORA["audio_flow_shift"]}
        elif PARTITION["locked_shifts"]:
            params["adapter"] = PARTITION["adapter"]
        else:
            params["flow_shift"] = float(payload.get("flow_shift", 12))
            params["audio_flow_shift"] = float(
                payload.get("audio_flow_shift", 3.0))
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
      <div><label for="steps">Steps <span class="note" id="steps-note"></span></label><input id="steps" type="number" value="20" min="1" max="200"></div>
      <div><label for="duration" data-i18n="label.duration"></label><input id="duration" type="number" value="2.0" step="0.5" min="0.5"></div>
      <div><label for="fps">FPS</label><input id="fps" type="number" value="24" readonly></div>
    </div>
    <div class="row3" id="shift-row">
      <div id="flow-wrap"><label for="flow">Flow shift</label><input id="flow" type="number" value="12" step="0.5"></div>
      <div id="aflow-wrap"><label for="aflow">Audio shift</label><input id="aflow" type="number" value="3.0" step="0.5"></div>
      <div><label for="seed">Seed</label>
        <div class="seed">
          <input id="seed" type="number" value="42" min="0" max="2147483647">
          <button type="button" class="dice" id="dice" data-i18n-title="dice.title">🎲</button>
        </div>
      </div>
    </div>
    <label class="toggle" id="turbo-wrap" style="display:none"><input id="turbo" type="checkbox"><span id="turbo-label"></span></label>
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
    "hdr.stale": " · reading is a moment old",
    "hdr.nodes": " · {busy}/{total} boxes busy",
    "hdr.nodeoff": " · {n} unreachable",
    "hdr.uierror": "frontend error",
    "hdr.adapter": " · {a}",
    "note.stepslocked": "fixed at {n} by {a}",
    "opt.turbo": "{label}",
    "note.turbo": "fixed at {n} by {a}",
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
    "est": "Roughly {t}, from a cost model fitted to timed renders on this box",
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
    "hdr.stale": " · 讀數稍舊",
    "hdr.nodes": " · {busy}/{total} 台忙碌",
    "hdr.nodeoff": " · {n} 台連不上",
    "hdr.uierror": "前端錯誤",
    "hdr.adapter": " · {a}",
    "note.stepslocked": "由 {a} 釘死在 {n} 步",
    "opt.turbo": "{label}",
    "note.turbo": "由 {a} 釘死在 {n} 步",
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
    "est": "大約 {t}，依本機實測算圖時間擬合出的成本模型推估",
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
// The cost model's coefficients arrive over /api/status. They used to be
// written out twice, once here and once in the server, which is exactly how
// the two drift apart.
let EST = {fixed: 0, per_mpxs: 11.88, per_mpxs_step: 6.05, per_mpxs2_step: 0.88};
let LORA = null, PINNED = 0, LOCKED = false, ADAPTER = null;
// What the steps box held before the adapter took it over, so unticking gives
// the number back rather than leaving the adapter's behind.
let STEPS_BEFORE = null;
let CONSTRAINTS = "";

function dims() {
  const p = $("preset").value;
  if (p === "auto") return [null, null];
  if (p === "custom") return [+$("width").value, +$("height").value];
  const [w, h] = p.split("x");
  return [+w, +h];
}
function estimate() {
  const [w, h] = dims();
  const steps = +$("steps").value;
  const x = (w || 1344) * (h || 768) / 1e6 * (+$("duration").value);
  const secs = EST.fixed + EST.per_mpxs * x + EST.per_mpxs_step * x * steps +
               EST.per_mpxs2_step * x * x * steps;
  $("est").textContent = tr("est", {t: fmt(Math.max(secs, 1))});
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

// What the upstream will and will not accept, applied from /api/status. The
// signature guard matters: poll() runs every five seconds and must not stomp a
// box the user is halfway through editing.
function applyConstraints(s) {
  const sig = JSON.stringify([s.pinned_steps, s.locked_shifts, s.adapter,
                              s.est, s.duration_min, s.duration_max,
                              s.request_lora]);
  if (sig === CONSTRAINTS) return;
  CONSTRAINTS = sig;
  if (s.est) EST = s.est;

  // A request-switchable adapter is offered rather than imposed: the checkbox
  // appears, and syncTurbo() applies its schedule only when it is ticked.
  LORA = s.request_lora || null;
  $("turbo-wrap").style.display = LORA ? "flex" : "none";
  if (LORA) $("turbo-label").textContent = tr("opt.turbo", {label: LORA.label});

  const pinned = s.pinned_steps || 0;
  PINNED = pinned; LOCKED = !!s.locked_shifts; ADAPTER = s.adapter;
  // readOnly, not disabled: a disabled field is not submitted.
  $("steps").readOnly = !!pinned;
  if (pinned) $("steps").value = pinned;
  $("steps-note").textContent = pinned
    ? tr("note.stepslocked", {n: pinned, a: s.adapter || "the checkpoint"}) : "";

  const locked = !!s.locked_shifts;
  $("flow-wrap").style.display = locked ? "none" : "block";
  $("aflow-wrap").style.display = locked ? "none" : "block";
  // Hiding two of three cells would leave the seed box alone in a three column
  // grid with two gaps.
  $("shift-row").style.gridTemplateColumns = locked ? "1fr" : "";

  if (s.duration_min != null) {
    $("duration").min = s.duration_min;
    if (+$("duration").value < s.duration_min) $("duration").value = s.duration_min;
  }
  if (s.duration_max != null) $("duration").max = s.duration_max;
  estimate();
  syncAlignHint();
  syncTurbo();
}

// The adapter brings its own schedule, so ticking it locks the same boxes a
// fused one does. Unticking restores whatever the server allows.
function syncTurbo() {
  if (!LORA) return;
  const on = $("turbo").checked;
  if (on && STEPS_BEFORE === null) STEPS_BEFORE = $("steps").value;
  $("steps").readOnly = on || !!PINNED;
  if (on) {
    $("steps").value = LORA.steps;
  } else {
    if (PINNED) $("steps").value = PINNED;
    else if (STEPS_BEFORE !== null) $("steps").value = STEPS_BEFORE;
    STEPS_BEFORE = null;
  }
  $("steps-note").textContent = on
    ? tr("note.turbo", {n: LORA.steps, a: LORA.name})
    : (PINNED ? tr("note.stepslocked", {n: PINNED, a: ADAPTER || "the checkpoint"}) : "");
  const hide = on || LOCKED;
  $("flow-wrap").style.display = hide ? "none" : "block";
  $("aflow-wrap").style.display = hide ? "none" : "block";
  $("shift-row").style.gridTemplateColumns = hide ? "1fr" : "";
  estimate();
}

function clampDuration() {
  const lo = +$("duration").min || 0;
  if (+$("duration").value < lo) $("duration").value = lo;
}

$("preset").onchange = () => {
  $("wh").style.display = $("preset").value === "custom" ? "grid" : "none";
  // The presets write a step count, which would silently unpin a fused
  // schedule.
  if (!$("steps").readOnly) {
    if ($("preset").value === "1344x768") $("steps").value = 50;
    if ($("preset").value === "768x448") $("steps").value = 20;
  }
  if ($("preset").value === "1344x768") $("duration").value = 4.0;
  if ($("preset").value === "768x448") $("duration").value = 2.0;
  clampDuration();
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
    applyConstraints(s);
    if (s.online) {
      // With more than one box the header has to say how many are working,
      // and call out any that dropped out rather than hiding it behind an
      // aggregate "online".
      const many = (s.backend_count || 1) > 1;
      $("svc").className = "pill ok";
      $("svc").textContent = tr(s.busy ? "hdr.busy" : "hdr.idle") +
        (s.waiting ? tr("hdr.queue", {n: s.waiting}) : "") +
        (many ? tr("hdr.nodes", {busy: s.busy_count, total: s.backend_count}) : "") +
        (s.offline_count ? tr("hdr.nodeoff", {n: s.offline_count}) : "") +
        (s.adapter ? tr("hdr.adapter", {a: s.adapter}) : "") +
        (s.stale ? tr("hdr.stale") : "");
      $("prof").textContent = s.attention + " / " + s.execution + " / cache: " + s.profile;
    } else if (s.starting) {
      // No probe has completed yet. Not the same as unreachable.
      $("svc").className = "pill";
      $("svc").textContent = tr("hdr.connecting");
      $("prof").textContent = s.detail || "";
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

$("turbo").onchange = syncTurbo;
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

  // A locked schedule rejects a request that names either shift, so the page
  // must not send what it is not showing.
  const shifts = $("flow-wrap").style.display === "none" ? {} :
    {flow_shift: +$("flow").value, audio_flow_shift: +$("aflow").value};
  const turbo = !!(LORA && $("turbo").checked);

  $("stat").textContent = tr("stat.sending");
  const res = await api("/api/generate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      task: $("task").value, ...text, width, height,
      steps: +$("steps").value, duration: +$("duration").value,
      fps: +$("fps").value, ...shifts, turbo, seed: +$("seed").value, attachments
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
    for index, upstream in enumerate(API_BASES):
        threading.Thread(target=worker_loop, args=(upstream,), daemon=True,
                         name=f"JobQueue-{index}").start()
        threading.Thread(target=status_prober_loop, args=(upstream,),
                         daemon=True, name=f"StatusProbe-{index}").start()
    print(f"H3 UI  ->  http://{UI_HOST}:{UI_PORT}")
    print(f"upstream: {', '.join(API_BASES)}  auth: {'on' if API_KEY else 'off'}")
    print(f"partition: {PARTITION['partition']}  tasks: {PARTITION['tasks']}"
          f"  contract: {SERVER_CONTRACT}")
    if PARTITION["adapter"]:
        print(f"adapter: {PARTITION['adapter']}  "
              f"steps pinned at {PARTITION['pinned_steps']}, shifts locked")
        if len(API_BASES) > 1:
            # Nothing on the wire advertises the fusion, so this cannot be
            # checked per box. A box started without --lora-path would accept
            # a four step request and return mush.
            print("warning: every upstream must have been started with "
                  "--lora-path; a mixed pair fails silently")
    ThreadingHTTPServer((UI_HOST, UI_PORT), Handler).serve_forever()
