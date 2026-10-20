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
import collections
import hashlib
import http.client
import json
import mimetypes
import os
import random
import re
import shutil
import sys
import threading
import traceback
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from email.utils import formatdate
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
        "attachment_unknown": "Unknown attachment field: {name}",
        "attachment_shape": "attachments must be an object, and videos a list",
        "body_json_object": "The request body must be a JSON object",
        "body_length": "Content-Length must be a whole number of bytes",
        "content_type": "Send the request body as application/json",
        "cross_origin": "Cross-origin requests are refused",
        "param_positive": "{name} must be a positive whole number",
        "no_upstream":
            "No configured upstream can serve this job: {reason}",
        "shift_number": "flow_shift and audio_flow_shift must be numbers",
        "reason_fastvideo":
            "FastVideo takes t2va only, without the request LoRA, between "
            "{lo} and {hi} seconds",
        "feature_contract":
            "{feature} needs a vLLM-Omni server on the current contract",
        "feature_off": "{feature} is not enabled on this frontend ({key})",
        "long_needed": "Above 15 seconds, tick long video",
        "long_mode": "Continuation is Ref2VA only; other modes render long video in full",
        "long_window":
            "Window and overlap must be on the 17n+5 grid, the window {lo} to "
            "{hi} frames and larger than the overlap",
        "long_prompts":
            "This clip runs {n} continuation windows, so give {n} prompts or none",
        "lock_source": "Locking the soundtrack needs ref2va with one reference audio",
        "edit_canvas": "Editing needs an explicit width and height, not auto",
        "edit_number": "Edit strengths and times must be numbers, strengths 0 to 1",
        "edit_noop": "This edit regenerates everything; nothing of the source is kept",
        "edit_source_video": "Keeping any of the picture needs a source video",
        "edit_source_audio": "Keeping any of the sound needs a source audio or a source video",
        "edit_sources_only": "A source clip only goes with an edit",
        "guide_shape": "Each guide needs a position and an image, a clip or audio",
        "guide_count": "At most {n} timeline guides",
        "guide_kind": "Guide {n}: the picture must be an image or a clip, the sound audio",
        "guide_size": "Guide {n}: {kind} over the {mb} MB limit",
        "guide_frame": "Guide {n} at {secs} s falls outside a {frames} frame clip",
        "upscale_number": "Upscale must be {lo} to {hi}x and refine strength above 0, at most 1",
        "conflict": "{a} cannot be combined with {b}",
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
        "attachment_unknown": "不認得的附件欄位：{name}",
        "attachment_shape": "attachments 必須是物件，videos 必須是陣列",
        "body_json_object": "請求內容必須是 JSON 物件",
        "body_length": "Content-Length 必須是整數位元組數",
        "content_type": "請以 application/json 送出請求內容",
        "cross_origin": "拒絕跨來源請求",
        "param_positive": "{name} 必須是正整數",
        "no_upstream": "目前設定的上游都無法執行這個工作：{reason}",
        "shift_number": "flow_shift 與 audio_flow_shift 必須是數字",
        "reason_fastvideo": "FastVideo 只接 t2va、不能搭配請求 LoRA，秒數需介於 {lo} 到 {hi} 秒",
        "feature_contract": "{feature} 需要 contract 為 current 的 vLLM-Omni 伺服器",
        "feature_off": "這個前端沒有開啟 {feature}（{key}）",
        "long_needed": "超過 15 秒請勾選長片",
        "long_mode": "續寫只限 ref2va，其他模式的長片以整段方式生成",
        "long_window": "視窗與重疊都必須落在 17n+5 的格點上，視窗介於 {lo} 到 {hi} 格，且大於重疊",
        "long_prompts": "這支片會切成 {n} 個續寫視窗，prompt 請給 {n} 段，或一段都不給",
        "lock_source": "鎖定音軌需要 ref2va 並附一段參考音訊",
        "edit_canvas": "編輯需要明確的寬與高，不能用自動",
        "edit_number": "編輯的強度與時間必須是數字，強度介於 0 到 1",
        "edit_noop": "這個編輯會全部重新生成，來源一點都沒有保留",
        "edit_source_video": "要保留任何畫面，必須附上來源影片",
        "edit_source_audio": "要保留任何聲音，必須附上來源音訊或來源影片",
        "edit_sources_only": "來源片段只能搭配編輯使用",
        "guide_shape": "每個 guide 都需要位置，以及一張圖片、一段短片或一段音訊",
        "guide_count": "timeline guide 最多 {n} 個",
        "guide_kind": "第 {n} 個 guide：畫面必須是圖片或短片，聲音必須是音訊",
        "guide_size": "第 {n} 個 guide：{kind} 超過 {mb} MB 上限",
        "guide_frame": "第 {n} 個 guide 位於 {secs} 秒，超出 {frames} 格的片長",
        "upscale_number": "放大倍率需介於 {lo} 到 {hi} 倍，精修強度需大於 0 且不超過 1",
        "conflict": "{a} 不能與 {b} 同時使用",
    },
}


FEATURE_NAMES = {
    "en": {"turbo": "Turbo", "adapter": "a distilled or fused schedule",
           "continuation": "long video continuation", "long": "long video",
           "edit": "latent edit", "guides": "timeline guides",
           "upscale": "latent upscale", "refine": "latent refine",
           "lock": "a locked soundtrack", "fastvideo": "FastVideo"},
    "zh": {"turbo": "Turbo", "adapter": "蒸餾或融合的排程",
           "continuation": "長片續寫", "long": "長片",
           "edit": "遮罩編輯", "guides": "timeline guides",
           "upscale": "latent 放大", "refine": "latent 精修",
           "lock": "鎖定音軌", "fastvideo": "FastVideo"},
}


def feature(lang, key):
    return (FEATURE_NAMES.get(lang) or FEATURE_NAMES["en"])[key]


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
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
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
API_KEY = setting("H3_API_KEY", "")

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
KINDS_PRESENT = set(BACKEND_KIND.values())

# The model alias FastVideo advertises. Its server rejects a request naming any
# other model, and the alias is an operator choice rather than a checkpoint
# property, so it has to be configurable.
FASTVIDEO_MODEL = setting("H3_FASTVIDEO_MODEL", "fasth3")

# Which request contract a vLLM-Omni upstream speaks. Three exist:
#
# "current" is v0.30.0 on. vllm-omni#7219 (merged 2026-09-24) made
# num_inference_steps count denoiser evaluations: N steps used to build N sigma
# points and so run N-1 forwards, and now build N+1 and run N. Turbo's check
# moved with it, so the 4 step adapter now wants 4 where it wanted 5.
#
# "2026-08" is the 2026-08-22 nightly through v0.29.x: t2va needs a named
# aspect ratio and the clip has to be four to fifteen seconds, but steps still
# count sigma points.
#
# "legacy" is the minimax-h3 image published on 2026-08-02, where the canvas
# alone was enough and nothing bounded the duration from below. It counts
# sigma points too.
#
# Anything unrecognised is treated as current, because that is the safe
# direction to guess in: sending a named ratio to an old image is ignored,
# while omitting one on a current server fails the request.
SERVER_CONTRACTS = ("current", "2026-08", "legacy")
SERVER_CONTRACT = setting("H3_SERVER_CONTRACT", "current").strip().lower()
if SERVER_CONTRACT not in SERVER_CONTRACTS:
    SERVER_CONTRACT = "current"
STEPS_COUNT_SIGMA_POINTS = SERVER_CONTRACT != "current"
VLLM_DURATION = (0.5, 60.0) if SERVER_CONTRACT == "legacy" else (4.0, 15.0)
CURRENT = SERVER_CONTRACT == "current"


def switch(key, default):
    """A yes/no setting. Anything but an explicit no counts as yes."""
    return setting(key, default).strip().lower() not in ("0", "false", "off", "no")


# Encode the MP4 on the worker while the VAE is still decoding (vllm-omni#7018,
# v0.30.0). The API process then receives finished bytes and skips the
# synchronous mux on its event loop, which #7095 would move off it and has not.
# On v0.30.0 that mux stalled the server for 0.67 s on a 1344x768/4 s clip, and
# 0.03 s with this on; the render takes as long and the MP4 is the same. The
# gain is on long clips, which never sit decoded in the API process.
PREENCODE_MP4 = CURRENT and switch("H3_PREENCODE_MP4", "1")
PREENCODE_BATCH_FRAMES = setting("H3_PREENCODE_BATCH_FRAMES", "")

# Run vLLM-Omni jobs through POST /v1/videos and poll, rather than holding one
# synchronous request open. The point is DELETE: since vllm-omni#7844 (v0.30.0)
# MiniMax-H3 stops a cancelled request at the next denoising boundary, so a
# running job can be cancelled instead of only a queued one.
ASYNC_JOBS = CURRENT and switch("H3_ASYNC_JOBS", "1")
POLL_INTERVAL = 2.0
# How long the upstream may stay unreachable while a job is on it before the
# job is given up. A server restart takes minutes; a busy one, seconds.
POLL_GIVE_UP_SECONDS = 300.0

# Long video (vllm-omni#7838, v0.30.0). Above 15 s a request has to opt in with
# long_video, and then full mode allows 30 s and Ref2VA continuation 300 s.
# These are the server's sanity bounds, not a promise the canvas fits memory.
LONG_FULL_MAX = 30.0
LONG_CONTINUATION_MAX = 300.0
# Continuation windows live on the 17n+5 frame grid; the defaults are upstream's.
CONTINUATION_WINDOW = 277
CONTINUATION_OVERLAP = 22
CONTINUATION_WINDOW_RANGE = (107, 345)

# Two features the server only has when it was started for them, and which an
# older one would not refuse but silently ignore. Neither is detectable over
# the wire, so both are opt in.
#
# Timeline guides (vllm-omni#8460, #8838) are on main from 2026-10-13 and in
# v0.32.0rc1. Latent upscale and refine (#8322, v0.31.0rc1) need the server
# started with --additional-config latent_upscaler_path; a refine alone needs
# no checkpoint but the same build.
TIMELINE_GUIDES = CURRENT and switch("H3_TIMELINE_GUIDES", "0")
LATENT_UPSCALE = CURRENT and switch("H3_LATENT_UPSCALE", "0")
GUIDE_MAX = 8
GUIDE_LIMITS = {"image": 30 << 20, "video": 50 << 20, "audio": 15 << 20}
# The upscaler was trained between 1x and 4x and only upscales.
UPSCALE_RANGE = (1.0, 4.0)

# FastVideo bounds the clip by frames on H3's 17n+5 grid, 124 to 362 at 24 fps.
# 4.5 s is the shortest duration that rounds onto 124 rather than 107, which it
# refuses; 15 s lands on 362, which it accepts since 2026-09-14.
FASTVIDEO_DURATION = (4.5, 15.0)
DURATION_BY_KIND = {"vllm-omni": VLLM_DURATION, "fastvideo": FASTVIDEO_DURATION}

# The form's bounds are the union over the configured boxes. A duration only one
# kind accepts is still valid; the queue just routes it to that kind.
DURATION_MIN = min(DURATION_BY_KIND[kind][0] for kind in KINDS_PRESENT)
DURATION_MAX = max(DURATION_BY_KIND[kind][1] for kind in KINDS_PRESENT)

# The named ratios t2va accepts. width and height still set the canvas; the
# ratio is a separate named field the server refuses to infer, so the nearest
# name is what it wants, not an exact match: the upstream recipe itself pairs
# 960x576 with 16:9.
NAMED_RATIOS = (("21:9", 21 / 9), ("16:9", 16 / 9), ("4:3", 4 / 3),
                ("1:1", 1.0), ("3:4", 3 / 4), ("9:16", 9 / 16))

# FastH3's five sigma points bound four denoiser evaluations. vLLM-Omni names
# the four and FastVideo names the five, and #7219 left fused and checkpoint
# schedules alone, so that holds on every contract. Only used as the default
# when H3_FASTH3 is a bare switch; an integer there wins, which is how the
# eight step V2 student is expressed: 8 on vLLM-Omni, 9 on FastVideo.
FASTH3_STEPS = 5 if KINDS_PRESENT == {"fastvideo"} else 4

# The fit found no constant term worth keeping: forcing it to zero was better
# held out than fitting it, so this ships as 0 and is here for a deployment
# whose own measurements disagree.
EST_FIXED = float(setting("H3_EST_FIXED_SECONDS", "0"))
EST_PER_MPXS = float(setting("H3_EST_PER_MPXS", "11.88"))
EST_PER_MPXS_STEP = float(setting("H3_EST_PER_MPXS_STEP", "6.05"))
EST_PER_MPXS2_STEP = float(setting("H3_EST_PER_MPXS2_STEP", "0.88"))


# The LightX2V Turbo filename names its own sampling policy, and vLLM-Omni reads
# it from there (vllm-omni#7062): task family, denoiser forwards and whether it
# is a 768p retrain. The pattern is upstream's, so a file it would refuse, such
# as a ComfyUI or fp8 export, falls back to configuration here as well.
TURBO_NAME = re.compile(
    r"minimax_h3_(?P<task>fl2v|ref2v)_turbo_(?P<steps>\d+)step"
    r"_v(?P<version>\d+\.\d+)(?P<res>_768p)?(?:_bf16)?\.safetensors")


def turbo_spec(path):
    """What a Turbo filename says about its schedule, or None.

    The 768p retrains moved to a video shift of 6; the 544p artifacts keep the
    checkpoint's 12. Audio stays at 3 for all of them. An fl2v file serves t2va
    and fl2va, and a ref2v file serves ref2va only, on a server started with
    --task-type ref2va.
    """
    match = TURBO_NAME.fullmatch(Path(path).name)
    if not match:
        return None
    forwards = int(match.group("steps"))
    family = match.group("task")
    return {
        "forwards": forwards,
        "version": match.group("version"),
        "flow_shift": 6.0 if match.group("res") else 12.0,
        "audio_flow_shift": 3.0,
        "tasks": ["ref2va"] if family == "ref2v" else ["t2va", "fl2va"],
    }


def request_lora():
    """A LoRA the server preloaded and each request may switch on.

    This is the other kind of adapter, and it is the opposite of a fused one.
    vLLM-Omni's --lora-path with --lora-backend peft keeps the adapter resident
    but inactive; a request carrying a `lora` field activates it, and one
    without renders on the base checkpoint. So it is a per-request choice
    rather than a property of the server, which is what makes a checkbox
    honest here and dishonest for FastH3.

    The adapter dictates its own sampling settings, and the filename says what
    they are. Sending anything else samples the student where it was never
    distilled, so the server refuses it. The step count is the one setting that
    depends on the server as well: a 4 step adapter is steps=4 on a current
    build and steps=5 on one that still counts sigma points.

    Anything set explicitly wins over the filename, so a deployment that
    pinned these by hand keeps exactly what it had.
    """
    path = setting("H3_REQUEST_LORA_PATH", "")
    if not path:
        return None
    spec = turbo_spec(path) or {"forwards": 4, "version": None,
                                "flow_shift": 6.0, "audio_flow_shift": 3.0,
                                "tasks": ["t2va", "fl2va"]}
    steps = spec["forwards"] + (1 if STEPS_COUNT_SIGMA_POINTS else 0)
    tasks = [t.strip() for t in
             setting("H3_REQUEST_LORA_TASKS", ",".join(spec["tasks"])).split(",")
             if t.strip()]
    version = f" v{spec['version']}" if spec["version"] else ""
    return {
        "name": setting("H3_REQUEST_LORA_NAME", "turbo"),
        "path": path,
        "scale": float(setting("H3_REQUEST_LORA_SCALE", "1.0")),
        "steps": int(setting("H3_REQUEST_LORA_STEPS", str(steps))),
        "flow_shift": float(setting("H3_REQUEST_LORA_FLOW_SHIFT",
                                    str(spec["flow_shift"]))),
        "audio_flow_shift": float(setting("H3_REQUEST_LORA_AUDIO_SHIFT",
                                          str(spec["audio_flow_shift"]))),
        "tasks": tasks,
        "label": setting("H3_REQUEST_LORA_LABEL",
                         f"Turbo{version}, {spec['forwards']} denoiser steps"),
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
    adapter pins. The integer form is what covers a student with a different
    count, such as FastH3 8-Step V2.
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


def fastvideo_contract(model_dir):
    """The schedule a FastVideo release declares, as (forwards, label), or None.

    FastVideo's distilled checkpoints carry no _minimax_h3 block. They ship
    fastvideo_inference.json instead, and both servers refuse a step count
    other than the one it names: FastH3 8-Step V2 runs eight forwards, and the
    V1 DataFree repos four. The file counts in either unit depending on its
    age, so the forwards are read from whichever field is present, most
    explicit first.
    """
    path = Path(model_dir) / "fastvideo_inference.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get("schema_version") != \
            "fasth3-inference-contract-v1":
        return None
    forwards = meta.get("transformer_forwards")
    if not isinstance(forwards, int):
        ladder = meta.get("dmd_denoising_steps")
        points = meta.get("num_inference_steps")
        forwards = (len(ladder) if isinstance(ladder, list) and ladder
                    else points - 1 if isinstance(points, int) and points > 1
                    else None)
    if not forwards:
        return None
    label = "FastH3 V2" if forwards == 8 else "FastH3"
    return forwards, f"{label} ({forwards} step)"


def load_partition():
    """The served checkpoint's partition, task list and pinned schedule.

    Three different things can pin the step count. A distilled checkpoint says
    so in its own metadata and the server pins the count from there. A FastVideo
    release says so in fastvideo_inference.json instead. A FastH3 adapter fused
    with --lora-path says nothing at all: /v1/models is unchanged and so is
    model_index.json, so that one can only come from configuration.
    """
    info = {"partition": "unknown", "tasks": ["t2va"],
            "pinned_steps": None, "locked_shifts": False, "adapter": None}
    model_dir = setting("MINIMAX_H3_MODEL_DIR", "")
    index = Path(model_dir) / "model_index.json" if model_dir else None
    if index and index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8")).get("_minimax_h3", {})
        info["partition"] = meta.get("partition", "unknown")
        info["tasks"] = list(meta.get("tasks") or ["t2va"])
        pinned = schedule_steps(meta)
        if pinned:
            info.update(pinned_steps=pinned, locked_shifts=True,
                        adapter="a distilled checkpoint")
    contract = fastvideo_contract(model_dir) if model_dir else None
    if contract and not info["pinned_steps"]:
        # The same release is steps=8 to vLLM-Omni and steps=9 to FastVideo.
        # The number is sent as given, so it has to be the one this fleet's
        # servers want; a mixed fleet is warned about at startup.
        forwards, label = contract
        steps = forwards + 1 if KINDS_PRESENT == {"fastvideo"} else forwards
        info.update(pinned_steps=steps, locked_shifts=True, tasks=["t2va"],
                    adapter=label)
        if info["partition"] == "unknown":
            info["partition"] = "fasth3"
    # The adapter is fused over whatever the checkpoint says, so it wins. A
    # FastH3 box may not set MINIMAX_H3_MODEL_DIR at all, which is why this
    # sits outside the blocks above.
    lora_path = setting("H3_LORA_PATH", "")
    override = fasth3_steps(setting("H3_FASTH3", ""))
    fused = override if override is not None else (FASTH3_STEPS if lora_path else 0)
    if fused:
        info.update(pinned_steps=fused, locked_shifts=True, tasks=["t2va"],
                    adapter=(adapter_label(lora_path) if lora_path
                             else f"fasth3 ({fused} step)"))
    return info


PARTITION = load_partition()

JOBS = {}
JOBS_LOCK = threading.Lock()
JOBS_READY = threading.Condition(JOBS_LOCK)

# One worker per upstream draining a shared FIFO, rather than a thread per job
# contending on a lock: threading.Lock has no ordering guarantee, so queued
# work used to start in an arbitrary order. Each GPU serialises its own work,
# so this makes the order you submitted the order work starts, across however
# many boxes are configured.
#
# A deque under JOBS_LOCK rather than a queue.Queue, because not every box can
# serve every job. A FastVideo worker takes the oldest job it can run and
# leaves the rest in order for a vLLM-Omni one, where a plain FIFO would hand
# it an fl2va job to fail. Holding the same lock as JOBS also makes taking a
# job and marking it running one step, so a cancel cannot land in between.
# Entries are (job_id, params, media, lang); only queued jobs are in it.
JOB_QUEUE = collections.deque()

# Finished jobs stay listed so the queue view keeps its history across a page
# reload, but not without bound.
MAX_FINISHED = 60
FINAL_STATES = ("done", "failed", "cancelled")
# On a box: running, or being aborted there and still holding the GPU.
ACTIVE_STATES = ("running", "cancelling")

ATTACHMENT_KEYS = ("image", "audio", "videos", "source_video", "source_audio",
                   "guide_files")


def auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def split_data_url(value, lang="en"):
    """Return (mime, raw_bytes) from a data: URL.

    The type is held to type/subtype characters because it is copied into a
    multipart header, where a CR or LF would start a header of its own.
    """
    match = re.fullmatch(r"data:([\w.+-]+/[\w.+-]+);base64,(.+)",
                         value if isinstance(value, str) else "", re.S)
    if not match:
        raise ValueError(t(lang, "attachment_data_url"))
    try:
        return match.group(1), base64.b64decode(match.group(2), validate=True)
    except binascii.Error as exc:
        raise ValueError(t(lang, "attachment_decode", error=exc)) from exc


# The server picks a decoder by the upload's suffix, and guess_extension has
# opinions of its own about some of these (audio/mpeg is not always .mp3).
EXTENSIONS = {"video/mp4": ".mp4", "video/quicktime": ".mov", "audio/wav": ".wav",
              "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/mpeg": ".mp3",
              "audio/mp3": ".mp3", "audio/flac": ".flac", "audio/x-flac": ".flac",
              "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def extension(mime):
    return EXTENSIONS.get(mime) or mimetypes.guess_extension(mime) or ".bin"


def media_kind(mime):
    """image, video or audio, from a MIME type, or None."""
    kind = mime.split("/", 1)[0]
    return kind if kind in ("image", "video", "audio") else None


def decode_attachments(raw, lang="en"):
    """Validate and decode attachments when they arrive, not when the job runs.

    A bad data URL used to surface only once a worker reached the job, which
    can be hours after the request that queued it. Decoding here also means the
    queue holds bytes rather than base64 text a third larger.

    Audio is the exception: an older vLLM-Omni takes it only as a URL inside a
    JSON field rather than as a file part, so it travels as the data URL it
    arrived as, checked but not kept decoded, and is decoded when the request
    is built for a server that takes it as a file.
    """
    if not isinstance(raw, dict):
        raise ValueError(t(lang, "attachment_shape"))
    for key in raw:
        if key not in ATTACHMENT_KEYS:
            raise ValueError(t(lang, "attachment_unknown", name=str(key)[:40]))
    videos = raw.get("videos") or []
    guides = raw.get("guide_files") or []
    if not isinstance(videos, list) or not isinstance(guides, list):
        raise ValueError(t(lang, "attachment_shape"))
    image, audio = raw.get("image"), raw.get("audio")
    if audio:
        split_data_url(audio, lang)
    single = {key: split_data_url(raw[key], lang) if raw.get(key) else None
              for key in ("source_video", "source_audio")}
    return {"image": split_data_url(image, lang) if image else None,
            "audio": audio or None,
            "videos": [split_data_url(item, lang) for item in videos],
            "guide_files": [split_data_url(item, lang) for item in guides],
            **single}


def attached_names(media):
    """Which attachment fields a job carried, from the fixed set only."""
    return [key for key in ATTACHMENT_KEYS if media.get(key)]


def encode_multipart(fields, files):
    """Build a multipart/form-data body.

    fields: {name: text}. files: [(name, filename, mime, bytes)]. Returned as
    the bytearray it was built in, which urllib sends as is, rather than
    copying a body that can be hundreds of megabytes once more.
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
    return body, f"multipart/form-data; boundary={boundary}"


def can_serve(kind, params):
    """Whether a box of this kind can run a job, decided before it is taken.

    FastVideo takes text only, has no request LoRA, and bounds the clip by its
    own frame grid. Everything else is vLLM-Omni's, within its contract.
    """
    if kind == "fastvideo":
        lo, hi = FASTVIDEO_DURATION
        return (lo <= params["duration"] <= hi and params["task"] == "t2va"
                and not params.get("lora")
                and not any(params.get(key) for key in VLLM_ONLY))
    lo, hi = VLLM_DURATION
    if params.get("long_video"):
        hi = long_video_max(params["task"], params["long_video"]["mode"])
    return lo <= params["duration"] <= hi


# Request features only a current vLLM-Omni speaks. A job carrying one never
# goes to a FastVideo box.
VLLM_ONLY = ("long_video", "audio_mode", "edit", "guides", "latent_upscale",
             "latent_refine")


def long_video_max(task, mode):
    return LONG_CONTINUATION_MAX if mode == "continuation" else LONG_FULL_MAX


def continuation_windows(frames, window, overlap):
    """How many windows a continuation render samples for this many frames.

    Each window after the first starts `window - overlap` frames on, and the
    last may be short. Upstream's 75 s example, 1807 frames at 277/22, is 7.
    """
    if frames <= window:
        return 1
    stride = window - overlap
    return 1 + -(-(frames - window) // stride)


def video_token_of_frame(frame):
    """The video latent a 24 fps output frame folds into.

    The causal video VAE groups each 17 frame clip into 5 latents: frame 0 on
    its own, then four frames each. A 17n+5 clip ends on a partial group, so
    its latent count is 5n+2, which is what the server's grid says.
    """
    clip, local = divmod(frame, 17)
    return clip * 5 + (0 if local == 0 else (local + 3) // 4)


def video_latent_count(frames):
    return 2 if frames <= 5 else (frames - 5) // 17 * 5 + 2


def edit_masks(params):
    """The video and audio noise masks for an edit, as JSON-ready values.

    vllm-omni v0.30.0 reads a video mask as a scalar or on its token grid,
    [latents, H/32, W/32], here sent flat; raw frame-space masks only arrive
    with #7947. So a time range is resolved onto that grid on this side, each
    latent taking the largest value of the frames it covers, so any frame the
    range touches is regenerated. The audio mask is one value per 40 Hz latent.

    A strength applies inside the range and 0 (keep) outside it; with no
    range it applies everywhere. Exactly 1 everywhere means "no mask", which
    the server treats as a no-op anyway, so it is not sent.
    """
    edit = params["edit"]
    frames = frame_count(params["duration"], params["fps"])
    start, end = edit.get("start"), edit.get("end")
    ranged = start is not None or end is not None
    first = 0 if start is None else max(int(round(start * params["fps"])), 0)
    last = frames if end is None else min(int(round(end * params["fps"])), frames)

    masks = {}
    video = edit["video"]
    if not ranged:
        if video != 1.0:
            masks["video"] = video
    else:
        tokens = [0.0] * video_latent_count(frames)
        for frame in range(first, last):
            tokens[video_token_of_frame(frame)] = video
        if any(value != 1.0 for value in tokens):
            cells = (params["height"] // 32) * (params["width"] // 32)
            masks["video"] = [value for value in tokens for _ in range(cells)]
    audio = edit["audio"]
    latents = int(round(frames / params["fps"] * 40))
    if not ranged:
        if audio != 1.0:
            masks["audio"] = audio
    else:
        lo, hi = first / params["fps"], last / params["fps"]
        row = [audio if lo <= index / 40 < hi else 0.0 for index in range(latents)]
        if any(value != 1.0 for value in row):
            masks["audio"] = row
    return masks


def build_fastvideo_request(params, media, lang="en"):
    """Translate UI parameters into FastVideo's OpenAI compatible video request.

    Four differences from the vLLM-Omni body matter. The transport is JSON
    rather than multipart. Frames are named directly instead of being derived
    from a duration, because FastH3 only accepts counts on its causal VAE grid
    and rounding a duration into one behind the caller's back would silently
    change what was measured. Neither shift is ever sent: the released
    checkpoint owns its schedule, and a request naming a shift is refused
    rather than ignored.

    And num_inference_steps counts something else here. FastVideo names sigma
    points, five for FastH3 V1 and nine for V2, where a current vLLM-Omni names
    the four and eight denoiser evaluations between them. So the same render is
    steps=4 on one and steps=5 on the other, and neither is wrong. The number
    is passed through as given rather than translated, so a sidecar always
    records what was actually sent.

    Reference media would ride along as image_reference here. Nothing needs it
    yet, and the queue never routes an attached job to this kind, so an
    attachment reaching this point is refused rather than dropped quietly.

    Returns (body, content_type, sent), where `sent` is the body less the
    prompt, which the sidecar already carries.
    """
    if attached_names(media):
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
    sent = {key: value for key, value in body.items() if key != "prompt"}
    return json.dumps(body).encode(), "application/json", sent


def frame_count(duration, fps):
    """Frames for a duration, snapped up to MiniMax H3's 17n+5 grid.

    Both servers land on that grid. vLLM-Omni snaps for you and reports the
    count it used; FastVideo rejects anything off grid. Snapping here means the
    number in the sidecar is the number that was rendered either way.
    """
    raw = max(int(round(float(duration) * float(fps))), 5)
    return raw + (5 - raw) % 17


def build_request(params, media, lang="en"):
    """Translate UI parameters into the vLLM-Omni video request.

    Returns (fields, files, sent). `sent` is what went on the wire less the
    prompt and the attachment bytes, with the JSON valued fields decoded, so
    the sidecar can say exactly what the server was asked for.
    """
    extra = {"task": params["task"], "duration": params["duration"]}
    fields = {
        "prompt": params["prompt"],
        "num_inference_steps": params["steps"],
        "seed": params["seed"],
        "fps": params["fps"],
    }
    if PREENCODE_MP4:
        extra["preencode_mp4"] = True
        if PREENCODE_BATCH_FRAMES:
            extra["preencode_batch_frames"] = int(PREENCODE_BATCH_FRAMES)
    long_video = params.get("long_video")
    if long_video:
        extra["long_video"] = True
        extra["long_video_mode"] = long_video["mode"]
        for key, name in (("window", "continuation_window_frames"),
                          ("overlap", "continuation_overlap_frames"),
                          ("prompts", "continuation_prompts")):
            if long_video.get(key):
                extra[name] = long_video[key]
    for key in ("audio_mode", "latent_upscale", "latent_refine"):
        if params.get(key):
            extra[key] = params[key]
    # Continuation runs uncached only, and guides need the native path, so
    # both name it rather than rely on how the server was started.
    if params.get("guides") or (long_video or {}).get("mode") == "continuation":
        fields["quality"] = "lossless"
    lora = None
    # An activated request LoRA brings its own schedule, so it decides the
    # step count and both shifts rather than the form.
    if params.get("lora") and REQUEST_LORA:
        fields["num_inference_steps"] = REQUEST_LORA["steps"]
        fields["flow_shift"] = REQUEST_LORA["flow_shift"]
        extra["audio_flow_shift"] = REQUEST_LORA["audio_flow_shift"]
        lora = {"name": REQUEST_LORA["name"], "path": REQUEST_LORA["path"],
                "scale": REQUEST_LORA["scale"]}
        fields["lora"] = json.dumps(lora)
    # A fused schedule owns both shifts: measured against the 2026-08-31 nightly,
    # the server accepts a value only when it equals the checkpoint's own (12 and
    # 3 for FastH3) and answers "FastH3 requires flow_shift=12, got 8" otherwise.
    # Omitting them is always accepted, and it is the only option that cannot
    # disagree with a schedule this process never sees.
    elif not PARTITION["locked_shifts"]:
        fields["flow_shift"] = params["flow_shift"]
        extra["audio_flow_shift"] = params["audio_flow_shift"]
    # t2va on a 2026-08-22 or later build wants a named ratio and refuses to
    # infer one from the canvas. fl2va takes its ratio from the input image and
    # ref2va defaults to 16:9, so neither is sent one.
    if SERVER_CONTRACT != "legacy" and params["task"] == "t2va":
        fields["aspect_ratio"] = named_ratio(params.get("width"),
                                             params.get("height"))
    fields["extra_params"] = json.dumps(extra)
    # fl2va derives the canvas from the reference image when width/height are
    # omitted; the UI exposes that as "follow the image".
    if params.get("width") and params.get("height"):
        fields["width"] = params["width"]
        fields["height"] = params["height"]

    files = []
    image = media.get("image")
    videos = media.get("videos") or []
    audio = media.get("audio")

    # On a current server an image + audio pair goes up as two files under
    # input_references, which H3 sorts by type. As a data URL inside the
    # audio_reference field it would be a plain form field, and the server
    # refuses any form field over 1 MiB: a reference track of a few seconds
    # already is one, and the request died as a reset connection.
    pair = CURRENT and bool(image) and bool(audio)
    if image:
        mime, blob = image
        # input_reference is sniffed server-side into an image or video.
        files.append(("input_references" if pair else "input_reference",
                      "reference" + (mimetypes.guess_extension(mime) or ".png"),
                      mime, blob))
    if videos:
        name = "input_references" if len(videos) > 1 else "input_reference"
        for index, (mime, blob) in enumerate(videos):
            files.append((name, f"reference-{index}" +
                          (mimetypes.guess_extension(mime) or ".mp4"), mime, blob))
    if pair:
        mime, blob = split_data_url(audio, lang)
        files.append(("input_references", "reference-audio" + extension(mime),
                      mime, blob))
    elif audio:
        fields["audio_reference"] = json.dumps({"audio_url": audio})

    # A latent edit sends its sources as files and each mask as a JSON file
    # part, which is the only form the server reads them in.
    masks = edit_masks(params) if params.get("edit") else {}
    for key in ("source_video", "source_audio"):
        if media.get(key):
            mime, blob = media[key]
            files.append((key, key + extension(mime), mime, blob))
    for key, value in masks.items():
        files.append((f"{key}_noise_mask", f"{key}-mask.json", "application/json",
                      json.dumps(value, separators=(",", ":")).encode()))

    # Timeline guides: a manifest naming uploads by position, then the uploads
    # in that order. Each row brings its own files, so none is shared.
    manifest = []
    for guide in params.get("guides") or []:
        entry = {"frame_index": guide["frame_index"]}
        for slot in ("image", "video", "audio"):
            if guide.get(slot) is not None:
                mime, blob = media["guide_files"][guide[slot]]
                entry[slot] = {"upload_index": sum(1 for f in files
                                                   if f[0] == "guide_files")}
                files.append(("guide_files", f"guide-{len(manifest)}-{slot}"
                              + extension(mime), mime, blob))
        manifest.append(entry)
    if manifest:
        fields["timeline_guides"] = json.dumps(manifest)

    sent = {key: value for key, value in fields.items()
            if key not in ("prompt", "extra_params", "lora", "audio_reference",
                           "timeline_guides")}
    sent["extra_params"] = extra
    if manifest:
        sent["timeline_guides"] = manifest
    if masks:
        # A token grid is thousands of numbers; the sidecar says what it was
        # built from instead, which is in params already.
        sent["noise_masks"] = {key: value if isinstance(value, float)
                               else f"{len(value)} values"
                               for key, value in masks.items()}
    if lora:
        sent["lora"] = lora
    if files:
        sent["files"] = [{"field": name, "type": mime, "bytes": len(blob)}
                         for name, _, mime, blob in files]
    if audio and not pair:
        sent["audio_reference"] = {"type": audio[5:audio.find(";")]}
    return fields, files, sent


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


def write_atomically(target, data):
    """Write a file under a temporary name and rename it into place.

    A crash or a full disk mid-write would otherwise leave a truncated file
    that the gallery serves as if it were a result.
    """
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, target)


def run_job(job_id, params, media, lang, api_base, started):
    """Run one generation to completion. Called only by a queue worker.

    `lang` is the submitting page's language: a job can fail long after the
    request that queued it, so the language travels with the job rather than
    being read off whichever request happens to collect the error.

    `api_base` is the upstream this worker owns, and the job was already marked
    running on it when the worker took it.
    """
    def touch(**kw):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(kw)

    kind = BACKEND_KIND.get(api_base, "vllm-omni")
    try:
        if kind == "fastvideo":
            body, content_type, sent = build_fastvideo_request(params, media, lang)
        else:
            fields, files, sent = build_request(params, media, lang)
            body, content_type = encode_multipart(fields, files)
    except ValueError as exc:
        touch(state="failed", error=str(exc), elapsed=0)
        return
    if kind == "vllm-omni" and ASYNC_JOBS:
        run_async_job(job_id, params, media, api_base, started, body,
                      content_type, sent, touch)
        return
    request = urllib.request.Request(
        f"{api_base}/v1/videos/sync", data=body,
        headers={"Content-Type": content_type, **auth_headers()},
        method="POST")

    MEDIA.mkdir(exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id[:8]}.mp4"
    target = MEDIA / name
    partial = MEDIA / (name + ".part")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            content_kind = response.headers.get("Content-Type", "")
            metrics = server_metrics(response.headers)
            # Look at the head before committing anything to disk: an error
            # page served with a 200 is text, and an MP4 opens on a box size.
            head = response.read(64 * 1024)
            if "video" not in content_kind and not head.startswith(b"\x00\x00\x00"):
                touch(state="failed", elapsed=time.time() - started,
                      error=f"non-video response ({content_kind}): "
                            f"{head[:1200].decode('utf-8', 'replace')}")
                return
            # Streamed rather than read whole: a long 1344x768 clip is large,
            # and holding it twice in memory bought nothing.
            with partial.open("wb") as handle:
                handle.write(head)
                shutil.copyfileobj(response, handle, 1024 * 1024)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1200]
        touch(state="failed", error=f"HTTP {exc.code}: {detail}",
              elapsed=time.time() - started)
        return
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
        partial.unlink(missing_ok=True)
        touch(state="failed", error=f"{type(exc).__name__}: {exc}",
              elapsed=time.time() - started)
        return

    finish_job(touch, params, media, api_base, kind, sent, metrics, started,
               partial, target)


def finish_job(touch, params, media, api_base, kind, sent, metrics, started,
               partial, target):
    """Move a finished download into place, write its sidecar, mark it done."""
    name = target.name
    elapsed = time.time() - started
    os.replace(partial, target)
    # The sidecar lands last, so a crash before it leaves a file the history
    # does not list rather than a record pointing at nothing.
    write_atomically(MEDIA / (name + ".json"), json.dumps(
        {**params, "elapsed": elapsed, "file": name,
         "backend": api_base, "backend_kind": kind, "sent": sent,
         **({"server_metrics": metrics} if metrics else {}),
         "attached": attached_names(media)},
        indent=2, ensure_ascii=False).encode("utf-8"))
    touch(state="done", elapsed=elapsed, file=name, size=target.stat().st_size)


def call_upstream(api_base, method, path, body=None, content_type=None, timeout=30):
    """One call to an upstream: (status, decoded JSON or None, raw bytes).

    An HTTP error status is a reply, not an exception, because the async API
    answers a failed job's GET with the failure's own status and a JSON body
    describing it. Only a transport failure raises.
    """
    headers = dict(auth_headers())
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(f"{api_base}{path}", data=body,
                                     headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        data = json.loads(raw) if raw else None
    except ValueError:
        data = None
    return status, data, raw


def upstream_error(status, data, raw):
    """The message a failed upstream call carries, as the UI shows it."""
    detail = None
    if isinstance(data, dict):
        error = data.get("error")
        detail = (error.get("message") if isinstance(error, dict) else error) \
            or data.get("detail") or data.get("message")
    if not detail:
        detail = raw[:1200].decode("utf-8", "replace")
    return f"HTTP {status}: {str(detail)[:1200]}"


def cancel_requested(job_id):
    with JOBS_LOCK:
        return JOBS.get(job_id, {}).get("state") == "cancelling"


def run_async_job(job_id, params, media, api_base, started, body,
                  content_type, sent, touch):
    """Run one job through vLLM-Omni's job API: create, poll, fetch, delete.

    Holding no request open is what makes a running job cancellable. A cancel
    marks the job `cancelling`; this loop sees it, asks the server to abort,
    and calls the job cancelled once the server says it is gone. That answer
    comes before the GPU is free: the server stops MiniMax-H3 at the next
    denoising boundary, measured at 10 s after a 2 s confirmation on a GB10.
    The box's next job may be sent in between and waits on the server, which
    runs one request at a time, so the two never share the GPU.

    The server's own record answers whether it has started (queued means it
    is still waiting for the scheduler) and, when done, how long it ran and
    its peak memory, which is what the sync path read from headers.
    """
    status, data, raw = call_upstream(api_base, "POST", "/v1/videos", body,
                                 content_type, timeout=600)
    if status != 200 or not isinstance(data, dict) or not data.get("id"):
        touch(state="failed", error=upstream_error(status, data, raw),
              elapsed=time.time() - started)
        return
    remote = data["id"]
    touch(remote_id=remote, remote_status=data.get("status") or "queued")

    last_seen = time.time()
    while True:
        if cancel_requested(job_id):
            abort_remote(api_base, remote, touch, started)
            return
        try:
            status, data, raw = call_upstream(api_base, "GET", f"/v1/videos/{remote}")
        except (OSError, http.client.HTTPException) as exc:
            # A busy or restarting server. Give it the time a restart takes
            # before calling the job lost.
            if time.time() - last_seen > POLL_GIVE_UP_SECONDS:
                touch(state="failed", elapsed=time.time() - started,
                      error=f"{type(exc).__name__}: {exc}")
                return
            time.sleep(POLL_INTERVAL)
            continue
        last_seen = time.time()
        state = data.get("status") if isinstance(data, dict) else None
        if status == 200 and state in ("queued", "in_progress"):
            touch(remote_status=state)
            time.sleep(POLL_INTERVAL)
            continue
        if status == 200 and state == "completed":
            break
        # A failed job answers with its failure's status; a 404 means the
        # server forgot the job, which a restart does.
        touch(state="failed", elapsed=time.time() - started,
              error=upstream_error(status, data, raw))
        upstream_quietly(api_base, "DELETE", f"/v1/videos/{remote}")
        return

    metrics = {key: data[name] for key, name in
               (("inference_seconds", "inference_time_s"),
                ("peak_memory_mb", "peak_memory_mb"),
                ("stages", "stage_durations"))
               if data.get(name) not in (None, {}, 0)}
    MEDIA.mkdir(exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id[:8]}.mp4"
    target = MEDIA / name
    partial = MEDIA / (name + ".part")
    request = urllib.request.Request(f"{api_base}/v1/videos/{remote}/content",
                                     headers=auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            head = response.read(64 * 1024)
            if not head.startswith(b"\x00\x00\x00"):
                touch(state="failed", elapsed=time.time() - started,
                      error="non-video content: "
                            + head[:1200].decode("utf-8", "replace"))
                return
            with partial.open("wb") as handle:
                handle.write(head)
                shutil.copyfileobj(response, handle, 1024 * 1024)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
        partial.unlink(missing_ok=True)
        touch(state="failed", elapsed=time.time() - started,
              error=f"{type(exc).__name__}: {exc}")
        return
    finally:
        # The server keeps every finished job's file until it is deleted.
        upstream_quietly(api_base, "DELETE", f"/v1/videos/{remote}")
    finish_job(touch, params, media, api_base, "vllm-omni", sent, metrics,
               started, partial, target)


def upstream_quietly(api_base, method, path):
    """A best-effort call whose outcome changes nothing here."""
    try:
        return call_upstream(api_base, method, path)[0]
    except (OSError, http.client.HTTPException):
        return None


def abort_remote(api_base, remote, touch, started):
    """Cancel a job on the server and wait until it says the job is gone.

    DELETE answers 409 while the abort is still in progress, so it is asked
    again until it answers 200 or 404. If the server never confirms, the job is
    still marked cancelled rather than left hanging, with the reason attached.
    """
    deadline = time.time() + POLL_GIVE_UP_SECONDS
    note = None
    while time.time() < deadline:
        status = upstream_quietly(api_base, "DELETE", f"/v1/videos/{remote}")
        if status in (200, 404):
            break
        note = f"abort unconfirmed (last answer {status})"
        time.sleep(POLL_INTERVAL)
    touch(state="cancelled", elapsed=time.time() - started,
          **({"error": note} if note and status not in (200, 404) else {}))


def take_job(api_base, timeout):
    """The oldest queued job this box can serve, now marked running, or None.

    Waits up to `timeout` for one to arrive. Taking and marking happen under
    one lock, so a job is either cancelled or running, never cancelled after a
    worker has already committed to it.
    """
    kind = BACKEND_KIND.get(api_base, "vllm-omni")
    with JOBS_READY:
        for entry in JOB_QUEUE:
            if can_serve(kind, entry[1]):
                JOB_QUEUE.remove(entry)
                started = time.time()
                JOBS[entry[0]].update(state="running", started=started,
                                      backend=api_base, backend_kind=kind,
                                      cancellable=kind == "vllm-omni" and ASYNC_JOBS)
                return (*entry, started)
        JOBS_READY.wait(timeout)
    return None


def worker_loop(api_base):
    """Drain the shared queue forever, one job at a time on this upstream.

    One worker per upstream, all pulling from the same FIFO: whichever box
    frees up first takes the next job it can serve, and submission order is
    still the order work starts.
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
        taken = take_job(api_base, PROBE_INTERVAL)
        if taken is None:
            continue          # recheck this box's health, then wait again
        job_id, params, media, lang, started = taken
        try:
            run_job(job_id, params, media, lang, api_base, started)
        except Exception as exc:  # noqa: BLE001 - a bad job must not end the worker
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id].update(
                        state="failed", error=f"{type(exc).__name__}: {exc}",
                        elapsed=time.time() - started)
        finally:
            # Release the attachments now rather than at the next job.
            taken = media = None
            prune_jobs()


def prune_jobs():
    """Drop the oldest finished jobs once the history grows past the cap."""
    with JOBS_LOCK:
        finished = [j for j in JOBS.values() if j.get("state") in FINAL_STATES]
        for job in sorted(finished, key=lambda j: j.get("created", 0))[:-MAX_FINISHED]:
            JOBS.pop(job["id"], None)


def queue_positions():
    """{job_id: 1-based place in line}. Call with JOBS_LOCK held."""
    return {entry[0]: index for index, entry in enumerate(JOB_QUEUE, 1)}


def job_list():
    """Every job the queue view needs, newest first, without the attachments."""
    now = time.time()
    with JOBS_LOCK:
        jobs = [dict(job) for job in JOBS.values()]
        positions = queue_positions()
    for job in jobs:
        if job.get("state") in ACTIVE_STATES and job.get("started"):
            job["elapsed"] = now - job["started"]
        if job["id"] in positions:
            job["position"] = positions[job["id"]]
    jobs.sort(key=lambda j: j.get("created", 0), reverse=True)
    return jobs


def enqueue(params, media, lang):
    """Queue a job and wake the workers. Returns (job_id, position)."""
    job_id = uuid.uuid4().hex
    with JOBS_READY:
        JOBS[job_id] = {"id": job_id, "state": "queued", "params": params,
                        "estimate": estimate_seconds(params),
                        "created": time.time()}
        JOB_QUEUE.append((job_id, params, media, lang))
        JOBS_READY.notify_all()
        return job_id, len(JOB_QUEUE)


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
    """Cancel a job. Returns the resulting state.

    A queued job is cancelled on the spot. A running one can be cancelled only
    on a box that runs jobs through the async API: it becomes `cancelling`,
    and its worker aborts it upstream and finishes the transition.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        if job.get("state") == "running" and job.get("cancellable"):
            job["state"] = "cancelling"
            return "cancelling"
        if job.get("state") != "queued":
            # On a synchronous box the work is on the GPU behind one open
            # request, so there is nothing safe to interrupt.
            return job.get("state")
        job["state"] = "cancelled"
        # Out of the line too, so its attachments are released now rather
        # than whenever a worker would have skipped past it.
        for entry in JOB_QUEUE:
            if entry[0] == job_id:
                JOB_QUEUE.remove(entry)
                break
        return "cancelled"


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
    mpx = width * height / 1e6

    def render(seconds, step_count, area=mpx):
        x = area * seconds
        return (EST_PER_MPXS * x + EST_PER_MPXS_STEP * x * step_count
                + EST_PER_MPXS2_STEP * x * x * step_count)

    # Continuation samples each window on its own, so the attention term grows
    # with the window rather than with the whole clip. Only the decode is paid
    # on the full length.
    long_video = params.get("long_video") or {}
    seconds = params["duration"]
    if long_video.get("mode") == "continuation":
        window = long_video.get("window") or CONTINUATION_WINDOW
        count = continuation_windows(frame_count(seconds, params["fps"]), window,
                                     long_video.get("overlap") or CONTINUATION_OVERLAP)
        span = window / params["fps"]
        total = count * render(span, steps) - (count * span - seconds) \
            * EST_PER_MPXS * mpx
    else:
        total = render(seconds, steps)
    # An upscale decodes at the target size, and a refine runs its share of
    # the steps there. Neither has been timed on this box yet, so this is the
    # same model applied to the larger canvas, not a fit.
    scale = upscale_factor(params.get("latent_upscale"), width, height)
    if scale > 1:
        total += EST_PER_MPXS * mpx * (scale * scale - 1) * seconds
    refine = params.get("latent_refine") or 0
    if refine:
        total += render(seconds, max(round(steps * refine), 1),
                        mpx * scale * scale) - EST_PER_MPXS * mpx * scale * scale * seconds
    return max(round(EST_FIXED + total), 1)


def upscale_factor(value, width, height):
    """The linear scale a latent_upscale request asks for, 1 when none."""
    if not value:
        return 1.0
    if isinstance(value, (int, float)):
        return float(value)
    if value.get("scale"):
        return float(value["scale"])
    if value.get("width") and value.get("height"):
        return max(value["width"] / width, value["height"] / height)
    if value.get("megapixels"):
        return (value["megapixels"] * 1e6 / (width * height)) ** 0.5
    return 1.0


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
    return {"base": api_base, "kind": BACKEND_KIND.get(api_base, "vllm-omni"),
            "state": state,
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
            # Which request features the form should offer. Each needs a
            # current vLLM-Omni; guides and upscale are also opt in.
            "features": {"long": VLLM_FEATURES, "edit": VLLM_FEATURES,
                         "lock": VLLM_FEATURES,
                         "guides": VLLM_FEATURES and TIMELINE_GUIDES,
                         "upscale": VLLM_FEATURES and LATENT_UPSCALE,
                         "cancel_running": ASYNC_JOBS and "vllm-omni" in KINDS_PRESENT,
                         "preencode": PREENCODE_MP4},
            "long_max": {"full": LONG_FULL_MAX,
                         "continuation": LONG_CONTINUATION_MAX},
            "continuation": {"window": CONTINUATION_WINDOW,
                             "overlap": CONTINUATION_OVERLAP},
            "guide_max": GUIDE_MAX,
            "est": {"fixed": EST_FIXED, "per_mpxs": EST_PER_MPXS,
                    "per_mpxs_step": EST_PER_MPXS_STEP,
                    "per_mpxs2_step": EST_PER_MPXS2_STEP},
            "labels": TASK_LABELS.get(lang, TASK_LABELS["en"])}
    with JOBS_LOCK:
        busy_bases = {j.get("backend") for j in JOBS.values()
                      if j.get("state") in ACTIVE_STATES}
        waiting = len(JOB_QUEUE)
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
            "profile": setting("H3_CACHE_BACKEND", "none"),
            "attention": setting("H3_DIFFUSION_ATTENTION_BACKEND", ""),
            "execution": setting("H3_EXECUTION_MODE", "")}


def history():
    """The newest results' sidecars. A file deleted mid-scan is skipped."""
    if not MEDIA.is_dir():
        return []
    items = []
    for meta in sorted(MEDIA.glob("*.mp4.json"), reverse=True)[:40]:
        try:
            items.append(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return items


class Rejected(ValueError):
    """A request refused with a message meant for the person who sent it."""


def whole_number(payload, key, default, lang, message="param_positive"):
    """A positive integer field, the default when absent, else Rejected.

    A fractional number is refused rather than truncated: 4.5 steps silently
    becoming 4 would leave a sidecar that disagrees with the request.
    """
    value = payload.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or (isinstance(value, float)
                                   and not value.is_integer()):
        raise Rejected(t(lang, message, name=key))
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise Rejected(t(lang, message, name=key)) from None
    if number < 1:
        raise Rejected(t(lang, message, name=key))
    return number


def number(payload, key, default, lang, message):
    """A finite float field, the default when absent, else Rejected."""
    value = payload.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise Rejected(t(lang, message)) from None
    if isinstance(value, bool) or result != result or result in (
            float("inf"), float("-inf")):
        raise Rejected(t(lang, message))
    return result


def parse_generate(payload, lang="en"):
    """Validate a /api/generate body into (params, media), or raise Rejected.

    Mirrors the pipeline's own constraints so the UI fails fast instead of
    spending a cold request on a rejected combination, and so a malformed field
    answers with a 400 that names it rather than a dropped connection.
    """
    if not isinstance(payload, dict):
        raise Rejected(t(lang, "body_json_object"))

    task = str(payload.get("task", "t2va"))
    if task not in PARTITION["tasks"]:
        # Blaming the checkpoint would be a lie when it is the fused adapter
        # that narrowed the task list.
        raise Rejected(t(lang, "task_adapter", adapter=PARTITION["adapter"],
                         tasks=PARTITION["tasks"])
                       if PARTITION["adapter"] else
                       t(lang, "task_unsupported",
                         partition=PARTITION["partition"],
                         tasks=PARTITION["tasks"]))

    duration = number(payload, "duration", DURATION_MIN, lang, "duration_number")
    long_video = parse_long_video(payload.get("long_video"), task, duration, lang)
    hi = long_video_max(task, long_video["mode"]) if long_video else DURATION_MAX
    if not DURATION_MIN <= duration <= hi:
        if duration > hi and not long_video and VLLM_FEATURES:
            raise Rejected(t(lang, "long_needed"))
        raise Rejected(t(lang, "duration_range", lo=DURATION_MIN, hi=hi))

    turbo = bool(payload.get("turbo"))
    if turbo:
        if not REQUEST_LORA:
            raise Rejected(t(lang, "turbo_unavailable"))
        if PARTITION["adapter"]:
            # A fused checkpoint refuses a request that also names a LoRA, so
            # the two adapters are mutually exclusive by construction.
            raise Rejected(t(lang, "turbo_fused", adapter=PARTITION["adapter"]))
        if task not in REQUEST_LORA["tasks"]:
            raise Rejected(t(lang, "turbo_task", name=REQUEST_LORA["name"],
                             task=task, tasks=REQUEST_LORA["tasks"]))

    pinned = PARTITION["pinned_steps"]
    steps = whole_number(payload, "steps", pinned or 20, lang, "steps_number")
    # Reject rather than quietly force. A request that asked for 20 steps and
    # got 4 would leave a sidecar that says something the render never did.
    if pinned and steps != pinned:
        raise Rejected(t(lang, "steps_pinned", adapter=PARTITION["adapter"],
                         steps=pinned))
    if PARTITION["locked_shifts"] and (
            "flow_shift" in payload or "audio_flow_shift" in payload):
        raise Rejected(t(lang, "shifts_locked", adapter=PARTITION["adapter"]))

    prompt = compose_prompt(payload, task, duration)
    if not prompt:
        raise Rejected(t(lang, "prompt_required"))

    try:
        media = decode_attachments(payload.get("attachments") or {}, lang)
    except ValueError as exc:
        raise Rejected(str(exc)) from None
    image, audio, videos = media["image"], media["audio"], media["videos"]
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
        raise Rejected(t(lang, problem))

    if turbo:
        steps = REQUEST_LORA["steps"]

    lock = bool(payload.get("lock_audio"))
    if lock:
        require_feature(True, lang, "lock", "")
        if task != "ref2va" or not audio:
            raise Rejected(t(lang, "lock_source"))

    params = {
        "task": task,
        "prompt": prompt,
        "width": whole_number(payload, "width", None, lang),
        "height": whole_number(payload, "height", None, lang),
        "steps": steps,
        "duration": duration,
        "fps": whole_number(payload, "fps", 24, lang),
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
        params["flow_shift"] = number(payload, "flow_shift", 12, lang,
                                      "shift_number")
        params["audio_flow_shift"] = number(payload, "audio_flow_shift", 3.0,
                                            lang, "shift_number")
    # Keep the sections as written when the structured form was used, so a
    # result can be reopened and edited rather than only re-run verbatim.
    for key in ("description", "soundscape", "music"):
        value = str(payload.get(key) or "").strip()
        if value:
            params[key] = value

    if long_video:
        params["long_video"] = long_video
    if lock:
        params["audio_mode"] = "lock_source"
    parse_upscale(payload, params, lang)
    parse_edit(payload, params, media, lang)
    parse_guides(payload, params, media, lang)
    check_conflicts(params, lang)

    # Queued work nobody can take would wait forever, so say so now. Only a
    # FastVideo-only fleet can get here: vLLM-Omni's bounds are the form's.
    if not any(can_serve(kind, params) for kind in KINDS_PRESENT):
        lo, hi = FASTVIDEO_DURATION
        raise Rejected(t(lang, "no_upstream",
                         reason=t(lang, "reason_fastvideo", lo=lo, hi=hi)))
    return params, media


# Whether any configured box speaks the current vLLM-Omni contract, which every
# request feature below needs.
VLLM_FEATURES = CURRENT and "vllm-omni" in KINDS_PRESENT


def require_feature(enabled, lang, name, key):
    if not VLLM_FEATURES:
        raise Rejected(t(lang, "feature_contract", feature=feature(lang, name)))
    if not enabled:
        raise Rejected(t(lang, "feature_off", feature=feature(lang, name), key=key))


def fraction(value, lang, message, low=0.0, high=1.0, low_open=False):
    """A number within bounds, or Rejected."""
    if isinstance(value, bool):
        raise Rejected(t(lang, message, **BOUNDS))
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise Rejected(t(lang, message, **BOUNDS)) from None
    if result != result or result > high or result < low or (
            low_open and result == low):
        raise Rejected(t(lang, message, **BOUNDS))
    return result


BOUNDS = {"lo": UPSCALE_RANGE[0], "hi": UPSCALE_RANGE[1]}


def parse_long_video(raw, task, duration, lang):
    """{"mode", "window", "overlap", "prompts"} for a long render, or None.

    The mode defaults the way the server's does: continuation for Ref2VA,
    full for everything else, where continuation is refused. Prompts, if any,
    have to come one per window, so the window count is worked out here from
    the same grid the server plans on.
    """
    if not raw:
        return None
    require_feature(True, lang, "long", "")
    raw = raw if isinstance(raw, dict) else {}
    mode = raw.get("mode") or ("continuation" if task == "ref2va" else "full")
    if mode not in ("full", "continuation") or (
            mode == "continuation" and task != "ref2va"):
        raise Rejected(t(lang, "long_mode"))
    result = {"mode": mode}
    if mode != "continuation":
        return result
    lo, hi = CONTINUATION_WINDOW_RANGE
    window = whole_number(raw, "window", None, lang)
    overlap = whole_number(raw, "overlap", None, lang)
    w = window or CONTINUATION_WINDOW
    o = overlap or CONTINUATION_OVERLAP
    if w % 17 != 5 or o % 17 != 5 or not lo <= w <= hi or w <= o:
        raise Rejected(t(lang, "long_window", lo=lo, hi=hi))
    for key, value in (("window", window), ("overlap", overlap)):
        if value:
            result[key] = value
    prompts = raw.get("prompts") or []
    if not isinstance(prompts, list):
        prompts = []
    prompts = [str(p).strip() for p in prompts if str(p).strip()]
    if prompts:
        count = continuation_windows(frame_count(duration, 24), w, o)
        if len(prompts) != count:
            raise Rejected(t(lang, "long_prompts", n=count))
        result["prompts"] = prompts
    return result


def parse_upscale(payload, params, lang):
    """latent_upscale (a scale factor) and latent_refine (a strength)."""
    scale, refine = payload.get("latent_upscale"), payload.get("latent_refine")
    if not scale and not refine:
        return
    require_feature(LATENT_UPSCALE, lang, "upscale" if scale else "refine",
                    "H3_LATENT_UPSCALE")
    if scale:
        value = fraction(scale, lang, "upscale_number", *UPSCALE_RANGE)
        if value > 1:
            params["latent_upscale"] = value
    if refine:
        params["latent_refine"] = fraction(refine, lang, "upscale_number",
                                           low_open=True)


def parse_edit(payload, params, media, lang):
    """A latent edit: which share of the source to regenerate, and where.

    {"video": s, "audio": s, "start": seconds, "end": seconds}. Each strength
    is how much to regenerate, 0 keeping the source and 1 replacing it. The
    masks are built here because the server's grid depends on the canvas,
    which is why an edit needs an explicit one.
    """
    raw = payload.get("edit")
    sources = media.get("source_video") or media.get("source_audio")
    if not raw:
        if sources:
            raise Rejected(t(lang, "edit_sources_only"))
        return
    require_feature(True, lang, "edit", "")
    if not isinstance(raw, dict):
        raise Rejected(t(lang, "edit_number"))
    if not params["width"] or not params["height"]:
        raise Rejected(t(lang, "edit_canvas"))
    edit = {}
    for key in ("video", "audio"):
        edit[key] = fraction(raw.get(key, 1.0), lang, "edit_number")
    for key in ("start", "end"):
        value = raw.get(key)
        edit[key] = None if value in (None, "") else fraction(
            value, lang, "edit_number", 0.0, params["duration"])
    if edit["start"] is not None and edit["end"] is not None \
            and edit["end"] <= edit["start"]:
        raise Rejected(t(lang, "edit_number"))
    params["edit"] = edit
    masks = edit_masks(params)
    if not masks:
        raise Rejected(t(lang, "edit_noop"))
    if "video" in masks and not media.get("source_video"):
        raise Rejected(t(lang, "edit_source_video"))
    if "audio" in masks and not sources:
        raise Rejected(t(lang, "edit_source_audio"))
    params["edit_masks"] = sorted(masks)


def parse_guides(payload, params, media, lang):
    """Timeline guides: a picture, a clip or audio pinned to a moment.

    [{"at": seconds, "visual": n, "audio": n}], where n indexes the uploaded
    guide files. A negative position counts back from the end, as the server's
    frame_index does. The server packs a clip on the 17n+5 grid and checks it
    fits; a still only has to land inside the clip, which is checked here.
    """
    raw = payload.get("guides") or []
    files = media.get("guide_files") or []
    if not raw and not files:
        return
    require_feature(TIMELINE_GUIDES, lang, "guides", "H3_TIMELINE_GUIDES")
    if not isinstance(raw, list) or not raw:
        raise Rejected(t(lang, "guide_shape"))
    if len(raw) > GUIDE_MAX:
        raise Rejected(t(lang, "guide_count", n=GUIDE_MAX))
    frames = frame_count(params["duration"], params["fps"])
    used, guides = set(), []
    for number_, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise Rejected(t(lang, "guide_shape"))
        at = entry.get("at")
        try:
            at = float(at)
        except (TypeError, ValueError):
            raise Rejected(t(lang, "guide_shape")) from None
        guide = {"at": at,
                 "frame_index": (int(round(at * params["fps"])) if at >= 0
                                 else -int(round(-at * params["fps"])))}
        for slot, kinds in (("visual", ("image", "video")), ("audio", ("audio",))):
            index = entry.get(slot)
            if index is None:
                continue
            if isinstance(index, bool) or not isinstance(index, int) \
                    or not 0 <= index < len(files) or index in used:
                raise Rejected(t(lang, "guide_shape"))
            mime, blob = files[index]
            kind = media_kind(mime)
            if kind not in kinds:
                raise Rejected(t(lang, "guide_kind", n=number_))
            if len(blob) > GUIDE_LIMITS[kind]:
                raise Rejected(t(lang, "guide_size", n=number_, kind=kind,
                                 mb=GUIDE_LIMITS[kind] >> 20))
            used.add(index)
            guide[kind] = index
        if not any(key in guide for key in ("image", "video", "audio")):
            raise Rejected(t(lang, "guide_shape"))
        resolved = guide["frame_index"] + (frames if guide["frame_index"] < 0 else 0)
        if not 0 <= resolved < frames:
            raise Rejected(t(lang, "guide_frame", n=number_, secs=at, frames=frames))
        guides.append(guide)
    # The server refuses an upload no entry names.
    if used != set(range(len(files))):
        raise Rejected(t(lang, "guide_shape"))
    params["guides"] = guides


def check_conflicts(params, lang):
    """Refuse the combinations upstream refuses, before they take a GPU.

    Guides want base weights and the native path; refine is a second
    request-mode pass that continuation and editing cannot share; a locked
    soundtrack and an audio mask both claim the audio rows.
    """
    on = {"turbo": bool(params.get("lora")),
          "adapter": bool(PARTITION["adapter"]),
          "continuation": (params.get("long_video") or {}).get("mode") == "continuation",
          "edit": bool(params.get("edit")),
          "guides": bool(params.get("guides")),
          "upscale": bool(params.get("latent_upscale")),
          "refine": bool(params.get("latent_refine")),
          "lock": bool(params.get("audio_mode"))}
    pairs = [("guides", other) for other in
             ("turbo", "adapter", "continuation", "edit", "upscale", "refine", "lock")]
    pairs += [("refine", "continuation"), ("refine", "edit"),
              ("edit", "continuation")]
    for a, b in pairs:
        if on[a] and on[b]:
            raise Rejected(t(lang, "conflict", a=feature(lang, a),
                             b=feature(lang, b)))
    if on["lock"] and "audio" in params.get("edit_masks", ()):
        raise Rejected(t(lang, "conflict", a=feature(lang, "lock"),
                         b=feature(lang, "edit")))


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A browser dropping an idle keep-alive connection, or abandoning a
        # video range it no longer needs, is routine. The stock handler prints
        # a traceback for each, which buried the log in hundreds of them.
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Without it an idle keep-alive connection holds its thread forever.
    timeout = 120

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
        if self.command != "HEAD":
            self.wfile.write(body)

    def _refuse(self, code, error):
        """Answer an error and drop the connection.

        Used wherever a request body may still be unread: on HTTP/1.1 the
        leftover bytes would otherwise be parsed as the next request.
        """
        self._send(code, json.dumps({"error": error}),
                   extra={"Connection": "close"})

    def _safely(self, handler):
        """Run one route, turning an unexpected exception into a JSON 500.

        Without this, any bug answered with a dropped connection and a
        traceback on stderr, which the page could only report as a frontend
        error.
        """
        try:
            handler()
        except ConnectionError:
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            traceback.print_exc()
            self.close_connection = True
            try:
                self._send(500, json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}))
            except OSError:
                pass

    def same_origin(self):
        """Whether a browser request came from this page rather than another site.

        Any site the user visits can make the browser send a request here, and
        the ones that need no preflight arrive with an Origin header naming
        that site. Tools like curl send none, and are not a browser being used
        against its owner.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        host = (self.headers.get("Host") or "").lower()
        return urllib.parse.urlsplit(origin).netloc.lower() == host

    def do_GET(self):
        self._safely(self._get)

    def do_HEAD(self):
        self._safely(self._get)

    def do_POST(self):
        self._safely(self._post)

    def do_DELETE(self):
        self._safely(self._delete)

    def _get(self):
        path = self.path.split("?")[0]
        if path == "/":
            if self.headers.get("If-None-Match") == INDEX_ETAG:
                self.send_response(304)
                self.send_header("ETag", INDEX_ETAG)
                self.end_headers()
                return
            self._send(200, INDEX_BYTES, "text/html; charset=utf-8",
                       {"ETag": INDEX_ETAG, "Cache-Control": "no-cache"})
        elif path == "/api/status":
            self._send(200, json.dumps(service_status(self.lang)))
        elif path.startswith("/api/job/"):
            with JOBS_LOCK:
                job = dict(JOBS.get(path.rsplit("/", 1)[-1], {}))
                positions = queue_positions()
            if not job:
                self._send(404, json.dumps({"error": "unknown job"}))
                return
            if job.get("state") in ACTIVE_STATES:
                job["elapsed"] = time.time() - job["started"]
            if job["id"] in positions:
                job["position"] = positions[job["id"]]
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
        stat = target.stat()
        size = stat.st_size
        # Results are written once under a unique name and never change, so
        # the browser may keep them. The gallery re-renders up to forty
        # <video> elements on every completion and used to refetch each.
        etag = f'"{size:x}-{stat.st_mtime_ns:x}"'
        cache = {"ETag": etag,
                 "Last-Modified": formatdate(stat.st_mtime, usegmt=True),
                 "Cache-Control": ("public, max-age=31536000, immutable"
                                   if name.endswith(".mp4") else "no-cache")}
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            for key, value in cache.items():
                self.send_header(key, value)
            self.end_headers()
            return
        span = self.parse_range(self.headers.get("Range"), size)
        start, end = span if span else (0, size - 1)
        length = end - start + 1

        self.send_response(206 if span else 200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        for key, value in cache.items():
            self.send_header(key, value)
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
            self.close_connection = True

    def _delete(self):
        if not self.same_origin():
            self._refuse(403, t(self.lang, "cross_origin"))
            return
        path = self.path.split("?")[0]
        if path.startswith("/api/history/"):
            self.delete_media(path[len("/api/history/"):])
        elif path == "/api/jobs/finished":
            self._send(200, json.dumps({"forgotten": forget_finished_jobs()}))
        elif path.startswith("/api/job/"):
            job_id = path[len("/api/job/"):]
            self.job_action(job_id, forget_job(job_id), "forgotten",
                            "forget_state")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def job_action(self, job_id, state, success, refusal):
        """Answer a cancel or a forget: 404, 200, or 409 naming the state."""
        if state is None:
            self._send(404, json.dumps({"error": "unknown job"}))
        elif state in (success if isinstance(success, tuple) else (success,)):
            self._send(200, json.dumps({"id": job_id, "state": state}))
        else:
            self._send(409, json.dumps(
                {"error": t(self.lang, refusal, state=state), "state": state}))

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

    def _post(self):
        lang = self.lang
        if not self.same_origin():
            self._refuse(403, t(lang, "cross_origin"))
            return
        path = self.path.split("?")[0]
        if path.startswith("/api/job/") and path.endswith("/cancel"):
            if self.headers.get("Content-Length", "0") != "0":
                self.close_connection = True
            job_id = path[len("/api/job/"):-len("/cancel")]
            self.job_action(job_id, cancel_job(job_id),
                            ("cancelled", "cancelling"), "cancel_state")
            return
        if path != "/api/generate":
            self._refuse(404, "not found")
            return
        # A form or a no-cors fetch from another site can only send the
        # simple content types, and needs no preflight to do it. Insisting on
        # JSON turns that into a preflight this server never answers.
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";")[0].strip().lower() != "application/json":
            self._refuse(415, t(lang, "content_type"))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0:
            self._refuse(400, t(lang, "body_length"))
            return
        if length > MAX_BODY:
            self._refuse(413, t(lang, "body_too_large"))
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, json.dumps({"error": "invalid JSON"}))
            return
        try:
            params, media = parse_generate(payload, lang)
        except Rejected as exc:
            self._send(400, json.dumps({"error": str(exc)}))
            return
        del payload
        job_id, position = enqueue(params, media, lang)
        self._send(200, json.dumps({"id": job_id, "position": position}))


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
  .sidecar { margin-top: 10px; }
  .sidecar summary { font-size: 12px; color: var(--muted); cursor: pointer;
    user-select: none; }
  .sidecar pre { font-family: ui-monospace, monospace; font-size: 12px;
    line-height: 1.55; margin: 8px 0 0; padding: 12px 14px; overflow-x: auto;
    border: 1px solid var(--line); border-radius: 8px; background: #0d0f14; }
  .sidecar button { font: inherit; font-size: 12px; margin-left: 10px;
    padding: 2px 9px; border: 1px solid var(--line); border-radius: 6px;
    background: transparent; color: var(--muted); cursor: pointer; }
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
  .toggle.legend { margin: 0; }
  .dice.add { margin-top: 10px; padding: 6px 12px; font-size: 12px; font-weight: 400; }
  .guide { display: grid; grid-template-columns: 74px 1fr 1fr auto; gap: 8px;
    align-items: end; margin-top: 8px; }
  .guide label { margin-top: 0; }
  .guide .x { width: auto; margin: 0; padding: 7px 9px; font-size: 12px;
    background: transparent; color: var(--muted); border: 1px solid var(--line); }
  @media (max-width: 480px) { .guide { grid-template-columns: 1fr 1fr; } }
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
      <label class="toggle" id="lock-wrap" style="display:none"><input id="lock" type="checkbox"><span data-i18n="lock.toggle"></span></label>
      <p class="hint" id="att-rule"></p>
    </fieldset>

    <fieldset id="long-wrap" style="display:none">
      <legend><label class="toggle legend"><input id="long" type="checkbox"><span data-i18n="long.toggle"></span></label></legend>
      <div id="long-body" style="display:none">
        <div id="long-mode-wrap">
          <label for="long-mode" data-i18n="long.mode"></label>
          <select id="long-mode">
            <option value="continuation" data-i18n="long.continuation"></option>
            <option value="full" data-i18n="long.full"></option>
          </select>
        </div>
        <div id="long-cont">
          <div class="row">
            <div><label for="long-window" data-i18n="long.window"></label><input id="long-window" type="number" step="17" min="107" max="345" placeholder="277"></div>
            <div><label for="long-overlap" data-i18n="long.overlap"></label><input id="long-overlap" type="number" step="17" min="5" placeholder="22"></div>
          </div>
          <label for="long-prompts" data-i18n="long.prompts"></label>
          <textarea id="long-prompts" class="short"></textarea>
        </div>
        <p class="hint" id="long-hint"></p>
      </div>
    </fieldset>

    <fieldset id="edit-wrap" style="display:none">
      <legend><label class="toggle legend"><input id="edit" type="checkbox"><span data-i18n="edit.toggle"></span></label></legend>
      <div id="edit-body" style="display:none">
        <label for="f-source-video" data-i18n="edit.video"></label>
        <input id="f-source-video" type="file" accept="video/mp4,video/quicktime,.mp4,.mov">
        <label for="f-source-audio" data-i18n="edit.audio"></label>
        <input id="f-source-audio" type="file" accept="audio/wav,audio/mpeg,.wav,.mp3">
        <div class="row">
          <div><label for="edit-vs" data-i18n="edit.vstrength"></label><input id="edit-vs" type="number" min="0" max="1" step="0.05" value="1"></div>
          <div><label for="edit-as" data-i18n="edit.astrength"></label><input id="edit-as" type="number" min="0" max="1" step="0.05" value="0"></div>
        </div>
        <div class="row">
          <div><label for="edit-from" data-i18n="edit.from"></label><input id="edit-from" type="number" min="0" step="0.1"></div>
          <div><label for="edit-to" data-i18n="edit.to"></label><input id="edit-to" type="number" min="0" step="0.1"></div>
        </div>
        <p class="hint" data-i18n="edit.hint"></p>
      </div>
    </fieldset>

    <fieldset id="guide-wrap" style="display:none">
      <legend data-i18n="guide.legend"></legend>
      <div id="guides"></div>
      <button type="button" class="dice add" id="guide-add" data-i18n="guide.add"></button>
      <p class="hint" data-i18n="guide.hint"></p>
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
    <div class="row" id="up-row" style="display:none">
      <div><label for="upscale" data-i18n="up.scale"></label>
        <select id="upscale">
          <option value="" data-i18n="up.off"></option>
          <option value="1.5">1.5×</option>
          <option value="2">2×</option>
        </select></div>
      <div><label for="refine" data-i18n="up.refine"></label><input id="refine" type="number" min="0" max="1" step="0.05" value="0"></div>
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
    "sidecar.title": "Reproduction record",
    "sidecar.copy": "Copy",
    "sidecar.copied": "Copied",
    "sidecar.missing": "No sidecar was written beside this file.",
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
    "align.hint": "Prepended automatically: Picture 1 (from Shot 1) → 0.00 s; Picture 2 (from Shot {shot}) → {secs} s",
    "lock.toggle": "Drive the clip with this audio (keep it as the soundtrack)",
    "long.toggle": "Long video (over 15 s)",
    "long.mode": "Long video mode",
    "long.continuation": "Continuation: render in windows, up to 300 s",
    "long.full": "Full: the whole clip at once, up to 30 s",
    "long.window": "Window (frames)",
    "long.overlap": "Overlap (frames)",
    "long.prompts": "Per-window prompts (optional, separated by a line of ---)",
    "long.hint.cont": "{f} frames in {n} windows. Per-window prompts: give {n}, or none to reuse the prompt.",
    "long.hint.full": "{f} frames sampled at once. Memory grows with the whole clip.",
    "edit.toggle": "Edit an existing clip (latent mask)",
    "edit.video": "Source video (.mp4 or .mov)",
    "edit.audio": "Source audio (optional; otherwise the video's own track)",
    "edit.vstrength": "Picture: regenerate 0 to 1",
    "edit.astrength": "Sound: regenerate 0 to 1",
    "edit.from": "From (s, optional)",
    "edit.to": "To (s, optional)",
    "edit.hint": "0 keeps the source, 1 replaces it. With a time range, the strengths apply inside it and everything outside is kept.",
    "guide.legend": "Timeline guides",
    "guide.add": "Add a guide",
    "guide.at": "At (s)",
    "guide.visual": "Image or clip",
    "guide.audio": "Audio",
    "guide.hint": "Pin a picture, a clip or audio to a moment; a negative time counts back from the end. Base weights only: no Turbo, editing or upscale.",
    "up.scale": "Latent upscale",
    "up.off": "Off",
    "up.refine": "Refine strength (0 = off)",
    "state.cancelling": "cancelling",
    "job.server_queued": " · waiting on the server",
    "job.cancelling": " · stopping at the next step"
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
    "sidecar.title": "重現紀錄",
    "sidecar.copy": "複製",
    "sidecar.copied": "已複製",
    "sidecar.missing": "這個檔案旁邊沒有參數紀錄。",
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
    "align.hint": "會自動加在最前面：Picture 1 (from Shot 1) → 0.00 秒；Picture 2 (from Shot {shot}) → {secs} 秒",
    "lock.toggle": "以這段音訊驅動影片（保留為音軌）",
    "long.toggle": "長片（超過 15 秒）",
    "long.mode": "長片模式",
    "long.continuation": "續寫：分視窗生成，最長 300 秒",
    "long.full": "整段：一次生成整支，最長 30 秒",
    "long.window": "視窗（格）",
    "long.overlap": "重疊（格）",
    "long.prompts": "逐段 prompt（選填，以一行 --- 分隔）",
    "long.hint.cont": "共 {f} 格，切成 {n} 個視窗。逐段 prompt 請給 {n} 段，或留空沿用同一段 prompt。",
    "long.hint.full": "共 {f} 格，一次取樣。記憶體用量隨整支片長成長。",
    "edit.toggle": "編輯既有片段（latent 遮罩）",
    "edit.video": "來源影片（.mp4 或 .mov）",
    "edit.audio": "來源音訊（選填，否則用影片本身的音軌）",
    "edit.vstrength": "畫面：重新生成 0 到 1",
    "edit.astrength": "聲音：重新生成 0 到 1",
    "edit.from": "起點（秒，選填）",
    "edit.to": "終點（秒，選填）",
    "edit.hint": "0 保留來源，1 完全取代。指定時間範圍時，強度只套用在範圍內，範圍外一律保留。",
    "guide.legend": "Timeline guides",
    "guide.add": "新增 guide",
    "guide.at": "位置（秒）",
    "guide.visual": "圖片或短片",
    "guide.audio": "音訊",
    "guide.hint": "把圖片、短片或音訊釘在某個時間點；負數表示從結尾倒數。只能用原始權重，不能搭配 Turbo、編輯或放大。",
    "up.scale": "Latent 放大",
    "up.off": "關閉",
    "up.refine": "精修強度（0 為關閉）",
    "state.cancelling": "取消中",
    "job.server_queued": " · 在伺服器端等待",
    "job.cancelling": " · 會在下一步停下"
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
// null, not ["t2va"]: a t2va-only server matched the initial value, so the
// mode list was never filled and every submit named an empty task.
let TASKS = null;
// The cost model's coefficients arrive over /api/status. They used to be
// written out twice, once here and once in the server, which is exactly how
// the two drift apart.
let EST = {fixed: 0, per_mpxs: 11.88, per_mpxs_step: 6.05, per_mpxs2_step: 0.88};
let LORA = null, PINNED = 0, LOCKED = false, ADAPTER = null;
// What the steps box held before the adapter took it over, so unticking gives
// the number back rather than leaving the adapter's behind.
let STEPS_BEFORE = null;
let CONSTRAINTS = "";
// Request features the server side offers, from /api/status.
let FEATURES = {}, LONG_MAX = {full: 30, continuation: 300}, DUR_MAX = 15;
let CONT = {window: 277, overlap: 22}, GUIDE_MAX = 8;

// The same grid the server plans on: frames snap up to 17n+5, and a
// continuation window after the first starts window - overlap frames on.
const snapFrames = secs => { const r = Math.max(Math.round(secs * 24), 5); return r + ((5 - r) % 17 + 17) % 17; };
function contWindows(frames) {
  const w = +$("long-window").value || CONT.window, o = +$("long-overlap").value || CONT.overlap;
  return frames <= w ? 1 : 1 + Math.ceil((frames - w) / Math.max(w - o, 1));
}
const longMode = () => $("task").value === "ref2va" ? $("long-mode").value : "full";
const longOn = () => !!FEATURES.long && $("long").checked;

function dims() {
  const p = $("preset").value;
  if (p === "auto") return [null, null];
  if (p === "custom") return [+$("width").value, +$("height").value];
  const [w, h] = p.split("x");
  return [+w, +h];
}
// The server's estimate_seconds(), in the browser: continuation pays attention
// per window, and an upscale or refine is the same model on the larger canvas.
function estimate() {
  const [w, h] = dims();
  const steps = +$("steps").value, dur = +$("duration").value;
  const mpx = (w || 1344) * (h || 768) / 1e6;
  const render = (s, n, area = mpx) => {
    const x = area * s;
    return EST.per_mpxs * x + EST.per_mpxs_step * x * n + EST.per_mpxs2_step * x * x * n;
  };
  let secs;
  if (longOn() && longMode() === "continuation") {
    const n = contWindows(snapFrames(dur)), span = (+$("long-window").value || CONT.window) / 24;
    secs = n * render(span, steps) - (n * span - dur) * EST.per_mpxs * mpx;
  } else {
    secs = render(dur, steps);
  }
  if (FEATURES.upscale) {
    const k = +$("upscale").value || 1, r = +$("refine").value || 0;
    if (k > 1) secs += EST.per_mpxs * mpx * (k * k - 1) * dur;
    if (r > 0) secs += render(dur, Math.max(Math.round(steps * r), 1), mpx * k * k)
                       - EST.per_mpxs * mpx * k * k * dur;
  }
  $("est").textContent = tr("est", {t: fmt(Math.max(EST.fixed + secs, 1))});
}
function syncTask() {
  const t = $("task").value;
  $("att").style.display = t === "t2va" ? "none" : "block";
  $("att-image").style.display = (t === "fl2va" || t === "ref2va") ? "block" : "none";
  $("att-audio").style.display = t === "ref2va" ? "block" : "none";
  $("att-video").style.display = t === "ref2va" ? "block" : "none";
  $("att-rule").textContent = RULES[t] ? tr(RULES[t]) : "";
  syncFeatures();
  syncAlignHint();
}

// Show the feature panels this server offers, for this mode. Continuation is
// Ref2VA's only; a long clip in any other mode renders in full.
function syncFeatures() {
  const t = $("task").value, show = (id, on) => { $(id).style.display = on ? "" : "none"; };
  show("lock-wrap", FEATURES.lock && t === "ref2va");
  show("long-wrap", FEATURES.long);
  show("long-body", longOn());
  show("long-mode-wrap", t === "ref2va");
  show("long-cont", longMode() === "continuation");
  show("edit-wrap", FEATURES.edit);
  show("edit-body", FEATURES.edit && $("edit").checked);
  show("guide-wrap", FEATURES.guides);
  show("up-row", FEATURES.upscale);
  const max = longOn() ? LONG_MAX[longMode()] : DUR_MAX;
  $("duration").max = max;
  if (+$("duration").value > max) $("duration").value = max;
  const frames = snapFrames(+$("duration").value);
  $("long-hint").textContent = longMode() === "continuation"
    ? tr("long.hint.cont", {f: frames, n: contWindows(frames)})
    : tr("long.hint.full", {f: frames});
  estimate();
}
["long", "long-mode", "edit"].forEach(id => $(id).addEventListener("change", syncFeatures));
["long-window", "long-overlap", "duration"].forEach(id => $(id).addEventListener("input", syncFeatures));
["upscale", "refine"].forEach(id => $(id).addEventListener("input", estimate));

function addGuide() {
  if ($("guides").children.length >= GUIDE_MAX) return;
  const row = document.createElement("div");
  row.className = "guide";
  row.innerHTML = `
    <div><label>${tr("guide.at")}</label><input class="g-at" type="number" step="0.1" value="0"></div>
    <div><label>${tr("guide.visual")}</label><input class="g-vis" type="file" accept="image/*,video/mp4,video/quicktime,.mp4,.mov"></div>
    <div><label>${tr("guide.audio")}</label><input class="g-aud" type="file" accept="audio/wav,audio/mpeg,audio/flac,.wav,.mp3,.flac"></div>
    <button type="button" class="x">✕</button>`;
  row.querySelector(".x").onclick = () => row.remove();
  $("guides").appendChild(row);
}
$("guide-add").onclick = addGuide;
$("task").onchange = syncTask;

// What the upstream will and will not accept, applied from /api/status. The
// signature guard matters: poll() runs every five seconds and must not stomp a
// box the user is halfway through editing.
function applyConstraints(s) {
  const sig = JSON.stringify([s.pinned_steps, s.locked_shifts, s.adapter,
                              s.est, s.duration_min, s.duration_max,
                              s.request_lora, s.features, s.long_max]);
  if (sig === CONSTRAINTS) return;
  CONSTRAINTS = sig;
  if (s.est) EST = s.est;
  FEATURES = s.features || {};
  if (s.long_max) LONG_MAX = s.long_max;
  if (s.continuation) CONT = s.continuation;
  if (s.guide_max) GUIDE_MAX = s.guide_max;
  if (s.duration_max != null) DUR_MAX = s.duration_max;

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
  syncFeatures();
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

// An object URL rather than a data URL: the preview needs no copy of the file,
// and the submit reads it once on its own.
let previewUrl = null;
$("f-image").onchange = () => {
  const f = $("f-image").files[0];
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  if (!f) { $("pv-image").style.display = "none"; return; }
  previewUrl = URL.createObjectURL(f);
  $("pv-image").src = previewUrl;
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
// Polling runs on a timer that skips its turn while the tab is hidden, and the
// queue view backs off when nothing is waiting or running. Coming back to the
// tab catches up at once rather than at the next tick.
function every(ms, fn, idleMs) {
  let timer = null;
  const tick = async () => {
    let busy = true;
    if (!document.hidden) busy = (await fn()) !== false;
    timer = setTimeout(tick, busy || !idleMs ? ms : idleMs);
  };
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    clearTimeout(timer); tick();
  });
  tick();
}
every(5000, poll);

// Every field in a sidecar is escaped on the way in. The files are written by
// this server, but anything that can reach the media directory can write one,
// and a sidecar is not a place markup should be able to run from.
const SAFE_FILE = /^[A-Za-z0-9._-]+$/;

// Old <video> elements keep fetching until collected, so they are stopped and
// emptied before the markup that holds them is replaced.
function setVideos(el, html) {
  if (el.innerHTML === html) return;
  el.querySelectorAll("video").forEach(v => {
    v.pause(); v.removeAttribute("src"); v.load();
  });
  el.innerHTML = html;
}

let HIST_HTML = null;
async function loadHistory() {
  let items;
  try { items = await (await api("/api/history")).json(); } catch (e) { return; }
  const html = items.filter(i => SAFE_FILE.test(i.file || "")).map(i => `
    <div class="card">
      <button class="del" title="${esc(tr("del.title"))}"
        onclick="delMedia('${i.file}')">✕</button>
      <video src="/media/${i.file}" muted preload="metadata"
        onclick="show('${i.file}')"></video>
      <div onclick="show('${i.file}')">${esc(i.task || "t2va")} ·
        ${esc(i.width || "auto")}×${esc(i.height || "auto")} ·
        ${esc(i.steps)} steps · seed ${esc(i.seed)}${esc(featureTags(i))}<br>${fmt(+i.elapsed || 0)}${
        Array.isArray(i.attached) && i.attached.length
          ? " · " + tr("hist.attached") + esc(i.attached.join(",")) : ""}</div>
    </div>`).join("") || `<p class="hint">${tr("hist.empty")}</p>`;
  if (html === HIST_HTML) return;
  HIST_HTML = html;
  setVideos($("hist"), html);
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
  if (shownFile === file) { setVideos($("out"), ""); shownFile = null; }
  loadHistory();
}
loadHistory();

// The sidecar is the point of the sidecar: a clip is only reproducible if the
// numbers that produced it are in front of you when you decide you want it
// again. Reading it meant opening a file on the server before this, which is
// exactly the friction that stops anyone doing it.
async function show(file, deep) {
  if (!SAFE_FILE.test(file || "")) return;
  shownFile = file;
  setVideos($("out"), `<video src="/media/${file}" controls autoplay></video>
    <p class="meta">${file} · <a href="/media/${file}" download>${tr("link.download")}</a> ·
      <a href="#" onclick="delMedia('${file}');return false">${tr("link.delete")}</a></p>
    <details class="sidecar" id="sidecar"${deep ? " open" : ""}></details>`);
  let text;
  try {
    const r = await api("/media/" + encodeURIComponent(file) + ".json");
    if (!r.ok) throw new Error(r.status);
    text = JSON.stringify(await r.json(), null, 2);
  } catch (e) {
    text = null;
  }
  // A late arrival must not overwrite a clip the reader has since moved on to.
  if (shownFile !== file) return;
  const panel = $("sidecar");
  if (!panel) return;
  if (text === null) {
    panel.innerHTML = `<summary>${tr("sidecar.title")}</summary>
      <p class="meta">${tr("sidecar.missing")}</p>`;
    return;
  }
  panel.innerHTML = `<summary>${tr("sidecar.title")}
      <button onclick="copySidecar(event)">${tr("sidecar.copy")}</button>
    </summary><pre>${esc(text)}</pre>`;
}

function copySidecar(ev) {
  ev.preventDefault();
  ev.stopPropagation();
  const pre = $("sidecar").querySelector("pre");
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).then(() => {
    ev.target.textContent = tr("sidecar.copied");
    setTimeout(() => { ev.target.textContent = tr("sidecar.copy"); }, 1600);
  });
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

const STATE_TAG = {
  queued: ["", "state.queued"], running: ["run", "state.running"],
  done: ["", "state.done"], failed: ["bad", "state.failed"],
  cancelled: ["", "state.cancelled"], cancelling: ["run", "state.cancelling"]
};
const ACTIVE = ["queued", "running", "cancelling"];

// What a result asked for beyond the plain render, for the card and queue.
function featureTags(p) {
  const tags = [];
  if (p.lora) tags.push("turbo");
  if (p.long_video) tags.push("long:" + p.long_video.mode);
  if (p.audio_mode) tags.push("lock");
  if (p.edit) tags.push("edit");
  if (p.guides) tags.push("guides " + p.guides.length);
  if (p.latent_upscale) tags.push("↑" + p.latent_upscale + "×");
  if (p.latent_refine) tags.push("refine " + p.latent_refine);
  return tags.length ? " · " + tags.join(" · ") : "";
}

// null until the first poll: without it, every job already finished before the
// page opened would be treated as newly done and yank the player around.
let seenDone = null;

let QUEUE_HTML = null;
async function loadQueue() {
  let jobs;
  try { jobs = await (await api("/api/jobs")).json(); } catch (e) { return true; }

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

  const pending = jobs.filter(j => ACTIVE.includes(j.state));
  const shown = pending.length ? pending
    : jobs.slice(0, 3);   // nothing waiting: keep the last few for context
  let html = shown.map(j => {
    const [cls, key] = STATE_TAG[j.state] || ["", null];
    const text = key ? tr(key) : j.state;
    const p = j.params || {};
    let sub = esc(`${p.task || "t2va"} · ${p.steps} steps · ${p.duration}s · seed ${p.seed}`
                  + featureTags(p));
    if (j.state === "running" || j.state === "cancelling") {
      sub += tr("job.progress",
                {done: fmt(j.elapsed || 0), est: fmt(j.estimate || 0)});
      if (j.state === "cancelling") sub += tr("job.cancelling");
      else if (j.remote_status === "queued") sub += tr("job.server_queued");
    }
    else if (j.state === "queued") sub += tr("job.position", {n: j.position || "?"});
    else if (j.state === "done") sub += ` · ${fmt(j.elapsed || 0)}`;
    else if (j.state === "failed") sub = esc((j.error || "").split("\n")[0]).slice(0, 120);
    const done = j.state === "done" && SAFE_FILE.test(j.file || "");
    const final = done || j.state === "failed" || j.state === "cancelled";
    return `<div class="q${done ? " click" : ""}"${
        done ? ` onclick="show('${j.file}')"` : ""}>
      <span class="tag ${cls}">${text}</span>
      <span class="txt">
        <span class="line">${esc(p.description || p.prompt) || tr("job.noprompt")}</span>
        <span class="line sub">${sub}</span></span>
      ${j.state === "queued" || (j.state === "running" && j.cancellable)
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
    html += `<p class="hint" style="margin-top:12px">
      <a href="#" onclick="forgetFinished();return false">${
        tr("forget.all", {n: finished.length})}</a></p>`;
  }
  // Rewriting identical markup every two seconds swallowed clicks on Cancel
  // that landed mid-rewrite.
  if (html !== QUEUE_HTML) { QUEUE_HTML = html; $("queue").innerHTML = html; }
  return pending.length > 0;
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
    const {error} = await r.json().catch(() => ({error: "HTTP " + r.status}));
    $("stat").className = "status err"; $("stat").textContent = error;
  }
  loadQueue();
}

every(2000, loadQueue, 10000);

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

  // Whatever goes wrong below, the button comes back. A failed read or a
  // dropped connection used to leave it disabled until a reload.
  try {
    const attachments = {};
    if ($("f-image").files[0] && $("att-image").style.display !== "none")
      attachments.image = await toDataUrl($("f-image").files[0]);
    if ($("f-audio").files[0] && $("att-audio").style.display !== "none")
      attachments.audio = await toDataUrl($("f-audio").files[0]);
    if ($("f-video").files.length && $("att-video").style.display !== "none")
      attachments.videos = await Promise.all([...$("f-video").files].map(toDataUrl));

    // Each panel only sends what it is showing, so a hidden or unticked one
    // never rides along on a request it was not meant for.
    const extras = {};
    if (FEATURES.lock && $("task").value === "ref2va" && $("lock").checked)
      extras.lock_audio = true;
    if (longOn()) {
      const prompts = $("long-prompts").value.split(/^\s*---\s*$/m)
        .map(p => p.trim()).filter(Boolean);
      extras.long_video = {mode: longMode()};
      if (longMode() === "continuation") Object.assign(extras.long_video, {
        window: +$("long-window").value || null,
        overlap: +$("long-overlap").value || null, prompts});
    }
    if (FEATURES.edit && $("edit").checked) {
      const num = id => $(id).value.trim() === "" ? null : +$(id).value;
      extras.edit = {video: num("edit-vs") ?? 1, audio: num("edit-as") ?? 1,
                     start: num("edit-from"), end: num("edit-to")};
      if ($("f-source-video").files[0])
        attachments.source_video = await toDataUrl($("f-source-video").files[0]);
      if ($("f-source-audio").files[0])
        attachments.source_audio = await toDataUrl($("f-source-audio").files[0]);
    }
    if (FEATURES.guides) {
      const files = [], guides = [];
      for (const row of $("guides").children) {
        const entry = {at: +row.querySelector(".g-at").value || 0};
        for (const [cls, slot] of [[".g-vis", "visual"], [".g-aud", "audio"]]) {
          const f = row.querySelector(cls).files[0];
          if (f) { entry[slot] = files.length; files.push(await toDataUrl(f)); }
        }
        if ("visual" in entry || "audio" in entry) guides.push(entry);
      }
      if (guides.length) { extras.guides = guides; attachments.guide_files = files; }
    }
    if (FEATURES.upscale) {
      if (+$("upscale").value > 1) extras.latent_upscale = +$("upscale").value;
      if (+$("refine").value > 0) extras.latent_refine = +$("refine").value;
    }

    if ($("rand").checked) rollSeed();

    // A locked schedule rejects a request that names either shift, so the page
    // must not send what it is not showing.
    const shifts = $("flow-wrap").style.display === "none" ? {} :
      {flow_shift: +$("flow").value, audio_flow_shift: +$("aflow").value};
    const turbo = !!(LORA && $("turbo").checked);
    // An empty seed box is "draw one", not the 0 that +"" would make it.
    const seed = $("seed").value.trim() === "" ? null : +$("seed").value;

    $("stat").textContent = tr("stat.sending");
    const res = await api("/api/generate", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        task: $("task").value, ...text, width, height,
        steps: +$("steps").value, duration: +$("duration").value,
        fps: +$("fps").value, ...shifts, turbo, seed, ...extras, attachments
      })
    });
    const {position, error} = await res.json()
      .catch(() => ({error: "HTTP " + res.status}));
    if (error) {
      $("stat").className = "status err"; $("stat").textContent = error;
      return;
    }
    $("stat").className = "status";
    $("stat").textContent = position > 1
      ? tr("stat.queued", {n: position - 1})
      : tr("stat.next");
    loadQueue();
  } catch (e) {
    $("stat").className = "status err"; $("stat").textContent = String(e);
  } finally {
    $("go").disabled = false;
  }
};

// Open one clip straight from the address bar. A render worth talking about is
// worth linking to, and the link has to carry the reproduction record with it
// rather than an instruction to go and find the right card.
if (location.hash.startsWith("#file=")) {
  const wanted = decodeURIComponent(location.hash.slice("#file=".length));
  if (/^[A-Za-z0-9._-]+$/.test(wanted)) show(wanted, true);
}
</script>
</body>
</html>
"""
INDEX_BYTES = INDEX_HTML.encode("utf-8")
INDEX_ETAG = '"' + hashlib.sha256(INDEX_BYTES).hexdigest()[:16] + '"'


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    for index, upstream in enumerate(API_BASES):
        threading.Thread(target=worker_loop, args=(upstream,), daemon=True,
                         name=f"JobQueue-{index}").start()
        threading.Thread(target=status_prober_loop, args=(upstream,),
                         daemon=True, name=f"StatusProbe-{index}").start()
    # flush: stdout is block buffered when redirected to a log, and these
    # lines used to be lost whenever the process was killed rather than
    # stopped.
    say = lambda text: print(text, flush=True)
    say(f"H3 UI  ->  http://{UI_HOST}:{UI_PORT}")
    say(f"upstream: {', '.join(f'{b} ({BACKEND_KIND[b]})' for b in API_BASES)}"
        f"  auth: {'on' if API_KEY else 'off'}")
    say(f"partition: {PARTITION['partition']}  tasks: {PARTITION['tasks']}"
        f"  contract: {SERVER_CONTRACT}")
    if VLLM_FEATURES:
        on = [name for name, flag in (
            ("async jobs", ASYNC_JOBS), ("preencode", PREENCODE_MP4),
            ("timeline guides", TIMELINE_GUIDES),
            ("latent upscale", LATENT_UPSCALE)) if flag]
        say(f"features: long video, edit, {', '.join(on) or 'nothing opt in'}")
    if REQUEST_LORA:
        say(f"request lora: {REQUEST_LORA['name']}  steps {REQUEST_LORA['steps']}"
            f"  shifts {REQUEST_LORA['flow_shift']:g}/"
            f"{REQUEST_LORA['audio_flow_shift']:g}  tasks {REQUEST_LORA['tasks']}")
    if PARTITION["adapter"]:
        say(f"adapter: {PARTITION['adapter']}  "
            f"steps pinned at {PARTITION['pinned_steps']}, shifts locked")
        if len(API_BASES) > 1:
            # Nothing on the wire advertises the fusion, so this cannot be
            # checked per box. A box started without --lora-path would accept
            # a four step request and return mush.
            say("warning: every upstream must have been started with the same "
                "adapter or checkpoint; a mixed pair fails silently")
        if len(KINDS_PRESENT) > 1:
            # The pinned number is sent as given, and the two servers count
            # the same schedule differently, so one kind will refuse it.
            say("warning: vLLM-Omni and FastVideo count a pinned schedule "
                "differently (4 and 5, or 8 and 9); one kind will refuse it")
    server = Server((UI_HOST, UI_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
