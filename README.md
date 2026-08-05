# h3-ui

A thin browser frontend for video generation on a [vLLM-Omni](https://github.com/vllm-project/vllm-omni) server.

One file, standard library only, no build step. Point it at a running server and open a browser.

It exists because talking to `/v1/videos/sync` by hand is tedious: multipart bodies, base64 attachments, an API key you don't want in your shell history, and generations long enough that a dropped connection loses the result. This sits in front of that and keeps the key server-side.

## What it does

- **Text, first-frame, and reference conditioning** — task list is read from the served checkpoint, so the UI matches whatever partition is loaded rather than offering options the server will reject.
- **Attachment rules enforced before submitting** — each task states what it accepts, and the form refuses mismatched combinations instead of letting the server 400.
- **Jobs survive the page** — generation runs server-side against a job id, so closing the tab or losing Wi-Fi doesn't kill a 10-minute render.
- **Seeds are always recorded** — every result writes a sidecar JSON with the exact parameters. A good result is reproducible by pasting its seed back.
- **Random seeds** — 🎲 draws one on demand; the checkbox draws a fresh one per generation. Either way the drawn value is written into the box and shown on completion, so nothing is lost to chance you can't recover.
- **History** — past generations are listed with their parameters and play inline.

Requests are serialised behind a lock: one generation at a time, which is what a single-GPU box wants anyway.

## Requirements

- Python 3.10+ (standard library only — nothing to install)
- A reachable vLLM-Omni server serving a video model

## Setup

```sh
cp .env.example .env
$EDITOR .env          # set H3_API_BASE, and H3_API_KEY if the server needs one
python3 server.py
```

If you run the [MiniMax-H3-DGX-Spark](https://github.com/joeynyc/MiniMax-H3-DGX-Spark) deployment repo, its `.env` is picked up automatically and a local one is optional.

Configuration resolves in this order: process environment → `.env` → default.

| Variable | Default | Meaning |
|---|---|---|
| `H3_API_BASE` | `http://127.0.0.1:8000` | vLLM-Omni base URL (a trailing `/v1` is stripped) |
| `H3_API_KEY` | *(empty)* | Sent as `Authorization: Bearer` when set |
| `H3_UI_HOST` | `127.0.0.1` | UI bind address |
| `H3_UI_PORT` | `8080` | UI port |
| `H3_UI_ENV_FILE` | *(auto)* | Explicit path to a `.env` |

**On binding beyond loopback:** this process holds your API key and will proxy anything a browser asks it to. Binding to a LAN address hands that to everyone on the network — do it deliberately, on a network you trust.

## Scope

The server contract is vLLM-Omni's `POST /v1/videos/sync`. The **task vocabulary** (`t2va`, `fl2va`, `ref2va`) and the partition probe that reads `_minimax_h3` from `model_index.json` are MiniMax-H3 specific — a different video model on the same server needs those two touched, but nothing about the transport or the job handling changes. There is no dependency on DGX Spark or any particular GPU.

Image generation is **not** implemented. vLLM-Omni does expose `/v1/images/generations` and `/v1/images/edits`, so adding it is a matter of a task mode and a shorter result pane rather than new plumbing — whether it produces anything depends on the model actually loaded.

## Reproducing a result

Every `media/*.mp4` has a `.mp4.json` beside it:

```json
{
  "task": "t2va",
  "prompt": "Rain on a window at night, soft patter.",
  "width": 768, "height": 448,
  "steps": 20, "duration": 2.0, "fps": 24,
  "flow_shift": 12.0, "audio_flow_shift": 3.0,
  "seed": 43538620,
  "elapsed": 118.6
}
```

Paste those values back into the form — with the random checkbox off — and you get the same video.

## A note on long generations

vLLM-Omni bounds the wait for a finished step's background copy with `_ASYNC_OUTPUT_TIMEOUT`, which upstream sets to 30 s. If a single denoise step runs longer than that, the output future is cancelled and the server's result-pump thread dies — after which `/health` keeps answering 200 while no request ever returns again. High step counts on long durations reach that easily: 50 steps at 4 s ran 44–48 s per step here.

Reported as [vllm-project/vllm-omni#5821](https://github.com/vllm-project/vllm-omni/issues/5821), with a build-time fix in [joeynyc/MiniMax-H3-DGX-Spark#4](https://github.com/joeynyc/MiniMax-H3-DGX-Spark/pull/4). Worth patching before you push step counts up; this frontend can't work around it.

## License

MIT
