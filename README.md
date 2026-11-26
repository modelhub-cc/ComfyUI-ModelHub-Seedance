# ComfyUI-ModelHub-Seedance

ComfyUI nodes for **ByteDance Seedance 2.0 / 2.5** video generation through the
[ModelHub](https://modelhub.cc) API. Text-to-video, reference-to-video (images / video / audio) and
first/last-frame video, with real-time status, interrupt support and output that plugs straight into
ComfyUI's **Save Video** node.

## Why

- **Per-second billing.** You pay for the seconds you generate. Reference video length counts toward billed duration.
- **Failed or moderated jobs are not charged.**
- **Photorealistic character references are accepted** as reference images.
- **No hard-coded endpoint.** Base URL is configurable (`MODELHUB_BASE_URL` or the Config node).
- **Model list is live.** The model dropdown is read from `GET /v1/models` at startup, with a built-in fallback list when offline.

## Install

**ComfyUI-Manager / Comfy Registry:** search for `ModelHub Seedance` and install.

**Comfy CLI:**

```bash
comfy node install modelhub-seedance
```

**Manual:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/modelhub-cc/ComfyUI-ModelHub-Seedance
pip install -r ComfyUI-ModelHub-Seedance/requirements.txt   # only needs `requests`
```

Restart ComfyUI. Nodes appear under **ModelHub/Seedance**.

## API key

Get a key at [modelhub.cc](https://modelhub.cc). The key is looked up in this order:

1. `api_key` field of the optional **ModelHub Config** node (*not recommended*: anything typed into a
   widget is saved in the workflow JSON and embedded in saved PNG/video metadata).
2. Environment variable **`MODELHUB_API_KEY`** (recommended):
   ```bash
   export MODELHUB_API_KEY=sk-...        # macOS / Linux
   set MODELHUB_API_KEY=sk-...           # Windows cmd
   python main.py
   ```
3. A key file: `ComfyUI/user/modelhub/api_key.txt`, or `modelhub_api_key.txt` in this node's folder
   (git-ignored). One line, the key only.

The generation node has no key widget, so shared workflows never contain your key unless you
type it into the Config node yourself.

## Nodes

### ModelHub Seedance Video

Submits one job, polls until done (progress bar follows the task's reported progress), downloads the `.mp4` into `ComfyUI/output/ModelHub/`.

| Input | Notes |
|---|---|
| `model` | `seedance-2.0`, `seedance-2.0-fast`, `seedance-2.0-mini`, `seedance-2.5`, `seedance-2.5-flash` (live list, Seedance models only) |
| `prompt` | Text prompt |
| `ratio` | `9:16`, `16:9`, `1:1`, `3:4`, `4:3`, `21:9`, `auto`, `adaptive` (support varies by model) |
| `resolution` | `480p` … `4k` (support varies by model; `seedance-2.5-flash` is 480p/720p only) |
| `duration` | Seconds. Seedance 2.0 family: 4–15 s; Seedance 2.5 and 2.5 Flash: 4–30 s |
| `generate_audio` | Native audio track. Not sent for `seedance-2.5-flash` (no such parameter). Some models may still return an audio track when this is off |
| `function_mode` | Used only when references are given. `omni_reference`: images/videos/audio as references. `first_last_frames`: exactly two images (first frame, last frame), no video/audio; output size follows the input images rather than `ratio`. With only one image the job is sent as `omni_reference` |
| `variation` | Not sent to the API. See *Caching* below |
| `image_1` … `image_9` | Optional IMAGE inputs, uploaded with the request |
| `video_1` … `video_3` | Optional VIDEO inputs (e.g. **Load Video**), uploaded with the request (`omni_reference` only) |
| `audio_1` … `audio_3` | Optional AUDIO inputs (e.g. **Load Audio**), uploaded as WAV (`omni_reference` only) |
| `image_urls` / `video_urls` / `audio_urls` | Optional public https URLs (or `data:` URIs), one per line |
| `config` | Optional ModelHub Config |
| `filename_prefix` | Default `ModelHub/seedance` → `output/ModelHub/seedance_<task_id>.mp4` |
| `poll_timeout_s` | How long to wait before giving up polling (the task keeps running) |

Outputs: `video` (ComfyUI `VIDEO`, connect to **Save Video**, **Get Video Components** for frames, etc.),
`video_path`, `video_url`, `task_id`.

Invalid combinations (e.g. 20 s on a 15 s model, `1080p` on `seedance-2.5-flash`) are rejected
**locally, before anything is submitted**.

### ModelHub Seedance Fetch Task

Polls an existing `task_id` and downloads the result. Never creates a job. Use it after a poll
timeout, a ComfyUI interrupt, or a restart.

### ModelHub Config

Optional: override base URL, request timeout, or key.

## Reference media

Images, videos and audio can all be used as references (`omni_reference`). Two ways for each,
and you can mix them:

1. **Node inputs**: `image_1..9` (IMAGE, e.g. **Load Image**; each frame is encoded as JPEG and every
   frame of a batch counts as one image), `video_1..3` (VIDEO, e.g. **Load Video**) and `audio_1..3`
   (AUDIO, e.g. **Load Audio**; encoded as 16-bit WAV). They are uploaded directly with the generation
   request (`multipart/form-data`). No external hosting needed.
2. **URLs**: public `https://` links in `image_urls` / `video_urls` / `audio_urls` (one per line).
   The API rejects `data:` URIs, so the node decodes any `data:` URI you paste and uploads it as a file.

Numbering sent to the API: node inputs first (`image_1` with all its frames, …), then the URLs.

Per-model limits for `omni_reference` (checked locally before submitting):

| Model | Images | Videos | Audio |
|---|---|---|---|
| `seedance-2.0`, `-fast`, `-mini` | 9 | 3 | 3 |
| `seedance-2.5` | 30 | 10 | 10 |
| `seedance-2.5-flash` | 10 | 5 | 5 |

The server may also cap the combined number of references per request. The length of reference
videos is added to the billed seconds (rounded up to whole seconds).

## Reliability & billing safety

- **Submit is never retried automatically.** If the submit request times out or the server
  returns 5xx, the outcome is unknown (the job may exist). The node stops with an error saying so;
  check your ModelHub task history before re-queueing.
- Polling (read-only) retries transient network errors. A task ends as `completed`, `failed`
  (reason from the API is shown) or `cancelled`; failed and cancelled tasks stop the node
  immediately (failed tasks are not charged). An unknown or expired `task_id` stops with the API's
  "task not found or expired" error.
- **Interrupt** (the ComfyUI *Cancel* button) stops waiting within ~0.5 s. The job keeps running on
  ModelHub; recover it with *Fetch Task* and the `task_id` printed in the console.
- **Caching:** ComfyUI only re-runs a node when its inputs change. Re-queueing an unchanged
  workflow returns the cached video and does **not** pay again. Change `variation` to
  deliberately generate a new take.

## Verified against the live API

The request/response handling (statuses, error fields, reference limits) follows ModelHub's
API contract and is covered by mocked unit tests. The live end-to-end run below only exercised
text-to-video, one IMAGE reference and Fetch Task; VIDEO/AUDIO inputs, cancelled tasks and the
reference limits have not been run against the live API yet.

Tested end to end on 2026-11-25 by calling the node functions directly (outside the ComfyUI UI)
with a real key, `seedance-2.0-mini`, 480p, 4 s, 9:16:

- Text-to-video: job submitted, polled to `completed`, mp4 downloaded to the output folder
  (h264 496×864, 4.07 s, plus an AAC audio track).
- IMAGE input (`image_1` from a JPEG): sent as multipart, the reference person and scene were kept in the output.
- Fetch Task on a finished `task_id`: downloaded the same file without creating a new job.
- `generate_audio = false` still returned an audio track on this model.
- The API reported `progress: 0` until completion, so the progress bar is a time-based estimate
  until the task reports more.

Not yet tested inside a running ComfyUI UI (Save Video wiring, preview, Cancel button).

## Example workflows

`example_workflows/` (also listed in ComfyUI's template browser for installed nodes):

- `modelhub_seedance_text_to_video.json` — text → video → Save Video
- `modelhub_seedance_reference_image.json` — reference image URL → video → Save Video (or connect **Load Image** to `image_1` instead)

The example files were saved with `image_1..4` slots only; ComfyUI adds the newer inputs
(`image_5..9`, `video_*`, `audio_*`) when it loads them.


## Environment variables

| Variable | Purpose |
|---|---|
| `MODELHUB_API_KEY` | API key |
| `MODELHUB_BASE_URL` | Default `https://modelhub.cc/v1` |
| `MODELHUB_OFFLINE` | `1` = don't fetch `/v1/models` at startup, use built-in list |

## Development

```bash
pip install requests pytest numpy pillow
pytest -q
```

Tests mock all HTTP; they never call the real API.

## License

MIT
