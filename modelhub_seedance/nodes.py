"""ComfyUI nodes (classic ``NODE_CLASS_MAPPINGS`` API, works on old and new ComfyUI)."""
from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from . import models as mh_models
from .client import (
    DEFAULT_BASE_URL,
    FUNCTION_MODES,
    InterruptedError_,
    ModelHubClient,
    build_generation_payload,
    resolve_api_key,
    task_credits,
)

CATEGORY = "ModelHub/Seedance"
CONFIG_TYPE = "MODELHUB_CONFIG"
# Socket counts. Seedance 2.0 family accepts up to 9 images / 3 videos / 3 audios
# as omni_reference; 2.5 accepts 30/10/10 and 2.5 Flash 10/5/5. Image batches and
# the URL fields can carry more than one item per socket, so the sockets do not
# cap the total -- the per-model limits (client.reference_limits) do.
MAX_IMAGE_INPUTS = 9
MAX_VIDEO_INPUTS = 3
MAX_AUDIO_INPUTS = 3
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Optional ComfyUI integrations. Everything degrades gracefully so the module
# also imports outside ComfyUI (unit tests).
# --------------------------------------------------------------------------
try:  # pragma: no cover - only inside ComfyUI
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover
    folder_paths = None

try:  # pragma: no cover
    import comfy.model_management as model_management  # type: ignore
except Exception:  # pragma: no cover
    model_management = None

try:  # pragma: no cover
    import comfy.utils as comfy_utils  # type: ignore
except Exception:  # pragma: no cover
    comfy_utils = None

VideoFromFile = None
try:  # pragma: no cover - newer ComfyUI
    from comfy_api.input_impl import VideoFromFile  # type: ignore
except Exception:  # pragma: no cover
    try:
        from comfy_api.latest import InputImpl  # type: ignore

        VideoFromFile = InputImpl.VideoFromFile
    except Exception:
        VideoFromFile = None


def _output_dir() -> str:
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.environ.get("MODELHUB_OUTPUT_DIR") or os.path.join(tempfile.gettempdir(), "modelhub_output")


def _key_file_candidates() -> List[str]:
    paths = []
    if folder_paths is not None and hasattr(folder_paths, "get_user_directory"):
        try:
            paths.append(os.path.join(folder_paths.get_user_directory(), "modelhub", "api_key.txt"))
        except Exception:
            pass
    paths.append(os.path.join(PACK_DIR, "modelhub_api_key.txt"))
    return paths


def _get_key(explicit: Optional[str]) -> str:
    last_exc = None
    for path in _key_file_candidates() or [None]:
        try:
            return resolve_api_key(explicit, key_file=path)
        except Exception as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _interrupted() -> bool:
    if model_management is None:
        return False
    try:
        return bool(model_management.processing_interrupted())
    except Exception:
        return False


def _raise_comfy_interrupt(message: str) -> None:
    print(f"[ModelHub] {message}")
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()
    raise InterruptedError_(message)


class _Progress:
    """ProgressBar + status text. Shows the larger of the task's ``progress``
    (0-100) and a time-based estimate, because the API may stay at 0 until done."""

    def __init__(self, node_id: Optional[str], expected_s: float):
        self.node_id = node_id
        self.expected = max(30.0, expected_s)
        self.bar = None
        if comfy_utils is not None:
            try:
                self.bar = comfy_utils.ProgressBar(100)
            except Exception:
                self.bar = None
        self.server = None
        try:  # pragma: no cover
            from server import PromptServer  # type: ignore

            self.server = PromptServer.instance
        except Exception:
            self.server = None

    def text(self, msg: str) -> None:
        print(f"[ModelHub] {msg}")
        if self.server is not None and self.node_id is not None and hasattr(self.server, "send_progress_text"):
            try:
                self.server.send_progress_text(msg, self.node_id)
            except Exception:
                pass

    def update(self, data: Dict[str, Any], elapsed: float) -> None:
        status = str(data.get("status", "?"))
        # Live API often reports progress 0 until the very end, so never show
        # less than a time-based estimate (capped at 95% until completed).
        estimate = min(95.0, 100.0 * elapsed / self.expected)
        try:
            api_pct = max(0.0, min(100.0, float(data.get("progress"))))
        except (TypeError, ValueError):
            api_pct = 0.0
        pct = 100.0 if status.lower() == "completed" else max(api_pct, estimate)
        if self.bar is not None:
            try:
                self.bar.update_absolute(int(pct), 100)
            except Exception:
                pass
        self.text(f"{status} · ~{int(pct)}% · {int(elapsed)}s")

    def done(self) -> None:
        if self.bar is not None:
            try:
                self.bar.update_absolute(100, 100)
            except Exception:
                pass


def _split_urls(text: Optional[str]) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for token in re.split(r"\s+", text):
        # data: URIs contain a comma, so only split plain URLs on commas.
        out.extend([token] if token.startswith("data:") else token.split(","))
    return [u.strip() for u in out if u.strip()]


def _tensor_to_jpegs(image: Any, prefix: str, quality: int = 95) -> List[Tuple[str, bytes, str]]:
    """Encode every frame of an IMAGE tensor ([B,H,W,C] float 0..1) as JPEG,
    returned as ``(filename, bytes, mime)`` parts for a multipart upload."""
    import io

    import numpy as np  # ComfyUI dependency
    from PIL import Image

    arr = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
    if arr.ndim == 3:
        arr = arr[None]
    parts = []
    for i, frame in enumerate(arr):
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        img = Image.fromarray(frame).convert("RGB")  # drop alpha; JPEG has none
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        parts.append((f"{prefix}_{i}.jpg", buf.getvalue(), "image/jpeg"))
    return parts


def _read_file_part(path: str, prefix: str, default_mime: str) -> Tuple[str, bytes, str]:
    import mimetypes

    with open(path, "rb") as fh:
        data = fh.read()
    if not data:
        raise ValueError(f"{prefix}: file is empty: {path}")
    mime = mimetypes.guess_type(path)[0] or default_mime
    ext = os.path.splitext(path)[1] or ".mp4"
    return f"{prefix}{ext}", data, mime


def _video_to_part(video: Any, prefix: str) -> Tuple[str, bytes, str]:
    """Turn a ComfyUI VIDEO input (or a file path) into a ``(filename, bytes, mime)`` part.

    Uses ``get_stream_source()`` (path or file-like) when available, else
    ``save_to()`` into a temp mp4.
    """
    if isinstance(video, (str, os.PathLike)):
        return _read_file_part(os.fspath(video), prefix, "video/mp4")
    src = None
    if hasattr(video, "get_stream_source"):
        try:
            src = video.get_stream_source()
        except Exception:
            src = None
    if isinstance(src, (str, os.PathLike)) and os.path.isfile(os.fspath(src)):
        return _read_file_part(os.fspath(src), prefix, "video/mp4")
    if src is not None and hasattr(src, "read"):
        if hasattr(src, "seek"):
            src.seek(0)
        data = src.read()
        if data:
            return f"{prefix}.mp4", data, "video/mp4"
    if hasattr(video, "save_to"):
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            video.save_to(tmp)
            return _read_file_part(tmp, prefix, "video/mp4")
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise ValueError(f"{prefix}: unsupported VIDEO input ({type(video).__name__})")


def _audio_to_wavs(audio: Any, prefix: str) -> List[Tuple[str, bytes, str]]:
    """Encode a ComfyUI AUDIO input (``{"waveform": [B, C, T] float -1..1, "sample_rate"}``)
    as 16-bit PCM WAV parts, one per batch item."""
    import io
    import wave

    import numpy as np

    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(f"{prefix}: unsupported AUDIO input ({type(audio).__name__})")
    wav = audio["waveform"]
    arr = wav.detach().cpu().numpy() if hasattr(wav, "cpu") else np.asarray(wav)
    if arr.ndim == 1:
        arr = arr[None, None]
    elif arr.ndim == 2:
        arr = arr[None]
    parts = []
    for i, item in enumerate(arr):  # item: [C, T]
        pcm = (np.clip(item, -1.0, 1.0) * 32767.0).astype("<i2").T  # -> [T, C] interleaved
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(int(item.shape[0]))
            w.setsampwidth(2)
            w.setframerate(int(audio["sample_rate"]))
            w.writeframes(pcm.tobytes())
        parts.append((f"{prefix}_{i}.wav", buf.getvalue(), "audio/wav"))
    return parts


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80] or "task"


def _client_from_config(config: Optional[Dict[str, Any]]) -> ModelHubClient:
    config = config or {}
    key = _get_key(config.get("api_key"))
    base = config.get("base_url") or os.environ.get("MODELHUB_BASE_URL") or DEFAULT_BASE_URL
    return ModelHubClient(api_key=key, base_url=base, timeout=float(config.get("request_timeout", 60)))


def _wait_and_download(
    client: ModelHubClient,
    task_id: str,
    filename_prefix: str,
    poll_timeout_s: float,
    node_id: Optional[str],
    expected_s: float,
    submitted: bool = True,
) -> Tuple[Any, ...]:
    progress = _Progress(node_id, expected_s)
    progress.text(f"{'submitted' if submitted else 'fetching'} · task {task_id}")
    try:
        data = client.wait(
            task_id,
            poll_interval=5.0,
            timeout=poll_timeout_s,
            on_status=progress.update,
            check_interrupt=_interrupted,
        )
    except InterruptedError_ as exc:
        _raise_comfy_interrupt(str(exc))
        raise
    url = data["result"]["url"]
    progress.text("downloading video")

    prefix = (filename_prefix or "ModelHub/seedance").replace("\\", "/").strip("/")
    subfolder, _, stem = prefix.rpartition("/")
    out_dir = os.path.join(_output_dir(), subfolder)
    filename = f"{_safe(stem or 'seedance')}_{_safe(task_id)}.mp4"
    path = client.download(url, os.path.join(out_dir, filename))
    progress.done()
    credits = task_credits(data)
    progress.text(f"completed · {filename}" + (f" · task credits {credits}" if credits is not None else ""))

    video = VideoFromFile(path) if VideoFromFile is not None else path
    ui = {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}], "animated": (True,)}
    return ui, (video, path, url, task_id)


# ==========================================================================
# Nodes
# ==========================================================================
class ModelHubConfig:
    """Optional. Without this node the key comes from MODELHUB_API_KEY or the key file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Leave EMPTY (recommended): the key is read from the MODELHUB_API_KEY "
                               "environment variable or the key file. A key typed here is stored in the "
                               "workflow JSON and in every PNG/video you save with embedded workflow.",
                }),
                "request_timeout": ("INT", {"default": 60, "min": 10, "max": 600}),
            },
        }

    RETURN_TYPES = (CONFIG_TYPE,)
    RETURN_NAMES = ("config",)
    FUNCTION = "make"
    CATEGORY = CATEGORY

    def make(self, base_url: str, api_key: str = "", request_timeout: int = 60):
        return ({"base_url": base_url.strip() or DEFAULT_BASE_URL, "api_key": api_key.strip() or None,
                 "request_timeout": request_timeout},)


class ModelHubSeedanceVideo:
    """Text / image / reference-to-video through ModelHub. Submits one job, polls, downloads the mp4."""

    @classmethod
    def INPUT_TYPES(cls):
        catalogue = mh_models.get_models()
        ids = [m["id"] for m in catalogue]
        default_model = "seedance-2.0" if "seedance-2.0" in ids else ids[0]
        ratios = mh_models.union("ratio", catalogue, mh_models.RATIO_ORDER) or ["9:16"]
        resolutions = mh_models.union("resolution", catalogue, mh_models.RESOLUTION_ORDER) or ["720p"]
        optional: Dict[str, Any] = {
            "config": (CONFIG_TYPE, {"tooltip": "Optional ModelHub Config (base URL / key override)."}),
        }
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            optional[f"image_{i}"] = ("IMAGE", {
                "tooltip": "Reference image, uploaded directly with the request (multipart). "
                           "Every frame of a batch counts as one image. Limits: Seedance 2.0 family 9 images, "
                           "2.5 30, 2.5 Flash 10."})
        for i in range(1, MAX_VIDEO_INPUTS + 1):
            optional[f"video_{i}"] = ("VIDEO", {
                "tooltip": "Reference video (omni_reference only), uploaded with the request. "
                           "Its length is added to the billed seconds (rounded up). "
                           "Limits: 2.0 family 3, 2.5 10, 2.5 Flash 5."})
        for i in range(1, MAX_AUDIO_INPUTS + 1):
            optional[f"audio_{i}"] = ("AUDIO", {
                "tooltip": "Reference audio (omni_reference only), uploaded as WAV. "
                           "Limits: 2.0 family 3, 2.5 10, 2.5 Flash 5."})
        optional.update({
            "image_urls": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Public https image URLs, one per line. Sent after the IMAGE inputs."}),
            "video_urls": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Public https reference video URLs (omni_reference only), "
                                                 "sent after the VIDEO inputs. Reference video length is billed too."}),
            "audio_urls": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Public https reference audio URLs (omni_reference only), "
                                                 "sent after the AUDIO inputs."}),
            "filename_prefix": ("STRING", {"default": "ModelHub/seedance"}),
            "poll_timeout_s": ("INT", {"default": 1800, "min": 60, "max": 7200, "step": 60}),
        })
        return {
            "required": {
                "model": (ids, {"default": default_model}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "ratio": (ratios, {"default": "9:16" if "9:16" in ratios else ratios[0]}),
                "resolution": (resolutions, {"default": "720p" if "720p" in resolutions else resolutions[0]}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1,
                                     "tooltip": "Seconds. Billed per second. Seedance 2.0 family 4-15 s, "
                                                "2.5 and 2.5 Flash 4-30 s."}),
                "generate_audio": ("BOOLEAN", {"default": True,
                                               "tooltip": "Ignored for models without audio output. Note: some models still "
                                                          "return an audio track when this is off."}),
                "function_mode": (list(FUNCTION_MODES), {
                    "default": "omni_reference",
                    "tooltip": "Only used when reference media are given. omni_reference: images/videos/audio "
                               "as references. first_last_frames: exactly 2 images (first frame, last frame), no "
                               "video/audio; with only 1 image the job is sent as omni_reference."}),
                "variation": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                      "tooltip": "Not sent to the API. ComfyUI caches results by inputs, so "
                                                 "re-queueing an unchanged node does NOT pay again. Change this "
                                                 "number to deliberately generate a new take."}),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "video_url", "task_id")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Generate a Seedance video via ModelHub. Billed per second; failed or moderated "
                   "jobs are not charged.")

    def generate(self, model, prompt, ratio, resolution, duration, generate_audio, function_mode, variation=0,
                 config=None, image_urls="", video_urls="", audio_urls="", filename_prefix="ModelHub/seedance",
                 poll_timeout_s=1800, unique_id=None, **kwargs):
        # 1. Validate everything locally before any network call that costs money.
        model_info = mh_models.find_model(model)
        if model_info is None:
            raise ValueError(f"Unknown model {model!r}. Restart ComfyUI to refresh the model list.")
        opts = mh_models.validate_options(model_info, ratio, resolution, int(duration))

        images: List[Any] = []
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            tensor = kwargs.get(f"image_{i}")
            if tensor is not None:
                images.extend(_tensor_to_jpegs(tensor, f"image_{i}"))
        images += _split_urls(image_urls)
        videos: List[Any] = []
        for i in range(1, MAX_VIDEO_INPUTS + 1):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                videos.append(_video_to_part(v, f"video_{i}"))
        videos += _split_urls(video_urls)
        audios: List[Any] = []
        for i in range(1, MAX_AUDIO_INPUTS + 1):
            a = kwargs.get(f"audio_{i}")
            if a is not None:
                audios.extend(_audio_to_wavs(a, f"audio_{i}"))
        audios += _split_urls(audio_urls)

        payload = build_generation_payload(
            model=model,
            prompt=prompt,
            ratio=opts["ratio"],
            resolution=opts["resolution"],
            duration=int(duration),
            generate_audio=bool(generate_audio) if model_info.get("generate_audio") else None,
            function_mode=function_mode,
            image_urls=images,
            video_urls=videos,
            audio_urls=audios,
        )
        client = _client_from_config(config)
        if _interrupted():
            _raise_comfy_interrupt("Interrupted before submit; nothing was sent.")

        # 2. Submit exactly once (never retried; see client.submit).
        task_id = client.submit(payload)
        ui, result = _wait_and_download(client, task_id, filename_prefix, poll_timeout_s, unique_id,
                                        expected_s=60 + 12 * int(duration))
        return {"ui": ui, "result": result}


class ModelHubSeedanceFetchTask:
    """Resume a task by id (after a timeout, interrupt or ComfyUI restart). Does not submit anything."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"task_id": ("STRING", {"default": ""})},
            "optional": {
                "config": (CONFIG_TYPE,),
                "filename_prefix": ("STRING", {"default": "ModelHub/seedance"}),
                "poll_timeout_s": ("INT", {"default": 1800, "min": 0, "max": 7200, "step": 60}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "video_url", "task_id")
    OUTPUT_NODE = True
    FUNCTION = "fetch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Poll an existing ModelHub video task and download the result. Never creates a new job."

    def fetch(self, task_id, config=None, filename_prefix="ModelHub/seedance", poll_timeout_s=1800, unique_id=None):
        task_id = (task_id or "").strip()
        if not task_id:
            raise ValueError("task_id is required")
        client = _client_from_config(config)
        ui, result = _wait_and_download(client, task_id, filename_prefix, poll_timeout_s, unique_id, expected_s=120,
                                        submitted=False)
        return {"ui": ui, "result": result}


NODE_CLASS_MAPPINGS = {
    "ModelHubConfig": ModelHubConfig,
    "ModelHubSeedanceVideo": ModelHubSeedanceVideo,
    "ModelHubSeedanceFetchTask": ModelHubSeedanceFetchTask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelHubConfig": "ModelHub Config",
    "ModelHubSeedanceVideo": "ModelHub Seedance Video",
    "ModelHubSeedanceFetchTask": "ModelHub Seedance Fetch Task",
}
