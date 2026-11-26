"""Minimal ModelHub video API client.

Pure Python + ``requests``; no ComfyUI imports so it can be unit-tested
in isolation.

API facts this client relies on (verified against the live service):

* Base URL ``https://modelhub.cc/v1``, auth ``Authorization: Bearer <key>``.
* ``GET  /models``                    -> ``{"data": [{id, display_name, media_type, parameters}]}`` (public)
* ``POST /videos/generations``        -> flat ``{"task_id", "status", "requested_model", "effective_model", ...}``
* ``GET  /videos/tasks/{task_id}``    -> flat ``{"task_id", "status", "progress", "creditsConsumed",
  "result": {"url"}, "error", ...}``. Public statuses: ``processing`` (``progress`` stays 0 until the end),
  ``completed`` (``result.url``), ``failed`` (reason in top-level ``error`` string, plus ``error_code`` /
  ``failureType``; not charged) and ``cancelled`` (top-level ``error``). Unknown/expired task:
  HTTP 404 ``{"error": {"message", "code": "-2011"}}``. The billing field is ``creditsConsumed`` live;
  the public docs example says ``credits_consumed`` -- both are read.
* Errors: HTTP 4xx with ``{"error": {"message", "type", "code"}}`` (not charged).
  A ``{"code": 0, "data": {...}}`` wrapper (older behaviour) is still tolerated.
* Reference media are numbered fields ``image_file_1..N`` / ``video_file_1..N``
  / ``audio_file_1..N`` together with ``functionMode``. Each one is either a
  public http(s) URL (string) or a **local file sent as a multipart part**.
  ``data:`` URIs are rejected by the server, so this client decodes them and
  uploads them as multipart parts instead. An ``image_files`` array is
  silently ignored by the server, so it is never sent.
* ``first_last_frames`` takes at most 2 images and no video/audio; some
  channels require exactly 2, so a single image is sent as ``omni_reference``.
* ``omni_reference`` limits per model (images / videos / audios): Seedance 2.0
  family 9/3/3, Seedance 2.5 30/10/10, Seedance 2.5 Flash 10/5/5. The server
  may additionally cap the combined count per channel.

Billing safety: a submit that times out or returns 5xx has an *unknown*
outcome (the job may have been created and may be charged). Such submits are
never retried automatically; :class:`UnknownOutcomeError` is raised instead.
"""
from __future__ import annotations

import base64
import binascii
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

DEFAULT_BASE_URL = "https://modelhub.cc/v1"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ComfyUI-ModelHub-Seedance/0.1"
)

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
CANCELLED_STATUSES = (STATUS_CANCELLED, "canceled")
TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED) + CANCELLED_STATUSES

FUNCTION_MODES = ("omni_reference", "first_last_frames")


class ModelHubError(RuntimeError):
    """Base error for everything raised by this client."""


class ModelHubAuthError(ModelHubError):
    """Missing or rejected API key."""


class ModelHubAPIError(ModelHubError):
    """The API answered and rejected the request (code != 0 or 4xx)."""

    def __init__(self, message: str, status_code: Optional[int] = None, code: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class UnknownOutcomeError(ModelHubError):
    """Submit timed out / got a 5xx: the job may or may not exist.

    Never resubmit automatically. Check your ModelHub task history first.
    """


class TaskFailedError(ModelHubError):
    """The task reached status ``failed`` (failed jobs are not charged)."""

    def __init__(self, message: str, task_id: str):
        super().__init__(message)
        self.task_id = task_id


class TaskCancelledError(TaskFailedError):
    """The task reached status ``cancelled``."""


class PollTimeoutError(ModelHubError):
    """Polling gave up before the task finished. The task may still complete."""

    def __init__(self, message: str, task_id: str):
        super().__init__(message)
        self.task_id = task_id


class InterruptedError_(ModelHubError):
    """Raised when the caller-supplied interrupt check fires."""


def resolve_api_key(explicit: Optional[str] = None, key_file: Optional[str] = None) -> str:
    """Return an API key from (in order) the explicit value, ``MODELHUB_API_KEY``,
    or a key file. Raises :class:`ModelHubAuthError` if none is found."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("MODELHUB_API_KEY", "").strip()
    if env:
        return env
    if key_file and os.path.isfile(key_file):
        with open(key_file, "r", encoding="utf-8") as fh:
            value = fh.read().strip()
        if value:
            return value
    raise ModelHubAuthError(
        "No ModelHub API key found. Set the MODELHUB_API_KEY environment variable "
        "before starting ComfyUI, or put the key in the key file "
        "(see README: 'API key')."
    )


# (images, videos, audios) accepted by ``omni_reference``, per model family.
_REF_LIMITS_2_0 = (9, 3, 3)
_REF_LIMITS_2_5 = (30, 10, 10)
_REF_LIMITS_2_5_FLASH = (10, 5, 5)


def reference_limits(model: str) -> Optional[Tuple[int, int, int]]:
    """``(images, videos, audios)`` limits for ``omni_reference``, or None if unknown."""
    m = (model or "").lower()
    if m.startswith("seedance-2.5-flash"):
        return _REF_LIMITS_2_5_FLASH
    if m.startswith("seedance-2.5"):
        return _REF_LIMITS_2_5
    if m.startswith("seedance-2.0"):
        return _REF_LIMITS_2_0
    return None


_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
        "video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm",
        "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
        "audio/mp4": "m4a", "audio/aac": "aac", "audio/ogg": "ogg"}


def data_uri_to_part(uri: str, stem: str) -> Tuple[str, bytes, str]:
    """Decode a ``data:<mime>;base64,...`` URI into a ``(filename, bytes, mime)`` multipart part."""
    header, sep, body = uri.partition(",")
    if not sep or not header.startswith("data:"):
        raise ValueError("malformed data: URI")
    meta = header[5:].split(";")
    mime = meta[0] or "application/octet-stream"
    try:
        raw = base64.b64decode(body, validate=True) if "base64" in meta[1:] else body.encode("utf-8")
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"malformed base64 in data: URI ({exc})") from exc
    if not raw:
        raise ValueError("empty data: URI")
    return f"{stem}.{_EXT.get(mime.lower(), 'bin')}", raw, mime


def task_credits(data: Dict[str, Any]) -> Any:
    """Credits charged for a task: live field ``creditsConsumed``, docs field ``credits_consumed``."""
    value = data.get("creditsConsumed")
    return data.get("credits_consumed") if value is None else value


def task_error_reason(data: Dict[str, Any], default: str = "unknown reason") -> str:
    """Failure / cancel reason: top-level ``error`` (string or ``{message}``), then ``result.error``."""
    err = data.get("error")
    reason = err.get("message") if isinstance(err, dict) else err
    if not reason:
        result = data.get("result")
        reason = result.get("error") if isinstance(result, dict) else None
    reason = reason or data.get("message") or default
    extras = [f"{k}={data[k]}" for k in ("failureType", "error_code") if data.get(k) not in (None, "")]
    return f"{reason}" + (f" ({', '.join(extras)})" if extras else "")


def build_generation_payload(
    model: str,
    prompt: str,
    ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    duration: Optional[float] = None,
    generate_audio: Optional[bool] = None,
    function_mode: Optional[str] = None,
    image_urls: Sequence[Any] = (),
    video_urls: Sequence[str] = (),
    audio_urls: Sequence[str] = (),
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON body for ``POST /videos/generations``.

    ``functionMode`` is only sent when reference media are present. If media
    are present and no mode is given, ``omni_reference`` is used.

    ``image_urls`` / ``video_urls`` / ``audio_urls`` items may be http(s)
    URL strings, ``data:`` URIs (decoded into file parts, because the server
    rejects data URIs) or ``(filename, bytes, mime)`` tuples for local files;
    file parts make :meth:`ModelHubClient.submit` send multipart/form-data.

    ``first_last_frames`` with a single image is sent as ``omni_reference``
    (some channels require exactly 2 images). Reference counts are checked
    against the per-model ``omni_reference`` limits before anything is sent.
    """
    if not model:
        raise ValueError("model is required")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    payload: Dict[str, Any] = {"model": model, "prompt": prompt}
    if ratio:
        payload["ratio"] = ratio
    if resolution:
        payload["resolution"] = resolution
    if duration is not None:
        d = float(duration)
        payload["duration"] = int(d) if d.is_integer() else d
    if generate_audio is not None:
        payload["generate_audio"] = bool(generate_audio)

    def clean(items: Sequence[Any], kind: str) -> List[Any]:
        out: List[Any] = []
        for item in items:
            if isinstance(item, tuple):
                out.append(item)
                continue
            url = (item or "").strip()
            if not url:
                continue
            if url.startswith("data:"):
                out.append(data_uri_to_part(url, f"{kind}_{len(out) + 1}"))
            elif url.startswith("https://") or url.startswith("http://"):
                out.append(url)
            else:
                raise ValueError(f"Reference {kind} must be a public http(s) URL, got: {url[:80]!r}")
        return out

    images = clean(image_urls, "image")
    videos = clean(video_urls, "video")
    audios = clean(audio_urls, "audio")

    if images or videos or audios:
        mode = function_mode or "omni_reference"
        if mode not in FUNCTION_MODES:
            raise ValueError(f"functionMode must be one of {FUNCTION_MODES}, got {mode!r}")
        if mode == "first_last_frames":
            if videos or audios:
                raise ValueError("first_last_frames does not take reference video/audio; use omni_reference")
            if len(images) > 2:
                raise ValueError(f"first_last_frames takes at most 2 images (first, last), got {len(images)}")
            if len(images) < 2:
                # Some channels reject first_last_frames with a single image (400);
                # a lone first frame works as an omni_reference image.
                mode = "omni_reference"
        if mode == "omni_reference":
            limits = reference_limits(model)
            if limits:
                for kind, got, cap in zip(("images", "videos", "audios"),
                                          (len(images), len(videos), len(audios)), limits):
                    if got > cap:
                        raise ValueError(f"{model} accepts at most {cap} reference {kind} "
                                         f"(omni_reference), got {got}")
        payload["functionMode"] = mode
        for i, url in enumerate(images, 1):
            payload[f"image_file_{i}"] = url
        for i, url in enumerate(videos, 1):
            payload[f"video_file_{i}"] = url
        for i, url in enumerate(audios, 1):
            payload[f"audio_file_{i}"] = url
    if extra:
        payload.update(extra)
    return payload


class ModelHubClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    # ------------------------------------------------------------------ utils
    def _headers(self, auth: bool = True) -> Dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if auth:
            if not self.api_key:
                raise ModelHubAuthError("API key is required for this request")
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _json(resp: requests.Response) -> Dict[str, Any]:
        try:
            body = resp.json()
        except ValueError:
            raise ModelHubAPIError(
                f"Non-JSON response (HTTP {resp.status_code}): {resp.text[:300]}",
                status_code=resp.status_code,
            )
        if not isinstance(body, dict):
            raise ModelHubAPIError(f"Unexpected response shape: {str(body)[:300]}", status_code=resp.status_code)
        return body

    @staticmethod
    def _unwrap(body: Dict[str, Any]) -> Dict[str, Any]:
        """Return the task object from a flat body or a ``{code, data}`` wrapper."""
        data = body.get("data")
        if isinstance(data, dict) and ("task_id" in data or "status" in data or "id" in data):
            return data
        return body

    @staticmethod
    def _is_error(resp: requests.Response, body: Dict[str, Any]) -> bool:
        if resp.status_code >= 400 or body.get("error"):
            return True
        code = body.get("code")
        if code is None or code in (0, "0"):
            return False
        # Flat success bodies carry task_id at the top level; a non-zero code
        # without one is a legacy-style error.
        return "task_id" not in body

    @staticmethod
    def _error_code(body: Dict[str, Any]) -> Any:
        err = body.get("error")
        if isinstance(err, dict) and err.get("code") is not None:
            return err.get("code")
        return body.get("code")

    @staticmethod
    def _error_message(body: Dict[str, Any]) -> str:
        msg = None
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
        elif isinstance(err, str):
            msg = err
        msg = msg or body.get("message") or body.get("msg")
        return str(msg or body)[:500]

    # ----------------------------------------------------------------- models
    def list_models(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """``GET /models`` (public; no key needed)."""
        resp = self.session.get(
            f"{self.base_url}/models", headers=self._headers(auth=False), timeout=timeout or self.timeout
        )
        resp.raise_for_status()
        body = self._json(resp)
        data = body.get("data")
        if not isinstance(data, list):
            raise ModelHubAPIError("models response has no 'data' list")
        return data

    # ----------------------------------------------------------------- submit
    @staticmethod
    def _to_multipart(payload: Dict[str, Any]) -> Tuple[Dict[str, str], List[Tuple[str, Tuple[str, bytes, str]]]]:
        fields: Dict[str, str] = {}
        files: List[Tuple[str, Tuple[str, bytes, str]]] = []
        for k, v in payload.items():
            if isinstance(v, tuple):
                files.append((k, v))
            elif isinstance(v, bool):
                fields[k] = "true" if v else "false"
            else:
                fields[k] = str(v)
        return fields, files

    def submit(self, payload: Dict[str, Any]) -> str:
        """Submit a generation job and return its ``task_id``.

        JSON body normally; multipart/form-data when the payload contains local
        files (``(filename, bytes, mime)`` tuples).

        Never retries. Timeouts / connection drops after sending / 5xx raise
        :class:`UnknownOutcomeError`.
        """
        url = f"{self.base_url}/videos/generations"
        headers = self._headers()
        has_files = any(isinstance(v, tuple) for v in payload.values())
        if has_files:
            fields, files = self._to_multipart(payload)
            kwargs: Dict[str, Any] = {"data": fields, "files": files}
        else:
            headers["Content-Type"] = "application/json"
            kwargs = {"json": payload}
        try:
            resp = self.session.post(url, headers=headers, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectTimeout as exc:
            # Connection was never established: the request did not reach the server.
            raise ModelHubError(f"Could not connect to ModelHub ({exc}). Nothing was submitted.") from exc
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise UnknownOutcomeError(
                "Submit request failed without a definite answer "
                f"({type(exc).__name__}). The job MAY have been created. "
                "Check your ModelHub task history before submitting again."
            ) from exc

        if resp.status_code >= 500:
            raise UnknownOutcomeError(
                f"ModelHub returned HTTP {resp.status_code} on submit; the job MAY have been created. "
                "Check your ModelHub task history before submitting again. "
                f"Body: {resp.text[:300]}"
            )
        if resp.status_code in (401, 403):
            raise ModelHubAuthError(f"API key rejected (HTTP {resp.status_code}): {resp.text[:300]}")
        body = self._json(resp)
        if self._is_error(resp, body):
            raise ModelHubAPIError(
                f"Submit rejected: {self._error_message(body)}",
                status_code=resp.status_code,
                code=self._error_code(body),
            )
        data = self._unwrap(body)
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise UnknownOutcomeError(
                f"Submit returned success but no task_id; the job MAY exist. Body: {str(body)[:300]}"
            )
        return str(task_id)

    # ------------------------------------------------------------------- poll
    def get_task(self, task_id: str) -> Dict[str, Any]:
        """``GET /videos/tasks/{id}`` -> the task object (flat, or unwrapped from ``data``)."""
        resp = self.session.get(
            f"{self.base_url}/videos/tasks/{task_id}", headers=self._headers(), timeout=self.timeout
        )
        if resp.status_code in (401, 403):
            raise ModelHubAuthError(f"API key rejected (HTTP {resp.status_code})")
        if resp.status_code >= 500:
            resp.raise_for_status()
        body = self._json(resp)
        data = self._unwrap(body)
        # A failed/cancelled task answers 200 with a top-level ``error`` string:
        # that is the task's failure reason, not a rejected query.
        is_task = resp.status_code < 400 and "status" in data
        if not is_task and self._is_error(resp, body):
            raise ModelHubAPIError(
                f"Task query rejected: {self._error_message(body)}",
                status_code=resp.status_code,
                code=self._error_code(body),
            )
        if "status" not in data:
            raise ModelHubAPIError(f"Task response has no status: {str(body)[:300]}")
        return data

    def wait(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
        on_status: Optional[Callable[[Dict[str, Any], float], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        max_transient_errors: int = 10,
    ) -> Dict[str, Any]:
        """Poll until the task completes. Returns the final ``data`` dict.

        ``failed`` raises :class:`TaskFailedError`, ``cancelled`` raises
        :class:`TaskCancelledError` immediately; an unknown/expired task (404)
        raises :class:`ModelHubAPIError`.

        Polling is read-only, so transient network / 5xx errors are retried
        (up to ``max_transient_errors`` in a row).
        """
        sleep = sleep or time.sleep
        start = clock()
        transient = 0
        while True:
            if check_interrupt and check_interrupt():
                raise InterruptedError_(
                    f"Interrupted while waiting for task {task_id}. The task keeps running on "
                    "ModelHub; use the 'ModelHub Seedance Fetch Task' node with this task_id to get the result."
                )
            try:
                data = self.get_task(task_id)
                transient = 0
            except (requests.exceptions.RequestException, ) as exc:
                transient += 1
                if transient > max_transient_errors:
                    raise ModelHubError(f"Polling task {task_id} failed repeatedly: {exc}") from exc
                data = None
            elapsed = clock() - start
            if data is not None:
                status = str(data.get("status", "")).lower()
                if on_status:
                    on_status(data, elapsed)
                if status == STATUS_COMPLETED:
                    url = (data.get("result") or {}).get("url")
                    if not url:
                        raise ModelHubError(f"Task {task_id} completed but has no result.url: {str(data)[:300]}")
                    return data
                if status == STATUS_FAILED:
                    raise TaskFailedError(
                        f"Task {task_id} failed: {task_error_reason(data)} (failed jobs are not charged)", task_id
                    )
                if status in CANCELLED_STATUSES:
                    raise TaskCancelledError(
                        f"Task {task_id} was cancelled: {task_error_reason(data, 'task cancelled')}", task_id
                    )
            if elapsed >= timeout:
                raise PollTimeoutError(
                    f"Task {task_id} still not finished after {int(elapsed)}s. It may still complete; "
                    "use the 'ModelHub Seedance Fetch Task' node with this task_id later.",
                    task_id,
                )
            # Sleep in small slices so interrupts are noticed quickly.
            remaining = poll_interval
            while remaining > 0:
                step = min(0.5, remaining)
                sleep(step)
                remaining -= step
                if check_interrupt and check_interrupt():
                    break

    # --------------------------------------------------------------- download
    def download(self, url: str, dest_path: str, chunk_size: int = 1 << 20) -> str:
        """Stream ``url`` to ``dest_path`` (atomic rename). Returns ``dest_path``."""
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        tmp = dest_path + ".part"
        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        with self.session.get(url, headers=headers, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
        if os.path.getsize(tmp) == 0:
            os.remove(tmp)
            raise ModelHubError(f"Downloaded video is empty: {url}")
        os.replace(tmp, dest_path)
        return dest_path
