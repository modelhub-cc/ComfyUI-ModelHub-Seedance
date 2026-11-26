import base64

import pytest
import requests

from conftest import FakeResponse
from modelhub_seedance.client import (
    InterruptedError_,
    ModelHubAPIError,
    ModelHubAuthError,
    ModelHubClient,
    ModelHubError,
    PollTimeoutError,
    TaskCancelledError,
    TaskFailedError,
    UnknownOutcomeError,
    build_generation_payload,
    reference_limits,
    task_credits,
    resolve_api_key,
)


def client(session):
    return ModelHubClient(api_key="sk-test", session=session)


# ---------------------------------------------------------------- payload
def test_payload_text_only_has_no_function_mode():
    p = build_generation_payload("seedance-2.0", "a cat", ratio="9:16", resolution="720p", duration=5,
                                 generate_audio=True)
    assert p == {"model": "seedance-2.0", "prompt": "a cat", "ratio": "9:16", "resolution": "720p",
                 "duration": 5, "generate_audio": True}
    assert isinstance(p["duration"], int)


def test_payload_numbered_reference_fields():
    p = build_generation_payload("seedance-2.5", "x", image_urls=["https://a/1.png", " https://a/2.png "],
                                 video_urls=["https://v/1.mp4"], audio_urls=["https://s/1.mp3"])
    assert p["functionMode"] == "omni_reference"
    assert p["image_file_1"] == "https://a/1.png"
    assert p["image_file_2"] == "https://a/2.png"
    assert p["video_file_1"] == "https://v/1.mp4"
    assert p["audio_file_1"] == "https://s/1.mp3"
    assert "image_files" not in p


def test_payload_first_last_frames_rules():
    p = build_generation_payload("seedance-2.0", "x", function_mode="first_last_frames",
                                 image_urls=["https://a/1.png", "https://a/2.png"])
    assert p["functionMode"] == "first_last_frames"
    with pytest.raises(ValueError):
        build_generation_payload("seedance-2.0", "x", function_mode="first_last_frames",
                                 image_urls=["https://a/1.png"] * 3)


def test_payload_data_uri_becomes_file_part():
    # The server rejects data: URIs, so they must go out as multipart file parts.
    p = build_generation_payload("seedance-2.0", "x", image_urls=["data:image/png;base64,iVBORw0K"],
                                 video_urls=["data:video/mp4;base64,AAAAGGZ0eXA="],
                                 audio_urls=["data:audio/mpeg;base64,SUQz"])
    assert p["image_file_1"] == ("image_1.png", base64.b64decode("iVBORw0K"), "image/png")
    assert p["video_file_1"][0] == "video_1.mp4" and p["video_file_1"][2] == "video/mp4"
    assert p["audio_file_1"] == ("audio_1.mp3", b"ID3", "audio/mpeg")
    assert not any(isinstance(v, str) and v.startswith("data:") for v in p.values())


def test_payload_malformed_data_uri_is_rejected_locally():
    with pytest.raises(ValueError, match="base64"):
        build_generation_payload("seedance-2.0", "x", image_urls=["data:image/png;base64,@@@"])


def test_first_last_frames_with_one_image_falls_back_to_omni():
    p = build_generation_payload("seedance-2.0", "x", function_mode="first_last_frames",
                                 image_urls=["https://a/1.png"])
    assert p["functionMode"] == "omni_reference" and p["image_file_1"] == "https://a/1.png"


def test_first_last_frames_rejects_video_and_audio():
    with pytest.raises(ValueError, match="does not take reference video/audio"):
        build_generation_payload("seedance-2.0", "x", function_mode="first_last_frames",
                                 image_urls=["https://a/1.png", "https://a/2.png"], video_urls=["https://v/1.mp4"])
    with pytest.raises(ValueError, match="does not take reference video/audio"):
        build_generation_payload("seedance-2.0", "x", function_mode="first_last_frames",
                                 image_urls=["https://a/1.png"], audio_urls=["https://s/1.mp3"])


@pytest.mark.parametrize("model,limits", [
    ("seedance-2.0", (9, 3, 3)), ("seedance-2.0-fast", (9, 3, 3)), ("seedance-2.0-mini", (9, 3, 3)),
    ("seedance-2.5", (30, 10, 10)), ("seedance-2.5-flash", (10, 5, 5)),
])
def test_omni_reference_limits_per_model(model, limits):
    assert reference_limits(model) == limits
    imgs, vids, auds = limits
    p = build_generation_payload(model, "x", image_urls=[f"https://a/{i}.png" for i in range(imgs)],
                                 video_urls=[f"https://v/{i}.mp4" for i in range(vids)],
                                 audio_urls=[f"https://s/{i}.mp3" for i in range(auds)])
    assert f"image_file_{imgs}" in p and f"video_file_{vids}" in p and f"audio_file_{auds}" in p
    for kind, field, n in (("images", "image_urls", imgs), ("videos", "video_urls", vids),
                           ("audios", "audio_urls", auds)):
        ext = {"images": "png", "videos": "mp4", "audios": "mp3"}[kind]
        with pytest.raises(ValueError, match=f"at most {n} reference {kind}"):
            build_generation_payload(model, "x", **{field: [f"https://x/{i}.{ext}" for i in range(n + 1)]})


def test_more_than_four_images_are_numbered():
    p = build_generation_payload("seedance-2.5", "x", image_urls=[f"https://a/{i}.png" for i in range(12)])
    assert p["image_file_12"] == "https://a/11.png"


def test_payload_accepts_local_file_tuples():
    part = ("a.jpg", b"\xff\xd8jpeg", "image/jpeg")
    p = build_generation_payload("seedance-2.0", "x", image_urls=[part, "https://a/2.png"])
    assert p["image_file_1"] == part and p["image_file_2"] == "https://a/2.png"
    assert p["functionMode"] == "omni_reference"


def test_payload_rejects_bad_input():
    with pytest.raises(ValueError):
        build_generation_payload("seedance-2.0", "  ")
    with pytest.raises(ValueError):
        build_generation_payload("seedance-2.0", "x", image_urls=["/local/file.png"])
    with pytest.raises(ValueError):
        build_generation_payload("seedance-2.0", "x", image_urls=["https://a"], function_mode="bogus")


# ---------------------------------------------------------------- api key
def test_resolve_api_key_order(monkeypatch, tmp_path):
    f = tmp_path / "k.txt"
    f.write_text("from-file\n")
    assert resolve_api_key(None, str(f)) == "from-file"
    monkeypatch.setenv("MODELHUB_API_KEY", "from-env")
    assert resolve_api_key(None, str(f)) == "from-env"
    assert resolve_api_key("explicit", str(f)) == "explicit"


def test_resolve_api_key_missing():
    with pytest.raises(ModelHubAuthError):
        resolve_api_key(None, None)


# ---------------------------------------------------------------- submit
SUBMIT_OK = {"created": 1764000000, "task_id": "t1", "status": "processing",
             "requested_model": "seedance-2.0", "effective_model": "seedance-2.0"}


def test_submit_ok_sends_bearer_and_browser_ua(fake_session):
    fake_session.add("POST", "/videos/generations", FakeResponse(200, SUBMIT_OK))
    assert client(fake_session).submit({"model": "m", "prompt": "p"}) == "t1"
    _, url, kw = fake_session.calls[0]
    assert url == "https://modelhub.cc/v1/videos/generations"
    assert kw["headers"]["Authorization"] == "Bearer sk-test"
    assert kw["headers"]["User-Agent"].startswith("Mozilla/5.0")
    assert kw["json"] == {"model": "m", "prompt": "p"}


def test_submit_tolerates_legacy_wrapper(fake_session):
    fake_session.add("POST", "/videos/generations", FakeResponse(200, {"code": 0, "data": {"task_id": "t2"}}))
    assert client(fake_session).submit({}) == "t2"


def test_submit_multipart_when_local_files(fake_session):
    fake_session.add("POST", "/videos/generations", FakeResponse(200, SUBMIT_OK))
    part = ("ref.jpg", b"\xff\xd8jpeg", "image/jpeg")
    payload = build_generation_payload("seedance-2.0", "p", ratio="9:16", resolution="720p", duration=5,
                                       generate_audio=False, image_urls=[part, "https://a/2.png"])
    assert client(fake_session).submit(payload) == "t1"
    kw = fake_session.calls[0][2]
    assert "json" not in kw and "Content-Type" not in kw["headers"]  # requests sets the multipart boundary
    assert kw["files"] == [("image_file_1", part)]
    assert kw["data"] == {"model": "seedance-2.0", "prompt": "p", "ratio": "9:16", "resolution": "720p",
                          "duration": "5", "generate_audio": "false", "functionMode": "omni_reference",
                          "image_file_2": "https://a/2.png"}


def test_submit_4xx_error_object(fake_session):
    fake_session.add("POST", "/videos/generations", FakeResponse(400, {"error": {
        "message": "invalid image_file_1", "type": "invalid_request_error", "code": "-2000"}}))
    with pytest.raises(ModelHubAPIError, match="invalid image_file_1") as ei:
        client(fake_session).submit({})
    assert ei.value.status_code == 400 and ei.value.code == "-2000"


def test_submit_code_nonzero_is_api_error(fake_session):
    fake_session.add("POST", "/videos/generations", FakeResponse(200, {"code": 1001, "message": "bad ratio"}))
    with pytest.raises(ModelHubAPIError, match="bad ratio"):
        client(fake_session).submit({})


def test_submit_401_is_auth_error(fake_session):
    fake_session.add("POST", "/videos/generations", FakeResponse(401, {"message": "invalid key"}))
    with pytest.raises(ModelHubAuthError):
        client(fake_session).submit({})


@pytest.mark.parametrize("failure", [
    FakeResponse(502, None, text="Bad Gateway"),
    requests.exceptions.ReadTimeout("read timed out"),
    requests.exceptions.ConnectionError("reset by peer"),
])
def test_submit_unknown_outcome_never_retried(fake_session, failure):
    fake_session.add("POST", "/videos/generations", failure)
    with pytest.raises(UnknownOutcomeError, match="MAY have been created"):
        client(fake_session).submit({})
    assert len([c for c in fake_session.calls if c[0] == "POST"]) == 1


def test_submit_connect_timeout_is_definite_failure(fake_session):
    fake_session.add("POST", "/videos/generations", requests.exceptions.ConnectTimeout("no route"))
    with pytest.raises(ModelHubError, match="Nothing was submitted") as ei:
        client(fake_session).submit({})
    assert not isinstance(ei.value, UnknownOutcomeError)


def test_submit_success_without_task_id_is_unknown(fake_session):
    fake_session.add("POST", "/videos/generations", FakeResponse(200, {"status": "processing"}))
    with pytest.raises(UnknownOutcomeError):
        client(fake_session).submit({})


def test_submit_requires_key(fake_session):
    with pytest.raises(ModelHubAuthError):
        ModelHubClient(api_key=None, session=fake_session).submit({})


# ---------------------------------------------------------------- polling
def task(status, url=None, error=None, **extra):
    """Flat task body, as the live API returns it."""
    data = {"task_id": "t1", "status": status, "result": {"url": url, "error": error}, "created_at": 1764000000000}
    data.update(extra)
    return FakeResponse(200, data)


def test_wait_until_completed(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", task("queued"), task("processing", progress=40),
                     task("completed", url="https://media.modelhub.cc/api/proxy/video?url=x", progress=100,
                          creditsConsumed=60))
    seen = []
    data = client(fake_session).wait("t1", poll_interval=1, sleep=lambda s: None,
                                     on_status=lambda d, e: seen.append(d["status"]))
    assert data["result"]["url"].startswith("https://media.modelhub.cc/")
    assert data["creditsConsumed"] == 60
    assert seen == ["queued", "processing", "completed"]


def test_get_task_tolerates_legacy_wrapper(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", FakeResponse(200, {"code": 0, "data": {
        "status": "completed", "result": {"url": "https://cdn/v.mp4"}}}))
    assert client(fake_session).get_task("t1")["status"] == "completed"


def test_get_task_4xx_error_object(fake_session):
    fake_session.add("GET", "/videos/tasks/nope", FakeResponse(404, {"error": {"message": "task not found"}}))
    with pytest.raises(ModelHubAPIError, match="task not found"):
        client(fake_session).get_task("nope")


def test_wait_failed_raises_with_reason(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", task("failed", error="content moderation"))
    with pytest.raises(TaskFailedError, match="content moderation.*not charged"):
        client(fake_session).wait("t1", sleep=lambda s: None)


def test_wait_failed_reads_top_level_error(fake_session):
    # Live shape: reason in top-level `error`, plus error_code / failureType, creditsConsumed null.
    fake_session.add("GET", "/videos/tasks/t1", FakeResponse(200, {
        "task_id": "t1", "status": "failed", "error": "prompt blocked by moderation", "error_code": "-3001",
        "failureType": "blocked", "creditsConsumed": None, "result": {}}))
    with pytest.raises(TaskFailedError, match="prompt blocked by moderation.*failureType=blocked.*-3001"):
        client(fake_session).wait("t1", sleep=lambda s: None)


def test_wait_failed_reads_error_object_message(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", FakeResponse(200, {
        "task_id": "t1", "status": "failed", "error": {"message": "upstream exploded"}}))
    with pytest.raises(TaskFailedError, match="upstream exploded"):
        client(fake_session).wait("t1", sleep=lambda s: None)


@pytest.mark.parametrize("status", ["cancelled", "canceled", "CANCELLED"])
def test_wait_cancelled_is_terminal(fake_session, status):
    fake_session.add("GET", "/videos/tasks/t1", FakeResponse(200, {
        "task_id": "t1", "status": status, "error": "任务已取消", "creditsConsumed": None}))
    with pytest.raises(TaskCancelledError, match="cancelled: 任务已取消") as ei:
        client(fake_session).wait("t1", poll_interval=5, timeout=1800, sleep=lambda s: None)
    assert ei.value.task_id == "t1"
    assert isinstance(ei.value, TaskFailedError)
    assert len(fake_session.calls) == 1  # no further polling


def test_wait_cancelled_without_reason_uses_default(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", FakeResponse(200, {"task_id": "t1", "status": "cancelled"}))
    with pytest.raises(TaskCancelledError, match="task cancelled"):
        client(fake_session).wait("t1", sleep=lambda s: None)


def test_wait_unknown_task_404_stops_immediately(fake_session):
    fake_session.add("GET", "/videos/tasks/gone", FakeResponse(404, {
        "error": {"message": "任务不存在或已过期", "code": "-2011"}}))
    with pytest.raises(ModelHubAPIError, match="任务不存在或已过期") as ei:
        client(fake_session).wait("gone", sleep=lambda s: None)
    assert ei.value.status_code == 404 and ei.value.code == "-2011"
    assert len(fake_session.calls) == 1


def test_task_credits_accepts_both_spellings():
    assert task_credits({"creditsConsumed": 60}) == 60
    assert task_credits({"credits_consumed": 45}) == 45
    assert task_credits({"creditsConsumed": 0, "credits_consumed": 9}) == 0
    assert task_credits({"creditsConsumed": None}) is None


def test_wait_retries_transient_poll_errors(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", requests.exceptions.ReadTimeout("x"),
                     FakeResponse(503, None, text="busy"), task("completed", url="https://cdn/v.mp4"))
    data = client(fake_session).wait("t1", sleep=lambda s: None)
    assert data["status"] == "completed"


def test_wait_gives_up_after_many_transient_errors(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", requests.exceptions.ReadTimeout("x"))
    with pytest.raises(ModelHubError, match="failed repeatedly"):
        client(fake_session).wait("t1", sleep=lambda s: None, max_transient_errors=3)


def test_wait_timeout(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", task("processing"))
    t = {"now": 0.0}

    def sleep(s):
        t["now"] += s

    with pytest.raises(PollTimeoutError) as ei:
        client(fake_session).wait("t1", poll_interval=5, timeout=20, sleep=sleep, clock=lambda: t["now"])
    assert ei.value.task_id == "t1"


def test_wait_respects_interrupt(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", task("processing"))
    calls = {"n": 0}

    def interrupted():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(InterruptedError_, match="Fetch Task"):
        client(fake_session).wait("t1", sleep=lambda s: None, check_interrupt=interrupted)


def test_completed_without_url_is_error(fake_session):
    fake_session.add("GET", "/videos/tasks/t1", task("completed"))
    with pytest.raises(ModelHubError, match="no result.url"):
        client(fake_session).wait("t1", sleep=lambda s: None)


# ---------------------------------------------------------------- download
def test_download_streams_to_file(fake_session, tmp_path):
    fake_session.add("GET", "cdn.example", FakeResponse(200, content=b"\x00\x00\x00\x18ftypmp42" * 100))
    dest = tmp_path / "sub" / "v.mp4"
    client(fake_session).download("https://cdn.example/v.mp4", str(dest), chunk_size=64)
    assert dest.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")
    assert not (tmp_path / "sub" / "v.mp4.part").exists()
    assert fake_session.calls[0][2]["headers"]["User-Agent"].startswith("Mozilla/5.0")


def test_download_empty_is_error(fake_session, tmp_path):
    fake_session.add("GET", "cdn.example", FakeResponse(200, content=b""))
    with pytest.raises(ModelHubError):
        client(fake_session).download("https://cdn.example/v.mp4", str(tmp_path / "v.mp4"))
