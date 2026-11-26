import json
import os

import numpy as np
import pytest

from conftest import FakeResponse, FakeSession
from modelhub_seedance import models, nodes
from modelhub_seedance.client import ModelHubAuthError, ModelHubClient, UnknownOutcomeError

LIVE_SAMPLE = {
    "data": [
        {"id": "seedance-2.5", "display_name": "Seedance 2.5", "media_type": "video",
         "parameters": [
             {"name": "ratio", "accepted_values": ["16:9", "9:16"], "default": "9:16"},
             {"name": "resolution", "accepted_values": ["480p", "720p", "1080p", "4K"], "default": "720p"},
             {"name": "duration", "accepted_values": [str(i) for i in range(4, 31)], "default": "5"},
             {"name": "generate_audio", "accepted_values": ["true", "false"], "default": "true"},
         ]},
        {"id": "seedance-2.5-flash", "media_type": "video",
         "parameters": [{"name": "ratio", "accepted_values": ["9:16"]}]},
        {"id": "some-image-model", "media_type": "image", "parameters": []},
        {"id": "wan-3.0", "media_type": "video", "parameters": []},
    ]
}


# ---------------------------------------------------------------- catalogue
def test_fetch_models_filters_video_and_normalizes():
    s = FakeSession().add("GET", "/models", FakeResponse(200, LIVE_SAMPLE))
    out = models.fetch_models(client=ModelHubClient(session=s))
    assert [m["id"] for m in out] == ["seedance-2.5", "seedance-2.5-flash"]
    assert out[0]["generate_audio"] is True and out[1]["generate_audio"] is False
    assert out[0]["defaults"]["ratio"] == "9:16"
    assert "Authorization" not in s.calls[0][2]["headers"]  # public endpoint


def test_get_models_live_then_cached(monkeypatch):
    monkeypatch.delenv("MODELHUB_OFFLINE")
    s = FakeSession().add("GET", "/models", FakeResponse(200, LIVE_SAMPLE))
    c = ModelHubClient(session=s)
    assert models.get_models(client=c)[0]["id"] == "seedance-2.5"
    models.get_models(client=c)
    assert len(s.calls) == 1
    assert models.cache_source() == "live"


def test_get_models_falls_back_on_error(monkeypatch):
    monkeypatch.delenv("MODELHUB_OFFLINE")
    s = FakeSession().add("GET", "/models", FakeResponse(500, None, text="down"))
    out = models.get_models(client=ModelHubClient(session=s))
    assert out == models.FALLBACK_MODELS
    assert models.cache_source() == "fallback"


def test_validate_options_matches_case_and_ranges():
    m = models.find_model("seedance-2.5", models.FALLBACK_MODELS)
    assert models.validate_options(m, "9:16", "4k", 30)["resolution"] == "4K"
    with pytest.raises(ValueError, match="duration 31s"):
        models.validate_options(m, "9:16", "720p", 31)
    flash = models.find_model("seedance-2.5-flash", models.FALLBACK_MODELS)
    with pytest.raises(ValueError, match="resolution"):
        models.validate_options(flash, "9:16", "1080p", 5)
    with pytest.raises(ValueError, match="ratio"):
        models.validate_options(flash, "auto", "720p", 5)


def test_union_dedupes_case_insensitively():
    r = models.union("resolution", models.FALLBACK_MODELS, models.RESOLUTION_ORDER)
    assert r == ["480p", "720p", "1080p", "2k", "4k"]


# ---------------------------------------------------------------- nodes
def test_node_input_types_are_valid():
    it = nodes.ModelHubSeedanceVideo.INPUT_TYPES()
    req = it["required"]
    assert "seedance-2.0" in req["model"][0]
    assert req["model"][1]["default"] in req["model"][0]
    assert req["ratio"][1]["default"] == "9:16"
    assert "4k" in req["resolution"][0] and "4K" not in req["resolution"][0]
    assert all(f"image_{i}" in it["optional"] for i in range(1, nodes.MAX_IMAGE_INPUTS + 1))
    assert nodes.MAX_IMAGE_INPUTS > 4  # not hard-coded to 4 any more
    assert all(it["optional"][f"video_{i}"][0] == "VIDEO" for i in range(1, nodes.MAX_VIDEO_INPUTS + 1))
    assert all(it["optional"][f"audio_{i}"][0] == "AUDIO" for i in range(1, nodes.MAX_AUDIO_INPUTS + 1))
    assert req["duration"][1]["min"] == 4
    assert not [m for m in req["model"][0] if not m.startswith("seedance-")]
    assert it["hidden"]["unique_id"] == "UNIQUE_ID"
    # API key must never be a required widget of the generation node
    assert "api_key" not in req and "api_key" not in it["optional"]
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)


def test_example_workflow_references_known_nodes():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(here, "example_workflows")
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    assert files
    for f in files:
        wf = json.load(open(os.path.join(folder, f)))
        for n in wf["nodes"]:
            if n["type"].startswith("ModelHub"):
                assert n["type"] in nodes.NODE_CLASS_MAPPINGS
        assert "sk-" not in json.dumps(wf)


class _FakeClientFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self, config):
        key = nodes._get_key((config or {}).get("api_key"))
        return ModelHubClient(api_key=key, session=self.session)


def _run_generate(monkeypatch, tmp_path, session, **kw):
    monkeypatch.setenv("MODELHUB_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MODELHUB_API_KEY", "sk-env")
    monkeypatch.setattr(nodes, "_client_from_config", _FakeClientFactory(session))
    monkeypatch.setattr("modelhub_seedance.client.time.sleep", lambda s: None)
    args = dict(model="seedance-2.5", prompt="a red fox in snow", ratio="16:9", resolution="4k", duration=8,
                generate_audio=True, function_mode="omni_reference")
    args.update(kw)
    return nodes.ModelHubSeedanceVideo().generate(**args)


def _ok_session():
    return (FakeSession()
            .add("POST", "/videos/generations", FakeResponse(200, {"task_id": "abc/123", "status": "processing"}))
            .add("GET", "/videos/tasks/",
                 FakeResponse(200, {"task_id": "abc/123", "status": "processing", "progress": 50}),
                 FakeResponse(200, {"task_id": "abc/123", "status": "completed", "progress": 100,
                                    "result": {"url": "https://cdn.example/out.mp4"}, "creditsConsumed": 60}))
            .add("GET", "cdn.example", FakeResponse(200, content=b"mp4data")))


def test_generate_end_to_end(monkeypatch, tmp_path):
    s = _ok_session()
    out = _run_generate(monkeypatch, tmp_path, s, image_urls="https://img/1.png\nhttps://img/2.png")
    video, path, url, task_id = out["result"]
    assert task_id == "abc/123" and url == "https://cdn.example/out.mp4"
    assert path == str(tmp_path / "ModelHub" / "seedance_abc_123.mp4")
    assert open(path, "rb").read() == b"mp4data"
    assert out["ui"]["images"][0] == {"filename": "seedance_abc_123.mp4", "subfolder": "ModelHub", "type": "output"}
    sent = [c for c in s.calls if c[0] == "POST"][0][2]["json"]
    assert sent["resolution"] == "4K" and sent["duration"] == 8 and sent["generate_audio"] is True
    assert sent["functionMode"] == "omni_reference"
    assert sent["image_file_1"] == "https://img/1.png" and sent["image_file_2"] == "https://img/2.png"
    assert [c for c in s.calls if c[0] == "POST"][0][2]["headers"]["Authorization"] == "Bearer sk-env"


def test_generate_omits_audio_flag_for_model_without_audio(monkeypatch, tmp_path):
    s = _ok_session()
    _run_generate(monkeypatch, tmp_path, s, model="seedance-2.5-flash", resolution="720p", ratio="9:16")
    sent = [c for c in s.calls if c[0] == "POST"][0][2]["json"]
    assert "generate_audio" not in sent and "functionMode" not in sent


def test_generate_tensor_images_sent_as_multipart_jpeg(monkeypatch, tmp_path):
    s = _ok_session()
    img = np.random.rand(2, 8, 8, 3).astype("float32")  # batch of 2 -> two images
    rgba = np.random.rand(1, 8, 8, 4).astype("float32")
    _run_generate(monkeypatch, tmp_path, s, image_1=img, image_3=rgba, image_urls="https://img/x.png")
    kw = [c for c in s.calls if c[0] == "POST"][0][2]
    names = [name for name, _ in kw["files"]]
    assert names == ["image_file_1", "image_file_2", "image_file_3"]
    for _, (fname, data, mime) in kw["files"]:
        assert fname.endswith(".jpg") and mime == "image/jpeg" and data[:2] == b"\xff\xd8"
    assert kw["data"]["image_file_4"] == "https://img/x.png"
    assert kw["data"]["functionMode"] == "omni_reference"
    assert kw["data"]["generate_audio"] == "true" and kw["data"]["resolution"] == "4K"


def test_generate_data_uri_sent_as_multipart_part(monkeypatch, tmp_path):
    s = _ok_session()
    _run_generate(monkeypatch, tmp_path, s, image_urls="data:image/png;base64,iVBORw0K\nhttps://img/2.png")
    kw = [c for c in s.calls if c[0] == "POST"][0][2]
    assert kw["files"] == [("image_file_1", ("image_1.png", b"\x89PNG\r\n", "image/png"))]
    assert kw["data"]["image_file_2"] == "https://img/2.png"
    assert "data:" not in json.dumps(kw["data"])


def test_generate_malformed_data_uri_fails_before_submit(monkeypatch, tmp_path):
    s = _ok_session()
    with pytest.raises(ValueError, match="base64"):
        _run_generate(monkeypatch, tmp_path, s, image_urls="data:image/png;base64,@@@@")
    assert not s.calls


def test_split_urls_keeps_data_uri_commas():
    assert nodes._split_urls("https://a/1.png, https://a/2.png\ndata:image/png;base64,AAAA") == [
        "https://a/1.png", "https://a/2.png", "data:image/png;base64,AAAA"]


class _FakeVideo:
    """Stands in for comfy_api VideoFromFile: exposes get_stream_source()."""

    def __init__(self, src):
        self.src = src

    def get_stream_source(self):
        return self.src


class _SaveOnlyVideo:
    def save_to(self, path, **kw):
        with open(path, "wb") as fh:
            fh.write(b"saved-mp4")


def test_generate_video_and_audio_inputs_sent_as_multipart(monkeypatch, tmp_path):
    import io
    import wave

    s = _ok_session()
    clip = tmp_path / "ref.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    audio = {"waveform": np.zeros((1, 2, 800), dtype="float32"), "sample_rate": 16000}
    _run_generate(monkeypatch, tmp_path, s, video_1=_FakeVideo(str(clip)), video_2=_FakeVideo(io.BytesIO(b"mem")),
                  video_3=_SaveOnlyVideo(), video_urls="https://v/4.mp4", audio_1=audio,
                  audio_urls="https://s/2.mp3")
    kw = [c for c in s.calls if c[0] == "POST"][0][2]
    files = dict(kw["files"])
    assert files["video_file_1"] == ("video_1.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")
    assert files["video_file_2"] == ("video_2.mp4", b"mem", "video/mp4")
    assert files["video_file_3"][1] == b"saved-mp4"
    assert kw["data"]["video_file_4"] == "https://v/4.mp4"
    name, wav_bytes, mime = files["audio_file_1"]
    assert name.endswith(".wav") and mime == "audio/wav"
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 16000, 800)
    assert kw["data"]["audio_file_2"] == "https://s/2.mp3"
    assert kw["data"]["functionMode"] == "omni_reference"


def test_generate_over_model_limit_fails_before_submit(monkeypatch, tmp_path):
    s = _ok_session()
    urls = "\n".join(f"https://v/{i}.mp4" for i in range(4))  # 2.0 family: max 3 videos
    with pytest.raises(ValueError, match="at most 3 reference videos"):
        _run_generate(monkeypatch, tmp_path, s, model="seedance-2.0", resolution="720p", video_urls=urls)
    assert not s.calls


def test_generate_first_last_single_image_goes_omni(monkeypatch, tmp_path):
    s = _ok_session()
    _run_generate(monkeypatch, tmp_path, s, function_mode="first_last_frames", image_urls="https://img/1.png")
    sent = [c for c in s.calls if c[0] == "POST"][0][2]["json"]
    assert sent["functionMode"] == "omni_reference"


def test_generate_cancelled_task_raises(monkeypatch, tmp_path):
    from modelhub_seedance.client import TaskCancelledError

    s = (FakeSession()
         .add("POST", "/videos/generations", FakeResponse(200, {"task_id": "c1", "status": "submitted"}))
         .add("GET", "/videos/tasks/", FakeResponse(200, {"task_id": "c1", "status": "cancelled",
                                                          "error": "任务已取消"})))
    with pytest.raises(TaskCancelledError, match="任务已取消"):
        _run_generate(monkeypatch, tmp_path, s)
    assert len([c for c in s.calls if c[0] == "GET"]) == 1


def test_generate_reports_snake_case_credits(monkeypatch, tmp_path, capsys):
    s = (FakeSession()
         .add("POST", "/videos/generations", FakeResponse(200, {"task_id": "k1", "status": "submitted"}))
         .add("GET", "/videos/tasks/", FakeResponse(200, {"task_id": "k1", "status": "completed",
                                                          "result": {"url": "https://cdn.example/o.mp4"},
                                                          "credits_consumed": 42}))
         .add("GET", "cdn.example", FakeResponse(200, content=b"mp4")))
    _run_generate(monkeypatch, tmp_path, s)
    assert "task credits 42" in capsys.readouterr().out


def test_progress_uses_api_progress_field():
    p = nodes._Progress(None, expected_s=100)
    seen = []
    p.bar = type("Bar", (), {"update_absolute": lambda self, v, t: seen.append(v)})()
    p.update({"status": "processing", "progress": 42}, elapsed=1)   # API ahead of estimate
    p.update({"status": "processing", "progress": 0}, elapsed=50)   # API stuck at 0 -> estimate
    p.update({"status": "processing"}, elapsed=500)                 # missing -> estimate, capped
    p.update({"status": "completed", "progress": 0}, elapsed=60)
    assert seen == [42, 50, 95, 100]


def test_generate_invalid_option_fails_before_submit(monkeypatch, tmp_path):
    s = _ok_session()
    with pytest.raises(ValueError, match="duration"):
        _run_generate(monkeypatch, tmp_path, s, model="seedance-2.0", resolution="720p", duration=20)
    assert not s.calls


def test_generate_unknown_outcome_surfaces_and_does_not_poll(monkeypatch, tmp_path):
    s = FakeSession().add("POST", "/videos/generations", FakeResponse(504, None, text="Gateway Timeout"))
    with pytest.raises(UnknownOutcomeError):
        _run_generate(monkeypatch, tmp_path, s)
    assert len(s.calls) == 1


def test_missing_key_is_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_key_file_candidates", lambda: [str(tmp_path / "none.txt")])
    with pytest.raises(ModelHubAuthError, match="MODELHUB_API_KEY"):
        nodes._client_from_config(None)


def test_config_node_key_overrides_env(monkeypatch):
    monkeypatch.setenv("MODELHUB_API_KEY", "sk-env")
    (cfg,) = nodes.ModelHubConfig().make("https://example.test/v1/", api_key=" sk-node ")
    c = nodes._client_from_config(cfg)
    assert c.api_key == "sk-node" and c.base_url == "https://example.test/v1"
    (cfg2,) = nodes.ModelHubConfig().make("", api_key="")
    assert nodes._client_from_config(cfg2).api_key == "sk-env"


def test_fetch_task_node_never_posts(monkeypatch, tmp_path):
    s = _ok_session()
    monkeypatch.setenv("MODELHUB_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MODELHUB_API_KEY", "sk-env")
    monkeypatch.setattr(nodes, "_client_from_config", _FakeClientFactory(s))
    monkeypatch.setattr("modelhub_seedance.client.time.sleep", lambda s: None)
    out = nodes.ModelHubSeedanceFetchTask().fetch(" t-9 ", filename_prefix="mine/clip")
    assert out["result"][1] == str(tmp_path / "mine" / "clip_t-9.mp4")
    assert not [c for c in s.calls if c[0] == "POST"]
