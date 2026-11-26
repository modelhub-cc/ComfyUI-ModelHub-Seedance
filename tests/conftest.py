import json
import os

import pytest
import requests

os.environ.setdefault("MODELHUB_OFFLINE", "1")


class FakeResponse:
    def __init__(self, status=200, body=None, content=b"", text=None):
        self.status_code = status
        self._body = body
        self._content = content
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    """Queue of responses (or exceptions) per (method, url-substring)."""

    def __init__(self):
        self.routes = []  # (method, needle, [responses])
        self.calls = []

    def add(self, method, needle, *responses):
        self.routes.append((method, needle, list(responses)))
        return self

    def _dispatch(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for m, needle, queue in self.routes:
            if m == method and needle in url and queue:
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kw):
        return self._dispatch("GET", url, **kw)

    def post(self, url, **kw):
        return self._dispatch("POST", url, **kw)


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MODELHUB_API_KEY", raising=False)
    monkeypatch.setenv("MODELHUB_OFFLINE", "1")
    from modelhub_seedance import models

    models.reset_cache()
    yield
    models.reset_cache()
