"""Shared fixtures + the .env contract for credentialed tests.

Test tiers (markers registered in pyproject.toml):

  unit         — no network, no credentials, no browser. Always runnable:
                     uv run pytest -m "not credentialed"
  credentialed — integration (deployment API) and e2e (Playwright/Firefox
                 driving the published UI). These REQUIRE a git-ignored
                 `.env` at the repo root — see MISSING_ENV_MSG below.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

MISSING_ENV_MSG = f"""
################################################################################
Credentialed tests (integration + e2e) need an env file that does not exist:

    {ENV_PATH}

Create it (it is git-ignored — NEVER commit these values) with:

    CREWAI_DEPLOYMENT_URL=https://<your-deployment>.crewai.com
    CREWAI_DEPLOYMENT_TOKEN=<bearer token from the deployment's AMP page>
    OPENAI_API_KEY=sk-...        # only needed by mic-related tests
    # optional: UI_URL=<override for the published UI under test>

These are the same three values the UI's Settings panel asks for — both are
shown on the deployment's page in CrewAI AMP.

To run only the credential-free unit tests instead:

    uv run pytest -m "not credentialed"
################################################################################
"""


def _load_env() -> None:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env()


@pytest.fixture(scope="session")
def deployment() -> dict:
    url = os.environ.get("CREWAI_DEPLOYMENT_URL", "").rstrip("/")
    token = os.environ.get("CREWAI_DEPLOYMENT_TOKEN", "")
    if not url or not token:
        pytest.fail(MISSING_ENV_MSG, pytrace=False)
    return {"url": url, "token": token}


@pytest.fixture(scope="session")
def ui_url() -> str:
    return os.environ.get(
        "UI_URL", "https://crewaiinc-fde.github.io/crewai-audio-quickstart/"
    ).rstrip("/") + "/"


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args: dict, browser_name: str) -> dict:
    """Fake microphone per engine, so mic-path tests run headless anywhere.

    Firefox is the suite default (strictest MediaRecorder — ogg). Chromium
    covers the Chrome path (webm/opus):
        uv run pytest -m credentialed -o addopts="" --browser chromium
    (`-o addopts=""` clears the default --browser firefox; passing both
    browsers runs every test on each.)"""
    if browser_name == "firefox":
        return {
            **browser_type_launch_args,
            "firefox_user_prefs": {
                "media.navigator.streams.fake": True,
                "media.navigator.permission.disabled": True,
            },
        }
    if browser_name == "chromium":
        return {
            **browser_type_launch_args,
            "args": [
                *browser_type_launch_args.get("args", []),
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ],
        }
    return browser_type_launch_args


# -- tiny deployment client (stdlib only, mirrors client/ask.py) --------------

def _ask():
    spec = importlib.util.spec_from_file_location(
        "audio_quickstart_client_ask", REPO_ROOT / "client" / "ask.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPO_ROOT / 'client' / 'ask.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _request(url: str, token: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if payload is not None else "GET",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode() or "{}")


def run_turn(dep: dict, message: str, session_id: str | None = None,
             timeout_s: int = 240) -> tuple[str, str]:
    """One AMP chat turn → SSE. Returns (session_id, result)."""
    ask = _ask()
    url, token = dep["url"], dep["token"]
    if not session_id:
        started = _request(f"{url}/chat/start", token, {})
        session_id = str(started.get("session_id") or started.get("id") or "")
        if not session_id:
            raise AssertionError(f"POST /chat/start returned no session_id: {started}")
    sent = _request(f"{url}/chat/{session_id}/message", token,
                    {"message": message, "stream": True})
    kid = sent.get("kickoff_id") or sent.get("id")
    if not kid:
        raise AssertionError(f"POST /chat/.../message returned no kickoff_id: {sent}")
    deadline = time.monotonic() + timeout_s
    sse_url = (f"{url}/chat/{session_id}/stream/events"
               "?events=*&last_event_id=0-0")
    last_409 = ""
    resp = None
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            sse_url, method="GET",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "text/event-stream"},
        )
        try:
            resp = urllib.request.urlopen(
                req, timeout=max(1.0, deadline - time.monotonic())
            )
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code == 409:
                last_409 = detail
                time.sleep(0.05)
                continue
            raise AssertionError(f"{exc.code} from {sse_url}\n{detail}") from exc
    if resp is None:
        raise AssertionError(
            f"timed out attaching to chat SSE for {kid}"
            + (f"\nLast 409: {last_409}" if last_409 else "")
        )
    try:
        outcome, text = ask.consume_sse_frames(resp.readline, deadline)
    finally:
        resp.close()
    if outcome == "failed":
        raise AssertionError(f"execution {kid} failed: {text}")
    if outcome == "timeout":
        raise AssertionError(f"timed out waiting for execution {kid}")
    if text.strip():
        return session_id, text
    history = _request(f"{url}/chat/{session_id}/history", token)
    for msg in reversed(history.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            return session_id, str(msg["content"])
    raise AssertionError(f"turn {kid} ended with no assistant text")
