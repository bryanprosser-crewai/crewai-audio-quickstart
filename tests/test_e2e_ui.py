"""End-to-end tier — Playwright driving the published UI in real Firefox
(needs .env; Firefox is the default browser via pyproject addopts because
it has the strictest MediaRecorder behavior of the majors — its default
container is audio/ogg, which is exactly what broke the mic pre-fix).

The mic test uses Firefox's fake-microphone mode (see conftest launch args),
so it verifies the recording pipeline without audio hardware or an OpenAI
call. Conversation tests type instead of talk — same kickoff path.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.credentialed

TURN_TIMEOUT_MS = 90_000

MIC_PROBE_JS = """
async () => {
  const prefs = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4',
                 'audio/ogg;codecs=opus', 'audio/wav'];
  const picked = prefs.find(m => MediaRecorder.isTypeSupported(m)) || null;
  if (!picked) return { picked: null };
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const rec = new MediaRecorder(stream, { mimeType: picked });
  const chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  const stopped = new Promise((res) => { rec.onstop = res; });
  rec.start();
  await new Promise((r) => setTimeout(r, 500));
  rec.stop();
  await stopped;
  stream.getTracks().forEach((t) => t.stop());
  const blob = new Blob(chunks, { type: picked.split(';')[0] });
  return { picked, actual: rec.mimeType, bytes: blob.size };
}
"""


def _configure(page: Page, ui_url: str, deployment: dict) -> None:
    page.goto(ui_url)
    url_box = page.get_by_placeholder("https://your-deployment....crewai.com")
    expect(url_box).to_be_visible()
    url_box.fill(deployment["url"])
    url_box.press("Tab")
    token_box = page.get_by_placeholder("bearer token from the deployment page")
    token_box.fill(deployment["token"])
    token_box.press("Tab")


def _send(page: Page, text: str) -> None:
    box = page.get_by_placeholder("type a message — or use the mic")
    box.fill(text)
    page.get_by_role("button", name="Send", exact=True).click()


def test_mic_recording_pipeline_produces_valid_container(page: Page, ui_url,
                                                         deployment):
    """The exact failure mode from 2026-07-13: a container the client
    mislabels. Assert the browser can (a) negotiate a supported format from
    the app's preference list and (b) actually record bytes in it — and that
    the container is one the app knows how to name for the OpenAI upload."""
    page.goto(ui_url)
    result = page.evaluate(MIC_PROBE_JS)
    assert result["picked"], "no MediaRecorder format from the app's list is supported"
    assert result["bytes"] > 0, "recorder produced an empty blob"
    base = (result["actual"] or result["picked"]).split(";")[0]
    assert base in {"audio/webm", "audio/mp4", "audio/ogg", "audio/wav",
                    "audio/x-wav"}, f"unmapped container: {base}"


def test_conversation_with_session_chain(page: Page, ui_url, deployment):
    _configure(page, ui_url, deployment)

    _send(page, "What is the latest output reading for PUMP A1?")
    first = page.locator(".msg.assistant").last
    expect(first).to_contain_text("PUMP A1", timeout=TURN_TIMEOUT_MS)
    expect(first).to_contain_text("units")

    _send(page, "And what about its energy use?")
    expect(page.locator(".msg.assistant")).to_have_count(2, timeout=TURN_TIMEOUT_MS)
    second = page.locator(".msg.assistant").last
    expect(second).to_contain_text("PUMP A1")
    expect(second).to_contain_text("kWh")

    expect(page.locator(".sess")).not_to_contain_text("(new conversation)")


def test_footer_shows_build_version(page: Page, ui_url, deployment):
    page.goto(ui_url)
    expect(page.locator(".sess")).to_contain_text(
        re.compile(r"build v\d+\.\d+\.\d+"))


def test_new_conversation_resets_chain(page: Page, ui_url, deployment):
    _configure(page, ui_url, deployment)
    _send(page, "goodbye")
    expect(page.locator(".msg.assistant").last).to_contain_text(
        "Goodbye", timeout=TURN_TIMEOUT_MS)
    page.get_by_role("button", name="new conversation").click()
    expect(page.locator(".sess")).to_contain_text("(new conversation)")
    expect(page.locator(".msg")).to_have_count(0)
