"""Integration tier — real kickoffs against the deployed flow (needs .env).

Exercises the AMP chat contract the clients depend on: single-turn answers
and cross-turn continuity via POST /chat/start + /chat/{id}/message.
"""

from __future__ import annotations

import pytest

from tests.conftest import run_turn

pytestmark = pytest.mark.credentialed


def test_asset_discovery_routes_to_data_agent(deployment):
    """Regression (2026-07-13): asset-DISCOVERY questions used to fall into
    the unknown-intent fallback because the classifier only covered readings;
    list_assets was unreachable. The demo hand-off leads with this phrase."""
    _, reply = run_turn(deployment, "List the assets I can ask about.")
    assert "PUMP A1" in reply.upper().replace("-", " ")


def test_single_turn_answers_from_data(deployment):
    _, reply = run_turn(deployment, "What is the latest output reading for PUMP A1?")
    up = reply.upper()
    assert "PUMP A1" in up.replace("-", " ")
    assert "UNITS" in up


def test_session_id_carries_context(deployment):
    sid, _ = run_turn(deployment, "What is the latest output reading for PUMP A1?")
    _, reply2 = run_turn(deployment, "And what about its energy use?",
                         session_id=sid)
    up2 = reply2.upper()
    assert "PUMP A1" in up2.replace("-", " "), \
        "turn 2 must resolve the pronoun from session history"
    assert "KWH" in up2

    _, reply3 = run_turn(deployment, "And its runtime hours?", session_id=sid)
    up3 = reply3.upper()
    assert "PUMP A1" in up3.replace("-", " "), \
        "later turns on the same session id must still see the history"


def test_deterministic_turn_no_llm(deployment):
    _, reply = run_turn(deployment, "goodbye")
    assert "goodbye" in reply.lower()
