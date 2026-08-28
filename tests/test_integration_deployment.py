"""Integration tier — real kickoffs against the deployed flow (needs .env).

Exercises the wire contract the clients depend on: single-turn answers and
cross-kickoff session continuity via the restoreFromStateId chain (fresh
UUID per turn; the platform deprecates inputs.id reuse).
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


def test_restore_chain_carries_context(deployment):
    t1, _ = run_turn(deployment, "What is the latest output reading for PUMP A1?")
    t2, reply2 = run_turn(deployment, "And what about its energy use?",
                          restore_from=t1)
    up2 = reply2.upper()
    assert "PUMP A1" in up2.replace("-", " "), \
        "turn 2 must resolve the pronoun from restored turn-1 history"
    assert "KWH" in up2

    _, reply3 = run_turn(deployment, "And its runtime hours?", restore_from=t2)
    up3 = reply3.upper()
    assert "PUMP A1" in up3.replace("-", " "), \
        "state written during a fork run must itself be restorable"


def test_deterministic_turn_no_llm(deployment):
    _, reply = run_turn(deployment, "goodbye")
    assert "goodbye" in reply.lower()
