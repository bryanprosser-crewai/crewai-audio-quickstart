"""Unit tier — no network, no credentials, no browser, no LLM calls.

Covers the pure logic layer (fuzzy matching, OFFSET-1 latest, validation,
form session state) and the flow's deterministic routing ladder, which is
reachable without any LLM call by construction.
"""

from __future__ import annotations

import sqlite3

import pytest

from audio_quickstart.data import ASSETS, connect
from audio_quickstart.flow import AssistantFlow
from audio_quickstart.forms import FORM_SCHEMAS, FormSession, validate_field
from audio_quickstart.tools import (
    ListAssetsTool,
    SetFieldTool,
    SubmitFormTool,
    build_form_tools,
    fuzzy_match_asset,
    query_latest,
    query_period,
)


@pytest.fixture()
def conn(tmp_path, monkeypatch) -> sqlite3.Connection:
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path))
    return connect()


# -- fuzzy matching -----------------------------------------------------------

@pytest.mark.parametrize("spoken,expected", [
    ("pump a1", "PUMP A1"),
    ("Pump A-1", "PUMP A1"),
    ("compressor b2", "COMPRESSOR B2"),
    ("generator c one", "GENERATOR C1"),
])
def test_fuzzy_match_resolves_to_canonical(spoken, expected):
    assert fuzzy_match_asset(spoken) == expected


def test_fuzzy_match_rejects_nonsense():
    assert fuzzy_match_asset("quarterly revenue") is None


# -- OFFSET-1 "latest" (skip the in-progress day) -----------------------------

def test_latest_skips_current_day(conn):
    result = query_latest(conn, "PUMP A1", ["output_units"])
    dates = [r[0] for r in conn.execute(
        "SELECT RECORD_DATE FROM DAILY_ASSET_READINGS WHERE ASSET_NAME='PUMP A1' "
        "ORDER BY RECORD_DATE DESC LIMIT 2")]
    assert result["record_date"] == dates[1], "must skip the newest (in-progress) day"
    assert result["values"]["output"]["unit"] == "units"
    assert "note" not in result


def test_latest_single_day_falls_back_with_note(conn):
    conn.execute("DELETE FROM DAILY_ASSET_READINGS "
                 "WHERE ASSET_NAME='PUMP A2' AND RECORD_DATE < "
                 "(SELECT MAX(RECORD_DATE) FROM DAILY_ASSET_READINGS "
                 " WHERE ASSET_NAME='PUMP A2')")
    result = query_latest(conn, "PUMP A2", ["energy_kwh"])
    assert "note" in result, "single-day data must carry the in-progress caveat"


def test_latest_unknown_asset_errors(conn):
    assert "error" in query_latest(conn, "NO SUCH ASSET", ["output_units"])


def test_period_stats_aggregates(conn):
    result = query_period(conn, "PUMP A1", ["output_units"], "weekly_avg")
    assert result["aggregation"] == "weekly_avg"
    assert result["values"]["output"]["value"] > 0


# -- tools (LLM-free `_run` paths) --------------------------------------------

def test_list_assets_tool_returns_all():
    hits = ListAssetsTool()._run("")
    for asset in ASSETS:
        assert asset in hits


def test_form_tools_validate_and_gate_submission():
    tools, session = build_form_tools("maintenance_report")
    set_field = next(t for t in tools if isinstance(t, SetFieldTool))
    submit = next(t for t in tools if isinstance(t, SubmitFormTool))

    assert "ERROR" in submit._run(), "must refuse to submit an empty form"
    assert "ERROR" in set_field._run("TimeSpentHours", "about an hour")
    assert "OK" in set_field._run("TimeSpentHours", "1.5")
    assert "ERROR" in set_field._run("NoSuchField", "x")

    for name, value in [("AssetId", "PUMP A1"), ("WorkDone", "greased bearing"),
                        ("CompletionDate", "July 13 2026"), ("ReportedBy", "T-100")]:
        assert "OK" in set_field._run(name, value)
    assert session.data["CompletionDate"] == "2026-07-13", "dates normalise to ISO"
    assert "MOCK SUBMIT OK" in submit._run()
    assert session.submitted


# -- field validation ---------------------------------------------------------

@pytest.mark.parametrize("value,ok", [
    ("2026-07-13", True), ("July 13 2026", True), ("07/13/2026", True),
    ("yesterday", False),
])
def test_validate_date(value, ok):
    field = next(f for f in FORM_SCHEMAS["maintenance_report"].fields
                 if f.field_type == "date")
    _, error = validate_field(field, value)
    assert (error is None) is ok


def test_validate_choice_is_case_insensitive():
    field = next(f for f in FORM_SCHEMAS["incident_report"].fields
                 if f.field_type == "choice")
    normalised, error = validate_field(field, "hIgH")
    assert error is None and normalised == "High"


def test_form_session_tracks_missing():
    session = FormSession(FORM_SCHEMAS["incident_report"])
    assert {f.name for f in session.missing_required()} == \
           {"AssetId", "Severity", "Description", "ReportedBy"}
    session.data["AssetId"] = "COMPRESSOR B1"
    assert "AssetId" not in {f.name for f in session.missing_required()}


# -- deterministic router ladder (zero LLM calls by construction) -------------

@pytest.fixture()
def flow(tmp_path, monkeypatch) -> AssistantFlow:
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path))
    return AssistantFlow()


@pytest.mark.parametrize("message", ["goodbye", "ok QUIT now", "bye!"])
def test_router_quit_is_deterministic(flow, message):
    flow.state.message = message
    assert flow.route() == "END"


def test_router_cancel_inside_form(flow):
    flow.state.active_mode = "form"
    flow.state.message = "cancel that"
    assert flow.route() == "CANCEL"


def test_router_form_mode_continues(flow):
    flow.state.active_mode = "form"
    flow.state.form_type = "maintenance_report"
    flow.state.message = "the asset id is PUMP A1"
    assert flow.route() == "FORM"
