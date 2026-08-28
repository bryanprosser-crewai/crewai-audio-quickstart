"""AssistantFlow — a conversational field-assistant, one handle_turn per utterance.

Same shape as CrewAI conversational Flows (see the ClickHouse dashboards
example): each user line is a new graph run with the SAME session id.
The framework owns message history; we keep extra domain state (the form
wizard) on a ConversationState subclass and persist it across turns.

    handle_turn(message, session_id=S)
        │
        ▼
    deterministic-first router          zero LLM calls where possible:
        ├─ quit/goodbye regex  ──────►  canned goodbye  (no LLM)
        ├─ form active + cancel ─────►  canned cancel   (no LLM)
        ├─ form active ──────────────►  FORM (extract + canned next prompt)
        └─ otherwise: one small LLM
           classification call ──────►  ASSET_DATA | START_FORM:<type> | UNKNOWN
        │
        ▼
    handlers
        ├─ ASSET_DATA  — data agent.kickoff (tools; native LLM loop)
        ├─ START_FORM  — canned first question (zero LLM)
        ├─ FORM        — 1× LLM.call JSON extract + validate_field + canned prompt
        └─ CANCEL / GOODBYE / UNKNOWN — canned
        │
        ▼
    @persist state keyed on `id` (= session_id) — messages + form progress
    survive across turn executions (on CrewAI AMP SaaS, state lands on the
    persistent volume by default).

Session contract:
    local:   flow.handle_turn(message, session_id=S)
    AMP:     POST /chat/start → S; then POST /chat/{S}/message {"message": ...}
             Clients wait on GET /chat/{S}/stream/events (not /status poll).
             (AMP's chat worker calls handle_turn. conversation_start still
             hydrates a raw /kickoff if something hits that path.)
"""

from __future__ import annotations

import json
import re

from pydantic import Field

from crewai import LLM, Flow
from crewai.experimental.conversational import ConversationConfig, ConversationState
from crewai.flow import listen
from crewai.flow.persistence import persist

from audio_quickstart.agents import build_data_agent
from audio_quickstart.data import connect
from audio_quickstart.forms import (
    FORM_SCHEMAS,
    FormSession,
    ask_field,
    extract_system_prompt,
    parse_extract_json,
    readback_and_confirm,
    start_form_prompt,
    validate_field,
)

_QUIT_RE = re.compile(r"\b(quit|exit|goodbye|bye)\b", re.IGNORECASE)
_CANCEL_RE = re.compile(r"\b(cancel|abort)\b", re.IGNORECASE)
_CONFIRM_RE = re.compile(
    r"\b(confirm|submit|that's right|looks good|yes|yeah|yep)\b",
    re.IGNORECASE,
)
_FORM_CLOSED_RE = re.compile(
    r"form cancelled|mock submit ok|report (has been )?filed",
    re.IGNORECASE,
)
_FORM_INTENT_LABELS = frozenset({"START_FORM", "FORM"})
_FORM_MENTIONS = (
    (re.compile(r"mainten\w*\s+report|maintenance_report", re.IGNORECASE), "maintenance_report"),
    (re.compile(r"incident\s+report|incident_report", re.IGNORECASE), "incident_report"),
)

_FORM_TOKENS = "\n".join(
    f"- start_form:{ft}  — user wants to fill in: {schema.title}"
    for ft, schema in FORM_SCHEMAS.items()
)

_CLASSIFIER_SYSTEM = f"""You are a routing agent for a field assistant. \
Given a user utterance, respond with exactly one classification token.

Respond with exactly one of:
- asset_data_query        — anything about the assets or their data: readings, \
output, energy, runtime — INCLUDING which assets exist, what can be asked \
about, or an asset mentioned by name
{_FORM_TOKENS}
- quit                    — user says goodbye or wants to stop
- unknown                 — anything else: off-topic, greetings, unclear requests

Reply with the token only. No punctuation, no explanation."""

_UNKNOWN_REPLY = ("I'm not sure what you'd like to do. You can ask about asset "
                  "readings or fill out a maintenance or incident report.")

_MAX_TURNS = 10  # history depth (pairs)


class AssistantState(ConversationState):
    """Framework chat fields (id, messages, current_user_message, …) plus the
    form wizard. `message` is the AMP /kickoff input; handle_turn uses
    current_user_message instead."""

    message: str = ""
    active_mode: str | None = None                     # None | "form"
    form_type: str | None = None
    form_data: dict = Field(default_factory=dict)      # mirror of FormSession.data


# AMP chat runs one kickoff process per /message and never calls
# finalize_session_traces(). Deferring FlowFinished leaves those kickoffs
# Pending — the chat worker often sets instance `defer_trace_finalization=True`
# even when ConversationConfig says False, and that attr is checked first.
# One-trace-per-session is for a long-lived in-process REPL; AMP already
# treats each utterance as its own execution.
@ConversationConfig(defer_trace_finalization=False)
@persist()
class AssistantFlow(Flow[AssistantState]):
    """One handle_turn = one utterance. Live objects are rebuilt lazily per execution."""

    conversational = True
    defer_trace_finalization = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.defer_trace_finalization = False
        self._conn = None
        self._data_agent = None
        self._form_session = None
        self._classifier_llm = LLM(model="openai/gpt-4o", temperature=0, timeout=30)

    def _should_defer_trace_finalization(self) -> bool:
        """AMP never finalizes the session; always emit FlowFinished per turn."""
        return False

    # -- AMP kickoff bridge -------------------------------------------------

    def conversation_start(self) -> None:
        """Hydrate a turn that arrived via AMP /kickoff rather than handle_turn.

        handle_turn already sets current_user_message and appends the user
        line. A raw kickoff only overlays inputs onto state (`id`, `message`),
        so copy `message` into the conversational turn if needed.
        """
        self.defer_trace_finalization = False
        incoming = (self.state.current_user_message or self.state.message or "").strip()
        if incoming and not (self.state.current_user_message or "").strip():
            self.receive_user_message(incoming)
        if incoming:
            self.state.message = incoming

    # -- lazy builders (state restores across pods; objects don't) ----------

    def _data(self):
        if self._data_agent is None:
            self._conn = connect()
            self._data_agent = build_data_agent(self._conn)
        return self._data_agent

    def _ensure_form_session(self) -> FormSession | None:
        """Rebuild the wizard from persisted form_type + form_data (AMP restores state, not objects)."""
        form_type = self.state.form_type
        if form_type not in FORM_SCHEMAS:
            return None
        if self._form_session is None or self._form_session.schema.form_type != form_type:
            self._form_session = FormSession(FORM_SCHEMAS[form_type])
        self._form_session.data = dict(self.state.form_data or {})
        return self._form_session

    def _clear_form(self) -> None:
        self.state.active_mode = None
        self.state.form_type = None
        self.state.form_data = {}
        self._form_session = None

    def _utterance(self) -> str:
        return (self.state.current_user_message or self.state.message or "").strip()

    @staticmethod
    def _msg_field(message, key, default=None):
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    def _form_checkpoint(self) -> dict:
        return {
            "active_mode": self.state.active_mode,
            "form_type": self.state.form_type,
            "form_data": dict(self.state.form_data or {}),
        }

    def _apply_form_checkpoint(self, checkpoint: dict) -> None:
        form_type = checkpoint.get("form_type")
        if form_type in FORM_SCHEMAS:
            self.state.form_type = form_type
        if checkpoint.get("form_data") and not self.state.form_data:
            self.state.form_data = dict(checkpoint["form_data"])
        if checkpoint.get("active_mode") == "form" or form_type in FORM_SCHEMAS:
            self.state.active_mode = "form"

    def _form_type_mentioned(self, text: str) -> str | None:
        for needle, form_type in _FORM_MENTIONS:
            if needle.search(text):
                return form_type
        lowered = text.lower()
        for form_type, schema in FORM_SCHEMAS.items():
            if schema.title.lower() in lowered:
                return form_type
        return None

    def _infer_open_form(self) -> tuple[str | None, dict, bool]:
        """Recover an in-progress wizard when AMP restored messages but not form fields."""
        open_type = None
        checkpoint: dict = {}
        saw_form = False
        for message in self.state.messages:
            role = self._msg_field(message, "role")
            content = self._msg_field(message, "content")
            text = content if isinstance(content, str) else ""
            metadata = self._msg_field(message, "metadata") or {}
            stored = metadata.get("form") if isinstance(metadata, dict) else None
            if isinstance(stored, dict) and stored.get("form_type") in FORM_SCHEMAS:
                saw_form = True
                open_type = stored["form_type"]
                checkpoint = stored
                continue
            if role == "user" and _CANCEL_RE.search(text):
                open_type = None
                checkpoint = {}
                continue
            if role == "assistant" and _FORM_CLOSED_RE.search(text):
                open_type = None
                checkpoint = {}
                continue
            mentioned = self._form_type_mentioned(text)
            if mentioned:
                saw_form = True
                open_type = mentioned
        return open_type, checkpoint, saw_form

    def _open_form(self) -> tuple[str | None, dict]:
        """Form type to continue, plus any checkpoint recovered from history."""
        inferred, checkpoint, saw_form = self._infer_open_form()
        if inferred:
            return inferred, checkpoint
        if saw_form:
            return None, {}
        if self.state.active_mode == "form" and self.state.form_type in FORM_SCHEMAS:
            return self.state.form_type, {}
        if self.state.last_intent in _FORM_INTENT_LABELS and self.state.form_type in FORM_SCHEMAS:
            return self.state.form_type, {}
        return None, {}

    def _continue_form(self, form_type: str, checkpoint: dict | None = None) -> None:
        if checkpoint:
            self._apply_form_checkpoint(checkpoint)
        self.state.form_type = form_type
        self.state.active_mode = "form"

    def _reply_form(self, reply: str) -> str:
        self.append_assistant_message(reply, metadata={"form": self._form_checkpoint()})
        return reply

    # -- routing (deterministic first; a returned label skips the LLM router)

    def route_turn(self, context: dict) -> str:
        del context
        message = self._utterance()
        if _QUIT_RE.search(message):
            return "GOODBYE"
        form_type, checkpoint = self._open_form()
        if form_type or self.state.active_mode == "form":
            if form_type:
                self._continue_form(form_type, checkpoint)
            if _CANCEL_RE.search(message):
                return "CANCEL"
            return "FORM"
        if _CANCEL_RE.search(message) and self._infer_open_form()[2]:
            return "CANCEL"
        try:
            intent = str(self._classifier_llm.call(messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": message},
            ])).strip().lower()
        except Exception:
            intent = "asset_data_query"  # safe fallback
        if intent == "asset_data_query":
            return "ASSET_DATA"
        if intent.startswith("start_form:"):
            form_type = intent.split(":", 1)[1]
            if form_type in FORM_SCHEMAS:
                self.state.form_type = form_type
                return "START_FORM"
        if intent == "quit":
            return "GOODBYE"
        return "UNKNOWN"

    # -- helpers --------------------------------------------------------------

    def _context(self) -> list[dict]:
        """Canonical chat history for the agents (user line already appended)."""
        lines: list[dict] = []
        for message in self.state.messages[-_MAX_TURNS * 2:]:
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                lines.append({"role": role, "content": content})
        return lines

    def _extract_field_value(self, field, utterance: str) -> dict:
        raw = str(self._classifier_llm.call(messages=[
            {"role": "system", "content": extract_system_prompt(field)},
            {"role": "user", "content": utterance},
        ]))
        return parse_extract_json(raw)

    def _submit_form(self, session: FormSession) -> str:
        missing = session.missing_required()
        if missing:
            field = missing[0]
            return f"Can't submit yet — still need {field.label}. {ask_field(field)}"
        payload = json.dumps(session.data)
        self._clear_form()
        return f"MOCK SUBMIT OK (no records system connected): {payload}"

    def _run_form(self, *, starting: bool = False) -> str:
        """One field per turn: canned start, then 1× LLM.call extract + Python validate."""
        session = self._ensure_form_session()
        if session is None:
            return "I don't have a form in progress. Ask to file a maintenance or incident report."
        if starting:
            return start_form_prompt(session)

        message = self._utterance()
        if not session.missing_required() and _CONFIRM_RE.search(message):
            return self._submit_form(session)

        field = session.next_open_field()
        if field is None:
            return readback_and_confirm(session)

        try:
            extracted = self._extract_field_value(field, message)
        except Exception:
            return f"I didn't catch that. {ask_field(field)}"

        if extracted.get("skip") and not field.required:
            session.data[field.name] = ""
            self.state.form_data = dict(session.data)
            nxt = session.next_open_field()
            if nxt is None:
                return f"Skipping {field.label}. {readback_and_confirm(session)}"
            return f"Skipping {field.label}. {ask_field(nxt)}"

        if extracted.get("error") or "value" not in extracted:
            return f"I didn't catch a {field.label}. {ask_field(field)}"

        normalised, error = validate_field(field, str(extracted["value"]).strip())
        if error:
            return f"{error} {ask_field(field)}"

        session.data[field.name] = normalised
        self.state.form_data = dict(session.data)
        nxt = session.next_open_field()
        ack = f"Got it — {field.label} is {normalised}."
        if nxt is None:
            return f"{ack} {readback_and_confirm(session)}"
        return f"{ack} {ask_field(nxt)}"

    # -- handlers (method name ≠ route label, or the graph re-triggers) ------

    @listen("ASSET_DATA")
    def handle_asset_data(self) -> str:
        """Questions about asset readings, output, energy, runtime, or which assets exist."""
        reply = str(self._data().kickoff(self._context()))
        self.append_assistant_message(reply)
        return reply

    @listen("START_FORM")
    def handle_start_form(self) -> str:
        """User wants to file a maintenance or incident report — canned first question."""
        self._form_session = None
        self.state.active_mode = "form"
        self.state.form_data = {}
        reply = self._run_form(starting=True)
        return self._reply_form(reply)

    @listen("FORM")
    def handle_form(self) -> str:
        """Continue the wizard: extract one value, validate, canned next prompt."""
        reply = self._run_form()
        return self._reply_form(reply)

    @listen("CANCEL")
    def handle_cancel(self) -> str:
        """Abort the in-progress form and return to the open assistant."""
        self._clear_form()
        reply = "Form cancelled. How can I help you?"
        self.append_assistant_message(reply)
        return reply

    @listen("GOODBYE")
    def handle_goodbye(self) -> str:
        """User is done with this conversation."""
        reply = "Goodbye."
        self.append_assistant_message(reply)
        return reply

    @listen("UNKNOWN")
    def handle_unknown(self) -> str:
        """Off-topic, greeting, or unclear — point the user at what we can do."""
        self.append_assistant_message(_UNKNOWN_REPLY)
        return _UNKNOWN_REPLY
