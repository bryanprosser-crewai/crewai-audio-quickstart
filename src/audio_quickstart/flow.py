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
        ├─ form active ──────────────►  form agent (continues the wizard)
        └─ otherwise: one small LLM
           classification call ──────►  ASSET_DATA | START_FORM:<type> | UNKNOWN
        │
        ▼
    agent handlers (each agent has its OWN LLM instance — usage counters
    are scoped per LLM object, sharing one pools the numbers)
        │
        ▼
    @persist state keyed on `id` (= session_id) — messages + form progress
    survive across turn executions (on CrewAI AMP SaaS, state lands on the
    persistent volume by default).

Session contract:
    local:   flow.handle_turn(message, session_id=S)
    AMP:     POST /chat/start → S; then POST /chat/{S}/message {"message": ...}
             (AMP's chat worker calls handle_turn. conversation_start still
             hydrates a raw /kickoff if something hits that path.)
"""

from __future__ import annotations

import re

from pydantic import Field

from crewai import LLM, Flow
from crewai.experimental.conversational import ConversationConfig, ConversationState
from crewai.flow import listen
from crewai.flow.persistence import persist

from audio_quickstart.agents import build_data_agent, build_form_agent
from audio_quickstart.data import connect
from audio_quickstart.forms import FORM_SCHEMAS

_QUIT_RE = re.compile(r"\b(quit|exit|goodbye|bye)\b", re.IGNORECASE)
_CANCEL_RE = re.compile(r"\b(cancel|abort)\b", re.IGNORECASE)

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
# "live" with "Span orphaned — execution ended before completion event".
# One-trace-per-session is for a long-lived in-process REPL; AMP already
# treats each utterance as its own execution.
@ConversationConfig(defer_trace_finalization=False)
@persist()
class AssistantFlow(Flow[AssistantState]):
    """One handle_turn = one utterance. Live objects are rebuilt lazily per execution."""

    conversational = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._conn = None
        self._data_agent = None
        self._form_agent = None
        self._form_session = None
        self._classifier_llm = LLM(model="openai/gpt-4o", temperature=0, timeout=30)

    # -- AMP kickoff bridge -------------------------------------------------

    def conversation_start(self) -> None:
        """Hydrate a turn that arrived via AMP /kickoff rather than handle_turn.

        handle_turn already sets current_user_message and appends the user
        line. A raw kickoff only overlays inputs onto state (`id`, `message`),
        so copy `message` into the conversational turn if needed.
        """
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

    def _form(self):
        if self._form_agent is None:
            self._form_agent, self._form_session = build_form_agent(self.state.form_type)
            self._form_session.data.update(self.state.form_data)  # re-seed after restore
        return self._form_agent

    def _clear_form(self) -> None:
        self.state.active_mode = None
        self.state.form_type = None
        self.state.form_data = {}
        self._form_agent = self._form_session = None

    def _utterance(self) -> str:
        return (self.state.current_user_message or self.state.message or "").strip()

    # -- routing (deterministic first; a returned label skips the LLM router)

    def route_turn(self, context: dict) -> str:
        del context
        message = self._utterance()
        if _QUIT_RE.search(message):
            return "GOODBYE"
        if self.state.active_mode == "form":
            if _CANCEL_RE.search(message):
                return "CANCEL"
            return "FORM"
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

    def _run_form(self) -> str:
        agent = self._form()
        reply = str(agent.kickoff(self._context()))
        self.state.form_data = dict(self._form_session.data)
        if self._form_session.submitted:
            self._clear_form()
        return reply

    # -- handlers (method name ≠ route label, or the graph re-triggers) ------

    @listen("ASSET_DATA")
    def handle_asset_data(self) -> str:
        """Questions about asset readings, output, energy, runtime, or which assets exist."""
        reply = str(self._data().kickoff(self._context()))
        self.append_assistant_message(reply)
        return reply

    @listen("START_FORM")
    def handle_start_form(self) -> str:
        """User wants to file a maintenance or incident report — start the wizard."""
        self._form_agent = self._form_session = None
        self.state.active_mode = "form"
        self.state.form_data = {}
        reply = self._run_form()
        self.append_assistant_message(reply)
        return reply

    @listen("FORM")
    def handle_form(self) -> str:
        """Continue the in-progress voice-guided form (one field per turn)."""
        reply = self._run_form()
        self.append_assistant_message(reply)
        return reply

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
