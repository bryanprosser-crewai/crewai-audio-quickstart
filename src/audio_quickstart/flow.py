"""AssistantFlow — a conversational field-assistant flow, one kickoff per turn.

Architecture (the shape we see working in production engagements):

    kickoff inputs {id, message}
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
    @persist state keyed on `id` — history + form progress survive across
    kickoff executions (on CrewAI AMP SaaS, state lands on the persistent
    volume by default).

Session contract (one kickoff = one conversational turn):
    inputs: {"id": "<FRESH uuid>", "message": "<user text>"}
    plus, from turn 2 on, TOP-LEVEL "restoreFromStateId": <previous turn's id>
    result: the assistant's reply (string)

Continuity is CHAINED, not keyed: every turn sends a fresh id (the platform
deprecates reusing inputs.id across kickoffs — it corrupts traces, the
executions list, and metrics) and hydrates the previous turn's persisted
state via restoreFromStateId. Omit restoreFromStateId to start fresh.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from crewai import LLM, Flow
from crewai.flow.flow import listen, router, start
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


class AssistantState(BaseModel):
    """Serializable session state — `id` is the @persist restore key."""

    id: str = ""
    message: str = ""
    history: list[dict] = Field(default_factory=list)  # {role, content}
    active_mode: str | None = None                     # None | "form"
    form_type: str | None = None
    form_data: dict = Field(default_factory=dict)      # mirror of FormSession.data


@persist()
class AssistantFlow(Flow[AssistantState]):
    """One kickoff = one turn. Live objects are rebuilt lazily per execution."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._conn = None
        self._data_agent = None
        self._form_agent = None
        self._form_session = None
        self._classifier_llm = LLM(model="openai/gpt-4o", temperature=0, timeout=30)

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

    # -- routing (deterministic first) ---------------------------------------

    @start()
    def ingest(self) -> str:
        return self.state.message or ""

    @router(ingest)
    def route(self) -> str:
        message = self.state.message or ""
        if _QUIT_RE.search(message):
            return "END"
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
            return "END"
        return "UNKNOWN"

    # -- helpers --------------------------------------------------------------

    def _context(self) -> list[dict]:
        return [*self.state.history[-_MAX_TURNS * 2:],
                {"role": "user", "content": self.state.message}]

    def _finish(self, reply: str) -> str:
        self.state.history.append({"role": "user", "content": self.state.message})
        self.state.history.append({"role": "assistant", "content": reply})
        self.state.history = self.state.history[-_MAX_TURNS * 2:]
        return reply

    def _run_form(self) -> str:
        agent = self._form()
        reply = str(agent.kickoff(self._context()))
        self.state.form_data = dict(self._form_session.data)
        if self._form_session.submitted:
            self._clear_form()
        return reply

    # -- handlers -------------------------------------------------------------

    @listen("ASSET_DATA")
    def asset_data_turn(self) -> str:
        return self._finish(str(self._data().kickoff(self._context())))

    @listen("START_FORM")
    def start_form_turn(self) -> str:
        self._form_agent = self._form_session = None
        self.state.active_mode = "form"
        self.state.form_data = {}
        return self._finish(self._run_form())

    @listen("FORM")
    def form_turn(self) -> str:
        return self._finish(self._run_form())

    @listen("CANCEL")
    def cancel_turn(self) -> str:
        self._clear_form()
        return self._finish("Form cancelled. How can I help you?")

    @listen("END")
    def end_turn(self) -> str:
        return self._finish("Goodbye.")

    @listen("UNKNOWN")
    def unknown_turn(self) -> str:
        return self._finish(_UNKNOWN_REPLY)
