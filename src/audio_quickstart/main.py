#!/usr/bin/env python
"""Entry points. Local smoke: three turns, one session id via handle_turn."""

from uuid import uuid4

from audio_quickstart.flow import AssistantFlow


def kickoff() -> None:
    """Scripted 3-turn smoke session — same session_id, not a restore chain."""
    flow = AssistantFlow()
    session_id = str(uuid4())
    try:
        for message in (
            "List the assets I can ask about.",
            "What was the latest output on pump A1?",
            "I'd like to file a maintenance report.",
        ):
            print(f"\nYOU: {message}")
            reply = flow.handle_turn(message, session_id=session_id)
            print(f"ASSISTANT: {reply}\n(session: {session_id})")
    finally:
        flow.finalize_session_traces()


def chat() -> None:
    """Interactive terminal chat (CrewAI conversational REPL)."""
    AssistantFlow().chat()


def plot() -> None:
    AssistantFlow().plot()


if __name__ == "__main__":
    kickoff()
