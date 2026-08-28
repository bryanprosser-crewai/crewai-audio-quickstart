#!/usr/bin/env python
"""Entry points. Local smoke run: two data turns + a form turn, one session."""

from audio_quickstart.flow import AssistantFlow


def kickoff() -> None:
    # Chain recipe: fresh state per turn, restored from the previous turn's
    # persisted id (inputs.id reuse is deprecated platform-wide).
    prev: str | None = None
    for message in (
        "List the assets I can ask about.",
        "What was the latest output on pump A1?",
        "I'd like to file a maintenance report.",
    ):
        print(f"\nYOU: {message}")
        flow = AssistantFlow()
        reply = flow.kickoff(inputs={"message": message},
                             restore_from_state_id=prev)
        prev = flow.state.id
        print(f"ASSISTANT: {reply}\n(state id: {prev})")


def plot() -> None:
    AssistantFlow().plot()


if __name__ == "__main__":
    kickoff()
