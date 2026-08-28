# Streaming voice plan (no ElevenLabs)

Optimize **time-to-first-audio** (end of user speech → first played sample), not
“AMP timer until the full reply.” CrewAI stays the brain. STT/TTS stay on
**OpenAI + the browser**.

This branch (`streaming-voice`) starts from `conversational-chat` (AMP chat SSE).
`kickoff-poll` cannot overlap TTS with a running turn.

## Cascade

```
mic ──► STT (today: gpt-4o-transcribe after Stop; later: gpt-live-transcribe)
            │
            ▼
     CrewAI chat turn (SSE tokens; form path = 1× LLM.call + canned prompt)
            │
            ▼
     browser speechSynthesis per sentence  (later: OpenAI streaming TTS)
```

Tool calls still **block speech** until the tool (or Python `validate_field`)
returns. Hide STT wait and TTS wait; do not wait for `turn_completed` to start
audio when tokens or a canned sentence already exist.

## Phases

### 0 — Baseline

Keep STT / AMP timers. **to audio** is first `speechSynthesis` `onstart`
(first sentence, not the full blob).

### 1 — Sentence TTS from SSE (started)

Buffer AMP `token` frames to a sentence/clause, `speak()` as each completes,
flush leftovers on `turn_completed`. Do not speak the full reply again if
sentences already started. If AMP emits no tokens (canned form line), split
the history reply and speak the same way.

### 2 — Streaming STT (not started)

`gpt-4o-transcribe` on a blob after Stop cannot stream the mic. Switch the
live path to OpenAI Realtime transcription (`gpt-live-transcribe`). Optional:
start the CrewAI turn on a stable partial.

### 3 — Form path: one `LLM.call` (started)

Stop `Agent.kickoff()` on form turns (that is the LLM → `set_field` → LLM
loop). Start-form is a canned first question (zero LLM). Continuations:
regex confirm/cancel, one `LLM.call` for JSON `{value}`, Python
`validate_field`, canned next prompt. Asset questions still use the data
agent.

### 4 — Optional gateway

Local process for Realtime STT + AMP SSE + TTS if WASM/CORS is painful.

## What not to do

- Faster `/status` poll
- `max_iter=1` on the form agent
- Passing tools into `LLM.call` and feeding results back into a second
  completion
- Expecting AMP `/kickoff` to grow a token stream
