# crewai-audio-quickstart

**A voice-ready, conversational field assistant on CrewAI AMP — the full reference
pattern:** audio → transcription → a conversational Flow (`handle_turn`, same
session id) with intent routing, tool-using agents, and session memory → answer
(text you can speak back).

The flow mirrors an architecture running in real field deployments, with every
customer-specific piece swapped for a generic, self-contained stand-in (the data
layer is a seeded SQLite file — no external services, no credentials).

## Architecture

```
audio file / mic ──► OpenAI transcription — the CLIENT's key, used only here.
                          │  text          (The deployed flow's agents make their
                          │                 own OpenAI calls server-side with the
                          │                 key configured ON THE DEPLOYMENT.)
                          ▼
             POST {deployment}/kickoff
               {"inputs": {"id": "<session uuid>", "message": "..."}}
                          │  reuse the SAME id every turn of a conversation
                          ▼
        AssistantFlow  (conversational=True; one handle_turn / kickoff = one utterance)
        ├─ deterministic-first router: quit / cancel / form-continuation
        │  are handled with ZERO LLM calls; everything else gets one small
        │  classification call (asset_data | start_form:<type> | unknown)
        ├─ Asset-data agent — 3 tools over the SQLite readings table:
        │    get_latest_reading · get_period_stats · list_assets
        │    (fuzzy name matching → canonical names; "latest" skips the
        │    in-progress day and says so when it can't)
        ├─ Form agent — voice-guided wizard (maintenance / incident report):
        │    one field per turn, typed validation, read-back, explicit
        │    "confirm" before a (mock) submission
        └─ ConversationState + @persist keyed on `id`: messages + form progress
           survive across turns (on AMP SaaS, state lands on the persistent
           volume by default)
```

**Session contract:** each turn sends `{"id": "<session uuid>", "message": "<text>"}`.
Reuse that id for the whole conversation — locally via `flow.handle_turn(message,
session_id=S)`, on AMP via `/kickoff` with the same `inputs.id`. A new conversation
is a new id. Ids must be full, valid UUIDs (an AMP-side rule). The reply is a string.

This is CrewAI's conversational Flow model (same shape as the ClickHouse dashboards
example). The previous `restoreFromStateId` chain — a fresh UUID per turn — is gone.

## Deploy (CrewAI AMP)

1. Connect this repo (Deploy From Code → Git Repository), branch `main`.
   The repo auto-deploys on new commits once connected.
2. The deployment needs an LLM: an **OpenAI LLM connection** in your org, or an
   `OPENAI_API_KEY` environment variable on the deployment (the agents and the
   intent classifier use `openai/gpt-4o`).
3. No other configuration. No database env vars (SQLite is seeded on first use),
   and no storage env vars (AMP persists flow state by default).

## Ask it something (audio in, answer out)

```bash
cp .env.example .env         # fill in the two deployment values
python3 client/ask.py samples/question.wav
```

`client/ask.py` is stdlib-only and does the whole loop: transcribe → discover the
deployment's input contract (`GET /inputs` — never assume key names) → kickoff →
poll → print. It keeps a `.session` file holding the conversation's session id so
consecutive questions continue ONE conversation — which is what makes the form
wizard work by voice:

```bash
python3 client/ask.py clips/file-a-report.wav      # "I'd like to file a maintenance report"
python3 client/ask.py clips/asset-id.wav           # "Pump A1"
python3 client/ask.py clips/work-done.wav          # ... one field per clip ...
python3 client/ask.py --new-session clips/next.wav # fresh conversation
```

Things to try:
- "List the assets I can ask about." (phrase it readings-flavored — a bare
  "what do you have?" routes to the deliberate unknown-intent fallback)
- "What was the latest output on pump A1?"
- "Average energy use on compressor B1 this week?"
- "I'd like to file an incident report." → then answer its questions → "confirm"

Record clips with `scripts/make_sample_audio.sh`, your phone, or anything that
produces wav/mp3/m4a.

## Browser client (`ui/`)

A fully client-side web UI (Rust/Leptos → WASM): paste your deployment URL,
bearer token, and OpenAI key, then talk — mic capture → transcription → kickoff →
spoken reply via the browser's speech synthesis. The three values are stored only
in your browser's localStorage; the browser sends the deployment values only to
your deployment endpoint and the OpenAI key only to OpenAI's transcription API.
(The deployed flow's agents use the key configured on the deployment — the
browser key does transcription and nothing else.) See
[`ui/README.md`](ui/README.md) for build/run.

> Browser → deployment CORS is verified working (the API sends CORS headers).
> If a corporate proxy still blocks it, `client/ask.py` is the same loop with
> no browser rules.

## Local development

```bash
uv sync
OPENAI_API_KEY=sk-... uv run kickoff     # scripted 3-turn smoke session (handle_turn)
OPENAI_API_KEY=sk-... uv run chat        # interactive terminal REPL
```

The data layer (`src/audio_quickstart/data.py`) seeds deterministic synthetic
readings for five assets × 45 days. Swap `connect()`/the tool internals for your
warehouse or API to make this real — the agents, router, and session machinery
don't change.

## Layout

```
src/audio_quickstart/
├── flow.py      # AssistantFlow: conversational router + handlers + @persist
├── agents.py    # data agent + form agent (one LLM instance per agent)
├── tools.py     # 3 data tools (SQLite) + 3 form tools; caching disabled
├── forms.py     # form schemas, typed validation, session state machine
├── data.py      # synthetic SQLite readings (the stubbed "warehouse")
└── main.py      # local smoke (handle_turn) + chat() REPL
client/ask.py    # stdlib audio client (transcribe → kickoff → poll)
ui/              # Leptos (Rust/WASM) browser client
```

## Releasing the browser UI (GitHub Pages)

The UI is published at https://crewaiinc-fde.github.io/crewai-audio-quickstart/ —
built locally, no CI:

```
./scripts/release-ui.sh
```

The script trunk-builds `ui/` with the Pages subpath baked in and force-pushes
the snapshot to the `gh-pages` branch, which Pages serves (Settings → Pages →
Deploy from a branch → `gh-pages` / root). Everything runs client-side: users
paste their own deployment URL, bearer token, and (for the mic) OpenAI key,
stored only in the browser's localStorage.

## Testing

Three tiers, all under `tests/` (deps install with `uv sync`; Playwright
needs a one-time `uv run playwright install firefox`):

```
uv run pytest -m "not credentialed"   # unit: no network, no creds, ~2s
uv run pytest -m credentialed         # integration + e2e (see below)
uv run pytest                         # everything
```

Credentialed tests (real kickoffs against your deployment + Playwright
driving the published UI in Firefox) require a **git-ignored `.env` at the
repo root**:

```
CREWAI_DEPLOYMENT_URL=https://<your-deployment>.crewai.com
CREWAI_DEPLOYMENT_TOKEN=<bearer token from the deployment's AMP page>
OPENAI_API_KEY=sk-...        # only for mic-related tests
```

Without it the suite stops with these same instructions. To run the e2e
tier on the Chrome engine instead (mic negotiates webm/opus there vs.
Firefox's ogg — both verified):

```
uv run playwright install chromium
uv run pytest -m credentialed -o addopts="" --browser chromium
```

Firefox is the default e2e browser deliberately — it has the strictest MediaRecorder
behavior (default container `audio/ogg`), so a green mic test there covers
the rest. The mic pipeline runs against Firefox's fake microphone, so no
audio hardware (or OpenAI spend) is needed.

Each release bumps the UI version (footer shows `build vX.Y.Z`) and all
assets are content-hashed, so a stale cached page is both visible and
mechanically distinct; GitHub Pages' ~10-minute index.html TTL is the only
residual staleness.
