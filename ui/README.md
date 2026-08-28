# ui — browser client (Leptos / Rust → WASM)

All client-side: the page talks directly to (1) your CrewAI deployment and
(2) OpenAI's transcription API — from the browser, with keys you paste into
Settings (persisted only to your browser's localStorage).

## Run

```bash
rustup target add wasm32-unknown-unknown   # once
cargo install trunk                        # once
cd ui && trunk serve                       # http://127.0.0.1:8082
```

Fill in Settings:

| Field | Where it comes from |
|---|---|
| Deployment URL | your deployment's page in CrewAI AMP |
| Deployment bearer token | same page |
| OpenAI API key | only needed for the 🎤 mic button — it is used for transcription and nothing else (the deployed flow's agents run on the key configured on the deployment, not this one) |

Then type — or click **🎤 Talk**, speak, click **Stop**. The reply can be
spoken aloud (checkbox in Settings) via the browser's speech synthesis.

Conversation continuity is a `restoreFromStateId` **chain**: every turn sends
a fresh UUID and restores the previous turn's state (the footer shows the
current chain head, advanced only when a turn succeeds — never reuse one id
across kickoffs). **"New conversation"** clears the chain; keeping it is what
makes the form wizard work.

## Cross-origin note

The browser calls the deployment API cross-origin; AMP sends the CORS headers
for this (verified against a live deployment from the published Pages origin,
2026-07-13). If a corporate proxy or extension still blocks it, use
`client/ask.py` — identical loop, no browser rules.
