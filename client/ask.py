#!/usr/bin/env python3
"""Audio in, answer out — against a deployed CrewAI crew.

The whole path, in four steps:

    1. TRANSCRIBE  audio file -> text, via OpenAI's transcription API
                   (this is the ONLY step that needs an OpenAI API key)
    2. DISCOVER    GET {deployment}/inputs -> the exact input keys the
                   deployment expects (never assume them)
    3. KICKOFF     POST {deployment}/kickoff with {"inputs": {...}} — plus,
                   when continuing a conversation, top-level
                   "restoreFromStateId": <previous turn's id> (each turn
                   gets a FRESH id; never reuse one across kickoffs)
    4. POLL        GET {deployment}/status/{kickoff_id} until it finishes

Usage:

    python3 client/ask.py samples/question.wav

Configuration (env vars, or a local .env file next to where you run this):

    OPENAI_API_KEY           key for the transcription call (client-side only;
                             also read from ~/.openai-key if the env var is unset)
    CREWAI_DEPLOYMENT_URL    your deployment's base URL   } both shown on the
    CREWAI_DEPLOYMENT_TOKEN  your deployment's bearer token } deployment's page
                                                              in CrewAI AMP

No third-party packages needed — stdlib only, so you can run it with any
Python 3.10+ without installing anything.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"

# /status states, normalized to upper-case. Deployments report e.g.
# PENDING -> RUNNING -> SUCCESS; be liberal in what we accept.
RUNNING_STATES = {"PENDING", "QUEUED", "STARTED", "RUNNING", "PROCESSING"}
SUCCESS_STATES = {"SUCCESS", "SUCCEEDED", "COMPLETED", "COMPLETE", "FINISHED"}
FAILURE_STATES = {"FAILED", "FAILURE", "ERROR", "CANCELLED"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Real env vars win."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def http(method: str, url: str, token: str, body: dict | None = None) -> dict:
    """One JSON request to the deployment; explain auth/input errors clearly."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        if exc.code in (401, 403):
            sys.exit(
                f"{exc.code} from {url}\n{detail}\n\n"
                "Auth problem. Check CREWAI_DEPLOYMENT_TOKEN: it must be the "
                "bearer token shown on THIS deployment's page in CrewAI AMP — "
                "every deployment has its own token (and its own URL)."
            )
        if exc.code == 422:
            sys.exit(
                f"422 from {url}\n{detail}\n\n"
                "The deployment rejected the request body. Two usual causes:\n"
                "  1. Wrong input keys — run with --show-inputs to see the exact\n"
                "     keys this deployment expects, and send exactly those.\n"
                "  2. An id/session input that isn't a FULL, valid UUID —\n"
                "     never truncate UUIDs."
            )
        sys.exit(f"{exc.code} from {url}\n{detail}")


# --------------------------------------------------------------------------
# step 1: audio -> text (OpenAI transcription API)
# --------------------------------------------------------------------------

def transcribe(path: str, api_key: str, model: str) -> str:
    """Upload the audio file as multipart/form-data; return the transcript.

    Accepts whatever the API accepts: wav, mp3, m4a, webm, flac, ogg...
    Built by hand so the client stays dependency-free.
    """
    boundary = f"----audioquickstart{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        audio = fh.read()

    def field(name: str, value: str) -> bytes:
        return (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{name}"\r\n\r\n{value}\r\n').encode()

    body = b"".join([
        field("model", model),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f'filename="{os.path.basename(path)}"\r\n'
         f"Content-Type: {content_type}\r\n\r\n").encode(),
        audio,
        f"\r\n--{boundary}--\r\n".encode(),
    ])

    req = urllib.request.Request(
        OPENAI_TRANSCRIPTION_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())["text"].strip()
    except urllib.error.HTTPError as exc:
        sys.exit(f"OpenAI transcription error {exc.code}:\n"
                 f"{exc.read().decode(errors='replace')}")


# --------------------------------------------------------------------------
# steps 2-4: discover inputs, kickoff, poll
# --------------------------------------------------------------------------

def discover_inputs(base: str, token: str) -> list[str]:
    """GET /inputs — the deployment's actual input contract.

    ALWAYS check this instead of assuming key names. Sending the wrong keys
    (or missing one) is the #1 cause of 422s.
    """
    data = http("GET", f"{base}/inputs", token)
    # Response is {"inputs": [...]} on current deployments; tolerate a bare list.
    return data.get("inputs", data) if isinstance(data, dict) else data


SESSION_FILE = ".session"  # stores the PREVIOUS turn's id — the chain head


def chain_head(new: bool = False) -> str | None:
    """Conversation continuity works by CHAINING, not by reusing one id:
    every turn sends a fresh UUID as its id (the platform deprecates reusing
    inputs.id across kickoffs — it corrupts traces/metrics), plus the previous
    turn's id as top-level `restoreFromStateId`, which hydrates that turn's
    persisted state. Returns the stored chain head, or None for a fresh
    conversation (first run, --new-session, or an unparseable file)."""
    if new or not os.path.exists(SESSION_FILE):
        return None
    stored = open(SESSION_FILE).read().strip()
    try:
        uuid.UUID(stored)
        return stored
    except ValueError:
        return None


def save_chain_head(turn_id: str) -> None:
    """Advance the chain only after a turn SUCCEEDS — a failed turn may have
    persisted nothing, and restoring from it silently drops the context."""
    with open(SESSION_FILE, "w") as fh:
        fh.write(turn_id)


def build_inputs(required: list[str], transcript: str, session_id: str) -> dict:
    """Map the transcript (plus the session id) onto the required keys."""
    inputs: dict[str, str] = {}
    text_keys = [k for k in required if k in ("query", "question", "message", "text", "input")]
    id_keys = [k for k in required if k == "id" or k.endswith("_id")]

    for key in id_keys:
        # Session/run ids must be full, valid UUIDs.
        inputs[key] = session_id

    if text_keys:
        inputs[text_keys[0]] = transcript
    elif len(required) - len(id_keys) == 1:
        # Exactly one non-id input: that's where the transcript goes.
        inputs[next(k for k in required if k not in inputs)] = transcript
    elif not required:
        pass  # deployment takes no inputs (some flows); kickoff with {}
    else:
        sys.exit(
            f"Can't tell which of the required inputs {required} should "
            "receive the transcript. Edit build_inputs() in this script "
            "to match your deployment's contract."
        )
    return inputs


def kickoff(base: str, token: str, inputs: dict,
            restore_from: str | None = None) -> str:
    # The body MUST wrap your values in an "inputs" object. Conversation
    # continuity: `restoreFromStateId` sits at the TOP level, beside "inputs".
    body: dict = {"inputs": inputs}
    if restore_from:
        body["restoreFromStateId"] = restore_from
    data = http("POST", f"{base}/kickoff", token, body=body)
    return data["kickoff_id"]


def wait_for_answer(base: str, token: str, kickoff_id: str,
                    timeout_s: int = 300, poll_s: float = 0.4) -> str:
    """Poll /status/{kickoff_id} until the run reaches a terminal state.

    Starts at ``poll_s`` (default 400ms) and backs off ×1.5 to a 2s cap so a
    typical turn is noticed quickly without a tight /status loop forever.
    """
    deadline = time.monotonic() + timeout_s
    last_state = ""
    delay = poll_s
    while time.monotonic() < deadline:
        data = http("GET", f"{base}/status/{kickoff_id}", token)
        state = str(data.get("state", data.get("status", ""))).upper()
        if state != last_state:
            print(f"   state: {state or '?'}", file=sys.stderr)
            last_state = state
        if state in SUCCESS_STATES:
            return str(data.get("result") or data.get("result_json") or data)
        if state in FAILURE_STATES:
            sys.exit(f"Run failed:\n{json.dumps(data, indent=2)}")
        time.sleep(delay)
        delay = min(delay * 1.5, 2.0)
    sys.exit(f"Timed out after {timeout_s}s waiting for {kickoff_id}")


# --------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="Path to an audio file (wav/mp3/m4a/...)")
    ap.add_argument("--stt-model", default="gpt-4o-transcribe",
                    help="OpenAI transcription model (default: %(default)s; "
                         "whisper-1 also works)")
    ap.add_argument("--show-inputs", action="store_true",
                    help="Just print the deployment's required inputs and exit")
    ap.add_argument("--new-session", action="store_true",
                    help="Start a fresh conversation (rotates the stored session id)")
    args = ap.parse_args()

    base = os.environ.get("CREWAI_DEPLOYMENT_URL", "").rstrip("/")
    token = os.environ.get("CREWAI_DEPLOYMENT_TOKEN", "")
    if not base or not token:
        ap.error("Set CREWAI_DEPLOYMENT_URL and CREWAI_DEPLOYMENT_TOKEN "
                 "(both are on your deployment's page in CrewAI AMP).")

    if args.show_inputs:
        print(json.dumps(discover_inputs(base, token), indent=2))
        return
    if not args.audio:
        ap.error("Provide an audio file (or use --show-inputs).")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    key_file = os.path.expanduser("~/.openai-key")
    if not api_key and os.path.exists(key_file):
        api_key = open(key_file).read().strip()
    if not api_key:
        ap.error("Set OPENAI_API_KEY (or put the key in ~/.openai-key). "
                 "It is used ONLY for the transcription call.")

    print(f"1) transcribing {args.audio} with {args.stt_model} ...")
    transcript = transcribe(args.audio, api_key, args.stt_model)
    print(f"   transcript: {transcript!r}")

    print("2) discovering the deployment's input contract (GET /inputs) ...")
    required = discover_inputs(base, token)
    print(f"   required inputs: {required}")

    prev = chain_head(new=args.new_session)
    turn_id = str(uuid.uuid4())
    print(f"   turn id: {turn_id}"
          + (f" (chained to {prev})" if prev else " (new conversation)"))
    inputs = build_inputs(required, transcript, turn_id)
    print(f"3) kicking off with inputs: {json.dumps(inputs)}"
          + (" + restoreFromStateId" if prev else ""))
    kickoff_id = kickoff(base, token, inputs, restore_from=prev)
    print(f"   kickoff_id: {kickoff_id}")

    print("4) polling for the result ...")
    answer = wait_for_answer(base, token, kickoff_id)
    save_chain_head(turn_id)  # only on success — wait_for_answer exits on failure
    print(f"\nanswer:\n{answer}")


if __name__ == "__main__":
    main()
