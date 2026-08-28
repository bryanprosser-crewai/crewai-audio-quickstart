#!/usr/bin/env python3
"""Audio in, answer out — one chat turn against a deployed conversational Flow.

AMP chat API (see https://docs-platform.crewai.com/platform/en/guides/conversational-flow-chat):

    1. TRANSCRIBE   audio file -> text, via OpenAI's transcription API
                    (this is the ONLY step that needs an OpenAI API key)
    2. SESSION      POST {deployment}/chat/start  -> session_id
                    (reused from .session on later turns; --new-session starts over)
    3. MESSAGE      POST {deployment}/chat/{session_id}/message
                    {"message": "<transcript>", "stream": true}  -> kickoff_id
    4. STREAM       GET  {deployment}/chat/{session_id}/stream/events
                    until turn_completed (tokens as they arrive)
    5. HISTORY      GET  {deployment}/chat/{session_id}/history
                    only if the stream had no token text (canned replies)

Usage:

    python3 client/ask.py samples/question.wav

Requires GET /inspect to report flow.chat.conversational and handle_turn.
Confirm with:  python3 client/ask.py --inspect

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
SESSION_FILE = ".session"

SSE_ATTACH_PAUSE_S = 0.05


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
    """One JSON request to the deployment; explain auth/chat errors clearly."""
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
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        if exc.code in (401, 403):
            sys.exit(
                f"{exc.code} from {url}\n{detail}\n\n"
                "Auth problem. Check CREWAI_DEPLOYMENT_TOKEN: it must be the "
                "bearer token shown on THIS deployment's page in CrewAI AMP."
            )
        if exc.code == 404:
            sys.exit(
                f"404 from {url}\n{detail}\n\n"
                "Chat is not available (or this session_id is unknown). "
                "GET /inspect must show flow.chat.conversational and "
                "handle_turn. Redeploy this conversational Flow, or pass "
                "--new-session if the stored session expired."
            )
        if exc.code == 409:
            sys.exit(
                f"409 from {url}\n{detail}\n\n"
                "This session already has an active turn. Wait for it to "
                "finish (or start --new-session) before sending another clip."
            )
        if exc.code == 422:
            sys.exit(
                f"422 from {url}\n{detail}\n\n"
                "session_id must be a full, valid UUID — never truncate it."
            )
        sys.exit(f"{exc.code} from {url}\n{detail}")


def transcribe(path: str, api_key: str, model: str) -> str:
    """Upload the audio file as multipart/form-data; return the transcript."""
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


def inspect_chat(base: str, token: str) -> dict:
    data = http("GET", f"{base}/inspect", token)
    return ((data.get("flow") or {}).get("chat") or {})


def start_session(base: str, token: str) -> str:
    data = http("POST", f"{base}/chat/start", token, body={})
    sid = data.get("session_id") or data.get("id")
    if not sid:
        sys.exit(f"POST /chat/start returned no session_id: {data}")
    return str(sid)


def send_turn(base: str, token: str, session_id: str, message: str) -> str:
    data = http(
        "POST", f"{base}/chat/{session_id}/message", token,
        body={"message": message, "stream": True},
    )
    kid = data.get("kickoff_id") or data.get("id")
    if not kid:
        sys.exit(f"POST /chat/.../message returned no kickoff_id: {data}")
    return str(kid)


def consume_sse_frames(readline, deadline: float) -> tuple[str, str]:
    """Read AMP SSE frames until the turn ends.

    Returns (outcome, text) where outcome is completed | failed | closed | timeout.
    Token frames are concatenated; canned replies often have no tokens.
    """
    pending: list[str] = []
    tokens: list[str] = []

    def flush_block() -> tuple[str, str] | None:
        if not pending:
            return None
        data = "\n".join(pending)
        pending.clear()
        try:
            frame = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(frame, dict):
            return None
        kind = str(frame.get("type") or "")
        payload = frame.get("data") if isinstance(frame.get("data"), dict) else {}
        if kind == "token":
            piece = payload.get("content") or payload.get("text") or ""
            if piece:
                tokens.append(str(piece))
            return None
        if kind in {"message", "conversation_message"}:
            role = payload.get("role") or frame.get("role") or ""
            content = payload.get("content") or payload.get("text") or ""
            if str(role) == "assistant" and content:
                tokens.append(str(content))
            return None
        if kind == "turn_completed":
            extra = payload.get("content") or payload.get("text") or payload.get("result") or ""
            return ("completed", "".join(tokens) or str(extra))
        if kind in {"turn_failed", "error"}:
            msg = payload.get("message") or payload.get("error") or data
            return ("failed", str(msg))
        return None

    while time.monotonic() < deadline:
        raw = readline()
        if not raw:
            terminal = flush_block()
            if terminal:
                return terminal
            return ("closed", "".join(tokens))
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if line.endswith("\n"):
            line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
        if line == "":
            terminal = flush_block()
            if terminal:
                return terminal
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            pending.append(line[5:].lstrip())
    return ("timeout", "".join(tokens))


def open_chat_sse(base: str, token: str, session_id: str, deadline: float):
    """Attach to the active turn's SSE. 409 = not published yet; retry briefly."""
    url = (f"{base}/chat/{session_id}/stream/events"
           "?events=*&last_event_id=0-0")
    last_409 = ""
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            url, method="GET",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "text/event-stream",
            },
        )
        remaining = max(1.0, deadline - time.monotonic())
        try:
            return urllib.request.urlopen(req, timeout=remaining)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code == 409:
                last_409 = detail
                time.sleep(SSE_ATTACH_PAUSE_S)
                continue
            if exc.code in (401, 403):
                sys.exit(
                    f"{exc.code} from {url}\n{detail}\n\n"
                    "Auth problem. Check CREWAI_DEPLOYMENT_TOKEN."
                )
            sys.exit(f"{exc.code} from {url}\n{detail}")
    sys.exit(
        f"Timed out attaching to {url}"
        + (f"\nLast 409: {last_409}" if last_409 else "")
    )


def wait_for_turn_sse(base: str, token: str, session_id: str,
                      timeout_s: int = 300) -> str:
    """Wait on chat SSE; fall back to /history when the stream has no text."""
    deadline = time.monotonic() + timeout_s
    resp = open_chat_sse(base, token, session_id, deadline)
    try:
        outcome, text = consume_sse_frames(resp.readline, deadline)
    finally:
        resp.close()
    if outcome == "failed":
        sys.exit(f"Turn failed:\n{text}")
    if outcome == "timeout":
        sys.exit(f"Timed out after {timeout_s}s waiting on chat SSE ({session_id})")
    if text.strip():
        return text
    reply = last_assistant_reply(base, token, session_id)
    if reply:
        return reply
    sys.exit(f"Turn ended with no assistant text (session {session_id})")


def last_assistant_reply(base: str, token: str, session_id: str) -> str:
    history = http("GET", f"{base}/chat/{session_id}/history", token)
    for msg in reversed(history.get("messages") or []):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "assistant" and content:
            return str(content)
    return ""


def load_session_id() -> str | None:
    if not os.path.exists(SESSION_FILE):
        return None
    stored = open(SESSION_FILE).read().strip()
    try:
        uuid.UUID(stored)
        return stored
    except ValueError:
        return None


def save_session_id(session_id: str) -> None:
    with open(SESSION_FILE, "w") as fh:
        fh.write(session_id)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="Path to an audio file (wav/mp3/m4a/...)")
    ap.add_argument("--stt-model", default="gpt-4o-transcribe",
                    help="OpenAI transcription model (default: %(default)s; "
                         "whisper-1 also works)")
    ap.add_argument("--inspect", action="store_true",
                    help="Print GET /inspect (chat capability) and exit")
    ap.add_argument("--new-session", action="store_true",
                    help="Start a fresh AMP chat session")
    args = ap.parse_args()

    base = os.environ.get("CREWAI_DEPLOYMENT_URL", "").rstrip("/")
    token = os.environ.get("CREWAI_DEPLOYMENT_TOKEN", "")
    if not base or not token:
        ap.error("Set CREWAI_DEPLOYMENT_URL and CREWAI_DEPLOYMENT_TOKEN "
                 "(both are on your deployment's page in CrewAI AMP).")

    if args.inspect:
        print(json.dumps(http("GET", f"{base}/inspect", token), indent=2))
        return
    if not args.audio:
        ap.error("Provide an audio file (or use --inspect).")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    key_file = os.path.expanduser("~/.openai-key")
    if not api_key and os.path.exists(key_file):
        api_key = open(key_file).read().strip()
    if not api_key:
        ap.error("Set OPENAI_API_KEY (or put the key in ~/.openai-key). "
                 "It is used ONLY for the transcription call.")

    chat = inspect_chat(base, token)
    if not (chat.get("conversational") and chat.get("handle_turn")):
        sys.exit(
            "This deployment does not expose conversational chat.\n"
            f"GET /inspect flow.chat = {json.dumps(chat)}\n"
            "Redeploy the conversational Flow from this repo, then retry."
        )

    print(f"1) transcribing {args.audio} with {args.stt_model} ...")
    transcript = transcribe(args.audio, api_key, args.stt_model)
    print(f"   transcript: {transcript!r}")

    sid = None if args.new_session else load_session_id()
    if sid:
        print(f"2) continuing session {sid}")
    else:
        print("2) POST /chat/start ...")
        sid = start_session(base, token)
        save_session_id(sid)
        print(f"   session_id: {sid}")

    print(f"3) POST /chat/{sid}/message (stream) ...")
    kickoff_id = send_turn(base, token, sid, transcript)
    print(f"   kickoff_id: {kickoff_id}")

    print("4) GET /chat/.../stream/events ...")
    answer = wait_for_turn_sse(base, token, sid)
    print(f"\nanswer:\n{answer}")


if __name__ == "__main__":
    main()
