//! Field Assistant — all-client-side browser UI (Leptos CSR → WASM).
//!
//! Pattern: paste three values (deployment URL, deployment bearer token,
//! OpenAI key), all persisted to localStorage and sent nowhere except to
//! those two APIs, directly from the browser. Mic → OpenAI transcription →
//! AMP chat (/chat/start, /chat/{id}/message, SSE /stream/events) → reply,
//! optionally spoken via the browser's speech synthesis.

use std::cell::RefCell;
use std::rc::Rc;

use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use js_sys::{Reflect, Uint8Array};

const SETTINGS_KEY: &str = "audio_quickstart_settings";
const OPENAI_TRANSCRIPTIONS: &str = "https://api.openai.com/v1/audio/transcriptions";

#[derive(Clone, Default, Serialize, Deserialize, PartialEq)]
struct Settings {
    deployment_url: String,
    deployment_token: String,
    openai_key: String,
    speak_replies: bool,
}

impl Settings {
    fn ready(&self) -> bool {
        !self.deployment_url.is_empty() && !self.deployment_token.is_empty()
    }
    fn load() -> Self {
        storage()
            .and_then(|s| s.get_item(SETTINGS_KEY).ok().flatten())
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }
    fn save(&self) {
        if let (Some(s), Ok(raw)) = (storage(), serde_json::to_string(self)) {
            let _ = s.set_item(SETTINGS_KEY, &raw);
        }
    }
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
}

#[derive(Clone, PartialEq)]
struct Msg {
    role: &'static str, // "user" | "assistant"
    text: String,
}

// ---------------------------------------------------------------------------
// AMP chat API (start → message → SSE → history fallback)
// ---------------------------------------------------------------------------

fn cors_hint(op: &str, e: impl std::fmt::Display) -> String {
    format!("{op} failed: {e} (if this is a CORS error, use client/ask.py)")
}

async fn authorized_json(
    cfg: &Settings,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let url = format!("{}{path}", cfg.deployment_url.trim_end_matches('/'));
    let mut req = match method {
        "POST" => gloo_net::http::Request::post(&url),
        _ => gloo_net::http::Request::get(&url),
    };
    req = req
        .header("Authorization", &format!("Bearer {}", cfg.deployment_token))
        .header("Content-Type", "application/json");
    let built = if let Some(body) = body {
        req.body(body.to_string()).map_err(|e| e.to_string())?
    } else {
        req.build().map_err(|e| e.to_string())?
    };
    let resp = built.send().await.map_err(|e| cors_hint(path, e))?;
    let text = resp.text().await.unwrap_or_default();
    if !resp.ok() {
        return Err(format!("{path} HTTP {}: {text}", resp.status()));
    }
    if text.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

fn json_id(v: &serde_json::Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(|x| x.as_str()).map(String::from))
}

async fn require_chat(cfg: &Settings) -> Result<(), String> {
    let v = authorized_json(cfg, "GET", "/inspect", None).await?;
    let chat = v
        .get("flow")
        .and_then(|f| f.get("chat"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let conversational = chat.get("conversational").and_then(|x| x.as_bool()).unwrap_or(false);
    let handle_turn = chat.get("handle_turn").and_then(|x| x.as_bool()).unwrap_or(false);
    if conversational && handle_turn {
        Ok(())
    } else {
        Err(format!(
            "this deployment does not expose conversational chat \
             (GET /inspect flow.chat = {chat}). Redeploy this Flow."
        ))
    }
}

async fn start_session(cfg: &Settings) -> Result<String, String> {
    require_chat(cfg).await?;
    let v = authorized_json(cfg, "POST", "/chat/start", Some(serde_json::json!({}))).await?;
    json_id(&v, &["session_id", "id"])
        .ok_or_else(|| format!("POST /chat/start returned no session_id: {v}"))
}

async fn send_turn(cfg: &Settings, session_id: &str, message: &str) -> Result<String, String> {
    let path = format!("/chat/{session_id}/message");
    let v = authorized_json(
        cfg, "POST", &path,
        Some(serde_json::json!({ "message": message, "stream": true })),
    ).await?;
    json_id(&v, &["kickoff_id", "id"])
        .ok_or_else(|| format!("POST {path} returned no kickoff_id: {v}"))
}

async fn last_assistant_reply(cfg: &Settings, session_id: &str) -> String {
    let path = format!("/chat/{session_id}/history");
    if let Ok(v) = authorized_json(cfg, "GET", &path, None).await {
        if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
            for msg in msgs.iter().rev() {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if role == "assistant" {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            return content.to_string();
                        }
                    }
                }
            }
        }
    }
    String::new()
}

fn parse_sse_block(block: &str) -> Option<serde_json::Value> {
    let mut data = String::new();
    for line in block.split('\n') {
        let line = line.trim_end_matches('\r');
        if line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data.is_empty() {
        return None;
    }
    serde_json::from_str(&data).ok()
}

fn take_sse_frames(buf: &mut String) -> Vec<serde_json::Value> {
    let normalized = buf.replace("\r\n", "\n").replace('\r', "\n");
    *buf = normalized;
    let mut out = Vec::new();
    while let Some(idx) = buf.find("\n\n") {
        let raw = buf[..idx].to_string();
        *buf = buf[idx + 2..].to_string();
        if let Some(v) = parse_sse_block(&raw) {
            out.push(v);
        }
    }
    out
}

fn frame_token_delta(frame: &serde_json::Value) -> Option<String> {
    let kind = frame.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let payload = frame.get("data").filter(|d| d.is_object());
    match kind {
        "token" => payload
            .and_then(|d| d.get("content").or_else(|| d.get("text")))
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from),
        "message" | "conversation_message" => {
            let role = payload
                .and_then(|d| d.get("role"))
                .or_else(|| frame.get("role"))
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role != "assistant" {
                return None;
            }
            payload
                .and_then(|d| d.get("content").or_else(|| d.get("text")))
                .and_then(|c| c.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from)
        }
        _ => None,
    }
}

fn frame_terminal(frame: &serde_json::Value) -> Option<Result<String, String>> {
    let kind = frame.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let payload = frame.get("data").cloned().unwrap_or_else(|| serde_json::json!({}));
    match kind {
        "turn_completed" => {
            let extra = payload.get("content")
                .or_else(|| payload.get("text"))
                .or_else(|| payload.get("result"))
                .and_then(|c| c.as_str())
                .unwrap_or("");
            Some(Ok(extra.to_string()))
        }
        "turn_failed" | "error" => {
            let msg = payload.get("message")
                .or_else(|| payload.get("error"))
                .and_then(|m| m.as_str())
                .map(String::from)
                .unwrap_or_else(|| frame.to_string());
            Some(Err(format!("turn failed: {msg}")))
        }
        _ => None,
    }
}

async fn read_sse_chunk(
    reader: &web_sys::ReadableStreamDefaultReader,
) -> Result<Option<Vec<u8>>, String> {
    let result = JsFuture::from(reader.read()).await.map_err(js_err)?;
    let done = Reflect::get(&result, &JsValue::from_str("done"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    if done {
        return Ok(None);
    }
    let value = Reflect::get(&result, &JsValue::from_str("value")).map_err(js_err)?;
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }
    let arr = Uint8Array::new(&value);
    let mut buf = vec![0u8; arr.length() as usize];
    arr.copy_to(&mut buf);
    Ok(Some(buf))
}

async fn attach_chat_sse(cfg: &Settings, session_id: &str) -> Result<web_sys::Response, String> {
    let url = format!(
        "{}/chat/{session_id}/stream/events?events=*&last_event_id=0-0",
        cfg.deployment_url.trim_end_matches('/')
    );
    let window = web_sys::window().ok_or_else(|| "no window".to_string())?;
    let headers = web_sys::Headers::new().map_err(js_err)?;
    headers
        .set("Authorization", &format!("Bearer {}", cfg.deployment_token))
        .map_err(js_err)?;
    headers.set("Accept", "text/event-stream").map_err(js_err)?;
    let init = web_sys::RequestInit::new();
    init.set_method("GET");
    init.set_headers(&headers);
    let mut last_409 = String::new();
    for _ in 0..20 {
        let request = web_sys::Request::new_with_str_and_init(&url, &init).map_err(js_err)?;
        let resp_val = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| cors_hint("/chat/.../stream/events", format!("{e:?}")))?;
        let resp: web_sys::Response = resp_val.dyn_into().map_err(js_err)?;
        if resp.status() == 409 {
            last_409 = JsFuture::from(resp.text().map_err(js_err)?)
                .await
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            gloo_timers::future::TimeoutFuture::new(50).await;
            continue;
        }
        if !resp.ok() {
            let text = JsFuture::from(resp.text().map_err(js_err)?)
                .await
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            return Err(format!(
                "/chat/.../stream/events HTTP {}: {text}",
                resp.status()
            ));
        }
        return Ok(resp);
    }
    Err(format!(
        "timed out attaching to chat SSE (last 409: {last_409})"
    ))
}

/// Wait on AMP chat SSE. `on_token` receives each token/message delta so the
/// UI can paint before `turn_completed`. Returns concatenated token text
/// (empty for canned replies that never emit tokens).
async fn wait_for_sse<F>(cfg: &Settings, session_id: &str, mut on_token: F) -> Result<String, String>
where
    F: FnMut(&str),
{
    let resp = attach_chat_sse(cfg, session_id).await?;
    let body = resp.body().ok_or_else(|| "chat SSE had no body".to_string())?;
    let reader: web_sys::ReadableStreamDefaultReader = body
        .get_reader()
        .dyn_into()
        .map_err(|_| "chat SSE body is not a readable stream".to_string())?;
    let mut rest = String::new();
    let mut tokens = String::new();
    loop {
        let Some(bytes) = read_sse_chunk(&reader).await? else {
            break;
        };
        rest.push_str(&String::from_utf8_lossy(&bytes));
        for frame in take_sse_frames(&mut rest) {
            if let Some(delta) = frame_token_delta(&frame) {
                tokens.push_str(&delta);
                on_token(&delta);
            }
            if let Some(terminal) = frame_terminal(&frame) {
                let extra = terminal?;
                if tokens.is_empty() && !extra.is_empty() {
                    on_token(&extra);
                    tokens.push_str(&extra);
                }
                return Ok(tokens);
            }
        }
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// OpenAI transcription (browser FormData multipart)
// ---------------------------------------------------------------------------

async fn transcribe(
    cfg: &Settings,
    audio: web_sys::Blob,
    filename: &str,
) -> Result<String, String> {
    if cfg.openai_key.is_empty() {
        return Err("Set the OpenAI key in Settings to use the mic.".into());
    }
    let form = web_sys::FormData::new().map_err(js_err)?;
    form.append_with_str("model", "gpt-4o-transcribe").map_err(js_err)?;
    form.append_with_blob_and_filename("file", &audio, filename).map_err(js_err)?;
    let resp = gloo_net::http::Request::post(OPENAI_TRANSCRIPTIONS)
        .header("Authorization", &format!("Bearer {}", cfg.openai_key))
        .body(form)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("transcription HTTP {}: {}", resp.status(),
                           resp.text().await.unwrap_or_default()));
    }
    let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    v.get("text").and_then(|t| t.as_str()).map(|t| t.trim().to_string())
        .ok_or_else(|| format!("no text in transcription response: {v}"))
}

fn js_err(e: JsValue) -> String {
    format!("{e:?}")
}

fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or_else(|| js_sys::Date::now())
}

fn fmt_ms(ms: f64) -> String {
    if ms < 1000.0 {
        format!("{ms:.0}ms")
    } else {
        format!("{:.2}s", ms / 1000.0)
    }
}

fn format_timing(stt_ms: Option<f64>, amp_ms: f64, speak_ms: Option<Result<f64, String>>) -> String {
    let mut parts = Vec::new();
    if let Some(ms) = stt_ms {
        parts.push(format!("before AMP (STT) {}", fmt_ms(ms)));
    }
    parts.push(format!("AMP {}", fmt_ms(amp_ms)));
    match speak_ms {
        Some(Ok(ms)) => parts.push(format!("to audio {}", fmt_ms(ms))),
        Some(Err(_)) => parts.push("to audio failed".into()),
        None => {}
    }
    parts.join(" · ")
}

/// Speak the reply and return ms until playback actually starts (`onstart`).
async fn speak_until_start(text: &str) -> Result<f64, String> {
    let window = web_sys::window().ok_or_else(|| "no window".to_string())?;
    let synth = window.speech_synthesis().map_err(js_err)?;
    let utt = Rc::new(
        web_sys::SpeechSynthesisUtterance::new_with_text(text).map_err(js_err)?,
    );
    let t0 = now_ms();
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let utt = utt.clone();
        let resolve = resolve.clone();
        let on_start = Closure::<dyn FnMut()>::once(move || {
            let _ = resolve.call1(&JsValue::UNDEFINED, &JsValue::from_f64(now_ms() - t0));
        });
        utt.set_onstart(Some(on_start.as_ref().unchecked_ref()));
        on_start.forget();

        let on_error = Closure::<dyn FnMut()>::once(move || {
            let _ = reject.call1(&JsValue::UNDEFINED, &JsValue::from_str("speech synthesis error"));
        });
        utt.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();

        synth.speak(&utt);
    });
    let value = JsFuture::from(promise).await.map_err(js_err)?;
    value.as_f64().ok_or_else(|| "speech start returned no time".into())
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[component]
fn App() -> impl IntoView {
    let initial = Settings::load();
    let configured = initial.ready();
    let (settings, set_settings) = signal(initial);
    let (msgs, set_msgs) = signal(Vec::<Msg>::new());
    let (session_id, set_session_id) = signal(Option::<String>::None);
    let (busy, set_busy) = signal(false);
    let (status, set_status) = signal(String::new());
    let (recording, set_recording) = signal(false);
    let (draft, set_draft) = signal(String::new());
    let (show_settings, set_show_settings) = signal(!configured);
    let (timing, set_timing) = signal(String::new());

    let recorder: Rc<RefCell<Option<web_sys::MediaRecorder>>> = Rc::new(RefCell::new(None));

    // one user turn: start session if needed → /message → SSE (tokens paint live)
    let send_text = move |text: String, stt_ms: Option<f64>| {
        if text.trim().is_empty() || busy.get_untracked() {
            return;
        }
        let cfg = settings.get_untracked();
        if !cfg.ready() {
            set_status.set("Set the deployment URL + token in Settings first.".into());
            set_show_settings.set(true);
            return;
        }
        set_msgs.update(|m| m.push(Msg { role: "user", text: text.clone() }));
        set_busy.set(true);
        set_timing.set(String::new());
        let existing = session_id.get_untracked();
        spawn_local(async move {
            let amp_t0 = now_ms();
            let sid = match existing {
                Some(s) => s,
                None => {
                    set_status.set("starting session…".into());
                    match start_session(&cfg).await {
                        Ok(s) => {
                            set_session_id.set(Some(s.clone()));
                            s
                        }
                        Err(e) => {
                            set_status.set(format!("error: {e}"));
                            set_busy.set(false);
                            return;
                        }
                    }
                }
            };
            set_status.set("sending turn…".into());
            let mut painted = false;
            let outcome = match send_turn(&cfg, &sid, &text).await {
                Ok(kid) => {
                    set_status.set(format!("streaming ({kid})…"));
                    match wait_for_sse(&cfg, &sid, |delta| {
                        if !painted {
                            set_msgs.update(|m| {
                                m.push(Msg { role: "assistant", text: delta.to_string() })
                            });
                            painted = true;
                        } else {
                            set_msgs.update(|m| {
                                if let Some(last) = m.last_mut() {
                                    if last.role == "assistant" {
                                        last.text.push_str(delta);
                                    }
                                }
                            });
                        }
                    }).await {
                        Ok(tokens) => {
                            let reply = if tokens.is_empty() {
                                last_assistant_reply(&cfg, &sid).await
                            } else {
                                tokens
                            };
                            if reply.is_empty() {
                                Err("turn ended with no assistant text".into())
                            } else {
                                if !painted {
                                    set_msgs.update(|m| {
                                        m.push(Msg { role: "assistant", text: reply.clone() })
                                    });
                                }
                                Ok(reply)
                            }
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            };
            let amp_ms = now_ms() - amp_t0;
            match outcome {
                Ok(reply) => {
                    let speak_ms = if cfg.speak_replies {
                        set_status.set("speaking…".into());
                        Some(speak_until_start(&reply).await)
                    } else {
                        None
                    };
                    set_timing.set(format_timing(stt_ms, amp_ms, speak_ms));
                    set_status.set(String::new());
                }
                Err(e) => {
                    set_timing.set(format_timing(stt_ms, amp_ms, None));
                    set_status.set(format!("error: {e}"));
                }
            }
            set_busy.set(false);
        });
    };

    // mic toggle: record → transcribe → send_text
    let rec_handle = recorder.clone();
    let toggle_mic = move |_| {
        if recording.get_untracked() {
            if let Some(r) = rec_handle.borrow_mut().take() {
                let _ = r.stop(); // onstop handles the rest
            }
            set_recording.set(false);
            return;
        }
        let cfg = settings.get_untracked();
        if cfg.openai_key.is_empty() {
            set_status.set("Mic needs the OpenAI key (Settings) — it does the transcription.".into());
            set_show_settings.set(true);
            return;
        }
        let rec_slot = rec_handle.clone();
        set_status.set("requesting microphone…".into());
        spawn_local(async move {
            let Some(devices) = web_sys::window()
                .map(|w| w.navigator())
                .and_then(|n| n.media_devices().ok())
            else {
                set_status.set("no mediaDevices in this browser".into());
                return;
            };
            let constraints = web_sys::MediaStreamConstraints::new();
            constraints.set_audio(&JsValue::TRUE);
            let stream_promise = match devices.get_user_media_with_constraints(&constraints) {
                Ok(p) => p,
                Err(e) => {
                    set_status.set(format!("mic error: {e:?}"));
                    return;
                }
            };
            let stream: web_sys::MediaStream = match JsFuture::from(stream_promise).await {
                Ok(s) => s.unchecked_into(),
                Err(e) => {
                    set_status.set(format!("mic denied: {e:?}"));
                    return;
                }
            };
            // OpenAI rejects uploads whose filename/extension doesn't match the
            // actual container, and MediaRecorder's default container differs by
            // browser (Chrome: webm/opus, Safari: mp4/AAC). Negotiate a format
            // explicitly; the upload is later named after what was really used.
            let picked = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4",
                          "audio/ogg;codecs=opus", "audio/wav"]
                .into_iter()
                .find(|m| web_sys::MediaRecorder::is_type_supported(m));
            let made = match picked {
                Some(m) => {
                    let opts = web_sys::MediaRecorderOptions::new();
                    opts.set_mime_type(m);
                    web_sys::MediaRecorder::new_with_media_stream_and_media_recorder_options(
                        &stream, &opts)
                }
                None => web_sys::MediaRecorder::new_with_media_stream(&stream),
            };
            let rec = match made {
                Ok(r) => r,
                Err(e) => {
                    set_status.set(format!("recorder error: {e:?}"));
                    return;
                }
            };

            let chunks: Rc<RefCell<Vec<web_sys::Blob>>> = Rc::new(RefCell::new(Vec::new()));
            let chunks_in = chunks.clone();
            let on_data = Closure::<dyn FnMut(web_sys::BlobEvent)>::new(move |e: web_sys::BlobEvent| {
                if let Some(b) = e.data() {
                    chunks_in.borrow_mut().push(b);
                }
            });
            rec.set_ondataavailable(Some(on_data.as_ref().unchecked_ref()));
            on_data.forget();

            let stream_stop = stream.clone();
            let rec_mime_src = rec.clone();
            let on_stop = Closure::<dyn FnMut()>::new(move || {
                // release the mic indicator
                for track in stream_stop.get_tracks().iter() {
                    track.unchecked_into::<web_sys::MediaStreamTrack>().stop();
                }
                let parts = js_sys::Array::new();
                for b in chunks.borrow().iter() {
                    parts.push(b);
                }
                // Name the upload after the container the recorder REALLY used.
                let mime = rec_mime_src.mime_type();
                let (blob_type, filename) = match mime.split(';').next().unwrap_or("") {
                    "audio/mp4" => ("audio/mp4", "clip.mp4"),
                    "audio/ogg" => ("audio/ogg", "clip.ogg"),
                    "audio/wav" | "audio/x-wav" => ("audio/wav", "clip.wav"),
                    _ => ("audio/webm", "clip.webm"),
                };
                let opts = web_sys::BlobPropertyBag::new();
                opts.set_type(blob_type);
                let Ok(blob) = web_sys::Blob::new_with_blob_sequence_and_options(&parts, &opts)
                else {
                    set_status.set("could not assemble the recording".into());
                    return;
                };
                let cfg2 = settings.get_untracked();
                set_status.set("transcribing…".into());
                spawn_local(async move {
                    let stt_t0 = now_ms();
                    match transcribe(&cfg2, blob, filename).await {
                        Ok(text) if !text.is_empty() => {
                            let stt_ms = now_ms() - stt_t0;
                            set_status.set(String::new());
                            send_text(text, Some(stt_ms));
                        }
                        Ok(_) => set_status.set("heard nothing — try again".into()),
                        Err(e) => set_status.set(format!("error: {e}")),
                    }
                });
            });
            rec.set_onstop(Some(on_stop.as_ref().unchecked_ref()));
            on_stop.forget();

            if rec.start().is_ok() {
                *rec_slot.borrow_mut() = Some(rec);
                set_recording.set(true);
                set_status.set("recording — click Stop when done".into());
            } else {
                set_status.set("could not start recording".into());
            }
        });
    };

    let submit_draft = move |_| {
        let text = draft.get_untracked();
        set_draft.set(String::new());
        send_text(text, None);
    };

    view! {
        <main>
            <h1>"Field Assistant" <span class="tag">"voice-first asset operations"</span></h1>

            <div class="card">
                <details prop:open=move || show_settings.get()>
                    <summary>"Settings (stored only in this browser's localStorage)"</summary>
                    <label>"Deployment URL"</label>
                    <input type="text" placeholder="https://your-deployment....crewai.com"
                        prop:value=move || settings.get().deployment_url
                        on:change=move |ev| {
                            set_settings.update(|s| s.deployment_url = event_target_value(&ev));
                            settings.get_untracked().save();
                        } />
                    <label>"Deployment bearer token"</label>
                    <input type="password" placeholder="bearer token from the deployment page"
                        prop:value=move || settings.get().deployment_token
                        on:change=move |ev| {
                            set_settings.update(|s| s.deployment_token = event_target_value(&ev));
                            settings.get_untracked().save();
                        } />
                    <label>"OpenAI API key (only for mic transcription; leave empty for text-only)"</label>
                    <input type="password" placeholder="sk-..."
                        prop:value=move || settings.get().openai_key
                        on:change=move |ev| {
                            set_settings.update(|s| s.openai_key = event_target_value(&ev));
                            settings.get_untracked().save();
                        } />
                    <label>
                        <input type="checkbox"
                            prop:checked=move || settings.get().speak_replies
                            on:change=move |ev| {
                                set_settings.update(|s| s.speak_replies = event_target_checked(&ev));
                                settings.get_untracked().save();
                            } />
                        " speak replies aloud"
                    </label>
                </details>
            </div>

            <div class="card">
                <div class="msgs">
                    <For each=move || msgs.get().into_iter().enumerate()
                         key=|(i, _)| *i
                         children=move |(_, m)| {
                             view! { <div class=format!("msg {}", m.role)>{m.text.clone()}</div> }
                         } />
                </div>
                <p class="status">{move || status.get()}</p>
                <p class="timing" class:hidden=move || timing.get().is_empty()>
                    {move || timing.get()}
                </p>
                <div class="row">
                    <input type="text" placeholder="type a message — or use the mic"
                        prop:value=move || draft.get()
                        on:input=move |ev| set_draft.set(event_target_value(&ev))
                        on:keydown=move |ev| {
                            if ev.key() == "Enter" {
                                let text = draft.get_untracked();
                                set_draft.set(String::new());
                                send_text(text, None);
                            }
                        } />
                    <button disabled=move || busy.get() on:click=submit_draft>"Send"</button>
                    <button class=move || if recording.get() { "rec" } else { "" }
                        disabled=move || busy.get()
                        on:click=toggle_mic>
                        {move || if recording.get() { "Stop" } else { "🎤 Talk" }}
                    </button>
                </div>
                <p class="sess">
                    {concat!("build v", env!("CARGO_PKG_VERSION"), " · session: ")}
                    {move || session_id.get().unwrap_or_else(|| "(new conversation)".into())}
                    <button class="ghost" on:click=move |_| {
                        set_session_id.set(None);
                        set_msgs.set(Vec::new());
                        set_timing.set(String::new());
                        set_status.set("new conversation started".into());
                    }>"new conversation"</button>
                </p>
            </div>
        </main>
    }
}

fn main() {
    console_error_panic_hook_lite();
    leptos::mount::mount_to_body(App);
}

/// Tiny inline panic hook (avoids a dependency): log panics to the console.
fn console_error_panic_hook_lite() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
    }));
}
