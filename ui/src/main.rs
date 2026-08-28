//! Field Assistant — all-client-side browser UI (Leptos CSR → WASM).
//!
//! Pattern: paste three values (deployment URL, deployment bearer token,
//! OpenAI key), all persisted to localStorage and sent nowhere except to
//! those two APIs, directly from the browser. Mic → OpenAI transcription →
//! deployment kickoff (fresh id per turn, chained via restoreFromStateId) →
//! poll → reply, optionally spoken via the browser's speech synthesis.

use std::cell::RefCell;
use std::rc::Rc;

use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

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
// Deployment API (kickoff → poll)
// ---------------------------------------------------------------------------

async fn kickoff(
    cfg: &Settings,
    turn_id: &str,
    restore_from: Option<&str>,
    message: &str,
) -> Result<String, String> {
    let url = format!("{}/kickoff", cfg.deployment_url.trim_end_matches('/'));
    // Chain recipe: the platform deprecates reusing an id across kickoffs, so
    // every turn gets a FRESH id and continuity comes from restoreFromStateId
    // (top-level, beside "inputs") pointing at the previous turn's id.
    let mut body = serde_json::json!({ "inputs": { "id": turn_id, "message": message } });
    if let Some(prev) = restore_from {
        body["restoreFromStateId"] = serde_json::json!(prev);
    }
    let resp = gloo_net::http::Request::post(&url)
        .header("Authorization", &format!("Bearer {}", cfg.deployment_token))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| format!("kickoff request failed: {e} (if this is a CORS error, use client/ask.py)"))?;
    if !resp.ok() {
        return Err(format!("kickoff HTTP {}: {}", resp.status(),
                           resp.text().await.unwrap_or_default()));
    }
    let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    ["kickoff_id", "id", "execution_id"]
        .iter()
        .find_map(|k| v.get(*k).and_then(|x| x.as_str()).map(String::from))
        .ok_or_else(|| format!("no kickoff id in response: {v}"))
}

async fn poll_result(cfg: &Settings, kickoff_id: &str) -> Result<String, String> {
    let url = format!("{}/status/{}", cfg.deployment_url.trim_end_matches('/'), kickoff_id);
    // Immediate first check, then backoff: 100ms → 150 → 225 → cap 400ms.
    let mut delay_ms: u32 = 100;
    for _ in 0..400 {
        let resp = gloo_net::http::Request::get(&url)
            .header("Authorization", &format!("Bearer {}", cfg.deployment_token))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let state = v.get("state").or_else(|| v.get("status"))
            .and_then(|s| s.as_str()).unwrap_or("").to_uppercase();
        match state.as_str() {
            "SUCCESS" | "SUCCEEDED" | "COMPLETED" | "COMPLETE" | "FINISHED" => {
                return Ok(v.get("result").and_then(|r| r.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| v.to_string()));
            }
            "FAILED" | "FAILURE" | "ERROR" | "CANCELLED" => {
                return Err(format!("run ended in {state}: {v}"));
            }
            _ => {
                gloo_timers::future::TimeoutFuture::new(delay_ms).await;
                delay_ms = delay_ms.saturating_mul(3).saturating_div(2).min(400);
            }
        }
    }
    Err("timed out waiting for the run".into())
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
    // Last successful turn's state id — the conversation's chain head.
    // None = fresh conversation; advanced only after a turn SUCCEEDS (a failed
    // turn may have persisted nothing; restoring from it would silently drop
    // the whole context).
    let (chain_head, set_chain_head) = signal(Option::<String>::None);
    let (busy, set_busy) = signal(false);
    let (status, set_status) = signal(String::new());
    let (recording, set_recording) = signal(false);
    let (draft, set_draft) = signal(String::new());
    let (show_settings, set_show_settings) = signal(!configured);
    let (timing, set_timing) = signal(String::new());

    let recorder: Rc<RefCell<Option<web_sys::MediaRecorder>>> = Rc::new(RefCell::new(None));

    // one user turn: send text → kickoff → poll → append reply (+ speak)
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
        set_status.set("kicking off…".into());
        let prev = chain_head.get_untracked();
        let turn_id = uuid::Uuid::new_v4().to_string();
        spawn_local(async move {
            let amp_t0 = now_ms();
            let outcome = match kickoff(&cfg, &turn_id, prev.as_deref(), &text).await {
                Ok(kid) => {
                    set_status.set(format!("running ({kid})…"));
                    poll_result(&cfg, &kid).await
                }
                Err(e) => Err(e),
            };
            let amp_ms = now_ms() - amp_t0;
            match outcome {
                Ok(reply) => {
                    set_chain_head.set(Some(turn_id.clone()));
                    let speak_ms = if cfg.speak_replies {
                        set_status.set("speaking…".into());
                        Some(speak_until_start(&reply).await)
                    } else {
                        None
                    };
                    set_timing.set(format_timing(stt_ms, amp_ms, speak_ms));
                    set_msgs.update(|m| m.push(Msg { role: "assistant", text: reply }));
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
                    {concat!("build v", env!("CARGO_PKG_VERSION"), " · chain head: ")}
                    {move || chain_head.get().unwrap_or_else(|| "(new conversation)".into())}
                    <button class="ghost" on:click=move |_| {
                        set_chain_head.set(None);
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
