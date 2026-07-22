use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::converter;
use crate::utf8_stream::Utf8StreamBuffer;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub client: reqwest::Client,
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn responses_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    info!("Received /v1/responses request");
    debug!(body_size = body.len(), "Request body size");

    if body.is_empty() {
        warn!("Request body is empty");
        return Err(AppError::bad_request(
            "Request body cannot be empty".to_string(),
        ));
    }

    let body_str = String::from_utf8_lossy(&body);
    debug!(request_body = %body_str, "Raw request content");

    let original_request: Value = match serde_json::from_slice(&body) {
        Ok(req) => {
            debug!("JSON parsed successfully");
            req
        }
        Err(e) => {
            error!(error = %e, body = %body_str, "JSON parse failed");
            return Err(AppError::bad_request(format!(
                "Invalid request JSON: {}",
                e
            )));
        }
    };

    debug!(request = ?original_request, "Parsed request object");

    let is_stream = original_request
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    debug!(is_stream = is_stream, "Stream mode");

    let model = original_request
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    debug!(model = %model, "Original model");

    let mapped_model = state.config.map_model(&model);
    debug!(mapped_model = %mapped_model, "Mapped model");

    debug!(model = %model, mapped_model = %mapped_model, stream = is_stream, "Converting request");

    let converted =
        match converter::convert_responses_to_chat_request(&body, mapped_model, is_stream) {
            Ok(conv) => {
                debug!("Request conversion succeeded");
                conv
            }
            Err(e) => {
                error!(error = %e, "Request conversion failed");
                return Err(e.into());
            }
        };

    debug!(converted_body = %String::from_utf8_lossy(&converted.body), "Converted request body");

    debug!("Sending to upstream: {}", state.config.upstream.url);

    let mut req = state
        .client
        .post(&state.config.upstream.url)
        .header(
            "Authorization",
            format!("Bearer {}", state.config.upstream.api_key),
        )
        .header("Content-Type", "application/json")
        .body(converted.body.clone());

    for (key, value) in &state.config.upstream.extra_headers {
        debug!(header_key = %key, header_value = %value, "Adding custom header");
        req = req.header(key.as_str(), value.as_str());
    }

    if let Some(content_type) = headers.get("content-type") {
        if let Ok(ct) = content_type.to_str() {
            debug!(original_content_type = %ct, "Detected original Content-Type (will be ignored)");
            if ct != "application/json" {
                debug!("Original Content-Type is not application/json, keeping application/json");
            }
        }
    }

    let resp = match req.send().await {
        Ok(r) => {
            debug!(status = %r.status(), "Upstream response succeeded");
            r
        }
        Err(e) => {
            error!(error = %e, url = %state.config.upstream.url, "Upstream request failed");
            return Err(AppError::upstream_error(format!(
                "Upstream request failed: {}",
                e
            )));
        }
    };

    let status = resp.status();
    let upstream_headers = resp.headers().clone();
    debug!(upstream_status = %status, is_stream = is_stream, "Processing upstream response");

    if is_stream {
        info!("Entering stream response handling");
        handle_stream_response(
            resp,
            status,
            upstream_headers,
            original_request,
            &state.config.streaming,
        )
        .await
    } else {
        info!("Entering non-stream response handling");
        handle_non_stream_response(
            resp,
            status,
            upstream_headers,
            original_request,
            mapped_model,
        )
        .await
    }
}

async fn handle_non_stream_response(
    resp: reqwest::Response,
    status: StatusCode,
    upstream_headers: HeaderMap,
    original_request: Value,
    mapped_model: &str,
) -> Result<Response, AppError> {
    debug!("Reading non-stream response body");

    let body_bytes = match resp.bytes().await {
        Ok(bytes) => {
            debug!(
                body_size = bytes.len(),
                "Upstream response body read successfully"
            );
            bytes
        }
        Err(e) => {
            error!(error = %e, "Failed to read upstream response body");
            return Err(AppError::upstream_error(format!(
                "Failed to read upstream response body: {}",
                e
            )));
        }
    };

    if status.is_client_error() || status.is_server_error() {
        warn!(status = %status, "Upstream returned error status, passing through response");
        let body_str = String::from_utf8_lossy(&body_bytes);
        debug!(error_body = %body_str, "Error response content");
        return Ok(forward_upstream_response(
            status,
            upstream_headers,
            body_bytes,
        ));
    }

    debug!("Converting Chat response to Responses format");

    let responses_resp = converter::convert_chat_response_to_responses(
        &body_bytes,
        &original_request,
        Some(mapped_model),
    );

    match responses_resp {
        Ok(resp_data) => {
            debug!("Chat response conversion succeeded");
            let json_body = match serde_json::to_vec(&resp_data) {
                Ok(json) => {
                    debug!(
                        response_size = json.len(),
                        "Response serialized successfully"
                    );
                    json
                }
                Err(e) => {
                    error!(error = %e, "Response serialization failed");
                    return Err(AppError::internal(
                        "Response serialization failed".to_string(),
                    ));
                }
            };

            let response = match Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(Body::from(json_body))
            {
                Ok(resp) => resp,
                Err(e) => {
                    error!(error = %e, "Failed to build response");
                    return Err(AppError::internal(format!(
                        "Failed to build response: {}",
                        e
                    )));
                }
            };

            debug!("Returning converted Responses format response");
            Ok(response)
        }
        Err(e) => {
            warn!(error = %e, "Chat response conversion failed, falling back to upstream pass-through");
            Ok(forward_upstream_response(
                status,
                upstream_headers,
                body_bytes,
            ))
        }
    }
}

async fn handle_stream_response(
    mut resp: reqwest::Response,
    status: StatusCode,
    upstream_headers: HeaderMap,
    original_request: Value,
    streaming_config: &crate::config::StreamingConfig,
) -> Result<Response, AppError> {
    debug!("Processing stream response");

    if status.is_client_error() || status.is_server_error() {
        warn!(status = %status, "Upstream returned error status (stream)");
        let body_bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "Failed to read upstream error response body");
                return Err(AppError::upstream_error(
                    "Failed to read upstream response body".to_string(),
                ));
            }
        };
        let error_body = String::from_utf8_lossy(&body_bytes);
        error!(upstream_status = %status, error_response = %error_body, "Upstream error response details");
        return Ok(forward_upstream_response(
            status,
            upstream_headers,
            body_bytes,
        ));
    }

    debug!(original_request = ?original_request, "Original request for stream conversion");

    // Stream preflight check (if enabled)
    let mut preflight_chunk: Option<Bytes> = None;
    if streaming_config.enable_preflight {
        debug!("Running stream preflight check");
        let (saved_chunk, preflight_result) =
            perform_stream_preflight_check(&mut resp, streaming_config.preflight_timeout_secs)
                .await;
        if let Some(preflight_result) = preflight_result {
            warn!(
                preflight_type = %preflight_result.detection_type,
                reason = %preflight_result.reason,
                "Stream preflight check failed, rejecting request"
            );

            let error_response = create_preflight_error_response(&preflight_result);
            return Ok(error_response);
        }
        preflight_chunk = saved_chunk;
        info!("Stream preflight check passed");
    }

    let original_request = Arc::new(original_request);
    let stream_state = Arc::new(std::sync::Mutex::new(
        None::<crate::converter::stream::StreamState>,
    ));
    let completed_sent = Arc::new(AtomicBool::new(false));
    let line_buf: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    let utf8_buf: Arc<std::sync::Mutex<Utf8StreamBuffer>> =
        Arc::new(std::sync::Mutex::new(Utf8StreamBuffer::new()));
    let mut initial_chunks: Vec<Result<Bytes, reqwest::Error>> = Vec::new();
    if let Some(chunk) = preflight_chunk {
        initial_chunks.push(Ok(chunk));
    }
    let stream = futures::stream::iter(initial_chunks).chain(resp.bytes_stream());
    let original_request_clone = original_request.clone();
    let completed_sent_clone = completed_sent.clone();
    let stream_state_for_fallback = stream_state.clone();
    let original_request_for_fallback = original_request.clone();
    let completed_sent_for_fallback = completed_sent.clone();
    let line_buf_clone = line_buf.clone();
    let utf8_buf_clone = utf8_buf.clone();

    let converted_stream = stream.filter_map(move |chunk| {
        let original_request = original_request_clone.clone();
        let stream_state = stream_state.clone();
        let completed_sent = completed_sent_clone.clone();
        let line_buf = line_buf_clone.clone();
        let utf8_buf = utf8_buf_clone.clone();
        async move {
            match chunk {
                Ok(bytes) => {
                    let mut partial = match line_buf.lock() {
                        Ok(guard) => guard,
                        Err(e) => {
                            error!(error = %e, "Failed to acquire line buffer lock");
                            return Some(Err(axum::Error::new(std::io::Error::other(
                                "Line buffer lock poisoned",
                            ))));
                        }
                    };
                    let mut utf8_guard = match utf8_buf.lock() {
                        Ok(guard) => guard,
                        Err(e) => {
                            error!(error = %e, "Failed to acquire UTF-8 buffer lock");
                            return Some(Err(axum::Error::new(std::io::Error::other(
                                "UTF-8 buffer lock poisoned",
                            ))));
                        }
                    };
                    let chunk_text = utf8_guard.process_bytes(&bytes);
                    drop(utf8_guard);
                    let full_text = format!("{}{}", &*partial, chunk_text);
                    let ends_with_newline = bytes.last() == Some(&b'\n');
                    *partial = String::new();

                    debug!(chunk_size = bytes.len(), chunk_content = %chunk_text, "Received upstream stream data chunk");

                    let mut events = Vec::new();
                    let lines: Vec<&str> = full_text.lines().collect();
                    let last_is_partial = !ends_with_newline && !lines.is_empty();

                    for (i, line) in lines.iter().enumerate() {
                        if last_is_partial && i == lines.len() - 1 {
                            partial.push_str(line);
                            debug!(partial_line = %line, "Buffering incomplete SSE line");
                            break;
                        }
                        debug!(line = %line, "Processing SSE line");
                        let mut state_guard = match stream_state.lock() {
                            Ok(guard) => guard,
                            Err(e) => {
                                error!(error = %e, "Failed to acquire stream state lock");
                                return Some(Err(axum::Error::new(std::io::Error::other(
                                    "Stream state lock poisoned",
                                ))));
                            }
                        };
                        let state = state_guard.get_or_insert_with(converter::stream::StreamState::new);
                        let line_events = converter::convert_chat_stream_line(
                            line,
                            state,
                            &original_request,
                        );

                        // Detect response.completed event
                        if line_events.iter().any(|e| e.contains("response.completed")) {
                            debug!("Detected response.completed event");
                            completed_sent.store(true, Ordering::SeqCst);
                        }

                        debug!(event_count = line_events.len(), "Generated event count");
                        events.extend(line_events);
                    }

                    if events.is_empty() {
                        debug!("No events generated from current data chunk");
                        None
                    } else {
                        debug!(total_events = events.len(), "Returning event stream");
                        Some(Ok::<Bytes, axum::Error>(Bytes::from(events.join(""))))
                    }
                }
                Err(e) => {
                    error!(error = %e, "Error reading upstream stream data");
                    Some(Err(axum::Error::new(e)))
                }
            }
        }
    });

    debug!(
        keepalive_interval_secs = streaming_config.keepalive_interval_secs,
        total_timeout_secs = streaming_config.total_timeout_secs,
        enable_usage_injection = streaming_config.enable_usage_injection,
        "Creating enhanced stream response handler"
    );

    // Use Box::pin to wrap non-Unpin stream
    let pinned_converted = Box::pin(converted_stream);

    // Use async_stream macro to create complete stream with Keep-Alive, timeout control, and fallback mechanism
    let final_stream = create_enhanced_stream_async(
        pinned_converted,
        streaming_config,
        stream_state_for_fallback,
        original_request_for_fallback,
        completed_sent_for_fallback,
    );

    debug!("Building final response body");

    let body = Body::from_stream(Box::pin(final_stream));

    let response = match Response::builder()
        .status(200)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(body)
    {
        Ok(resp) => resp,
        Err(e) => {
            error!(error = %e, "Failed to build streaming response");
            return Err(AppError::internal(format!(
                "Failed to build streaming response: {}",
                e
            )));
        }
    };

    info!(
        "Stream response handler started (Keep-Alive={}s, Timeout={}s, Usage injection={})",
        streaming_config.keepalive_interval_secs,
        streaming_config.total_timeout_secs,
        streaming_config.enable_usage_injection
    );
    Ok(response)
}

fn create_enhanced_stream_async(
    mut inner_stream: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, axum::Error>> + Send>>,
    config: &crate::config::StreamingConfig,
    stream_state: Arc<std::sync::Mutex<Option<crate::converter::stream::StreamState>>>,
    original_request: Arc<Value>,
    completed_sent: Arc<AtomicBool>,
) -> impl Stream<Item = Result<Bytes, axum::Error>> {
    use async_stream::stream;

    let keepalive_interval = Duration::from_secs(config.keepalive_interval_secs);
    let total_timeout = Duration::from_secs(config.total_timeout_secs);
    let enable_usage_injection = config.enable_usage_injection;

    stream! {
        let keepalive_timer = interval(keepalive_interval);
        tokio::pin!(keepalive_timer);
        let start_time = std::time::Instant::now();
        let mut _fallback_triggered = false;

        loop {
            // Timeout check
            if start_time.elapsed() >= total_timeout {
                warn!(
                    elapsed_secs = start_time.elapsed().as_secs(),
                    timeout_secs = total_timeout.as_secs(),
                    "Stream response timeout, forcing end"
                );

                // Trigger fallback mechanism
                if !completed_sent.load(Ordering::SeqCst) && !_fallback_triggered {
                    _fallback_triggered = true;
                    if let Some(fallback_data) = trigger_fallback_mechanism(
                        &stream_state,
                        &original_request,
                        enable_usage_injection,
                    ) {
                        yield Ok(fallback_data);
                    }
                }
                break;
            }

            tokio::select! {
                // Main data stream event
                item = inner_stream.next() => {
                    match item {
                        Some(Ok(bytes)) => {
                            // Reset Keep-Alive timer
                            keepalive_timer.as_mut().reset_after(keepalive_interval);
                            yield Ok(bytes);
                        }
                        Some(Err(e)) => {
                            if is_client_disconnect_error(&e) {
                                info!("Client disconnected (normal behavior)");
                            } else {
                                error!(error = %e, "Stream transport error");
                            }
                            yield Err(e);
                        }
                        None => {
                            // Stream ended, trigger fallback mechanism
                            if !completed_sent.load(Ordering::SeqCst) && !_fallback_triggered {
                                _fallback_triggered = true;
                                warn!("Upstream stream ended without [DONE] marker, triggering fallback completion");

                                if let Some(fallback_data) = trigger_fallback_mechanism(
                                    &stream_state,
                                    &original_request,
                                    enable_usage_injection,
                                ) {
                                    info!("Fallback completion event sent");
                                    yield Ok(fallback_data);
                                } else {
                                    warn!("Fallback mechanism generated no events");
                                }
                            }

                            info!("Stream response fully ended");
                            break;
                        }
                    }
                }

                // Keep-Alive heartbeat
                _ = keepalive_timer.tick() => {
                    debug!("Sending SSE Keep-Alive heartbeat");
                    let keepalive_msg = Bytes::from_static(b": keepalive\n\n");
                    yield Ok(keepalive_msg);
                }
            }
        }
    }
}

fn trigger_fallback_mechanism(
    stream_state: &Arc<std::sync::Mutex<Option<crate::converter::stream::StreamState>>>,
    original_request: &Arc<Value>,
    enable_usage_injection: bool,
) -> Option<Bytes> {
    let mut state = match stream_state.lock() {
        Ok(guard) => guard,
        Err(e) => {
            error!(error = %e, "Failed to acquire stream state lock during fallback");
            return None;
        }
    };

    if let Some(ref mut st) = *state {
        let mut fallback_events =
            crate::converter::stream::generate_completed_events_fallback(st, original_request);

        if !fallback_events.is_empty() && enable_usage_injection {
            // If usage injection enabled, check and patch usage field of last event
            if let Some(last_event) = fallback_events.last_mut() {
                if last_event.contains("response.completed") {
                    if let Some(patched_event) = inject_usage_if_needed(last_event, st) {
                        *last_event = patched_event;
                        info!("Injected/patched usage field into response.completed event");
                    }
                }
            }
        }

        if !fallback_events.is_empty() {
            info!(
                "Fallback mechanism succeeded - generated {} completion events",
                fallback_events.len()
            );
            return Some(Bytes::from(fallback_events.join("")));
        } else {
            warn!("Fallback mechanism generated no events");
        }
    } else {
        warn!("StreamState is empty, cannot generate fallback events");
    }

    None
}

fn inject_usage_if_needed(
    completed_event: &str,
    st: &crate::converter::stream::StreamState,
) -> Option<String> {
    use serde_json::json;

    // Parse existing event JSON
    let mut event_json: serde_json::Value = match serde_json::from_str(completed_event) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "Cannot parse response.completed event for usage injection");
            return None;
        }
    };

    // Check if valid usage data already exists
    let has_valid_usage = event_json
        .pointer("/response/usage/output_tokens")
        .and_then(|v| v.as_i64())
        .map(|t| t > 0)
        .unwrap_or(false);

    if has_valid_usage {
        debug!("response.completed already contains valid usage data, no injection needed");
        return None;
    }

    // Local token estimation
    let text_len = st.get_text_len() as i64;
    let reasoning_len = st.get_reasoning_len() as i64;

    // Simple estimate: ~1 token per 4 characters
    let estimated_output_tokens = (text_len / 4).max(1);
    let estimated_reasoning_tokens = if reasoning_len > 0 {
        reasoning_len / 4
    } else {
        0
    };

    let estimated_input_tokens = st.get_input_tokens().max(10); // At least 10 input tokens

    info!(
        input_tokens = estimated_input_tokens,
        output_tokens = estimated_output_tokens,
        reasoning_tokens = estimated_reasoning_tokens,
        text_length = text_len,
        reasoning_length = reasoning_len,
        "Injecting locally estimated usage data"
    );

    // Build usage object
    let total_tokens =
        estimated_input_tokens + estimated_output_tokens + estimated_reasoning_tokens;
    let mut usage = json!({
        "input_tokens": estimated_input_tokens,
        "output_tokens": estimated_output_tokens,
        "total_tokens": total_tokens,
    });

    // Add reasoning tokens details (if any)
    if estimated_reasoning_tokens > 0 {
        usage["output_tokens_details"] = json!({
            "reasoning_tokens": estimated_reasoning_tokens
        });
    }

    // Inject into event
    if let Some(resp_obj) = event_json.get_mut("response") {
        resp_obj["usage"] = usage;
    }

    match serde_json::to_string(&event_json) {
        Ok(patched) => Some(patched),
        Err(e) => {
            warn!(error = %e, "Failed to serialize patched event");
            None
        }
    }
}

fn is_client_disconnect_error(e: &axum::Error) -> bool {
    let error_msg = e.to_string();
    let lower = error_msg.to_ascii_lowercase();
    lower.contains("connection reset")
        || lower.contains("broken pipe")
        || lower.contains("client disconnected")
        || lower.contains("aborted")
}

#[derive(Debug, Clone)]
struct PreflightDetectionResult {
    detection_type: String,
    reason: String,
    status_code: u16,
}

async fn perform_stream_preflight_check(
    resp: &mut reqwest::Response,
    timeout_secs: u64,
) -> (Option<Bytes>, Option<PreflightDetectionResult>) {
    use tokio::time::timeout as tokio_timeout;

    let preflight_timeout = Duration::from_secs(timeout_secs);

    let initial_bytes_result = tokio_timeout(preflight_timeout, resp.chunk()).await;

    match initial_bytes_result {
        Ok(Ok(Some(chunk))) => {
            let mut preflight_utf8 = Utf8StreamBuffer::new();
            let text = preflight_utf8.process_bytes(&chunk);

            if text.trim().is_empty() || text.trim() == "{}" {
                return (
                    None,
                    Some(PreflightDetectionResult {
                        detection_type: "empty_response".to_string(),
                        reason: format!(
                            "Upstream returned empty response or empty object, content length: {}",
                            text.len()
                        ),
                        status_code: 502,
                    }),
                );
            }

            let lower_text = text.to_lowercase();
            if lower_text.contains("invalid_api_key")
                || lower_text.contains("authentication_error")
                || lower_text.contains("unauthorized")
                || lower_text.contains("401")
                || lower_text.contains("api key") && lower_text.contains("invalid")
            {
                return (
                    None,
                    Some(PreflightDetectionResult {
                        detection_type: "auth_error".to_string(),
                        reason:
                            "Upstream returned authentication error, API key invalid or expired"
                                .to_string(),
                        status_code: 502,
                    }),
                );
            }

            if lower_text.contains("insufficient_quota")
                || lower_text.contains("billing") && lower_text.contains("error")
                || lower_text.contains("quota_exceeded")
                || lower_text.contains("rate_limit")
            {
                return (
                    None,
                    Some(PreflightDetectionResult {
                        detection_type: "quota_error".to_string(),
                        reason: "Upstream returned quota exceeded or rate limit error".to_string(),
                        status_code: 429,
                    }),
                );
            }

            if lower_text.contains("tool_calls")
                && (lower_text.contains("malformed") || lower_text.contains("parse_error"))
            {
                return (
                    None,
                    Some(PreflightDetectionResult {
                        detection_type: "malformed_tool_call".to_string(),
                        reason: "Upstream returned malformed tool call format".to_string(),
                        status_code: 500,
                    }),
                );
            }

            debug!(
                preflight_content_len = text.len(),
                "Preflight passed - response looks normal"
            );
            (Some(chunk), None)
        }
        Ok(Ok(None)) => {
            warn!("Upstream stream ended immediately during preflight");
            (
                None,
                Some(PreflightDetectionResult {
                    detection_type: "immediate_stream_end".to_string(),
                    reason: "Upstream stream ended immediately during preflight, no data"
                        .to_string(),
                    status_code: 502,
                }),
            )
        }
        Ok(Err(e)) => {
            warn!(error = %e, "Stream read failed during preflight");
            (
                None,
                Some(PreflightDetectionResult {
                    detection_type: "stream_read_error".to_string(),
                    reason: format!("Failed to read upstream stream: {}", e),
                    status_code: 502,
                }),
            )
        }
        Err(_) => {
            warn!(
                timeout_secs = timeout_secs,
                "Preflight timeout, upstream response too slow"
            );
            (
                None,
                Some(PreflightDetectionResult {
                    detection_type: "preflight_timeout".to_string(),
                    reason: format!(
                        "Preflight timeout ({}s), upstream response too slow",
                        timeout_secs
                    ),
                    status_code: 504,
                }),
            )
        }
    }
}

fn create_preflight_error_response(result: &PreflightDetectionResult) -> Response {
    use serde_json::json;

    let error_body = json!({
        "error": {
            "type": result.detection_type.clone(),
            "message": result.reason.clone(),
            "code": result.status_code
        },
        "type": "error",
        "id": uuid::Uuid::new_v4().to_string()
    });

    match Response::builder()
        .status(result.status_code)
        .header("Content-Type", "application/json")
        .body(Body::from(error_body.to_string()))
    {
        Ok(resp) => resp,
        Err(e) => {
            error!(error = %e, "Failed to build preflight error response");
            Response::builder()
                .status(500)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"error":{"type":"internal_error","message":"Failed to build response"}}"#,
                ))
                .unwrap_or_else(|_| {
                    // Final fallback, should not be reached in theory
                    axum::http::response::Response::new(Body::empty())
                })
        }
    }
}

fn forward_upstream_response(
    status: StatusCode,
    upstream_headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    let mut response = Response::builder().status(status);

    if let Some(ct) = upstream_headers.get("content-type") {
        if let Ok(v) = ct.to_str() {
            response = response.header("Content-Type", v);
        }
    }

    match response.body(Body::from(body_bytes)) {
        Ok(resp) => resp,
        Err(e) => {
            error!(error = %e, "Failed to build upstream forward response");
            Response::builder()
                .status(500)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"error":{"type":"internal_error","message":"Failed to build response"}}"#,
                ))
                .unwrap_or_else(|_| axum::http::response::Response::new(Body::empty()))
        }
    }
}

#[derive(Debug)]
pub struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg,
        }
    }

    fn upstream_error(msg: String) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: msg,
        }
    }

    fn internal(msg: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "type": "api_error",
                "message": self.message,
            }
        });
        let json = serde_json::to_vec(&body).unwrap_or_default();
        match Response::builder()
            .status(self.status)
            .header("Content-Type", "application/json")
            .body(Body::from(json))
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to build error response");
                axum::http::response::Response::new(Body::empty())
            }
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        error!(error = %e, "Internal error");
        AppError::internal(format!("{}", e))
    }
}
