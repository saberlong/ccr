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
    info!("收到 /v1/responses 请求");
    debug!(body_size = body.len(), "请求体大小");

    if body.is_empty() {
        warn!("请求体为空");
        return Err(AppError::bad_request("请求体不能为空".to_string()));
    }

    if body.len() > state.config.server.max_body_size {
        error!(
            body_size = body.len(),
            max_size = state.config.server.max_body_size,
            "请求体超过最大限制"
        );
        return Err(AppError::bad_request(format!(
            "请求体大小 ({}) 超过最大限制 ({})",
            body.len(),
            state.config.server.max_body_size
        )));
    }

    let body_str = String::from_utf8_lossy(&body);
    debug!(request_body = %body_str, "原始请求内容");

    let original_request: Value = match serde_json::from_slice(&body) {
        Ok(req) => {
            debug!("JSON 解析成功");
            req
        }
        Err(e) => {
            error!(error = %e, body = %body_str, "JSON 解析失败");
            return Err(AppError::bad_request(format!("无效的请求 JSON: {}", e)));
        }
    };

    debug!(request = ?original_request, "解析后的请求对象");

    let is_stream = original_request
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    debug!(is_stream = is_stream, "流式模式");

    let model = original_request
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    debug!(model = %model, "原始模型");

    let mapped_model = state.config.map_model(&model);
    debug!(mapped_model = %mapped_model, "映射后模型");

    debug!(model = %model, mapped_model = %mapped_model, stream = is_stream, "转换请求");

    let converted =
        match converter::convert_responses_to_chat_request(&body, &mapped_model, is_stream) {
            Ok(conv) => {
                debug!("请求转换成功");
                conv
            }
            Err(e) => {
                error!(error = %e, "请求转换失败");
                return Err(e.into());
            }
        };

    debug!(converted_body = %String::from_utf8_lossy(&converted.body), "转换后的请求体");

    debug!("发送到上游: {}", state.config.upstream.url);

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
        debug!(header_key = %key, header_value = %value, "添加自定义头");
        req = req.header(key.as_str(), value.as_str());
    }

    if let Some(content_type) = headers.get("content-type") {
        if let Ok(ct) = content_type.to_str() {
            debug!(original_content_type = %ct, "检测到原始 Content-Type (将被忽略)");
            if ct != "application/json" {
                debug!("原始 Content-Type 不是 application/json，保持使用 application/json");
            }
        }
    }

    let resp = match req.send().await {
        Ok(r) => {
            debug!(status = %r.status(), "上游响应成功");
            r
        }
        Err(e) => {
            error!(error = %e, url = %state.config.upstream.url, "上游请求失败");
            return Err(AppError::upstream_error(format!("上游请求失败: {}", e)));
        }
    };

    let status = resp.status();
    let upstream_headers = resp.headers().clone();
    debug!(upstream_status = %status, is_stream = is_stream, "处理上游响应");

    if is_stream {
        info!("进入流式响应处理");
        handle_stream_response(
            resp,
            status,
            upstream_headers,
            original_request,
            &state.config.streaming,
        )
        .await
    } else {
        info!("进入非流式响应处理");
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
    mapped_model: String,
) -> Result<Response, AppError> {
    debug!("开始读取非流式响应体");

    let body_bytes = match resp.bytes().await {
        Ok(bytes) => {
            debug!(body_size = bytes.len(), "读取上游响应成功");
            bytes
        }
        Err(e) => {
            error!(error = %e, "读取上游响应失败");
            return Err(AppError::upstream_error(format!("读取上游响应失败: {}", e)));
        }
    };

    if status.is_client_error() || status.is_server_error() {
        warn!(status = %status, "上游返回错误状态码，透传响应");
        let body_str = String::from_utf8_lossy(&body_bytes);
        debug!(error_body = %body_str, "错误响应内容");
        return Ok(forward_upstream_response(
            status,
            upstream_headers,
            body_bytes,
        ));
    }

    debug!("开始转换 Chat 响应为 Responses 格式");

    let responses_resp = converter::convert_chat_response_to_responses(
        &body_bytes,
        &original_request,
        Some(&mapped_model),
    );

    match responses_resp {
        Ok(resp_data) => {
            debug!("Chat 响应转换成功");
            let json_body = match serde_json::to_vec(&resp_data) {
                Ok(json) => {
                    debug!(response_size = json.len(), "序列化响应成功");
                    json
                }
                Err(e) => {
                    error!(error = %e, "序列化响应失败");
                    return Err(AppError::internal("序列化响应失败".to_string()));
                }
            };

            let response = match Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(Body::from(json_body))
            {
                Ok(resp) => resp,
                Err(e) => {
                    error!(error = %e, "构建响应失败");
                    return Err(AppError::internal(format!("构建响应失败: {}", e)));
                }
            };

            debug!("返回转换后的 Responses 格式响应");
            Ok(response)
        }
        Err(e) => {
            warn!(error = %e, "Chat响应转换失败，回退透传上游响应");
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
    debug!("开始处理流式响应");

    if status.is_client_error() || status.is_server_error() {
        warn!(status = %status, "上游返回错误状态码（流式）");
        let body_bytes = resp.bytes().await.unwrap_or_default();
        let error_body = String::from_utf8_lossy(&body_bytes);
        error!(upstream_status = %status, error_response = %error_body, "上游错误响应详情");
        return Ok(forward_upstream_response(
            status,
            upstream_headers,
            body_bytes,
        ));
    }

    debug!(original_request = ?original_request, "流式转换使用的原始请求");

    // 流预检测机制（如果启用）
    let mut preflight_chunk: Option<Bytes> = None;
    if streaming_config.enable_preflight {
        debug!("执行流预检测");
        let (saved_chunk, preflight_result) =
            perform_stream_preflight_check(&mut resp, streaming_config.preflight_timeout_secs).await;
        if let Some(preflight_result) = preflight_result {
            warn!(
                preflight_type = %preflight_result.detection_type,
                reason = %preflight_result.reason,
                "🚫 流预检测失败，拒绝请求"
            );

            let error_response = create_preflight_error_response(&preflight_result);
            return Ok(error_response);
        }
        preflight_chunk = saved_chunk;
        info!("✅ 流预检测通过");
    }

    let original_request = Arc::new(original_request);
    let stream_state = Arc::new(std::sync::Mutex::new(
        None::<crate::converter::stream::StreamState>,
    ));
    let completed_sent = Arc::new(AtomicBool::new(false));
    let line_buf: Arc<std::sync::Mutex<String>> =
        Arc::new(std::sync::Mutex::new(String::new()));
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

    let converted_stream = stream.filter_map(move |chunk| {
        let original_request = original_request_clone.clone();
        let stream_state = stream_state.clone();
        let completed_sent = completed_sent_clone.clone();
        let line_buf = line_buf_clone.clone();
        async move {
            match chunk {
                Ok(bytes) => {
                    let mut partial = match line_buf.lock() {
                        Ok(guard) => guard,
                        Err(e) => {
                            error!(error = %e, "获取行缓冲区锁失败");
                            return Some(Err(axum::Error::new(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                "行缓冲区锁被污染",
                            ))));
                        }
                    };
                    let chunk_text = String::from_utf8_lossy(&bytes);
                    let full_text = format!("{}{}", &*partial, chunk_text);
                    let ends_with_newline = bytes.last() == Some(&b'\n');
                    *partial = String::new();

                    debug!(chunk_size = bytes.len(), chunk_content = %chunk_text, "收到上游流数据块");

                    let mut events = Vec::new();
                    let lines: Vec<&str> = full_text.lines().collect();
                    let last_is_partial = !ends_with_newline && !lines.is_empty();

                    for (i, line) in lines.iter().enumerate() {
                        if last_is_partial && i == lines.len() - 1 {
                            partial.push_str(line);
                            debug!(partial_line = %line, "缓存不完整的 SSE 行");
                            break;
                        }
                        debug!(line = %line, "处理 SSE 行");
                        let mut state = match stream_state.lock() {
                            Ok(guard) => guard,
                            Err(e) => {
                                error!(error = %e, "获取流状态锁失败");
                                return Some(Err(axum::Error::new(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    "流状态锁被污染",
                                ))));
                            }
                        };
                        let line_events = converter::convert_chat_stream_line(
                            line,
                            &mut state,
                            &original_request,
                        );

                        // 检测 response.completed 事件
                        if line_events.iter().any(|e| e.contains("response.completed")) {
                            debug!("检测到 response.completed 事件");
                            completed_sent.store(true, Ordering::SeqCst);
                        }

                        debug!(event_count = line_events.len(), "生成的事件数量");
                        events.extend(line_events);
                    }

                    if events.is_empty() {
                        debug!("当前数据块没有生成事件");
                        None
                    } else {
                        debug!(total_events = events.len(), "返回事件流");
                        Some(Ok::<Bytes, axum::Error>(Bytes::from(events.join(""))))
                    }
                }
                Err(e) => {
                    error!(error = %e, "读取上游流数据错误");
                    Some(Err(axum::Error::new(e)))
                }
            }
        }
    });

    debug!(
        keepalive_interval_secs = streaming_config.keepalive_interval_secs,
        total_timeout_secs = streaming_config.total_timeout_secs,
        enable_usage_injection = streaming_config.enable_usage_injection,
        "创建增强版流式响应处理器"
    );

    // 使用 Box::pin 包装以支持非 Unpin 流
    let pinned_converted = Box::pin(converted_stream);

    // 使用 async_stream 宏创建带 Keep-Alive、超时控制和兜底机制的完整流
    let final_stream = create_enhanced_stream_async(
        pinned_converted,
        streaming_config,
        stream_state_for_fallback,
        original_request_for_fallback,
        completed_sent_for_fallback,
    );

    debug!("构建最终响应体");

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
            error!(error = %e, "构建流式响应失败");
            return Err(AppError::internal(format!("构建流式响应失败: {}", e)));
        }
    };

    info!(
        "✅ 流式响应处理器已启动（Keep-Alive={}s, 超时={}s, Usage注入={})",
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
            // 超时检查
            if start_time.elapsed() >= total_timeout {
                warn!(
                    elapsed_secs = start_time.elapsed().as_secs(),
                    timeout_secs = total_timeout.as_secs(),
                    "⏰ 流式响应超时，强制结束"
                );

                // 触发兜底机制
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
                // 主数据流事件
                item = inner_stream.next() => {
                    match item {
                        Some(Ok(bytes)) => {
                            // 重置 Keep-Alive 定时器
                            keepalive_timer.as_mut().reset_after(keepalive_interval);
                            yield Ok(bytes);
                        }
                        Some(Err(e)) => {
                            if is_client_disconnect_error(&e) {
                                info!("客户端断开连接（正常行为）");
                            } else {
                                error!(error = %e, "流传输错误");
                            }
                            yield Err(e);
                        }
                        None => {
                            // 流结束，触发兜底机制
                            if !completed_sent.load(Ordering::SeqCst) && !_fallback_triggered {
                                _fallback_triggered = true;
                                warn!("🔧 上游流结束但未收到 [DONE] 标记，触发兜底完成机制");

                                if let Some(fallback_data) = trigger_fallback_mechanism(
                                    &stream_state,
                                    &original_request,
                                    enable_usage_injection,
                                ) {
                                    info!("📤 兜底完成事件已发送");
                                    yield Ok(fallback_data);
                                } else {
                                    warn!("⚠️ 兜底机制未生成任何事件");
                                }
                            }

                            info!("📤 流式响应完全结束");
                            break;
                        }
                    }
                }

                // Keep-Alive 心跳
                _ = keepalive_timer.tick() => {
                    debug!("💓 发送 SSE Keep-Alive 心跳");
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
            error!(error = %e, "触发兜底机制时获取流状态锁失败");
            return None;
        }
    };

    if let Some(ref mut st) = *state {
        let mut fallback_events =
            crate::converter::stream::generate_completed_events_fallback(st, original_request);

        if !fallback_events.is_empty() && enable_usage_injection {
            // 如果启用 usage 注入，检查并修补最后一个事件的 usage 字段
            if let Some(last_event) = fallback_events.last_mut() {
                if last_event.contains("response.completed") {
                    if let Some(patched_event) = inject_usage_if_needed(last_event, st) {
                        *last_event = patched_event;
                        info!("✅ 已注入/修补 usage 字段到 response.completed 事件");
                    }
                }
            }
        }

        if !fallback_events.is_empty() {
            info!(
                "✅ 兜底机制成功 - 生成 {} 个完成事件",
                fallback_events.len()
            );
            return Some(Bytes::from(fallback_events.join("")));
        } else {
            warn!("⚠️ 兜底机制未生成任何事件");
        }
    } else {
        warn!("⚠️ StreamState 为空，无法生成兜底事件");
    }

    None
}

fn inject_usage_if_needed(
    completed_event: &str,
    st: &crate::converter::stream::StreamState,
) -> Option<String> {
    use serde_json::json;

    // 解析现有的事件 JSON
    let mut event_json: serde_json::Value = match serde_json::from_str(completed_event) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "无法解析 response.completed 事件进行 usage 注入");
            return None;
        }
    };

    // 检查是否已有有效的 usage 数据
    let has_valid_usage = event_json
        .pointer("/response/usage/output_tokens")
        .and_then(|v| v.as_i64())
        .map(|t| t > 0)
        .unwrap_or(false);

    if has_valid_usage {
        debug!("response.completed 已包含有效 usage 数据，无需注入");
        return None;
    }

    // 本地估算 token 统计
    let text_len = st.get_text_len() as i64;
    let reasoning_len = st.get_reasoning_len() as i64;

    // 简单估算：每4个字符约等于1个token
    let estimated_output_tokens = (text_len / 4).max(1);
    let estimated_reasoning_tokens = if reasoning_len > 0 {
        reasoning_len / 4
    } else {
        0
    };

    let estimated_input_tokens = st.get_input_tokens().max(10); // 至少10个input tokens

    info!(
        input_tokens = estimated_input_tokens,
        output_tokens = estimated_output_tokens,
        reasoning_tokens = estimated_reasoning_tokens,
        text_length = text_len,
        reasoning_length = reasoning_len,
        "📊 注入本地估算的 usage 数据"
    );

    // 构建 usage 对象
    let total_tokens =
        estimated_input_tokens + estimated_output_tokens + estimated_reasoning_tokens;
    let mut usage = json!({
        "input_tokens": estimated_input_tokens,
        "output_tokens": estimated_output_tokens,
        "total_tokens": total_tokens,
    });

    // 添加 reasoning tokens 详情（如果有）
    if estimated_reasoning_tokens > 0 {
        usage["output_tokens_details"] = json!({
            "reasoning_tokens": estimated_reasoning_tokens
        });
    }

    // 注入到事件中
    if let Some(resp_obj) = event_json.get_mut("response") {
        resp_obj["usage"] = usage;
    }

    match serde_json::to_string(&event_json) {
        Ok(patched) => Some(patched),
        Err(e) => {
            warn!(error = %e, "序列化修补后的事件失败");
            None
        }
    }
}

fn is_client_disconnect_error(e: &axum::Error) -> bool {
    let error_msg = e.to_string().to_lowercase();
    error_msg.contains("connection reset")
        || error_msg.contains("broken pipe")
        || error_msg.contains("client disconnected")
        || error_msg.contains("aborted")
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
            let text = String::from_utf8_lossy(&chunk).to_string();

            if text.trim().is_empty() || text.trim() == "{}" {
                return (None, Some(PreflightDetectionResult {
                    detection_type: "empty_response".to_string(),
                    reason: format!("上游返回空响应或仅包含空对象，内容长度: {}", text.len()),
                    status_code: 502,
                }));
            }

            let lower_text = text.to_lowercase();
            if lower_text.contains("invalid_api_key")
                || lower_text.contains("authentication_error")
                || lower_text.contains("unauthorized")
                || lower_text.contains("401")
                || lower_text.contains("api key") && lower_text.contains("invalid")
            {
                return (None, Some(PreflightDetectionResult {
                    detection_type: "auth_error".to_string(),
                    reason: "上游返回认证错误，API Key 无效或已过期".to_string(),
                    status_code: 502,
                }));
            }

            if lower_text.contains("insufficient_quota")
                || lower_text.contains("billing") && lower_text.contains("error")
                || lower_text.contains("quota_exceeded")
                || lower_text.contains("rate_limit")
            {
                return (None, Some(PreflightDetectionResult {
                    detection_type: "quota_error".to_string(),
                    reason: "上游返回余额不足或速率限制错误".to_string(),
                    status_code: 429,
                }));
            }

            if lower_text.contains("tool_calls")
                && (lower_text.contains("malformed") || lower_text.contains("parse_error"))
            {
                return (None, Some(PreflightDetectionResult {
                    detection_type: "malformed_tool_call".to_string(),
                    reason: "上游返回畸形工具调用格式".to_string(),
                    status_code: 500,
                }));
            }

            debug!(
                preflight_content_len = text.len(),
                "✅ 预检测通过 - 响应看起来正常"
            );
            (Some(chunk), None)
        }
        Ok(Ok(None)) => {
            warn!("上游流在预检测阶段立即结束");
            (None, Some(PreflightDetectionResult {
                detection_type: "immediate_stream_end".to_string(),
                reason: "上游流在预检测阶段立即结束，无任何数据".to_string(),
                status_code: 502,
            }))
        }
        Ok(Err(e)) => {
            warn!(error = %e, "预检测阶段读取流失败");
            (None, Some(PreflightDetectionResult {
                detection_type: "stream_read_error".to_string(),
                reason: format!("读取上游流失败: {}", e),
                status_code: 502,
            }))
        }
        Err(_) => {
            warn!(timeout_secs = timeout_secs, "⏰ 预检测超时");
            (None, None)
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
            error!(error = %e, "构建预检测错误响应失败");
            Response::builder()
                .status(500)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"error":{"type":"internal_error","message":"构建响应失败"}}"#,
                ))
                .unwrap_or_else(|_| {
                    // 最后的兜底，理论上不会执行
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
            error!(error = %e, "构建上游转发响应失败");
            Response::builder()
                .status(500)
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{"error":{"type":"internal_error","message":"构建响应失败"}}"#,
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
                error!(error = %e, "构建错误响应失败");
                axum::http::response::Response::new(Body::empty())
            }
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        error!(error = %e, "内部错误");
        AppError::internal(format!("{}", e))
    }
}
