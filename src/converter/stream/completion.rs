use serde_json::{json, Value};
use tracing::{info, warn};

use super::emit_sse_event;
use super::handlers::{close_func_blocks, close_reasoning_block, close_text_block};
use super::images::extract_image_urls_from_text;
use super::state::StreamState;

pub(crate) fn generate_completed_events(
    st: &mut StreamState,
    original_request: &Value,
) -> Vec<String> {
    warn!(
        "Generating completion events - reasoning_buf_len: {}, text_buf_len: {}, should_synthesize: {}",
        st.reasoning.buf.len(),
        st.text.buf.len(),
        st.should_synthesize_message_from_reasoning()
    );
    generate_completed_events_internal(st, original_request)
}

pub fn generate_completed_events_fallback(
    st: &mut StreamState,
    original_request: &Value,
) -> Vec<String> {
    warn!(
        "Fallback mechanism triggered - upstream did not send [DONE], sending response.completed event\n\
         status: reasoning_active={}, in_text_block={}, in_func_block={}\n\
         content: reasoning_buf_len={}, text_buf_len={}\n\
         should_synthesize_message_from_reasoning={}",
        st.reasoning.active,
        st.text.in_block,
        st.func.in_block,
        st.reasoning.buf.len(),
        st.text.buf.len(),
        st.should_synthesize_message_from_reasoning(),
    );

    let events = generate_completed_events_internal(st, original_request);

    info!(
        "Fallback completion events generated - {} events total",
        events.len()
    );

    events
}

fn generate_completed_events_internal(
    st: &mut StreamState,
    original_request: &Value,
) -> Vec<String> {
    let mut events = Vec::new();

    if st.should_synthesize_message_from_reasoning() {
        let summary = extract_summary_from_reasoning(&st.reasoning.buf);
        if !summary.is_empty() {
            let output_index = st.text.current_output_index;
            st.text.current_output_index += 1;
            st.text.msg_output_index = output_index;
            st.text.msg_id = format!("msg_{}_{}", st.session.response_id, output_index);
            st.text.buf = summary;
            st.text.in_block = true;
        }
    }

    if st.reasoning.active {
        events.extend(close_reasoning_block(st));
    }
    if st.text.in_block {
        events.extend(close_text_block(st));
    }
    if st.func.in_block {
        events.extend(close_func_blocks(st));
    }

    let status = if st.session.finish_reason.as_deref() == Some("length") {
        "incomplete"
    } else {
        "completed"
    };

    let mut completed = json!({
        "type": "response.completed",
        "sequence_number": st.next_seq(),
        "response": {
            "id": st.session.response_id,
            "object": "response",
            "created_at": st.session.created_at,
            "status": status,
            "background": false,
            "error": null,
        }
    });

    if st.session.finish_reason.as_deref() == Some("length") {
        completed["response"]["incomplete_details"] = json!({
            "reason": "max_output_tokens"
        });
    }

    if let Some(v) = original_request
        .get("instructions")
        .and_then(|v| v.as_str())
    {
        completed["response"]["instructions"] = Value::String(v.to_string());
    }
    if let Some(v) = original_request.get("model") {
        completed["response"]["model"] = v.clone();
    }
    if let Some(v) = original_request.get("max_output_tokens") {
        completed["response"]["max_output_tokens"] = v.clone();
    }
    if let Some(v) = original_request.get("parallel_tool_calls") {
        completed["response"]["parallel_tool_calls"] = v.clone();
    }
    if let Some(v) = original_request.get("previous_response_id") {
        completed["response"]["previous_response_id"] = v.clone();
    }
    if let Some(v) = original_request.get("temperature") {
        completed["response"]["temperature"] = v.clone();
    }
    if let Some(v) = original_request.get("tool_choice") {
        completed["response"]["tool_choice"] = v.clone();
    }
    if let Some(v) = original_request.get("tools") {
        completed["response"]["tools"] = v.clone();
    }
    if let Some(v) = original_request.get("top_p") {
        completed["response"]["top_p"] = v.clone();
    }
    if let Some(v) = original_request.get("reasoning") {
        completed["response"]["reasoning"] = v.clone();
    }
    if let Some(v) = original_request.get("metadata") {
        completed["response"]["metadata"] = v.clone();
    }

    let mut outputs: Vec<Value> = Vec::new();

    if !st.reasoning.buf.is_empty() || st.reasoning.part_added {
        outputs.push(json!({
            "id": st.reasoning.item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [{
                "type": "summary_text",
                "text": st.reasoning.buf,
            }]
        }));
    }

    if !st.text.buf.is_empty() || !st.text.msg_id.is_empty() {
        let image_parts = extract_image_urls_from_text(&st.text.buf);
        let mut content_parts: Vec<serde_json::Value> = vec![json!({
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": st.text.buf,
        })];
        content_parts.extend(image_parts);
        outputs.push(json!({
            "id": st.text.msg_id,
            "type": "message",
            "status": "completed",
            "content": content_parts,
            "role": "assistant"
        }));
    }

    let mut idxs: Vec<i64> = st.func.args.keys().copied().collect();
    idxs.sort();
    for idx in idxs {
        let args = st
            .func
            .args
            .get(&idx)
            .cloned()
            .unwrap_or_else(|| "{}".to_string());
        let call_id = st.func.call_ids.get(&idx).cloned().unwrap_or_default();
        let name = st.func.names.get(&idx).cloned().unwrap_or_default();

        outputs.push(json!({
            "id": format!("fc_{}", call_id),
            "type": "function_call",
            "status": "completed",
            "arguments": args,
            "call_id": call_id,
            "name": name,
        }));
    }

    if !outputs.is_empty() {
        completed["response"]["output"] = Value::Array(outputs);
    }

    let input_tokens = st.usage.input;
    let output_tokens = st.usage.output;
    let cached_tokens = st.usage.cached;
    let mut reasoning_tokens = st.usage.reasoning;

    if reasoning_tokens == 0 && !st.reasoning.buf.is_empty() {
        reasoning_tokens = (st.reasoning.buf.len() / 4) as i64;
    }

    let actual_input = if cached_tokens > 0 {
        (input_tokens - cached_tokens).max(0)
    } else {
        input_tokens
    };

    let mut total = st.usage.total;
    if total == 0 || cached_tokens > 0 {
        total = actual_input + output_tokens + cached_tokens;
    }

    completed["response"]["usage"] = json!({
        "input_tokens": actual_input,
        "output_tokens": output_tokens,
        "total_tokens": total,
    });

    if cached_tokens > 0 {
        completed["response"]["usage"]["input_tokens_details"] = json!({
            "cached_tokens": cached_tokens,
        });
    }

    if reasoning_tokens > 0 {
        completed["response"]["usage"]["output_tokens_details"] = json!({
            "reasoning_tokens": reasoning_tokens,
        });
    }

    if cached_tokens > 0 {
        completed["response"]["usage"]["cache_read_input_tokens"] = json!(cached_tokens);
    }

    events.push(emit_sse_event(
        "response.completed",
        &serde_json::to_string(&completed).unwrap_or_default(),
    ));

    events
}

const MAX_REASONING_SUMMARY_LENGTH: usize = 500;

fn extract_summary_from_reasoning(reasoning_text: &str) -> String {
    if reasoning_text.is_empty() {
        return String::new();
    }

    let trimmed = reasoning_text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.len() <= MAX_REASONING_SUMMARY_LENGTH {
        return trimmed.to_string();
    }

    let candidates = ["\n", "。", ". ", ",", "，", " "];
    let mut best_pos: i64 = -1;
    let mut best_sep_len: usize = 0;

    for sep in &candidates {
        if let Some(pos) = trimmed[..MAX_REASONING_SUMMARY_LENGTH].rfind(sep) {
            if pos as i64 > best_pos {
                best_pos = pos as i64;
                best_sep_len = sep.len();
            }
        }
    }

    if best_pos > 0 {
        let end_pos = (best_pos as usize) + best_sep_len;
        let result = &trimmed[..end_pos.min(trimmed.len())];
        return result.trim().to_string();
    }

    trimmed[..MAX_REASONING_SUMMARY_LENGTH].to_string()
}
