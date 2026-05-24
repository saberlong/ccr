use chrono::Utc;
use serde_json::{json, Value};
use tracing::{error, warn};
use uuid::Uuid;

use super::completion::generate_completed_events;
use super::emit_sse_event;
use super::handlers::*;
use super::state::StreamState;
use crate::types::ChatStreamChunk;

pub fn convert_chat_stream_line(
    line: &str,
    state: &mut StreamState,
    original_request: &Value,
) -> Vec<String> {
    let st = state;

    let trimmed = line.trim();
    if trimmed.is_empty() {
        return vec![];
    }

    if !trimmed.starts_with("data:") {
        return vec![];
    }

    let json_str = match trimmed.strip_prefix("data:") {
        Some(s) => s.trim(),
        None => {
            error!("SSE data format error: missing data: prefix");
            return vec![];
        }
    };

    if json_str == "[DONE]" {
        return generate_completed_events(st, original_request);
    }

    let chunk: ChatStreamChunk = match serde_json::from_str(json_str) {
        Ok(c) => c,
        Err(e) => {
            let preview = if json_str.len() > 500 {
                &json_str[..500]
            } else {
                json_str
            };
            warn!(
                error = %e,
                json_str = %preview,
                "SSE data line JSON parse failed, skipping line"
            );
            return vec![];
        }
    };

    let mut events: Vec<String> = Vec::new();

    if st.session.first_chunk {
        st.session.first_chunk = false;
        st.session.response_id = chunk
            .id
            .clone()
            .unwrap_or_else(|| format!("resp_{}", Uuid::new_v4().simple()));
        st.session.created_at = chunk.created.unwrap_or_else(|| Utc::now().timestamp());

        st.session.seq = 0;

        let created = json!({
            "type": "response.created",
            "sequence_number": st.next_seq(),
            "response": {
                "id": st.session.response_id,
                "object": "response",
                "created_at": st.session.created_at,
                "status": "in_progress",
                "background": false,
                "error": null,
                "instructions": ""
            }
        });
        events.push(emit_sse_event(
            "response.created",
            &serde_json::to_string(&created).unwrap_or_default(),
        ));

        let in_progress = json!({
            "type": "response.in_progress",
            "sequence_number": st.next_seq(),
            "response": {
                "id": st.session.response_id,
                "object": "response",
                "created_at": st.session.created_at,
                "status": "in_progress"
            }
        });
        events.push(emit_sse_event(
            "response.in_progress",
            &serde_json::to_string(&in_progress).unwrap_or_default(),
        ));
    }

    if let Some(usage) = &chunk.usage {
        st.usage.seen = true;
        if let Some(v) = usage.prompt_tokens {
            st.usage.input = v;
        }
        if let Some(v) = usage.completion_tokens {
            st.usage.output = v;
        }
        if let Some(v) = usage.total_tokens {
            st.usage.total = v;
        }
        if let Some(details) = &usage.prompt_tokens_details {
            if let Some(v) = details.cached_tokens {
                st.usage.cached = v;
            }
        }
        if let Some(details) = &usage.completion_tokens_details {
            if let Some(v) = details.reasoning_tokens {
                st.usage.reasoning = v;
            }
        }
    }

    if let Some(choices) = &chunk.choices {
        for choice in choices {
            if let Some(delta) = &choice.delta {
                if let Some(reasoning) = &delta.reasoning_content {
                    if !reasoning.is_empty() {
                        events.extend(handle_reasoning_part(st, reasoning));
                    }
                }

                if let Some(content) = &delta.content {
                    if !content.is_empty() {
                        let (reasoning_parts, content_parts) = st.feed_think_tag(content);
                        for rp in reasoning_parts {
                            if !rp.is_empty() {
                                if st.reasoning.active {
                                    events.extend(close_reasoning_block(st));
                                }
                                events.extend(handle_reasoning_part(st, &rp));
                            }
                        }
                        for cp in content_parts {
                            if !cp.is_empty() {
                                if st.reasoning.active {
                                    events.extend(close_reasoning_block(st));
                                }
                                events.extend(handle_content_part(st, &cp));
                            }
                        }
                    }
                }

                if let Some(tool_calls) = &delta.tool_calls {
                    if st.reasoning.active {
                        events.extend(close_reasoning_block(st));
                    }
                    if st.text.in_block {
                        events.extend(close_text_block(st));
                    }
                    for tc in tool_calls {
                        let idx = tc.index;

                        st.func.args.entry(idx).or_default();
                        st.func.names.entry(idx).or_default();
                        st.func.call_ids.entry(idx).or_default();

                        if let Some(id) = &tc.id {
                            if !id.is_empty() {
                                st.func.call_ids.insert(idx, id.clone());
                            }
                        }

                        let mut func_args_delta: Option<String> = None;
                        if let Some(func) = &tc.function {
                            if let Some(name) = &func.name {
                                if !name.is_empty() {
                                    st.func.names.insert(idx, name.clone());
                                }
                            }
                            if let Some(args) = &func.arguments {
                                if !args.is_empty() {
                                    if let Some(buf) = st.func.args.get_mut(&idx) {
                                        buf.push_str(args);
                                    }
                                    func_args_delta = Some(args.clone());
                                }
                            }
                        }

                        let has_id = st
                            .func
                            .call_ids
                            .get(&idx)
                            .map(|s| !s.is_empty())
                            .unwrap_or(false);
                        let has_name = st
                            .func
                            .names
                            .get(&idx)
                            .map(|s| !s.is_empty())
                            .unwrap_or(false);
                        if has_id && has_name && !st.func.item_added.contains(&idx) {
                            st.session.fc_id =
                                st.func.call_ids.get(&idx).cloned().unwrap_or_default();
                            st.func.in_block = true;
                            events.extend(add_func_item(st, idx));
                        }

                        if let Some(args) = func_args_delta {
                            if !st.func.item_added.contains(&idx) {
                                events.extend(add_func_item(st, idx));
                            }

                            let output_index =
                                st.func.output_index.get(&idx).copied().unwrap_or_default();
                            let item_id = format!(
                                "fc_{}",
                                st.func
                                    .call_ids
                                    .get(&idx)
                                    .cloned()
                                    .unwrap_or_else(|| format!("fc_{}", Uuid::new_v4().simple()))
                            );
                            let delta = json!({
                                "type": "response.function_call_arguments.delta",
                                "sequence_number": st.next_seq(),
                                "item_id": item_id,
                                "output_index": output_index,
                                "delta": args,
                            });
                            events.push(emit_sse_event(
                                "response.function_call_arguments.delta",
                                &serde_json::to_string(&delta).unwrap_or_default(),
                            ));
                        }
                    }
                }
            }

            if let Some(finish_reason) = &choice.finish_reason {
                if finish_reason != "null" && !finish_reason.is_empty() {
                    st.session.finish_reason = Some(finish_reason.clone());
                    let (remaining, is_reasoning) = st.flush_think_buf();
                    if !remaining.is_empty() {
                        if is_reasoning {
                            events.extend(handle_reasoning_part(st, &remaining));
                        } else {
                            events.extend(handle_content_part(st, &remaining));
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
                }
            }
        }
    }

    events
}
