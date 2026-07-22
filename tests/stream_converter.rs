use ccr::converter::stream::{
    convert_chat_stream_line, generate_completed_events_fallback, StreamState,
};
use serde_json::{json, Value};

fn sse_line(json_val: &Value) -> String {
    format!("data: {}", serde_json::to_string(json_val).unwrap())
}

fn original_request() -> Value {
    json!({
        "model": "gpt-5",
        "input": [{"type": "message", "role": "user", "content": "hello"}],
        "stream": true
    })
}

fn event_type(event: &str) -> &str {
    event
        .lines()
        .find(|l| l.starts_with("event:"))
        .map(|l| l.strip_prefix("event: ").unwrap_or(""))
        .unwrap_or("")
}

fn event_data(event: &str) -> Value {
    let data = event
        .lines()
        .find(|l| l.starts_with("data:"))
        .map(|l| l.strip_prefix("data: ").unwrap_or(""))
        .unwrap_or("{}");
    serde_json::from_str(data).unwrap_or(Value::Null)
}

fn event_types(events: &[String]) -> Vec<String> {
    events.iter().map(|e| event_type(e).to_string()).collect()
}

fn events_by_type(events: &[String], ty: &str) -> Vec<Value> {
    events
        .iter()
        .filter(|e| event_type(e) == ty)
        .map(|e| event_data(e))
        .collect()
}

#[test]
fn stream_state_new_is_empty() {
    let st = StreamState::new();
    assert_eq!(st.get_text_len(), 0);
    assert_eq!(st.get_reasoning_len(), 0);
    assert_eq!(st.get_input_tokens(), 0);
}

#[test]
fn empty_line_returns_no_events() {
    let mut st = StreamState::new();
    let req = original_request();
    let events = convert_chat_stream_line("", &mut st, &req);
    assert!(events.is_empty());
}

#[test]
fn whitespace_only_line_returns_no_events() {
    let mut st = StreamState::new();
    let req = original_request();
    let events = convert_chat_stream_line("   \n  ", &mut st, &req);
    assert!(events.is_empty());
}

#[test]
fn line_without_data_prefix_returns_no_events() {
    let mut st = StreamState::new();
    let req = original_request();
    let events = convert_chat_stream_line("event: something", &mut st, &req);
    assert!(events.is_empty());
}

#[test]
fn invalid_json_returns_no_events() {
    let mut st = StreamState::new();
    let req = original_request();
    let events = convert_chat_stream_line("data: not json", &mut st, &req);
    assert!(events.is_empty());
}

#[test]
fn first_chunk_creates_response_created_and_in_progress() {
    let mut st = StreamState::new();
    let req = original_request();
    let chunk = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    let events = convert_chat_stream_line(&sse_line(&chunk), &mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.created".to_string()));
    assert!(types.contains(&"response.in_progress".to_string()));
}

#[test]
fn first_chunk_without_id_generates_fallback_id() {
    let mut st = StreamState::new();
    let req = original_request();
    let chunk = json!({
        "object": "chat.completion.chunk", "created": 1700000000,
        "model": "deepseek-chat", "choices": []
    });
    let events = convert_chat_stream_line(&sse_line(&chunk), &mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.created".to_string()));
    assert!(types.contains(&"response.in_progress".to_string()));
    let created = events_by_type(&events, "response.created");
    let id = created[0]["response"]["id"].as_str().unwrap();
    assert!(id.starts_with("resp_"));
}

#[test]
fn content_delta_produces_output_events() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    let delta = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {"content": "Hello"}}]
    });
    let events = convert_chat_stream_line(&sse_line(&delta), &mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.output_item.added".to_string()));
    assert!(types.contains(&"response.content_part.added".to_string()));
    assert!(types.contains(&"response.output_text.delta".to_string()));
}

#[test]
fn multiple_content_deltas_accumulate() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": " World"}}]
        })),
        &mut st,
        &req,
    );
    let types = event_types(&events);
    assert!(!types.contains(&"response.output_item.added".to_string()));
    assert!(types.contains(&"response.output_text.delta".to_string()));
    assert_eq!(st.get_text_len(), 11);
}

#[test]
fn reasoning_delta_produces_reasoning_events() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    let delta = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {"reasoning_content": "I need to think"}}]
    });
    let events = convert_chat_stream_line(&sse_line(&delta), &mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.output_item.added".to_string()));
    assert!(types.contains(&"response.reasoning_summary_part.added".to_string()));
    assert!(types.contains(&"response.reasoning_summary_text.delta".to_string()));
    assert_eq!(st.get_reasoning_len(), 15);
}

#[test]
fn think_tag_content_becomes_reasoning() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    let delta = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {
            "content": "<thinking>Let me analyze this</thinking>The answer is 42"
        }}]
    });
    let events = convert_chat_stream_line(&sse_line(&delta), &mut st, &req);
    let types = event_types(&events);
    assert!(
        types.contains(&"response.reasoning_summary_text.delta".to_string()),
        "expected reasoning events for <thinking> content, got: {:?}",
        types
    );
    assert!(
        types.contains(&"response.output_text.delta".to_string()),
        "expected content events for non-thinking part, got: {:?}",
        types
    );
}

#[test]
fn think_tag_split_across_chunks() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    let chunk1 = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {"content": "<think"}}]
    });
    let events1 = convert_chat_stream_line(&sse_line(&chunk1), &mut st, &req);
    let types1 = event_types(&events1);
    assert!(!types1.contains(&"response.reasoning_summary_text.delta".to_string()));
    let chunk2 = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {"content": "ing>analysis here</thinking>result"}}]
    });
    let events2 = convert_chat_stream_line(&sse_line(&chunk2), &mut st, &req);
    let types2 = event_types(&events2);
    assert!(types2.contains(&"response.reasoning_summary_text.delta".to_string()));
    assert!(types2.contains(&"response.output_text.delta".to_string()));
}

#[test]
fn think_tag_not_at_start_is_treated_as_content() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    let delta = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {
            "content": "Some text <thinking>hidden</thinking> more"
        }}]
    });
    let events = convert_chat_stream_line(&sse_line(&delta), &mut st, &req);
    let types = event_types(&events);
    assert!(!types.contains(&"response.reasoning_summary_text.delta".to_string()));
    assert!(types.contains(&"response.output_text.delta".to_string()));
}

#[test]
fn tool_call_delta_produces_function_call_events() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    let delta = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {
            "tool_calls": [{
                "index": 0, "id": "call_abc",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"beijing\"}"}
            }]
        }}]
    });
    let events = convert_chat_stream_line(&sse_line(&delta), &mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.output_item.added".to_string()));
    assert!(types.contains(&"response.function_call_arguments.delta".to_string()));
}

#[test]
fn tool_call_id_and_name_arrive_separately() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_abc"}]}}]
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"name": "search"}}]
            }}]
        })),
        &mut st,
        &req,
    );
    let types = event_types(&events);
    assert!(types.contains(&"response.output_item.added".to_string()));
}

#[test]
fn finish_reason_closes_open_blocks() {
    let mut st = StreamState::new();
    let req = original_request();
    let first = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": []
    });
    convert_chat_stream_line(&sse_line(&first), &mut st, &req);
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": "stop"}]
        })),
        &mut st,
        &req,
    );
    let types = event_types(&events);
    assert!(types.contains(&"response.output_text.done".to_string()));
    assert!(types.contains(&"response.content_part.done".to_string()));
    assert!(types.contains(&"response.output_item.done".to_string()));
}

#[test]
fn usage_is_tracked_from_chunk() {
    let mut st = StreamState::new();
    let req = original_request();
    let chunk = json!({
        "id": "chatcmpl-123", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "deepseek-chat", "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    convert_chat_stream_line(&sse_line(&chunk), &mut st, &req);
    assert_eq!(st.get_input_tokens(), 10);
}

#[test]
fn done_marker_generates_response_completed() {
    let mut st = StreamState::new();
    let req = original_request();
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": "stop"}]
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line("data: [DONE]", &mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.completed".to_string()));
}

#[test]
fn done_marker_with_length_finish_reason_is_incomplete() {
    let mut st = StreamState::new();
    let req = original_request();
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "truncated"}}]
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": "length"}]
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line("data: [DONE]", &mut st, &req);
    let completed = events_by_type(&events, "response.completed");
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0]["response"]["status"], "incomplete");
    assert_eq!(
        completed[0]["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
}

#[test]
fn fallback_generates_completed_event() {
    let mut st = StreamState::new();
    let req = original_request();
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        })),
        &mut st,
        &req,
    );
    let events = generate_completed_events_fallback(&mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.output_text.done".to_string()));
    assert!(types.contains(&"response.content_part.done".to_string()));
    assert!(types.contains(&"response.output_item.done".to_string()));
    assert!(types.contains(&"response.completed".to_string()));
}

#[test]
fn fallback_with_no_content_produces_completed() {
    let mut st = StreamState::new();
    let req = original_request();
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "chatcmpl-123", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    let events = generate_completed_events_fallback(&mut st, &req);
    let types = event_types(&events);
    assert!(types.contains(&"response.completed".to_string()));
}

#[test]
fn full_lifecycle_content_only() {
    let mut st = StreamState::new();
    let req = original_request();
    let e1 = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-1", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    assert_eq!(
        event_types(&e1),
        vec!["response.created", "response.in_progress"]
    );
    let e2 = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-1", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        })),
        &mut st,
        &req,
    );
    let t2 = event_types(&e2);
    assert!(t2.contains(&"response.output_item.added".to_string()));
    assert!(t2.contains(&"response.content_part.added".to_string()));
    assert!(t2.contains(&"response.output_text.delta".to_string()));
    let e3 = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-1", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": "stop"}]
        })),
        &mut st,
        &req,
    );
    let t3 = event_types(&e3);
    assert!(t3.contains(&"response.output_text.done".to_string()));
    assert!(t3.contains(&"response.content_part.done".to_string()));
    assert!(t3.contains(&"response.output_item.done".to_string()));
    let e4 = convert_chat_stream_line("data: [DONE]", &mut st, &req);
    let t4 = event_types(&e4);
    assert!(t4.contains(&"response.completed".to_string()));
    let completed = events_by_type(&e4, "response.completed");
    let output = completed[0]["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
}

#[test]
fn full_lifecycle_with_tool_calls() {
    let mut st = StreamState::new();
    let req = original_request();
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-2", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-2", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {
                "tool_calls": [{
                    "index": 0, "id": "call_x",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"beijing\"}"}
                }]
            }}]
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-2", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": "stop"}]
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line("data: [DONE]", &mut st, &req);
    let completed = events_by_type(&events, "response.completed");
    let output = completed[0]["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "function_call");
}

#[test]
fn full_lifecycle_with_usage() {
    let mut st = StreamState::new();
    let req = original_request();
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-3", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-3", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}]
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-3", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 2, "total_tokens": 17}
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line("data: [DONE]", &mut st, &req);
    let completed = events_by_type(&events, "response.completed");
    let usage = &completed[0]["response"]["usage"];
    assert_eq!(usage["input_tokens"], 15);
    assert_eq!(usage["output_tokens"], 2);
    assert_eq!(usage["total_tokens"], 17);
}

#[test]
fn content_after_reasoning_closes_reasoning_block_first() {
    let mut st = StreamState::new();
    let req = original_request();
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-4", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-4", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}]
        })),
        &mut st,
        &req,
    );
    let events = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-4", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": "answer"}}]
        })),
        &mut st,
        &req,
    );
    let types = event_types(&events);
    assert!(
        types.contains(&"response.reasoning_summary_text.done".to_string()),
        "should close reasoning before content, got: {:?}",
        types
    );
    assert!(types.contains(&"response.output_item.added".to_string()));
}

#[test]
fn sse_event_format_is_correct() {
    let mut st = StreamState::new();
    let req = original_request();
    let events = convert_chat_stream_line(
        &sse_line(&json!({
            "id": "cmpl-6", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "deepseek-chat", "choices": []
        })),
        &mut st,
        &req,
    );
    for event in &events {
        assert!(event.starts_with("event: "), "bad event start: {}", event);
        assert!(event.contains("\ndata: "), "missing data: {}", event);
        assert!(event.ends_with("\n\n"), "bad event end: {}", event);
    }
}
