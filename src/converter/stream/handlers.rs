use serde_json::json;
use tracing::debug;
use uuid::Uuid;

use super::emit_sse_event;
use super::images::extract_image_urls_from_text;
use super::state::StreamState;

pub(crate) fn handle_reasoning_part(st: &mut StreamState, text: &str) -> Vec<String> {
    let mut events = Vec::new();

    if !st.reasoning.active {
        let output_index = st.text.current_output_index;
        st.text.current_output_index += 1;
        st.reasoning.output_index = output_index;
        st.reasoning.active = true;
        st.reasoning.buf.clear();
        st.reasoning.item_id = format!("rs_{}_{}", st.session.response_id, st.reasoning.sequence);

        let item = json!({
            "type": "response.output_item.added",
            "sequence_number": st.next_seq(),
            "output_index": output_index,
            "item": {
                "id": st.reasoning.item_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": []
            }
        });
        events.push(emit_sse_event(
            "response.output_item.added",
            &serde_json::to_string(&item).unwrap_or_default(),
        ));

        let part = json!({
            "type": "response.reasoning_summary_part.added",
            "sequence_number": st.next_seq(),
            "item_id": st.reasoning.item_id,
            "output_index": output_index,
            "summary_index": 0,
            "part": {
                "type": "summary_text",
                "text": ""
            }
        });
        events.push(emit_sse_event(
            "response.reasoning_summary_part.added",
            &serde_json::to_string(&part).unwrap_or_default(),
        ));
        st.reasoning.part_added = true;
    }

    st.reasoning.buf.push_str(text);

    let delta = json!({
        "type": "response.reasoning_summary_text.delta",
        "sequence_number": st.next_seq(),
        "item_id": st.reasoning.item_id,
        "output_index": st.reasoning.output_index,
        "summary_index": 0,
        "text": text,
    });
    events.push(emit_sse_event(
        "response.reasoning_summary_text.delta",
        &serde_json::to_string(&delta).unwrap_or_default(),
    ));

    events
}

pub(crate) fn handle_content_part(st: &mut StreamState, text: &str) -> Vec<String> {
    let mut events = Vec::new();

    if !st.text.in_block {
        st.text.in_block = true;
        st.text.buf.clear();

        let output_index = st.text.current_output_index;
        st.text.current_output_index += 1;
        st.text.msg_output_index = output_index;

        st.text.msg_id = format!("msg_{}_{}", st.session.response_id, output_index);

        let item = json!({
            "type": "response.output_item.added",
            "sequence_number": st.next_seq(),
            "output_index": output_index,
            "item": {
                "id": st.text.msg_id,
                "type": "message",
                "status": "in_progress",
                "content": [],
                "role": "assistant"
            }
        });
        events.push(emit_sse_event(
            "response.output_item.added",
            &serde_json::to_string(&item).unwrap_or_default(),
        ));

        let part = json!({
            "type": "response.content_part.added",
            "sequence_number": st.next_seq(),
            "item_id": st.text.msg_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": ""
            }
        });
        events.push(emit_sse_event(
            "response.content_part.added",
            &serde_json::to_string(&part).unwrap_or_default(),
        ));
    }

    st.text.buf.push_str(text);

    let delta = json!({
        "type": "response.output_text.delta",
        "sequence_number": st.next_seq(),
        "item_id": st.text.msg_id,
        "output_index": st.text.msg_output_index,
        "content_index": 0,
        "delta": text,
        "logprobs": []
    });
    events.push(emit_sse_event(
        "response.output_text.delta",
        &serde_json::to_string(&delta).unwrap_or_default(),
    ));

    events
}

pub(crate) fn add_func_item(st: &mut StreamState, idx: i64) -> Vec<String> {
    let mut events = Vec::new();

    let call_id = st
        .func
        .call_ids
        .get(&idx)
        .cloned()
        .unwrap_or_else(|| format!("fc_{}", Uuid::new_v4().simple()));
    let name = st.func.names.get(&idx).cloned().unwrap_or_default();

    let item_id = format!("fc_{}", call_id);

    let output_index = st.text.current_output_index;
    st.text.current_output_index += 1;
    st.func.output_index.insert(idx, output_index);

    let item = json!({
        "type": "response.output_item.added",
        "sequence_number": st.next_seq(),
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": "function_call",
            "status": "in_progress",
            "arguments": "",
            "call_id": call_id,
            "name": name,
        }
    });
    events.push(emit_sse_event(
        "response.output_item.added",
        &serde_json::to_string(&item).unwrap_or_default(),
    ));

    st.func.item_added.insert(idx);
    events
}

pub(crate) fn close_reasoning_block(st: &mut StreamState) -> Vec<String> {
    if !st.reasoning.active {
        return vec![];
    }

    let mut events = Vec::new();
    let full_text = st.reasoning.buf.clone();

    if !full_text.is_empty() {
        st.reasoning.for_func_calls = Some(full_text.clone());
    }

    let text_done = json!({
        "type": "response.reasoning_summary_text.done",
        "sequence_number": st.next_seq(),
        "item_id": st.reasoning.item_id,
        "output_index": st.reasoning.output_index,
        "summary_index": 0,
        "text": full_text,
    });
    events.push(emit_sse_event(
        "response.reasoning_summary_text.done",
        &serde_json::to_string(&text_done).unwrap_or_default(),
    ));

    let part_done = json!({
        "type": "response.reasoning_summary_part.done",
        "sequence_number": st.next_seq(),
        "item_id": st.reasoning.item_id,
        "output_index": st.reasoning.output_index,
        "summary_index": 0,
        "part": {
            "type": "summary_text",
            "text": full_text,
        }
    });
    events.push(emit_sse_event(
        "response.reasoning_summary_part.done",
        &serde_json::to_string(&part_done).unwrap_or_default(),
    ));

    let item_done = json!({
        "type": "response.output_item.done",
        "sequence_number": st.next_seq(),
        "output_index": st.reasoning.output_index,
        "item": {
            "id": st.reasoning.item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [{
                "type": "summary_text",
                "text": full_text
            }]
        }
    });
    events.push(emit_sse_event(
        "response.output_item.done",
        &serde_json::to_string(&item_done).unwrap_or_default(),
    ));

    st.reasoning.active = false;
    st.reasoning.buf.clear();
    st.reasoning.sequence += 1;
    events
}

pub(crate) fn close_text_block(st: &mut StreamState) -> Vec<String> {
    if !st.text.in_block {
        return vec![];
    }

    let mut events = Vec::new();
    let output_index = st.text.msg_output_index;
    let full_text = st.text.buf.clone();

    let done = json!({
        "type": "response.output_text.done",
        "sequence_number": st.next_seq(),
        "item_id": st.text.msg_id,
        "output_index": output_index,
        "content_index": 0,
        "text": full_text,
        "logprobs": []
    });
    events.push(emit_sse_event(
        "response.output_text.done",
        &serde_json::to_string(&done).unwrap_or_default(),
    ));

    let part_done = json!({
        "type": "response.content_part.done",
        "sequence_number": st.next_seq(),
        "item_id": st.text.msg_id,
        "output_index": output_index,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": full_text,
        }
    });
    events.push(emit_sse_event(
        "response.content_part.done",
        &serde_json::to_string(&part_done).unwrap_or_default(),
    ));

    let image_parts = extract_image_urls_from_text(&full_text);
    let item_done = {
        let mut content_parts: Vec<serde_json::Value> = vec![json!({
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": full_text,
        })];
        content_parts.extend(image_parts);
        json!({
            "type": "response.output_item.done",
            "sequence_number": st.next_seq(),
            "output_index": output_index,
            "item": {
                "id": st.text.msg_id,
                "type": "message",
                "status": "completed",
                "content": content_parts,
                "role": "assistant"
            }
        })
    };
    events.push(emit_sse_event(
        "response.output_item.done",
        &serde_json::to_string(&item_done).unwrap_or_default(),
    ));

    st.text.in_block = false;
    events
}

pub(crate) fn close_func_blocks(st: &mut StreamState) -> Vec<String> {
    let mut events = Vec::new();

    let mut idxs: Vec<i64> = st.func.item_added.iter().copied().collect();
    idxs.sort();

    let reasoning_text = st.reasoning.for_func_calls.take();

    for idx in &idxs {
        let args = st
            .func
            .args
            .get(idx)
            .cloned()
            .unwrap_or_else(|| "{}".to_string());
        let call_id = st
            .func
            .call_ids
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("fc_{}", Uuid::new_v4().simple()));
        let name = st.func.names.get(idx).cloned().unwrap_or_default();

        let output_index = st.func.output_index.get(idx).copied().unwrap_or_default();

        if let Some(ref reasoning) = reasoning_text {
            if !reasoning.is_empty() {
                crate::converter::set_reasoning(call_id.clone(), reasoning.clone());
                debug!(
                    call_id = %call_id,
                    reasoning_len = reasoning.len(),
                    "Stored reasoning_text by call_id"
                );
            }
        }

        let fc_done = json!({
            "type": "response.function_call_arguments.done",
            "sequence_number": st.next_seq(),
            "item_id": format!("fc_{}", call_id),
            "output_index": output_index,
            "arguments": args,
        });
        events.push(emit_sse_event(
            "response.function_call_arguments.done",
            &serde_json::to_string(&fc_done).unwrap_or_default(),
        ));

        let item_done = json!({
            "type": "response.output_item.done",
            "sequence_number": st.next_seq(),
            "output_index": output_index,
            "item": {
                "id": format!("fc_{}", call_id),
                "type": "function_call",
                "status": "completed",
                "arguments": args,
                "call_id": call_id,
                "name": name,
            }
        });
        events.push(emit_sse_event(
            "response.output_item.done",
            &serde_json::to_string(&item_done).unwrap_or_default(),
        ));
    }

    st.func.in_block = false;
    events
}
