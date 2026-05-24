use chrono::Utc;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::types::ChatStreamChunk;

#[derive(Debug, Clone)]
pub struct StreamState {
    seq: i64,
    response_id: String,
    created_at: i64,
    msg_id: String,
    fc_id: String,
    in_text_block: bool,
    in_func_block: bool,
    reasoning_active: bool,
    reasoning_item_id: String,
    reasoning_buf: String,
    reasoning_part_added: bool,
    current_output_index: i64,
    reasoning_output_index: i64,
    msg_output_index: i64,
    func_output_index: BTreeMap<i64, i64>,
    reasoning_sequence: i64,
    text_buf: String,
    func_args: BTreeMap<i64, String>,
    func_names: BTreeMap<i64, String>,
    func_call_ids: BTreeMap<i64, String>,
    func_item_added: BTreeMap<i64, bool>,
    first_chunk: bool,
    usage_seen: bool,
    input_tokens: i64,
    output_tokens: i64,
    total_tokens: i64,
    cached_tokens: i64,
    reasoning_tokens: i64,
    finish_reason: Option<String>,
    reasoning_for_func_calls: Option<String>,
    think_buf: String,
    think_in_tag: bool,
    think_state: ThinkTagState,
    think_can_start: bool,
    think_leading_ws: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ThinkTagState {
    None,
    Inside,
    Done,
}

impl Drop for StreamState {
    fn drop(&mut self) {
        info!(
            "StreamState 销毁 - reasoning_active: {}, in_text_block: {}, \
             in_func_block: {}, reasoning_buf_len: {}, text_buf_len: {}, \
             should_synthesize: {}",
            self.reasoning_active,
            self.in_text_block,
            self.in_func_block,
            self.reasoning_buf.len(),
            self.text_buf.len(),
            self.should_synthesize_message_from_reasoning(),
        );
    }
}

impl StreamState {
    fn new() -> Self {
        Self {
            seq: 0,
            response_id: String::new(),
            created_at: 0,
            msg_id: String::new(),
            fc_id: String::new(),
            in_text_block: false,
            in_func_block: false,
            reasoning_active: false,
            reasoning_item_id: String::new(),
            reasoning_buf: String::new(),
            reasoning_part_added: false,
            current_output_index: 0,
            reasoning_output_index: 0,
            msg_output_index: 0,
            func_output_index: BTreeMap::new(),
            reasoning_sequence: 0,
            text_buf: String::new(),
            func_args: BTreeMap::new(),
            func_names: BTreeMap::new(),
            func_call_ids: BTreeMap::new(),
            func_item_added: BTreeMap::new(),
            first_chunk: true,
            usage_seen: false,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_tokens: 0,
            reasoning_tokens: 0,
            finish_reason: None,
            reasoning_for_func_calls: None,
            think_buf: String::new(),
            think_in_tag: false,
            think_state: ThinkTagState::None,
            think_can_start: true,
            think_leading_ws: String::new(),
        }
    }

    fn next_seq(&mut self) -> i64 {
        self.seq += 1;
        self.seq
    }

    pub fn get_text_len(&self) -> usize {
        self.text_buf.len()
    }

    pub fn get_reasoning_len(&self) -> usize {
        self.reasoning_buf.len()
    }

    pub fn get_input_tokens(&self) -> i64 {
        self.input_tokens
    }

    fn feed_think_tag(&mut self, text: &str) -> (Vec<String>, Vec<String>) {
        let mut reasoning_parts = Vec::new();
        let mut content_parts = Vec::new();

        if text.is_empty() {
            return (reasoning_parts, content_parts);
        }

        debug!(
            text_len = text.len(),
            think_state_before = ?self.think_state,
            think_can_start_before = self.think_can_start,
            think_buf_len_before = self.think_buf.len(),
            "🔍 开始处理 Think 标签（增强状态机）"
        );

        const THINK_OPEN_TAG: &str = "<thinking>";
        const THINK_CLOSE_TAG: &str = "</thinking>";

        let mut pending = format!("{}{}", self.think_buf, text);
        self.think_buf.clear();

        while !pending.is_empty() {
            match self.think_state {
                ThinkTagState::None => {
                    // 如果不允许开始，直接作为正文
                    if !self.think_can_start {
                        content_parts.push(pending.clone());
                        pending.clear();
                        continue;
                    }

                    // 1. 提取前导空白字符
                    let ws_end = pending
                        .find(|c: char| c != ' ' && c != '\t' && c != '\r' && c != '\n')
                        .unwrap_or(pending.len());

                    if ws_end > 0 {
                        self.think_leading_ws.push_str(&pending[..ws_end]);
                        pending = pending[ws_end..].to_string();

                        if pending.is_empty() {
                            debug!("当前 chunk 全是空白字符，继续等待");
                            return (reasoning_parts, content_parts);
                        }
                    }

                    // 2. 检查  标签
                    if let Some(idx) = pending.find(THINK_OPEN_TAG) {
                        if idx > 0 {
                            //  不在最开头，关闭检测窗口
                            warn!(idx = idx, "⚠️  出现在非起始位置，关闭 Think 检测窗口");

                            if !self.think_leading_ws.is_empty() {
                                content_parts.push(self.think_leading_ws.clone());
                                self.think_leading_ws.clear();
                            }
                            content_parts.push(pending.clone());
                            pending.clear();
                            self.think_can_start = false;
                            continue;
                        }

                        // 匹配成功！丢弃前导空白，进入 Inside 状态
                        debug!("✅ 匹配到  标签，进入推理模式");
                        self.think_leading_ws.clear();
                        self.think_state = ThinkTagState::Inside;
                        pending = pending[THINK_OPEN_TAG.len()..].to_string();
                        continue;
                    }

                    // 未找到完整 ：检查是否是前缀
                    if Self::is_strict_prefix(THINK_OPEN_TAG, &pending) {
                        debug!(pending = %pending, "缓存可能的前缀片段");
                        self.think_buf = pending.clone();
                        return (reasoning_parts, content_parts);
                    }

                    // 否则关闭检测窗口，作为正文输出
                    if !self.think_leading_ws.is_empty() {
                        content_parts.push(self.think_leading_ws.clone());
                        self.think_leading_ws.clear();
                    }
                    content_parts.push(pending.clone());
                    pending.clear();
                    self.think_can_start = false;
                }

                ThinkTagState::Inside => {
                    // 等待
                    if let Some(idx) = pending.find(THINK_CLOSE_TAG) {
                        if idx > 0 {
                            reasoning_parts.push(pending[..idx].to_string());
                            debug!(
                                reasoning_len = idx,
                                total_reasoning_parts = reasoning_parts.len(),
                                "✅ 提取推理内容块"
                            );
                        }
                        pending = pending[idx + THINK_CLOSE_TAG.len()..].to_string();
                        self.think_state = ThinkTagState::Done;
                        info!("✅ Think 推理结束标记匹配完成");
                        continue;
                    }

                    // 未找到完整 ：缓存末尾可能是前缀的部分
                    let keep = Self::suffix_that_could_be_prefix(&pending, THINK_CLOSE_TAG);
                    if keep > 0 {
                        if pending.len() > keep {
                            reasoning_parts.push(pending[..pending.len() - keep].to_string());
                        }
                        self.think_buf = pending[pending.len() - keep..].to_string();
                        debug!(buf_len = self.think_buf.len(), "⏳ 缓存可能的结束标签前缀");
                        return (reasoning_parts, content_parts);
                    }

                    reasoning_parts.push(pending.clone());
                    pending.clear();
                }

                ThinkTagState::Done => {
                    content_parts.push(pending.clone());
                    pending.clear();
                }
            }
        }

        debug!(
            reasoning_count = reasoning_parts.len(),
            content_count = content_parts.len(),
            think_state_after = ?self.think_state,
            think_buf_remaining = self.think_buf.len(),
            "📊 Think 增强状态机处理完成"
        );

        (reasoning_parts, content_parts)
    }

    fn is_strict_prefix(full: &str, s: &str) -> bool {
        s.is_empty() || s.len() >= full.len() || full[..s.len()] == *s
    }

    fn suffix_that_could_be_prefix(s: &str, tag: &str) -> usize {
        let max_len = s.len().min(tag.len() - 1);

        for k in (1..=max_len).rev() {
            if &s[s.len() - k..] == &tag[..k] {
                return k;
            }
        }

        0
    }

    fn flush_think_buf(&mut self) -> (String, bool) {
        if self.think_buf.is_empty()
            && self.think_state == ThinkTagState::None
            && self.think_leading_ws.is_empty()
        {
            return (String::new(), false);
        }

        let mut content = String::new();
        let is_reasoning = self.think_state == ThinkTagState::Inside;

        match self.think_state {
            ThinkTagState::None => {
                // 如果还在等待状态，合并前导空白和缓冲区内容
                if !self.think_leading_ws.is_empty() {
                    content.push_str(&self.think_leading_ws);
                    self.think_leading_ws.clear();
                }
                if !self.think_buf.is_empty() {
                    content.push_str(&self.think_buf);
                    self.think_buf.clear();
                }
            }
            ThinkTagState::Inside | ThinkTagState::Done => {
                if !self.think_buf.is_empty() {
                    content = self.think_buf.clone();
                    self.think_buf.clear();
                }
            }
        }

        warn!(
            flushed_len = content.len(),
            was_in_tag = is_reasoning,
            state_before_flush = ?self.think_state,
            "⚠️ 强制刷新 Think 缓冲区（可能缺少结束标记）"
        );

        // 重置状态
        self.think_state = ThinkTagState::None;
        self.think_in_tag = false;

        (content, is_reasoning)
    }
}

fn emit_sse_event(event: &str, payload: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event, payload)
}

pub fn convert_chat_stream_line(
    line: &str,
    state: &mut Option<StreamState>,
    original_request: &Value,
) -> Vec<String> {
    if state.is_none() {
        *state = Some(StreamState::new());
    }

    let st = match state.as_mut() {
        Some(s) => s,
        None => {
            error!("流状态初始化失败");
            return vec![];
        }
    };

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
            error!("SSE 数据格式错误：缺少 data: 前缀");
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
                "SSE data 行 JSON 解析失败，跳过该行"
            );
            return vec![];
        }
    };

    let mut events: Vec<String> = Vec::new();

    if st.first_chunk {
        st.first_chunk = false;
        st.response_id = chunk
            .id
            .clone()
            .unwrap_or_else(|| format!("resp_{}", Uuid::new_v4().simple()));
        st.created_at = chunk.created.unwrap_or_else(|| Utc::now().timestamp());

        st.seq = 0;

        let created = json!({
            "type": "response.created",
            "sequence_number": st.next_seq(),
            "response": {
                "id": st.response_id,
                "object": "response",
                "created_at": st.created_at,
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
                "id": st.response_id,
                "object": "response",
                "created_at": st.created_at,
                "status": "in_progress"
            }
        });
        events.push(emit_sse_event(
            "response.in_progress",
            &serde_json::to_string(&in_progress).unwrap_or_default(),
        ));
    }

    if let Some(usage) = &chunk.usage {
        st.usage_seen = true;
        if let Some(v) = usage.prompt_tokens {
            st.input_tokens = v;
        }
        if let Some(v) = usage.completion_tokens {
            st.output_tokens = v;
        }
        if let Some(v) = usage.total_tokens {
            st.total_tokens = v;
        }
        if let Some(details) = &usage.prompt_tokens_details {
            if let Some(v) = details.cached_tokens {
                st.cached_tokens = v;
            }
        }
        if let Some(details) = &usage.completion_tokens_details {
            if let Some(v) = details.reasoning_tokens {
                st.reasoning_tokens = v;
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
                                if st.reasoning_active {
                                    events.extend(close_reasoning_block(st));
                                }
                                events.extend(handle_reasoning_part(st, &rp));
                            }
                        }
                        for cp in content_parts {
                            if !cp.is_empty() {
                                if st.reasoning_active {
                                    events.extend(close_reasoning_block(st));
                                }
                                events.extend(handle_content_part(st, &cp));
                            }
                        }
                    }
                }

                if let Some(tool_calls) = &delta.tool_calls {
                    if st.reasoning_active {
                        events.extend(close_reasoning_block(st));
                    }
                    if st.in_text_block {
                        events.extend(close_text_block(st));
                    }
                    for tc in tool_calls {
                        let idx = tc.index;

                        st.func_args.entry(idx).or_insert_with(String::new);
                        st.func_names.entry(idx).or_insert_with(String::new);
                        st.func_call_ids.entry(idx).or_insert_with(String::new);
                        st.func_item_added.entry(idx).or_insert(false);

                        if let Some(id) = &tc.id {
                            if !id.is_empty() {
                                st.func_call_ids.insert(idx, id.clone());
                            }
                        }

                        let mut func_args_delta: Option<String> = None;
                        if let Some(func) = &tc.function {
                            if let Some(name) = &func.name {
                                if !name.is_empty() {
                                    st.func_names.insert(idx, name.clone());
                                }
                            }
                            if let Some(args) = &func.arguments {
                                if !args.is_empty() {
                                    if let Some(buf) = st.func_args.get_mut(&idx) {
                                        buf.push_str(args);
                                    }
                                    func_args_delta = Some(args.clone());
                                }
                            }
                        }

                        let has_id = st.func_call_ids.get(&idx).map(|s| !s.is_empty()).unwrap_or(false);
                        let has_name = st.func_names.get(&idx).map(|s| !s.is_empty()).unwrap_or(false);
                        if has_id && has_name && !st.func_item_added.get(&idx).unwrap_or(&false) {
                            st.fc_id = st.func_call_ids.get(&idx).cloned().unwrap_or_default();
                            st.in_func_block = true;
                            events.extend(add_func_item(st, idx));
                        }

                        if let Some(args) = func_args_delta {
                            if !st.func_item_added.get(&idx).unwrap_or(&false) {
                                events.extend(add_func_item(st, idx));
                            }

                            let output_index = st
                                .func_output_index
                                .get(&idx)
                                .copied()
                                .unwrap_or_default();
                            let item_id = format!(
                                "fc_{}",
                                st.func_call_ids.get(&idx).cloned().unwrap_or_else(
                                    || format!("fc_{}", Uuid::new_v4().simple())
                                )
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
                    st.finish_reason = Some(finish_reason.clone());
                    let (remaining, is_reasoning) = st.flush_think_buf();
                    if !remaining.is_empty() {
                        if is_reasoning {
                            events.extend(handle_reasoning_part(st, &remaining));
                        } else {
                            events.extend(handle_content_part(st, &remaining));
                        }
                    }
                    if st.reasoning_active {
                        events.extend(close_reasoning_block(st));
                    }
                    if st.in_text_block {
                        events.extend(close_text_block(st));
                    }
                    if st.in_func_block {
                        events.extend(close_func_blocks(st));
                    }
                }
            }
        }
    }

    events
}

fn handle_reasoning_part(st: &mut StreamState, text: &str) -> Vec<String> {
    let mut events = Vec::new();

    if !st.reasoning_active {
        let output_index = st.current_output_index;
        st.current_output_index += 1;
        st.reasoning_output_index = output_index;
        st.reasoning_active = true;
        st.reasoning_buf.clear();
        st.reasoning_item_id = format!("rs_{}_{}", st.response_id, st.reasoning_sequence);

        let item = json!({
            "type": "response.output_item.added",
            "sequence_number": st.next_seq(),
            "output_index": output_index,
            "item": {
                "id": st.reasoning_item_id,
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
            "item_id": st.reasoning_item_id,
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
        st.reasoning_part_added = true;
    }

    st.reasoning_buf.push_str(text);

    let delta = json!({
        "type": "response.reasoning_summary_text.delta",
        "sequence_number": st.next_seq(),
        "item_id": st.reasoning_item_id,
        "output_index": st.reasoning_output_index,
        "summary_index": 0,
        "text": text,
    });
    events.push(emit_sse_event(
        "response.reasoning_summary_text.delta",
        &serde_json::to_string(&delta).unwrap_or_default(),
    ));

    events
}

fn handle_content_part(st: &mut StreamState, text: &str) -> Vec<String> {
    let mut events = Vec::new();

    if !st.in_text_block {
        st.in_text_block = true;
        st.text_buf.clear();

        let output_index = st.current_output_index;
        st.current_output_index += 1;
        st.msg_output_index = output_index;

        st.msg_id = format!("msg_{}_{}", st.response_id, output_index);

        let item = json!({
            "type": "response.output_item.added",
            "sequence_number": st.next_seq(),
            "output_index": output_index,
            "item": {
                "id": st.msg_id,
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
            "item_id": st.msg_id,
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

    st.text_buf.push_str(text);

    let delta = json!({
        "type": "response.output_text.delta",
        "sequence_number": st.next_seq(),
        "item_id": st.msg_id,
        "output_index": st.msg_output_index,
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

fn add_func_item(st: &mut StreamState, idx: i64) -> Vec<String> {
    let mut events = Vec::new();

    let call_id = st
        .func_call_ids
        .get(&idx)
        .cloned()
        .unwrap_or_else(|| format!("fc_{}", Uuid::new_v4().simple()));
    let name = st.func_names.get(&idx).cloned().unwrap_or_default();

    let item_id = format!("fc_{}", call_id);

    let output_index = st.current_output_index;
    st.current_output_index += 1;
    st.func_output_index.insert(idx, output_index);

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

    st.func_item_added.insert(idx, true);
    events
}

fn close_reasoning_block(st: &mut StreamState) -> Vec<String> {
    if !st.reasoning_active {
        return vec![];
    }

    let mut events = Vec::new();
    let full_text = st.reasoning_buf.clone();

    if !full_text.is_empty() {
        st.reasoning_for_func_calls = Some(full_text.clone());
    }

    let text_done = json!({
        "type": "response.reasoning_summary_text.done",
        "sequence_number": st.next_seq(),
        "item_id": st.reasoning_item_id,
        "output_index": st.reasoning_output_index,
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
        "item_id": st.reasoning_item_id,
        "output_index": st.reasoning_output_index,
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
        "output_index": st.reasoning_output_index,
        "item": {
            "id": st.reasoning_item_id,
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

    st.reasoning_active = false;
    st.reasoning_buf.clear();
    st.reasoning_sequence += 1;
    events
}

fn close_text_block(st: &mut StreamState) -> Vec<String> {
    if !st.in_text_block {
        return vec![];
    }

    let mut events = Vec::new();
    let output_index = st.msg_output_index;
    let full_text = st.text_buf.clone();

    let done = json!({
        "type": "response.output_text.done",
        "sequence_number": st.next_seq(),
        "item_id": st.msg_id,
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
        "item_id": st.msg_id,
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

    let item_done = json!({
        "type": "response.output_item.done",
        "sequence_number": st.next_seq(),
        "output_index": output_index,
        "item": {
            "id": st.msg_id,
            "type": "message",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": full_text,
            }],
            "role": "assistant"
        }
    });
    events.push(emit_sse_event(
        "response.output_item.done",
        &serde_json::to_string(&item_done).unwrap_or_default(),
    ));

    st.in_text_block = false;
    events
}

fn close_func_blocks(st: &mut StreamState) -> Vec<String> {
    let mut events = Vec::new();

    let mut idxs: Vec<i64> = st
        .func_item_added
        .iter()
        .filter(|(_, added)| **added)
        .map(|(idx, _)| *idx)
        .collect();
    idxs.sort();

    let reasoning_text = st.reasoning_for_func_calls.take();

    for idx in &idxs {
        let args = st
            .func_args
            .get(idx)
            .cloned()
            .unwrap_or_else(|| "{}".to_string());
        let call_id = st
            .func_call_ids
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("fc_{}", Uuid::new_v4().simple()));
        let name = st.func_names.get(idx).cloned().unwrap_or_default();

        let output_index = st
            .func_output_index
            .get(idx)
            .copied()
            .unwrap_or_default();

        if let Some(ref reasoning) = reasoning_text {
            if !reasoning.is_empty() {
                crate::converter::set_reasoning(call_id.clone(), reasoning.clone());
                debug!(
                    call_id = %call_id,
                    reasoning_len = reasoning.len(),
                    "已按 call_id 存储 reasoning_text"
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

    st.in_func_block = false;
    events
}

fn generate_completed_events(st: &mut StreamState, original_request: &Value) -> Vec<String> {
    warn!(
        "生成完成事件 - reasoning_buf长度: {}, text_buf长度: {}, should_synthesize: {}",
        st.reasoning_buf.len(),
        st.text_buf.len(),
        st.should_synthesize_message_from_reasoning()
    );
    generate_completed_events_internal(st, original_request)
}

pub fn generate_completed_events_fallback(
    st: &mut StreamState,
    original_request: &Value,
) -> Vec<String> {
    warn!(
        "🔧 兜底机制触发 - 上游未发送 [DONE]，补发 response.completed 事件\n\
         状态: reasoning_active={}, in_text_block={}, in_func_block={}\n\
         内容: reasoning_buf_len={}, text_buf_len={}\n\
         should_synthesize_message_from_reasoning={}",
        st.reasoning_active,
        st.in_text_block,
        st.in_func_block,
        st.reasoning_buf.len(),
        st.text_buf.len(),
        st.should_synthesize_message_from_reasoning(),
    );

    let events = generate_completed_events_internal(st, original_request);

    info!("✅ 兜底完成事件已生成 - 共 {} 个事件", events.len());

    events
}

fn generate_completed_events_internal(
    st: &mut StreamState,
    original_request: &Value,
) -> Vec<String> {
    let mut events = Vec::new();

    // 兜底：如果只有 reasoning 内容但没有文本内容，从 reasoning 合成 message
    if st.should_synthesize_message_from_reasoning() {
        let summary = extract_summary_from_reasoning(&st.reasoning_buf);
        if !summary.is_empty() {
            let output_index = st.current_output_index;
            st.current_output_index += 1;
            st.msg_output_index = output_index;
            st.msg_id = format!("msg_{}_{}", st.response_id, output_index);
            st.text_buf = summary;
            st.in_text_block = true;
        }
    }

    if st.reasoning_active {
        events.extend(close_reasoning_block(st));
    }
    if st.in_text_block {
        events.extend(close_text_block(st));
    }
    if st.in_func_block {
        events.extend(close_func_blocks(st));
    }

    let status = if st.finish_reason.as_deref() == Some("length") {
        "incomplete"
    } else {
        "completed"
    };

    let mut completed = json!({
        "type": "response.completed",
        "sequence_number": st.next_seq(),
        "response": {
            "id": st.response_id,
            "object": "response",
            "created_at": st.created_at,
            "status": status,
            "background": false,
            "error": null,
        }
    });

    if st.finish_reason.as_deref() == Some("length") {
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

    if !st.reasoning_buf.is_empty() || st.reasoning_part_added {
        outputs.push(json!({
            "id": st.reasoning_item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [{
                "type": "summary_text",
                "text": st.reasoning_buf,
            }]
        }));
    }

    if !st.text_buf.is_empty() || !st.msg_id.is_empty() {
        outputs.push(json!({
            "id": st.msg_id,
            "type": "message",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": st.text_buf,
            }],
            "role": "assistant"
        }));
    }

    let mut idxs: Vec<i64> = st.func_args.keys().copied().collect();
    idxs.sort();
    for idx in idxs {
        let args = st
            .func_args
            .get(&idx)
            .cloned()
            .unwrap_or_else(|| "{}".to_string());
        let call_id = st.func_call_ids.get(&idx).cloned().unwrap_or_default();
        let name = st.func_names.get(&idx).cloned().unwrap_or_default();

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

    let input_tokens = st.input_tokens;
    let output_tokens = st.output_tokens;
    let cached_tokens = st.cached_tokens;
    let mut reasoning_tokens = st.reasoning_tokens;

    if reasoning_tokens == 0 && !st.reasoning_buf.is_empty() {
        reasoning_tokens = (st.reasoning_buf.len() / 4) as i64;
    }

    let actual_input = if cached_tokens > 0 {
        (input_tokens - cached_tokens).max(0)
    } else {
        input_tokens
    };

    let mut total = st.total_tokens;
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

impl StreamState {

    fn should_synthesize_message_from_reasoning(&self) -> bool {
        let has_no_text_content = self.text_buf.is_empty() && self.msg_id.is_empty();
        let has_reasoning_content = !self.reasoning_buf.is_empty();
        let reasoning_not_already_sent = !self.reasoning_part_added;
        has_no_text_content && has_reasoning_content && reasoning_not_already_sent
    }
}
