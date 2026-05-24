use std::collections::{BTreeMap, BTreeSet};
use tracing::{debug, info, warn};

// ── ThinkTagStateMachine ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ThinkTagState {
    None,
    Inside,
    Done,
}

pub(crate) struct ThinkStateMachine {
    pub(crate) buf: String,
    pub(crate) in_tag: bool,
    pub(crate) state: ThinkTagState,
    pub(crate) can_start: bool,
    pub(crate) leading_ws: String,
}

impl ThinkStateMachine {
    pub(crate) fn new() -> Self {
        Self {
            buf: String::new(),
            in_tag: false,
            state: ThinkTagState::None,
            can_start: true,
            leading_ws: String::new(),
        }
    }

    /// Feeds text through the `<thinking>` tag state machine.
    /// Returns (reasoning_parts, content_parts).
    pub(crate) fn feed(&mut self, text: &str) -> (Vec<String>, Vec<String>) {
        let mut reasoning_parts = Vec::new();
        let mut content_parts = Vec::new();

        if text.is_empty() {
            return (reasoning_parts, content_parts);
        }

        debug!(
            text_len = text.len(),
            think_state_before = ?self.state,
            think_can_start_before = self.can_start,
            think_buf_len_before = self.buf.len(),
            "Processing Think tags (enhanced state machine)"
        );

        const THINK_OPEN_TAG: &str = "<thinking>";
        const THINK_CLOSE_TAG: &str = "</thinking>";

        let mut pending = format!("{}{}", self.buf, text);
        self.buf.clear();

        while !pending.is_empty() {
            match self.state {
                ThinkTagState::None => {
                    if !self.can_start {
                        content_parts.push(pending.clone());
                        pending.clear();
                        continue;
                    }

                    let ws_end = pending
                        .find(|c: char| c != ' ' && c != '\t' && c != '\r' && c != '\n')
                        .unwrap_or(pending.len());

                    if ws_end > 0 {
                        self.leading_ws.push_str(&pending[..ws_end]);
                        pending = pending[ws_end..].to_string();

                        if pending.is_empty() {
                            debug!("Current chunk is all whitespace, continuing to wait");
                            return (reasoning_parts, content_parts);
                        }
                    }

                    if let Some(idx) = pending.find(THINK_OPEN_TAG) {
                        if idx > 0 {
                            warn!(idx = idx, "<thinking> found not at start position, closing Think detection window");

                            if !self.leading_ws.is_empty() {
                                content_parts.push(self.leading_ws.clone());
                                self.leading_ws.clear();
                            }
                            content_parts.push(pending.clone());
                            pending.clear();
                            self.can_start = false;
                            continue;
                        }

                        debug!("Matched <thinking> tag, entering reasoning mode");
                        self.leading_ws.clear();
                        self.state = ThinkTagState::Inside;
                        pending = pending[THINK_OPEN_TAG.len()..].to_string();
                        continue;
                    }

                    if Self::is_strict_prefix(THINK_OPEN_TAG, &pending) {
                        debug!(pending = %pending, "Buffering possible prefix fragment");
                        self.buf = pending.clone();
                        return (reasoning_parts, content_parts);
                    }

                    if !self.leading_ws.is_empty() {
                        content_parts.push(self.leading_ws.clone());
                        self.leading_ws.clear();
                    }
                    content_parts.push(pending.clone());
                    pending.clear();
                    self.can_start = false;
                }

                ThinkTagState::Inside => {
                    if let Some(idx) = pending.find(THINK_CLOSE_TAG) {
                        if idx > 0 {
                            reasoning_parts.push(pending[..idx].to_string());
                            debug!(
                                reasoning_len = idx,
                                total_reasoning_parts = reasoning_parts.len(),
                                "Extracted reasoning content chunk"
                            );
                        }
                        pending = pending[idx + THINK_CLOSE_TAG.len()..].to_string();
                        self.state = ThinkTagState::Done;
                        info!("Think reasoning end tag match completed");
                        continue;
                    }

                    let keep = Self::suffix_that_could_be_prefix(&pending, THINK_CLOSE_TAG);
                    if keep > 0 {
                        if pending.len() > keep {
                            reasoning_parts.push(pending[..pending.len() - keep].to_string());
                        }
                        self.buf = pending[pending.len() - keep..].to_string();
                        debug!(
                            buf_len = self.buf.len(),
                            "Buffering possible end tag prefix"
                        );
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
            think_state_after = ?self.state,
            think_buf_remaining = self.buf.len(),
            "Think enhanced state machine processing complete"
        );

        (reasoning_parts, content_parts)
    }

    /// Force-flush the think buffer. Returns (content, is_reasoning).
    pub(crate) fn flush(&mut self) -> (String, bool) {
        if self.buf.is_empty() && self.state == ThinkTagState::None && self.leading_ws.is_empty() {
            return (String::new(), false);
        }

        let mut content = String::new();
        let is_reasoning = self.state == ThinkTagState::Inside;

        match self.state {
            ThinkTagState::None => {
                if !self.leading_ws.is_empty() {
                    content.push_str(&self.leading_ws);
                    self.leading_ws.clear();
                }
                if !self.buf.is_empty() {
                    content.push_str(&self.buf);
                    self.buf.clear();
                }
            }
            ThinkTagState::Inside | ThinkTagState::Done => {
                if !self.buf.is_empty() {
                    content = self.buf.clone();
                    self.buf.clear();
                }
            }
        }

        warn!(
            flushed_len = content.len(),
            was_in_tag = is_reasoning,
            state_before_flush = ?self.state,
            "Force-flushing Think buffer (end tag may be missing)"
        );

        self.state = ThinkTagState::None;
        self.in_tag = false;

        (content, is_reasoning)
    }

    fn is_strict_prefix(full: &str, s: &str) -> bool {
        s.is_empty() || (s.len() <= full.len() && full[..s.len()] == *s)
    }

    fn suffix_that_could_be_prefix(s: &str, tag: &str) -> usize {
        let max_len = s.len().min(tag.len() - 1);

        for k in (1..=max_len).rev() {
            if s[s.len() - k..] == tag[..k] {
                return k;
            }
        }

        0
    }
}

// ── ReasoningState ──────────────────────────────────────────────────────

pub(crate) struct ReasoningState {
    pub(crate) active: bool,
    pub(crate) item_id: String,
    pub(crate) buf: String,
    pub(crate) part_added: bool,
    pub(crate) output_index: i64,
    pub(crate) sequence: i64,
    pub(crate) for_func_calls: Option<String>,
}

impl ReasoningState {
    pub(crate) fn new() -> Self {
        Self {
            active: false,
            item_id: String::new(),
            buf: String::new(),
            part_added: false,
            output_index: 0,
            sequence: 0,
            for_func_calls: None,
        }
    }
}

// ── TextContentState ────────────────────────────────────────────────────

pub(crate) struct TextContentState {
    pub(crate) in_block: bool,
    pub(crate) buf: String,
    pub(crate) msg_id: String,
    pub(crate) current_output_index: i64,
    pub(crate) msg_output_index: i64,
}

impl TextContentState {
    pub(crate) fn new() -> Self {
        Self {
            in_block: false,
            buf: String::new(),
            msg_id: String::new(),
            current_output_index: 0,
            msg_output_index: 0,
        }
    }
}

// ── FuncCallState ───────────────────────────────────────────────────────

pub(crate) struct FuncCallState {
    pub(crate) in_block: bool,
    pub(crate) args: BTreeMap<i64, String>,
    pub(crate) names: BTreeMap<i64, String>,
    pub(crate) call_ids: BTreeMap<i64, String>,
    pub(crate) item_added: BTreeSet<i64>,
    pub(crate) output_index: BTreeMap<i64, i64>,
}

impl FuncCallState {
    pub(crate) fn new() -> Self {
        Self {
            in_block: false,
            args: BTreeMap::new(),
            names: BTreeMap::new(),
            call_ids: BTreeMap::new(),
            item_added: BTreeSet::new(),
            output_index: BTreeMap::new(),
        }
    }
}

// ── SessionMeta ────────────────────────────────────────────────��────────

pub(crate) struct SessionMeta {
    pub(crate) seq: i64,
    pub(crate) response_id: String,
    pub(crate) created_at: i64,
    pub(crate) first_chunk: bool,
    pub(crate) finish_reason: Option<String>,
    pub(crate) fc_id: String,
}

impl SessionMeta {
    pub(crate) fn new() -> Self {
        Self {
            seq: 0,
            response_id: String::new(),
            created_at: 0,
            first_chunk: true,
            finish_reason: None,
            fc_id: String::new(),
        }
    }

    pub(crate) fn next_seq(&mut self) -> i64 {
        self.seq += 1;
        self.seq
    }
}

// ── UsageTracker ────────────────────────────────────────────────────────

pub(crate) struct UsageTracker {
    pub(crate) seen: bool,
    pub(crate) input: i64,
    pub(crate) output: i64,
    pub(crate) total: i64,
    pub(crate) cached: i64,
    pub(crate) reasoning: i64,
}

impl UsageTracker {
    pub(crate) fn new() -> Self {
        Self {
            seen: false,
            input: 0,
            output: 0,
            total: 0,
            cached: 0,
            reasoning: 0,
        }
    }
}

// ── StreamState ─────────────────────────────────────────────────────────

pub struct StreamState {
    pub(crate) session: SessionMeta,
    pub(crate) reasoning: ReasoningState,
    pub(crate) text: TextContentState,
    pub(crate) func: FuncCallState,
    pub(crate) think: ThinkStateMachine,
    pub(crate) usage: UsageTracker,
}

impl Drop for StreamState {
    fn drop(&mut self) {
        debug!(
            "StreamState destroyed - reasoning_active: {}, in_text_block: {}, \
             in_func_block: {}, reasoning_buf_len: {}, text_buf_len: {}, \
             should_synthesize: {}",
            self.reasoning.active,
            self.text.in_block,
            self.func.in_block,
            self.reasoning.buf.len(),
            self.text.buf.len(),
            self.should_synthesize_message_from_reasoning(),
        );
    }
}

impl StreamState {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            session: SessionMeta::new(),
            reasoning: ReasoningState::new(),
            text: TextContentState::new(),
            func: FuncCallState::new(),
            think: ThinkStateMachine::new(),
            usage: UsageTracker::new(),
        }
    }

    pub(crate) fn next_seq(&mut self) -> i64 {
        self.session.next_seq()
    }

    pub fn get_text_len(&self) -> usize {
        self.text.buf.len()
    }

    pub fn get_reasoning_len(&self) -> usize {
        self.reasoning.buf.len()
    }

    pub fn get_input_tokens(&self) -> i64 {
        self.usage.input
    }

    pub(crate) fn feed_think_tag(&mut self, text: &str) -> (Vec<String>, Vec<String>) {
        self.think.feed(text)
    }

    pub(crate) fn flush_think_buf(&mut self) -> (String, bool) {
        self.think.flush()
    }

    pub(crate) fn should_synthesize_message_from_reasoning(&self) -> bool {
        let has_no_text_content = self.text.buf.is_empty() && self.text.msg_id.is_empty();
        let has_reasoning_content = !self.reasoning.buf.is_empty();
        let reasoning_not_already_sent = !self.reasoning.part_added;
        has_no_text_content && has_reasoning_content && reasoning_not_already_sent
    }
}
