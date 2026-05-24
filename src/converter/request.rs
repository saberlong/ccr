use serde_json::{json, Value};
use tracing::debug;

pub struct ConvertedRequest {
    pub body: Vec<u8>,
}

pub fn convert_responses_to_chat_request(
    body: &[u8],
    mapped_model: &str,
    stream: bool,
) -> Result<ConvertedRequest, anyhow::Error> {
    let root: Value = serde_json::from_slice(body)?;

    let mut chat = json!({
        "model": mapped_model,
        "messages": [],
        "stream": stream,
    });

    if stream {
        chat["stream_options"] = json!({"include_usage": true});
    }

    if let Some(v) = root.get("max_output_tokens") {
        chat["max_tokens"] = v.clone();
    }
    if let Some(v) = root.get("parallel_tool_calls") {
        chat["parallel_tool_calls"] = v.clone();
    }
    if let Some(v) = root.get("temperature") {
        chat["temperature"] = v.clone();
    }
    if let Some(v) = root.get("top_p") {
        chat["top_p"] = v.clone();
    }
    if let Some(v) = root.get("user") {
        chat["user"] = v.clone();
    }

    if let Some(instructions) = root.get("instructions").and_then(|v| v.as_str()) {
        if !instructions.is_empty() {
            let sys_msg = json!({"role": "system", "content": instructions});
            if let Some(messages) = chat.get_mut("messages").and_then(|m| m.as_array_mut()) {
                messages.push(sys_msg);
            }
        }
    }

    if let Some(input) = root.get("input") {
        convert_input_to_messages(input, &mut chat);
    }

    if let Some(tools) = root.get("tools").and_then(|t| t.as_array()) {
        convert_tools(tools, &mut chat);
    }

    if let Some(v) = root.get("tool_choice") {
        chat["tool_choice"] = v.clone();
    }

    if let Some(reasoning) = root.get("reasoning") {
        if let Some(effort) = reasoning.get("effort").and_then(|e| e.as_str()) {
            let effort = match effort {
                "none" => "none",
                "auto" => "auto",
                "minimal" | "low" => "low",
                "medium" => "medium",
                "high" => "high",
                "xhigh" => "xhigh",
                _ => "auto",
            };
            chat["reasoning_effort"] = Value::String(effort.to_string());
        }
    }

    let body = serde_json::to_vec(&chat)?;

    Ok(ConvertedRequest { body })
}

fn convert_input_to_messages(input: &Value, chat: &mut Value) {
    match input {
        Value::String(text) => {
            let msg = json!({"role": "user", "content": text});
            if let Some(messages) = chat.get_mut("messages").and_then(|m| m.as_array_mut()) {
                messages.push(msg);
            }
        }
        Value::Array(items) => {
            let mut pending_reasoning: Option<String> = None;
            let mut pending_func_calls: Vec<FuncCallItem> = Vec::new();
            let mut pending_assistant_content: Option<String> = None;
            let mut pending_assistant_reasoning: Option<String> = None;
            let mut has_active_tool_turn = false;

            let flush_assistant_with_tool_calls =
                |chat: &mut Value,
                 funcs: &mut Vec<FuncCallItem>,
                 assistant_content: &mut Option<String>,
                 assistant_reasoning: &mut Option<String>,
                 pending_reasoning: &Option<String>,
                 active_tool_turn: &mut bool| {
                    if funcs.is_empty() && assistant_content.is_none() {
                        return;
                    }
                    *active_tool_turn = true;
                    let first_call_id = funcs.first().map(|fc| fc.call_id.clone());
                    let mut tool_calls: Vec<Value> = Vec::new();
                    let count = funcs.len();
                    for fc in funcs.drain(..) {
                        tool_calls.push(json!({
                            "id": fc.call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": fc.arguments,
                            }
                        }));
                    }
                    let mut msg = json!({"role": "assistant"});
                    if let Some(content) = assistant_content.take() {
                        if !content.is_empty() {
                            msg["content"] = json!(content);
                        }
                    }
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = json!(tool_calls);
                    }
                    let reasoning_text = assistant_reasoning
                        .take()
                        .or_else(|| pending_reasoning.clone())
                        .or_else(|| {
                            first_call_id
                                .and_then(|id| super::get_and_remove_reasoning(&id))
                        });
                    if let Some(ref text) = reasoning_text {
                        if !text.is_empty() {
                            msg["reasoning_content"] = json!(text);
                            debug!(
                                reasoning_len = text.len(),
                                tool_calls_count = count,
                                "已为合并的 function_call 消息注入 reasoning_content"
                            );
                        }
                    }
                    if let Some(messages) =
                        chat.get_mut("messages").and_then(|m| m.as_array_mut())
                    {
                        messages.push(msg);
                    }
                };

            let flush_pending_assistant =
                |chat: &mut Value,
                 assistant_content: &mut Option<String>,
                 assistant_reasoning: &mut Option<String>| {
                    if let Some(content) = assistant_content.take() {
                        let mut msg = json!({"role": "assistant", "content": content});
                        if let Some(reasoning) = assistant_reasoning.take() {
                            if !reasoning.is_empty() {
                                msg["reasoning_content"] = json!(reasoning);
                            }
                        }
                        if let Some(messages) =
                            chat.get_mut("messages").and_then(|m| m.as_array_mut())
                        {
                            messages.push(msg);
                        }
                    }
                };

            let discard_pending = |funcs: &mut Vec<FuncCallItem>,
                                   assistant_content: &mut Option<String>,
                                   assistant_reasoning: &mut Option<String>,
                                   pending_reasoning: &mut Option<String>,
                                   active_tool_turn: &mut bool| {
                funcs.clear();
                *assistant_content = None;
                *assistant_reasoning = None;
                *pending_reasoning = None;
                *active_tool_turn = false;
            };

            for item in items {
                let item_type = item
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");

                if item_type.is_empty() && item.get("role").is_some() {
                    let raw_role = item
                        .get("role")
                        .and_then(|r| r.as_str())
                        .unwrap_or("user");
                    let normalized = normalize_role(raw_role);

                    if normalized == "assistant" {
                        let content = extract_message_text_content(item);
                        pending_assistant_content = content;
                        pending_assistant_reasoning = pending_reasoning.take().or_else(|| {
                            item.get("reasoning")
                                .and_then(extract_reasoning_text)
                        });
                    } else {
                        discard_pending(
                            &mut pending_func_calls,
                            &mut pending_assistant_content,
                            &mut pending_assistant_reasoning,
                            &mut pending_reasoning,
                            &mut has_active_tool_turn,
                        );
                        convert_message_item(item, chat, &mut pending_reasoning);
                    }
                } else {
                    match item_type {
                        "message" => {
                            let raw_role = item
                                .get("role")
                                .and_then(|r| r.as_str())
                                .unwrap_or("user");
                            let normalized = normalize_role(raw_role);

                            if normalized == "assistant" {
                                let content = extract_message_text_content(item);
                                pending_assistant_content = content;
                                pending_assistant_reasoning =
                                    pending_reasoning.take().or_else(|| {
                                        item.get("reasoning")
                                            .and_then(extract_reasoning_text)
                                    });
                            } else {
                                discard_pending(
                                    &mut pending_func_calls,
                                    &mut pending_assistant_content,
                                    &mut pending_assistant_reasoning,
                                    &mut pending_reasoning,
                                    &mut has_active_tool_turn,
                                );
                                convert_message_item(item, chat, &mut pending_reasoning);
                            }
                        }
                        "reasoning" => {
                            pending_reasoning = extract_reasoning_text(item);
                        }
                        "function_call" => {
                            let call_id = item
                                .get("call_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let arguments = item
                                .get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("{}")
                                .to_string();

                            if !call_id.is_empty() {
                                pending_func_calls.push(FuncCallItem {
                                    call_id,
                                    name,
                                    arguments,
                                });
                            }
                        }
                        "function_call_output" => {
                            flush_assistant_with_tool_calls(
                                chat,
                                &mut pending_func_calls,
                                &mut pending_assistant_content,
                                &mut pending_assistant_reasoning,
                                &pending_reasoning,
                                &mut has_active_tool_turn,
                            );
                            if has_active_tool_turn {
                                convert_function_call_output_item(item, chat);
                            }
                        }
                        _ => {
                            discard_pending(
                                &mut pending_func_calls,
                                &mut pending_assistant_content,
                                &mut pending_assistant_reasoning,
                                &mut pending_reasoning,
                                &mut has_active_tool_turn,
                            );
                            debug!(%item_type, "跳过未处理的 input item 类型");
                        }
                    }
                }
            }

            flush_pending_assistant(
                chat,
                &mut pending_assistant_content,
                &mut pending_assistant_reasoning,
            );
        }
        _ => {}
    }
}

struct FuncCallItem {
    call_id: String,
    name: String,
    arguments: String,
}

fn extract_message_text_content(item: &Value) -> Option<String> {
    item.get("content")
        .and_then(|c| match c {
            Value::String(text) => {
                let t = text.trim();
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_string())
                }
            }
            Value::Array(parts) => {
                let text: Vec<&str> = parts
                    .iter()
                    .filter_map(|p| {
                        let pt = p.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if pt == "input_text" || pt == "output_text" || pt == "text" {
                            p.get("text").and_then(|t| t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if text.is_empty() {
                    None
                } else {
                    Some(text.join("\n"))
                }
            }
            _ => None,
        })
}

fn convert_message_item(item: &Value, chat: &mut Value, pending_reasoning: &mut Option<String>) {
    let raw_role = item
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("user")
        .to_string();

    let role = normalize_role(&raw_role);
    debug!(original_role = %raw_role, normalized_role = %role, "角色标准化");

    let mut msg = json!({"role": role, "content": ""});

    if let Some(content) = item.get("content") {
        match content {
            Value::String(text) => {
                msg["content"] = Value::String(text.clone());
            }
            Value::Array(parts) => {
                let mut has_media = false;
                let mut text_parts: Vec<String> = Vec::new();
                let mut chat_content: Vec<Value> = Vec::new();

                for part in parts {
                    let part_type = part
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("input_text");

                    match part_type {
                        "input_text" | "output_text" | "text" => {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                text_parts.push(text.to_string());
                                chat_content.push(json!({"type": "text", "text": text}));
                            }
                        }
                        "input_image" | "image_url" => {
                            if let Some(block) = convert_image_content(part) {
                                chat_content.push(block);
                                has_media = true;
                            }
                        }
                        _ => {}
                    }
                }

                if has_media {
                    msg["content"] = Value::Array(chat_content);
                } else if !text_parts.is_empty() {
                    msg["content"] = Value::String(text_parts.join("\n"));
                }
            }
            _ => {}
        }
    }

    if role == "assistant" {
        if let Some(reasoning) = pending_reasoning.take() {
            if !reasoning.is_empty() {
                msg["reasoning_content"] = Value::String(reasoning);
            }
        } else if let Some(reasoning) = item.get("reasoning") {
            if let Some(text) = extract_reasoning_text(reasoning) {
                if !text.is_empty() {
                    msg["reasoning_content"] = Value::String(text);
                }
            }
        }
    }

    if let Some(messages) = chat.get_mut("messages").and_then(|m| m.as_array_mut()) {
        messages.push(msg);
    }
}

fn convert_image_content(item: &Value) -> Option<Value> {
    if let Some(image_url) = item.get("image_url") {
        if image_url.is_string() {
            let url = image_url.as_str()?;
            if url.is_empty() {
                return None;
            }
            return Some(json!({
                "type": "image_url",
                "image_url": {"url": url}
            }));
        }
        let url = image_url.get("url").and_then(|u| u.as_str())?;
        if url.is_empty() {
            return None;
        }
        let mut block = json!({
            "type": "image_url",
            "image_url": {"url": url}
        });
        if let Some(detail) = image_url.get("detail") {
            block["image_url"]["detail"] = detail.clone();
        }
        return Some(block);
    }

    let source = item.get("source").or_else(|| item.get("image"))?;
    match source.get("type").and_then(|t| t.as_str()) {
        Some("url") => {
            let url = source.get("url").and_then(|t| t.as_str())?;
            let mut block = json!({
                "type": "image_url",
                "image_url": {"url": url}
            });
            if let Some(detail) = item.get("detail") {
                block["image_url"]["detail"] = detail.clone();
            }
            Some(block)
        }
        _ => None,
    }
}

fn convert_function_call_output_item(item: &Value, chat: &mut Value) {
    let call_id = item
        .get("call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let output = item
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let msg = json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": output,
    });

    if let Some(messages) = chat.get_mut("messages").and_then(|m| m.as_array_mut()) {
        messages.push(msg);
    }
}

fn extract_reasoning_text(item: &Value) -> Option<String> {
    if let Some(text) = item.as_str() {
        let t = text.trim();
        if t.is_empty() {
            return None;
        }
        return Some(t.to_string());
    }

    if let Some(obj) = item.as_object() {
        if let Some(summary) = obj.get("summary") {
            if let Some(arr) = summary.as_array() {
                for entry in arr {
                    let entry_type = entry
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    if entry_type == "summary_text" {
                        if let Some(t) = entry.get("text").and_then(|v| v.as_str()) {
                            let trimmed = t.trim();
                            if !trimmed.is_empty() {
                                return Some(trimmed.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

fn normalize_role(raw: &str) -> String {
    match raw {
        "user" | "assistant" | "system" | "tool" => raw.to_string(),
        "developer" => "system".to_string(),
        _ => "user".to_string(),
    }
}

fn convert_tools(tools: &[Value], chat: &mut Value) {
    let mut chat_tools: Vec<Value> = Vec::new();

    for tool in tools {
        let tool_type = tool
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("");

        if tool_type != "function" && !tool_type.is_empty() {
            continue;
        }

        let name = tool
            .get("name")
            .or_else(|| tool.get("function.name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if name.is_empty() {
            continue;
        }

        let description = tool
            .get("description")
            .or_else(|| tool.get("function.description"))
            .and_then(|v| v.as_str());

        let parameters = tool
            .get("parameters")
            .or_else(|| tool.get("function.parameters"));

        let mut params = match parameters {
            Some(p) => p.clone(),
            None => json!({}),
        };

        normalize_tool_parameters(&mut params);

        let mut func = json!({
            "name": name,
            "parameters": params,
        });
        if let Some(desc) = description {
            func["description"] = Value::String(desc.to_string());
        }

        chat_tools.push(json!({
            "type": "function",
            "function": func,
        }));
    }

    if !chat_tools.is_empty() {
        chat["tools"] = Value::Array(chat_tools);
    }
}

fn normalize_tool_parameters(params: &mut Value) {
    if let Some(props) = params.get_mut("properties") {
        if let Some(fields) = props.as_object_mut() {
            for (key, field) in fields.iter_mut() {
                if !key.is_empty() {
                    field["required"] = json!([]);
                }
            }
        }
    }
}