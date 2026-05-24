use crate::types::*;
use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

pub fn convert_chat_response_to_responses(
    chat_body: &[u8],
    original_request: &Value,
    model_name: Option<&str>,
) -> Result<ResponsesResponse, anyhow::Error> {
    let chat: ChatResponse = serde_json::from_slice(chat_body)?;

    let chat_id = chat
        .id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::new_v4().simple()));
    let real_response_id = format!("resp_{}", &chat_id);
    let created_at = chat.created.unwrap_or_else(|| Utc::now().timestamp());

    let mut resp = ResponsesResponse {
        id: real_response_id,
        object: "response".to_string(),
        created_at,
        status: "completed".to_string(),
        model: model_name.or(chat.model.as_deref()).map(|s| s.to_string()),
        instructions: original_request
            .get("instructions")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        previous_response_id: original_request
            .get("previous_response_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        temperature: original_request.get("temperature").and_then(|v| v.as_f64()),
        top_p: original_request.get("top_p").and_then(|v| v.as_f64()),
        max_output_tokens: original_request
            .get("max_output_tokens")
            .and_then(|v| v.as_i64()),
        parallel_tool_calls: original_request
            .get("parallel_tool_calls")
            .and_then(|v| v.as_bool()),
        tool_choice: original_request.get("tool_choice").cloned(),
        tools: original_request
            .get("tools")
            .cloned()
            .map(|t| serde_json::from_value(t).unwrap_or_default()),
        reasoning: original_request.get("reasoning").cloned().map(|r| {
            serde_json::from_value(r).unwrap_or(ReasoningConfig {
                effort: None,
                summary: None,
            })
        }),
        metadata: original_request.get("metadata").cloned(),
        output: None,
        usage: None,
        error: None,
    };

    let mut outputs: Vec<OutputItem> = Vec::new();
    let mut reasoning_texts: Vec<String> = Vec::new();
    let mut content_texts: Vec<String> = Vec::new();
    let mut function_calls: Vec<OutputItem> = Vec::new();

    if let Some(choices) = &chat.choices {
        for choice in choices {
            if let Some(msg) = &choice.message {
                if let Some(rc) = extract_reasoning_content(msg) {
                    reasoning_texts.push(rc);
                }

                if let Some(content) = &msg.content {
                    let content_text = match content {
                        MessageContent::String(s) => s.clone(),
                        MessageContent::Array(parts) => parts
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    };
                    if !content_text.is_empty() {
                        content_texts.push(content_text);
                    }
                }

                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        if let Some(id) = &tc.id {
                            let name = tc.function.name.clone().unwrap_or_default();
                            let arguments = tc
                                .function
                                .arguments
                                .clone()
                                .unwrap_or_else(|| "{}".to_string());

                            function_calls.push(OutputItem {
                                id: Some(format!("fc_{}", id)),
                                item_type: "function_call".to_string(),
                                status: Some("completed".to_string()),
                                role: None,
                                content: None,
                                summary: None,
                                arguments: Some(arguments),
                                call_id: Some(id.clone()),
                                name: Some(name),
                            });
                        }
                    }
                }
            }
        }
    }

    if !reasoning_texts.is_empty() {
        let reasoning_id = format!("rs_{}_0", chat_id);
        let full_text = reasoning_texts.join("");
        outputs.push(OutputItem {
            id: Some(reasoning_id),
            item_type: "reasoning".to_string(),
            status: Some("completed".to_string()),
            role: None,
            content: None,
            summary: Some(vec![OutputSummary {
                summary_type: "summary_text".to_string(),
                text: Some(full_text),
            }]),
            arguments: None,
            call_id: None,
            name: None,
        });
    }

    if !content_texts.is_empty() {
        let msg_id = format!("msg_{}_0", chat_id);
        let full_text = content_texts.join("");
        outputs.push(OutputItem {
            id: Some(msg_id),
            item_type: "message".to_string(),
            status: Some("completed".to_string()),
            role: Some("assistant".to_string()),
            content: Some(vec![OutputContent {
                content_type: "output_text".to_string(),
                text: Some(full_text),
                annotations: Some(vec![]),
                logprobs: Some(vec![]),
            }]),
            summary: None,
            arguments: None,
            call_id: None,
            name: None,
        });
    }

    for fc in function_calls {
        outputs.push(fc);
    }

    if !outputs.is_empty() {
        resp.output = Some(outputs);
    }

    if let Some(usage) = &chat.usage {
        let input_tokens = usage.prompt_tokens.unwrap_or(0);
        let output_tokens = usage.completion_tokens.unwrap_or(0);
        let cached_tokens = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);
        let reasoning_tokens = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens);

        let actual_input = if let Some(cached) = cached_tokens {
            (input_tokens - cached).max(0)
        } else {
            input_tokens
        };

        let total = actual_input + output_tokens + cached_tokens.unwrap_or(0);

        let mut input_details = None;
        if let Some(cached) = cached_tokens {
            if cached > 0 {
                input_details = Some(InputTokensDetails {
                    cached_tokens: Some(cached),
                });
            }
        }

        let mut output_details = None;
        if let Some(rt) = reasoning_tokens {
            if rt > 0 {
                output_details = Some(OutputTokensDetails {
                    reasoning_tokens: Some(rt),
                });
            }
        }

        resp.usage = Some(ResponsesUsage {
            input_tokens: actual_input,
            output_tokens,
            total_tokens: total,
            cache_read_input_tokens: cached_tokens,
            cache_creation_input_tokens: None,
            cache_creation_5m_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            input_tokens_details: input_details,
            output_tokens_details: output_details,
        });
    }

    Ok(resp)
}

fn extract_reasoning_content(msg: &ChatMessage) -> Option<String> {
    if let Some(content) = &msg.content {
        match content {
            MessageContent::String(s) => {
                let trimmed = s.trim();
                if trimmed.starts_with("```reasoning") || trimmed.starts_with("```thinking") {
                    let inner = trimmed
                        .strip_prefix("```reasoning\n")
                        .or_else(|| trimmed.strip_prefix("```thinking\n"))
                        .or_else(|| trimmed.strip_prefix("```reasoning"))
                        .or_else(|| trimmed.strip_prefix("```thinking"))
                        .unwrap_or("")
                        .trim_end_matches("\n```")
                        .trim_end_matches("```");
                    if !inner.is_empty() {
                        return Some(inner.to_string());
                    }
                }
                None
            }
            _ => None,
        }
    } else {
        None
    }
}
