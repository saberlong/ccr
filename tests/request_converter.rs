use ccr::converter::convert_responses_to_chat_request;
use serde_json::{json, Value};

fn call(body: &str, model: &str, stream: bool) -> Value {
    let result = convert_responses_to_chat_request(body.as_bytes(), model, stream).unwrap();
    serde_json::from_slice(&result.body).unwrap()
}

fn messages_from(chat: &Value) -> &Vec<Value> {
    chat["messages"].as_array().unwrap()
}

mod basic_input_types {
    use super::*;

    #[test]
    fn string_input_becomes_user_message() {
        let input = json!({"model": "gpt-5", "input": "hello world"});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello world");
    }

    #[test]
    fn empty_array_input_produces_no_messages() {
        let input = json!({"model": "gpt-5", "input": []});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn null_input_produces_no_messages() {
        let input = json!({"model": "gpt-5"});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn instructions_becomes_system_message_at_beginning() {
        let input = json!({
            "model": "gpt-5",
            "instructions": "You are helpful",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn empty_instructions_are_skipped() {
        let input = json!({
            "model": "gpt-5",
            "instructions": "",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }
}

mod simple_message_conversion {
    use super::*;

    #[test]
    fn single_user_message() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello");
    }

    #[test]
    fn multiple_user_messages() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "first"},
                {"type": "message", "role": "user", "content": "second"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "first");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "second");
    }

    #[test]
    fn developer_role_maps_to_system() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "developer", "content": "system prompt"}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "system prompt");
    }

    #[test]
    fn item_without_type_uses_role_field() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"role": "user", "content": "bare item"}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "bare item");
    }

    #[test]
    fn unknown_role_defaults_to_user() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "unknown_role", "content": "text"}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn user_assistant_user_sequence_pending_assistant_discarded() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "q1"},
                {"type": "message", "role": "assistant", "content": "a1"},
                {"type": "message", "role": "user", "content": "q2"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(
            msgs.len(),
            2,
            "pending assistant text discarded before user message"
        );
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "q1");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "q2");
    }
}

mod content_formats {
    use super::*;

    #[test]
    fn content_as_string() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "plain text"}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs[0]["content"], "plain text");
    }

    #[test]
    fn content_as_text_parts_array() {
        let input = json!({
            "model": "gpt-5",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "part1"},
                    {"type": "input_text", "text": "part2"}
                ]
            }]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs[0]["content"], "part1\npart2");
    }

    #[test]
    fn content_array_with_image_uses_chat_format() {
        let input = json!({
            "model": "gpt-5",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"}
                ]
            }]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        let content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "describe this");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "https://example.com/img.png"
        );
    }

    #[test]
    fn empty_content_produces_empty_string() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": ""}]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs[0]["content"], "");
    }
}

mod function_call_core {
    use super::*;

    #[test]
    fn single_function_call_with_output() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "what is the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\": \"beijing\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny"
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", true);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        let tc = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0]["id"], "call_1");
        assert_eq!(tc[0]["function"]["name"], "get_weather");
        assert_eq!(tc[0]["function"]["arguments"], "{\"city\": \"beijing\"}");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_1");
        assert_eq!(msgs[2]["content"], "sunny");
    }

    #[test]
    fn multiple_function_calls_merged_into_single_assistant() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\": \"beijing\"}"
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_time",
                    "arguments": "{\"tz\": \"UTC\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "14:00"
                },
                {"type": "message", "role": "developer", "content": "next turn"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", true);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        let tc = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tc.len(), 2);
        assert_eq!(tc[0]["id"], "call_1");
        assert_eq!(tc[1]["id"], "call_2");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_1");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "call_2");
        assert_eq!(msgs[4]["role"], "system");
    }

    #[test]
    fn function_calls_without_outputs_are_discarded() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}"
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_time",
                    "arguments": "{}"
                },
                {"type": "message", "role": "developer", "content": "next turn"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "system");
    }

    #[test]
    fn orphan_function_call_output_is_skipped() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "function_call_output",
                    "call_id": "call_orphan",
                    "output": "result"
                },
                {"type": "message", "role": "user", "content": "next"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn function_call_with_empty_call_id_is_ignored() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "",
                    "name": "get_weather",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "orphan",
                    "output": "result"
                },
                {"type": "message", "role": "user", "content": "next"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "user");
    }
}

mod assistant_text_handling {
    use super::*;

    #[test]
    fn assistant_text_merged_with_tool_calls() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": "Let me check"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny"
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Let me check");
        let tc = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0]["id"], "call_1");
        assert_eq!(msgs[2]["role"], "tool");
    }

    #[test]
    fn pending_assistant_flushed_at_end() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": "The weather is sunny"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "The weather is sunny");
    }

    #[test]
    fn only_function_calls_without_assistant_text() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny"
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert!(msgs[1].get("content").is_none());
        assert!(msgs[1].get("tool_calls").is_some());
        assert_eq!(msgs[2]["role"], "tool");
    }
}

mod complex_sequences {
    use super::*;

    #[test]
    fn two_turns_of_tool_calls() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "turn1"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "f1",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "r1"
                },
                {"type": "message", "role": "user", "content": "turn2"},
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "f2",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": "r2"
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "turn1");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["tool_calls"].as_array().unwrap()[0]["id"], "c1");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "c1");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[3]["content"], "turn2");
        assert_eq!(msgs[4]["role"], "assistant");
        assert_eq!(msgs[4]["tool_calls"].as_array().unwrap()[0]["id"], "c2");
        assert_eq!(msgs[5]["role"], "tool");
        assert_eq!(msgs[5]["tool_call_id"], "c2");
    }

    #[test]
    fn user_then_tool_calls_then_user_again() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "q1"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "search",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "found"
                },
                {"type": "message", "role": "user", "content": "q2"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "q1");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[3]["content"], "q2");
    }

    #[test]
    fn interleaved_assistant_and_function_calls() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "q"},
                {"type": "message", "role": "assistant", "content": "thinking..."},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "a",
                    "arguments": "{}"
                },
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "b",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "ra"
                },
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": "rb"
                },
                {"type": "message", "role": "user", "content": "followup"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "q");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "thinking...");
        let tc = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tc.len(), 2);
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[4]["role"], "user");
        assert_eq!(msgs[4]["content"], "followup");
    }
}

mod reasoning_handling {
    use super::*;

    #[test]
    fn reasoning_before_function_call_produces_reasoning_content() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I need to call get_weather"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny"
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["reasoning_content"], "I need to call get_weather");
        assert!(msgs[1].get("tool_calls").is_some());
        assert_eq!(msgs[2]["role"], "tool");
    }

    #[test]
    fn reasoning_with_assistant_text_merged() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Let me think about this"}]
                },
                {"type": "message", "role": "assistant", "content": "Here is my answer"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "results"
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Here is my answer");
        assert_eq!(msgs[1]["reasoning_content"], "Let me think about this");
        assert!(msgs[1].get("tool_calls").is_some());
        assert_eq!(msgs[2]["role"], "tool");
    }

    #[test]
    fn reasoning_with_only_assistant_text_no_tool_calls() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "simple answer"}]
                },
                {"type": "message", "role": "assistant", "content": "done"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "done");
        assert_eq!(msgs[1]["reasoning_content"], "simple answer");
    }

    #[test]
    fn message_with_inline_reasoning_field() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "answer",
                    "reasoning": {
                        "summary": [{"type": "summary_text", "text": "thinking process"}]
                    }
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "answer");
        assert_eq!(msgs[1]["reasoning_content"], "thinking process");
    }

    #[test]
    fn plain_string_reasoning() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I am thinking..."}]},
                {"type": "message", "role": "assistant", "content": "answer"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "answer");
        assert_eq!(msgs[1]["reasoning_content"], "I am thinking...");
    }
}

mod tools_conversion {
    use super::*;

    #[test]
    fn tools_array_is_converted() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["description"], "Get weather info");
    }

    #[test]
    fn tool_parameters_normalized_with_required_empty_arrays() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string"}
                    }
                }
            }]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let tools = chat["tools"].as_array().unwrap();
        let params = &tools[0]["function"]["parameters"];
        assert!(
            params["properties"]["city"].get("required").is_none(),
            "required should not be injected"
        );
        assert!(
            params["properties"]["unit"].get("required").is_none(),
            "required should not be injected"
        );
    }

    #[test]
    fn tool_without_description_is_valid() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "name": "simple_tool",
                "parameters": {}
            }]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "simple_tool");
    }

    #[test]
    fn empty_tools_produces_no_tools_key() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": []
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert!(chat.get("tools").is_none());
    }

    #[test]
    fn non_function_tool_type_is_skipped() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": [{
                "type": "web_search",
                "name": "search"
            }]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert!(chat.get("tools").is_none());
    }

    #[test]
    fn tool_without_name_is_skipped() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "description": "no name"
            }]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert!(chat.get("tools").is_none());
    }

    #[test]
    fn multiple_tools_mixed_valid_and_invalid() {
        let input = json!({
            "model": "gpt-5",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": [
                {"type": "function", "name": "valid_tool"},
                {"type": "function"},
                {"type": "function", "name": "another_valid"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["function"]["name"], "valid_tool");
        assert_eq!(tools[1]["function"]["name"], "another_valid");
    }
}

mod top_level_parameters {
    use super::*;

    #[test]
    fn model_is_mapped() {
        let input = json!({"model": "gpt-5", "input": []});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["model"], "deepseek-chat");
    }

    #[test]
    fn stream_true_adds_stream_options() {
        let input = json!({"model": "gpt-5", "input": []});
        let chat = call(&input.to_string(), "deepseek-chat", true);
        assert_eq!(chat["stream"], true);
        assert_eq!(chat["stream_options"]["include_usage"], true);
    }

    #[test]
    fn stream_false_no_stream_options() {
        let input = json!({"model": "gpt-5", "input": []});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["stream"], false);
        assert!(chat.get("stream_options").is_none());
    }

    #[test]
    fn max_output_tokens_maps_to_max_tokens() {
        let input = json!({"model": "gpt-5", "input": [], "max_output_tokens": 100});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["max_tokens"], 100);
    }

    #[test]
    fn temperature_is_passed_through() {
        let input = json!({"model": "gpt-5", "input": [], "temperature": 0.7});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["temperature"], 0.7);
    }

    #[test]
    fn top_p_is_passed_through() {
        let input = json!({"model": "gpt-5", "input": [], "top_p": 0.9});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["top_p"], 0.9);
    }

    #[test]
    fn user_field_is_passed_through() {
        let input = json!({"model": "gpt-5", "input": [], "user": "user_123"});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["user"], "user_123");
    }

    #[test]
    fn parallel_tool_calls_is_passed_through() {
        let input = json!({"model": "gpt-5", "input": [], "parallel_tool_calls": false});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["parallel_tool_calls"], false);
    }

    #[test]
    fn tool_choice_is_passed_through() {
        let input = json!({"model": "gpt-5", "input": [], "tool_choice": "auto"});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert_eq!(chat["tool_choice"], "auto");
    }

    #[test]
    fn reasoning_effort_is_normalized() {
        let cases = [
            ("none", "none"),
            ("auto", "auto"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("unknown", "auto"),
        ];
        for (input_effort, expected) in &cases {
            let input = json!({
                "model": "gpt-5",
                "input": [],
                "reasoning": {"effort": input_effort}
            });
            let chat = call(&input.to_string(), "deepseek-chat", false);
            assert_eq!(
                chat["reasoning_effort"],
                expected.to_string(),
                "effort '{}' should map to '{}'",
                input_effort,
                expected
            );
        }
    }

    #[test]
    fn missing_optional_fields_are_omitted() {
        let input = json!({"model": "gpt-5", "input": []});
        let chat = call(&input.to_string(), "deepseek-chat", false);
        assert!(chat.get("max_tokens").is_none());
        assert!(chat.get("temperature").is_none());
        assert!(chat.get("user").is_none());
        assert!(chat.get("tool_choice").is_none());
        assert!(chat.get("reasoning_effort").is_none());
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn function_call_output_only_first_item() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "orphan",
                    "output": "result"
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn all_outputs_without_calls() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "function_call_output", "call_id": "o1", "output": "r1"},
                {"type": "function_call_output", "call_id": "o2", "output": "r2"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn unknown_item_type_is_gracefully_skipped() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "unknown_type", "data": "ignored"},
                {"type": "message", "role": "user", "content": "world"}
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["content"], "hello");
        assert_eq!(msgs[1]["content"], "world");
    }

    #[test]
    fn reasoning_only_no_messages() {
        let input = json!({
            "model": "gpt-5",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "just thinking"}]
                }
            ]
        });
        let chat = call(&input.to_string(), "deepseek-chat", false);
        let msgs = messages_from(&chat);
        assert_eq!(msgs.len(), 0);
    }
}
