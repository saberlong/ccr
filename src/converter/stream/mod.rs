pub mod completion;
pub mod conversion;
pub mod handlers;
pub mod images;
pub mod state;

pub use completion::generate_completed_events_fallback;
pub use conversion::convert_chat_stream_line;
pub use state::StreamState;

pub(crate) fn emit_sse_event(event: &str, payload: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event, payload)
}
