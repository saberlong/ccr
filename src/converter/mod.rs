use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;

pub mod request;
pub mod response;
pub mod stream;

pub use request::convert_responses_to_chat_request;
pub use response::convert_chat_response_to_responses;
pub use stream::convert_chat_stream_line;

struct ReasoningEntry {
    text: String,
    stored_at: Instant,
}

static REASONING_STORE: LazyLock<DashMap<String, ReasoningEntry>> = LazyLock::new(DashMap::new);

static CLEANUP_STARTED: OnceLock<()> = OnceLock::new();

const REASONING_TTL_SECS: u64 = 3600;
const CLEANUP_INTERVAL_SECS: u64 = 60;

/// Start the background cleanup task. Safe to call multiple times; only the
/// first call spawns a daemon thread that periodically removes expired
/// entries from the store.
///
/// Cleanup uses `DashMap::retain`, which locks only one shard at a time,
/// so concurrent lookups on other shards are unaffected.
pub fn init_reasoning_cleanup() {
    CLEANUP_STARTED.get_or_init(|| {
        std::thread::spawn(|| loop {
            std::thread::sleep(Duration::from_secs(CLEANUP_INTERVAL_SECS));
            REASONING_STORE.retain(|_, v| v.stored_at.elapsed().as_secs() < REASONING_TTL_SECS);
        });
    });
}

pub fn get_and_remove_reasoning(call_id: &str) -> Option<String> {
    if let Some(entry) = REASONING_STORE.get(call_id) {
        if entry.stored_at.elapsed().as_secs() < REASONING_TTL_SECS {
            drop(entry);
            return REASONING_STORE.remove(call_id).map(|(_, e)| e.text);
        }
        // Entry exists but expired — remove it
        drop(entry);
        REASONING_STORE.remove(call_id);
    }
    None
}

pub fn set_reasoning(call_id: String, text: String) {
    let now = Instant::now();
    REASONING_STORE.insert(
        call_id,
        ReasoningEntry {
            text,
            stored_at: now,
        },
    );
    // Inline eager cleanup of already-expired entries
    REASONING_STORE.retain(|_, v| v.stored_at.elapsed().as_secs() < REASONING_TTL_SECS);
}
