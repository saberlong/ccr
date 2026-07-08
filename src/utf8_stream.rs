/// Buffer that accumulates byte chunks and produces valid UTF-8 strings,
/// handling multi-byte characters split across chunk boundaries.
///
/// When a TCP stream delivers bytes, a multi-byte UTF-8 character (e.g. Chinese
/// characters encoded as 3 bytes) can be split across two chunks. Using
/// String::from_utf8_lossy directly on each chunk will replace the incomplete
/// trailing bytes with the replacement character U+FFFD (�).
///
/// This buffer holds incomplete trailing bytes and prepends them to the next
/// chunk, producing clean UTF-8 output.
pub struct Utf8StreamBuffer {
    incomplete: Vec<u8>,
}

impl Utf8StreamBuffer {
    pub fn new() -> Self {
        Self {
            incomplete: Vec::new(),
        }
    }

    /// Process a new byte chunk, returning the decoded UTF-8 string.
    /// Incomplete multi-byte sequences at the end are retained internally
    /// and prepended to the next process_bytes call.
    pub fn process_bytes(&mut self, bytes: &[u8]) -> String {
        let mut combined = std::mem::take(&mut self.incomplete);
        combined.extend_from_slice(bytes);

        if combined.is_empty() {
            return String::new();
        }

        let valid_len = find_incomplete_utf8_at_end(&combined);
        let valid_bytes = &combined[..valid_len];
        let result = String::from_utf8_lossy(valid_bytes).into_owned();
        self.incomplete = combined[valid_len..].to_vec();

        result
    }

    /// Finalize, returning any remaining bytes decoded with lossy replacement.
    pub fn finalize(self) -> String {
        if self.incomplete.is_empty() {
            String::new()
        } else {
            String::from_utf8_lossy(&self.incomplete).into_owned()
        }
    }
}

impl Default for Utf8StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Scan backwards from the end of ytes to find where valid UTF-8 ends.
/// Returns the index of the first byte of an incomplete multi-byte sequence,
/// or ytes.len() if all bytes form complete characters.
pub fn find_incomplete_utf8_at_end(bytes: &[u8]) -> usize {
    let len = bytes.len();
    let max_lookback = 4.min(len);
    for i in 1..=max_lookback {
        let b = bytes[len - i];
        if b < 0x80 {
            // ASCII byte — cannot be part of a multi-byte prefix
            break;
        } else if (0xC0..=0xF7).contains(&b) {
            // Leading byte of a multi-byte sequence
            let expected_len: usize = if b < 0xE0 {
                2
            } else if b < 0xF0 {
                3
            } else {
                4
            };
            if i < expected_len {
                // This multi-byte character is truncated; return its start
                return len - i;
            }
            break;
        }
        // Continuation byte (0x80–0xBF) — keep scanning backwards
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_only() {
        let mut buf = Utf8StreamBuffer::new();
        assert_eq!(buf.process_bytes(b"hello"), "hello");
        assert_eq!(buf.process_bytes(b" world"), " world");
        assert_eq!(buf.finalize(), "");
    }

    #[test]
    fn test_multi_byte_split() {
        // "你好" in UTF-8: e4 bd a0 e5 a5 bd
        let mut buf = Utf8StreamBuffer::new();

        // First 3 bytes: e4 bd a0 → "你"
        assert_eq!(buf.process_bytes(b"\xe4\xbd\xa0"), "\u{4f60}");

        // Next 2 bytes: e5 a5 (incomplete, missing bd)
        // Should produce empty string and buffer the incomplete bytes
        assert_eq!(buf.process_bytes(b"\xe5\xa5"), "");

        // Last byte: bd completes "好"
        assert_eq!(buf.process_bytes(b"\xbd"), "\u{597d}");

        assert_eq!(buf.finalize(), "");
    }

    #[test]
    fn test_single_byte_split() {
        let mut buf = Utf8StreamBuffer::new();
        // 3-byte char split: first 2 bytes then last 1 byte
        assert_eq!(buf.process_bytes(b"\xe4\xbd"), "");
        assert_eq!(buf.process_bytes(b"\xa0"), "\u{4f60}");
    }

    #[test]
    fn test_invalid_utf8_mid_buffer() {
        let mut buf = Utf8StreamBuffer::new();
        // Invalid byte in the middle should be handled by from_utf8_lossy
        let result = buf.process_bytes(b"abc\xfe\xffxyz");
        assert!(result.contains("abc"));
        assert!(result.contains("xyz"));
    }

    #[test]
    fn test_empty_chunk() {
        let mut buf = Utf8StreamBuffer::new();
        assert_eq!(buf.process_bytes(b""), "");
        assert_eq!(buf.finalize(), "");
    }

    #[test]
    fn test_find_incomplete_utf8() {
        // Complete 3-byte sequence
        assert_eq!(find_incomplete_utf8_at_end(b"\xe4\xbd\xa0"), 3);
        // Incomplete 3-byte: only 2 bytes
        assert_eq!(find_incomplete_utf8_at_end(b"\xe4\xbd"), 0);
        // Incomplete 3-byte: only 1 byte
        assert_eq!(find_incomplete_utf8_at_end(b"\xe4"), 0);
        // ASCII only
        assert_eq!(find_incomplete_utf8_at_end(b"hello"), 5);
        // Mixed: complete multi-byte + ASCII
        assert_eq!(find_incomplete_utf8_at_end(b"\xe4\xbd\xa0abc"), 6);
        // Mixed: multi-byte + incomplete multi-byte
        assert_eq!(find_incomplete_utf8_at_end(b"a\xe4\xbd\xa0\xe5"), 4);
    }
}
