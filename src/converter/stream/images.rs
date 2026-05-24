/// Extract image URLs from text content. Handles:
/// - Markdown image syntax: ![alt](url)
/// - Raw data: URLs: data:image/...
pub(crate) fn extract_image_urls_from_text(text: &str) -> Vec<serde_json::Value> {
    let mut images: Vec<serde_json::Value> = Vec::new();
    if text.is_empty() {
        return images;
    }

    // Scan for data:image URLs
    let mut remaining = text;
    while let Some(start) = remaining.find("data:image/") {
        let slice = &remaining[start..];
        let end = slice
            .find(|c: char| c.is_whitespace() || c == ')' || c == '>')
            .unwrap_or(slice.len());
        let url = slice[..end]
            .trim_end_matches(')')
            .trim_end_matches('>')
            .to_string();
        if !url.is_empty()
            && !images
                .iter()
                .any(|v| v["image_url"]["url"].as_str() == Some(&url))
        {
            images.push(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": url},
            }));
        }
        remaining = &slice[end..];
        if remaining.is_empty() {
            break;
        }
    }

    // Scan for markdown image syntax: ![alt](url)
    let mut remaining = text;
    while let Some(start) = remaining.find("![") {
        let slice = &remaining[start..];
        if let Some(bracket_end) = slice.find(']') {
            let after_alt = &slice[bracket_end + 1..];
            if let Some(paren_content) = after_alt.strip_prefix('(') {
                if let Some(paren_end) = paren_content.find(')') {
                    let url = paren_content[..paren_end].to_string();
                    if !url.is_empty()
                        && (url.starts_with("http://")
                            || url.starts_with("https://")
                            || url.starts_with("data:"))
                        && !images
                            .iter()
                            .any(|v| v["image_url"]["url"].as_str() == Some(&url))
                    {
                        images.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": {"url": url},
                        }));
                    }
                    remaining = &paren_content[paren_end + 1..];
                    continue;
                }
            }
        }
        remaining = &slice[2..];
        if remaining.is_empty() {
            break;
        }
    }

    images
}
