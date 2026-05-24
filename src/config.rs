use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub upstream: UpstreamConfig,
    #[serde(default)]
    pub logging: LogConfig,
    #[serde(default)]
    pub streaming: StreamingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamConfig {
    pub url: String,
    pub api_key: String,
    #[serde(default)]
    pub extra_headers: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub model_mapping: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LogConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub json: bool,
    #[serde(default = "default_log_dir")]
    pub dir: String,
    #[serde(default = "default_log_file_prefix")]
    pub file_prefix: String,
    #[serde(default = "default_log_rotation")]
    pub rotation: bool,
    #[serde(default = "default_retention_days")]
    pub retention_days: usize,
}

fn default_log_dir() -> String {
    "logs".to_string()
}

fn default_log_file_prefix() -> String {
    "ccr".to_string()
}

fn default_retention_days() -> usize {
    3
}

fn default_log_rotation() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamingConfig {
    #[serde(default = "default_keepalive_interval")]
    pub keepalive_interval_secs: u64,
    #[serde(default = "default_enable_usage_injection")]
    pub enable_usage_injection: bool,
    #[serde(default = "default_preflight_timeout")]
    pub preflight_timeout_secs: u64,
    #[serde(default = "default_total_timeout")]
    pub total_timeout_secs: u64,
    #[serde(default = "default_enable_preflight")]
    pub enable_preflight: bool,
}

fn default_keepalive_interval() -> u64 {
    15
}
fn default_enable_usage_injection() -> bool {
    true
}
fn default_preflight_timeout() -> u64 {
    30
}
fn default_total_timeout() -> u64 {
    600
}
fn default_enable_preflight() -> bool {
    true
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            keepalive_interval_secs: default_keepalive_interval(),
            enable_usage_injection: default_enable_usage_injection(),
            preflight_timeout_secs: default_preflight_timeout(),
            total_timeout_secs: default_total_timeout(),
            enable_preflight: default_enable_preflight(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8180
}

fn default_connect_timeout_secs() -> u64 {
    30
}

fn default_request_timeout_secs() -> u64 {
    600
}

fn default_max_body_size() -> usize {
    10 * 1024 * 1024
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Cannot read config file: {:?}", path.as_ref()))?;
        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {:?}", path.as_ref()))?;
        Ok(config)
    }

    pub fn map_model<'a>(&'a self, model: &'a str) -> &'a str {
        if let Some(mapped) = self.upstream.model_mapping.get(model) {
            mapped.as_str()
        } else {
            model
        }
    }
}
