use std::fs;

use crate::config::LogConfig;
use tracing_subscriber::{
    fmt, layer::Layer, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};
use tracing_appender::rolling::{RollingFileAppender, Rotation};

pub fn init_tracing(config: &LogConfig) -> anyhow::Result<()> {
    let env_filter = format!("ccr={}", config.level);

    if !config.dir.is_empty() && config.rotation {
        fs::create_dir_all(&config.dir)?;
        let file_appender = RollingFileAppender::builder()
            .rotation(Rotation::DAILY)
            .filename_prefix(&config.file_prefix)
            .filename_suffix("log")
            .max_log_files(config.retention_days)
            .build(&config.dir)?;

        let file_layer = if config.json {
            fmt::layer()
                .with_target(false)
                .json()
                .with_writer(file_appender)
                .boxed()
        } else {
            fmt::layer()
                .with_target(false)
                .with_ansi(false)
                .with_writer(file_appender)
                .boxed()
        };

        tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&env_filter)))
            .with(file_layer)
            .init();
    } else if config.json {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&env_filter)),
            )
            .with_target(false)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&env_filter)),
            )
            .with_target(false)
            .init();
    }

    Ok(())
}

pub fn setup_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        let location = panic_info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "未知位置".to_string());

        let payload = panic_info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "非字符串 panic".to_string());

        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("未命名线程");

        let backtrace = std::backtrace::Backtrace::capture();

        tracing::error!(
            panic_payload = %payload,
            location = %location,
            thread = %thread_name,
            backtrace = %backtrace,
            "🚨 应用程序发生 Panic"
        );

        eprintln!("\n🚨 应用程序发生 Panic");
        eprintln!("📍 位置: {}", location);
        eprintln!("💬 信息: {}", payload);
        eprintln!("🧵 线程: {}", thread_name);
        eprintln!("\n📋 调用栈:");
        eprintln!("{}", backtrace);
        eprintln!("\n请检查日志获取详细信息");
    }));
}