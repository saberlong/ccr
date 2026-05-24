use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{routing::get, routing::post, Router};
use tower_http::limit::RequestBodyLimitLayer;
use tokio::net::TcpListener;
use tracing::info;

use ccr::config::Config;
use ccr::handler::AppState;
use ccr::log;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    log::setup_panic_hook();

    ccr::converter::init_reasoning_cleanup();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());

    let config = Config::load(&config_path)?;
    log::init_tracing(&config.logging)?;

    info!("加载配置文件: {}", config_path);
    let config = Arc::new(config);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.server.request_timeout_secs))
        .connect_timeout(Duration::from_secs(config.server.connect_timeout_secs))
        .build()?;

    let state = AppState { config, client };

    let host = state.config.server.host.clone();
    let port = state.config.server.port;
    let upstream_url = state.config.upstream.url.clone();

    let app = Router::new()
        .route("/health", get(ccr::handler::health))
        .route("/v1/responses", post(ccr::handler::responses_handler))
        .layer(RequestBodyLimitLayer::new(state.config.server.max_body_size))
        .with_state(state);

    let addr = SocketAddr::new(host.parse()?, port);

    info!("CCR 代理启动: http://{}", addr);
    info!("端点: POST /v1/responses → {}", upstream_url);

    let listener = TcpListener::bind(addr).await?;

    axum::serve(listener, app).await?;

    Ok(())
}