use muse_extract_service::{app, config::Config, telemetry};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::from_env()?;
    telemetry::init(&config)?;

    let app_state = app::build_state(&config);
    let router = app::build_router(app_state);
    let listener = tokio::net::TcpListener::bind(config.listen_addr).await?;

    tracing::info!(address = %config.listen_addr, "starting extraction service");
    axum::serve(listener, router).await?;

    Ok(())
}
