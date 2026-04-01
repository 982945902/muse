use crate::config::Config;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init(config: &Config) -> anyhow::Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_new(config.log_filter.clone())
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init()?;

    Ok(())
}
