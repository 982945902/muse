use super::{Parser, shared};
use crate::{domain::BlockSourceKind, ingestion::ParseInput};
use async_trait::async_trait;

#[derive(Default)]
pub struct UrlParser;

#[async_trait]
impl Parser for UrlParser {
    fn name(&self) -> &'static str {
        "url-parser"
    }

    async fn parse(&self, input: ParseInput) -> anyhow::Result<crate::domain::DocumentIr> {
        let text = input
            .url
            .as_ref()
            .map(|url| format!("url ingestion placeholder: {url}"))
            .unwrap_or_default();

        Ok(shared::build_document(
            input,
            text,
            BlockSourceKind::Synthetic,
            self.name(),
        ))
    }
}
