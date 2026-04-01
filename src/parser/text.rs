use super::{Parser, shared};
use crate::{domain::BlockSourceKind, ingestion::ParseInput};
use async_trait::async_trait;

#[derive(Default)]
pub struct TextParser;

#[async_trait]
impl Parser for TextParser {
    fn name(&self) -> &'static str {
        "text-parser"
    }

    async fn parse(&self, input: ParseInput) -> anyhow::Result<crate::domain::DocumentIr> {
        let text = shared::decode_text_payload(input.text.clone(), input.bytes.clone())?;
        Ok(shared::build_document(
            input,
            text,
            BlockSourceKind::NativeText,
            self.name(),
        ))
    }
}
