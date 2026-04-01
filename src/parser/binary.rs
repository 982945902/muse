use super::{Parser, shared};
use crate::ingestion::ParseInput;
use async_trait::async_trait;

#[derive(Default)]
pub struct PlaceholderBinaryParser;

#[async_trait]
impl Parser for PlaceholderBinaryParser {
    fn name(&self) -> &'static str {
        "binary-placeholder-parser"
    }

    async fn parse(&self, input: ParseInput) -> anyhow::Result<crate::domain::DocumentIr> {
        let summary = match input.source_type {
            crate::domain::SourceType::Image => "image OCR provider not implemented yet",
            crate::domain::SourceType::Pdf => "pdf parser/OCR provider not implemented yet",
            crate::domain::SourceType::Docx => "docx parser provider not implemented yet",
            crate::domain::SourceType::Unknown => "input type could not be inferred yet",
            _ => "binary parser placeholder",
        };

        let block_source_kind = shared::infer_block_source_kind(&input.source_type);
        let text = shared::placeholder_message(&input, summary);

        Ok(shared::build_document(
            input,
            text,
            block_source_kind,
            self.name(),
        ))
    }
}
