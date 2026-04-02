use super::{Parser, shared};
use crate::{
    domain::BlockSourceKind,
    ingestion::ParseInput,
    ocr::{OcrProvider, OcrRequest},
};
use async_trait::async_trait;
use std::sync::Arc;

pub struct ImageParser {
    ocr: Arc<dyn OcrProvider>,
}

impl ImageParser {
    pub fn new(ocr: Arc<dyn OcrProvider>) -> Self {
        Self { ocr }
    }
}

#[async_trait]
impl Parser for ImageParser {
    fn name(&self) -> &'static str {
        "image-parser"
    }

    async fn parse(&self, input: ParseInput) -> anyhow::Result<crate::domain::DocumentIr> {
        let bytes = input
            .bytes
            .clone()
            .ok_or_else(|| anyhow::anyhow!("image parser requires uploaded bytes"))?;

        let ocr_output = self
            .ocr
            .recognize(OcrRequest {
                file_name: input.file_name.clone(),
                mime_type: input.mime_type.clone(),
                bytes,
            })
            .await?;

        let text = ocr_output
            .lines
            .iter()
            .map(|line| line.text.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        let blocks = ocr_output
            .lines
            .iter()
            .map(|line| shared::BlockInput {
                text: line.text.clone(),
                page_no: line.page_no.unwrap_or(1),
                bbox: line.bbox.clone(),
                confidence: line.confidence,
            })
            .collect::<Vec<_>>();
        let mut document =
            shared::build_document_from_blocks(input, blocks, BlockSourceKind::Ocr, self.name());
        if document.plain_text.trim().is_empty() {
            document.plain_text = text;
        }
        document.metadata.extra.insert(
            "ocr_provider".to_string(),
            ocr_output
                .provider
                .clone()
                .unwrap_or_else(|| self.ocr.name().to_string()),
        );
        if let Some(model) = ocr_output.model {
            document
                .metadata
                .extra
                .insert("ocr_model".to_string(), model);
        }
        document.metadata.extra.insert(
            "ocr_transport".to_string(),
            self.ocr.transport_name().to_string(),
        );
        Ok(document)
    }
}
