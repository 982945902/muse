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
                request_id: None,
                source_type: Some(input.source_type.clone()),
                page_no_hint: None,
                metadata: std::collections::HashMap::new(),
            })
            .await?;

        let text = ocr_output
            .lines
            .iter()
            .map(|line| line.text.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        let blocks = if !ocr_output.blocks.is_empty() {
            ocr_output
                .blocks
                .iter()
                .map(|block| shared::BlockInput {
                    block_id: Some(block.block_id.clone()),
                    text: block.text.clone(),
                    page_no: block.page_no.unwrap_or(1),
                    bbox: block.bbox.clone(),
                    confidence: block.confidence,
                })
                .collect::<Vec<_>>()
        } else {
            ocr_output
                .lines
                .iter()
                .map(|line| shared::BlockInput {
                    block_id: line.block_id.clone(),
                    text: line.text.clone(),
                    page_no: line.page_no.unwrap_or(1),
                    bbox: line.bbox.clone(),
                    confidence: line.confidence,
                })
                .collect::<Vec<_>>()
        };
        let mut document =
            shared::build_document_from_blocks(input, blocks, BlockSourceKind::Ocr, self.name());
        if document.plain_text.trim().is_empty() {
            document.plain_text = text;
        }
        for page in &mut document.pages {
            if let Some(ocr_page) = ocr_output
                .pages
                .iter()
                .find(|ocr_page| ocr_page.page_no == page.page_no)
            {
                page.width = ocr_page.width;
                page.height = ocr_page.height;
                if let Some(rotation) = ocr_page.rotation_degrees {
                    document.metadata.extra.insert(
                        format!("ocr_page_{}_rotation_degrees", page.page_no),
                        rotation.to_string(),
                    );
                }
                if let Some(request_id) = ocr_page.request_id.as_deref() {
                    document.metadata.extra.insert(
                        format!("ocr_page_{}_request_id", page.page_no),
                        request_id.to_string(),
                    );
                }
                if let Some(timing_ms) = ocr_page.timing_ms {
                    document.metadata.extra.insert(
                        format!("ocr_page_{}_timing_ms", page.page_no),
                        timing_ms.to_string(),
                    );
                }
                if !ocr_page.warnings.is_empty() {
                    document.metadata.extra.insert(
                        format!("ocr_page_{}_warning_count", page.page_no),
                        ocr_page.warnings.len().to_string(),
                    );
                    for (index, warning) in ocr_page.warnings.iter().enumerate() {
                        document.metadata.extra.insert(
                            format!("ocr_page_{}_warning_{}", page.page_no, index + 1),
                            warning.clone(),
                        );
                    }
                }
            }
            let page_no = page.page_no;
            document.metadata.extra.insert(
                format!("ocr_page_{}_line_count", page_no),
                ocr_output
                    .lines
                    .iter()
                    .filter(|line| line.page_no.unwrap_or(1) == page_no)
                    .count()
                    .to_string(),
            );
            document.metadata.extra.insert(
                format!("ocr_page_{}_block_count", page_no),
                if ocr_output.blocks.is_empty() {
                    ocr_output
                        .lines
                        .iter()
                        .filter(|line| line.page_no.unwrap_or(1) == page_no)
                        .count()
                } else {
                    ocr_output
                        .blocks
                        .iter()
                        .filter(|block| block.page_no.unwrap_or(1) == page_no)
                        .count()
                }
                .to_string(),
            );
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
        if let Some(request_id) = ocr_output.request_id {
            document
                .metadata
                .extra
                .insert("ocr_request_id".to_string(), request_id);
        }
        if let Some(timing_ms) = ocr_output.timing_ms {
            document
                .metadata
                .extra
                .insert("ocr_timing_ms".to_string(), timing_ms.to_string());
        }
        if !ocr_output.warnings.is_empty() {
            document.metadata.extra.insert(
                "ocr_warning_count".to_string(),
                ocr_output.warnings.len().to_string(),
            );
            for (index, warning) in ocr_output.warnings.into_iter().enumerate() {
                document
                    .metadata
                    .extra
                    .insert(format!("ocr_warning_{}", index + 1), warning);
            }
        }
        Ok(document)
    }
}
