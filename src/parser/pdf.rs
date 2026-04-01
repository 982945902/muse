use super::{Parser, shared};
use crate::{
    domain::BlockSourceKind,
    ingestion::ParseInput,
    ocr::{OcrProvider, OcrRequest},
    pdf::{PdfProvider, PdfRequest},
};
use async_trait::async_trait;
use std::sync::Arc;

pub struct PdfParser {
    pdf: Arc<dyn PdfProvider>,
    ocr: Arc<dyn OcrProvider>,
}

impl PdfParser {
    pub fn new(pdf: Arc<dyn PdfProvider>, ocr: Arc<dyn OcrProvider>) -> Self {
        Self { pdf, ocr }
    }
}

#[async_trait]
impl Parser for PdfParser {
    fn name(&self) -> &'static str {
        "pdf-parser"
    }

    async fn parse(&self, input: ParseInput) -> anyhow::Result<crate::domain::DocumentIr> {
        let bytes = input
            .bytes
            .clone()
            .ok_or_else(|| anyhow::anyhow!("pdf parser requires uploaded bytes"))?;

        let pdf_output = self
            .pdf
            .extract(PdfRequest {
                file_name: input.file_name.clone(),
                mime_type: input.mime_type.clone(),
                bytes: bytes.clone(),
            })
            .await?;

        let fallback_to_ocr = !pdf_output.extracted_text_layer;
        let (text, page_blocks, block_source_kind, ocr_metadata) = if fallback_to_ocr {
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
            let page_blocks = vec![
                ocr_output
                    .lines
                    .iter()
                    .map(|line| shared::BlockInput {
                        text: line.text.clone(),
                        page_no: line.page_no.unwrap_or(1),
                        bbox: line.bbox.clone(),
                        confidence: line.confidence,
                    })
                    .collect::<Vec<_>>(),
            ];
            (text, page_blocks, BlockSourceKind::Ocr, Some(ocr_output))
        } else {
            let page_blocks = pdf_output
                .page_texts
                .iter()
                .enumerate()
                .map(|(index, page_text)| {
                    page_text
                        .lines()
                        .map(|line| shared::BlockInput {
                            text: line.to_string(),
                            page_no: (index + 1) as u32,
                            bbox: None,
                            confidence: None,
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (
                pdf_output.text,
                page_blocks,
                BlockSourceKind::Synthetic,
                None,
            )
        };

        let mut document = shared::build_document_from_page_blocks(
            input,
            page_blocks,
            block_source_kind,
            self.name(),
        );
        if document.plain_text.trim().is_empty() {
            document.plain_text = text;
        }
        document
            .metadata
            .extra
            .insert("pdf_provider".to_string(), self.pdf.name().to_string());
        if let Some(page_count) = pdf_output.page_count {
            document
                .metadata
                .extra
                .insert("page_count".to_string(), page_count.to_string());
        }
        document.metadata.extra.insert(
            "extracted_text_layer".to_string(),
            pdf_output.extracted_text_layer.to_string(),
        );
        document.metadata.extra.insert(
            "pdf_parse_mode".to_string(),
            if fallback_to_ocr {
                "ocr_fallback".to_string()
            } else {
                "text_layer".to_string()
            },
        );
        if let Some(ocr_output) = ocr_metadata {
            document.metadata.extra.insert(
                "ocr_provider".to_string(),
                ocr_output
                    .provider
                    .unwrap_or_else(|| self.ocr.name().to_string()),
            );
            if let Some(model) = ocr_output.model {
                document
                    .metadata
                    .extra
                    .insert("ocr_model".to_string(), model);
            }
            document
                .metadata
                .extra
                .insert("ocr_transport".to_string(), self.ocr.name().to_string());
        }
        Ok(document)
    }
}
