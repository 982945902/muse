use super::{Parser, shared};
use crate::{
    domain::BlockSourceKind,
    ingestion::ParseInput,
    ocr::{OcrLine, OcrOutput, OcrProvider, OcrRequest},
    pdf::{PdfOcrPage, PdfProvider, PdfRequest},
};
use async_trait::async_trait;
use std::collections::BTreeSet;
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
        let (mut document, ocr_metadata, pdf_ocr_input) = if fallback_to_ocr {
            let (ocr_output, pdf_ocr_input) = if pdf_output.raster_pages.is_empty() {
                (
                    self.ocr
                        .recognize(OcrRequest {
                            file_name: input.file_name.clone(),
                            mime_type: input.mime_type.clone(),
                            bytes,
                        })
                        .await?,
                    "original_pdf_bytes",
                )
            } else {
                validate_raster_pages(&pdf_output.raster_pages)?;
                (
                    self.recognize_raster_pages(&input, &pdf_output.raster_pages)
                        .await?,
                    "page_rasters",
                )
            };

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
            let mut document = shared::build_document_from_blocks(
                input,
                blocks,
                BlockSourceKind::Ocr,
                self.name(),
            );
            if document.plain_text.trim().is_empty() {
                document.plain_text = ocr_output
                    .lines
                    .iter()
                    .map(|line| line.text.trim())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            (document, Some(ocr_output), Some(pdf_ocr_input.to_string()))
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
                shared::build_document_from_page_blocks(
                    input,
                    page_blocks,
                    BlockSourceKind::Synthetic,
                    self.name(),
                ),
                None,
                None,
            )
        };
        document
            .metadata
            .extra
            .insert("pdf_provider".to_string(), self.pdf.name().to_string());
        if let Some(raster_provider) = pdf_output.raster_provider.clone() {
            document
                .metadata
                .extra
                .insert("pdf_raster_provider".to_string(), raster_provider);
        }
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
        if let Some(pdf_ocr_input) = pdf_ocr_input {
            document
                .metadata
                .extra
                .insert("pdf_ocr_input".to_string(), pdf_ocr_input);
        }
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
            document.metadata.extra.insert(
                "ocr_transport".to_string(),
                self.ocr.transport_name().to_string(),
            );
        }
        Ok(document)
    }
}

fn validate_raster_pages(raster_pages: &[PdfOcrPage]) -> anyhow::Result<()> {
    let mut seen_page_nos = BTreeSet::new();

    for page in raster_pages {
        if page.page_no == 0 {
            anyhow::bail!("pdf raster_pages must use page_no >= 1");
        }
        if page.bytes.is_empty() {
            anyhow::bail!(
                "pdf raster page {} must contain non-empty image bytes",
                page.page_no
            );
        }
        if !seen_page_nos.insert(page.page_no) {
            anyhow::bail!(
                "pdf raster_pages contains duplicated page_no `{}`",
                page.page_no
            );
        }
    }

    Ok(())
}

impl PdfParser {
    async fn recognize_raster_pages(
        &self,
        input: &ParseInput,
        raster_pages: &[PdfOcrPage],
    ) -> anyhow::Result<OcrOutput> {
        let mut lines = Vec::new();
        let mut provider = None;
        let mut model = None;

        for raster_page in raster_pages {
            let output = self
                .ocr
                .recognize(OcrRequest {
                    file_name: input.file_name.clone(),
                    mime_type: raster_page.mime_type.clone(),
                    bytes: raster_page.bytes.clone(),
                })
                .await?;

            if provider.is_none() {
                provider = output.provider.clone();
            }
            if model.is_none() {
                model = output.model.clone();
            }

            lines.extend(output.lines.into_iter().map(|line| OcrLine {
                text: line.text,
                page_no: Some(raster_page.page_no.max(1)),
                bbox: line.bbox,
                confidence: line.confidence,
            }));
        }

        Ok(OcrOutput {
            lines,
            provider,
            model,
        })
    }
}
