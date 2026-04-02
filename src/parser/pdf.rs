use super::{Parser, shared};
use crate::{
    domain::{BlockSourceKind, PageIr},
    ingestion::ParseInput,
    ocr::{OcrLine, OcrOutput, OcrPage, OcrProvider, OcrRequest},
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
                            request_id: None,
                            source_type: Some(input.source_type.clone()),
                            page_no_hint: None,
                            metadata: std::collections::HashMap::from([(
                                "pdf_ocr_input".to_string(),
                                "original_pdf_bytes".to_string(),
                            )]),
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
            let mut document = shared::build_document_from_blocks(
                input,
                blocks,
                BlockSourceKind::Ocr,
                self.name(),
            );
            apply_ocr_page_metadata(
                &mut document.pages,
                &ocr_output,
                &ocr_output.pages,
                &mut document.metadata.extra,
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
                            block_id: None,
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
        let mut blocks = Vec::new();
        let mut pages = Vec::new();
        let mut provider = None;
        let mut model = None;

        for raster_page in raster_pages {
            let output = self
                .ocr
                .recognize(OcrRequest {
                    file_name: input.file_name.clone(),
                    mime_type: raster_page.mime_type.clone(),
                    bytes: raster_page.bytes.clone(),
                    request_id: None,
                    source_type: Some(input.source_type.clone()),
                    page_no_hint: Some(raster_page.page_no.max(1)),
                    metadata: std::collections::HashMap::from([(
                        "pdf_ocr_input".to_string(),
                        "page_rasters".to_string(),
                    )]),
                })
                .await?;

            if provider.is_none() {
                provider = output.provider.clone();
            }
            if model.is_none() {
                model = output.model.clone();
            }

            if let Some(page) = output.pages.first() {
                pages.push(OcrPage {
                    page_no: raster_page.page_no.max(1),
                    width: page.width,
                    height: page.height,
                    rotation_degrees: page.rotation_degrees,
                    request_id: page
                        .request_id
                        .clone()
                        .or_else(|| output.request_id.clone()),
                    timing_ms: page.timing_ms.or(output.timing_ms),
                    warnings: if page.warnings.is_empty() {
                        output.warnings.clone()
                    } else {
                        page.warnings.clone()
                    },
                });
            } else {
                pages.push(OcrPage {
                    page_no: raster_page.page_no.max(1),
                    width: image::load_from_memory(&raster_page.bytes)
                        .ok()
                        .map(|image| image.width() as f32),
                    height: image::load_from_memory(&raster_page.bytes)
                        .ok()
                        .map(|image| image.height() as f32),
                    rotation_degrees: None,
                    request_id: output.request_id.clone(),
                    timing_ms: output.timing_ms,
                    warnings: output.warnings.clone(),
                });
            }

            lines.extend(output.lines.into_iter().map(|line| OcrLine {
                block_id: line.block_id,
                text: line.text,
                page_no: Some(raster_page.page_no.max(1)),
                bbox: line.bbox,
                confidence: line.confidence,
            }));
            blocks.extend(output.blocks.into_iter().map(|block| crate::ocr::OcrBlock {
                block_id: block.block_id,
                text: block.text,
                page_no: Some(raster_page.page_no.max(1)),
                bbox: block.bbox,
                confidence: block.confidence,
                line_count: block.line_count,
            }));
        }

        let request_id = aggregate_ocr_request_id(&pages);
        let timing_ms = aggregate_ocr_timing_ms(&pages);
        let warnings = aggregate_ocr_warnings(&pages);

        Ok(OcrOutput {
            pages,
            blocks,
            lines,
            request_id,
            timing_ms,
            warnings,
            provider,
            model,
        })
    }
}

fn apply_ocr_page_metadata(
    pages: &mut [PageIr],
    ocr_output: &OcrOutput,
    ocr_pages: &[OcrPage],
    metadata: &mut std::collections::HashMap<String, String>,
) {
    for page in pages {
        let page_no = page.page_no;
        if let Some(ocr_page) = ocr_pages
            .iter()
            .find(|ocr_page| ocr_page.page_no == page_no)
        {
            page.width = ocr_page.width;
            page.height = ocr_page.height;
            if let Some(rotation) = ocr_page.rotation_degrees {
                metadata.insert(
                    format!("ocr_page_{}_rotation_degrees", page_no),
                    rotation.to_string(),
                );
            }
            if let Some(request_id) = ocr_page.request_id.as_deref() {
                metadata.insert(
                    format!("ocr_page_{}_request_id", page_no),
                    request_id.to_string(),
                );
            }
            if let Some(timing_ms) = ocr_page.timing_ms {
                metadata.insert(
                    format!("ocr_page_{}_timing_ms", page_no),
                    timing_ms.to_string(),
                );
            }
            if !ocr_page.warnings.is_empty() {
                metadata.insert(
                    format!("ocr_page_{}_warning_count", page_no),
                    ocr_page.warnings.len().to_string(),
                );
                for (index, warning) in ocr_page.warnings.iter().enumerate() {
                    metadata.insert(
                        format!("ocr_page_{}_warning_{}", page_no, index + 1),
                        warning.clone(),
                    );
                }
            }
        }
        metadata.insert(
            format!("ocr_page_{}_line_count", page_no),
            ocr_output
                .lines
                .iter()
                .filter(|line| line.page_no.unwrap_or(1) == page_no)
                .count()
                .to_string(),
        );
        metadata.insert(
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
}

fn aggregate_ocr_request_id(pages: &[OcrPage]) -> Option<String> {
    let mut unique = pages
        .iter()
        .filter_map(|page| page.request_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if unique.len() == 1 {
        unique.pop()
    } else {
        None
    }
}

fn aggregate_ocr_timing_ms(pages: &[OcrPage]) -> Option<u64> {
    let mut total = 0_u64;
    let mut seen = false;
    for timing_ms in pages.iter().filter_map(|page| page.timing_ms) {
        seen = true;
        total = total.saturating_add(timing_ms);
    }
    seen.then_some(total)
}

fn aggregate_ocr_warnings(pages: &[OcrPage]) -> Vec<String> {
    let mut warnings = Vec::new();
    for warning in pages.iter().flat_map(|page| page.warnings.iter()) {
        if !warnings.iter().any(|existing| existing == warning) {
            warnings.push(warning.clone());
        }
    }
    warnings
}
