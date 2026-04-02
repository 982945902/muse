use super::{Parser, shared};
use crate::{
    docx::{DocxProvider, DocxRequest},
    domain::BlockSourceKind,
    ingestion::ParseInput,
};
use async_trait::async_trait;
use std::sync::Arc;

pub struct DocxParser {
    docx: Arc<dyn DocxProvider>,
}

impl DocxParser {
    pub fn new(docx: Arc<dyn DocxProvider>) -> Self {
        Self { docx }
    }
}

#[async_trait]
impl Parser for DocxParser {
    fn name(&self) -> &'static str {
        "docx-parser"
    }

    async fn parse(&self, input: ParseInput) -> anyhow::Result<crate::domain::DocumentIr> {
        let bytes = input
            .bytes
            .clone()
            .ok_or_else(|| anyhow::anyhow!("docx parser requires uploaded bytes"))?;

        let docx_output = self
            .docx
            .extract(DocxRequest {
                file_name: input.file_name.clone(),
                mime_type: input.mime_type.clone(),
                bytes,
            })
            .await?;

        let page_blocks = vec![
            docx_output
                .paragraphs
                .iter()
                .map(|paragraph| shared::BlockInput {
                    block_id: None,
                    text: paragraph.clone(),
                    page_no: 1,
                    bbox: None,
                    confidence: None,
                })
                .collect::<Vec<_>>(),
        ];
        let mut document = shared::build_document_from_page_blocks(
            input,
            page_blocks,
            BlockSourceKind::Synthetic,
            self.name(),
        );
        if document.plain_text.trim().is_empty() {
            document.plain_text = docx_output.text;
        }
        document
            .metadata
            .extra
            .insert("docx_provider".to_string(), self.docx.name().to_string());
        if let Some(paragraph_count) = docx_output.paragraph_count {
            document
                .metadata
                .extra
                .insert("paragraph_count".to_string(), paragraph_count.to_string());
        }
        Ok(document)
    }
}
