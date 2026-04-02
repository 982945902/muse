use crate::{
    domain::{BBox, BlockSourceKind, DocumentIr, DocumentMetadata, PageIr, SourceType, TextBlock},
    ingestion::ParseInput,
};
use std::collections::BTreeMap;
use uuid::Uuid;

pub fn build_document(
    input: ParseInput,
    text: String,
    block_source_kind: BlockSourceKind,
    parser_provider: &'static str,
) -> DocumentIr {
    build_document_from_page_blocks(
        input,
        vec![vec![BlockInput {
            block_id: None,
            text,
            page_no: 1,
            bbox: None,
            confidence: None,
        }]],
        block_source_kind,
        parser_provider,
    )
}

#[derive(Clone, Debug)]
pub struct BlockInput {
    pub block_id: Option<String>,
    pub text: String,
    pub page_no: u32,
    pub bbox: Option<BBox>,
    pub confidence: Option<f32>,
}

pub fn build_document_from_page_blocks(
    input: ParseInput,
    page_blocks: Vec<Vec<BlockInput>>,
    block_source_kind: BlockSourceKind,
    parser_provider: &'static str,
) -> DocumentIr {
    let mut extra = std::collections::HashMap::new();
    extra.insert("parser_provider".to_string(), parser_provider.to_string());

    let mut pages = page_blocks
        .into_iter()
        .enumerate()
        .map(|(index, blocks)| PageIr {
            page_no: (index + 1) as u32,
            width: None,
            height: None,
            blocks: blocks
                .into_iter()
                .filter_map(|block| {
                    let text = block.text.trim().to_string();
                    if text.is_empty() {
                        return None;
                    }

                    Some(TextBlock {
                        block_id: block_id_or_uuid(block.block_id),
                        page_no: block.page_no.max(1),
                        text,
                        bbox: block.bbox,
                        confidence: block.confidence,
                        source_kind: block_source_kind.clone(),
                    })
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    if pages.is_empty() {
        pages.push(PageIr {
            page_no: 1,
            width: None,
            height: None,
            blocks: vec![],
        });
    }

    let plain_text = pages
        .iter()
        .flat_map(|page| page.blocks.iter())
        .map(|block| block.text.as_str())
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    DocumentIr {
        doc_id: Uuid::new_v4().to_string(),
        source_type: input.source_type,
        pages,
        plain_text,
        metadata: DocumentMetadata {
            file_name: input.file_name,
            mime_type: input.mime_type,
            extra,
        },
    }
}

fn block_id_or_uuid(block_id: Option<String>) -> String {
    block_id.unwrap_or_else(|| Uuid::new_v4().to_string())
}

pub fn build_document_from_blocks(
    input: ParseInput,
    blocks: Vec<BlockInput>,
    block_source_kind: BlockSourceKind,
    parser_provider: &'static str,
) -> DocumentIr {
    let mut grouped = BTreeMap::<u32, Vec<BlockInput>>::new();

    for mut block in blocks {
        block.page_no = block.page_no.max(1);
        grouped.entry(block.page_no).or_default().push(block);
    }

    let page_blocks = if grouped.is_empty() {
        vec![Vec::new()]
    } else {
        grouped.into_values().collect::<Vec<_>>()
    };

    build_document_from_page_blocks(input, page_blocks, block_source_kind, parser_provider)
}

pub fn decode_text_payload(text: Option<String>, bytes: Option<Vec<u8>>) -> anyhow::Result<String> {
    if let Some(text) = text {
        return Ok(text);
    }

    if let Some(bytes) = bytes {
        let text = String::from_utf8(bytes).map_err(|_| {
            anyhow::anyhow!("text upload is not valid UTF-8 in the current bootstrap")
        })?;
        return Ok(text);
    }

    Ok(String::new())
}

pub fn placeholder_message(input: &ParseInput, summary: &str) -> String {
    let name = input
        .file_name
        .clone()
        .unwrap_or_else(|| "unnamed".to_string());
    let mime = input
        .mime_type
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let size = input
        .bytes
        .as_ref()
        .map(|bytes| bytes.len())
        .unwrap_or_default();

    format!("{summary}\nfile_name: {name}\nmime_type: {mime}\nbyte_size: {size}")
}

pub fn infer_block_source_kind(source_type: &SourceType) -> BlockSourceKind {
    match source_type {
        SourceType::Image | SourceType::Pdf | SourceType::Docx | SourceType::Unknown => {
            BlockSourceKind::Synthetic
        }
        _ => BlockSourceKind::NativeText,
    }
}
