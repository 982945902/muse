use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub const NORMALIZED_PROTOCOL_VERSION: &str = "1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Text,
    Image,
    Pdf,
    Docx,
    Html,
    Markdown,
    Url,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockSourceKind {
    NativeText,
    Ocr,
    Synthetic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextBlock {
    pub block_id: String,
    pub page_no: u32,
    pub text: String,
    pub bbox: Option<BBox>,
    pub confidence: Option<f32>,
    pub source_kind: BlockSourceKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PageIr {
    pub page_no: u32,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub blocks: Vec<TextBlock>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub extra: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentIr {
    pub doc_id: String,
    pub source_type: SourceType,
    pub pages: Vec<PageIr>,
    pub plain_text: String,
    pub metadata: DocumentMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalizedDocument {
    pub source_type: SourceType,
    pub pages: Vec<NormalizedPage>,
    pub plain_text: String,
    pub metadata: DocumentMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalizedPage {
    pub page_no: u32,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub blocks: Vec<NormalizedTextBlock>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormalizedTextBlock {
    pub page_no: u32,
    pub text: String,
    pub bbox: Option<BBox>,
    pub confidence: Option<f32>,
    pub source_kind: BlockSourceKind,
}

impl NormalizedDocument {
    pub fn validate(&self) -> anyhow::Result<()> {
        let protocol_version = self.metadata.extra.get("protocol_version");
        let sdk_version = self.metadata.extra.get("sdk_version");
        if protocol_version.is_none() && sdk_version.is_none() {
            anyhow::bail!(
                "normalized document metadata.extra must include `protocol_version` or `sdk_version`"
            );
        }

        if let Some(protocol_version) = protocol_version {
            if protocol_version != NORMALIZED_PROTOCOL_VERSION {
                anyhow::bail!(
                    "unsupported normalized document protocol_version `{protocol_version}`, expected `{NORMALIZED_PROTOCOL_VERSION}`"
                );
            }
        }

        if self.plain_text.trim().is_empty() && self.pages.iter().all(|page| page.blocks.is_empty())
        {
            anyhow::bail!("normalized document must contain plain_text or at least one text block");
        }

        let mut seen_pages = HashSet::new();
        for (page_index, page) in self.pages.iter().enumerate() {
            if page.page_no == 0 {
                anyhow::bail!("normalized page at index {page_index} must use page_no >= 1");
            }

            if !seen_pages.insert(page.page_no) {
                anyhow::bail!(
                    "normalized document contains duplicated page_no `{}`",
                    page.page_no
                );
            }

            for (block_index, block) in page.blocks.iter().enumerate() {
                if block.text.trim().is_empty() {
                    anyhow::bail!(
                        "normalized block at page {} index {} must not have empty text",
                        page.page_no,
                        block_index
                    );
                }

                if block.page_no != page.page_no {
                    anyhow::bail!(
                        "normalized block at page {} index {} must use matching page_no",
                        page.page_no,
                        block_index
                    );
                }
            }
        }

        Ok(())
    }

    pub fn into_document_ir(self) -> DocumentIr {
        let mut metadata = self.metadata;
        metadata
            .extra
            .entry("ingest_mode".to_string())
            .or_insert_with(|| "normalized".to_string());
        metadata
            .extra
            .entry("protocol_version".to_string())
            .or_insert_with(|| NORMALIZED_PROTOCOL_VERSION.to_string());

        let pages = self
            .pages
            .into_iter()
            .map(|page| PageIr {
                page_no: page.page_no,
                width: page.width,
                height: page.height,
                blocks: page
                    .blocks
                    .into_iter()
                    .map(|block| TextBlock {
                        block_id: Uuid::new_v4().to_string(),
                        page_no: block.page_no,
                        text: block.text,
                        bbox: block.bbox,
                        confidence: block.confidence,
                        source_kind: block.source_kind,
                    })
                    .collect(),
            })
            .collect();

        DocumentIr {
            doc_id: Uuid::new_v4().to_string(),
            source_type: self.source_type,
            pages,
            plain_text: self.plain_text,
            metadata,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    String,
    Number,
    Boolean,
    Object,
    Array,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldSpec {
    pub key: String,
    pub field_type: FieldType,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub multiple: bool,
    #[serde(default)]
    pub children: Vec<FieldSpec>,
    #[serde(default)]
    pub hints: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaSpec {
    pub name: String,
    pub version: String,
    pub fields: Vec<FieldSpec>,
}

impl SchemaSpec {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.name.trim().is_empty() {
            anyhow::bail!("schema name must not be empty");
        }

        if self.version.trim().is_empty() {
            anyhow::bail!("schema version must not be empty");
        }

        if self.fields.is_empty() {
            anyhow::bail!("schema must define at least one field");
        }

        validate_field_list(&self.fields, "schema")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub page_no: Option<u32>,
    pub text: String,
    pub bbox: Option<BBox>,
    pub source_block_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldValue {
    pub key: String,
    pub value: Value,
    pub confidence: Option<f32>,
    pub evidences: Vec<Evidence>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub parse_ms: u64,
    pub extract_ms: u64,
    pub postprocess_ms: u64,
    pub total_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Created,
    Queued,
    Parsing,
    Extracting,
    Postprocessing,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub fields: Vec<FieldValue>,
    pub raw_text: Option<String>,
    pub timings: TimingBreakdown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskRecord {
    pub task_id: String,
    pub status: TaskStatus,
    pub result: Option<ExtractionResult>,
    pub message: Option<String>,
}

fn validate_field_list(fields: &[FieldSpec], scope: &str) -> anyhow::Result<()> {
    let mut seen = HashSet::new();

    for field in fields {
        let key = field.key.trim();
        if key.is_empty() {
            anyhow::bail!("{scope} contains a field with an empty key");
        }

        if !seen.insert(key.to_string()) {
            anyhow::bail!("{scope} contains duplicated field key `{key}`");
        }

        if !field.children.is_empty() {
            match field.field_type {
                FieldType::Object | FieldType::Array => {}
                _ => anyhow::bail!(
                    "field `{}` has children but field_type is not `object` or `array`",
                    field.key
                ),
            }

            let nested_scope = format!("field `{}`", field.key);
            validate_field_list(&field.children, &nested_scope)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_leaf(key: &str) -> FieldSpec {
        FieldSpec {
            key: key.to_string(),
            field_type: FieldType::String,
            required: false,
            multiple: false,
            children: vec![],
            hints: vec![],
        }
    }

    #[test]
    fn validates_schema_successfully() {
        let schema = SchemaSpec {
            name: "resume".to_string(),
            version: "1".to_string(),
            fields: vec![
                sample_leaf("岗位类型"),
                FieldSpec {
                    key: "测评维度".to_string(),
                    field_type: FieldType::Object,
                    required: false,
                    multiple: true,
                    children: vec![sample_leaf("答题策略")],
                    hints: vec![],
                },
            ],
        };

        assert!(schema.validate().is_ok());
    }

    #[test]
    fn rejects_empty_schema_fields() {
        let schema = SchemaSpec {
            name: "resume".to_string(),
            version: "1".to_string(),
            fields: vec![],
        };

        assert!(schema.validate().is_err());
    }

    #[test]
    fn rejects_duplicate_keys_in_same_scope() {
        let schema = SchemaSpec {
            name: "resume".to_string(),
            version: "1".to_string(),
            fields: vec![sample_leaf("岗位类型"), sample_leaf("岗位类型")],
        };

        assert!(schema.validate().is_err());
    }

    #[test]
    fn rejects_children_on_scalar_field() {
        let schema = SchemaSpec {
            name: "resume".to_string(),
            version: "1".to_string(),
            fields: vec![FieldSpec {
                key: "岗位类型".to_string(),
                field_type: FieldType::String,
                required: false,
                multiple: false,
                children: vec![sample_leaf("子字段")],
                hints: vec![],
            }],
        };

        assert!(schema.validate().is_err());
    }

    #[test]
    fn validates_normalized_document_successfully() {
        let mut metadata = DocumentMetadata::default();
        metadata.extra.insert(
            "protocol_version".to_string(),
            NORMALIZED_PROTOCOL_VERSION.to_string(),
        );
        let document = NormalizedDocument {
            source_type: SourceType::Image,
            plain_text: "岗位类型：SDK 预处理".to_string(),
            metadata,
            pages: vec![NormalizedPage {
                page_no: 1,
                width: None,
                height: None,
                blocks: vec![NormalizedTextBlock {
                    page_no: 1,
                    text: "岗位类型：SDK 预处理".to_string(),
                    bbox: None,
                    confidence: Some(0.9),
                    source_kind: BlockSourceKind::Ocr,
                }],
            }],
        };

        assert!(document.validate().is_ok());
        let ir = document.into_document_ir();
        assert_eq!(ir.pages.len(), 1);
        assert_eq!(ir.pages[0].blocks.len(), 1);
        assert_eq!(
            ir.metadata.extra.get("ingest_mode").map(String::as_str),
            Some("normalized")
        );
    }

    #[test]
    fn rejects_normalized_document_without_protocol_or_sdk_version() {
        let document = NormalizedDocument {
            source_type: SourceType::Image,
            plain_text: "岗位类型：SDK 预处理".to_string(),
            metadata: DocumentMetadata::default(),
            pages: vec![],
        };

        assert!(document.validate().is_err());
    }

    #[test]
    fn rejects_normalized_document_when_block_page_mismatches() {
        let mut metadata = DocumentMetadata::default();
        metadata
            .extra
            .insert("sdk_version".to_string(), "0.1.0".to_string());
        let document = NormalizedDocument {
            source_type: SourceType::Image,
            plain_text: "岗位类型：SDK 预处理".to_string(),
            metadata,
            pages: vec![NormalizedPage {
                page_no: 1,
                width: None,
                height: None,
                blocks: vec![NormalizedTextBlock {
                    page_no: 2,
                    text: "岗位类型：SDK 预处理".to_string(),
                    bbox: None,
                    confidence: Some(0.9),
                    source_kind: BlockSourceKind::Ocr,
                }],
            }],
        };

        assert!(document.validate().is_err());
    }
}
