use crate::domain::SourceType;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    Text,
    Image,
    Pdf,
    Docx,
    Html,
    Markdown,
    Url,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractionInput {
    pub kind: InputKind,
    pub content: Option<String>,
    pub url: Option<String>,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParseInput {
    pub source_type: SourceType,
    pub text: Option<String>,
    pub bytes: Option<Vec<u8>>,
    pub url: Option<String>,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
}

impl ParseInput {
    pub fn from_upload(file_name: String, mime_type: Option<String>, bytes: Vec<u8>) -> Self {
        let source_type = detect_source_type(Some(&file_name), mime_type.as_deref());

        Self {
            source_type,
            text: None,
            bytes: Some(bytes),
            url: None,
            file_name: Some(file_name),
            mime_type,
        }
    }
}

impl TryFrom<ExtractionInput> for ParseInput {
    type Error = anyhow::Error;

    fn try_from(value: ExtractionInput) -> Result<Self, Self::Error> {
        let source_type = match value.kind {
            InputKind::Text => SourceType::Text,
            InputKind::Image => SourceType::Image,
            InputKind::Pdf => SourceType::Pdf,
            InputKind::Docx => SourceType::Docx,
            InputKind::Html => SourceType::Html,
            InputKind::Markdown => SourceType::Markdown,
            InputKind::Url => SourceType::Url,
        };

        if matches!(
            source_type,
            SourceType::Text | SourceType::Html | SourceType::Markdown
        ) && value.content.as_deref().unwrap_or("").trim().is_empty()
        {
            anyhow::bail!("text-like input requires a non-empty `content` field");
        }

        if matches!(source_type, SourceType::Url)
            && value.url.as_deref().unwrap_or("").trim().is_empty()
        {
            anyhow::bail!("url input requires a non-empty `url` field");
        }

        Ok(Self {
            source_type,
            text: value.content,
            bytes: None,
            url: value.url,
            file_name: value.file_name,
            mime_type: value.mime_type,
        })
    }
}

pub fn detect_source_type(file_name: Option<&str>, mime_type: Option<&str>) -> SourceType {
    if let Some(mime) = mime_type {
        let mime = mime.to_ascii_lowercase();
        if mime.starts_with("image/") {
            return SourceType::Image;
        }
        if mime == "application/pdf" {
            return SourceType::Pdf;
        }
        if mime.contains("wordprocessingml") {
            return SourceType::Docx;
        }
        if mime == "text/html" {
            return SourceType::Html;
        }
        if mime == "text/markdown" || mime == "text/x-markdown" {
            return SourceType::Markdown;
        }
        if mime.starts_with("text/") {
            return SourceType::Text;
        }
    }

    if let Some(name) = file_name {
        let lowered = name.to_ascii_lowercase();
        if lowered.ends_with(".pdf") {
            return SourceType::Pdf;
        }
        if lowered.ends_with(".docx") {
            return SourceType::Docx;
        }
        if lowered.ends_with(".html") || lowered.ends_with(".htm") {
            return SourceType::Html;
        }
        if lowered.ends_with(".md") || lowered.ends_with(".markdown") {
            return SourceType::Markdown;
        }
        if lowered.ends_with(".txt") {
            return SourceType::Text;
        }
        if [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
            .iter()
            .any(|suffix| lowered.ends_with(suffix))
        {
            return SourceType::Image;
        }
    }

    SourceType::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_image_from_mime_type() {
        let source_type = detect_source_type(Some("note.bin"), Some("image/png"));
        assert!(matches!(source_type, SourceType::Image));
    }

    #[test]
    fn detects_pdf_from_file_extension() {
        let source_type = detect_source_type(Some("resume.PDF"), None);
        assert!(matches!(source_type, SourceType::Pdf));
    }

    #[test]
    fn build_parse_input_from_upload() {
        let parse_input = ParseInput::from_upload(
            "note.txt".to_string(),
            Some("text/plain".to_string()),
            b"hello".to_vec(),
        );

        assert!(matches!(parse_input.source_type, SourceType::Text));
        assert_eq!(parse_input.bytes.as_deref(), Some(&b"hello"[..]));
        assert_eq!(parse_input.file_name.as_deref(), Some("note.txt"));
    }
}
