use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Read};
use zip::ZipArchive;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocxRequest {
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocxOutput {
    pub text: String,
    pub paragraph_count: Option<u32>,
    pub paragraphs: Vec<String>,
}

#[async_trait]
pub trait DocxProvider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn extract(&self, request: DocxRequest) -> anyhow::Result<DocxOutput>;
}

#[derive(Default)]
pub struct ZipDocxProvider;

#[async_trait]
impl DocxProvider for ZipDocxProvider {
    fn name(&self) -> &'static str {
        "zip-docx"
    }

    async fn extract(&self, request: DocxRequest) -> anyhow::Result<DocxOutput> {
        let mut archive = ZipArchive::new(Cursor::new(request.bytes))
            .map_err(|error| anyhow::anyhow!("failed to open docx archive: {error}"))?;

        let document_xml = read_zip_entry(&mut archive, "word/document.xml")?;
        let paragraph_count = count_paragraphs(&document_xml);
        let paragraphs = extract_docx_paragraphs(&document_xml)?;
        let text = paragraphs.join("\n");

        Ok(DocxOutput {
            text,
            paragraph_count: Some(paragraph_count),
            paragraphs,
        })
    }
}

fn read_zip_entry<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    entry_name: &str,
) -> anyhow::Result<String> {
    let mut entry = archive
        .by_name(entry_name)
        .map_err(|error| anyhow::anyhow!("missing `{entry_name}` in docx archive: {error}"))?;
    let mut xml = String::new();
    entry
        .read_to_string(&mut xml)
        .map_err(|error| anyhow::anyhow!("failed to read `{entry_name}` as UTF-8: {error}"))?;
    Ok(xml)
}

fn count_paragraphs(document_xml: &str) -> u32 {
    document_xml.matches("</w:p>").count() as u32
}

fn extract_docx_paragraphs(document_xml: &str) -> anyhow::Result<Vec<String>> {
    let tab_regex = Regex::new(r"<w:tab\s*/>")?;
    let break_regex = Regex::new(r"<w:br[^>]*/>|<w:cr[^>]*/>")?;
    let text_regex = Regex::new(r"(?s)<w:t[^>]*>(.*?)</w:t>")?;

    let normalized = tab_regex.replace_all(document_xml, "\t");
    let normalized = break_regex.replace_all(&normalized, "\n");

    let mut paragraphs = Vec::new();
    for paragraph_xml in normalized.split("</w:p>") {
        let parts = text_regex
            .captures_iter(paragraph_xml)
            .filter_map(|caps| caps.get(1).map(|m| decode_xml_entities(m.as_str())))
            .collect::<Vec<_>>();

        if !parts.is_empty() {
            paragraphs.push(parts.join(""));
        }
    }

    Ok(paragraphs)
}

fn decode_xml_entities(value: &str) -> String {
    value
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::{ZipWriter, write::SimpleFileOptions};

    #[tokio::test]
    async fn extracts_text_from_minimal_docx() {
        let provider = ZipDocxProvider;
        let bytes = make_minimal_docx(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>岗位类型：文档运营</w:t></w:r></w:p>
    <w:p><w:r><w:t>人设要点：结构化表达</w:t></w:r></w:p>
  </w:body>
</w:document>"#,
        );

        let output = provider
            .extract(DocxRequest {
                file_name: Some("resume.docx".to_string()),
                mime_type: Some(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        .to_string(),
                ),
                bytes,
            })
            .await
            .expect("docx extraction should succeed");

        assert_eq!(output.paragraph_count, Some(2));
        assert_eq!(output.text, "岗位类型：文档运营\n人设要点：结构化表达");
        assert_eq!(
            output.paragraphs,
            vec!["岗位类型：文档运营", "人设要点：结构化表达"]
        );
    }

    #[tokio::test]
    async fn decodes_basic_xml_entities() {
        let provider = ZipDocxProvider;
        let bytes = make_minimal_docx(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Tom &amp; Jerry</w:t></w:r></w:p>
    <w:p><w:r><w:t>5 &lt; 10</w:t></w:r></w:p>
  </w:body>
</w:document>"#,
        );

        let output = provider
            .extract(DocxRequest {
                file_name: Some("entities.docx".to_string()),
                mime_type: None,
                bytes,
            })
            .await
            .expect("docx extraction should succeed");

        assert_eq!(output.text, "Tom & Jerry\n5 < 10");
        assert_eq!(output.paragraphs, vec!["Tom & Jerry", "5 < 10"]);
    }

    fn make_minimal_docx(document_xml: &str) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = ZipWriter::new(&mut cursor);
            let options = SimpleFileOptions::default();

            writer
                .start_file("[Content_Types].xml", options)
                .expect("start content types");
            writer
                .write_all(
                    br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#,
                )
                .expect("write content types");

            writer
                .start_file("_rels/.rels", options)
                .expect("start rels");
            writer
                .write_all(
                    br#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#,
                )
                .expect("write rels");

            writer
                .start_file("word/document.xml", options)
                .expect("start document xml");
            writer
                .write_all(document_xml.as_bytes())
                .expect("write document xml");

            writer.finish().expect("finish zip");
        }

        cursor.into_inner()
    }
}
