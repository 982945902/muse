use async_trait::async_trait;
use lopdf::Document;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdfRequest {
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdfOutput {
    pub text: String,
    pub page_count: Option<u32>,
    pub extracted_text_layer: bool,
    pub page_texts: Vec<String>,
}

#[async_trait]
pub trait PdfProvider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn extract(&self, request: PdfRequest) -> anyhow::Result<PdfOutput>;
}

#[derive(Default)]
pub struct LopdfTextLayerProvider;

#[async_trait]
impl PdfProvider for LopdfTextLayerProvider {
    fn name(&self) -> &'static str {
        "lopdf-text-layer"
    }

    async fn extract(&self, request: PdfRequest) -> anyhow::Result<PdfOutput> {
        let document = Document::load_mem(&request.bytes)
            .map_err(|error| anyhow::anyhow!("failed to open pdf: {error}"))?;

        let pages = document.get_pages();
        let page_numbers = pages.keys().copied().collect::<Vec<_>>();
        let page_count = page_numbers.len() as u32;

        let page_texts = page_numbers
            .iter()
            .map(|page_no| {
                document
                    .extract_text(&[*page_no])
                    .map(|raw| normalize_text(&raw))
                    .map_err(|error| anyhow::anyhow!("failed to extract pdf text layer: {error}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let text = page_texts
            .iter()
            .map(String::as_str)
            .filter(|page| !page.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(PdfOutput {
            extracted_text_layer: !text.trim().is_empty(),
            text,
            page_count: Some(page_count),
            page_texts,
        })
    }
}

fn normalize_text(raw_text: &str) -> String {
    raw_text
        .replace('\r', "\n")
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn extracts_text_from_generated_pdf() {
        let provider = LopdfTextLayerProvider;
        let bytes = build_single_page_pdf("role_type: pdf text\npersona_hint: no ocr");

        let output = provider
            .extract(PdfRequest {
                file_name: Some("sample.pdf".to_string()),
                mime_type: Some("application/pdf".to_string()),
                bytes,
            })
            .await
            .expect("pdf extraction should succeed");

        assert_eq!(output.page_count, Some(1));
        assert!(output.extracted_text_layer);
        assert!(output.text.contains("role_type: pdf text"));
        assert!(output.text.contains("persona_hint: no ocr"));
        assert_eq!(output.page_texts.len(), 1);
        assert!(output.page_texts[0].contains("role_type: pdf text"));
    }

    fn build_single_page_pdf(text: &str) -> Vec<u8> {
        use lopdf::{
            Object, Stream,
            content::{Content, Operation},
            dictionary,
        };

        let mut document = Document::with_version("1.5");

        let pages_id = document.new_object_id();
        let page_id = document.new_object_id();
        let font_id = document.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });

        let resources_id = document.add_object(dictionary! {
            "Font" => dictionary! {
                "F1" => font_id,
            }
        });

        let mut operations = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
        ];

        for (index, line) in text.lines().enumerate() {
            let y = 720_i64 - (index as i64 * 18);
            operations.push(Operation::new("Td", vec![72.into(), y.into()]));
            operations.push(Operation::new("Tj", vec![Object::string_literal(line)]));
        }
        operations.push(Operation::new("ET", vec![]));

        let content = Content { operations };
        let content_id = document.add_object(Stream::new(
            dictionary! {},
            content.encode().expect("encode content"),
        ));

        let page = dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        };
        document.objects.insert(page_id, Object::Dictionary(page));

        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        };
        document.objects.insert(pages_id, Object::Dictionary(pages));

        let catalog_id = document.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        document.trailer.set("Root", catalog_id);

        document.compress();

        let mut buffer = Vec::new();
        document.save_to(&mut buffer).expect("save pdf");
        buffer
    }
}
