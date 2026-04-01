mod binary;
mod docx;
mod image;
mod pdf;
mod shared;
mod text;
mod url;

use crate::{domain::DocumentIr, ingestion::ParseInput};
use async_trait::async_trait;
use std::sync::Arc;

pub use binary::PlaceholderBinaryParser;
pub use docx::DocxParser;
pub use image::ImageParser;
pub use pdf::PdfParser;
pub use text::TextParser;
pub use url::UrlParser;

#[async_trait]
pub trait Parser: Send + Sync {
    fn name(&self) -> &'static str;
    async fn parse(&self, input: ParseInput) -> anyhow::Result<DocumentIr>;
}

pub struct DefaultParser {
    text: TextParser,
    url: UrlParser,
    image: ImageParser,
    pdf: PdfParser,
    docx: DocxParser,
    binary: PlaceholderBinaryParser,
}

impl DefaultParser {
    pub fn new(
        ocr: Arc<dyn crate::ocr::OcrProvider>,
        pdf: Arc<dyn crate::pdf::PdfProvider>,
        docx: Arc<dyn crate::docx::DocxProvider>,
    ) -> Self {
        Self {
            text: TextParser,
            url: UrlParser,
            image: ImageParser::new(ocr.clone()),
            pdf: PdfParser::new(pdf, ocr),
            docx: DocxParser::new(docx),
            binary: PlaceholderBinaryParser,
        }
    }
}

impl Default for DefaultParser {
    fn default() -> Self {
        Self::new(
            Arc::new(crate::ocr::PlaceholderOcrProvider),
            Arc::new(crate::pdf::LopdfTextLayerProvider),
            Arc::new(crate::docx::ZipDocxProvider),
        )
    }
}

#[async_trait]
impl Parser for DefaultParser {
    fn name(&self) -> &'static str {
        "delegating-parser"
    }

    async fn parse(&self, input: ParseInput) -> anyhow::Result<DocumentIr> {
        self.route(&input).parse(input).await
    }
}

impl DefaultParser {
    fn route(&self, input: &ParseInput) -> &dyn Parser {
        use crate::domain::SourceType;

        match input.source_type {
            SourceType::Text | SourceType::Html | SourceType::Markdown => &self.text,
            SourceType::Url => &self.url,
            SourceType::Image => &self.image,
            SourceType::Pdf => &self.pdf,
            SourceType::Docx => &self.docx,
            SourceType::Unknown => &self.binary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docx::{DocxOutput, DocxProvider, DocxRequest};
    use crate::domain::{BBox, SourceType};
    use crate::ocr::{OcrLine, OcrOutput, OcrProvider, OcrRequest};
    use crate::pdf::{PdfOutput, PdfProvider, PdfRequest};
    use async_trait::async_trait;

    struct TestOcrProvider;
    struct TestPdfProvider;
    struct TestScanPdfProvider;
    struct TestDocxProvider;

    #[async_trait]
    impl OcrProvider for TestOcrProvider {
        fn name(&self) -> &'static str {
            "test-ocr"
        }

        async fn recognize(&self, _request: OcrRequest) -> anyhow::Result<OcrOutput> {
            Ok(OcrOutput {
                lines: vec![
                    OcrLine {
                        text: "岗位类型：图片运营".to_string(),
                        page_no: Some(1),
                        bbox: Some(BBox {
                            x1: 10.0,
                            y1: 20.0,
                            x2: 120.0,
                            y2: 48.0,
                        }),
                        confidence: Some(0.9),
                    },
                    OcrLine {
                        text: "人设要点：审美好".to_string(),
                        page_no: Some(1),
                        bbox: Some(BBox {
                            x1: 12.0,
                            y1: 60.0,
                            x2: 150.0,
                            y2: 90.0,
                        }),
                        confidence: Some(0.8),
                    },
                ],
                provider: None,
                model: Some("test-ocr-model".to_string()),
            })
        }
    }

    #[async_trait]
    impl PdfProvider for TestPdfProvider {
        fn name(&self) -> &'static str {
            "test-pdf"
        }

        async fn extract(&self, _request: PdfRequest) -> anyhow::Result<PdfOutput> {
            Ok(PdfOutput {
                text: "岗位类型：PDF 运营\n人设要点：信息密度高".to_string(),
                page_count: Some(2),
                extracted_text_layer: true,
                page_texts: vec![
                    "岗位类型：PDF 运营".to_string(),
                    "人设要点：信息密度高".to_string(),
                ],
            })
        }
    }

    #[async_trait]
    impl PdfProvider for TestScanPdfProvider {
        fn name(&self) -> &'static str {
            "test-scan-pdf"
        }

        async fn extract(&self, _request: PdfRequest) -> anyhow::Result<PdfOutput> {
            Ok(PdfOutput {
                text: String::new(),
                page_count: Some(3),
                extracted_text_layer: false,
                page_texts: vec![],
            })
        }
    }

    #[async_trait]
    impl DocxProvider for TestDocxProvider {
        fn name(&self) -> &'static str {
            "test-docx"
        }

        async fn extract(&self, _request: DocxRequest) -> anyhow::Result<DocxOutput> {
            Ok(DocxOutput {
                text: "岗位类型：文档运营\n人设要点：结构化表达".to_string(),
                paragraph_count: Some(4),
                paragraphs: vec![
                    "岗位类型：文档运营".to_string(),
                    "人设要点：结构化表达".to_string(),
                ],
            })
        }
    }

    #[tokio::test]
    async fn routes_uploaded_text_to_text_parser() {
        let parser = DefaultParser::default();
        let input = ParseInput::from_upload(
            "note.txt".to_string(),
            Some("text/plain".to_string()),
            "岗位类型：测试\n人设要点：认真".as_bytes().to_vec(),
        );

        let document = parser.parse(input).await.expect("parse should succeed");

        assert!(matches!(document.source_type, SourceType::Text));
        assert_eq!(document.plain_text, "岗位类型：测试\n人设要点：认真");
        assert_eq!(
            document
                .metadata
                .extra
                .get("parser_provider")
                .map(String::as_str),
            Some("text-parser")
        );
    }

    #[tokio::test]
    async fn routes_url_to_url_parser() {
        let parser = DefaultParser::default();
        let input = ParseInput {
            source_type: SourceType::Url,
            text: None,
            bytes: None,
            url: Some("https://example.com/post/123".to_string()),
            file_name: None,
            mime_type: None,
        };

        let document = parser.parse(input).await.expect("parse should succeed");

        assert_eq!(
            document
                .metadata
                .extra
                .get("parser_provider")
                .map(String::as_str),
            Some("url-parser")
        );
        assert!(document.plain_text.contains("https://example.com/post/123"));
    }

    #[tokio::test]
    async fn routes_pdf_to_pdf_parser() {
        let parser = DefaultParser::new(
            Arc::new(TestOcrProvider),
            Arc::new(TestPdfProvider),
            Arc::new(TestDocxProvider),
        );
        let input = ParseInput::from_upload(
            "resume.pdf".to_string(),
            Some("application/pdf".to_string()),
            b"%PDF-1.7".to_vec(),
        );

        let document = parser.parse(input).await.expect("parse should succeed");

        assert!(matches!(document.source_type, SourceType::Pdf));
        assert_eq!(
            document
                .metadata
                .extra
                .get("parser_provider")
                .map(String::as_str),
            Some("pdf-parser")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_provider")
                .map(String::as_str),
            Some("test-pdf")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("page_count")
                .map(String::as_str),
            Some("2")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("extracted_text_layer")
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_parse_mode")
                .map(String::as_str),
            Some("text_layer")
        );
        assert_eq!(document.pages.len(), 2);
        assert_eq!(document.pages[0].blocks.len(), 1);
        assert_eq!(document.pages[0].blocks[0].text, "岗位类型：PDF 运营");
        assert_eq!(document.pages[1].blocks[0].text, "人设要点：信息密度高");
        assert!(document.plain_text.contains("岗位类型：PDF 运营"));
    }

    #[tokio::test]
    async fn routes_image_to_image_parser_with_ocr_provider() {
        let parser = DefaultParser::new(
            Arc::new(TestOcrProvider),
            Arc::new(TestPdfProvider),
            Arc::new(TestDocxProvider),
        );
        let input = ParseInput::from_upload(
            "poster.png".to_string(),
            Some("image/png".to_string()),
            vec![1, 2, 3, 4],
        );

        let document = parser.parse(input).await.expect("parse should succeed");

        assert!(matches!(document.source_type, SourceType::Image));
        assert_eq!(
            document
                .metadata
                .extra
                .get("parser_provider")
                .map(String::as_str),
            Some("image-parser")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("ocr_provider")
                .map(String::as_str),
            Some("test-ocr")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("ocr_transport")
                .map(String::as_str),
            Some("test-ocr")
        );
        assert_eq!(
            document.metadata.extra.get("ocr_model").map(String::as_str),
            Some("test-ocr-model")
        );
        assert_eq!(document.pages.len(), 1);
        assert_eq!(document.pages[0].blocks.len(), 2);
        assert_eq!(document.pages[0].blocks[0].text, "岗位类型：图片运营");
        assert_eq!(document.pages[0].blocks[1].text, "人设要点：审美好");
        assert!(document.pages[0].blocks[0].bbox.is_some());
        assert!(document.plain_text.contains("岗位类型：图片运营"));
        assert!(document.plain_text.contains("人设要点：审美好"));
    }

    #[tokio::test]
    async fn routes_scanned_pdf_to_ocr_fallback() {
        let parser = DefaultParser::new(
            Arc::new(TestOcrProvider),
            Arc::new(TestScanPdfProvider),
            Arc::new(TestDocxProvider),
        );
        let input = ParseInput::from_upload(
            "scan.pdf".to_string(),
            Some("application/pdf".to_string()),
            b"%PDF-1.7 scanned".to_vec(),
        );

        let document = parser.parse(input).await.expect("parse should succeed");

        assert!(matches!(document.source_type, SourceType::Pdf));
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_provider")
                .map(String::as_str),
            Some("test-scan-pdf")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("extracted_text_layer")
                .map(String::as_str),
            Some("false")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_parse_mode")
                .map(String::as_str),
            Some("ocr_fallback")
        );
        assert_eq!(document.pages.len(), 1);
        assert_eq!(document.pages[0].blocks.len(), 2);
        assert_eq!(
            document
                .metadata
                .extra
                .get("ocr_provider")
                .map(String::as_str),
            Some("test-ocr")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("ocr_transport")
                .map(String::as_str),
            Some("test-ocr")
        );
        assert!(matches!(
            document.pages[0].blocks[0].source_kind,
            crate::domain::BlockSourceKind::Ocr
        ));
        assert!(document.plain_text.contains("岗位类型：图片运营"));
    }

    #[tokio::test]
    async fn routes_docx_to_docx_parser() {
        let parser = DefaultParser::new(
            Arc::new(TestOcrProvider),
            Arc::new(TestPdfProvider),
            Arc::new(TestDocxProvider),
        );
        let input = ParseInput::from_upload(
            "resume.docx".to_string(),
            Some(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    .to_string(),
            ),
            vec![0, 1, 2, 3],
        );

        let document = parser.parse(input).await.expect("parse should succeed");

        assert!(matches!(document.source_type, SourceType::Docx));
        assert_eq!(document.pages.len(), 1);
        assert_eq!(document.pages[0].blocks.len(), 2);
        assert_eq!(document.pages[0].blocks[0].text, "岗位类型：文档运营");
        assert_eq!(
            document
                .metadata
                .extra
                .get("parser_provider")
                .map(String::as_str),
            Some("docx-parser")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("docx_provider")
                .map(String::as_str),
            Some("test-docx")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("paragraph_count")
                .map(String::as_str),
            Some("4")
        );
        assert!(document.plain_text.contains("岗位类型：文档运营"));
    }
}
