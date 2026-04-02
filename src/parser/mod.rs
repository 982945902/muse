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
            Arc::new(crate::ocr::PlaceholderOcrProvider::default()),
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
    use crate::ocr::{
        LocalOnnxOcrProvider, OcrLine, OcrOutput, OcrProvider, OcrRequest, TestLocalOcrCandidate,
        TestLocalOcrFixture, TestLocalOcrPrediction,
    };
    use crate::pdf::{PdfOcrPage, PdfOutput, PdfProvider, PdfRequest};
    use ::image::{ColorType, GenericImageView, ImageEncoder, codecs::png::PngEncoder};
    use async_trait::async_trait;

    struct TestOcrProvider;
    struct RasterAwareOcrProvider;
    struct TestPdfProvider;
    struct TestScanPdfProvider;
    struct TestRasterScanPdfProvider;
    struct InvalidRasterScanPdfProvider;
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
    impl OcrProvider for RasterAwareOcrProvider {
        fn name(&self) -> &'static str {
            "raster-aware-ocr"
        }

        async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
            assert_eq!(request.mime_type.as_deref(), Some("image/png"));
            assert!(request.bytes.starts_with(&[0x89, b'P', b'N', b'G']));

            let image = ::image::load_from_memory(&request.bytes).expect("decode png");
            let pixel = image.get_pixel(0, 0).0;
            let text = if pixel[0] > pixel[2] {
                "第一页结果"
            } else {
                "第二页结果"
            };

            Ok(OcrOutput {
                lines: vec![OcrLine {
                    text: text.to_string(),
                    page_no: Some(1),
                    bbox: Some(BBox {
                        x1: 4.0,
                        y1: 8.0,
                        x2: 64.0,
                        y2: 28.0,
                    }),
                    confidence: Some(0.93),
                }],
                provider: Some("raster-aware-ocr".to_string()),
                model: Some("png-page-model".to_string()),
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
                raster_provider: None,
                raster_pages: vec![],
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
                raster_provider: Some("test-inline-rasterizer".to_string()),
                raster_pages: vec![PdfOcrPage {
                    page_no: 1,
                    mime_type: Some("image/png".to_string()),
                    bytes: encode_test_png(2, 1, &[255, 255, 255, 0, 0, 0]),
                }],
            })
        }
    }

    #[async_trait]
    impl PdfProvider for TestRasterScanPdfProvider {
        fn name(&self) -> &'static str {
            "test-raster-scan-pdf"
        }

        async fn extract(&self, _request: PdfRequest) -> anyhow::Result<PdfOutput> {
            Ok(PdfOutput {
                text: String::new(),
                page_count: Some(2),
                extracted_text_layer: false,
                page_texts: vec![],
                raster_provider: Some("test-inline-rasterizer".to_string()),
                raster_pages: vec![
                    PdfOcrPage {
                        page_no: 1,
                        mime_type: Some("image/png".to_string()),
                        bytes: encode_test_png(1, 1, &[255, 0, 0]),
                    },
                    PdfOcrPage {
                        page_no: 2,
                        mime_type: Some("image/png".to_string()),
                        bytes: encode_test_png(1, 1, &[0, 0, 255]),
                    },
                ],
            })
        }
    }

    #[async_trait]
    impl PdfProvider for InvalidRasterScanPdfProvider {
        fn name(&self) -> &'static str {
            "invalid-raster-scan-pdf"
        }

        async fn extract(&self, _request: PdfRequest) -> anyhow::Result<PdfOutput> {
            Ok(PdfOutput {
                text: String::new(),
                page_count: Some(2),
                extracted_text_layer: false,
                page_texts: vec![],
                raster_provider: Some("test-inline-rasterizer".to_string()),
                raster_pages: vec![
                    PdfOcrPage {
                        page_no: 1,
                        mime_type: Some("image/png".to_string()),
                        bytes: encode_test_png(1, 1, &[255, 0, 0]),
                    },
                    PdfOcrPage {
                        page_no: 1,
                        mime_type: Some("image/png".to_string()),
                        bytes: encode_test_png(1, 1, &[0, 0, 255]),
                    },
                ],
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
    async fn routes_image_through_local_onnx_provider_fixture() {
        let parser = DefaultParser::new(
            Arc::new(LocalOnnxOcrProvider::from_test_fixture(
                TestLocalOcrFixture {
                    model_summary: "det=/tmp/det.onnx, rec=/tmp/rec.onnx, charset=/tmp/keys.txt"
                        .to_string(),
                    charset: vec!["岗".to_string(), "位".to_string(), "类".to_string()],
                    candidates: vec![
                        TestLocalOcrCandidate {
                            bbox: BBox {
                                x1: 8.0,
                                y1: 18.0,
                                x2: 108.0,
                                y2: 52.0,
                            },
                            score: 0.91,
                        },
                        TestLocalOcrCandidate {
                            bbox: BBox {
                                x1: 12.0,
                                y1: 68.0,
                                x2: 160.0,
                                y2: 104.0,
                            },
                            score: 0.84,
                        },
                    ],
                    predictions: vec![
                        TestLocalOcrPrediction {
                            patch_index: 0,
                            text: "岗位类型".to_string(),
                            confidence: 0.88,
                        },
                        TestLocalOcrPrediction {
                            patch_index: 1,
                            text: "人设要点".to_string(),
                            confidence: 0.80,
                        },
                    ],
                },
            )),
            Arc::new(TestPdfProvider),
            Arc::new(TestDocxProvider),
        );
        let input = ParseInput::from_upload(
            "poster-local.png".to_string(),
            Some("image/png".to_string()),
            encode_test_png(
                4,
                2,
                &[
                    255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255,
                    255, 255, 0, 0, 0,
                ],
            ),
        );

        let document = parser.parse(input).await.expect("parse should succeed");

        assert!(matches!(document.source_type, SourceType::Image));
        assert_eq!(
            document
                .metadata
                .extra
                .get("ocr_provider")
                .map(String::as_str),
            Some("local-onnx-ocr")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("ocr_transport")
                .map(String::as_str),
            Some("inproc")
        );
        assert_eq!(
            document.metadata.extra.get("ocr_model").map(String::as_str),
            Some("det=/tmp/det.onnx, rec=/tmp/rec.onnx, charset=/tmp/keys.txt")
        );
        assert_eq!(document.pages.len(), 1);
        assert_eq!(document.pages[0].blocks.len(), 2);
        assert_eq!(document.pages[0].blocks[0].text, "岗位类型");
        assert_eq!(document.pages[0].blocks[1].text, "人设要点");
        assert_eq!(document.pages[0].blocks[0].confidence, Some(0.88));
        assert_eq!(document.pages[0].blocks[1].confidence, Some(0.80));
        assert!(document.pages[0].blocks[0].bbox.is_some());
        assert!(document.plain_text.contains("岗位类型"));
        assert!(document.plain_text.contains("人设要点"));
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
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_ocr_input")
                .map(String::as_str),
            Some("page_rasters")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_raster_provider")
                .map(String::as_str),
            Some("test-inline-rasterizer")
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
    async fn routes_scanned_pdf_raster_pages_through_per_page_ocr() {
        let parser = DefaultParser::new(
            Arc::new(RasterAwareOcrProvider),
            Arc::new(TestRasterScanPdfProvider),
            Arc::new(TestDocxProvider),
        );
        let input = ParseInput::from_upload(
            "scan-pages.pdf".to_string(),
            Some("application/pdf".to_string()),
            b"%PDF-1.7 scan pages".to_vec(),
        );

        let document = parser.parse(input).await.expect("parse should succeed");

        assert!(matches!(document.source_type, SourceType::Pdf));
        assert_eq!(document.pages.len(), 2);
        assert_eq!(document.pages[0].page_no, 1);
        assert_eq!(document.pages[1].page_no, 2);
        assert_eq!(document.pages[0].blocks.len(), 1);
        assert_eq!(document.pages[1].blocks.len(), 1);
        assert_eq!(document.pages[0].blocks[0].text, "第一页结果");
        assert_eq!(document.pages[1].blocks[0].text, "第二页结果");
        assert_eq!(document.pages[0].blocks[0].page_no, 1);
        assert_eq!(document.pages[1].blocks[0].page_no, 2);
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_ocr_input")
                .map(String::as_str),
            Some("page_rasters")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("pdf_raster_provider")
                .map(String::as_str),
            Some("test-inline-rasterizer")
        );
        assert_eq!(
            document
                .metadata
                .extra
                .get("ocr_provider")
                .map(String::as_str),
            Some("raster-aware-ocr")
        );
        assert!(document.plain_text.contains("第一页结果"));
        assert!(document.plain_text.contains("第二页结果"));
    }

    #[tokio::test]
    async fn rejects_scanned_pdf_with_duplicate_raster_page_numbers() {
        let parser = DefaultParser::new(
            Arc::new(RasterAwareOcrProvider),
            Arc::new(InvalidRasterScanPdfProvider),
            Arc::new(TestDocxProvider),
        );
        let input = ParseInput::from_upload(
            "scan-invalid.pdf".to_string(),
            Some("application/pdf".to_string()),
            b"%PDF-1.7 invalid scan pages".to_vec(),
        );

        let error = parser.parse(input).await.expect_err("parse should fail");
        assert!(error.to_string().contains("duplicated page_no"));
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

    fn encode_test_png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes)
            .write_image(rgb, width, height, ColorType::Rgb8.into())
            .expect("encode png");
        bytes
    }
}
