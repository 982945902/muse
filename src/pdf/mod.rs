use crate::config::Config;
use async_trait::async_trait;
use lopdf::Document;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, process::Command, sync::Arc};
use tokio::task;
use uuid::Uuid;

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
    pub raster_provider: Option<String>,
    #[serde(default)]
    pub raster_pages: Vec<PdfOcrPage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdfOcrPage {
    pub page_no: u32,
    pub mime_type: Option<String>,
    pub bytes: Vec<u8>,
}

#[async_trait]
pub trait PdfProvider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn extract(&self, request: PdfRequest) -> anyhow::Result<PdfOutput>;
}

#[async_trait]
pub trait PdfRasterizer: Send + Sync {
    fn name(&self) -> &'static str;
    async fn rasterize(&self, request: &PdfRequest) -> anyhow::Result<Vec<PdfOcrPage>>;
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
            raster_provider: None,
            raster_pages: vec![],
        })
    }
}

pub struct CompositePdfProvider {
    text_layer: Arc<dyn PdfProvider>,
    rasterizer: Option<Arc<dyn PdfRasterizer>>,
}

impl CompositePdfProvider {
    pub fn new(
        text_layer: Arc<dyn PdfProvider>,
        rasterizer: Option<Arc<dyn PdfRasterizer>>,
    ) -> Self {
        Self {
            text_layer,
            rasterizer,
        }
    }

    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let rasterizer = build_pdf_rasterizer(config)?;
        Ok(Self::new(Arc::new(LopdfTextLayerProvider), rasterizer))
    }
}

#[async_trait]
impl PdfProvider for CompositePdfProvider {
    fn name(&self) -> &'static str {
        if self.rasterizer.is_some() {
            "composite-pdf"
        } else {
            self.text_layer.name()
        }
    }

    async fn extract(&self, request: PdfRequest) -> anyhow::Result<PdfOutput> {
        let mut output = self.text_layer.extract(request.clone()).await?;
        if !output.extracted_text_layer {
            if let Some(rasterizer) = &self.rasterizer {
                output.raster_pages = rasterizer.rasterize(&request).await?;
                output.raster_provider = Some(rasterizer.name().to_string());
            }
        }
        Ok(output)
    }
}

pub struct PdftoppmRasterizer {
    binary: String,
}

impl PdftoppmRasterizer {
    pub fn new(binary: impl Into<String>) -> anyhow::Result<Self> {
        let binary = binary.into();
        if binary.trim().is_empty() {
            anyhow::bail!("pdftoppm rasterizer requires a non-empty binary path");
        }
        Ok(Self { binary })
    }
}

#[async_trait]
impl PdfRasterizer for PdftoppmRasterizer {
    fn name(&self) -> &'static str {
        "pdftoppm-rasterizer"
    }

    async fn rasterize(&self, request: &PdfRequest) -> anyhow::Result<Vec<PdfOcrPage>> {
        let binary = self.binary.clone();
        let request = request.clone();
        task::spawn_blocking(move || rasterize_pdf_with_pdftoppm(binary, request))
            .await
            .map_err(|error| anyhow::anyhow!("pdftoppm rasterization task join error: {error}"))?
    }
}

fn build_pdf_rasterizer(config: &Config) -> anyhow::Result<Option<Arc<dyn PdfRasterizer>>> {
    match config.pdf_raster_provider.as_str() {
        "none" => Ok(None),
        "pdftoppm" => {
            let binary = config
                .pdftoppm_bin
                .clone()
                .unwrap_or_else(|| "pdftoppm".to_string());
            Ok(Some(Arc::new(PdftoppmRasterizer::new(binary)?)))
        }
        other => anyhow::bail!("unsupported PDF raster provider `{other}`"),
    }
}

fn rasterize_pdf_with_pdftoppm(
    binary: String,
    request: PdfRequest,
) -> anyhow::Result<Vec<PdfOcrPage>> {
    let workdir = std::env::temp_dir().join(format!("muse-pdftoppm-{}", Uuid::new_v4()));
    fs::create_dir_all(&workdir)?;

    let result = (|| -> anyhow::Result<Vec<PdfOcrPage>> {
        let input_path = workdir.join("input.pdf");
        let output_prefix = workdir.join("page");
        fs::write(&input_path, &request.bytes)?;

        let output = Command::new(&binary)
            .arg("-png")
            .arg(&input_path)
            .arg(&output_prefix)
            .output()
            .map_err(|error| anyhow::anyhow!("failed to execute `{binary}`: {error}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "`{binary}` exited with {}: {}",
                output.status,
                stderr.trim()
            );
        }

        let mut page_files = fs::read_dir(&workdir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
                    && path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .map(|stem| stem.starts_with("page-"))
                        .unwrap_or(false)
            })
            .collect::<Vec<PathBuf>>();
        page_files.sort();

        let pages = page_files
            .into_iter()
            .enumerate()
            .map(|(index, path)| -> anyhow::Result<PdfOcrPage> {
                Ok(PdfOcrPage {
                    page_no: (index + 1) as u32,
                    mime_type: Some("image/png".to_string()),
                    bytes: fs::read(path)?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        if pages.is_empty() {
            anyhow::bail!("`{binary}` produced no rasterized pages")
        }

        Ok(pages)
    })();

    let _ = fs::remove_dir_all(&workdir);
    result
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
    use std::sync::{Arc, Mutex};

    struct EmptyTextPdfProvider;

    struct StaticRasterizer {
        page_count: usize,
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl PdfProvider for EmptyTextPdfProvider {
        fn name(&self) -> &'static str {
            "empty-text-pdf"
        }

        async fn extract(&self, _request: PdfRequest) -> anyhow::Result<PdfOutput> {
            Ok(PdfOutput {
                text: String::new(),
                page_count: Some(2),
                extracted_text_layer: false,
                page_texts: vec![],
                raster_provider: None,
                raster_pages: vec![],
            })
        }
    }

    #[async_trait]
    impl PdfRasterizer for StaticRasterizer {
        fn name(&self) -> &'static str {
            "static-rasterizer"
        }

        async fn rasterize(&self, request: &PdfRequest) -> anyhow::Result<Vec<PdfOcrPage>> {
            self.calls
                .lock()
                .expect("lock")
                .push(request.file_name.clone().unwrap_or_default());
            Ok((1..=self.page_count)
                .map(|page_no| PdfOcrPage {
                    page_no: page_no as u32,
                    mime_type: Some("image/png".to_string()),
                    bytes: vec![page_no as u8],
                })
                .collect())
        }
    }

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
        assert!(output.raster_pages.is_empty());
    }

    #[tokio::test]
    async fn composite_provider_uses_rasterizer_when_pdf_has_no_text_layer() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = CompositePdfProvider::new(
            Arc::new(EmptyTextPdfProvider),
            Some(Arc::new(StaticRasterizer {
                page_count: 2,
                calls: calls.clone(),
            })),
        );

        let output = provider
            .extract(PdfRequest {
                file_name: Some("scan.pdf".to_string()),
                mime_type: Some("application/pdf".to_string()),
                bytes: b"%PDF-1.7 fake".to_vec(),
            })
            .await
            .expect("composite pdf output");

        assert!(!output.extracted_text_layer);
        assert_eq!(output.raster_pages.len(), 2);
        assert_eq!(output.raster_pages[0].page_no, 1);
        assert_eq!(output.raster_pages[1].page_no, 2);
        assert_eq!(output.raster_provider.as_deref(), Some("static-rasterizer"));
        assert_eq!(calls.lock().expect("lock").as_slice(), &["scan.pdf"]);
    }

    #[tokio::test]
    async fn composite_provider_skips_rasterizer_when_text_layer_exists() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = CompositePdfProvider::new(
            Arc::new(LopdfTextLayerProvider),
            Some(Arc::new(StaticRasterizer {
                page_count: 1,
                calls: calls.clone(),
            })),
        );

        let output = provider
            .extract(PdfRequest {
                file_name: Some("sample.pdf".to_string()),
                mime_type: Some("application/pdf".to_string()),
                bytes: build_single_page_pdf("role_type: pdf text"),
            })
            .await
            .expect("composite pdf output");

        assert!(output.extracted_text_layer);
        assert!(output.raster_pages.is_empty());
        assert!(output.raster_provider.is_none());
        assert!(calls.lock().expect("lock").is_empty());
    }

    #[test]
    fn from_config_rejects_unsupported_pdf_raster_provider() {
        let config = Config {
            listen_addr: "127.0.0.1:0".parse().expect("socket addr"),
            service_name: "test".to_string(),
            log_filter: "info".to_string(),
            extractor_provider: "heuristic".to_string(),
            onnx_model_path: None,
            onnx_model_spec_path: None,
            onnx_threads: 1,
            onnx_input_text_name: "text".to_string(),
            onnx_input_schema_name: "schema".to_string(),
            onnx_output_json_name: "json_output".to_string(),
            ocr_provider: "placeholder".to_string(),
            ocr_fallback_provider: None,
            ocr_worker_url: None,
            ocr_timeout_ms: 5_000,
            ocr_worker_token: None,
            ocr_model_dir: None,
            ocr_threads: 1,
            ocr_prewarm: false,
            pdf_raster_provider: "mystery".to_string(),
            pdftoppm_bin: None,
        };

        let error = match CompositePdfProvider::from_config(&config) {
            Ok(_) => panic!("invalid provider should fail"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("unsupported PDF raster provider")
        );
    }

    #[test]
    fn pdftoppm_rasterizer_surfaces_missing_binary() {
        let error = rasterize_pdf_with_pdftoppm(
            "__definitely_missing_pdftoppm_binary__".to_string(),
            PdfRequest {
                file_name: Some("scan.pdf".to_string()),
                mime_type: Some("application/pdf".to_string()),
                bytes: build_single_page_pdf("scan page"),
            },
        )
        .expect_err("missing binary should fail");

        assert!(error.to_string().contains("failed to execute"));
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
