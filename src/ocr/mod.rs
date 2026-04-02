use crate::{config::Config, domain::BBox};
use anyhow::Context;
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView, RgbImage, imageops::FilterType};
#[cfg(feature = "onnx-runtime")]
use onnxruntime_sys_ng as ort_sys;
use reqwest::Client;
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "onnx-runtime", not(target_family = "windows")))]
use std::os::unix::ffi::OsStrExt;
#[cfg(all(feature = "onnx-runtime", target_family = "windows"))]
use std::os::windows::ffi::OsStrExt;
use std::{
    collections::VecDeque,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
#[cfg(feature = "onnx-runtime")]
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
    ptr,
    sync::OnceLock,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OcrRequest {
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    pub page_no: Option<u32>,
    pub bbox: Option<BBox>,
    pub confidence: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OcrOutput {
    pub lines: Vec<OcrLine>,
    pub provider: Option<String>,
    pub model: Option<String>,
}

#[async_trait]
pub trait OcrProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn transport_name(&self) -> &'static str {
        self.name()
    }
    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput>;
}

#[derive(Clone, Debug)]
struct OcrPreparedInput {
    file_name: Option<String>,
    mime_type: Option<String>,
    original_bytes: Vec<u8>,
    pages: Vec<OcrPreparedPage>,
}

#[derive(Clone, Debug)]
struct OcrPreparedPage {
    page_no: u32,
    mime_type: Option<String>,
    bytes: Vec<u8>,
    raster: Option<OcrPreparedRaster>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct OcrPreparedRaster {
    original_width: u32,
    original_height: u32,
    resized_width: u32,
    resized_height: u32,
    channels: u32,
    scale_x: f32,
    scale_y: f32,
    resized_rgb_u8: Vec<u8>,
    chw_f32: Vec<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct RawOcrOutput {
    lines: Vec<RawOcrLine>,
    provider: Option<String>,
    model: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RawOcrLine {
    text: String,
    page_no: Option<u32>,
    bbox: Option<BBox>,
    confidence: Option<f32>,
}

trait OcrPreprocessor: Send + Sync {
    fn prepare(&self, request: OcrRequest) -> anyhow::Result<OcrPreparedInput>;
}

#[async_trait]
trait OcrRuntime: Send + Sync {
    async fn recognize(&self, input: OcrPreparedInput) -> anyhow::Result<RawOcrOutput>;
}

trait OcrResultAdapter: Send + Sync {
    fn adapt(&self, raw: RawOcrOutput) -> anyhow::Result<OcrOutput>;
}

struct LayeredOcrProvider<P, R, A> {
    provider_name: &'static str,
    preprocessor: P,
    runtime: R,
    adapter: A,
    _marker: PhantomData<(P, R, A)>,
}

impl<P, R, A> LayeredOcrProvider<P, R, A> {
    fn new(provider_name: &'static str, preprocessor: P, runtime: R, adapter: A) -> Self {
        Self {
            provider_name,
            preprocessor,
            runtime,
            adapter,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<P, R, A> OcrProvider for LayeredOcrProvider<P, R, A>
where
    P: OcrPreprocessor + Send + Sync,
    R: OcrRuntime + Send + Sync,
    A: OcrResultAdapter + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.provider_name
    }

    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
        let prepared = self.preprocessor.prepare(request)?;
        let raw = self.runtime.recognize(prepared).await?;
        self.adapter.adapt(raw)
    }
}

#[derive(Default)]
struct PassthroughOcrPreprocessor;

impl OcrPreprocessor for PassthroughOcrPreprocessor {
    fn prepare(&self, request: OcrRequest) -> anyhow::Result<OcrPreparedInput> {
        Ok(OcrPreparedInput {
            file_name: request.file_name,
            mime_type: request.mime_type.clone(),
            original_bytes: request.bytes.clone(),
            pages: vec![OcrPreparedPage {
                page_no: 1,
                mime_type: request.mime_type,
                bytes: request.bytes,
                raster: None,
            }],
        })
    }
}

const LOCAL_ONNX_MAX_IMAGE_SIDE: u32 = 1536;
#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
const LOCAL_ONNX_DETECTION_THRESHOLD: f32 = 0.3;
#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
const LOCAL_ONNX_MIN_COMPONENT_PIXELS: usize = 4;
#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
const LOCAL_ONNX_RECOGNITION_TARGET_HEIGHT: u32 = 48;
#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
const LOCAL_ONNX_RECOGNITION_MAX_WIDTH: u32 = 320;

struct LocalOnnxImagePreprocessor {
    max_image_side: u32,
}

impl Default for LocalOnnxImagePreprocessor {
    fn default() -> Self {
        Self {
            max_image_side: LOCAL_ONNX_MAX_IMAGE_SIDE,
        }
    }
}

impl OcrPreprocessor for LocalOnnxImagePreprocessor {
    fn prepare(&self, request: OcrRequest) -> anyhow::Result<OcrPreparedInput> {
        let image = image::load_from_memory(&request.bytes).context(
            "local ONNX OCR preprocessor failed to decode image bytes; only raster image inputs are supported right now",
        )?;
        let raster = prepare_raster_page(&image, self.max_image_side)?;

        Ok(OcrPreparedInput {
            file_name: request.file_name,
            mime_type: request.mime_type.clone(),
            original_bytes: request.bytes.clone(),
            pages: vec![OcrPreparedPage {
                page_no: 1,
                mime_type: request.mime_type,
                bytes: request.bytes,
                raster: Some(raster),
            }],
        })
    }
}

fn prepare_raster_page(
    image: &DynamicImage,
    max_image_side: u32,
) -> anyhow::Result<OcrPreparedRaster> {
    let (original_width, original_height) = image.dimensions();
    if original_width == 0 || original_height == 0 {
        anyhow::bail!("OCR image dimensions must be greater than zero");
    }

    let (resized_width, resized_height) =
        resize_dimensions_within_limit(original_width, original_height, max_image_side);
    let resized = if resized_width != original_width || resized_height != original_height {
        image.resize_exact(resized_width, resized_height, FilterType::Triangle)
    } else {
        image.clone()
    };
    let rgb = resized.to_rgb8();
    let chw_f32 = rgb_to_chw_f32(rgb.as_raw());

    Ok(OcrPreparedRaster {
        original_width,
        original_height,
        resized_width,
        resized_height,
        channels: 3,
        scale_x: original_width as f32 / resized_width as f32,
        scale_y: original_height as f32 / resized_height as f32,
        resized_rgb_u8: rgb.into_raw(),
        chw_f32,
    })
}

fn resize_dimensions_within_limit(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    if width == 0 || height == 0 || max_side == 0 {
        return (width.max(1), height.max(1));
    }

    let current_max = width.max(height);
    if current_max <= max_side {
        return (width, height);
    }

    let scale = max_side as f32 / current_max as f32;
    let resized_width = ((width as f32) * scale).round().max(1.0) as u32;
    let resized_height = ((height as f32) * scale).round().max(1.0) as u32;
    (resized_width, resized_height)
}

#[derive(Default)]
struct NormalizeOcrResultAdapter;

impl OcrResultAdapter for NormalizeOcrResultAdapter {
    fn adapt(&self, raw: RawOcrOutput) -> anyhow::Result<OcrOutput> {
        let lines = raw
            .lines
            .into_iter()
            .filter_map(|line| {
                let text = line.text.trim().to_string();
                if text.is_empty() {
                    return None;
                }

                Some(OcrLine {
                    text,
                    page_no: line.page_no.filter(|page_no| *page_no > 0),
                    bbox: line.bbox,
                    confidence: line.confidence,
                })
            })
            .collect();

        Ok(OcrOutput {
            lines,
            provider: raw.provider,
            model: raw.model,
        })
    }
}

#[derive(Default)]
struct PlaceholderOcrRuntime;

#[async_trait]
impl OcrRuntime for PlaceholderOcrRuntime {
    async fn recognize(&self, input: OcrPreparedInput) -> anyhow::Result<RawOcrOutput> {
        let file_name = input.file_name.unwrap_or_else(|| "unnamed".to_string());
        let mime_type = input.mime_type.unwrap_or_else(|| "unknown".to_string());
        let byte_size = input.original_bytes.len();
        let page_no = input.pages.first().map(|page| page.page_no).unwrap_or(1);

        Ok(RawOcrOutput {
            lines: vec![RawOcrLine {
                text: format!(
                    "image OCR provider not implemented yet\nfile_name: {file_name}\nmime_type: {mime_type}\nbyte_size: {byte_size}"
                ),
                page_no: Some(page_no),
                bbox: None,
                confidence: Some(0.1),
            }],
            provider: Some("placeholder-ocr".to_string()),
            model: None,
        })
    }
}

struct HttpOcrRuntime {
    client: Client,
    endpoint: String,
    auth_token: Option<String>,
}

impl HttpOcrRuntime {
    fn new(
        endpoint: impl Into<String>,
        timeout: Duration,
        auth_token: Option<String>,
    ) -> anyhow::Result<Self> {
        let endpoint = endpoint.into();
        if endpoint.trim().is_empty() {
            anyhow::bail!("HTTP OCR provider requires a non-empty endpoint");
        }
        if timeout.is_zero() {
            anyhow::bail!("HTTP OCR provider timeout must be greater than zero");
        }

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .context("failed to build HTTP OCR client")?;

        Ok(Self {
            client,
            endpoint,
            auth_token,
        })
    }
}

#[async_trait]
impl OcrRuntime for HttpOcrRuntime {
    async fn recognize(&self, input: OcrPreparedInput) -> anyhow::Result<RawOcrOutput> {
        let mime_type = input
            .mime_type
            .clone()
            .or_else(|| input.pages.first().and_then(|page| page.mime_type.clone()))
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let payload_bytes = input
            .pages
            .first()
            .map(|page| page.bytes.clone())
            .unwrap_or_else(|| input.original_bytes.clone());

        let mut builder = self
            .client
            .post(&self.endpoint)
            .header("content-type", mime_type)
            .body(payload_bytes);

        if let Some(file_name) = input.file_name.as_deref() {
            builder = builder.header("x-file-name", file_name);
        }

        if let Some(token) = self.auth_token.as_deref() {
            builder = builder.bearer_auth(token);
        }

        let response = builder
            .send()
            .await
            .with_context(|| format!("failed to call OCR worker `{}`", self.endpoint))?;
        let status = response.status();

        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read error body>".to_string());
            anyhow::bail!(
                "OCR worker `{}` returned {}: {}",
                self.endpoint,
                status,
                error_body
            );
        }

        let payload: WorkerOcrResponse = response.json().await.with_context(|| {
            format!(
                "failed to decode OCR worker response from `{}`",
                self.endpoint
            )
        })?;

        Ok(RawOcrOutput {
            lines: payload.lines,
            provider: payload.provider,
            model: payload.model,
        })
    }
}

type PlaceholderPipeline = LayeredOcrProvider<
    PassthroughOcrPreprocessor,
    PlaceholderOcrRuntime,
    NormalizeOcrResultAdapter,
>;

pub struct PlaceholderOcrProvider {
    inner: PlaceholderPipeline,
}

impl Default for PlaceholderOcrProvider {
    fn default() -> Self {
        Self {
            inner: LayeredOcrProvider::new(
                "placeholder-ocr",
                PassthroughOcrPreprocessor,
                PlaceholderOcrRuntime,
                NormalizeOcrResultAdapter,
            ),
        }
    }
}

#[async_trait]
impl OcrProvider for PlaceholderOcrProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn transport_name(&self) -> &'static str {
        "inproc"
    }

    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
        self.inner.recognize(request).await
    }
}

type HttpPipeline =
    LayeredOcrProvider<PassthroughOcrPreprocessor, HttpOcrRuntime, NormalizeOcrResultAdapter>;

pub struct HttpOcrProvider {
    inner: HttpPipeline,
}

impl HttpOcrProvider {
    pub fn new(
        endpoint: impl Into<String>,
        timeout: Duration,
        auth_token: Option<String>,
    ) -> anyhow::Result<Self> {
        let runtime = HttpOcrRuntime::new(endpoint, timeout, auth_token)?;
        Ok(Self {
            inner: LayeredOcrProvider::new(
                "http-ocr-worker",
                PassthroughOcrPreprocessor,
                runtime,
                NormalizeOcrResultAdapter,
            ),
        })
    }

    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let endpoint = config.ocr_worker_url.clone().ok_or_else(|| {
            anyhow::anyhow!("`MUSE_OCR_WORKER_URL` is required when MUSE_OCR_PROVIDER=http")
        })?;

        Self::new(
            endpoint,
            Duration::from_millis(config.ocr_timeout_ms),
            config.ocr_worker_token.clone(),
        )
    }
}

#[async_trait]
impl OcrProvider for HttpOcrProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn transport_name(&self) -> &'static str {
        "http"
    }

    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
        self.inner.recognize(request).await
    }
}

pub struct FallbackOcrProvider {
    primary: Arc<dyn OcrProvider>,
    fallback: Arc<dyn OcrProvider>,
}

impl FallbackOcrProvider {
    pub fn new(primary: Arc<dyn OcrProvider>, fallback: Arc<dyn OcrProvider>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl OcrProvider for FallbackOcrProvider {
    fn name(&self) -> &'static str {
        "fallback-ocr"
    }

    fn transport_name(&self) -> &'static str {
        "composite"
    }

    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
        match self.primary.recognize(request.clone()).await {
            Ok(output) => Ok(output),
            Err(primary_error) => self.fallback.recognize(request).await.with_context(|| {
                format!(
                    "primary OCR provider `{}` failed and fallback provider `{}` also failed; primary error: {primary_error}",
                    self.primary.name(),
                    self.fallback.name(),
                )
            }),
        }
    }
}

#[derive(Clone, Debug)]
struct LocalOnnxModelFiles {
    det_model_path: PathBuf,
    rec_model_path: PathBuf,
    cls_model_path: Option<PathBuf>,
    rec_charset_path: PathBuf,
}

impl LocalOnnxModelFiles {
    fn discover(model_dir: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let model_dir = model_dir.into();
        if !model_dir.is_dir() {
            anyhow::bail!(
                "local ONNX OCR model directory was not found: {}",
                model_dir.display()
            );
        }

        Ok(Self {
            det_model_path: resolve_local_onnx_model_path(
                &model_dir,
                "detection",
                &["det.onnx", "ocr_det.onnx", "text_detection.onnx"],
                true,
            )?
            .expect("required OCR detection model should exist"),
            rec_model_path: resolve_local_onnx_model_path(
                &model_dir,
                "recognition",
                &["rec.onnx", "ocr_rec.onnx", "text_recognition.onnx"],
                true,
            )?
            .expect("required OCR recognition model should exist"),
            cls_model_path: resolve_local_onnx_model_path(
                &model_dir,
                "classification",
                &[
                    "cls.onnx",
                    "ocr_cls.onnx",
                    "text_classification.onnx",
                    "text_direction.onnx",
                ],
                false,
            )?,
            rec_charset_path: resolve_local_onnx_model_path(
                &model_dir,
                "recognition charset",
                &[
                    "ppocr_keys_v1.txt",
                    "dict.txt",
                    "ocr_keys.txt",
                    "keys.txt",
                    "charset.txt",
                ],
                true,
            )?
            .expect("required OCR recognition charset should exist"),
        })
    }

    fn summary(&self) -> String {
        let mut parts = vec![
            format!("det={}", self.det_model_path.display()),
            format!("rec={}", self.rec_model_path.display()),
            format!("charset={}", self.rec_charset_path.display()),
        ];
        if let Some(cls) = &self.cls_model_path {
            parts.push(format!("cls={}", cls.display()));
        }
        parts.join(", ")
    }
}

fn resolve_local_onnx_model_path(
    model_dir: &Path,
    stage_name: &str,
    candidates: &[&str],
    required: bool,
) -> anyhow::Result<Option<PathBuf>> {
    for candidate in candidates {
        let path = model_dir.join(candidate);
        if path.is_file() {
            return Ok(Some(path));
        }
    }

    if required {
        anyhow::bail!(
            "local ONNX OCR {stage_name} model was not found under `{}`; tried: {}",
            model_dir.display(),
            candidates.join(", ")
        );
    }

    Ok(None)
}

trait LocalOcrBackend: Send + Sync {
    fn prewarm(&self) -> anyhow::Result<()>;
    fn run_detection(&self, page: &OcrPreparedPage) -> anyhow::Result<OcrDetectionResult>;
    fn run_recognition(
        &self,
        patches: &[OcrRecognitionPatch],
        charset: &OcrCharset,
    ) -> anyhow::Result<OcrRecognitionRun>;
}

struct LocalOnnxOcrRuntime {
    model_summary: String,
    backend: Box<dyn LocalOcrBackend>,
    recognition_charset: OcrCharset,
}

impl LocalOnnxOcrRuntime {
    fn new(model_dir: impl Into<PathBuf>, threads: usize, prewarm: bool) -> anyhow::Result<Self> {
        if threads == 0 {
            anyhow::bail!("local ONNX OCR requires `MUSE_OCR_THREADS >= 1`");
        }

        let model_files = LocalOnnxModelFiles::discover(model_dir)?;
        let model_summary = model_files.summary();
        let recognition_charset = load_ocr_charset(&model_files.rec_charset_path)?;
        let backend: Box<dyn LocalOcrBackend> =
            Box::new(LocalOnnxRuntimeBackend::new(&model_files, threads)?);
        if prewarm {
            backend.prewarm()?;
        }

        Ok(Self {
            model_summary,
            backend,
            recognition_charset,
        })
    }

    #[cfg(test)]
    fn from_parts(
        model_summary: impl Into<String>,
        backend: Box<dyn LocalOcrBackend>,
        recognition_charset: OcrCharset,
    ) -> Self {
        Self {
            model_summary: model_summary.into(),
            backend,
            recognition_charset,
        }
    }
}

#[async_trait]
impl OcrRuntime for LocalOnnxOcrRuntime {
    async fn recognize(&self, input: OcrPreparedInput) -> anyhow::Result<RawOcrOutput> {
        let mut lines = Vec::new();

        for page in &input.pages {
            let detection = self.backend.run_detection(page)?;
            let recognition_patches = page
                .raster
                .as_ref()
                .map(|raster| {
                    build_recognition_patches(
                        raster,
                        &detection.candidates,
                        LOCAL_ONNX_RECOGNITION_TARGET_HEIGHT,
                        LOCAL_ONNX_RECOGNITION_MAX_WIDTH,
                    )
                })
                .unwrap_or_default();
            let recognition = self
                .backend
                .run_recognition(&recognition_patches, &self.recognition_charset)?;
            lines.extend(assemble_recognized_lines(
                page.page_no,
                &detection,
                &recognition,
            )?);
        }

        Ok(RawOcrOutput {
            lines,
            provider: Some("local-onnx-ocr".to_string()),
            model: Some(self.model_summary.clone()),
        })
    }
}

type LocalOnnxPipeline =
    LayeredOcrProvider<LocalOnnxImagePreprocessor, LocalOnnxOcrRuntime, NormalizeOcrResultAdapter>;

pub struct LocalOnnxOcrProvider {
    inner: LocalOnnxPipeline,
}

impl LocalOnnxOcrProvider {
    pub fn new(
        model_dir: impl Into<PathBuf>,
        threads: usize,
        prewarm: bool,
    ) -> anyhow::Result<Self> {
        let runtime = LocalOnnxOcrRuntime::new(model_dir, threads, prewarm)?;
        Ok(Self {
            inner: LayeredOcrProvider::new(
                "local-onnx-ocr",
                LocalOnnxImagePreprocessor::default(),
                runtime,
                NormalizeOcrResultAdapter,
            ),
        })
    }

    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let model_dir = config.ocr_model_dir.clone().ok_or_else(|| {
            anyhow::anyhow!("`MUSE_OCR_MODEL_DIR` is required when MUSE_OCR_PROVIDER=local-onnx")
        })?;

        Self::new(model_dir, config.ocr_threads, config.ocr_prewarm)
    }

    #[cfg(test)]
    pub(crate) fn from_test_fixture(fixture: TestLocalOcrFixture) -> Self {
        let runtime = LocalOnnxOcrRuntime::from_parts(
            fixture.model_summary,
            Box::new(StaticLocalOcrBackend {
                detection: OcrDetectionResult {
                    output_name: "fixture-detection".to_string(),
                    heatmap_shape: vec![],
                    candidates: fixture
                        .candidates
                        .into_iter()
                        .map(|candidate| OcrDetectionCandidate {
                            bbox: candidate.bbox,
                            score: candidate.score,
                        })
                        .collect(),
                },
                recognition: OcrRecognitionRun {
                    patch_count: fixture.predictions.len(),
                    executed_patches: fixture.predictions.len(),
                    predictions: fixture
                        .predictions
                        .into_iter()
                        .map(|prediction| OcrRecognitionPrediction {
                            patch_index: prediction.patch_index,
                            text: prediction.text,
                            confidence: prediction.confidence,
                        })
                        .collect(),
                    outputs: Vec::new(),
                },
            }),
            OcrCharset {
                tokens: fixture.charset,
            },
        );

        Self {
            inner: LayeredOcrProvider::new(
                "local-onnx-ocr",
                LocalOnnxImagePreprocessor::default(),
                runtime,
                NormalizeOcrResultAdapter,
            ),
        }
    }
}

#[async_trait]
impl OcrProvider for LocalOnnxOcrProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn transport_name(&self) -> &'static str {
        "inproc"
    }

    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
        self.inner.recognize(request).await
    }
}

#[cfg(feature = "onnx-runtime")]
type OrtStatusPtr = *mut ort_sys::OrtStatus;

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct OrtIoInfo {
    name: String,
    tensor_type: ort_sys::ONNXTensorElementDataType,
    dimensions: Vec<Option<u32>>,
}

#[cfg(feature = "onnx-runtime")]
fn ort_api() -> &'static ort_sys::OrtApi {
    static API: OnceLock<OrtApiRef> = OnceLock::new();
    let api_ref = API.get_or_init(|| {
        let base = unsafe { ort_sys::OrtGetApiBase() };
        assert!(!base.is_null(), "ONNX Runtime API base pointer is null");
        let get_api = unsafe { (*base).GetApi }.expect("ONNX Runtime GetApi function is missing");
        let api_ptr = unsafe { get_api(ort_sys::ORT_API_VERSION) };
        assert!(!api_ptr.is_null(), "ONNX Runtime API pointer is null");
        OrtApiRef(api_ptr)
    });
    unsafe { &*api_ref.0 }
}

#[cfg(feature = "onnx-runtime")]
struct OrtApiRef(*const ort_sys::OrtApi);

#[cfg(feature = "onnx-runtime")]
unsafe impl Send for OrtApiRef {}

#[cfg(feature = "onnx-runtime")]
unsafe impl Sync for OrtApiRef {}

#[cfg(feature = "onnx-runtime")]
fn ort_status_to_result(status: OrtStatusPtr, context: &str) -> anyhow::Result<()> {
    if status.is_null() {
        return Ok(());
    }

    let message = unsafe {
        let message_ptr = ort_api().GetErrorMessage.unwrap()(status);
        let text = CStr::from_ptr(message_ptr).to_string_lossy().into_owned();
        ort_api().ReleaseStatus.unwrap()(status);
        text
    };
    Err(anyhow::anyhow!("{context}: {message}"))
}

#[cfg(feature = "onnx-runtime")]
fn extract_allocated_string(
    allocator_ptr: *mut ort_sys::OrtAllocator,
    raw_ptr: *mut c_char,
) -> anyhow::Result<String> {
    if raw_ptr.is_null() {
        anyhow::bail!("ONNX Runtime returned a null string pointer")
    }

    let text = unsafe { CStr::from_ptr(raw_ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        let free_fn = (*allocator_ptr)
            .Free
            .expect("default ORT allocator must provide Free");
        free_fn(allocator_ptr, raw_ptr.cast::<c_void>());
    }
    Ok(text)
}

#[cfg(feature = "onnx-runtime")]
#[cfg(target_family = "windows")]
type OrtModelPathChar = u16;

#[cfg(feature = "onnx-runtime")]
#[cfg(not(target_family = "windows"))]
type OrtModelPathChar = c_char;

#[cfg(feature = "onnx-runtime")]
fn model_path_bytes(model_path: &Path) -> anyhow::Result<ModelPathBytes> {
    #[cfg(target_family = "windows")]
    {
        let encoded = model_path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<OrtModelPathChar>>();
        Ok(ModelPathBytes::Wide(encoded))
    }

    #[cfg(not(target_family = "windows"))]
    {
        let encoded = model_path
            .as_os_str()
            .as_bytes()
            .iter()
            .copied()
            .chain(std::iter::once(0))
            .map(|byte| byte as OrtModelPathChar)
            .collect::<Vec<OrtModelPathChar>>();
        Ok(ModelPathBytes::Narrow(encoded))
    }
}

#[cfg(feature = "onnx-runtime")]
enum ModelPathBytes {
    #[cfg(target_family = "windows")]
    Wide(Vec<OrtModelPathChar>),
    #[cfg(not(target_family = "windows"))]
    Narrow(Vec<OrtModelPathChar>),
}

#[cfg(feature = "onnx-runtime")]
impl ModelPathBytes {
    fn as_ptr(&self) -> *const OrtModelPathChar {
        match self {
            #[cfg(target_family = "windows")]
            Self::Wide(bytes) => bytes.as_ptr(),
            #[cfg(not(target_family = "windows"))]
            Self::Narrow(bytes) => bytes.as_ptr(),
        }
    }
}

#[cfg(feature = "onnx-runtime")]
fn extract_dimensions(
    tensor_info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo,
) -> anyhow::Result<Vec<Option<u32>>> {
    let mut count = 0;
    ort_status_to_result(
        unsafe { ort_api().GetDimensionsCount.unwrap()(tensor_info_ptr, &mut count) },
        "failed to read ONNX tensor dimension count",
    )?;

    if count == 0 {
        return Ok(vec![]);
    }

    let mut dimensions = vec![0_i64; count];
    ort_status_to_result(
        unsafe {
            ort_api().GetDimensions.unwrap()(tensor_info_ptr, dimensions.as_mut_ptr(), count)
        },
        "failed to read ONNX tensor dimensions",
    )?;

    Ok(dimensions
        .into_iter()
        .map(|dimension| {
            if dimension < 0 {
                None
            } else {
                Some(dimension as u32)
            }
        })
        .collect())
}

#[cfg(feature = "onnx-runtime")]
fn extract_io_info(
    session_ptr: *mut ort_sys::OrtSession,
    allocator_ptr: *mut ort_sys::OrtAllocator,
    count: usize,
    get_name: unsafe extern "C" fn(
        *const ort_sys::OrtSession,
        usize,
        *mut ort_sys::OrtAllocator,
        *mut *mut c_char,
    ) -> OrtStatusPtr,
    get_type_info: unsafe extern "C" fn(
        *const ort_sys::OrtSession,
        usize,
        *mut *mut ort_sys::OrtTypeInfo,
    ) -> OrtStatusPtr,
) -> anyhow::Result<Vec<OrtIoInfo>> {
    let mut infos = Vec::with_capacity(count);

    for index in 0..count {
        let mut name_ptr = ptr::null_mut();
        ort_status_to_result(
            unsafe { get_name(session_ptr, index, allocator_ptr, &mut name_ptr) },
            "failed to read ONNX input/output name",
        )?;
        let name = extract_allocated_string(allocator_ptr, name_ptr)?;

        let mut type_info_ptr = ptr::null_mut();
        ort_status_to_result(
            unsafe { get_type_info(session_ptr, index, &mut type_info_ptr) },
            "failed to read ONNX input/output type info",
        )?;

        let mut tensor_info_ptr = ptr::null();
        let result = (|| -> anyhow::Result<OrtIoInfo> {
            ort_status_to_result(
                unsafe {
                    ort_api().CastTypeInfoToTensorInfo.unwrap()(type_info_ptr, &mut tensor_info_ptr)
                },
                "failed to cast ONNX type info to tensor info",
            )?;

            let mut tensor_type =
                ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_status_to_result(
                unsafe {
                    ort_api().GetTensorElementType.unwrap()(tensor_info_ptr, &mut tensor_type)
                },
                "failed to read ONNX tensor element type",
            )?;

            Ok(OrtIoInfo {
                name,
                tensor_type,
                dimensions: extract_dimensions(tensor_info_ptr)?,
            })
        })();

        unsafe {
            ort_api().ReleaseTypeInfo.unwrap()(type_info_ptr);
        }
        infos.push(result?);
    }

    Ok(infos)
}

#[cfg(feature = "onnx-runtime")]
fn normalize_float_input_shape(
    input_name: &str,
    dimensions: &[Option<u32>],
    requested_shape: &[i64],
) -> anyhow::Result<Vec<usize>> {
    if dimensions.len() != requested_shape.len() {
        anyhow::bail!(
            "OCR ONNX input `{input_name}` expects rank {}, got {}",
            dimensions.len(),
            requested_shape.len()
        );
    }

    let mut shape = Vec::with_capacity(dimensions.len());
    for (axis, (declared, requested)) in dimensions.iter().zip(requested_shape.iter()).enumerate() {
        if *requested <= 0 {
            anyhow::bail!(
                "OCR ONNX input `{input_name}` has invalid requested shape {} at axis {axis}",
                requested
            );
        }

        let requested = *requested as usize;
        match declared {
            Some(value) if *value as usize != requested => anyhow::bail!(
                "OCR ONNX input `{input_name}` expects dimension {} at axis {axis}, got {}",
                value,
                requested
            ),
            Some(_) | None => shape.push(requested),
        }
    }

    Ok(shape)
}

#[cfg(feature = "onnx-runtime")]
struct OrtValueGuard {
    ptr: *mut ort_sys::OrtValue,
    _float_values: Option<Vec<f32>>,
}

#[cfg(feature = "onnx-runtime")]
impl OrtValueGuard {
    fn new(ptr: *mut ort_sys::OrtValue) -> Self {
        Self {
            ptr,
            _float_values: None,
        }
    }

    fn with_float_values(ptr: *mut ort_sys::OrtValue, float_values: Vec<f32>) -> Self {
        Self {
            ptr,
            _float_values: Some(float_values),
        }
    }

    fn as_const_ptr(&self) -> *const ort_sys::OrtValue {
        self.ptr.cast_const()
    }
}

#[cfg(feature = "onnx-runtime")]
impl Drop for OrtValueGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ort_api().ReleaseValue.unwrap()(self.ptr);
            }
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_float_input_value(
    input: &OrtIoInfo,
    values: &[f32],
    shape: &[i64],
) -> anyhow::Result<OrtValueGuard> {
    let normalized_shape = normalize_float_input_shape(&input.name, &input.dimensions, shape)?;
    let element_count = normalized_shape.iter().product::<usize>();
    if element_count != values.len() {
        anyhow::bail!(
            "OCR ONNX input `{}` expected {} values from shape {:?}, got {}",
            input.name,
            element_count,
            normalized_shape,
            values.len()
        );
    }

    let shape_i64 = normalized_shape
        .iter()
        .map(|dimension| *dimension as i64)
        .collect::<Vec<_>>();
    let mut owned_values = values.to_vec();
    let mut memory_info_ptr = ptr::null_mut();
    ort_status_to_result(
        unsafe {
            ort_api().CreateCpuMemoryInfo.unwrap()(
                ort_sys::OrtAllocatorType_OrtArenaAllocator,
                ort_sys::OrtMemType_OrtMemTypeDefault,
                &mut memory_info_ptr,
            )
        },
        "failed to create OCR ONNX CPU memory info for float input",
    )?;

    let mut value_ptr = ptr::null_mut();
    let create_result = ort_status_to_result(
        unsafe {
            ort_api().CreateTensorWithDataAsOrtValue.unwrap()(
                memory_info_ptr,
                owned_values.as_mut_ptr().cast::<c_void>(),
                owned_values.len() * std::mem::size_of::<f32>(),
                shape_i64.as_ptr(),
                shape_i64.len(),
                ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &mut value_ptr,
            )
        },
        &format!(
            "failed to create OCR ONNX float tensor for input `{}`",
            input.name
        ),
    );
    unsafe {
        ort_api().ReleaseMemoryInfo.unwrap()(memory_info_ptr);
    }
    create_result?;

    Ok(OrtValueGuard::with_float_values(value_ptr, owned_values))
}

#[cfg(feature = "onnx-runtime")]
fn read_float_tensor_output(
    output: &OrtIoInfo,
    value_ptr: *mut ort_sys::OrtValue,
) -> anyhow::Result<OcrTensorOutput> {
    if output.tensor_type != ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    {
        anyhow::bail!(
            "OCR ONNX output `{}` uses unsupported tensor type `{}`",
            output.name,
            output.tensor_type
        );
    }

    let mut tensor_info_ptr = ptr::null_mut();
    ort_status_to_result(
        unsafe { ort_api().GetTensorTypeAndShape.unwrap()(value_ptr, &mut tensor_info_ptr) },
        "failed to read OCR ONNX output tensor shape",
    )?;
    let dimensions = extract_dimensions(tensor_info_ptr)?;
    unsafe {
        ort_api().ReleaseTensorTypeAndShapeInfo.unwrap()(tensor_info_ptr);
    }

    let shape = dimensions
        .into_iter()
        .map(|dimension| {
            dimension.ok_or_else(|| {
                anyhow::anyhow!("OCR ONNX output `{}` contains a dynamic shape", output.name)
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .map(|dimension| dimension as usize)
        .collect::<Vec<_>>();
    let element_count = shape.iter().product::<usize>().max(1);

    let mut data_ptr = ptr::null_mut();
    ort_status_to_result(
        unsafe { ort_api().GetTensorMutableData.unwrap()(value_ptr, &mut data_ptr) },
        &format!("failed to access OCR ONNX output tensor `{}`", output.name),
    )?;
    let values =
        unsafe { std::slice::from_raw_parts(data_ptr.cast::<f32>(), element_count).to_vec() };

    Ok(OcrTensorOutput {
        name: output.name.clone(),
        shape,
        values,
    })
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn decode_detection_output(
    output: &OcrTensorOutput,
    raster: &OcrPreparedRaster,
    threshold: f32,
    min_component_pixels: usize,
) -> anyhow::Result<OcrDetectionResult> {
    let (heatmap_height, heatmap_width, heatmap_values) =
        normalize_detection_heatmap(&output.shape, &output.values)?;
    let candidates = extract_detection_candidates(
        heatmap_width,
        heatmap_height,
        heatmap_values,
        raster,
        threshold,
        min_component_pixels,
    );

    Ok(OcrDetectionResult {
        output_name: output.name.clone(),
        heatmap_shape: vec![heatmap_height, heatmap_width],
        candidates,
    })
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn normalize_detection_heatmap<'a>(
    shape: &[usize],
    values: &'a [f32],
) -> anyhow::Result<(usize, usize, &'a [f32])> {
    match shape {
        [height, width] => {
            if values.len() != height.saturating_mul(*width) {
                anyhow::bail!(
                    "OCR detection heatmap shape {:?} does not match value count {}",
                    shape,
                    values.len()
                );
            }
            Ok((*height, *width, values))
        }
        [channels, height, width] if *channels == 1 => {
            let plane_size = height.saturating_mul(*width);
            if values.len() < plane_size {
                anyhow::bail!(
                    "OCR detection heatmap shape {:?} requires at least {} values, got {}",
                    shape,
                    plane_size,
                    values.len()
                );
            }
            Ok((*height, *width, &values[..plane_size]))
        }
        [batch, channels, height, width] if *batch >= 1 && *channels >= 1 => {
            let plane_size = height.saturating_mul(*width);
            if values.len() < plane_size {
                anyhow::bail!(
                    "OCR detection heatmap shape {:?} requires at least {} values, got {}",
                    shape,
                    plane_size,
                    values.len()
                );
            }
            Ok((*height, *width, &values[..plane_size]))
        }
        other => anyhow::bail!(
            "unsupported OCR detection output shape {:?}; expected [H,W], [1,H,W], or [1,1,H,W]",
            other
        ),
    }
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn extract_detection_candidates(
    heatmap_width: usize,
    heatmap_height: usize,
    heatmap_values: &[f32],
    raster: &OcrPreparedRaster,
    threshold: f32,
    min_component_pixels: usize,
) -> Vec<OcrDetectionCandidate> {
    if heatmap_width == 0 || heatmap_height == 0 || heatmap_values.is_empty() {
        return Vec::new();
    }

    let mut visited = vec![false; heatmap_values.len()];
    let mut candidates = Vec::new();

    for row in 0..heatmap_height {
        for col in 0..heatmap_width {
            let start = row * heatmap_width + col;
            if visited[start] || heatmap_values[start] < threshold {
                continue;
            }

            let mut queue = VecDeque::from([(row, col)]);
            visited[start] = true;
            let mut pixel_count = 0_usize;
            let mut score_sum = 0.0_f32;
            let mut min_row = row;
            let mut max_row = row;
            let mut min_col = col;
            let mut max_col = col;

            while let Some((current_row, current_col)) = queue.pop_front() {
                let index = current_row * heatmap_width + current_col;
                let score = heatmap_values[index];
                if score < threshold {
                    continue;
                }

                pixel_count += 1;
                score_sum += score;
                min_row = min_row.min(current_row);
                max_row = max_row.max(current_row);
                min_col = min_col.min(current_col);
                max_col = max_col.max(current_col);

                for (next_row, next_col) in
                    connected_neighbors(current_row, current_col, heatmap_height, heatmap_width)
                {
                    let next_index = next_row * heatmap_width + next_col;
                    if visited[next_index] || heatmap_values[next_index] < threshold {
                        continue;
                    }
                    visited[next_index] = true;
                    queue.push_back((next_row, next_col));
                }
            }

            if pixel_count < min_component_pixels {
                continue;
            }

            let x_scale = raster.resized_width as f32 / heatmap_width as f32;
            let y_scale = raster.resized_height as f32 / heatmap_height as f32;
            let x1 = (min_col as f32) * x_scale * raster.scale_x;
            let y1 = (min_row as f32) * y_scale * raster.scale_y;
            let x2 = ((max_col + 1) as f32) * x_scale * raster.scale_x;
            let y2 = ((max_row + 1) as f32) * y_scale * raster.scale_y;

            candidates.push(OcrDetectionCandidate {
                bbox: BBox { x1, y1, x2, y2 },
                score: score_sum / pixel_count as f32,
            });
        }
    }

    candidates.sort_by(|left, right| {
        left.bbox
            .y1
            .partial_cmp(&right.bbox.y1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.bbox
                    .x1
                    .partial_cmp(&right.bbox.x1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    candidates
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn connected_neighbors(
    row: usize,
    col: usize,
    height: usize,
    width: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let up = row.checked_sub(1).map(|next_row| (next_row, col));
    let left = col.checked_sub(1).map(|next_col| (row, next_col));
    let down = (row + 1 < height).then_some((row + 1, col));
    let right = (col + 1 < width).then_some((row, col + 1));
    [up, left, down, right].into_iter().flatten()
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
#[derive(Clone, Debug)]
struct OcrTensorOutput {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
#[derive(Clone, Debug)]
struct OcrDetectionCandidate {
    bbox: BBox,
    score: f32,
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct OcrDetectionResult {
    output_name: String,
    heatmap_shape: Vec<usize>,
    candidates: Vec<OcrDetectionCandidate>,
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct OcrRecognitionPatch {
    bbox: BBox,
    width: u32,
    height: u32,
    channels: u32,
    chw_f32: Vec<f32>,
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct OcrRecognitionOutputSummary {
    patch_index: usize,
    output_name: String,
    shape: Vec<usize>,
    value_count: usize,
    max_value: Option<f32>,
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct OcrRecognitionRun {
    patch_count: usize,
    executed_patches: usize,
    predictions: Vec<OcrRecognitionPrediction>,
    outputs: Vec<OcrRecognitionOutputSummary>,
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
#[derive(Clone, Debug)]
struct OcrRecognitionPrediction {
    patch_index: usize,
    text: String,
    confidence: f32,
}

#[derive(Clone, Debug)]
struct OcrCharset {
    tokens: Vec<String>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct TestLocalOcrCandidate {
    pub bbox: BBox,
    pub score: f32,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct TestLocalOcrPrediction {
    pub patch_index: usize,
    pub text: String,
    pub confidence: f32,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct TestLocalOcrFixture {
    pub model_summary: String,
    pub charset: Vec<String>,
    pub candidates: Vec<TestLocalOcrCandidate>,
    pub predictions: Vec<TestLocalOcrPrediction>,
}

#[cfg(test)]
struct StaticLocalOcrBackend {
    detection: OcrDetectionResult,
    recognition: OcrRecognitionRun,
}

#[cfg(test)]
impl LocalOcrBackend for StaticLocalOcrBackend {
    fn prewarm(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn run_detection(&self, _page: &OcrPreparedPage) -> anyhow::Result<OcrDetectionResult> {
        Ok(self.detection.clone())
    }

    fn run_recognition(
        &self,
        _patches: &[OcrRecognitionPatch],
        _charset: &OcrCharset,
    ) -> anyhow::Result<OcrRecognitionRun> {
        Ok(self.recognition.clone())
    }
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn build_recognition_patches(
    raster: &OcrPreparedRaster,
    candidates: &[OcrDetectionCandidate],
    target_height: u32,
    max_width: u32,
) -> Vec<OcrRecognitionPatch> {
    if target_height == 0 || max_width == 0 {
        return Vec::new();
    }

    candidates
        .iter()
        .filter_map(|candidate| {
            build_recognition_patch(raster, candidate, target_height, max_width).ok()
        })
        .collect()
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn build_recognition_patch(
    raster: &OcrPreparedRaster,
    candidate: &OcrDetectionCandidate,
    target_height: u32,
    max_width: u32,
) -> anyhow::Result<OcrRecognitionPatch> {
    let (x1, y1, x2, y2) = project_bbox_to_resized_image(&candidate.bbox, raster)?;
    let crop_width = x2.saturating_sub(x1);
    let crop_height = y2.saturating_sub(y1);
    if crop_width == 0 || crop_height == 0 {
        anyhow::bail!("OCR recognition crop has zero-sized bounds");
    }

    let crop_rgb = crop_rgb_region(
        &raster.resized_rgb_u8,
        raster.resized_width,
        raster.resized_height,
        x1,
        y1,
        crop_width,
        crop_height,
    )?;
    let crop_image = RgbImage::from_raw(crop_width, crop_height, crop_rgb)
        .ok_or_else(|| anyhow::anyhow!("failed to build OCR crop image"))?;

    let target_width = ((crop_width as f32 / crop_height as f32) * target_height as f32)
        .round()
        .clamp(1.0, max_width as f32) as u32;
    let resized = image::imageops::resize(
        &crop_image,
        target_width,
        target_height,
        FilterType::Triangle,
    );
    let chw_f32 = rgb_to_chw_f32(resized.as_raw());

    Ok(OcrRecognitionPatch {
        bbox: candidate.bbox.clone(),
        width: target_width,
        height: target_height,
        channels: 3,
        chw_f32,
    })
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn project_bbox_to_resized_image(
    bbox: &BBox,
    raster: &OcrPreparedRaster,
) -> anyhow::Result<(u32, u32, u32, u32)> {
    let max_x = raster.resized_width as f32;
    let max_y = raster.resized_height as f32;
    let x1 = (bbox.x1 / raster.scale_x).floor().clamp(0.0, max_x - 1.0) as u32;
    let y1 = (bbox.y1 / raster.scale_y).floor().clamp(0.0, max_y - 1.0) as u32;
    let x2 = (bbox.x2 / raster.scale_x).ceil().clamp(1.0, max_x) as u32;
    let y2 = (bbox.y2 / raster.scale_y).ceil().clamp(1.0, max_y) as u32;

    if x2 <= x1 || y2 <= y1 {
        anyhow::bail!("OCR bbox cannot be projected into a non-empty resized crop");
    }

    Ok((x1, y1, x2, y2))
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn crop_rgb_region(
    rgb: &[u8],
    image_width: u32,
    image_height: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> anyhow::Result<Vec<u8>> {
    if width == 0 || height == 0 || image_width == 0 || image_height == 0 {
        anyhow::bail!("OCR crop dimensions must be greater than zero");
    }
    if x.saturating_add(width) > image_width || y.saturating_add(height) > image_height {
        anyhow::bail!("OCR crop region is outside the resized image bounds");
    }

    let channels = 3_usize;
    let image_width = image_width as usize;
    let x = x as usize;
    let y = y as usize;
    let width = width as usize;
    let height = height as usize;
    let mut cropped = vec![0_u8; width * height * channels];

    for row in 0..height {
        let src_offset = ((y + row) * image_width + x) * channels;
        let dst_offset = row * width * channels;
        let copy_len = width * channels;
        cropped[dst_offset..dst_offset + copy_len]
            .copy_from_slice(&rgb[src_offset..src_offset + copy_len]);
    }

    Ok(cropped)
}

fn rgb_to_chw_f32(rgb: &[u8]) -> Vec<f32> {
    let pixel_count = rgb.len() / 3;
    let mut chw_f32 = vec![0.0_f32; pixel_count * 3];

    for (index, pixel) in rgb.chunks_exact(3).enumerate() {
        chw_f32[index] = f32::from(pixel[0]) / 255.0;
        chw_f32[pixel_count + index] = f32::from(pixel[1]) / 255.0;
        chw_f32[(pixel_count * 2) + index] = f32::from(pixel[2]) / 255.0;
    }

    chw_f32
}

fn load_ocr_charset(path: &Path) -> anyhow::Result<OcrCharset> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read OCR charset file `{}`", path.display()))?;
    let tokens = raw
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = if index == 0 {
                line.strip_prefix('\u{feff}').unwrap_or(line)
            } else {
                line
            };
            if line.is_empty() {
                None
            } else {
                Some(line.to_string())
            }
        })
        .collect::<Vec<_>>();

    if tokens.is_empty() {
        anyhow::bail!("OCR charset file `{}` is empty", path.display());
    }

    Ok(OcrCharset { tokens })
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn assemble_recognized_lines(
    page_no: u32,
    detection: &OcrDetectionResult,
    recognition: &OcrRecognitionRun,
) -> anyhow::Result<Vec<RawOcrLine>> {
    let mut lines = Vec::new();

    for prediction in &recognition.predictions {
        let candidate = detection
            .candidates
            .get(prediction.patch_index)
            .ok_or_else(|| anyhow::anyhow!("OCR recognition prediction index is out of bounds"))?;
        let text = prediction.text.trim().to_string();
        if text.is_empty() {
            continue;
        }

        lines.push(RawOcrLine {
            text,
            page_no: Some(page_no),
            bbox: Some(candidate.bbox.clone()),
            confidence: Some(prediction.confidence.min(candidate.score)),
        });
    }

    Ok(lines)
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn decode_recognition_output(
    output: &OcrTensorOutput,
    charset: &OcrCharset,
) -> anyhow::Result<(String, f32)> {
    let (timesteps, classes, values) = normalize_recognition_logits(&output.shape, &output.values)?;
    let blank_index = infer_ctc_blank_index(classes, charset)?;
    let mut text = String::new();
    let mut previous_index = None;
    let mut confidence_sum = 0.0_f32;
    let mut confidence_count = 0_usize;

    for timestep in 0..timesteps {
        let start = timestep * classes;
        let slice = &values[start..start + classes];
        let (class_index, class_score) = slice
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow::anyhow!("OCR recognition timestep is empty"))?;

        if Some(class_index) == previous_index {
            continue;
        }
        previous_index = Some(class_index);

        if class_index == blank_index {
            continue;
        }

        let token = recognition_token_from_index(class_index, blank_index, charset)?;
        text.push_str(token);
        confidence_sum += class_score;
        confidence_count += 1;
    }

    let confidence = if confidence_count > 0 {
        confidence_sum / confidence_count as f32
    } else {
        0.0
    };
    Ok((text, confidence))
}

#[cfg_attr(not(any(test, feature = "onnx-runtime")), allow(dead_code))]
fn normalize_recognition_logits<'a>(
    shape: &[usize],
    values: &'a [f32],
) -> anyhow::Result<(usize, usize, &'a [f32])> {
    match shape {
        [timesteps, classes] => {
            if values.len() != timesteps.saturating_mul(*classes) {
                anyhow::bail!(
                    "OCR recognition logits shape {:?} does not match value count {}",
                    shape,
                    values.len()
                );
            }
            Ok((*timesteps, *classes, values))
        }
        [batch, timesteps, classes] if *batch >= 1 => {
            let plane_size = timesteps.saturating_mul(*classes);
            if values.len() < plane_size {
                anyhow::bail!(
                    "OCR recognition logits shape {:?} requires at least {} values, got {}",
                    shape,
                    plane_size,
                    values.len()
                );
            }
            Ok((*timesteps, *classes, &values[..plane_size]))
        }
        [batch, channels, timesteps, classes] if *batch >= 1 && *channels >= 1 => {
            let plane_size = timesteps.saturating_mul(*classes);
            if values.len() < plane_size {
                anyhow::bail!(
                    "OCR recognition logits shape {:?} requires at least {} values, got {}",
                    shape,
                    plane_size,
                    values.len()
                );
            }
            Ok((*timesteps, *classes, &values[..plane_size]))
        }
        other => anyhow::bail!(
            "unsupported OCR recognition output shape {:?}; expected [T,C], [1,T,C], or [1,1,T,C]",
            other
        ),
    }
}

fn infer_ctc_blank_index(classes: usize, charset: &OcrCharset) -> anyhow::Result<usize> {
    if charset.tokens.len() + 1 == classes {
        return Ok(0);
    }
    if charset.tokens.len() == classes {
        return Ok(classes.saturating_sub(1));
    }

    anyhow::bail!(
        "OCR recognition classes ({classes}) do not match charset size ({}) for current CTC decoder",
        charset.tokens.len()
    )
}

fn recognition_token_from_index<'a>(
    class_index: usize,
    blank_index: usize,
    charset: &'a OcrCharset,
) -> anyhow::Result<&'a str> {
    let token_index = if blank_index == 0 {
        class_index
            .checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("blank index should not be decoded into a token"))?
    } else {
        class_index
    };
    charset
        .tokens
        .get(token_index)
        .map(String::as_str)
        .ok_or_else(|| {
            anyhow::anyhow!("OCR recognition token index `{token_index}` is out of charset bounds")
        })
}

#[cfg(feature = "onnx-runtime")]
struct OcrOnnxSession {
    _model_path: PathBuf,
    env_ptr: *mut ort_sys::OrtEnv,
    session_ptr: *mut ort_sys::OrtSession,
    _allocator_ptr: *mut ort_sys::OrtAllocator,
    inputs: Vec<OrtIoInfo>,
    outputs: Vec<OrtIoInfo>,
}

#[cfg(feature = "onnx-runtime")]
unsafe impl Send for OcrOnnxSession {}

#[cfg(feature = "onnx-runtime")]
unsafe impl Sync for OcrOnnxSession {}

#[cfg(feature = "onnx-runtime")]
impl Drop for OcrOnnxSession {
    fn drop(&mut self) {
        unsafe {
            if !self.session_ptr.is_null() {
                ort_api().ReleaseSession.unwrap()(self.session_ptr);
                self.session_ptr = ptr::null_mut();
            }

            if !self.env_ptr.is_null() {
                ort_api().ReleaseEnv.unwrap()(self.env_ptr);
                self.env_ptr = ptr::null_mut();
            }
        }
    }
}

#[cfg(feature = "onnx-runtime")]
impl OcrOnnxSession {
    fn new(model_path: PathBuf, intra_threads: usize, env_name: &str) -> anyhow::Result<Self> {
        if !model_path.is_file() {
            anyhow::bail!(
                "OCR ONNX model file was not found: {}",
                model_path.display()
            );
        }

        let env_name = CString::new(env_name)?;
        let mut env_ptr = ptr::null_mut();
        ort_status_to_result(
            unsafe {
                ort_api().CreateEnv.unwrap()(
                    ort_sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
                    env_name.as_ptr(),
                    &mut env_ptr,
                )
            },
            "failed to create OCR ONNX Runtime environment",
        )?;

        let mut session_options_ptr = ptr::null_mut();
        let result = (|| -> anyhow::Result<Self> {
            ort_status_to_result(
                unsafe { ort_api().CreateSessionOptions.unwrap()(&mut session_options_ptr) },
                "failed to create OCR ONNX Runtime session options",
            )?;
            ort_status_to_result(
                unsafe {
                    ort_api().SetSessionGraphOptimizationLevel.unwrap()(
                        session_options_ptr,
                        ort_sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
                    )
                },
                "failed to configure OCR ONNX Runtime graph optimization level",
            )?;
            ort_status_to_result(
                unsafe {
                    ort_api().SetIntraOpNumThreads.unwrap()(
                        session_options_ptr,
                        i32::try_from(intra_threads).unwrap_or(1),
                    )
                },
                "failed to configure OCR ONNX Runtime intra-op threads",
            )?;

            let path_bytes = model_path_bytes(&model_path)?;
            let mut session_ptr = ptr::null_mut();
            ort_status_to_result(
                unsafe {
                    ort_api().CreateSession.unwrap()(
                        env_ptr,
                        path_bytes.as_ptr(),
                        session_options_ptr,
                        &mut session_ptr,
                    )
                },
                "failed to create OCR ONNX Runtime session",
            )?;

            let mut allocator_ptr = ptr::null_mut();
            ort_status_to_result(
                unsafe { ort_api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) },
                "failed to get OCR ONNX Runtime default allocator",
            )?;

            let mut input_count = 0;
            ort_status_to_result(
                unsafe { ort_api().SessionGetInputCount.unwrap()(session_ptr, &mut input_count) },
                "failed to read OCR ONNX input count",
            )?;
            let mut output_count = 0;
            ort_status_to_result(
                unsafe { ort_api().SessionGetOutputCount.unwrap()(session_ptr, &mut output_count) },
                "failed to read OCR ONNX output count",
            )?;

            let inputs = extract_io_info(
                session_ptr,
                allocator_ptr,
                input_count,
                ort_api().SessionGetInputName.unwrap(),
                ort_api().SessionGetInputTypeInfo.unwrap(),
            )?;
            let outputs = extract_io_info(
                session_ptr,
                allocator_ptr,
                output_count,
                ort_api().SessionGetOutputName.unwrap(),
                ort_api().SessionGetOutputTypeInfo.unwrap(),
            )?;

            Ok(Self {
                _model_path: model_path,
                env_ptr,
                session_ptr,
                _allocator_ptr: allocator_ptr,
                inputs,
                outputs,
            })
        })();

        if !session_options_ptr.is_null() {
            unsafe {
                ort_api().ReleaseSessionOptions.unwrap()(session_options_ptr);
            }
        }

        if result.is_err() && !env_ptr.is_null() {
            unsafe {
                ort_api().ReleaseEnv.unwrap()(env_ptr);
            }
        }

        result
    }

    fn validate(&self, stage_name: &str) -> anyhow::Result<()> {
        if self.inputs.is_empty() {
            anyhow::bail!("OCR ONNX {stage_name} session exposed no inputs");
        }
        if self.outputs.is_empty() {
            anyhow::bail!("OCR ONNX {stage_name} session exposed no outputs");
        }
        if self.inputs.iter().any(|input| input.name.trim().is_empty()) {
            anyhow::bail!("OCR ONNX {stage_name} session exposed an unnamed input");
        }
        if self
            .outputs
            .iter()
            .any(|output| output.name.trim().is_empty())
        {
            anyhow::bail!("OCR ONNX {stage_name} session exposed an unnamed output");
        }
        if self
            .inputs
            .iter()
            .chain(self.outputs.iter())
            .any(|io| io.dimensions.is_empty())
        {
            anyhow::bail!(
                "OCR ONNX {stage_name} session exposed a scalar tensor, which is unexpected for OCR models"
            );
        }
        if self.inputs.iter().chain(self.outputs.iter()).any(|io| {
            io.tensor_type
                == ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        }) {
            anyhow::bail!("OCR ONNX {stage_name} session exposed an undefined tensor element type");
        }
        Ok(())
    }

    fn find_primary_image_input(&self, stage_name: &str) -> anyhow::Result<&OrtIoInfo> {
        self.inputs
            .iter()
            .find(|input| {
                input.tensor_type
                    == ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    && input.dimensions.len() == 4
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "OCR ONNX {stage_name} session does not expose a float rank-4 image input"
                )
            })
    }

    fn run_single_float_input(
        &self,
        input_name: &str,
        values: &[f32],
        shape: &[i64],
    ) -> anyhow::Result<Vec<OcrTensorOutput>> {
        if self.inputs.len() != 1 {
            anyhow::bail!(
                "OCR ONNX session currently supports single-input models only; found {} inputs",
                self.inputs.len()
            );
        }

        let input = self
            .inputs
            .iter()
            .find(|candidate| candidate.name == input_name)
            .ok_or_else(|| anyhow::anyhow!("OCR ONNX input `{input_name}` was not found"))?;
        let input_value = build_float_input_value(input, values, shape)?;
        let input_name_cstring = CString::new(input.name.as_str())?;
        let input_name_ptrs = [input_name_cstring.as_ptr()];
        let input_value_ptrs = [input_value.as_const_ptr()];

        let output_name_cstrings = self
            .outputs
            .iter()
            .map(|output| CString::new(output.name.as_str()))
            .collect::<Result<Vec<_>, _>>()?;
        let output_name_ptrs = output_name_cstrings
            .iter()
            .map(|output_name| output_name.as_ptr())
            .collect::<Vec<_>>();
        let mut output_value_ptrs = vec![ptr::null_mut(); output_name_ptrs.len()];

        ort_status_to_result(
            unsafe {
                ort_api().Run.unwrap()(
                    self.session_ptr,
                    ptr::null(),
                    input_name_ptrs.as_ptr(),
                    input_value_ptrs.as_ptr(),
                    input_value_ptrs.len(),
                    output_name_ptrs.as_ptr(),
                    output_name_ptrs.len(),
                    output_value_ptrs.as_mut_ptr(),
                )
            },
            "failed to execute OCR ONNX Runtime session",
        )?;

        let output_values = output_value_ptrs
            .into_iter()
            .map(OrtValueGuard::new)
            .collect::<Vec<_>>();
        self.outputs
            .iter()
            .zip(output_values.iter())
            .map(|(output, value)| read_float_tensor_output(output, value.ptr))
            .collect()
    }
}

#[cfg(feature = "onnx-runtime")]
struct LocalOnnxRuntimeBackend {
    detection: OcrOnnxSession,
    recognition: OcrOnnxSession,
    _classification: Option<OcrOnnxSession>,
}

#[cfg(feature = "onnx-runtime")]
impl LocalOnnxRuntimeBackend {
    fn new(model_files: &LocalOnnxModelFiles, threads: usize) -> anyhow::Result<Self> {
        let detection = OcrOnnxSession::new(
            model_files.det_model_path.clone(),
            threads,
            "muse-ocr-detection",
        )
        .context("failed to bootstrap local OCR detection model")?;
        let recognition = OcrOnnxSession::new(
            model_files.rec_model_path.clone(),
            threads,
            "muse-ocr-recognition",
        )
        .context("failed to bootstrap local OCR recognition model")?;
        let classification = model_files
            .cls_model_path
            .as_ref()
            .map(|path| {
                OcrOnnxSession::new(path.clone(), threads, "muse-ocr-classification")
                    .context("failed to bootstrap local OCR classification model")
            })
            .transpose()?;

        Ok(Self {
            detection,
            recognition,
            _classification: classification,
        })
    }

    fn prewarm(&self) -> anyhow::Result<()> {
        self.detection.validate("detection")?;
        self.recognition.validate("recognition")?;
        if let Some(classification) = &self._classification {
            classification.validate("classification")?;
        }
        Ok(())
    }

    fn run_detection(&self, page: &OcrPreparedPage) -> anyhow::Result<OcrDetectionResult> {
        let raster = page
            .raster
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("local OCR page raster is missing"))?;
        let input = self.detection.find_primary_image_input("detection")?;
        let outputs = self.detection.run_single_float_input(
            &input.name,
            &raster.chw_f32,
            &[
                1,
                raster.channels as i64,
                raster.resized_height as i64,
                raster.resized_width as i64,
            ],
        )?;
        let best_output = outputs
            .iter()
            .max_by_key(|output| output.values.len())
            .ok_or_else(|| anyhow::anyhow!("OCR detection session returned no outputs"))?;
        decode_detection_output(
            best_output,
            raster,
            LOCAL_ONNX_DETECTION_THRESHOLD,
            LOCAL_ONNX_MIN_COMPONENT_PIXELS,
        )
    }

    fn run_recognition(
        &self,
        patches: &[OcrRecognitionPatch],
        charset: &OcrCharset,
    ) -> anyhow::Result<OcrRecognitionRun> {
        if patches.is_empty() {
            return Ok(OcrRecognitionRun {
                patch_count: 0,
                executed_patches: 0,
                predictions: Vec::new(),
                outputs: Vec::new(),
            });
        }

        let input = self.recognition.find_primary_image_input("recognition")?;
        let executed_patches = patches.len();
        let mut predictions = Vec::new();
        let mut outputs = Vec::new();

        for (patch_index, patch) in patches.iter().enumerate() {
            let patch_outputs = self.recognition.run_single_float_input(
                &input.name,
                &patch.chw_f32,
                &[
                    1,
                    patch.channels as i64,
                    patch.height as i64,
                    patch.width as i64,
                ],
            )?;
            let best_output = patch_outputs
                .iter()
                .max_by_key(|output| output.values.len())
                .ok_or_else(|| anyhow::anyhow!("OCR recognition session returned no outputs"))?;
            let (text, confidence) = decode_recognition_output(best_output, charset)?;
            predictions.push(OcrRecognitionPrediction {
                patch_index,
                text,
                confidence,
            });
            outputs.extend(
                patch_outputs
                    .into_iter()
                    .map(|output| OcrRecognitionOutputSummary {
                        patch_index,
                        output_name: output.name,
                        shape: output.shape,
                        value_count: output.values.len(),
                        max_value: output.values.iter().copied().reduce(f32::max),
                    }),
            );
        }

        Ok(OcrRecognitionRun {
            patch_count: patches.len(),
            executed_patches,
            predictions,
            outputs,
        })
    }
}

#[cfg(feature = "onnx-runtime")]
impl LocalOcrBackend for LocalOnnxRuntimeBackend {
    fn prewarm(&self) -> anyhow::Result<()> {
        Self::prewarm(self)
    }

    fn run_detection(&self, page: &OcrPreparedPage) -> anyhow::Result<OcrDetectionResult> {
        Self::run_detection(self, page)
    }

    fn run_recognition(
        &self,
        patches: &[OcrRecognitionPatch],
        charset: &OcrCharset,
    ) -> anyhow::Result<OcrRecognitionRun> {
        Self::run_recognition(self, patches, charset)
    }
}

#[cfg(not(feature = "onnx-runtime"))]
struct LocalOnnxRuntimeBackend;

#[cfg(not(feature = "onnx-runtime"))]
impl LocalOnnxRuntimeBackend {
    fn new(_model_files: &LocalOnnxModelFiles, _threads: usize) -> anyhow::Result<Self> {
        anyhow::bail!(
            "local ONNX OCR requires the `onnx-runtime` cargo feature to be enabled at build time"
        )
    }

    fn prewarm(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn run_detection(&self, _page: &OcrPreparedPage) -> anyhow::Result<OcrDetectionResult> {
        anyhow::bail!(
            "local ONNX OCR requires the `onnx-runtime` cargo feature to be enabled at build time"
        )
    }

    fn run_recognition(
        &self,
        _patches: &[OcrRecognitionPatch],
        _charset: &OcrCharset,
    ) -> anyhow::Result<OcrRecognitionRun> {
        anyhow::bail!(
            "local ONNX OCR requires the `onnx-runtime` cargo feature to be enabled at build time"
        )
    }
}

#[cfg(not(feature = "onnx-runtime"))]
impl LocalOcrBackend for LocalOnnxRuntimeBackend {
    fn prewarm(&self) -> anyhow::Result<()> {
        Self::prewarm(self)
    }

    fn run_detection(&self, page: &OcrPreparedPage) -> anyhow::Result<OcrDetectionResult> {
        Self::run_detection(self, page)
    }

    fn run_recognition(
        &self,
        patches: &[OcrRecognitionPatch],
        charset: &OcrCharset,
    ) -> anyhow::Result<OcrRecognitionRun> {
        Self::run_recognition(self, patches, charset)
    }
}

#[derive(Debug, Deserialize)]
struct WorkerOcrResponse {
    #[serde(default)]
    lines: Vec<RawOcrLine>,
    provider: Option<String>,
    model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::{
        Json, Router,
        body::Bytes,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
    };
    use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};
    use serde::Deserialize;
    use serde_json::json;
    use std::{fs, sync::Arc};
    use tokio::{net::TcpListener, sync::Mutex};

    #[derive(Debug, Deserialize)]
    struct OcrFixture {
        raster: OcrFixtureRaster,
        detection: OcrFixtureTensorOutput,
        charset: Vec<String>,
        recognition_outputs: Vec<OcrFixtureTensorOutput>,
        expected_lines: Vec<OcrFixtureExpectedLine>,
    }

    #[derive(Debug, Deserialize)]
    struct OcrFixtureRaster {
        original_width: u32,
        original_height: u32,
        resized_width: u32,
        resized_height: u32,
        scale_x: f32,
        scale_y: f32,
        fill_rgb_value: u8,
    }

    #[derive(Debug, Deserialize)]
    struct OcrFixtureTensorOutput {
        name: String,
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct OcrFixtureExpectedLine {
        text: String,
        page_no: u32,
        confidence: f32,
        bbox: BBox,
    }

    struct AlwaysFailOcrProvider;

    struct StaticOcrProvider {
        provider_name: &'static str,
        transport_name: &'static str,
        output: OcrOutput,
    }

    #[async_trait]
    impl OcrProvider for AlwaysFailOcrProvider {
        fn name(&self) -> &'static str {
            "always-fail-ocr"
        }

        fn transport_name(&self) -> &'static str {
            "inproc"
        }

        async fn recognize(&self, _request: OcrRequest) -> anyhow::Result<OcrOutput> {
            anyhow::bail!("simulated OCR failure")
        }
    }

    #[async_trait]
    impl OcrProvider for StaticOcrProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }

        fn transport_name(&self) -> &'static str {
            self.transport_name
        }

        async fn recognize(&self, _request: OcrRequest) -> anyhow::Result<OcrOutput> {
            Ok(self.output.clone())
        }
    }

    #[test]
    fn passthrough_preprocessor_preserves_original_bytes_and_single_page() {
        let prepared = PassthroughOcrPreprocessor
            .prepare(OcrRequest {
                file_name: Some("sample.png".to_string()),
                mime_type: Some("image/png".to_string()),
                bytes: b"image-bytes".to_vec(),
            })
            .expect("prepared input");

        assert_eq!(prepared.pages.len(), 1);
        assert_eq!(prepared.pages[0].page_no, 1);
        assert_eq!(prepared.original_bytes, b"image-bytes".to_vec());
        assert_eq!(prepared.pages[0].bytes, b"image-bytes".to_vec());
        assert!(prepared.pages[0].raster.is_none());
    }

    #[tokio::test]
    async fn fallback_provider_uses_secondary_when_primary_fails() {
        let provider = FallbackOcrProvider::new(
            Arc::new(AlwaysFailOcrProvider),
            Arc::new(StaticOcrProvider {
                provider_name: "secondary-ocr",
                transport_name: "http",
                output: OcrOutput {
                    lines: vec![OcrLine {
                        text: "岗位类型：回退成功".to_string(),
                        page_no: Some(1),
                        bbox: None,
                        confidence: Some(0.77),
                    }],
                    provider: Some("secondary-ocr".to_string()),
                    model: Some("fallback-model".to_string()),
                },
            }),
        );

        let output = provider
            .recognize(OcrRequest {
                file_name: Some("sample.png".to_string()),
                mime_type: Some("image/png".to_string()),
                bytes: b"fake".to_vec(),
            })
            .await
            .expect("fallback should succeed");

        assert_eq!(provider.name(), "fallback-ocr");
        assert_eq!(provider.transport_name(), "composite");
        assert_eq!(output.lines.len(), 1);
        assert_eq!(output.lines[0].text, "岗位类型：回退成功");
        assert_eq!(output.provider.as_deref(), Some("secondary-ocr"));
        assert_eq!(output.model.as_deref(), Some("fallback-model"));
    }

    #[test]
    fn local_onnx_preprocessor_decodes_image_and_builds_chw_tensor() {
        let png = encode_test_png(2, 1, &[255, 0, 0, 0, 255, 0]);
        let prepared = LocalOnnxImagePreprocessor::default()
            .prepare(OcrRequest {
                file_name: Some("sample.png".to_string()),
                mime_type: Some("image/png".to_string()),
                bytes: png,
            })
            .expect("prepared input");

        let raster = prepared.pages[0].raster.as_ref().expect("raster");
        assert_eq!(raster.original_width, 2);
        assert_eq!(raster.original_height, 1);
        assert_eq!(raster.resized_width, 2);
        assert_eq!(raster.resized_height, 1);
        assert_eq!(raster.channels, 3);
        assert_eq!(raster.chw_f32.len(), 6);
        assert_eq!(raster.chw_f32, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn local_onnx_preprocessor_resizes_large_images_within_limit() {
        let png = encode_test_png(4000, 1000, &vec![64_u8; 4000 * 1000 * 3]);
        let preprocessor = LocalOnnxImagePreprocessor {
            max_image_side: 1024,
        };
        let prepared = preprocessor
            .prepare(OcrRequest {
                file_name: Some("large.png".to_string()),
                mime_type: Some("image/png".to_string()),
                bytes: png,
            })
            .expect("prepared input");

        let raster = prepared.pages[0].raster.as_ref().expect("raster");
        assert_eq!(raster.original_width, 4000);
        assert_eq!(raster.original_height, 1000);
        assert_eq!(raster.resized_width, 1024);
        assert_eq!(raster.resized_height, 256);
        assert_eq!(raster.chw_f32.len(), 1024 * 256 * 3);
    }

    #[test]
    fn local_onnx_preprocessor_rejects_invalid_image_bytes() {
        let error = LocalOnnxImagePreprocessor::default()
            .prepare(OcrRequest {
                file_name: Some("broken.png".to_string()),
                mime_type: Some("image/png".to_string()),
                bytes: b"not-an-image".to_vec(),
            })
            .expect_err("invalid image bytes should fail");

        assert!(error.to_string().contains("failed to decode image bytes"));
    }

    #[test]
    fn decodes_detection_heatmap_into_scaled_bounding_boxes() {
        let raster = OcrPreparedRaster {
            original_width: 200,
            original_height: 100,
            resized_width: 100,
            resized_height: 50,
            channels: 3,
            scale_x: 2.0,
            scale_y: 2.0,
            resized_rgb_u8: vec![0; 100 * 50 * 3],
            chw_f32: vec![],
        };
        let output = OcrTensorOutput {
            name: "maps".to_string(),
            shape: vec![1, 1, 5, 10],
            values: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.7, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.7, 0.9, 0.8, 0.0, 0.0, 0.0, 0.6, 0.7, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.8, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
        };

        let detection = decode_detection_output(
            &output,
            &raster,
            LOCAL_ONNX_DETECTION_THRESHOLD.max(0.5),
            LOCAL_ONNX_MIN_COMPONENT_PIXELS.min(2),
        )
        .expect("decode detection");

        assert_eq!(detection.output_name, "maps");
        assert_eq!(detection.heatmap_shape, vec![5, 10]);
        assert_eq!(detection.candidates.len(), 2);
        assert_eq!(
            detection.candidates[0].bbox,
            BBox {
                x1: 20.0,
                y1: 20.0,
                x2: 80.0,
                y2: 60.0
            }
        );
        assert_eq!(
            detection.candidates[1].bbox,
            BBox {
                x1: 140.0,
                y1: 40.0,
                x2: 180.0,
                y2: 80.0
            }
        );
        assert!(detection.candidates[0].score > 0.7);
        assert!(detection.candidates[1].score > 0.6);
    }

    #[test]
    fn rejects_unsupported_detection_output_shape() {
        let raster = OcrPreparedRaster {
            original_width: 10,
            original_height: 10,
            resized_width: 10,
            resized_height: 10,
            channels: 3,
            scale_x: 1.0,
            scale_y: 1.0,
            resized_rgb_u8: vec![0; 10 * 10 * 3],
            chw_f32: vec![],
        };
        let output = OcrTensorOutput {
            name: "maps".to_string(),
            shape: vec![1, 2, 3, 4, 5],
            values: vec![0.0; 120],
        };

        let error = decode_detection_output(&output, &raster, 0.5, 1)
            .expect_err("unexpected detection shape should fail");
        assert!(
            error
                .to_string()
                .contains("unsupported OCR detection output shape")
        );
    }

    #[test]
    fn builds_recognition_patches_from_detection_candidates() {
        let raster = OcrPreparedRaster {
            original_width: 200,
            original_height: 100,
            resized_width: 100,
            resized_height: 50,
            channels: 3,
            scale_x: 2.0,
            scale_y: 2.0,
            resized_rgb_u8: vec![200; 100 * 50 * 3],
            chw_f32: vec![],
        };
        let candidates = vec![OcrDetectionCandidate {
            bbox: BBox {
                x1: 20.0,
                y1: 20.0,
                x2: 80.0,
                y2: 60.0,
            },
            score: 0.9,
        }];

        let patches = build_recognition_patches(
            &raster,
            &candidates,
            LOCAL_ONNX_RECOGNITION_TARGET_HEIGHT,
            LOCAL_ONNX_RECOGNITION_MAX_WIDTH,
        );

        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].bbox, candidates[0].bbox);
        assert_eq!(patches[0].height, LOCAL_ONNX_RECOGNITION_TARGET_HEIGHT);
        assert_eq!(patches[0].width, 72);
        assert_eq!(patches[0].channels, 3);
        assert_eq!(
            patches[0].chw_f32.len(),
            (patches[0].width * patches[0].height * 3) as usize
        );
    }

    #[test]
    fn decodes_recognition_logits_with_ctc_greedy() {
        let charset = OcrCharset {
            tokens: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        };
        let output = OcrTensorOutput {
            name: "logits".to_string(),
            shape: vec![1, 5, 4],
            values: vec![
                0.9, 0.1, 0.0, 0.0, //
                0.1, 0.8, 0.1, 0.0, //
                0.1, 0.7, 0.2, 0.0, //
                0.8, 0.1, 0.0, 0.1, //
                0.1, 0.1, 0.7, 0.1, //
            ],
        };

        let (text, confidence) =
            decode_recognition_output(&output, &charset).expect("decode recognition");

        assert_eq!(text, "AB");
        assert!(confidence > 0.7);
    }

    #[test]
    fn fixture_driven_local_onnx_pipeline_regression() {
        let raw = fs::read_to_string("fixtures/ocr/local_onnx_pipeline_fixture.json")
            .expect("read fixture");
        let fixture: OcrFixture = serde_json::from_str(&raw).expect("parse fixture");

        let raster = OcrPreparedRaster {
            original_width: fixture.raster.original_width,
            original_height: fixture.raster.original_height,
            resized_width: fixture.raster.resized_width,
            resized_height: fixture.raster.resized_height,
            channels: 3,
            scale_x: fixture.raster.scale_x,
            scale_y: fixture.raster.scale_y,
            resized_rgb_u8: vec![
                fixture.raster.fill_rgb_value;
                (fixture.raster.resized_width * fixture.raster.resized_height * 3)
                    as usize
            ],
            chw_f32: vec![],
        };
        let detection_output = OcrTensorOutput {
            name: fixture.detection.name,
            shape: fixture.detection.shape,
            values: fixture.detection.values,
        };
        let detection = decode_detection_output(
            &detection_output,
            &raster,
            LOCAL_ONNX_DETECTION_THRESHOLD.max(0.5),
            LOCAL_ONNX_MIN_COMPONENT_PIXELS.min(2),
        )
        .expect("decode detection");
        let patches = build_recognition_patches(
            &raster,
            &detection.candidates,
            LOCAL_ONNX_RECOGNITION_TARGET_HEIGHT,
            LOCAL_ONNX_RECOGNITION_MAX_WIDTH,
        );
        assert_eq!(patches.len(), fixture.recognition_outputs.len());

        let charset = OcrCharset {
            tokens: fixture.charset,
        };
        let predictions = fixture
            .recognition_outputs
            .into_iter()
            .enumerate()
            .map(|(patch_index, output)| {
                let tensor = OcrTensorOutput {
                    name: output.name,
                    shape: output.shape,
                    values: output.values,
                };
                let (text, confidence) =
                    decode_recognition_output(&tensor, &charset).expect("decode recognition");
                OcrRecognitionPrediction {
                    patch_index,
                    text,
                    confidence,
                }
            })
            .collect::<Vec<_>>();
        let lines = assemble_recognized_lines(
            1,
            &detection,
            &OcrRecognitionRun {
                patch_count: patches.len(),
                executed_patches: patches.len(),
                predictions,
                outputs: Vec::new(),
            },
        )
        .expect("assemble lines");

        assert_eq!(lines.len(), fixture.expected_lines.len());
        for (actual, expected) in lines.iter().zip(fixture.expected_lines.iter()) {
            assert_eq!(actual.text, expected.text);
            assert_eq!(actual.page_no, Some(expected.page_no));
            assert_eq!(actual.bbox, Some(expected.bbox.clone()));
            let confidence = actual.confidence.expect("confidence");
            assert!((confidence - expected.confidence).abs() < 0.0001);
        }
    }

    #[test]
    fn loads_ocr_charset_from_sidecar_file() {
        let temp_root =
            std::env::temp_dir().join(format!("muse-local-onnx-charset-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root).expect("temp dir");
        let path = temp_root.join("ppocr_keys_v1.txt");
        std::fs::write(&path, "\u{feff}你\n好\n").expect("write charset");

        let charset = load_ocr_charset(&path).expect("load charset");
        assert_eq!(charset.tokens, vec!["你".to_string(), "好".to_string()]);

        std::fs::remove_dir_all(&temp_root).expect("cleanup");
    }

    #[test]
    fn normalize_adapter_discards_blank_lines_and_invalid_page_numbers() {
        let output = NormalizeOcrResultAdapter
            .adapt(RawOcrOutput {
                lines: vec![
                    RawOcrLine {
                        text: "  岗位类型：图像策略  ".to_string(),
                        page_no: Some(0),
                        bbox: None,
                        confidence: Some(0.9),
                    },
                    RawOcrLine {
                        text: "  ".to_string(),
                        page_no: Some(1),
                        bbox: None,
                        confidence: Some(0.1),
                    },
                ],
                provider: Some("worker".to_string()),
                model: Some("model".to_string()),
            })
            .expect("normalized output");

        assert_eq!(output.lines.len(), 1);
        assert_eq!(output.lines[0].text, "岗位类型：图像策略");
        assert_eq!(output.lines[0].page_no, None);
        assert_eq!(output.provider.as_deref(), Some("worker"));
    }

    #[tokio::test]
    async fn placeholder_provider_returns_informative_text() {
        let provider = PlaceholderOcrProvider::default();
        let output = provider
            .recognize(OcrRequest {
                file_name: Some("sample.png".to_string()),
                mime_type: Some("image/png".to_string()),
                bytes: b"image-bytes".to_vec(),
            })
            .await
            .expect("placeholder OCR should succeed");

        assert_eq!(output.provider.as_deref(), Some("placeholder-ocr"));
        assert!(output.lines[0].text.contains("sample.png"));
        assert_eq!(output.lines[0].page_no, Some(1));
    }

    #[test]
    fn local_onnx_model_discovery_resolves_required_models() {
        let temp_root =
            std::env::temp_dir().join(format!("muse-local-onnx-ocr-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_root).expect("temp dir");
        std::fs::write(temp_root.join("det.onnx"), b"det").expect("write det");
        std::fs::write(temp_root.join("rec.onnx"), b"rec").expect("write rec");
        std::fs::write(temp_root.join("ppocr_keys_v1.txt"), "A\nB\n").expect("write charset");

        let model_files = LocalOnnxModelFiles::discover(&temp_root).expect("discover");
        assert_eq!(model_files.det_model_path, temp_root.join("det.onnx"));
        assert_eq!(model_files.rec_model_path, temp_root.join("rec.onnx"));
        assert_eq!(
            model_files.rec_charset_path,
            temp_root.join("ppocr_keys_v1.txt")
        );
        assert!(model_files.cls_model_path.is_none());

        std::fs::remove_dir_all(&temp_root).expect("cleanup");
    }

    #[tokio::test]
    async fn local_onnx_runtime_assembles_ocr_lines_from_backend_outputs() {
        let runtime = LocalOnnxOcrRuntime::from_parts(
            "det=/tmp/det.onnx, rec=/tmp/rec.onnx, charset=/tmp/keys.txt",
            Box::new(StaticLocalOcrBackend {
                detection: OcrDetectionResult {
                    output_name: "maps".to_string(),
                    heatmap_shape: vec![2, 2],
                    candidates: vec![
                        OcrDetectionCandidate {
                            bbox: BBox {
                                x1: 10.0,
                                y1: 20.0,
                                x2: 110.0,
                                y2: 60.0,
                            },
                            score: 0.95,
                        },
                        OcrDetectionCandidate {
                            bbox: BBox {
                                x1: 12.0,
                                y1: 80.0,
                                x2: 132.0,
                                y2: 120.0,
                            },
                            score: 0.55,
                        },
                        OcrDetectionCandidate {
                            bbox: BBox {
                                x1: 20.0,
                                y1: 130.0,
                                x2: 160.0,
                                y2: 168.0,
                            },
                            score: 0.66,
                        },
                        OcrDetectionCandidate {
                            bbox: BBox {
                                x1: 18.0,
                                y1: 180.0,
                                x2: 190.0,
                                y2: 220.0,
                            },
                            score: 0.91,
                        },
                    ],
                },
                recognition: OcrRecognitionRun {
                    patch_count: 4,
                    executed_patches: 4,
                    predictions: vec![
                        OcrRecognitionPrediction {
                            patch_index: 0,
                            text: "岗位类型".to_string(),
                            confidence: 0.87,
                        },
                        OcrRecognitionPrediction {
                            patch_index: 1,
                            text: "   ".to_string(),
                            confidence: 0.99,
                        },
                        OcrRecognitionPrediction {
                            patch_index: 2,
                            text: "第三行".to_string(),
                            confidence: 0.88,
                        },
                        OcrRecognitionPrediction {
                            patch_index: 3,
                            text: "第四行".to_string(),
                            confidence: 0.93,
                        },
                    ],
                    outputs: Vec::new(),
                },
            }),
            OcrCharset {
                tokens: vec!["岗".to_string(), "位".to_string()],
            },
        );

        let output = runtime
            .recognize(OcrPreparedInput {
                file_name: Some("sample.png".to_string()),
                mime_type: Some("image/png".to_string()),
                original_bytes: vec![1, 2, 3],
                pages: vec![OcrPreparedPage {
                    page_no: 1,
                    mime_type: Some("image/png".to_string()),
                    bytes: vec![1, 2, 3],
                    raster: Some(OcrPreparedRaster {
                        original_width: 100,
                        original_height: 50,
                        resized_width: 100,
                        resized_height: 50,
                        channels: 3,
                        scale_x: 1.0,
                        scale_y: 1.0,
                        resized_rgb_u8: vec![0; 100 * 50 * 3],
                        chw_f32: vec![],
                    }),
                }],
            })
            .await
            .expect("local runtime");

        assert_eq!(output.provider.as_deref(), Some("local-onnx-ocr"));
        assert_eq!(
            output.model.as_deref(),
            Some("det=/tmp/det.onnx, rec=/tmp/rec.onnx, charset=/tmp/keys.txt")
        );
        assert_eq!(output.lines.len(), 3);
        assert_eq!(output.lines[0].text, "岗位类型");
        assert_eq!(output.lines[0].page_no, Some(1));
        assert_eq!(output.lines[0].confidence, Some(0.87));
        assert_eq!(
            output.lines[0].bbox,
            Some(BBox {
                x1: 10.0,
                y1: 20.0,
                x2: 110.0,
                y2: 60.0,
            })
        );
        assert_eq!(output.lines[1].text, "第三行");
        assert_eq!(output.lines[1].confidence, Some(0.66));
        assert_eq!(
            output.lines[1].bbox,
            Some(BBox {
                x1: 20.0,
                y1: 130.0,
                x2: 160.0,
                y2: 168.0,
            })
        );
        assert_eq!(output.lines[2].text, "第四行");
        assert_eq!(output.lines[2].confidence, Some(0.91));
    }

    #[test]
    fn local_onnx_provider_requires_model_dir_in_config() {
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
            ocr_provider: "local-onnx".to_string(),
            ocr_fallback_provider: None,
            ocr_worker_url: None,
            ocr_timeout_ms: 5_000,
            ocr_worker_token: None,
            ocr_model_dir: None,
            ocr_threads: 1,
            ocr_prewarm: false,
            pdf_raster_provider: "none".to_string(),
            pdftoppm_bin: None,
        };

        let error = match LocalOnnxOcrProvider::from_config(&config) {
            Ok(_) => panic!("expected missing model dir error"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("MUSE_OCR_MODEL_DIR"));
    }

    #[tokio::test]
    async fn http_provider_sends_binary_payload_and_decodes_response() {
        let seen = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/v1/ocr", post(test_ocr_handler))
            .with_state(seen.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("server");
        });

        let provider = HttpOcrProvider::new(
            format!("http://{addr}/v1/ocr"),
            Duration::from_secs(2),
            Some("top-secret".to_string()),
        )
        .expect("provider");
        let output = provider
            .recognize(OcrRequest {
                file_name: Some("sample.png".to_string()),
                mime_type: Some("image/png".to_string()),
                bytes: b"fake-image".to_vec(),
            })
            .await
            .expect("HTTP OCR should succeed");

        let observed = seen.lock().await.clone().expect("captured request");
        assert_eq!(observed.0.as_deref(), Some("image/png"));
        assert_eq!(observed.1.as_deref(), Some("sample.png"));
        assert_eq!(observed.2.as_deref(), Some("Bearer top-secret"));
        assert_eq!(observed.3, b"fake-image".to_vec());

        assert_eq!(output.provider.as_deref(), Some("rapidocr-worker"));
        assert_eq!(output.model.as_deref(), Some("rapidocr-onnx"));
        assert_eq!(output.lines.len(), 2);
        assert_eq!(output.lines[0].text, "岗位类型：图像策略");
        assert_eq!(output.lines[0].page_no, Some(1));
        assert!(output.lines[0].bbox.is_some());

        server.abort();
    }

    #[tokio::test]
    async fn http_provider_surfaces_worker_error_response() {
        let app = Router::new().route(
            "/v1/ocr",
            post(|| async { (StatusCode::BAD_GATEWAY, "worker unavailable") }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("server");
        });

        let provider = HttpOcrProvider::new(
            format!("http://{addr}/v1/ocr"),
            Duration::from_secs(2),
            None,
        )
        .expect("provider");
        let error = provider
            .recognize(OcrRequest {
                file_name: None,
                mime_type: None,
                bytes: b"fake-image".to_vec(),
            })
            .await
            .expect_err("HTTP OCR should fail");

        assert!(error.to_string().contains("worker unavailable"));

        server.abort();
    }

    async fn test_ocr_handler(
        State(seen): State<
            Arc<Mutex<Option<(Option<String>, Option<String>, Option<String>, Vec<u8>)>>>,
        >,
        headers: HeaderMap,
        body: Bytes,
    ) -> Json<serde_json::Value> {
        *seen.lock().await = Some((
            headers
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .map(ToString::to_string),
            headers
                .get("x-file-name")
                .and_then(|value| value.to_str().ok())
                .map(ToString::to_string),
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .map(ToString::to_string),
            body.to_vec(),
        ));

        Json(json!({
            "provider": "rapidocr-worker",
            "model": "rapidocr-onnx",
            "lines": [
                {"text": "岗位类型：图像策略", "page_no": 1, "bbox": {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 44.0}, "confidence": 0.98},
                {"text": "  ", "confidence": 0.2},
                {"text": "人设要点：边缘预处理", "page_no": 1, "bbox": {"x1": 12.0, "y1": 60.0, "x2": 160.0, "y2": 84.0}, "confidence": 0.93}
            ]
        }))
    }

    fn encode_test_png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes)
            .write_image(rgb, width, height, ColorType::Rgb8.into())
            .expect("encode png");
        bytes
    }
}
