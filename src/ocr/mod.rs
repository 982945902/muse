use crate::{config::Config, domain::BBox};
use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

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
    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput>;
}

#[derive(Default)]
pub struct PlaceholderOcrProvider;

#[async_trait]
impl OcrProvider for PlaceholderOcrProvider {
    fn name(&self) -> &'static str {
        "placeholder-ocr"
    }

    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
        let file_name = request.file_name.unwrap_or_else(|| "unnamed".to_string());
        let mime_type = request.mime_type.unwrap_or_else(|| "unknown".to_string());
        let byte_size = request.bytes.len();

        Ok(OcrOutput {
            lines: vec![OcrLine {
                text: format!(
                    "image OCR provider not implemented yet\nfile_name: {file_name}\nmime_type: {mime_type}\nbyte_size: {byte_size}"
                ),
                page_no: Some(1),
                bbox: None,
                confidence: Some(0.1),
            }],
            provider: Some(self.name().to_string()),
            model: None,
        })
    }
}

pub struct HttpOcrProvider {
    client: Client,
    endpoint: String,
    auth_token: Option<String>,
}

impl HttpOcrProvider {
    pub fn new(
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

#[derive(Debug, Deserialize)]
struct WorkerOcrResponse {
    #[serde(default)]
    lines: Vec<OcrLine>,
    provider: Option<String>,
    model: Option<String>,
}

#[async_trait]
impl OcrProvider for HttpOcrProvider {
    fn name(&self) -> &'static str {
        "http-ocr-worker"
    }

    async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
        let mime_type = request
            .mime_type
            .clone()
            .unwrap_or_else(|| "application/octet-stream".to_string());

        let mut builder = self
            .client
            .post(&self.endpoint)
            .header("content-type", mime_type)
            .body(request.bytes);

        if let Some(file_name) = request.file_name.as_deref() {
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

        let lines = payload
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
            provider: payload.provider,
            model: payload.model,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        body::Bytes,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
    };
    use serde_json::json;
    use std::sync::Arc;
    use tokio::{net::TcpListener, sync::Mutex};

    #[tokio::test]
    async fn placeholder_provider_returns_informative_text() {
        let provider = PlaceholderOcrProvider;
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
}
