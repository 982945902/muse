use std::{env, net::SocketAddr};

#[derive(Clone, Debug)]
pub struct Config {
    pub listen_addr: SocketAddr,
    pub service_name: String,
    pub log_filter: String,
    pub storage_provider: String,
    pub storage_sqlite_path: Option<String>,
    pub extractor_provider: String,
    pub onnx_model_path: Option<String>,
    pub onnx_model_spec_path: Option<String>,
    pub onnx_threads: usize,
    pub onnx_input_text_name: String,
    pub onnx_input_schema_name: String,
    pub onnx_output_json_name: String,
    pub ocr_provider: String,
    pub ocr_fallback_provider: Option<String>,
    pub ocr_worker_url: Option<String>,
    pub ocr_timeout_ms: u64,
    pub ocr_worker_token: Option<String>,
    pub ocr_model_dir: Option<String>,
    pub ocr_threads: usize,
    pub ocr_prewarm: bool,
    pub pdf_raster_provider: String,
    pub pdftoppm_bin: Option<String>,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let host = env::var("MUSE_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = env::var("MUSE_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(3000);
        let listen_addr = format!("{host}:{port}").parse()?;
        let service_name =
            env::var("MUSE_SERVICE_NAME").unwrap_or_else(|_| "muse-extraction-service".to_string());
        let log_filter = env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
        let storage_provider =
            env::var("MUSE_STORAGE_PROVIDER").unwrap_or_else(|_| "memory".to_string());
        let storage_sqlite_path = env::var("MUSE_STORAGE_SQLITE_PATH").ok();
        let extractor_provider =
            env::var("MUSE_EXTRACTOR_PROVIDER").unwrap_or_else(|_| "heuristic".to_string());
        let onnx_model_path = env::var("MUSE_ONNX_MODEL_PATH").ok();
        let onnx_model_spec_path = env::var("MUSE_ONNX_MODEL_SPEC_PATH").ok();
        let onnx_threads = env::var("MUSE_ONNX_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let onnx_input_text_name =
            env::var("MUSE_ONNX_INPUT_TEXT_NAME").unwrap_or_else(|_| "text".to_string());
        let onnx_input_schema_name =
            env::var("MUSE_ONNX_INPUT_SCHEMA_NAME").unwrap_or_else(|_| "schema".to_string());
        let onnx_output_json_name =
            env::var("MUSE_ONNX_OUTPUT_JSON_NAME").unwrap_or_else(|_| "json_output".to_string());
        let ocr_provider = env::var("MUSE_OCR_PROVIDER").unwrap_or_else(|_| {
            if env::var("MUSE_OCR_WORKER_URL").is_ok() {
                "http".to_string()
            } else if env::var("MUSE_OCR_MODEL_DIR").is_ok() {
                "local-onnx".to_string()
            } else {
                "placeholder".to_string()
            }
        });
        let ocr_fallback_provider = env::var("MUSE_OCR_FALLBACK_PROVIDER")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let ocr_worker_url = env::var("MUSE_OCR_WORKER_URL").ok();
        let ocr_timeout_ms = env::var("MUSE_OCR_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(5_000);
        let ocr_worker_token = env::var("MUSE_OCR_WORKER_TOKEN").ok();
        let ocr_model_dir = env::var("MUSE_OCR_MODEL_DIR").ok();
        let ocr_threads = env::var("MUSE_OCR_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1);
        let ocr_prewarm = env::var("MUSE_OCR_PREWARM")
            .ok()
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        let pdf_raster_provider = env::var("MUSE_PDF_RASTER_PROVIDER").unwrap_or_else(|_| {
            if env::var("MUSE_PDFTOPPM_BIN").is_ok() {
                "pdftoppm".to_string()
            } else {
                "none".to_string()
            }
        });
        let pdftoppm_bin = env::var("MUSE_PDFTOPPM_BIN").ok();

        Ok(Self {
            listen_addr,
            service_name,
            log_filter,
            storage_provider,
            storage_sqlite_path,
            extractor_provider,
            onnx_model_path,
            onnx_model_spec_path,
            onnx_threads,
            onnx_input_text_name,
            onnx_input_schema_name,
            onnx_output_json_name,
            ocr_provider,
            ocr_fallback_provider,
            ocr_worker_url,
            ocr_timeout_ms,
            ocr_worker_token,
            ocr_model_dir,
            ocr_threads,
            ocr_prewarm,
            pdf_raster_provider,
            pdftoppm_bin,
        })
    }
}
