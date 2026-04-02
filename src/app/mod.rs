use crate::{
    api,
    config::Config,
    docx::{DocxProvider, ZipDocxProvider},
    events::EventHub,
    extractor::{Extractor, build_extractor},
    ocr::{
        FallbackOcrProvider, HttpOcrProvider, LocalOnnxOcrProvider, OcrProvider,
        PlaceholderOcrProvider,
    },
    parser::{DefaultParser, Parser},
    pdf::{CompositePdfProvider, PdfProvider},
    postprocess::{DefaultPostprocessor, Postprocessor},
    queue::{InMemoryQueue, TaskQueue},
    storage::{ExtractionStore, InMemoryStorage},
};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub parser: Arc<dyn Parser>,
    pub extractor: Arc<dyn Extractor>,
    pub postprocessor: Arc<dyn Postprocessor>,
    pub ocr: Arc<dyn OcrProvider>,
    pub pdf: Arc<dyn PdfProvider>,
    pub docx: Arc<dyn DocxProvider>,
    pub queue: Arc<dyn TaskQueue>,
    pub storage: Arc<dyn ExtractionStore>,
    pub events: Arc<EventHub>,
}

fn build_ocr_provider_by_name(config: &Config, provider_name: &str) -> Arc<dyn OcrProvider> {
    match provider_name {
        "http" => {
            let provider = HttpOcrProvider::from_config(config)
                .unwrap_or_else(|error| panic!("failed to configure HTTP OCR provider: {error}"));
            Arc::new(provider)
        }
        "local-onnx" => {
            let provider = LocalOnnxOcrProvider::from_config(config).unwrap_or_else(|error| {
                panic!("failed to configure local ONNX OCR provider: {error}")
            });
            Arc::new(provider)
        }
        "placeholder" => Arc::new(PlaceholderOcrProvider::default()),
        other => panic!("unsupported OCR provider `{other}`"),
    }
}

pub fn build_state_with_providers(
    config: &Config,
    ocr: Arc<dyn OcrProvider>,
    pdf: Arc<dyn PdfProvider>,
    docx: Arc<dyn DocxProvider>,
) -> AppState {
    let parser: Arc<dyn Parser> =
        Arc::new(DefaultParser::new(ocr.clone(), pdf.clone(), docx.clone()));

    AppState {
        config: config.clone(),
        parser,
        extractor: build_extractor(config),
        postprocessor: Arc::new(DefaultPostprocessor),
        ocr,
        pdf,
        docx,
        queue: Arc::new(InMemoryQueue),
        storage: Arc::new(InMemoryStorage::default()),
        events: Arc::new(EventHub::default()),
    }
}

pub fn build_state(config: &Config) -> AppState {
    let primary_ocr = build_ocr_provider_by_name(config, &config.ocr_provider);
    let ocr: Arc<dyn OcrProvider> = match config.ocr_fallback_provider.as_deref() {
        Some(fallback_provider) if fallback_provider != config.ocr_provider => {
            Arc::new(FallbackOcrProvider::new(
                primary_ocr.clone(),
                build_ocr_provider_by_name(config, fallback_provider),
            ))
        }
        _ => primary_ocr,
    };
    let pdf: Arc<dyn PdfProvider> = Arc::new(
        CompositePdfProvider::from_config(config)
            .unwrap_or_else(|error| panic!("failed to configure PDF provider: {error}")),
    );
    let docx: Arc<dyn DocxProvider> = Arc::new(ZipDocxProvider);
    build_state_with_providers(config, ocr, pdf, docx)
}

pub fn build_router(state: AppState) -> axum::Router {
    api::router(state)
}
