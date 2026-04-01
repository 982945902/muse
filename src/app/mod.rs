use crate::{
    api,
    config::Config,
    docx::{DocxProvider, ZipDocxProvider},
    events::EventHub,
    extractor::{Extractor, build_extractor},
    ocr::{HttpOcrProvider, OcrProvider, PlaceholderOcrProvider},
    parser::{DefaultParser, Parser},
    pdf::{LopdfTextLayerProvider, PdfProvider},
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

pub fn build_state(config: &Config) -> AppState {
    let ocr: Arc<dyn OcrProvider> = match config.ocr_provider.as_str() {
        "http" => {
            let provider = HttpOcrProvider::from_config(config)
                .unwrap_or_else(|error| panic!("failed to configure HTTP OCR provider: {error}"));
            Arc::new(provider)
        }
        "placeholder" => Arc::new(PlaceholderOcrProvider),
        other => panic!("unsupported OCR provider `{other}`"),
    };
    let pdf: Arc<dyn PdfProvider> = Arc::new(LopdfTextLayerProvider);
    let docx: Arc<dyn DocxProvider> = Arc::new(ZipDocxProvider);

    AppState {
        config: config.clone(),
        parser: Arc::new(DefaultParser::new(ocr.clone(), pdf.clone(), docx.clone())),
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

pub fn build_router(state: AppState) -> axum::Router {
    api::router(state)
}
