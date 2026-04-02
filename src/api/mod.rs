use crate::{
    app::AppState,
    domain::{
        DocumentIr, Evidence, ExtractionResult, FieldValue, NormalizedDocument, SchemaSpec,
        TaskRecord, TaskStatus,
    },
    events::StreamEvent,
    ingestion::{ExtractionInput, ParseInput},
    storage::CachedExtraction,
};
use async_stream::stream;
use axum::{
    Json, Router,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, convert::Infallible, sync::Arc, time::Instant};
use thiserror::Error;
use uuid::Uuid;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/version", get(version))
        .route("/v1/extractions", post(create_extraction))
        .route(
            "/v1/extractions/normalized",
            post(create_normalized_extraction),
        )
        .route("/v1/extractions/upload", post(upload_extraction))
        .route("/v1/extractions/{task_id}", get(get_extraction))
        .route(
            "/v1/extractions/{task_id}/events",
            get(stream_extraction_events),
        )
        .with_state(Arc::new(state))
}

#[derive(Debug, Deserialize)]
pub struct CreateExtractionRequest {
    pub mode: Option<ExecutionMode>,
    pub input: ExtractionInput,
    pub schema: SchemaSpec,
    pub options: Option<ExtractionOptions>,
}

#[derive(Debug, Deserialize)]
pub struct CreateNormalizedExtractionRequest {
    pub mode: Option<ExecutionMode>,
    pub document: NormalizedDocument,
    pub schema: SchemaSpec,
    pub options: Option<ExtractionOptions>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Sync,
    Async,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ExtractionOptions {
    #[serde(default)]
    pub return_raw_text: bool,
    #[serde(default)]
    pub return_evidence: bool,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub service: String,
    pub version: String,
    pub normalized_protocol_version: &'static str,
    pub normalized_accepts_sdk_version_fallback: bool,
    pub parser_provider: &'static str,
    pub extractor_provider: &'static str,
    pub postprocessor_provider: &'static str,
    pub ocr_provider: &'static str,
    pub ocr_transport: &'static str,
    pub pdf_provider: &'static str,
    pub pdf_raster_provider: String,
    pub docx_provider: &'static str,
    pub queue_provider: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CreateExtractionResponse {
    pub task_id: String,
    pub status: String,
    pub cached: bool,
    pub stream_url: String,
    pub result: Option<ExtractionResult>,
}

#[derive(Debug, Serialize)]
pub struct TaskLookupResponse {
    pub task_id: String,
    pub status: String,
    pub result: Option<ExtractionResult>,
    pub message: Option<String>,
}

async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn version(State(state): State<Arc<AppState>>) -> Json<VersionResponse> {
    Json(VersionResponse {
        service: state.config.service_name.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        normalized_protocol_version: crate::domain::NORMALIZED_PROTOCOL_VERSION,
        normalized_accepts_sdk_version_fallback: true,
        parser_provider: state.parser.name(),
        extractor_provider: state.extractor.name(),
        postprocessor_provider: state.postprocessor.name(),
        ocr_provider: state.ocr.name(),
        ocr_transport: state.ocr.transport_name(),
        pdf_provider: state.pdf.name(),
        pdf_raster_provider: state.config.pdf_raster_provider.clone(),
        docx_provider: state.docx.name(),
        queue_provider: state.queue.name(),
    })
}

async fn create_extraction(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateExtractionRequest>,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    let parse_input = ParseInput::try_from(request.input)?;
    dispatch_parsed_extraction(
        state,
        request.mode,
        request.schema,
        request.options,
        parse_input,
    )
    .await
}

async fn create_normalized_extraction(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateNormalizedExtractionRequest>,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    dispatch_normalized_extraction(
        state,
        request.mode,
        request.schema,
        request.options,
        request.document,
    )
    .await
}

async fn upload_extraction(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    let mut mode = None;
    let mut schema = None;
    let mut options = None;
    let mut file = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        let name = field.name().unwrap_or_default().to_string();

        match name.as_str() {
            "schema" => {
                let text = field
                    .text()
                    .await
                    .map_err(|error| ApiError::bad_request(error.to_string()))?;
                schema = Some(serde_json::from_str::<SchemaSpec>(&text)?);
            }
            "mode" => {
                let text = field
                    .text()
                    .await
                    .map_err(|error| ApiError::bad_request(error.to_string()))?;
                mode = Some(parse_mode(&text)?);
            }
            "options" => {
                let text = field
                    .text()
                    .await
                    .map_err(|error| ApiError::bad_request(error.to_string()))?;
                options = Some(serde_json::from_str::<ExtractionOptions>(&text)?);
            }
            "file" => {
                let file_name = field
                    .file_name()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "upload.bin".to_string());
                let mime_type = field.content_type().map(ToString::to_string);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| ApiError::bad_request(error.to_string()))?
                    .to_vec();
                file = Some(ParseInput::from_upload(file_name, mime_type, bytes));
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let schema =
        schema.ok_or_else(|| ApiError::bad_request("multipart field `schema` is required"))?;
    let parse_input =
        file.ok_or_else(|| ApiError::bad_request("multipart field `file` is required"))?;

    dispatch_parsed_extraction(state, mode, schema, options, parse_input).await
}

async fn dispatch_parsed_extraction(
    state: Arc<AppState>,
    mode: Option<ExecutionMode>,
    schema: SchemaSpec,
    options: Option<ExtractionOptions>,
    parse_input: ParseInput,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    schema
        .validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let options = options.unwrap_or_default();
    let cache_key = build_parsed_cache_key(&parse_input, &schema, &options)?;

    match mode.unwrap_or(ExecutionMode::Sync) {
        ExecutionMode::Sync => {
            run_parsed_extraction(state, schema, options, parse_input, cache_key).await
        }
        ExecutionMode::Async => {
            enqueue_parsed_extraction(state, schema, options, parse_input, cache_key).await
        }
    }
}

async fn dispatch_normalized_extraction(
    state: Arc<AppState>,
    mode: Option<ExecutionMode>,
    schema: SchemaSpec,
    options: Option<ExtractionOptions>,
    document: NormalizedDocument,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    schema
        .validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    document
        .validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let options = options.unwrap_or_default();
    let cache_key = build_normalized_cache_key(&document, &schema, &options)?;

    match mode.unwrap_or(ExecutionMode::Sync) {
        ExecutionMode::Sync => {
            run_normalized_extraction(state, schema, options, document, cache_key).await
        }
        ExecutionMode::Async => {
            enqueue_normalized_extraction(state, schema, options, document, cache_key).await
        }
    }
}

async fn run_parsed_extraction(
    state: Arc<AppState>,
    schema: SchemaSpec,
    options: ExtractionOptions,
    parse_input: ParseInput,
    cache_key: String,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    let task_id = Uuid::new_v4().to_string();
    state.events.ensure_task(&task_id).await;
    publish_event(
        &state,
        &task_id,
        "task.accepted",
        serde_json::json!({
            "mode": "sync",
            "ingest_mode": "parsed",
        }),
    )
    .await;
    let outcome = match execute_parsed_extraction(
        state.clone(),
        task_id.clone(),
        schema,
        options,
        parse_input,
        cache_key,
    )
    .await
    {
        Ok(outcome) => outcome,
        Err(error) => {
            handle_sync_execution_error(&state, &task_id, &error).await;
            return Err(error);
        }
    };

    Ok(Json(CreateExtractionResponse {
        task_id: task_id.clone(),
        status: "succeeded".to_string(),
        cached: outcome.cached,
        stream_url: format!("/v1/extractions/{}/events", task_id),
        result: Some(outcome.result),
    }))
}

async fn run_normalized_extraction(
    state: Arc<AppState>,
    schema: SchemaSpec,
    options: ExtractionOptions,
    document: NormalizedDocument,
    cache_key: String,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    let task_id = Uuid::new_v4().to_string();
    state.events.ensure_task(&task_id).await;
    publish_event(
        &state,
        &task_id,
        "task.accepted",
        serde_json::json!({
            "mode": "sync",
            "ingest_mode": "normalized",
        }),
    )
    .await;
    let outcome = match execute_normalized_extraction(
        state.clone(),
        task_id.clone(),
        schema,
        options,
        document,
        cache_key,
    )
    .await
    {
        Ok(outcome) => outcome,
        Err(error) => {
            handle_sync_execution_error(&state, &task_id, &error).await;
            return Err(error);
        }
    };

    Ok(Json(CreateExtractionResponse {
        task_id: task_id.clone(),
        status: "succeeded".to_string(),
        cached: outcome.cached,
        stream_url: format!("/v1/extractions/{}/events", task_id),
        result: Some(outcome.result),
    }))
}

async fn enqueue_parsed_extraction(
    state: Arc<AppState>,
    schema: SchemaSpec,
    options: ExtractionOptions,
    parse_input: ParseInput,
    cache_key: String,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    let task_id = Uuid::new_v4().to_string();
    state.events.ensure_task(&task_id).await;

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Queued,
            result: None,
            message: Some("task has been accepted for background processing".to_string()),
        })
        .await?;
    publish_event(
        &state,
        &task_id,
        "task.accepted",
        serde_json::json!({
            "mode": "async",
            "ingest_mode": "parsed",
        }),
    )
    .await;
    publish_stage_event(
        &state,
        &task_id,
        "queued",
        "accepted",
        Some("task has been accepted for background processing"),
    )
    .await;

    state
        .queue
        .dispatch(Box::pin({
            let state = state.clone();
            let task_id = task_id.clone();
            async move {
                if let Err(error) = execute_parsed_extraction(
                    state.clone(),
                    task_id.clone(),
                    schema,
                    options,
                    parse_input,
                    cache_key,
                )
                .await
                {
                    publish_failure_event(&state, &task_id, &error.to_string()).await;
                    let _ = state
                        .storage
                        .upsert(TaskRecord {
                            task_id,
                            status: TaskStatus::Failed,
                            result: None,
                            message: Some(error.to_string()),
                        })
                        .await;
                }
            }
        }))
        .await?;

    Ok(Json(CreateExtractionResponse {
        task_id: task_id.clone(),
        status: "queued".to_string(),
        cached: false,
        stream_url: format!("/v1/extractions/{}/events", task_id),
        result: None,
    }))
}

async fn enqueue_normalized_extraction(
    state: Arc<AppState>,
    schema: SchemaSpec,
    options: ExtractionOptions,
    document: NormalizedDocument,
    cache_key: String,
) -> Result<Json<CreateExtractionResponse>, ApiError> {
    let task_id = Uuid::new_v4().to_string();
    state.events.ensure_task(&task_id).await;

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Queued,
            result: None,
            message: Some("task has been accepted for background processing".to_string()),
        })
        .await?;
    publish_event(
        &state,
        &task_id,
        "task.accepted",
        serde_json::json!({
            "mode": "async",
            "ingest_mode": "normalized",
        }),
    )
    .await;
    publish_stage_event(
        &state,
        &task_id,
        "queued",
        "accepted",
        Some("task has been accepted for background processing"),
    )
    .await;

    state
        .queue
        .dispatch(Box::pin({
            let state = state.clone();
            let task_id = task_id.clone();
            async move {
                if let Err(error) = execute_normalized_extraction(
                    state.clone(),
                    task_id.clone(),
                    schema,
                    options,
                    document,
                    cache_key,
                )
                .await
                {
                    publish_failure_event(&state, &task_id, &error.to_string()).await;
                    let _ = state
                        .storage
                        .upsert(TaskRecord {
                            task_id,
                            status: TaskStatus::Failed,
                            result: None,
                            message: Some(error.to_string()),
                        })
                        .await;
                }
            }
        }))
        .await?;

    Ok(Json(CreateExtractionResponse {
        task_id: task_id.clone(),
        status: "queued".to_string(),
        cached: false,
        stream_url: format!("/v1/extractions/{}/events", task_id),
        result: None,
    }))
}

struct ExecutionOutcome {
    result: ExtractionResult,
    cached: bool,
}

struct ExtractionPassOutcome {
    result: ExtractionResult,
    extract_ms: u64,
    postprocess_ms: u64,
}

#[derive(Default)]
struct PartialResultTracker {
    last_fields: HashMap<String, FieldValue>,
    next_revision: usize,
}

struct PartialFieldDelta {
    revision: usize,
    field: FieldValue,
    new_evidences: Vec<Evidence>,
}

async fn publish_event(
    state: &Arc<AppState>,
    task_id: &str,
    event_type: &str,
    payload: serde_json::Value,
) {
    let _ = state.events.publish(task_id, event_type, payload).await;
}

async fn publish_stage_event(
    state: &Arc<AppState>,
    task_id: &str,
    stage: &str,
    status: &str,
    message: Option<&str>,
) {
    let mut payload = serde_json::json!({
        "stage": stage,
        "status": status,
    });
    if let Some(message) = message {
        payload["message"] = serde_json::Value::String(message.to_string());
    }
    publish_event(state, task_id, "stage.changed", payload).await;
}

async fn publish_result_snapshot_event(
    state: &Arc<AppState>,
    task_id: &str,
    result: &ExtractionResult,
    cached: bool,
) {
    publish_event(
        state,
        task_id,
        "result.snapshot",
        serde_json::json!({
            "cached": cached,
            "field_count": result.fields.len(),
            "result": result,
        }),
    )
    .await;
}

async fn publish_partial_field_events(
    state: &Arc<AppState>,
    task_id: &str,
    result: &ExtractionResult,
    cached: bool,
) {
    for (index, field) in result.fields.iter().enumerate() {
        publish_partial_field_event(state, task_id, field, index + 1, cached, None, None, None)
            .await;
    }
}

async fn publish_partial_field_event(
    state: &Arc<AppState>,
    task_id: &str,
    field: &FieldValue,
    revision: usize,
    cached: bool,
    page_no: Option<u32>,
    pages_processed: Option<usize>,
    page_count: Option<usize>,
) {
    let mut payload = serde_json::json!({
        "cached": cached,
        "revision": revision,
        "field": field,
    });

    if let Some(page_no) = page_no {
        payload["page_no"] = serde_json::json!(page_no);
    }
    if let Some(pages_processed) = pages_processed {
        payload["pages_processed"] = serde_json::json!(pages_processed);
    }
    if let Some(page_count) = page_count {
        payload["page_count"] = serde_json::json!(page_count);
    }

    publish_event(state, task_id, "result.partial", payload).await;
}

async fn publish_block_extracted_event(
    state: &Arc<AppState>,
    task_id: &str,
    field: &FieldValue,
    revision: usize,
    page_no: u32,
    new_evidences: &[Evidence],
) {
    if new_evidences.is_empty() {
        return;
    }

    let source_block_ids = new_evidences
        .iter()
        .flat_map(|evidence| evidence.source_block_ids.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let snippets = new_evidences
        .iter()
        .map(|evidence| evidence.text.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    publish_event(
        state,
        task_id,
        "block.extracted",
        serde_json::json!({
            "revision": revision,
            "page_no": page_no,
            "field_key": field.key,
            "field": field,
            "evidence_count": new_evidences.len(),
            "source_block_ids": source_block_ids,
            "snippets": snippets,
        }),
    )
    .await;
}

async fn publish_document_events(state: &Arc<AppState>, task_id: &str, document: &DocumentIr) {
    publish_event(
        state,
        task_id,
        "document.ready",
        serde_json::json!({
            "page_count": document.pages.len(),
            "plain_text_chars": document.plain_text.chars().count(),
            "source_type": document.source_type,
        }),
    )
    .await;

    for page in &document.pages {
        publish_event(
            state,
            task_id,
            "page.parsed",
            serde_json::json!({
                "page_no": page.page_no,
                "block_count": page.blocks.len(),
                "text_chars": page.blocks.iter().map(|block| block.text.chars().count()).sum::<usize>(),
            }),
        )
        .await;

        let ocr_block_count = page
            .blocks
            .iter()
            .filter(|block| matches!(block.source_kind, crate::domain::BlockSourceKind::Ocr))
            .count();
        if ocr_block_count > 0 {
            publish_event(
                state,
                task_id,
                "page.ocr_done",
                serde_json::json!({
                    "page_no": page.page_no,
                    "ocr_block_count": ocr_block_count,
                    "ocr_provider": document.metadata.extra.get("ocr_provider"),
                    "ocr_model": document.metadata.extra.get("ocr_model"),
                    "ocr_transport": document.metadata.extra.get("ocr_transport"),
                    "pdf_ocr_input": document.metadata.extra.get("pdf_ocr_input"),
                    "pdf_raster_provider": document.metadata.extra.get("pdf_raster_provider"),
                }),
            )
            .await;
        }
    }
}

async fn publish_completion_event(
    state: &Arc<AppState>,
    task_id: &str,
    result: &ExtractionResult,
    cached: bool,
) {
    publish_event(
        state,
        task_id,
        "completed",
        serde_json::json!({
            "cached": cached,
            "result": result,
        }),
    )
    .await;
}

async fn publish_failure_event(state: &Arc<AppState>, task_id: &str, error: &str) {
    publish_event(
        state,
        task_id,
        "failed",
        serde_json::json!({
            "message": error,
        }),
    )
    .await;
}

async fn handle_sync_execution_error(state: &Arc<AppState>, task_id: &str, error: &ApiError) {
    publish_stage_event(state, task_id, "failed", "failed", Some(&error.to_string())).await;
    publish_failure_event(state, task_id, &error.to_string()).await;
    let _ = state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.to_string(),
            status: TaskStatus::Failed,
            result: None,
            message: Some(error.to_string()),
        })
        .await;
}

fn build_incremental_documents(document: &DocumentIr) -> Vec<DocumentIr> {
    if document.pages.is_empty() {
        return vec![document.clone()];
    }

    (1..=document.pages.len())
        .map(|page_count| {
            let pages = document.pages[..page_count].to_vec();
            let plain_text = pages
                .iter()
                .flat_map(|page| page.blocks.iter())
                .map(|block| block.text.trim())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");

            DocumentIr {
                doc_id: document.doc_id.clone(),
                source_type: document.source_type.clone(),
                pages,
                plain_text,
                metadata: document.metadata.clone(),
            }
        })
        .collect()
}

fn sanitize_field_for_response(field: &FieldValue, options: &ExtractionOptions) -> FieldValue {
    let mut field = field.clone();
    if !options.return_evidence {
        field.evidences.clear();
    }
    field
}

fn apply_response_options(result: &mut ExtractionResult, options: &ExtractionOptions) {
    if !options.return_raw_text {
        result.raw_text = None;
    }

    if !options.return_evidence {
        for field in &mut result.fields {
            field.evidences.clear();
        }
    }
}

fn diff_new_evidences(previous: Option<&FieldValue>, current: &FieldValue) -> Vec<Evidence> {
    current
        .evidences
        .iter()
        .filter(|evidence| {
            previous
                .map(|field| !field.evidences.iter().any(|existing| existing == *evidence))
                .unwrap_or(true)
        })
        .cloned()
        .collect()
}

fn collect_partial_field_deltas(
    tracker: &mut PartialResultTracker,
    fields: &[FieldValue],
) -> Vec<PartialFieldDelta> {
    let mut deltas = Vec::new();

    for field in fields {
        let previous = tracker.last_fields.get(&field.key).cloned();
        if previous.as_ref() == Some(field) {
            continue;
        }

        tracker.next_revision += 1;
        let revision = tracker.next_revision;
        let new_evidences = diff_new_evidences(previous.as_ref(), field);

        tracker.last_fields.insert(field.key.clone(), field.clone());

        deltas.push(PartialFieldDelta {
            revision,
            field: field.clone(),
            new_evidences,
        });
    }

    deltas
}

async fn publish_incremental_result_events(
    state: &Arc<AppState>,
    task_id: &str,
    tracker: &mut PartialResultTracker,
    result: &ExtractionResult,
    options: &ExtractionOptions,
    page_no: u32,
    pages_processed: usize,
    page_count: usize,
) {
    for delta in collect_partial_field_deltas(tracker, &result.fields) {
        let response_field = sanitize_field_for_response(&delta.field, options);
        publish_partial_field_event(
            state,
            task_id,
            &response_field,
            delta.revision,
            false,
            Some(page_no),
            Some(pages_processed),
            Some(page_count),
        )
        .await;
        publish_block_extracted_event(
            state,
            task_id,
            &response_field,
            delta.revision,
            page_no,
            &delta.new_evidences,
        )
        .await;
    }
}

async fn run_extraction_pass(
    state: &Arc<AppState>,
    task_id: &str,
    document: &DocumentIr,
    schema: &SchemaSpec,
) -> Result<ExtractionPassOutcome, ApiError> {
    let extract_start = Instant::now();
    let mut result = state.extractor.extract(document, schema).await?;
    let extract_ms = extract_start.elapsed().as_millis() as u64;

    let postprocess_start = Instant::now();
    result.task_id = task_id.to_string();
    result = state.postprocessor.finalize(result).await?;
    let postprocess_ms = postprocess_start.elapsed().as_millis() as u64;

    Ok(ExtractionPassOutcome {
        result,
        extract_ms,
        postprocess_ms,
    })
}

async fn execute_parsed_extraction(
    state: Arc<AppState>,
    task_id: String,
    schema: SchemaSpec,
    options: ExtractionOptions,
    parse_input: ParseInput,
    cache_key: String,
) -> Result<ExecutionOutcome, ApiError> {
    if let Some(cached) = state.storage.get_cached(&cache_key).await? {
        return finish_cached_execution(state, task_id, cached).await;
    }

    let total_start = Instant::now();

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Parsing,
            result: None,
            message: None,
        })
        .await?;
    publish_stage_event(&state, &task_id, "parsing", "running", None).await;

    let parse_start = Instant::now();
    let document = state.parser.parse(parse_input).await?;
    let parse_ms = parse_start.elapsed().as_millis() as u64;
    publish_document_events(&state, &task_id, &document).await;

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Extracting,
            result: None,
            message: None,
        })
        .await?;
    publish_stage_event(&state, &task_id, "extracting", "running", None).await;

    execute_document_extraction(
        state,
        task_id,
        schema,
        options,
        document,
        cache_key,
        parse_ms,
        total_start,
    )
    .await
}

async fn execute_normalized_extraction(
    state: Arc<AppState>,
    task_id: String,
    schema: SchemaSpec,
    options: ExtractionOptions,
    document: NormalizedDocument,
    cache_key: String,
) -> Result<ExecutionOutcome, ApiError> {
    if let Some(cached) = state.storage.get_cached(&cache_key).await? {
        return finish_cached_execution(state, task_id, cached).await;
    }

    let total_start = Instant::now();

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Extracting,
            result: None,
            message: Some("normalized document accepted; parse stage skipped".to_string()),
        })
        .await?;
    publish_stage_event(
        &state,
        &task_id,
        "extracting",
        "running",
        Some("normalized document accepted; parse stage skipped"),
    )
    .await;
    publish_document_events(&state, &task_id, &document.clone().into_document_ir()).await;

    execute_document_extraction(
        state,
        task_id,
        schema,
        options,
        document.into_document_ir(),
        cache_key,
        0,
        total_start,
    )
    .await
}

async fn finish_cached_execution(
    state: Arc<AppState>,
    task_id: String,
    cached: CachedExtraction,
) -> Result<ExecutionOutcome, ApiError> {
    let mut result = cached.result;
    tracing::info!(
        cache_key = %cached.cache_key,
        cache_hit_count = cached.hit_count,
        cache_created_at_ms = cached.created_at_ms,
        cache_last_accessed_at_ms = cached.last_accessed_at_ms,
        "returning cached extraction result"
    );
    publish_event(
        &state,
        &task_id,
        "cache.hit",
        serde_json::json!({
            "cache_key": cached.cache_key,
            "hit_count": cached.hit_count,
            "created_at_ms": cached.created_at_ms,
            "last_accessed_at_ms": cached.last_accessed_at_ms,
        }),
    )
    .await;

    result.task_id = task_id.clone();
    result.timings.parse_ms = 0;
    result.timings.extract_ms = 0;
    result.timings.postprocess_ms = 0;
    result.timings.total_ms = 0;

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Succeeded,
            result: Some(result.clone()),
            message: Some(format!(
                "cache hit (hit_count={}, created_at_ms={})",
                cached.hit_count, cached.created_at_ms
            )),
        })
        .await?;
    publish_partial_field_events(&state, &task_id, &result, true).await;
    publish_result_snapshot_event(&state, &task_id, &result, true).await;
    publish_completion_event(&state, &task_id, &result, true).await;

    Ok(ExecutionOutcome {
        result,
        cached: true,
    })
}

async fn execute_document_extraction(
    state: Arc<AppState>,
    task_id: String,
    schema: SchemaSpec,
    options: ExtractionOptions,
    document: DocumentIr,
    cache_key: String,
    parse_ms: u64,
    total_start: Instant,
) -> Result<ExecutionOutcome, ApiError> {
    let mut partial_tracker = PartialResultTracker::default();
    let incremental_documents = build_incremental_documents(&document);
    let page_count = document.pages.len();
    let mut final_pass = None;
    let mut extract_ms = 0_u64;
    let mut postprocess_ms = 0_u64;

    for (index, partial_document) in incremental_documents.iter().enumerate() {
        let pass = run_extraction_pass(&state, &task_id, partial_document, &schema).await?;
        extract_ms += pass.extract_ms;
        postprocess_ms += pass.postprocess_ms;

        let page_no = partial_document
            .pages
            .last()
            .map(|page| page.page_no)
            .unwrap_or(1);
        publish_incremental_result_events(
            &state,
            &task_id,
            &mut partial_tracker,
            &pass.result,
            &options,
            page_no,
            index + 1,
            page_count.max(1),
        )
        .await;

        final_pass = Some(pass);
    }

    let mut extracted = final_pass
        .map(|pass| pass.result)
        .unwrap_or_else(|| ExtractionResult {
            task_id: task_id.clone(),
            status: TaskStatus::Succeeded,
            fields: vec![],
            raw_text: Some(document.plain_text.clone()),
            timings: Default::default(),
        });

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Postprocessing,
            result: None,
            message: None,
        })
        .await?;
    publish_stage_event(&state, &task_id, "postprocessing", "running", None).await;

    extracted.timings.parse_ms = parse_ms;
    extracted.timings.extract_ms = extract_ms;
    extracted.timings.postprocess_ms = postprocess_ms;
    extracted.timings.total_ms = total_start.elapsed().as_millis() as u64;

    apply_response_options(&mut extracted, &options);
    publish_result_snapshot_event(&state, &task_id, &extracted, false).await;

    let cache_entry = state
        .storage
        .put_cached(&cache_key, extracted.clone())
        .await?;
    tracing::info!(
        cache_key = %cache_entry.cache_key,
        cache_created_at_ms = cache_entry.created_at_ms,
        "stored extraction result in cache"
    );

    state
        .storage
        .upsert(TaskRecord {
            task_id: task_id.clone(),
            status: TaskStatus::Succeeded,
            result: Some(extracted.clone()),
            message: None,
        })
        .await?;
    publish_completion_event(&state, &task_id, &extracted, false).await;

    Ok(ExecutionOutcome {
        result: extracted,
        cached: false,
    })
}

async fn get_extraction(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> Result<Json<TaskLookupResponse>, ApiError> {
    let record = state.storage.get(&task_id).await?;

    let response = match record {
        Some(record) => TaskLookupResponse {
            task_id,
            status: format_task_status(&record.status),
            result: record.result,
            message: record.message,
        },
        None => TaskLookupResponse {
            task_id,
            status: "not_found".to_string(),
            result: None,
            message: Some("task result was not found in the in-memory store".to_string()),
        },
    };

    Ok(Json(response))
}

async fn stream_extraction_events(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let task_exists =
        state.storage.get(&task_id).await?.is_some() || state.events.exists(&task_id).await;
    if !task_exists {
        return Err(ApiError::NotFound(format!(
            "task `{task_id}` was not found for streaming"
        )));
    }

    let subscription = state
        .events
        .subscribe(&task_id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("task `{task_id}` has no event stream")))?;

    let stream = stream! {
        for item in subscription.history {
            yield Ok::<Event, Infallible>(to_sse_event(&item));
        }

        let mut receiver = subscription.receiver;
        loop {
            match receiver.recv().await {
                Ok(item) => yield Ok::<Event, Infallible>(to_sse_event(&item)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    let lagged = StreamEvent {
                        sequence: 0,
                        event_type: "stream.lagged".to_string(),
                        task_id: task_id.clone(),
                        created_at_ms: current_time_ms(),
                        payload: serde_json::json!({
                            "skipped": skipped,
                        }),
                    };
                    yield Ok::<Event, Infallible>(to_sse_event(&lagged));
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn to_sse_event(event: &StreamEvent) -> Event {
    Event::default()
        .event(event.event_type.clone())
        .data(serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()))
}

fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    NotImplemented(String),
    #[error("{0}")]
    Internal(String),
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self::Internal(error.to_string())
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(error: serde_json::Error) -> Self {
        Self::bad_request(error.to_string())
    }
}

impl From<axum::extract::rejection::JsonRejection> for ApiError {
    fn from(error: axum::extract::rejection::JsonRejection) -> Self {
        Self::bad_request(error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let body = Json(serde_json::json!({
            "error": self.to_string(),
        }));

        (status, body).into_response()
    }
}

fn parse_mode(value: &str) -> Result<ExecutionMode, ApiError> {
    match value.trim() {
        "sync" => Ok(ExecutionMode::Sync),
        "async" => Ok(ExecutionMode::Async),
        other => Err(ApiError::bad_request(format!(
            "invalid mode `{other}`, expected `sync` or `async`"
        ))),
    }
}

fn format_task_status(status: &TaskStatus) -> String {
    match status {
        TaskStatus::Created => "created",
        TaskStatus::Queued => "queued",
        TaskStatus::Parsing => "parsing",
        TaskStatus::Extracting => "extracting",
        TaskStatus::Postprocessing => "postprocessing",
        TaskStatus::Succeeded => "succeeded",
        TaskStatus::Failed => "failed",
    }
    .to_string()
}

fn build_parsed_cache_key(
    parse_input: &ParseInput,
    schema: &SchemaSpec,
    options: &ExtractionOptions,
) -> Result<String, ApiError> {
    let payload = serde_json::json!({
        "parse_input": parse_input,
        "schema": schema,
        "options": options,
    });
    build_cache_hash(&payload)
}

fn build_normalized_cache_key(
    document: &NormalizedDocument,
    schema: &SchemaSpec,
    options: &ExtractionOptions,
) -> Result<String, ApiError> {
    let payload = serde_json::json!({
        "normalized_document": document,
        "schema": schema,
        "options": options,
    });
    build_cache_hash(&payload)
}

fn build_cache_hash(payload: &serde_json::Value) -> Result<String, ApiError> {
    let bytes = serde_json::to_vec(payload)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        app,
        config::Config,
        docx::{DocxProvider, ZipDocxProvider},
        domain::{
            BBox, BlockSourceKind, DocumentMetadata, Evidence, FieldValue, NormalizedPage,
            NormalizedTextBlock, PageIr, SourceType, TextBlock,
        },
        ocr::{OcrLine, OcrOutput, OcrProvider, OcrRequest},
        pdf::{PdfOcrPage, PdfOutput, PdfProvider, PdfRequest},
    };
    use ::image::{ColorType, GenericImageView, ImageEncoder, codecs::png::PngEncoder};
    use async_trait::async_trait;
    use axum::{body::Body, http::Request};
    use serde::Deserialize;
    use serde_json::json;
    use std::{fs, sync::Arc};
    use tower::util::ServiceExt;

    struct RasterAwareOcrProvider;
    struct RasterScanPdfProvider;
    struct FailingOcrProvider;

    #[derive(Debug, Deserialize)]
    struct UploadRegressionFixture {
        schema: SchemaSpec,
        expected_raw_text: String,
        expected_field_key: String,
        expected_field_value: serde_json::Value,
        expected_evidence_page_nos: Vec<u32>,
        expected_page_ocr_done: UploadRegressionEventExpectation,
    }

    #[derive(Debug, Deserialize)]
    struct UploadRegressionEventExpectation {
        pdf_ocr_input: serde_json::Value,
        pdf_raster_provider: serde_json::Value,
        ocr_provider: serde_json::Value,
        ocr_transport: serde_json::Value,
    }

    #[derive(Debug, Deserialize)]
    struct UploadFailureRegressionFixture {
        schema: SchemaSpec,
        expected_http_status: u16,
        expected_error_contains: String,
        expected_failed_event: UploadFailureEventExpectation,
    }

    #[derive(Debug, Deserialize)]
    struct UploadSyncFailureRegressionFixture {
        schema: SchemaSpec,
        expected_http_status: u16,
        expected_error_contains: String,
        expected_failed_event: UploadFailureEventExpectation,
        expected_failed_stage: UploadFailureStageExpectation,
    }

    #[derive(Debug, Deserialize)]
    struct UploadCacheHitRegressionFixture {
        schema: SchemaSpec,
        expected_raw_text: String,
        expected_field_key: String,
        expected_field_value: serde_json::Value,
        expected_cache_hit: UploadCacheHitEventExpectation,
    }

    #[derive(Debug, Deserialize)]
    struct UploadCacheHitEventExpectation {
        hit_count: u64,
    }

    #[derive(Debug, Deserialize)]
    struct UploadFailureEventExpectation {
        message_contains: String,
    }

    #[derive(Debug, Deserialize)]
    struct UploadFailureStageExpectation {
        message_contains: String,
    }

    #[async_trait]
    impl OcrProvider for RasterAwareOcrProvider {
        fn name(&self) -> &'static str {
            "raster-aware-ocr"
        }

        fn transport_name(&self) -> &'static str {
            "inproc"
        }

        async fn recognize(&self, request: OcrRequest) -> anyhow::Result<OcrOutput> {
            assert_eq!(request.mime_type.as_deref(), Some("image/png"));
            assert!(request.bytes.starts_with(&[0x89, b'P', b'N', b'G']));

            let image = ::image::load_from_memory(&request.bytes).expect("decode png");
            let pixel = image.get_pixel(0, 0).0;
            let text = if pixel[1] > pixel[0] && pixel[1] > pixel[2] {
                "岗位类型：图片运营"
            } else if pixel[0] > pixel[2] {
                "岗位类型：第一页结果"
            } else {
                "岗位类型：第二页结果"
            };

            Ok(OcrOutput {
                lines: vec![OcrLine {
                    text: text.to_string(),
                    page_no: Some(1),
                    bbox: Some(BBox {
                        x1: 4.0,
                        y1: 8.0,
                        x2: 80.0,
                        y2: 28.0,
                    }),
                    confidence: Some(0.95),
                }],
                provider: Some("raster-aware-ocr".to_string()),
                model: Some("png-page-model".to_string()),
            })
        }
    }

    #[async_trait]
    impl OcrProvider for FailingOcrProvider {
        fn name(&self) -> &'static str {
            "failing-ocr"
        }

        fn transport_name(&self) -> &'static str {
            "inproc"
        }

        async fn recognize(&self, _request: OcrRequest) -> anyhow::Result<OcrOutput> {
            anyhow::bail!("simulated OCR worker failure")
        }
    }

    #[async_trait]
    impl PdfProvider for RasterScanPdfProvider {
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

    #[tokio::test]
    async fn normalized_endpoint_accepts_valid_document() {
        let router = test_router();
        let payload = normalized_request_payload();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extractions/normalized")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn normalized_endpoint_rejects_missing_protocol_metadata() {
        let router = test_router();
        let payload = serde_json::json!({
            "mode": "sync",
            "document": {
                "source_type": "image",
                "plain_text": "岗位类型：SDK 图片",
                "metadata": {
                    "file_name": "sdk-note.png",
                    "mime_type": "image/png",
                    "extra": {}
                },
                "pages": [
                    {
                        "page_no": 1,
                        "width": 100,
                        "height": 200,
                        "blocks": [
                            {
                                "page_no": 1,
                                "text": "岗位类型：SDK 图片",
                                "bbox": null,
                                "confidence": 0.98,
                                "source_kind": "ocr"
                            }
                        ]
                    }
                ]
            },
            "schema": {
                "name": "demo",
                "version": "1",
                "fields": [
                    {"key": "岗位类型", "field_type": "string", "required": false, "multiple": false, "children": [], "hints": []}
                ]
            }
        });

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extractions/normalized")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn version_endpoint_reports_normalized_protocol_contract() {
        let router = test_router();

        let response = router
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/version")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let payload: serde_json::Value = serde_json::from_slice(&body).expect("json");

        assert_eq!(
            payload.get("normalized_protocol_version"),
            Some(&serde_json::Value::String(
                crate::domain::NORMALIZED_PROTOCOL_VERSION.to_string()
            ))
        );
        assert_eq!(
            payload.get("normalized_accepts_sdk_version_fallback"),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            payload.get("ocr_provider"),
            Some(&serde_json::Value::String("placeholder-ocr".to_string()))
        );
        assert_eq!(
            payload.get("ocr_transport"),
            Some(&serde_json::Value::String("inproc".to_string()))
        );
        assert_eq!(
            payload.get("pdf_raster_provider"),
            Some(&serde_json::Value::String("none".to_string()))
        );
    }

    #[tokio::test]
    async fn normalized_endpoint_reuses_cache_on_repeated_requests() {
        let router = test_router();
        let payload = normalized_request_payload();

        let first = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extractions/normalized")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(first.status(), StatusCode::OK);
        let first_body = axum::body::to_bytes(first.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let first_payload: serde_json::Value =
            serde_json::from_slice(&first_body).expect("first json");
        assert_eq!(
            first_payload.get("cached"),
            Some(&serde_json::Value::Bool(false))
        );

        let second = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extractions/normalized")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(second.status(), StatusCode::OK);
        let second_body = axum::body::to_bytes(second.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let second_payload: serde_json::Value =
            serde_json::from_slice(&second_body).expect("second json");
        assert_eq!(
            second_payload.get("cached"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[tokio::test]
    async fn upload_endpoint_extracts_scanned_pdf_via_raster_pages() {
        let fixture =
            load_upload_regression_fixture("fixtures/extraction/scanned_pdf_upload_fixture.json");
        let state = Arc::new(test_state_with_providers(
            Arc::new(RasterAwareOcrProvider),
            Arc::new(RasterScanPdfProvider),
            Arc::new(ZipDocxProvider),
        ));
        let router = app::build_router((*state).clone());
        let boundary = "muse-boundary";
        run_upload_regression_fixture(
            router,
            state,
            fixture,
            "scan.pdf",
            "application/pdf",
            b"%PDF-1.7 fake scanned".to_vec(),
            boundary,
        )
        .await;
    }

    #[tokio::test]
    async fn upload_endpoint_extracts_image_via_fixture_regression() {
        let fixture =
            load_upload_regression_fixture("fixtures/extraction/image_upload_fixture.json");
        let state = Arc::new(test_state_with_providers(
            Arc::new(RasterAwareOcrProvider),
            Arc::new(RasterScanPdfProvider),
            Arc::new(ZipDocxProvider),
        ));
        let router = app::build_router((*state).clone());

        run_upload_regression_fixture(
            router,
            state,
            fixture,
            "poster.png",
            "image/png",
            encode_test_png(1, 1, &[0, 255, 0]),
            "muse-image-boundary",
        )
        .await;
    }

    #[tokio::test]
    async fn upload_endpoint_reuses_cache_via_fixture_regression() {
        let fixture = load_upload_cache_hit_regression_fixture(
            "fixtures/extraction/image_upload_cache_hit_fixture.json",
        );
        let state = Arc::new(test_state_with_providers(
            Arc::new(RasterAwareOcrProvider),
            Arc::new(RasterScanPdfProvider),
            Arc::new(ZipDocxProvider),
        ));
        let router = app::build_router((*state).clone());

        run_upload_cache_hit_regression_fixture(
            router,
            state,
            fixture,
            "poster.png",
            "image/png",
            encode_test_png(1, 1, &[0, 255, 0]),
            "muse-image-cache-boundary",
        )
        .await;
    }

    #[tokio::test]
    async fn upload_endpoint_sync_failure_matches_fixture_regression() {
        let fixture = load_upload_sync_failure_regression_fixture(
            "fixtures/extraction/image_upload_sync_failure_fixture.json",
        );
        let state = Arc::new(test_state_with_providers(
            Arc::new(FailingOcrProvider),
            Arc::new(RasterScanPdfProvider),
            Arc::new(ZipDocxProvider),
        ));
        let router = app::build_router((*state).clone());

        run_upload_sync_failure_regression_fixture(
            router,
            state,
            fixture,
            "broken.png",
            "image/png",
            encode_test_png(1, 1, &[0, 255, 0]),
            "muse-image-sync-failure-boundary",
        )
        .await;
    }

    #[tokio::test]
    async fn upload_endpoint_async_failure_matches_fixture_regression() {
        let fixture = load_upload_failure_regression_fixture(
            "fixtures/extraction/image_upload_async_failure_fixture.json",
        );
        let state = Arc::new(test_state_with_providers(
            Arc::new(FailingOcrProvider),
            Arc::new(RasterScanPdfProvider),
            Arc::new(ZipDocxProvider),
        ));
        let router = app::build_router((*state).clone());
        let boundary = "muse-image-failure-boundary";
        let payload = send_upload_request(
            router.clone(),
            boundary,
            &[
                multipart_text_part("mode", "async"),
                multipart_text_part(
                    "schema",
                    &serde_json::to_string(&fixture.schema).expect("schema json"),
                ),
                multipart_file_part(
                    "file",
                    "broken.png",
                    "image/png",
                    encode_test_png(1, 1, &[0, 255, 0]),
                ),
            ],
        )
        .await;

        assert_eq!(payload.status.as_u16(), fixture.expected_http_status);
        let task_id = payload
            .payload
            .get("task_id")
            .and_then(serde_json::Value::as_str)
            .expect("task id")
            .to_string();
        assert_eq!(
            payload.payload.get("status"),
            Some(&serde_json::json!("queued"))
        );

        wait_for_task_failure(&state, &task_id).await;

        let record = state
            .storage
            .get(&task_id)
            .await
            .expect("get task")
            .expect("task record");
        assert!(matches!(record.status, TaskStatus::Failed));
        assert!(
            record
                .message
                .as_deref()
                .unwrap_or_default()
                .contains(&fixture.expected_error_contains)
        );

        let task_lookup = fetch_task_lookup(router.clone(), &task_id).await;
        assert_eq!(task_lookup.status, StatusCode::OK);
        assert_eq!(
            task_lookup.payload.get("status"),
            Some(&serde_json::json!("failed"))
        );
        assert!(
            task_lookup
                .payload
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_error_contains)
        );

        let subscription = state
            .events
            .subscribe(&task_id)
            .await
            .expect("subscription");
        let failed_event = subscription
            .history
            .iter()
            .find(|event| event.event_type == "failed")
            .expect("failed event");
        assert!(
            failed_event
                .payload
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_failed_event.message_contains)
        );
    }

    #[tokio::test]
    async fn upload_endpoint_async_pdf_raster_failure_matches_fixture_regression() {
        let fixture = load_upload_failure_regression_fixture(
            "fixtures/extraction/pdf_raster_async_failure_fixture.json",
        );
        let mut config = test_config();
        config.pdf_raster_provider = "pdftoppm".to_string();
        config.pdftoppm_bin = Some("__definitely_missing_pdftoppm_binary__".to_string());

        let state = Arc::new(app::build_state(&config));
        let router = app::build_router((*state).clone());
        let boundary = "muse-pdf-raster-failure-boundary";
        let payload = send_upload_request(
            router.clone(),
            boundary,
            &[
                multipart_text_part("mode", "async"),
                multipart_text_part(
                    "schema",
                    &serde_json::to_string(&fixture.schema).expect("schema json"),
                ),
                multipart_file_part("file", "scan.pdf", "application/pdf", build_blank_pdf()),
            ],
        )
        .await;

        assert_eq!(payload.status.as_u16(), fixture.expected_http_status);
        let task_id = payload
            .payload
            .get("task_id")
            .and_then(serde_json::Value::as_str)
            .expect("task id")
            .to_string();
        assert_eq!(
            payload.payload.get("status"),
            Some(&serde_json::json!("queued"))
        );

        wait_for_task_failure(&state, &task_id).await;

        let record = state
            .storage
            .get(&task_id)
            .await
            .expect("get task")
            .expect("task record");
        assert!(matches!(record.status, TaskStatus::Failed));
        assert!(
            record
                .message
                .as_deref()
                .unwrap_or_default()
                .contains(&fixture.expected_error_contains)
        );

        let task_lookup = fetch_task_lookup(router.clone(), &task_id).await;
        assert_eq!(task_lookup.status, StatusCode::OK);
        assert_eq!(
            task_lookup.payload.get("status"),
            Some(&serde_json::json!("failed"))
        );
        assert!(
            task_lookup
                .payload
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_error_contains)
        );

        let subscription = state
            .events
            .subscribe(&task_id)
            .await
            .expect("subscription");
        let failed_event = subscription
            .history
            .iter()
            .find(|event| event.event_type == "failed")
            .expect("failed event");
        assert!(
            failed_event
                .payload
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_failed_event.message_contains)
        );
    }

    #[tokio::test]
    async fn page_ocr_done_event_includes_pdf_raster_metadata() {
        let state = Arc::new(app::build_state(&test_config()));
        let task_id = "task-pdf-events";
        state.events.ensure_task(task_id).await;

        let mut metadata = DocumentMetadata::default();
        metadata
            .extra
            .insert("ocr_provider".to_string(), "raster-aware-ocr".to_string());
        metadata
            .extra
            .insert("ocr_model".to_string(), "png-page-model".to_string());
        metadata
            .extra
            .insert("ocr_transport".to_string(), "inproc".to_string());
        metadata
            .extra
            .insert("pdf_ocr_input".to_string(), "page_rasters".to_string());
        metadata.extra.insert(
            "pdf_raster_provider".to_string(),
            "pdftoppm-rasterizer".to_string(),
        );

        publish_document_events(
            &state,
            task_id,
            &DocumentIr {
                doc_id: "doc-pdf-events".to_string(),
                source_type: SourceType::Pdf,
                pages: vec![PageIr {
                    page_no: 1,
                    width: None,
                    height: None,
                    blocks: vec![TextBlock {
                        block_id: "b1".to_string(),
                        page_no: 1,
                        text: "岗位类型：扫描件".to_string(),
                        bbox: None,
                        confidence: Some(0.92),
                        source_kind: BlockSourceKind::Ocr,
                    }],
                }],
                plain_text: "岗位类型：扫描件".to_string(),
                metadata,
            },
        )
        .await;

        let subscription = state.events.subscribe(task_id).await.expect("subscription");
        let page_ocr_done = subscription
            .history
            .iter()
            .find(|event| event.event_type == "page.ocr_done")
            .expect("page.ocr_done event");

        assert_eq!(
            page_ocr_done.payload.get("pdf_ocr_input"),
            Some(&serde_json::Value::String("page_rasters".to_string()))
        );
        assert_eq!(
            page_ocr_done.payload.get("pdf_raster_provider"),
            Some(&serde_json::Value::String(
                "pdftoppm-rasterizer".to_string()
            ))
        );
    }

    fn test_router() -> Router {
        let config = test_config();
        let state = app::build_state(&config);
        app::build_router(state)
    }

    fn test_state_with_providers(
        ocr: Arc<dyn OcrProvider>,
        pdf: Arc<dyn PdfProvider>,
        docx: Arc<dyn DocxProvider>,
    ) -> app::AppState {
        let config = test_config();
        app::build_state_with_providers(&config, ocr, pdf, docx)
    }

    fn test_config() -> Config {
        Config {
            listen_addr: "127.0.0.1:0".parse().expect("socket addr"),
            service_name: "test-service".to_string(),
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
            pdf_raster_provider: "none".to_string(),
            pdftoppm_bin: None,
        }
    }

    fn multipart_text_part(name: &str, value: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        bytes.extend_from_slice(value.as_bytes());
        bytes.extend_from_slice(b"\r\n");
        bytes
    }

    fn multipart_file_part(
        name: &str,
        file_name: &str,
        content_type: &str,
        bytes: Vec<u8>,
    ) -> Vec<u8> {
        let mut part = Vec::new();
        part.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{name}\"; filename=\"{file_name}\"\r\n"
            )
            .as_bytes(),
        );
        part.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
        part.extend_from_slice(&bytes);
        part.extend_from_slice(b"\r\n");
        part
    }

    fn multipart_body(boundary: &str, parts: &[Vec<u8>]) -> Vec<u8> {
        let mut body = Vec::new();
        for part in parts {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(part);
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    fn encode_test_png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let encoder = PngEncoder::new(&mut bytes);
        encoder
            .write_image(rgb, width, height, ColorType::Rgb8.into())
            .expect("encode png");
        bytes
    }

    fn build_blank_pdf() -> Vec<u8> {
        use lopdf::{
            Document, Object, Stream,
            content::{Content, Operation},
            dictionary,
        };

        let mut document = Document::with_version("1.5");
        let pages_id = document.new_object_id();
        let page_id = document.new_object_id();
        let resources_id = document.add_object(dictionary! {});
        let content = Content {
            operations: vec![Operation::new("BT", vec![]), Operation::new("ET", vec![])],
        };
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

    fn load_upload_regression_fixture(path: &str) -> UploadRegressionFixture {
        let raw = fs::read_to_string(path).expect("read upload fixture");
        serde_json::from_str(&raw).expect("parse upload fixture")
    }

    fn load_upload_failure_regression_fixture(path: &str) -> UploadFailureRegressionFixture {
        let raw = fs::read_to_string(path).expect("read upload failure fixture");
        serde_json::from_str(&raw).expect("parse upload failure fixture")
    }

    fn load_upload_sync_failure_regression_fixture(
        path: &str,
    ) -> UploadSyncFailureRegressionFixture {
        let raw = fs::read_to_string(path).expect("read upload sync failure fixture");
        serde_json::from_str(&raw).expect("parse upload sync failure fixture")
    }

    fn load_upload_cache_hit_regression_fixture(path: &str) -> UploadCacheHitRegressionFixture {
        let raw = fs::read_to_string(path).expect("read upload cache-hit fixture");
        serde_json::from_str(&raw).expect("parse upload cache-hit fixture")
    }

    async fn run_upload_regression_fixture(
        router: Router,
        state: Arc<app::AppState>,
        fixture: UploadRegressionFixture,
        file_name: &str,
        content_type: &str,
        file_bytes: Vec<u8>,
        boundary: &str,
    ) {
        let payload = send_standard_upload_request(
            router.clone(),
            "sync",
            &fixture.schema,
            file_name,
            content_type,
            file_bytes,
            boundary,
        )
        .await;

        assert_eq!(payload.status, StatusCode::OK);
        let task_id = payload
            .payload
            .get("task_id")
            .and_then(serde_json::Value::as_str)
            .expect("task id");
        let result = payload.payload.get("result").expect("result");
        assert_eq!(
            result.get("raw_text"),
            Some(&serde_json::Value::String(fixture.expected_raw_text))
        );
        assert_eq!(
            result.pointer("/fields/0/key"),
            Some(&serde_json::Value::String(fixture.expected_field_key))
        );
        assert_eq!(
            result.pointer("/fields/0/value"),
            Some(&fixture.expected_field_value)
        );
        for (index, page_no) in fixture.expected_evidence_page_nos.iter().enumerate() {
            assert_eq!(
                result.pointer(&format!("/fields/0/evidences/{index}/page_no")),
                Some(&serde_json::json!(page_no))
            );
        }

        let subscription = state.events.subscribe(task_id).await.expect("subscription");
        let page_ocr_done = subscription
            .history
            .iter()
            .find(|event| event.event_type == "page.ocr_done")
            .expect("page.ocr_done event");
        assert_eq!(
            page_ocr_done.payload.get("pdf_ocr_input"),
            Some(&fixture.expected_page_ocr_done.pdf_ocr_input)
        );
        assert_eq!(
            page_ocr_done.payload.get("pdf_raster_provider"),
            Some(&fixture.expected_page_ocr_done.pdf_raster_provider)
        );
        assert_eq!(
            page_ocr_done.payload.get("ocr_provider"),
            Some(&fixture.expected_page_ocr_done.ocr_provider)
        );
        assert_eq!(
            page_ocr_done.payload.get("ocr_transport"),
            Some(&fixture.expected_page_ocr_done.ocr_transport)
        );
    }

    async fn run_upload_cache_hit_regression_fixture(
        router: Router,
        state: Arc<app::AppState>,
        fixture: UploadCacheHitRegressionFixture,
        file_name: &str,
        content_type: &str,
        file_bytes: Vec<u8>,
        boundary: &str,
    ) {
        let first = run_upload_cache_hit_request(
            router.clone(),
            &fixture.schema,
            file_name,
            content_type,
            file_bytes.clone(),
            boundary,
        )
        .await;

        assert_eq!(first.cached, Some(false));
        assert_eq!(
            first.result.get("raw_text"),
            Some(&serde_json::Value::String(
                fixture.expected_raw_text.clone()
            ))
        );
        assert_eq!(
            first.result.pointer("/fields/0/key"),
            Some(&serde_json::Value::String(
                fixture.expected_field_key.clone()
            ))
        );
        assert_eq!(
            first.result.pointer("/fields/0/value"),
            Some(&fixture.expected_field_value)
        );

        let first_subscription = state
            .events
            .subscribe(&first.task_id)
            .await
            .expect("first subscription");
        assert!(
            first_subscription
                .history
                .iter()
                .any(|event| event.event_type == "page.ocr_done"),
            "first request should execute OCR before caching"
        );

        let second = run_upload_cache_hit_request(
            router,
            &fixture.schema,
            file_name,
            content_type,
            file_bytes,
            boundary,
        )
        .await;

        assert_eq!(second.cached, Some(true));
        assert_eq!(
            second.result.get("raw_text"),
            Some(&serde_json::Value::String(fixture.expected_raw_text))
        );
        assert_eq!(
            second.result.pointer("/fields/0/key"),
            Some(&serde_json::Value::String(fixture.expected_field_key))
        );
        assert_eq!(
            second.result.pointer("/fields/0/value"),
            Some(&fixture.expected_field_value)
        );

        let second_subscription = state
            .events
            .subscribe(&second.task_id)
            .await
            .expect("second subscription");
        let cache_hit = second_subscription
            .history
            .iter()
            .find(|event| event.event_type == "cache.hit")
            .expect("cache.hit event");
        assert_eq!(
            cache_hit.payload.get("hit_count"),
            Some(&serde_json::json!(fixture.expected_cache_hit.hit_count))
        );
        assert!(
            cache_hit
                .payload
                .get("cache_key")
                .and_then(serde_json::Value::as_str)
                .is_some(),
            "cache.hit should expose cache_key"
        );
        assert!(
            second_subscription
                .history
                .iter()
                .all(|event| event.event_type != "page.ocr_done"),
            "cached request should not rerun OCR"
        );
    }

    async fn run_upload_sync_failure_regression_fixture(
        router: Router,
        state: Arc<app::AppState>,
        fixture: UploadSyncFailureRegressionFixture,
        file_name: &str,
        content_type: &str,
        file_bytes: Vec<u8>,
        boundary: &str,
    ) {
        let before_task_ids = state.events.task_ids().await;
        let payload = send_standard_upload_request(
            router.clone(),
            "sync",
            &fixture.schema,
            file_name,
            content_type,
            file_bytes,
            boundary,
        )
        .await;

        assert_eq!(payload.status.as_u16(), fixture.expected_http_status);
        assert!(
            payload
                .payload
                .get("error")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_error_contains)
        );

        let after_task_ids = state.events.task_ids().await;
        let task_id = after_task_ids
            .into_iter()
            .find(|candidate| !before_task_ids.iter().any(|before| before == candidate))
            .expect("new sync failure task id");

        let record = state
            .storage
            .get(&task_id)
            .await
            .expect("get task")
            .expect("task record");
        assert!(matches!(record.status, TaskStatus::Failed));
        assert!(
            record
                .message
                .as_deref()
                .unwrap_or_default()
                .contains(&fixture.expected_error_contains)
        );

        let task_lookup = fetch_task_lookup(router.clone(), &task_id).await;
        assert_eq!(task_lookup.status, StatusCode::OK);
        assert_eq!(
            task_lookup.payload.get("status"),
            Some(&serde_json::json!("failed"))
        );
        assert!(
            task_lookup
                .payload
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_error_contains)
        );

        let subscription = state
            .events
            .subscribe(&task_id)
            .await
            .expect("subscription");
        let failed_stage = subscription
            .history
            .iter()
            .find(|event| {
                event.event_type == "stage.changed"
                    && event.payload.get("stage") == Some(&serde_json::json!("failed"))
            })
            .expect("failed stage event");
        assert_eq!(
            failed_stage.payload.get("status"),
            Some(&serde_json::json!("failed"))
        );
        assert!(
            failed_stage
                .payload
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_failed_stage.message_contains)
        );

        let failed_event = subscription
            .history
            .iter()
            .find(|event| event.event_type == "failed")
            .expect("failed event");
        assert!(
            failed_event
                .payload
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .contains(&fixture.expected_failed_event.message_contains)
        );
    }

    async fn run_upload_cache_hit_request(
        router: Router,
        schema: &SchemaSpec,
        file_name: &str,
        content_type: &str,
        file_bytes: Vec<u8>,
        boundary: &str,
    ) -> UploadRequestOutcome {
        let payload = send_standard_upload_request(
            router,
            "sync",
            schema,
            file_name,
            content_type,
            file_bytes,
            boundary,
        )
        .await;
        assert_eq!(payload.status, StatusCode::OK);
        UploadRequestOutcome {
            task_id: payload
                .payload
                .get("task_id")
                .and_then(serde_json::Value::as_str)
                .expect("task id")
                .to_string(),
            cached: payload
                .payload
                .get("cached")
                .and_then(serde_json::Value::as_bool),
            result: payload.payload.get("result").cloned().expect("result"),
        }
    }

    struct UploadRequestOutcome {
        task_id: String,
        cached: Option<bool>,
        result: serde_json::Value,
    }

    struct UploadHttpResponse {
        status: StatusCode,
        payload: serde_json::Value,
    }

    async fn send_standard_upload_request(
        router: Router,
        mode: &str,
        schema: &SchemaSpec,
        file_name: &str,
        content_type: &str,
        file_bytes: Vec<u8>,
        boundary: &str,
    ) -> UploadHttpResponse {
        let options = serde_json::json!({
            "return_raw_text": true,
            "return_evidence": true
        });
        send_upload_request(
            router,
            boundary,
            &[
                multipart_text_part("mode", mode),
                multipart_text_part(
                    "schema",
                    &serde_json::to_string(schema).expect("schema json"),
                ),
                multipart_text_part("options", &options.to_string()),
                multipart_file_part("file", file_name, content_type, file_bytes),
            ],
        )
        .await
    }

    async fn send_upload_request(
        router: Router,
        boundary: &str,
        parts: &[Vec<u8>],
    ) -> UploadHttpResponse {
        let body = multipart_body(boundary, parts);
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extractions/upload")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .expect("request"),
            )
            .await
            .expect("response");
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");

        UploadHttpResponse {
            status,
            payload: serde_json::from_slice(&body).expect("json"),
        }
    }

    async fn fetch_task_lookup(router: Router, task_id: &str) -> UploadHttpResponse {
        let response = router
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/v1/extractions/{task_id}"))
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");

        UploadHttpResponse {
            status,
            payload: serde_json::from_slice(&body).expect("json"),
        }
    }

    async fn wait_for_task_failure(state: &Arc<app::AppState>, task_id: &str) {
        for _ in 0..40 {
            if let Some(record) = state.storage.get(task_id).await.expect("get task") {
                if matches!(record.status, TaskStatus::Failed) {
                    return;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }

        panic!("task `{task_id}` did not reach failed status in time");
    }

    fn normalized_request_payload() -> serde_json::Value {
        serde_json::json!({
            "mode": "sync",
            "document": {
                "source_type": "image",
                "plain_text": "岗位类型：SDK 图片\n人设要点：边缘预处理",
                "metadata": {
                    "file_name": "sdk-note.png",
                    "mime_type": "image/png",
                    "extra": {
                        "protocol_version": crate::domain::NORMALIZED_PROTOCOL_VERSION,
                        "sdk_version": "0.1.0"
                    }
                },
                "pages": [
                    {
                        "page_no": 1,
                        "width": 100,
                        "height": 200,
                        "blocks": [
                            {
                                "page_no": 1,
                                "text": "岗位类型：SDK 图片",
                                "bbox": null,
                                "confidence": 0.98,
                                "source_kind": "ocr"
                            }
                        ]
                    }
                ]
            },
            "schema": {
                "name": "demo",
                "version": "1",
                "fields": [
                    {"key": "岗位类型", "field_type": "string", "required": false, "multiple": false, "children": [], "hints": []}
                ]
            },
            "options": {
                "return_raw_text": true,
                "return_evidence": true
            }
        })
    }

    #[allow(dead_code)]
    fn _normalized_document_example() -> NormalizedDocument {
        let mut metadata = DocumentMetadata::default();
        metadata.extra.insert(
            "protocol_version".to_string(),
            crate::domain::NORMALIZED_PROTOCOL_VERSION.to_string(),
        );
        metadata
            .extra
            .insert("sdk_version".to_string(), "0.1.0".to_string());

        NormalizedDocument {
            source_type: crate::domain::SourceType::Image,
            plain_text: "岗位类型：SDK 图片".to_string(),
            metadata,
            pages: vec![NormalizedPage {
                page_no: 1,
                width: None,
                height: None,
                blocks: vec![NormalizedTextBlock {
                    page_no: 1,
                    text: "岗位类型：SDK 图片".to_string(),
                    bbox: None,
                    confidence: Some(0.9),
                    source_kind: BlockSourceKind::Ocr,
                }],
            }],
        }
    }

    #[test]
    fn incremental_documents_follow_page_prefixes() {
        let document = DocumentIr {
            doc_id: "doc-1".to_string(),
            source_type: SourceType::Pdf,
            pages: vec![
                PageIr {
                    page_no: 1,
                    width: None,
                    height: None,
                    blocks: vec![TextBlock {
                        block_id: "b1".to_string(),
                        page_no: 1,
                        text: "第一页".to_string(),
                        bbox: None,
                        confidence: Some(0.9),
                        source_kind: BlockSourceKind::NativeText,
                    }],
                },
                PageIr {
                    page_no: 2,
                    width: None,
                    height: None,
                    blocks: vec![TextBlock {
                        block_id: "b2".to_string(),
                        page_no: 2,
                        text: "第二页".to_string(),
                        bbox: None,
                        confidence: Some(0.9),
                        source_kind: BlockSourceKind::Ocr,
                    }],
                },
            ],
            plain_text: "第一页\n第二页".to_string(),
            metadata: DocumentMetadata::default(),
        };

        let incremental = build_incremental_documents(&document);

        assert_eq!(incremental.len(), 2);
        assert_eq!(incremental[0].pages.len(), 1);
        assert_eq!(incremental[0].plain_text, "第一页");
        assert_eq!(incremental[1].pages.len(), 2);
        assert_eq!(incremental[1].plain_text, "第一页\n第二页");
    }

    #[test]
    fn partial_field_deltas_increment_revision_for_same_field() {
        let mut tracker = PartialResultTracker::default();
        let first = FieldValue {
            key: "岗位类型".to_string(),
            value: json!("运营"),
            confidence: Some(0.8),
            evidences: vec![Evidence {
                page_no: Some(1),
                text: "岗位类型：运营".to_string(),
                bbox: None,
                source_block_ids: vec!["b1".to_string()],
            }],
        };
        let second = FieldValue {
            key: "岗位类型".to_string(),
            value: json!("高级运营"),
            confidence: Some(0.9),
            evidences: vec![
                Evidence {
                    page_no: Some(1),
                    text: "岗位类型：运营".to_string(),
                    bbox: None,
                    source_block_ids: vec!["b1".to_string()],
                },
                Evidence {
                    page_no: Some(2),
                    text: "岗位类型：高级运营".to_string(),
                    bbox: None,
                    source_block_ids: vec!["b2".to_string()],
                },
            ],
        };

        let first_delta = collect_partial_field_deltas(&mut tracker, &[first]);
        let second_delta = collect_partial_field_deltas(&mut tracker, &[second]);

        assert_eq!(first_delta.len(), 1);
        assert_eq!(first_delta[0].revision, 1);
        assert_eq!(first_delta[0].new_evidences.len(), 1);
        assert_eq!(second_delta.len(), 1);
        assert_eq!(second_delta[0].revision, 2);
        assert_eq!(second_delta[0].new_evidences.len(), 1);
    }
}
