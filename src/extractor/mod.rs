use crate::{
    config::Config,
    domain::{
        DocumentIr, Evidence, ExtractionResult, FieldSpec, FieldType, FieldValue, SchemaSpec,
        TaskStatus, TextBlock, TimingBreakdown,
    },
};
use async_trait::async_trait;
#[cfg(feature = "onnx-runtime")]
use onnxruntime_sys_ng as ort_sys;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value, json};
#[cfg(all(feature = "onnx-runtime", not(target_family = "windows")))]
use std::os::unix::ffi::OsStrExt;
#[cfg(all(feature = "onnx-runtime", target_family = "windows"))]
use std::os::windows::ffi::OsStrExt;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
#[cfg(feature = "onnx-runtime")]
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
    ptr,
    sync::OnceLock,
};
use uuid::Uuid;
#[cfg(feature = "onnx-runtime")]
use tokenizers::{
    PaddingDirection as TokenizerPaddingDirection, PaddingParams, PaddingStrategy, Tokenizer,
    TruncationDirection, TruncationParams, TruncationStrategy,
};

#[async_trait]
pub trait Extractor: Send + Sync {
    fn name(&self) -> &'static str;
    async fn extract(
        &self,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult>;
}

#[derive(Default)]
pub struct HeuristicExtractor;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum OnnxDecodeStrategy {
    #[default]
    JsonStringScalar,
    UieSpan,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum OnnxRuntimeContractKind {
    #[default]
    StringIo,
    Tokenized,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OnnxTokenizedInputNames {
    input_ids: String,
    attention_mask: String,
    token_type_ids: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OnnxTokenizedOutputNames {
    start_probs: String,
    end_probs: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct OnnxModelSpec {
    #[serde(default = "default_onnx_spec_protocol_version")]
    protocol_version: u32,
    #[serde(default)]
    runtime_contract: OnnxRuntimeContractKind,
    input_text_name: Option<String>,
    input_schema_name: Option<String>,
    output_json_name: Option<String>,
    tokenizer_path: Option<String>,
    max_length: Option<usize>,
    inputs: Option<OnnxTokenizedInputNames>,
    outputs: Option<OnnxTokenizedOutputNames>,
    #[serde(default)]
    decode_strategy: OnnxDecodeStrategy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedOnnxTokenizedInputs {
    input_ids: String,
    attention_mask: String,
    token_type_ids: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedOnnxTokenizedOutputs {
    start_probs: String,
    end_probs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedOnnxTokenizedContract {
    tokenizer_path: PathBuf,
    max_length: usize,
    inputs: ResolvedOnnxTokenizedInputs,
    outputs: ResolvedOnnxTokenizedOutputs,
}

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
#[derive(Clone, Debug)]
struct ResolvedOnnxModelContract {
    runtime_contract: OnnxRuntimeContractKind,
    input_text_name: String,
    input_schema_name: String,
    output_json_name: String,
    tokenized: Option<ResolvedOnnxTokenizedContract>,
    _decode_strategy: OnnxDecodeStrategy,
    _spec_path: Option<PathBuf>,
}

fn default_onnx_spec_protocol_version() -> u32 {
    1
}

fn resolve_sidecar_relative_path(spec_path: &Path, raw_path: &str) -> PathBuf {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        return candidate;
    }

    spec_path
        .parent()
        .map(|parent| parent.join(candidate.clone()))
        .unwrap_or(candidate)
}

fn resolve_tokenized_contract(
    spec: &OnnxModelSpec,
    spec_path: &Path,
) -> anyhow::Result<ResolvedOnnxTokenizedContract> {
    let tokenizer_path = spec
        .tokenizer_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("tokenized ONNX model spec requires `tokenizer_path`"))?;
    let tokenizer_path = resolve_sidecar_relative_path(spec_path, tokenizer_path);
    if !tokenizer_path.is_file() {
        anyhow::bail!(
            "ONNX tokenizer file was not found: {}",
            tokenizer_path.display()
        );
    }

    let max_length = spec
        .max_length
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow::anyhow!("tokenized ONNX model spec requires `max_length > 0`"))?;
    let inputs = spec
        .inputs
        .clone()
        .ok_or_else(|| anyhow::anyhow!("tokenized ONNX model spec requires `inputs`"))?;
    let outputs = spec
        .outputs
        .clone()
        .ok_or_else(|| anyhow::anyhow!("tokenized ONNX model spec requires `outputs`"))?;

    Ok(ResolvedOnnxTokenizedContract {
        tokenizer_path,
        max_length,
        inputs: ResolvedOnnxTokenizedInputs {
            input_ids: inputs.input_ids,
            attention_mask: inputs.attention_mask,
            token_type_ids: inputs.token_type_ids,
        },
        outputs: ResolvedOnnxTokenizedOutputs {
            start_probs: outputs.start_probs,
            end_probs: outputs.end_probs,
        },
    })
}

fn maybe_load_onnx_model_spec(
    model_path: &Path,
    explicit_spec_path: Option<&str>,
) -> anyhow::Result<Option<(OnnxModelSpec, PathBuf)>> {
    let implicit_path = model_path.with_extension("json");
    let spec_path = explicit_spec_path
        .map(PathBuf::from)
        .or_else(|| implicit_path.is_file().then_some(implicit_path));

    let Some(spec_path) = spec_path else {
        return Ok(None);
    };

    if !spec_path.is_file() {
        anyhow::bail!(
            "ONNX model spec file was not found: {}",
            spec_path.display()
        );
    }

    let raw = fs::read_to_string(&spec_path)?;
    let spec = serde_json::from_str::<OnnxModelSpec>(&raw)?;
    if spec.protocol_version != default_onnx_spec_protocol_version() {
        anyhow::bail!(
            "unsupported ONNX model spec protocol_version `{}`, expected `{}`",
            spec.protocol_version,
            default_onnx_spec_protocol_version()
        );
    }

    Ok(Some((spec, spec_path)))
}

fn resolve_onnx_model_contract(
    model_path: &Path,
    explicit_spec_path: Option<&str>,
    input_text_name: &str,
    input_schema_name: &str,
    output_json_name: &str,
) -> anyhow::Result<ResolvedOnnxModelContract> {
    let spec = maybe_load_onnx_model_spec(model_path, explicit_spec_path)?;

    Ok(match spec {
        Some((spec, spec_path)) => {
            let runtime_contract = spec.runtime_contract.clone();
            let tokenized = match runtime_contract {
                OnnxRuntimeContractKind::StringIo => None,
                OnnxRuntimeContractKind::Tokenized => {
                    Some(resolve_tokenized_contract(&spec, &spec_path)?)
                }
            };

            ResolvedOnnxModelContract {
                runtime_contract,
                input_text_name: spec
                    .input_text_name
                    .clone()
                    .unwrap_or_else(|| input_text_name.to_string()),
                input_schema_name: spec
                    .input_schema_name
                    .clone()
                    .unwrap_or_else(|| input_schema_name.to_string()),
                output_json_name: spec
                    .output_json_name
                    .clone()
                    .unwrap_or_else(|| output_json_name.to_string()),
                tokenized,
                _decode_strategy: spec.decode_strategy,
                _spec_path: Some(spec_path),
            }
        }
        None => ResolvedOnnxModelContract {
            runtime_contract: OnnxRuntimeContractKind::StringIo,
            input_text_name: input_text_name.to_string(),
            input_schema_name: input_schema_name.to_string(),
            output_json_name: output_json_name.to_string(),
            tokenized: None,
            _decode_strategy: OnnxDecodeStrategy::JsonStringScalar,
            _spec_path: None,
        },
    })
}

#[cfg(feature = "onnx-runtime")]
fn schema_to_json(schema: &SchemaSpec) -> anyhow::Result<String> {
    serde_json::to_string(schema).map_err(Into::into)
}

#[cfg(feature = "onnx-runtime")]
fn normalize_onnx_json_result(
    value: &Value,
    doc: &DocumentIr,
    schema: &SchemaSpec,
) -> anyhow::Result<ExtractionResult> {
    let root = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("ONNX json_output must decode to a JSON object"))?;

    let fields = match root.get("fields") {
        Some(Value::Array(items)) => schema
            .fields
            .iter()
            .map(|field| {
                let matched = items.iter().find(|item| {
                    item.as_object()
                        .and_then(|object| object.get("key"))
                        .and_then(Value::as_str)
                        == Some(field.key.as_str())
                });

                matched
                    .map(|item| decode_onnx_field_item(field, item, doc))
                    .unwrap_or_else(|| Ok(empty_field_value(field)))
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        Some(other) => {
            anyhow::bail!("ONNX `fields` must be an array, got {other}")
        }
        None => schema
            .fields
            .iter()
            .map(|field| decode_onnx_field_from_object(field, root, doc))
            .collect::<anyhow::Result<Vec<_>>>()?,
    };

    Ok(ExtractionResult {
        task_id: Uuid::new_v4().to_string(),
        status: TaskStatus::Extracting,
        fields,
        raw_text: Some(doc.plain_text.clone()),
        timings: TimingBreakdown::default(),
    })
}

#[cfg(feature = "onnx-runtime")]
fn decode_onnx_field_from_object(
    field: &FieldSpec,
    root: &Map<String, Value>,
    doc: &DocumentIr,
) -> anyhow::Result<FieldValue> {
    let raw_value = root.get(&field.key);
    let confidence = root
        .get(&format!("{}_confidence", field.key))
        .and_then(parse_confidence_value);
    let evidences = root
        .get(&format!("{}_evidences", field.key))
        .or_else(|| root.get(&format!("{}_evidence", field.key)))
        .map(|value| decode_evidences_value(value, doc))
        .transpose()?
        .unwrap_or_else(|| infer_evidences_from_value(raw_value, doc));

    Ok(build_field_value(
        field,
        raw_value
            .cloned()
            .unwrap_or_else(|| default_value(&field.field_type)),
        confidence,
        evidences,
    ))
}

#[cfg(feature = "onnx-runtime")]
fn decode_onnx_field_item(
    field: &FieldSpec,
    item: &Value,
    doc: &DocumentIr,
) -> anyhow::Result<FieldValue> {
    let object = item
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("ONNX field item must be an object"))?;

    let value = object
        .get("value")
        .cloned()
        .unwrap_or_else(|| default_value(&field.field_type));
    let confidence = object
        .get("confidence")
        .and_then(parse_confidence_value)
        .or_else(|| object.get("score").and_then(parse_confidence_value));
    let evidences = object
        .get("evidences")
        .or_else(|| object.get("evidence"))
        .map(|value| decode_evidences_value(value, doc))
        .transpose()?
        .unwrap_or_else(|| infer_evidences_from_value(Some(&value), doc));

    Ok(build_field_value(field, value, confidence, evidences))
}

#[cfg(feature = "onnx-runtime")]
fn build_field_value(
    field: &FieldSpec,
    value: Value,
    confidence: Option<f32>,
    evidences: Vec<Evidence>,
) -> FieldValue {
    let confidence =
        confidence.or_else(|| Some(compute_confidence(&value, field.required, evidences.len())));

    FieldValue {
        key: field.key.clone(),
        value,
        confidence,
        evidences: dedupe_evidences(evidences),
    }
}

#[cfg(feature = "onnx-runtime")]
fn empty_field_value(field: &FieldSpec) -> FieldValue {
    FieldValue {
        key: field.key.clone(),
        value: default_value(&field.field_type),
        confidence: Some(compute_confidence(
            &default_value(&field.field_type),
            field.required,
            0,
        )),
        evidences: vec![],
    }
}

#[cfg(feature = "onnx-runtime")]
fn parse_confidence_value(value: &Value) -> Option<f32> {
    match value {
        Value::Number(number) => number.as_f64().map(|value| value as f32),
        Value::String(text) => text.trim().parse::<f32>().ok(),
        _ => None,
    }
    .map(|value| value.clamp(0.0, 1.0))
}

#[cfg(feature = "onnx-runtime")]
fn decode_evidences_value(value: &Value, doc: &DocumentIr) -> anyhow::Result<Vec<Evidence>> {
    match value {
        Value::Null => Ok(vec![]),
        Value::String(text) => Ok(infer_evidences_from_text(text, doc)),
        Value::Array(items) => {
            let mut evidences = Vec::new();
            for item in items {
                evidences.extend(decode_single_evidence(item, doc)?);
            }
            Ok(dedupe_evidences(evidences))
        }
        Value::Object(_) => decode_single_evidence(value, doc),
        other => anyhow::bail!("unsupported evidence payload: {other}"),
    }
}

#[cfg(feature = "onnx-runtime")]
fn decode_single_evidence(value: &Value, doc: &DocumentIr) -> anyhow::Result<Vec<Evidence>> {
    match value {
        Value::String(text) => Ok(infer_evidences_from_text(text, doc)),
        Value::Object(object) => {
            let text = object
                .get("text")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToString::to_string);
            let mut page_no = object
                .get("page_no")
                .and_then(Value::as_u64)
                .map(|value| value as u32);
            let block_ids = object
                .get("source_block_ids")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let mut evidences = if !block_ids.is_empty() {
                find_evidences_by_block_ids(doc, &block_ids)
            } else if let Some(text) = text.as_deref() {
                infer_evidences_from_text(text, doc)
            } else {
                vec![]
            };

            if evidences.is_empty() {
                if let Some(text) = text {
                    evidences.push(Evidence {
                        page_no,
                        text,
                        bbox: None,
                        source_block_ids: block_ids,
                    });
                }
            } else {
                if let Some(text) = text.as_deref() {
                    for evidence in &mut evidences {
                        if evidence.text.trim().is_empty() {
                            evidence.text = text.to_string();
                        }
                    }
                }

                if let Some(explicit_page_no) = page_no.take() {
                    for evidence in &mut evidences {
                        evidence.page_no = Some(explicit_page_no);
                    }
                }
            }

            Ok(dedupe_evidences(evidences))
        }
        other => anyhow::bail!("unsupported evidence item: {other}"),
    }
}

#[cfg(feature = "onnx-runtime")]
fn infer_evidences_from_value(value: Option<&Value>, doc: &DocumentIr) -> Vec<Evidence> {
    let Some(value) = value else {
        return vec![];
    };

    let mut snippets = Vec::new();
    collect_value_snippets(value, &mut snippets);

    let mut evidences = Vec::new();
    for snippet in snippets {
        evidences.extend(infer_evidences_from_text(&snippet, doc));
    }
    dedupe_evidences(evidences)
}

#[cfg(feature = "onnx-runtime")]
fn collect_value_snippets(value: &Value, snippets: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            let text = text.trim();
            if !text.is_empty() {
                snippets.push(text.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_value_snippets(item, snippets);
            }
        }
        Value::Object(map) => {
            for item in map.values() {
                collect_value_snippets(item, snippets);
            }
        }
        Value::Number(number) => snippets.push(number.to_string()),
        Value::Bool(boolean) => snippets.push(boolean.to_string()),
        Value::Null => {}
    }
}

#[cfg(feature = "onnx-runtime")]
fn infer_evidences_from_text(text: &str, doc: &DocumentIr) -> Vec<Evidence> {
    let needle = text.trim();
    if needle.is_empty() {
        return vec![];
    }

    let normalized_needle = needle.to_ascii_lowercase();
    let mut evidences = Vec::new();
    for page in &doc.pages {
        for block in &page.blocks {
            if block.text.to_ascii_lowercase().contains(&normalized_needle) {
                evidences.push(evidence_from_block(block));
            }
        }
    }

    if evidences.is_empty()
        && doc
            .plain_text
            .to_ascii_lowercase()
            .contains(&normalized_needle)
    {
        evidences.push(Evidence {
            page_no: None,
            text: needle.to_string(),
            bbox: None,
            source_block_ids: vec![],
        });
    }

    dedupe_evidences(evidences)
}

#[cfg(feature = "onnx-runtime")]
fn find_evidences_by_block_ids(doc: &DocumentIr, block_ids: &[String]) -> Vec<Evidence> {
    let expected = block_ids
        .iter()
        .map(|block_id| block_id.trim())
        .filter(|block_id| !block_id.is_empty())
        .collect::<HashSet<_>>();
    if expected.is_empty() {
        return vec![];
    }

    let mut evidences = Vec::new();
    for page in &doc.pages {
        for block in &page.blocks {
            if expected.contains(block.block_id.as_str()) {
                evidences.push(evidence_from_block(block));
            }
        }
    }
    dedupe_evidences(evidences)
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug, Default)]
struct OnnxInputs {
    items: Vec<OnnxNamedInput>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct OnnxNamedInput {
    name: String,
    payload: OnnxInputPayload,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
enum OnnxInputPayload {
    StringScalar(String),
    Int64Tensor { values: Vec<i64>, shape: Vec<i64> },
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug, Default)]
struct OnnxOutputs {
    items: HashMap<String, OnnxOutputPayload>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
enum OnnxOutputPayload {
    StringTensor(Vec<String>),
    FloatTensor { values: Vec<f32>, shape: Vec<usize> },
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct OnnxPreparedInputs {
    inputs: OnnxInputs,
    context: OnnxPreparedContext,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug, Default)]
enum OnnxPreparedContext {
    #[default]
    None,
    Uie(UiePreparedContext),
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug, Default)]
struct UiePreparedContext {
    plans: Vec<UieFieldWindowPlan>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct UieFieldWindowPlan {
    field_path: Vec<String>,
    token_spans: Vec<Option<UieTokenSpan>>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct UieTokenSpan {
    start_char: usize,
    end_char: usize,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct UieSpanCandidate {
    start_char: usize,
    end_char: usize,
    confidence: f32,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct UiePredictionItem {
    raw_value: String,
    confidence: f32,
    evidences: Vec<Evidence>,
    start_char: usize,
    end_char: usize,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct FloatTensorView<'a> {
    values: &'a [f32],
    batch_size: usize,
    sequence_len: usize,
}

#[cfg(feature = "onnx-runtime")]
impl<'a> FloatTensorView<'a> {
    fn new(values: &'a [f32], shape: &'a [usize]) -> anyhow::Result<Self> {
        let (batch_size, sequence_len) = match shape {
            [] => (1, 1),
            [sequence_len] => (1, *sequence_len),
            [batch_size, sequence_len] => (*batch_size, *sequence_len),
            [batch_size, sequence_len, 1] => (*batch_size, *sequence_len),
            [batch_size, 1, sequence_len] => (*batch_size, *sequence_len),
            other => anyhow::bail!(
                "unsupported UIE output tensor rank `{}` with shape {:?}",
                other.len(),
                other,
            ),
        };

        let expected_len = batch_size.saturating_mul(sequence_len);
        if expected_len != values.len() {
            anyhow::bail!(
                "UIE output tensor shape {:?} expects {} values, got {}",
                shape,
                expected_len,
                values.len(),
            )
        }

        Ok(Self {
            values,
            batch_size,
            sequence_len,
        })
    }

    fn row(&self, batch_index: usize) -> anyhow::Result<&'a [f32]> {
        if batch_index >= self.batch_size {
            anyhow::bail!(
                "UIE output row `{batch_index}` is out of bounds for batch size `{}`",
                self.batch_size,
            )
        }

        let start = batch_index.saturating_mul(self.sequence_len);
        let end = start.saturating_add(self.sequence_len);
        Ok(&self.values[start..end])
    }
}

#[cfg(feature = "onnx-runtime")]
trait OnnxInputAdapter: Send + Sync {
    fn build_inputs(
        &self,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<OnnxPreparedInputs>;
}

#[cfg(feature = "onnx-runtime")]
trait OnnxOutputAdapter: Send + Sync {
    fn output_names(&self) -> &[String];
    fn decode(
        &self,
        output: OnnxOutputs,
        context: &OnnxPreparedContext,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult>;
}

#[cfg(feature = "onnx-runtime")]
#[derive(Debug)]
struct JsonStringInputAdapter {
    input_text_name: String,
    input_schema_name: String,
}

#[cfg(feature = "onnx-runtime")]
impl OnnxInputAdapter for JsonStringInputAdapter {
    fn build_inputs(
        &self,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<OnnxPreparedInputs> {
        Ok(OnnxPreparedInputs {
            inputs: OnnxInputs {
                items: vec![
                    OnnxNamedInput {
                        name: self.input_text_name.clone(),
                        payload: OnnxInputPayload::StringScalar(doc.plain_text.clone()),
                    },
                    OnnxNamedInput {
                        name: self.input_schema_name.clone(),
                        payload: OnnxInputPayload::StringScalar(schema_to_json(schema)?),
                    },
                ],
            },
            context: OnnxPreparedContext::None,
        })
    }
}

#[cfg(feature = "onnx-runtime")]
#[derive(Debug)]
struct JsonStringOutputAdapter {
    output_json_name: String,
    output_names: Vec<String>,
}

#[cfg(feature = "onnx-runtime")]
impl JsonStringOutputAdapter {
    fn new(output_json_name: String) -> Self {
        Self {
            output_names: vec![output_json_name.clone()],
            output_json_name,
        }
    }
}

#[cfg(feature = "onnx-runtime")]
impl OnnxOutputAdapter for JsonStringOutputAdapter {
    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn decode(
        &self,
        output: OnnxOutputs,
        _context: &OnnxPreparedContext,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult> {
        let payload = output.items.get(&self.output_json_name).ok_or_else(|| {
            anyhow::anyhow!("ONNX output `{}` was not returned", self.output_json_name)
        })?;
        let output_json = match payload {
            OnnxOutputPayload::StringTensor(values) => {
                values.first().cloned().ok_or_else(|| {
                    anyhow::anyhow!("ONNX output `{}` is empty", self.output_json_name)
                })?
            }
            other => anyhow::bail!(
                "ONNX output `{}` expected a string tensor, got {:?}",
                self.output_json_name,
                other,
            ),
        };

        let parsed: Value = serde_json::from_str(&output_json).map_err(|error| {
            anyhow::anyhow!(
                "failed to parse ONNX json_output as JSON: {error}; raw output: {output_json}"
            )
        })?;

        normalize_onnx_json_result(&parsed, doc, schema)
    }
}

#[cfg(feature = "onnx-runtime")]
#[derive(Debug)]
struct TokenizedUieInputAdapter {
    tokenizer: Tokenizer,
    input_ids_name: String,
    attention_mask_name: String,
    token_type_ids_name: Option<String>,
    max_length: usize,
}

#[cfg(feature = "onnx-runtime")]
impl TokenizedUieInputAdapter {
    fn new(contract: &ResolvedOnnxTokenizedContract) -> anyhow::Result<Self> {
        let mut tokenizer = Tokenizer::from_file(&contract.tokenizer_path).map_err(|error| {
            anyhow::anyhow!(
                "failed to load tokenizer from `{}`: {error}",
                contract.tokenizer_path.display(),
            )
        })?;
        configure_tokenizer_for_uie(&mut tokenizer, contract.max_length)?;

        Ok(Self {
            tokenizer,
            input_ids_name: contract.inputs.input_ids.clone(),
            attention_mask_name: contract.inputs.attention_mask.clone(),
            token_type_ids_name: contract.inputs.token_type_ids.clone(),
            max_length: contract.max_length,
        })
    }
}

#[cfg(feature = "onnx-runtime")]
impl OnnxInputAdapter for TokenizedUieInputAdapter {
    fn build_inputs(
        &self,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<OnnxPreparedInputs> {
        let leaf_fields = collect_schema_leaf_fields(schema);
        if leaf_fields.is_empty() {
            anyhow::bail!("schema does not contain any leaf fields for UIE extraction")
        }

        let mut input_ids = Vec::new();
        let mut attention_mask = Vec::new();
        let mut token_type_ids = Vec::new();
        let mut plans = Vec::new();

        for leaf in leaf_fields {
            let prompt = build_uie_prompt(&leaf.path, leaf.field);
            let mut encoding = self
                .tokenizer
                .encode_char_offsets((prompt.as_str(), doc.plain_text.as_str()), true)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "failed to tokenize UIE prompt `{}`: {error}",
                        leaf.path.join("/"),
                    )
                })?;
            let mut windows = Vec::with_capacity(1 + encoding.get_overflowing().len());
            let overflowings = encoding.take_overflowing();
            windows.push(encoding);
            windows.extend(overflowings);

            for window in windows {
                if window.get_ids().len() != self.max_length {
                    anyhow::bail!(
                        "tokenizer produced sequence length `{}` but runtime expected fixed max_length `{}`",
                        window.get_ids().len(),
                        self.max_length,
                    )
                }

                input_ids.extend(window.get_ids().iter().map(|value| *value as i64));
                attention_mask.extend(window.get_attention_mask().iter().map(|value| *value as i64));
                if self.token_type_ids_name.is_some() {
                    token_type_ids.extend(window.get_type_ids().iter().map(|value| *value as i64));
                }

                let sequence_ids = window.get_sequence_ids();
                let token_spans = window
                    .get_offsets()
                    .iter()
                    .zip(sequence_ids.into_iter())
                    .map(|((start_char, end_char), sequence_id)| match sequence_id {
                        Some(1) if end_char > start_char => Some(UieTokenSpan {
                            start_char: *start_char,
                            end_char: *end_char,
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                plans.push(UieFieldWindowPlan {
                    field_path: leaf.path.clone(),
                    token_spans,
                });
            }
        }

        let batch_size = i64::try_from(plans.len())?;
        let sequence_len = i64::try_from(self.max_length)?;
        let mut items = vec![
            OnnxNamedInput {
                name: self.input_ids_name.clone(),
                payload: OnnxInputPayload::Int64Tensor {
                    values: input_ids,
                    shape: vec![batch_size, sequence_len],
                },
            },
            OnnxNamedInput {
                name: self.attention_mask_name.clone(),
                payload: OnnxInputPayload::Int64Tensor {
                    values: attention_mask,
                    shape: vec![batch_size, sequence_len],
                },
            },
        ];
        if let Some(token_type_ids_name) = &self.token_type_ids_name {
            items.push(OnnxNamedInput {
                name: token_type_ids_name.clone(),
                payload: OnnxInputPayload::Int64Tensor {
                    values: token_type_ids,
                    shape: vec![batch_size, sequence_len],
                },
            });
        }

        Ok(OnnxPreparedInputs {
            inputs: OnnxInputs { items },
            context: OnnxPreparedContext::Uie(UiePreparedContext { plans }),
        })
    }
}

#[cfg(feature = "onnx-runtime")]
#[derive(Debug)]
struct UieSpanOutputAdapter {
    start_probs_name: String,
    end_probs_name: String,
    output_names: Vec<String>,
}

#[cfg(feature = "onnx-runtime")]
impl UieSpanOutputAdapter {
    fn new(start_probs_name: String, end_probs_name: String) -> Self {
        Self {
            output_names: vec![start_probs_name.clone(), end_probs_name.clone()],
            start_probs_name,
            end_probs_name,
        }
    }
}

#[cfg(feature = "onnx-runtime")]
impl OnnxOutputAdapter for UieSpanOutputAdapter {
    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn decode(
        &self,
        output: OnnxOutputs,
        context: &OnnxPreparedContext,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult> {
        let context = match context {
            OnnxPreparedContext::Uie(context) => context,
            other => anyhow::bail!("UIE output adapter received unexpected context: {:?}", other),
        };

        let start_payload = output.items.get(&self.start_probs_name).ok_or_else(|| {
            anyhow::anyhow!("ONNX output `{}` was not returned", self.start_probs_name)
        })?;
        let end_payload = output.items.get(&self.end_probs_name).ok_or_else(|| {
            anyhow::anyhow!("ONNX output `{}` was not returned", self.end_probs_name)
        })?;

        let (start_values, start_shape) = match start_payload {
            OnnxOutputPayload::FloatTensor { values, shape } => (values.as_slice(), shape.as_slice()),
            other => anyhow::bail!(
                "ONNX output `{}` expected a float tensor, got {:?}",
                self.start_probs_name,
                other,
            ),
        };
        let (end_values, end_shape) = match end_payload {
            OnnxOutputPayload::FloatTensor { values, shape } => (values.as_slice(), shape.as_slice()),
            other => anyhow::bail!(
                "ONNX output `{}` expected a float tensor, got {:?}",
                self.end_probs_name,
                other,
            ),
        };

        let start_view = FloatTensorView::new(start_values, start_shape)?;
        let end_view = FloatTensorView::new(end_values, end_shape)?;
        if start_view.batch_size != context.plans.len() || end_view.batch_size != context.plans.len() {
            anyhow::bail!(
                "UIE output batch size mismatch: start={}, end={}, plans={}",
                start_view.batch_size,
                end_view.batch_size,
                context.plans.len(),
            )
        }

        let mut predictions: HashMap<String, Vec<UiePredictionItem>> = HashMap::new();
        for (batch_index, plan) in context.plans.iter().enumerate() {
            let start_row = start_view.row(batch_index)?;
            let end_row = end_view.row(batch_index)?;
            let decoded = finalize_uie_predictions(
                decode_uie_window_spans(plan, start_row, end_row),
                doc,
            );
            if decoded.is_empty() {
                continue;
            }

            predictions
                .entry(field_path_cache_key(&plan.field_path))
                .or_default()
                .extend(decoded);
        }

        let mut path = Vec::new();
        let fields = schema
            .fields
            .iter()
            .map(|field| build_uie_field_value(field, &mut path, &predictions))
            .collect::<Vec<_>>();

        Ok(ExtractionResult {
            task_id: Uuid::new_v4().to_string(),
            status: TaskStatus::Extracting,
            fields,
            raw_text: Some(doc.plain_text.clone()),
            timings: TimingBreakdown::default(),
        })
    }
}

#[cfg(feature = "onnx-runtime")]
struct OnnxRuntimePipeline {
    provider_name: &'static str,
    input_adapter: Box<dyn OnnxInputAdapter>,
    output_adapter: Box<dyn OnnxOutputAdapter>,
}

#[cfg(feature = "onnx-runtime")]
fn ensure_supported_runtime_contract(contract: &ResolvedOnnxModelContract) -> anyhow::Result<()> {
    match (&contract.runtime_contract, &contract._decode_strategy) {
        (OnnxRuntimeContractKind::StringIo, OnnxDecodeStrategy::JsonStringScalar) => Ok(()),
        (OnnxRuntimeContractKind::Tokenized, OnnxDecodeStrategy::UieSpan) => {
            let _ = contract.tokenized.as_ref().ok_or_else(|| {
                anyhow::anyhow!("tokenized ONNX contract metadata is missing after spec resolution")
            })?;
            Ok(())
        }
        (OnnxRuntimeContractKind::StringIo, other) => anyhow::bail!(
            "string_io runtime_contract does not support decode_strategy `{:?}`",
            other,
        ),
        (OnnxRuntimeContractKind::Tokenized, other) => anyhow::bail!(
            "tokenized runtime_contract does not support decode_strategy `{:?}` yet",
            other,
        ),
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_onnx_runtime_pipeline(
    contract: &ResolvedOnnxModelContract,
    inputs: &[OrtIoInfo],
    outputs: &[OrtIoInfo],
) -> anyhow::Result<OnnxRuntimePipeline> {
    ensure_supported_runtime_contract(contract)?;

    match contract.runtime_contract {
        OnnxRuntimeContractKind::StringIo => {
            validate_onnx_string_contract(inputs, outputs, contract)?;
            Ok(OnnxRuntimePipeline {
                provider_name: "onnx-runtime-cpu-json-contract",
                input_adapter: Box::new(JsonStringInputAdapter {
                    input_text_name: contract.input_text_name.clone(),
                    input_schema_name: contract.input_schema_name.clone(),
                }),
                output_adapter: Box::new(JsonStringOutputAdapter::new(
                    contract.output_json_name.clone(),
                )),
            })
        }
        OnnxRuntimeContractKind::Tokenized => {
            let tokenized = contract.tokenized.as_ref().ok_or_else(|| {
                anyhow::anyhow!("tokenized ONNX contract metadata is missing after spec resolution")
            })?;
            validate_onnx_tokenized_contract(inputs, outputs, tokenized)?;

            Ok(OnnxRuntimePipeline {
                provider_name: "onnx-runtime-cpu-tokenized-uie",
                input_adapter: Box::new(TokenizedUieInputAdapter::new(tokenized)?),
                output_adapter: Box::new(UieSpanOutputAdapter::new(
                    tokenized.outputs.start_probs.clone(),
                    tokenized.outputs.end_probs.clone(),
                )),
            })
        }
    }
}

#[cfg(feature = "onnx-runtime")]
fn configure_tokenizer_for_uie(tokenizer: &mut Tokenizer, max_length: usize) -> anyhow::Result<()> {
    let (pad_id, pad_token) = resolve_tokenizer_padding(tokenizer);
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(max_length),
        direction: TokenizerPaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id,
        pad_type_id: 0,
        pad_token,
    }));

    let mut stride = suggested_tokenized_stride(max_length);
    loop {
        let truncation = TruncationParams {
            direction: TruncationDirection::Right,
            max_length,
            strategy: TruncationStrategy::OnlySecond,
            stride,
        };
        match tokenizer.with_truncation(Some(truncation)) {
            Ok(_) => return Ok(()),
            Err(_) if stride > 0 => {
                stride /= 2;
            }
            Err(error) => anyhow::bail!("failed to configure tokenizer truncation: {error}"),
        }
    }
}

#[cfg(feature = "onnx-runtime")]
fn resolve_tokenizer_padding(tokenizer: &Tokenizer) -> (u32, String) {
    if let Some(padding) = tokenizer.get_padding() {
        return (padding.pad_id, padding.pad_token.clone());
    }

    for token in ["[PAD]", "<pad>", "<PAD>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            return (id, token.to_string());
        }
    }

    let pad_id = 0;
    let pad_token = tokenizer
        .id_to_token(pad_id)
        .unwrap_or_else(|| "[PAD]".to_string());
    (pad_id, pad_token)
}

#[cfg(feature = "onnx-runtime")]
fn suggested_tokenized_stride(max_length: usize) -> usize {
    if max_length <= 1 {
        return 0;
    }

    let mut stride = max_length / 4;
    if stride > 64 {
        stride = 64;
    }
    if stride >= max_length {
        stride = max_length.saturating_sub(1);
    }
    stride
}

#[cfg(feature = "onnx-runtime")]
struct SchemaLeafField<'a> {
    path: Vec<String>,
    field: &'a FieldSpec,
}

#[cfg(feature = "onnx-runtime")]
fn collect_schema_leaf_fields<'a>(schema: &'a SchemaSpec) -> Vec<SchemaLeafField<'a>> {
    let mut leaves = Vec::new();
    let mut path = Vec::new();
    collect_schema_leaf_fields_from_list(&schema.fields, &mut path, &mut leaves);
    leaves
}

#[cfg(feature = "onnx-runtime")]
fn collect_schema_leaf_fields_from_list<'a>(
    fields: &'a [FieldSpec],
    path: &mut Vec<String>,
    leaves: &mut Vec<SchemaLeafField<'a>>,
) {
    for field in fields {
        path.push(field.key.clone());
        if field.children.is_empty() {
            leaves.push(SchemaLeafField {
                path: path.clone(),
                field,
            });
        } else {
            collect_schema_leaf_fields_from_list(&field.children, path, leaves);
        }
        let _ = path.pop();
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_uie_prompt(path: &[String], field: &FieldSpec) -> String {
    let mut parts = Vec::new();
    if !path.is_empty() {
        parts.push(path.join("/"));
    } else {
        parts.push(field.key.clone());
    }
    if !field.hints.is_empty() {
        parts.push(format!("别名：{}", field.hints.join("，")));
    }
    parts.join("；")
}

#[cfg(feature = "onnx-runtime")]
fn field_path_cache_key(path: &[String]) -> String {
    path.join("::")
}

#[cfg(feature = "onnx-runtime")]
fn decode_uie_window_spans(
    plan: &UieFieldWindowPlan,
    start_row: &[f32],
    end_row: &[f32],
) -> Vec<UieSpanCandidate> {
    const UIE_THRESHOLD: f32 = 0.5;
    const MAX_TOKEN_SPAN: usize = 32;
    const MAX_CANDIDATES_PER_WINDOW: usize = 8;

    let limit = plan
        .token_spans
        .len()
        .min(start_row.len())
        .min(end_row.len());
    let mut start_candidates = Vec::new();
    let mut end_candidates = Vec::new();

    for index in 0..limit {
        let Some(span) = plan.token_spans[index].as_ref() else {
            continue;
        };
        if start_row[index] >= UIE_THRESHOLD {
            start_candidates.push((index, span, start_row[index]));
        }
        if end_row[index] >= UIE_THRESHOLD {
            end_candidates.push((index, span, end_row[index]));
        }
    }

    let mut candidates = Vec::new();
    for (start_index, start_span, start_prob) in &start_candidates {
        for (end_index, end_span, end_prob) in &end_candidates {
            if end_index < start_index || end_index.saturating_sub(*start_index) > MAX_TOKEN_SPAN {
                continue;
            }
            if end_span.end_char <= start_span.start_char {
                continue;
            }

            candidates.push(UieSpanCandidate {
                start_char: start_span.start_char,
                end_char: end_span.end_char,
                confidence: (start_prob * end_prob).sqrt().clamp(0.0, 1.0),
            });
        }
    }

    candidates.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.start_char.cmp(&right.start_char))
            .then_with(|| left.end_char.cmp(&right.end_char))
    });
    candidates.dedup_by(|left, right| {
        left.start_char == right.start_char && left.end_char == right.end_char
    });
    candidates.truncate(MAX_CANDIDATES_PER_WINDOW);
    candidates
}

#[cfg(feature = "onnx-runtime")]
fn finalize_uie_predictions(
    candidates: Vec<UieSpanCandidate>,
    doc: &DocumentIr,
) -> Vec<UiePredictionItem> {
    let mut items = Vec::new();

    for candidate in candidates {
        let Some(raw_value) = slice_text_by_char_range(
            &doc.plain_text,
            candidate.start_char,
            candidate.end_char,
        ) else {
            continue;
        };
        let raw_value = raw_value.trim().to_string();
        if raw_value.is_empty() {
            continue;
        }

        let evidences = {
            let inferred = infer_evidences_from_text(&raw_value, doc);
            if inferred.is_empty() {
                vec![Evidence {
                    page_no: None,
                    text: raw_value.clone(),
                    bbox: None,
                    source_block_ids: vec![],
                }]
            } else {
                inferred
            }
        };

        items.push(UiePredictionItem {
            raw_value,
            confidence: candidate.confidence,
            evidences,
            start_char: candidate.start_char,
            end_char: candidate.end_char,
        });
    }

    items.sort_by(|left, right| {
        left.start_char
            .cmp(&right.start_char)
            .then_with(|| left.end_char.cmp(&right.end_char))
            .then_with(|| right.confidence.total_cmp(&left.confidence))
    });
    items.dedup_by(|left, right| {
        left.start_char == right.start_char && left.end_char == right.end_char
    });
    items
}

#[cfg(feature = "onnx-runtime")]
fn slice_text_by_char_range(text: &str, start_char: usize, end_char: usize) -> Option<String> {
    if end_char <= start_char {
        return None;
    }

    let char_count = text.chars().count();
    if start_char >= char_count || end_char > char_count {
        return None;
    }

    let start_byte = text.char_indices().nth(start_char).map(|(idx, _)| idx)?;
    let end_byte = if end_char == char_count {
        text.len()
    } else {
        text.char_indices().nth(end_char).map(|(idx, _)| idx)?
    };
    text.get(start_byte..end_byte).map(ToString::to_string)
}

#[cfg(feature = "onnx-runtime")]
fn build_uie_field_value(
    field: &FieldSpec,
    path: &mut Vec<String>,
    predictions: &HashMap<String, Vec<UiePredictionItem>>,
) -> FieldValue {
    path.push(field.key.clone());
    let result = if field.children.is_empty() {
        build_uie_leaf_field_value(field, path, predictions)
    } else {
        let children = field
            .children
            .iter()
            .map(|child| build_uie_field_value(child, path, predictions))
            .collect::<Vec<_>>();
        let evidences = dedupe_evidences(
            children
                .iter()
                .flat_map(|child| child.evidences.clone())
                .collect(),
        );

        let value = match field.field_type {
            FieldType::Array => build_object_array(&children),
            _ => Value::Object(
                children
                    .iter()
                    .map(|child| (child.key.clone(), child.value.clone()))
                    .collect::<Map<String, Value>>(),
            ),
        };

        let meaningful_children = children
            .iter()
            .filter(|child| has_meaningful_value(&child.value))
            .count();
        let confidence = if meaningful_children == 0 {
            Some(compute_confidence(&value, field.required, evidences.len()))
        } else {
            Some(
                (children
                    .iter()
                    .filter_map(|child| child.confidence)
                    .sum::<f32>()
                    / meaningful_children as f32)
                    .clamp(0.0, 0.99),
            )
        };

        FieldValue {
            key: field.key.clone(),
            value,
            confidence,
            evidences,
        }
    };
    let _ = path.pop();
    result
}

#[cfg(feature = "onnx-runtime")]
fn build_uie_leaf_field_value(
    field: &FieldSpec,
    path: &[String],
    predictions: &HashMap<String, Vec<UiePredictionItem>>,
) -> FieldValue {
    let prediction_items = predictions
        .get(&field_path_cache_key(path))
        .cloned()
        .unwrap_or_default();
    if prediction_items.is_empty() {
        return empty_field_value(field);
    }

    if field.multiple {
        let mut values = Vec::new();
        let mut confidences = Vec::new();
        let mut evidences = Vec::new();
        for item in prediction_items {
            let Some(value) = coerce_value(&item.raw_value, &field.field_type) else {
                continue;
            };
            if !values.iter().any(|existing| existing == &value) {
                values.push(value);
                confidences.push(item.confidence);
                evidences.extend(item.evidences);
            }
        }

        if values.is_empty() {
            return empty_field_value(field);
        }

        let confidence = Some(
            (confidences.iter().copied().sum::<f32>() / confidences.len() as f32)
                .clamp(0.0, 1.0),
        );
        return FieldValue {
            key: field.key.clone(),
            value: Value::Array(values),
            confidence,
            evidences: dedupe_evidences(evidences),
        };
    }

    let best_item = prediction_items
        .into_iter()
        .filter_map(|item| {
            coerce_value(&item.raw_value, &field.field_type).map(|value| (item, value))
        })
        .max_by(|(left_item, _), (right_item, _)| {
            left_item.confidence.total_cmp(&right_item.confidence)
        });

    let Some((best_item, value)) = best_item else {
        return empty_field_value(field);
    };

    FieldValue {
        key: field.key.clone(),
        value,
        confidence: Some(best_item.confidence.clamp(0.0, 1.0)),
        evidences: dedupe_evidences(best_item.evidences),
    }
}

#[async_trait]
impl Extractor for HeuristicExtractor {
    fn name(&self) -> &'static str {
        "heuristic-extractor"
    }

    async fn extract(
        &self,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult> {
        Ok(heuristic_extract(doc, schema))
    }
}

pub fn build_extractor(config: &Config) -> Arc<dyn Extractor> {
    match config.extractor_provider.as_str() {
        "heuristic" => Arc::new(HeuristicExtractor),
        "onnx" => {
            let extractor = OnnxRuntimeExtractor::from_config(config)
                .unwrap_or_else(|error| panic!("failed to configure ONNX extractor: {error}"));
            Arc::new(extractor)
        }
        other => panic!("unsupported extractor provider `{other}`"),
    }
}

pub struct OnnxRuntimeExtractor {
    inner: OnnxRuntimeBackend,
}

impl std::fmt::Debug for OnnxRuntimeExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxRuntimeExtractor")
            .field("provider", &self.inner.provider_name())
            .finish()
    }
}

impl OnnxRuntimeExtractor {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let model_path = config.onnx_model_path.clone().ok_or_else(|| {
            anyhow::anyhow!("`MUSE_ONNX_MODEL_PATH` is required when MUSE_EXTRACTOR_PROVIDER=onnx")
        })?;
        let model_path = PathBuf::from(model_path);
        let contract = resolve_onnx_model_contract(
            &model_path,
            config.onnx_model_spec_path.as_deref(),
            &config.onnx_input_text_name,
            &config.onnx_input_schema_name,
            &config.onnx_output_json_name,
        )?;

        Ok(Self {
            inner: OnnxRuntimeBackend::new(model_path, config.onnx_threads, contract)?,
        })
    }
}

#[async_trait]
impl Extractor for OnnxRuntimeExtractor {
    fn name(&self) -> &'static str {
        self.inner.provider_name()
    }

    async fn extract(
        &self,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult> {
        self.inner.extract(doc, schema).await
    }
}

#[cfg(feature = "onnx-runtime")]
type OrtStatusPtr = *mut ort_sys::OrtStatus;

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Debug)]
struct OrtIoInfo {
    name: String,
    tensor_type: u32,
    dimensions: Vec<Option<u32>>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Debug)]
struct OrtValueGuard {
    ptr: *mut ort_sys::OrtValue,
}

#[cfg(feature = "onnx-runtime")]
impl OrtValueGuard {
    fn new(ptr: *mut ort_sys::OrtValue) -> Self {
        Self { ptr }
    }

    fn as_const_ptr(&self) -> *const ort_sys::OrtValue {
        self.ptr as *const ort_sys::OrtValue
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
fn ort_api() -> &'static ort_sys::OrtApi {
    static API: OnceLock<ort_sys::OrtApi> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let base = ort_sys::OrtGetApiBase();
        assert!(!base.is_null(), "OrtGetApiBase returned null");
        let get_api = (*base)
            .GetApi
            .expect("OrtApiBase::GetApi must be available");
        let api_ptr = get_api(ort_sys::ORT_API_VERSION);
        assert!(!api_ptr.is_null(), "OrtApiBase::GetApi returned null");
        *api_ptr
    })
}

#[cfg(feature = "onnx-runtime")]
fn ort_status_to_result(status: OrtStatusPtr, context: &str) -> anyhow::Result<()> {
    if status.is_null() {
        return Ok(());
    }

    let message = unsafe {
        let message_ptr = ort_api().GetErrorMessage.unwrap()(status);
        let message = if message_ptr.is_null() {
            "unknown ONNX Runtime error".to_string()
        } else {
            CStr::from_ptr(message_ptr).to_string_lossy().into_owned()
        };
        ort_api().ReleaseStatus.unwrap()(status);
        message
    };

    anyhow::bail!("{context}: {message}")
}

#[cfg(feature = "onnx-runtime")]
fn normalize_string_input_shape(
    input_name: &str,
    dimensions: &[Option<u32>],
) -> anyhow::Result<Vec<usize>> {
    if dimensions.is_empty() {
        return Ok(vec![]);
    }

    let mut shape = Vec::with_capacity(dimensions.len());
    for (index, dimension) in dimensions.iter().enumerate() {
        match dimension {
            Some(value) if *value > 0 => shape.push(*value as usize),
            Some(_) => anyhow::bail!(
                "ONNX input `{input_name}` has unsupported zero-sized dimension at axis {index}"
            ),
            None if index == 0 => shape.push(1),
            None => anyhow::bail!(
                "ONNX input `{input_name}` has unsupported dynamic dimension at axis {index}; current string adapter only supports a dynamic batch axis"
            ),
        }
    }
    Ok(shape)
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
fn validate_onnx_string_contract(
    inputs: &[OrtIoInfo],
    outputs: &[OrtIoInfo],
    contract: &ResolvedOnnxModelContract,
) -> anyhow::Result<()> {
    validate_named_string_input(inputs, &contract.input_text_name)?;
    validate_named_string_input(inputs, &contract.input_schema_name)?;
    validate_named_string_output(outputs, &contract.output_json_name)?;
    Ok(())
}


#[cfg(feature = "onnx-runtime")]
fn validate_onnx_tokenized_contract(
    inputs: &[OrtIoInfo],
    outputs: &[OrtIoInfo],
    contract: &ResolvedOnnxTokenizedContract,
) -> anyhow::Result<()> {
    validate_named_int64_input(inputs, &contract.inputs.input_ids)?;
    validate_named_int64_input(inputs, &contract.inputs.attention_mask)?;
    if let Some(token_type_ids) = &contract.inputs.token_type_ids {
        validate_named_int64_input(inputs, token_type_ids)?;
    }
    validate_named_float_output(outputs, &contract.outputs.start_probs)?;
    validate_named_float_output(outputs, &contract.outputs.end_probs)?;
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
fn validate_named_int64_input(inputs: &[OrtIoInfo], expected_name: &str) -> anyhow::Result<()> {
    let input = inputs
        .iter()
        .find(|input| input.name == expected_name)
        .ok_or_else(|| anyhow::anyhow!("ONNX input `{expected_name}` was not found in model"))?;

    if input.tensor_type != ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
        anyhow::bail!("ONNX input `{expected_name}` must be `int64`")
    }

    validate_rank_at_least(&input.name, &input.dimensions, 1)
}

#[cfg(feature = "onnx-runtime")]
fn validate_named_float_output(outputs: &[OrtIoInfo], expected_name: &str) -> anyhow::Result<()> {
    let output = outputs
        .iter()
        .find(|output| output.name == expected_name)
        .ok_or_else(|| anyhow::anyhow!("ONNX output `{expected_name}` was not found in model"))?;

    if output.tensor_type != ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT {
        anyhow::bail!("ONNX output `{expected_name}` must be `float`")
    }

    validate_rank_at_least(&output.name, &output.dimensions, 1)
}

#[cfg(feature = "onnx-runtime")]
fn validate_rank_at_least(
    tensor_name: &str,
    dimensions: &[Option<u32>],
    min_rank: usize,
) -> anyhow::Result<()> {
    if dimensions.len() < min_rank {
        anyhow::bail!(
            "ONNX tensor `{tensor_name}` must have rank >= {min_rank}, got {}",
            dimensions.len(),
        )
    }
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
fn validate_named_string_input(inputs: &[OrtIoInfo], expected_name: &str) -> anyhow::Result<()> {
    let input = inputs
        .iter()
        .find(|input| input.name == expected_name)
        .ok_or_else(|| anyhow::anyhow!("ONNX input `{expected_name}` was not found in model"))?;

    if input.tensor_type != ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    {
        anyhow::bail!("ONNX input `{expected_name}` must be `string`")
    }

    Ok(())
}

#[cfg(feature = "onnx-runtime")]
fn validate_named_string_output(outputs: &[OrtIoInfo], expected_name: &str) -> anyhow::Result<()> {
    let output = outputs
        .iter()
        .find(|output| output.name == expected_name)
        .ok_or_else(|| anyhow::anyhow!("ONNX output `{expected_name}` was not found in model"))?;

    if output.tensor_type != ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    {
        anyhow::bail!("ONNX output `{expected_name}` must be `string`")
    }

    Ok(())
}

#[cfg(feature = "onnx-runtime")]
fn build_string_input_value(
    allocator_ptr: *mut ort_sys::OrtAllocator,
    input: &OrtIoInfo,
    value: &str,
) -> anyhow::Result<OrtValueGuard> {
    let shape = normalize_string_input_shape(&input.name, &input.dimensions)?;
    let shape_i64 = shape
        .iter()
        .map(|dimension| *dimension as i64)
        .collect::<Vec<_>>();
    let element_count = shape.iter().product::<usize>();

    let mut value_ptr = ptr::null_mut();
    ort_status_to_result(
        unsafe {
            ort_api().CreateTensorAsOrtValue.unwrap()(
                allocator_ptr,
                shape_i64.as_ptr(),
                shape_i64.len(),
                ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
                &mut value_ptr,
            )
        },
        &format!(
            "failed to create ONNX string tensor for input `{}`",
            input.name
        ),
    )?;

    let c_string = CString::new(value)?;
    let string_pointers = vec![c_string.as_ptr(); element_count.max(1)];
    ort_status_to_result(
        unsafe {
            ort_api().FillStringTensor.unwrap()(
                value_ptr,
                string_pointers.as_ptr(),
                string_pointers.len(),
            )
        },
        &format!(
            "failed to fill ONNX string tensor for input `{}`",
            input.name
        ),
    )?;

    Ok(OrtValueGuard::new(value_ptr))
}

#[cfg(feature = "onnx-runtime")]
fn read_string_tensor_output_values(
    value_ptr: *mut ort_sys::OrtValue,
) -> anyhow::Result<Vec<String>> {
    let mut tensor_info_ptr = ptr::null_mut();
    ort_status_to_result(
        unsafe { ort_api().GetTensorTypeAndShape.unwrap()(value_ptr, &mut tensor_info_ptr) },
        "failed to read ONNX output tensor shape",
    )?;

    let dimensions = extract_dimensions(tensor_info_ptr)?;
    unsafe {
        ort_api().ReleaseTensorTypeAndShapeInfo.unwrap()(tensor_info_ptr);
    }

    let element_count = if dimensions.is_empty() {
        1
    } else {
        dimensions
            .into_iter()
            .map(|dimension| {
                dimension.ok_or_else(|| {
                    anyhow::anyhow!("ONNX output tensor contains an unexpected dynamic dimension")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .fold(1_usize, |count, dimension| {
                count.saturating_mul(dimension as usize)
            })
    };

    if element_count == 0 {
        anyhow::bail!("ONNX string output tensor is empty")
    }

    let mut values = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let mut length = 0;
        ort_status_to_result(
            unsafe {
                ort_api().GetStringTensorElementLength.unwrap()(value_ptr, index, &mut length)
            },
            "failed to read ONNX string output length",
        )?;

        let mut buffer = vec![0_u8; length];
        ort_status_to_result(
            unsafe {
                ort_api().GetStringTensorElement.unwrap()(
                    value_ptr,
                    length,
                    index,
                    buffer.as_mut_ptr().cast::<c_void>(),
                )
            },
            "failed to read ONNX string output contents",
        )?;
        values.push(String::from_utf8(buffer)?);
    }

    Ok(values)
}

#[cfg(feature = "onnx-runtime")]
fn build_ort_input_value(
    allocator_ptr: *mut ort_sys::OrtAllocator,
    input: &OrtIoInfo,
    payload: &OnnxInputPayload,
) -> anyhow::Result<OrtValueGuard> {
    match payload {
        OnnxInputPayload::StringScalar(value) => {
            build_string_input_value(allocator_ptr, input, value)
        }
    }
}

#[cfg(feature = "onnx-runtime")]
fn decode_ort_output_value(
    output: &OrtIoInfo,
    value_ptr: *mut ort_sys::OrtValue,
) -> anyhow::Result<OnnxOutputPayload> {
    match output.tensor_type {
        ort_sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => Ok(
            OnnxOutputPayload::StringTensor(read_string_tensor_output_values(value_ptr)?),
        ),
        other => anyhow::bail!(
            "ONNX output `{}` returned unsupported tensor type `{}` for current runtime adapter",
            output.name,
            other,
        ),
    }
}

#[cfg(feature = "onnx-runtime")]
struct OnnxSession {
    _model_path: PathBuf,
    env_ptr: *mut ort_sys::OrtEnv,
    session_ptr: *mut ort_sys::OrtSession,
    allocator_ptr: *mut ort_sys::OrtAllocator,
    inputs: Vec<OrtIoInfo>,
    outputs: Vec<OrtIoInfo>,
    run_lock: std::sync::Mutex<()>,
}

#[cfg(feature = "onnx-runtime")]
unsafe impl Send for OnnxSession {}

#[cfg(feature = "onnx-runtime")]
unsafe impl Sync for OnnxSession {}

#[cfg(feature = "onnx-runtime")]
impl Drop for OnnxSession {
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
impl OnnxSession {
    fn new(model_path: PathBuf, intra_threads: usize) -> anyhow::Result<Self> {
        if !model_path.is_file() {
            anyhow::bail!("ONNX model file was not found: {}", model_path.display());
        }

        let env_name = CString::new("muse-extractor")?;
        let mut env_ptr = ptr::null_mut();
        ort_status_to_result(
            unsafe {
                ort_api().CreateEnv.unwrap()(
                    ort_sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
                    env_name.as_ptr(),
                    &mut env_ptr,
                )
            },
            "failed to create ONNX Runtime environment",
        )?;

        let mut session_options_ptr = ptr::null_mut();
        let result = (|| -> anyhow::Result<Self> {
            ort_status_to_result(
                unsafe { ort_api().CreateSessionOptions.unwrap()(&mut session_options_ptr) },
                "failed to create ONNX Runtime session options",
            )?;
            ort_status_to_result(
                unsafe {
                    ort_api().SetSessionGraphOptimizationLevel.unwrap()(
                        session_options_ptr,
                        ort_sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
                    )
                },
                "failed to configure ONNX Runtime graph optimization level",
            )?;
            ort_status_to_result(
                unsafe {
                    ort_api().SetIntraOpNumThreads.unwrap()(
                        session_options_ptr,
                        i32::try_from(intra_threads).unwrap_or(1),
                    )
                },
                "failed to configure ONNX Runtime intra-op threads",
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
                "failed to create ONNX Runtime session",
            )?;

            let mut allocator_ptr = ptr::null_mut();
            ort_status_to_result(
                unsafe { ort_api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) },
                "failed to get ONNX Runtime default allocator",
            )?;

            let mut input_count = 0;
            ort_status_to_result(
                unsafe { ort_api().SessionGetInputCount.unwrap()(session_ptr, &mut input_count) },
                "failed to read ONNX input count",
            )?;
            let mut output_count = 0;
            ort_status_to_result(
                unsafe { ort_api().SessionGetOutputCount.unwrap()(session_ptr, &mut output_count) },
                "failed to read ONNX output count",
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
                allocator_ptr,
                inputs,
                outputs,
                run_lock: std::sync::Mutex::new(()),
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

    fn inputs(&self) -> &[OrtIoInfo] {
        &self.inputs
    }

    fn outputs(&self) -> &[OrtIoInfo] {
        &self.outputs
    }

    async fn run(
        &self,
        inputs: &OnnxInputs,
        output_names: &[String],
    ) -> anyhow::Result<OnnxOutputs> {
        if output_names.is_empty() {
            anyhow::bail!("ONNX runtime session requires at least one output name")
        }

        let _run_lock = self
            .run_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX Runtime run lock is poisoned"))?;
        let provided_inputs = inputs
            .items
            .iter()
            .map(|item| (item.name.as_str(), item))
            .collect::<HashMap<_, _>>();
        let requested_outputs = output_names
            .iter()
            .map(|output_name| {
                self.outputs
                    .iter()
                    .find(|output| output.name == *output_name)
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!("ONNX output `{output_name}` was not found in model")
                    })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut input_values = Vec::with_capacity(self.inputs.len());
        let mut input_name_cstrings = Vec::with_capacity(self.inputs.len());
        let mut input_name_ptrs = Vec::with_capacity(self.inputs.len());
        let mut input_value_ptrs = Vec::with_capacity(self.inputs.len());

        for input in &self.inputs {
            let provided = provided_inputs.get(input.name.as_str()).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "ONNX input `{}` is required by model but was not provided by the current runtime adapter",
                    input.name
                )
            })?;

            let input_value = build_ort_input_value(self.allocator_ptr, input, &provided.payload)?;
            let input_name = CString::new(input.name.as_str())?;
            input_name_ptrs.push(input_name.as_ptr());
            input_value_ptrs.push(input_value.as_const_ptr());
            input_name_cstrings.push(input_name);
            input_values.push(input_value);
        }

        let output_name_cstrings = output_names
            .iter()
            .map(|output_name| CString::new(output_name.as_str()))
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
            "failed to execute ONNX Runtime session",
        )?;

        let output_values = output_value_ptrs
            .into_iter()
            .map(OrtValueGuard::new)
            .collect::<Vec<_>>();
        let mut items = std::collections::HashMap::with_capacity(requested_outputs.len());
        for ((output_name, output_info), output_value) in output_names
            .iter()
            .zip(requested_outputs.iter())
            .zip(output_values.iter())
        {
            let decoded = decode_ort_output_value(output_info, output_value.ptr)?;
            items.insert(output_name.clone(), decoded);
        }

        Ok(OnnxOutputs { items })
    }
}

#[cfg(feature = "onnx-runtime")]
struct OnnxRuntimeBackend {
    _contract: ResolvedOnnxModelContract,
    session: OnnxSession,
    provider_name: &'static str,
    input_adapter: Box<dyn OnnxInputAdapter>,
    output_adapter: Box<dyn OnnxOutputAdapter>,
}

#[cfg(feature = "onnx-runtime")]
impl OnnxRuntimeBackend {
    fn new(
        model_path: PathBuf,
        intra_threads: usize,
        contract: ResolvedOnnxModelContract,
    ) -> anyhow::Result<Self> {
        let session = OnnxSession::new(model_path, intra_threads)?;
        let pipeline = build_onnx_runtime_pipeline(&contract, session.inputs(), session.outputs())?;

        Ok(Self {
            _contract: contract,
            session,
            provider_name: pipeline.provider_name,
            input_adapter: pipeline.input_adapter,
            output_adapter: pipeline.output_adapter,
        })
    }

    fn provider_name(&self) -> &'static str {
        self.provider_name
    }

    async fn extract(
        &self,
        doc: &DocumentIr,
        schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult> {
        let prepared = self.input_adapter.build_inputs(doc, schema)?;
        let outputs = self
            .session
            .run(&prepared.inputs, self.output_adapter.output_names())
            .await?;
        self.output_adapter
            .decode(outputs, &prepared.context, doc, schema)
    }
}

#[cfg(not(feature = "onnx-runtime"))]
#[derive(Debug)]
struct OnnxRuntimeBackend {
    _model_path: PathBuf,
    _contract: ResolvedOnnxModelContract,
}

#[cfg(not(feature = "onnx-runtime"))]
impl OnnxRuntimeBackend {
    fn new(
        model_path: PathBuf,
        _intra_threads: usize,
        contract: ResolvedOnnxModelContract,
    ) -> anyhow::Result<Self> {
        if !model_path.is_file() {
            anyhow::bail!("ONNX model file was not found: {}", model_path.display());
        }
        let _ = contract;
        anyhow::bail!(
            "this binary was built without the `onnx-runtime` Cargo feature; rebuild with `--features onnx-runtime`"
        )
    }

    fn provider_name(&self) -> &'static str {
        "onnx-runtime-disabled"
    }

    async fn extract(
        &self,
        _doc: &DocumentIr,
        _schema: &SchemaSpec,
    ) -> anyhow::Result<ExtractionResult> {
        anyhow::bail!(
            "this binary was built without the `onnx-runtime` Cargo feature; rebuild with `--features onnx-runtime`"
        )
    }
}

fn heuristic_extract(doc: &DocumentIr, schema: &SchemaSpec) -> ExtractionResult {
    let fields = schema
        .fields
        .iter()
        .map(|field| extract_field(doc, field))
        .collect::<Vec<_>>();

    ExtractionResult {
        task_id: Uuid::new_v4().to_string(),
        status: TaskStatus::Extracting,
        fields,
        raw_text: Some(doc.plain_text.clone()),
        timings: TimingBreakdown::default(),
    }
}

#[derive(Clone, Debug)]
struct CandidateMatch {
    raw_value: String,
    evidences: Vec<Evidence>,
}

fn extract_field(doc: &DocumentIr, field: &FieldSpec) -> FieldValue {
    if !field.children.is_empty() {
        return extract_composite_field(doc, field);
    }

    let matches = find_candidate_matches(doc, field);
    let typed_values = matches
        .iter()
        .filter_map(|candidate| coerce_value(&candidate.raw_value, &field.field_type))
        .collect::<Vec<_>>();
    let evidences = dedupe_evidences(
        matches
            .into_iter()
            .flat_map(|candidate| candidate.evidences)
            .collect(),
    );

    let value = if field.multiple {
        let mut items = Vec::new();
        for item in typed_values {
            if !items.iter().any(|existing| existing == &item) {
                items.push(item);
            }
        }

        Value::Array(items)
    } else {
        typed_values
            .into_iter()
            .next()
            .unwrap_or_else(|| default_value(&field.field_type))
    };

    let confidence = Some(compute_confidence(&value, field.required, evidences.len()));

    FieldValue {
        key: field.key.clone(),
        value,
        confidence,
        evidences,
    }
}

fn extract_composite_field(doc: &DocumentIr, field: &FieldSpec) -> FieldValue {
    let children = field
        .children
        .iter()
        .map(|child| extract_field(doc, child))
        .collect::<Vec<_>>();
    let evidences = dedupe_evidences(
        children
            .iter()
            .flat_map(|child| child.evidences.clone())
            .collect(),
    );

    let value = match field.field_type {
        FieldType::Array => build_object_array(&children),
        _ => Value::Object(
            children
                .iter()
                .map(|child| (child.key.clone(), child.value.clone()))
                .collect::<Map<String, Value>>(),
        ),
    };

    let meaningful_children = children
        .iter()
        .filter(|child| has_meaningful_value(&child.value))
        .count();
    let confidence = if meaningful_children == 0 {
        Some(compute_confidence(&value, field.required, evidences.len()))
    } else {
        Some(
            (children
                .iter()
                .filter_map(|child| child.confidence)
                .sum::<f32>()
                / meaningful_children as f32)
                .clamp(0.0, 0.99),
        )
    };

    FieldValue {
        key: field.key.clone(),
        value,
        confidence,
        evidences,
    }
}

fn find_candidate_matches(doc: &DocumentIr, field: &FieldSpec) -> Vec<CandidateMatch> {
    let mut matches = Vec::new();
    let mut seen = HashSet::new();

    for label in candidate_labels(field) {
        for raw_value in find_values_in_text(&doc.plain_text, &label) {
            let evidences = find_matching_block_evidence(doc, &label, &raw_value);
            let key = format!("text::{label}::{raw_value}");
            if seen.insert(key) {
                matches.push(CandidateMatch {
                    raw_value,
                    evidences,
                });
            }
        }

        for page in &doc.pages {
            for block in &page.blocks {
                for raw_value in find_values_in_text(&block.text, &label) {
                    let evidence = evidence_from_block(block);
                    let key = format!(
                        "block::{label}::{}::{}",
                        block.block_id,
                        raw_value.trim().to_lowercase()
                    );
                    if seen.insert(key) {
                        matches.push(CandidateMatch {
                            raw_value,
                            evidences: vec![evidence],
                        });
                    }
                }
            }
        }
    }

    matches
}

fn candidate_labels(field: &FieldSpec) -> Vec<String> {
    let mut labels = Vec::new();
    let mut seen = HashSet::new();

    for candidate in
        std::iter::once(field.key.as_str()).chain(field.hints.iter().map(String::as_str))
    {
        let candidate = candidate.trim();
        if !candidate.is_empty() && seen.insert(candidate.to_string()) {
            labels.push(candidate.to_string());
        }
    }

    labels
}

fn find_values_in_text(text: &str, label: &str) -> Vec<String> {
    let escaped = regex::escape(label);
    let pattern = format!(r"(?m)^\s*(?:[-*•]\s*)?{escaped}\s*[:：]\s*(?P<value>[^\n\r]+)$");
    let regex = Regex::new(&pattern).expect("valid extraction regex");

    regex
        .captures_iter(text)
        .filter_map(|captures| {
            captures
                .name("value")
                .map(|m| m.as_str().trim().to_string())
        })
        .filter(|value| !value.is_empty())
        .collect()
}

fn find_matching_block_evidence(doc: &DocumentIr, label: &str, raw_value: &str) -> Vec<Evidence> {
    let mut evidences = Vec::new();

    for page in &doc.pages {
        for block in &page.blocks {
            if block.text.contains(label) && block.text.contains(raw_value) {
                evidences.push(evidence_from_block(block));
            }
        }
    }

    dedupe_evidences(evidences)
}

fn evidence_from_block(block: &TextBlock) -> Evidence {
    Evidence {
        page_no: Some(block.page_no),
        text: block.text.clone(),
        bbox: block.bbox.clone(),
        source_block_ids: vec![block.block_id.clone()],
    }
}

fn build_object_array(children: &[FieldValue]) -> Value {
    let row_count = children.iter().map(array_row_count).max().unwrap_or(0);

    if row_count == 0 {
        return Value::Array(vec![]);
    }

    let rows = (0..row_count)
        .map(|row_index| {
            let mut object = Map::new();
            for child in children {
                object.insert(child.key.clone(), nth_value(child, row_index));
            }
            Value::Object(object)
        })
        .collect::<Vec<_>>();

    Value::Array(rows)
}

fn array_row_count(field: &FieldValue) -> usize {
    match &field.value {
        Value::Array(items) => items.len(),
        value if has_meaningful_value(value) => 1,
        _ => 0,
    }
}

fn nth_value(field: &FieldValue, row_index: usize) -> Value {
    match &field.value {
        Value::Array(items) => items.get(row_index).cloned().unwrap_or(Value::Null),
        value if row_index == 0 => value.clone(),
        _ => Value::Null,
    }
}

fn coerce_value(raw: &str, field_type: &FieldType) -> Option<Value> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    match field_type {
        FieldType::String | FieldType::Object | FieldType::Array => {
            Some(Value::String(raw.to_string()))
        }
        FieldType::Number => parse_number_value(raw),
        FieldType::Boolean => parse_bool_value(raw).map(Value::Bool),
    }
}

fn parse_number_value(raw: &str) -> Option<Value> {
    let normalized = raw.replace(',', "");
    let regex = Regex::new(r"[-+]?\d+(?:\.\d+)?").expect("valid number regex");
    let matched = regex.find(&normalized)?.as_str();

    if let Ok(integer) = matched.parse::<i64>() {
        return Some(Value::Number(Number::from(integer)));
    }

    let float = matched.parse::<f64>().ok()?;
    Number::from_f64(float).map(Value::Number)
}

fn parse_bool_value(raw: &str) -> Option<bool> {
    let lowered = raw.trim().to_ascii_lowercase();
    if matches!(
        lowered.as_str(),
        "true" | "yes" | "y" | "1" | "是" | "有" | "已" | "通过" | "开启"
    ) {
        return Some(true);
    }
    if matches!(
        lowered.as_str(),
        "false" | "no" | "n" | "0" | "否" | "无" | "未" | "失败" | "关闭"
    ) {
        return Some(false);
    }
    None
}

fn compute_confidence(value: &Value, required: bool, evidence_count: usize) -> f32 {
    if !has_meaningful_value(value) {
        return if required { 0.05 } else { 0.2 };
    }

    let mut confidence: f32 = 0.55;
    if evidence_count > 0 {
        confidence += 0.2;
    }
    if required {
        confidence += 0.05;
    }
    if matches!(value, Value::Array(items) if items.len() > 1) {
        confidence += 0.05;
    }

    confidence.clamp(0.0, 0.95)
}

fn dedupe_evidences(evidences: Vec<Evidence>) -> Vec<Evidence> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();

    for evidence in evidences {
        let key = format!(
            "{:?}|{}|{:?}",
            evidence.page_no, evidence.text, evidence.source_block_ids
        );
        if seen.insert(key) {
            deduped.push(evidence);
        }
    }

    deduped
}

fn default_value(field_type: &FieldType) -> Value {
    match field_type {
        FieldType::String => Value::Null,
        FieldType::Number => Value::Null,
        FieldType::Boolean => Value::Null,
        FieldType::Object => json!({}),
        FieldType::Array => json!([]),
    }
}

fn has_meaningful_value(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(items) => !items.is_empty() && items.iter().any(has_meaningful_value),
        Value::Object(map) => !map.is_empty() && map.values().any(has_meaningful_value),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BlockSourceKind, DocumentMetadata, PageIr, SourceType, TextBlock};
    use std::fs;

    #[tokio::test]
    async fn extracts_typed_scalar_fields_with_hints() {
        let extractor = HeuristicExtractor;
        let schema = SchemaSpec {
            name: "demo".to_string(),
            version: "1".to_string(),
            fields: vec![
                FieldSpec {
                    key: "岗位类型".to_string(),
                    field_type: FieldType::String,
                    required: true,
                    multiple: false,
                    children: vec![],
                    hints: vec!["岗位别名".to_string()],
                },
                FieldSpec {
                    key: "人数".to_string(),
                    field_type: FieldType::Number,
                    required: false,
                    multiple: false,
                    children: vec![],
                    hints: vec![],
                },
                FieldSpec {
                    key: "是否远程".to_string(),
                    field_type: FieldType::Boolean,
                    required: false,
                    multiple: false,
                    children: vec![],
                    hints: vec![],
                },
            ],
        };

        let result = extractor
            .extract(
                &sample_document(
                    "岗位别名：产品经理\n人数：42 人\n是否远程：是",
                    vec!["岗位别名：产品经理", "人数：42 人", "是否远程：是"],
                ),
                &schema,
            )
            .await
            .expect("extract");

        assert_eq!(
            result.fields[0].value,
            Value::String("产品经理".to_string())
        );
        assert_eq!(result.fields[1].value, Value::Number(Number::from(42)));
        assert_eq!(result.fields[2].value, Value::Bool(true));
        assert!(!result.fields[0].evidences.is_empty());
    }

    #[tokio::test]
    async fn aggregates_object_fields_from_children() {
        let extractor = HeuristicExtractor;
        let schema = SchemaSpec {
            name: "profile".to_string(),
            version: "1".to_string(),
            fields: vec![FieldSpec {
                key: "岗位画像".to_string(),
                field_type: FieldType::Object,
                required: false,
                multiple: false,
                children: vec![
                    FieldSpec {
                        key: "岗位类型".to_string(),
                        field_type: FieldType::String,
                        required: false,
                        multiple: false,
                        children: vec![],
                        hints: vec![],
                    },
                    FieldSpec {
                        key: "人设要点".to_string(),
                        field_type: FieldType::String,
                        required: false,
                        multiple: true,
                        children: vec![],
                        hints: vec![],
                    },
                ],
                hints: vec![],
            }],
        };

        let result = extractor
            .extract(
                &sample_document(
                    "岗位类型：策略运营\n人设要点：结构化表达\n人设要点：逻辑清晰",
                    vec![
                        "岗位类型：策略运营",
                        "人设要点：结构化表达",
                        "人设要点：逻辑清晰",
                    ],
                ),
                &schema,
            )
            .await
            .expect("extract");

        assert_eq!(
            result.fields[0].value,
            json!({
                "岗位类型": "策略运营",
                "人设要点": ["结构化表达", "逻辑清晰"]
            })
        );
        assert!(result.fields[0].confidence.unwrap_or_default() > 0.5);
        assert_eq!(result.fields[0].evidences.len(), 3);
    }

    #[tokio::test]
    async fn builds_array_objects_from_child_arrays() {
        let extractor = HeuristicExtractor;
        let schema = SchemaSpec {
            name: "assessment".to_string(),
            version: "1".to_string(),
            fields: vec![FieldSpec {
                key: "测评维度".to_string(),
                field_type: FieldType::Array,
                required: false,
                multiple: false,
                children: vec![
                    FieldSpec {
                        key: "维度".to_string(),
                        field_type: FieldType::String,
                        required: false,
                        multiple: true,
                        children: vec![],
                        hints: vec![],
                    },
                    FieldSpec {
                        key: "答题策略".to_string(),
                        field_type: FieldType::String,
                        required: false,
                        multiple: true,
                        children: vec![],
                        hints: vec![],
                    },
                ],
                hints: vec![],
            }],
        };

        let result = extractor
            .extract(
                &sample_document(
                    "维度：抗压能力\n维度：团队协作\n答题策略：给出真实案例\n答题策略：强调协作闭环",
                    vec![
                        "维度：抗压能力",
                        "维度：团队协作",
                        "答题策略：给出真实案例",
                        "答题策略：强调协作闭环",
                    ],
                ),
                &schema,
            )
            .await
            .expect("extract");

        assert_eq!(
            result.fields[0].value,
            json!([
                {"维度": "抗压能力", "答题策略": "给出真实案例"},
                {"维度": "团队协作", "答题策略": "强调协作闭环"}
            ])
        );
    }

    #[test]
    fn rejects_onnx_without_model_path() {
        let config = Config {
            listen_addr: "127.0.0.1:0".parse().expect("socket addr"),
            service_name: "test".to_string(),
            log_filter: "info".to_string(),
            extractor_provider: "onnx".to_string(),
            onnx_model_path: None,
            onnx_model_spec_path: None,
            onnx_threads: 1,
            onnx_input_text_name: "text".to_string(),
            onnx_input_schema_name: "schema".to_string(),
            onnx_output_json_name: "json_output".to_string(),
            ocr_provider: "placeholder".to_string(),
            ocr_worker_url: None,
            ocr_timeout_ms: 5_000,
            ocr_worker_token: None,
        };

        let error = OnnxRuntimeExtractor::from_config(&config).expect_err("missing model path");
        assert!(error.to_string().contains("MUSE_ONNX_MODEL_PATH"));
    }

    #[test]
    fn resolves_onnx_contract_from_sidecar_spec() {
        let temp_root = std::env::temp_dir().join(format!("muse-onnx-spec-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_root).expect("temp dir");
        let model_path = temp_root.join("uie.onnx");
        let spec_path = temp_root.join("uie.json");
        fs::write(&model_path, b"fake-model").expect("write model");
        fs::write(
            &spec_path,
            r#"{
                "protocol_version": 1,
                "input_text_name": "document_text",
                "input_schema_name": "schema_json",
                "output_json_name": "structured_json",
                "decode_strategy": "json_string_scalar"
            }"#,
        )
        .expect("write spec");

        let contract = resolve_onnx_model_contract(&model_path, None, "text", "schema", "output")
            .expect("contract");

        assert_eq!(contract.input_text_name, "document_text");
        assert_eq!(contract.input_schema_name, "schema_json");
        assert_eq!(contract.output_json_name, "structured_json");
        assert_eq!(contract._spec_path.as_deref(), Some(spec_path.as_path()));

        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn resolves_tokenized_onnx_contract_from_sidecar_spec() {
        let temp_root =
            std::env::temp_dir().join(format!("muse-onnx-tokenized-spec-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_root).expect("temp dir");
        let model_path = temp_root.join("uie.onnx");
        let spec_path = temp_root.join("uie.json");
        let tokenizer_path = temp_root.join("tokenizer.json");
        fs::write(&model_path, b"fake-model").expect("write model");
        fs::write(&tokenizer_path, b"{}").expect("write tokenizer");
        fs::write(
            &spec_path,
            r#"{
                "protocol_version": 1,
                "runtime_contract": "tokenized",
                "tokenizer_path": "./tokenizer.json",
                "max_length": 512,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "token_type_ids": "token_type_ids"
                },
                "outputs": {
                    "start_probs": "start_probs",
                    "end_probs": "end_probs"
                },
                "decode_strategy": "uie_span"
            }"#,
        )
        .expect("write spec");

        let contract = resolve_onnx_model_contract(&model_path, None, "text", "schema", "output")
            .expect("contract");
        let tokenized = contract.tokenized.as_ref().expect("tokenized contract");

        assert_eq!(
            contract.runtime_contract,
            OnnxRuntimeContractKind::Tokenized
        );
        assert_eq!(contract._decode_strategy, OnnxDecodeStrategy::UieSpan);
        assert_eq!(tokenized.tokenizer_path, tokenizer_path);
        assert_eq!(tokenized.max_length, 512);
        assert_eq!(tokenized.inputs.input_ids, "input_ids");
        assert_eq!(tokenized.inputs.attention_mask, "attention_mask");
        assert_eq!(
            tokenized.inputs.token_type_ids.as_deref(),
            Some("token_type_ids")
        );
        assert_eq!(tokenized.outputs.start_probs, "start_probs");
        assert_eq!(tokenized.outputs.end_probs, "end_probs");

        let _ = fs::remove_dir_all(temp_root);
    }

    #[cfg(feature = "onnx-runtime")]
    #[test]
    fn tokenized_runtime_contract_reports_clear_not_implemented_error() {
        let temp_root =
            std::env::temp_dir().join(format!("muse-onnx-tokenized-runtime-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_root).expect("temp dir");
        let model_path = temp_root.join("uie.onnx");
        let spec_path = temp_root.join("uie.json");
        let tokenizer_path = temp_root.join("tokenizer.json");
        fs::write(&model_path, b"fake-model").expect("write model");
        fs::write(&tokenizer_path, b"{}").expect("write tokenizer");
        fs::write(
            &spec_path,
            r#"{
                "protocol_version": 1,
                "runtime_contract": "tokenized",
                "tokenizer_path": "./tokenizer.json",
                "max_length": 256,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask"
                },
                "outputs": {
                    "start_probs": "start_probs",
                    "end_probs": "end_probs"
                },
                "decode_strategy": "uie_span"
            }"#,
        )
        .expect("write spec");

        let contract = resolve_onnx_model_contract(&model_path, None, "text", "schema", "output")
            .expect("contract");
        let error =
            ensure_supported_runtime_contract(&contract).expect_err("tokenized should be pending");

        assert!(error.to_string().contains("runtime_contract `tokenized`"));
        assert!(error.to_string().contains("tokenizer_path"));

        let _ = fs::remove_dir_all(temp_root);
    }

    #[cfg(feature = "onnx-runtime")]
    #[test]
    fn normalizes_onnx_object_payload_into_schema_fields() {
        let schema = SchemaSpec {
            name: "demo".to_string(),
            version: "1".to_string(),
            fields: vec![
                FieldSpec {
                    key: "公司名称".to_string(),
                    field_type: FieldType::String,
                    required: true,
                    multiple: false,
                    children: vec![],
                    hints: vec![],
                },
                FieldSpec {
                    key: "是否远程".to_string(),
                    field_type: FieldType::Boolean,
                    required: false,
                    multiple: false,
                    children: vec![],
                    hints: vec![],
                },
            ],
        };
        let doc = DocumentIr {
            doc_id: Uuid::new_v4().to_string(),
            source_type: SourceType::Text,
            pages: vec![PageIr {
                page_no: 1,
                width: None,
                height: None,
                blocks: vec![
                    TextBlock {
                        block_id: "company-block".to_string(),
                        page_no: 1,
                        text: "公司名称：Muse".to_string(),
                        bbox: None,
                        confidence: Some(0.98),
                        source_kind: BlockSourceKind::NativeText,
                    },
                    TextBlock {
                        block_id: "remote-block".to_string(),
                        page_no: 1,
                        text: "支持远程办公".to_string(),
                        bbox: None,
                        confidence: Some(0.95),
                        source_kind: BlockSourceKind::NativeText,
                    },
                ],
            }],
            plain_text: "公司名称：Muse\n支持远程办公".to_string(),
            metadata: DocumentMetadata::default(),
        };

        let result = normalize_onnx_json_result(
            &json!({
                "公司名称": "Muse",
                "公司名称_confidence": 0.91,
                "公司名称_evidences": [{
                    "source_block_ids": ["company-block"]
                }]
            }),
            &doc,
            &schema,
        )
        .expect("normalize");

        assert_eq!(result.fields[0].value, Value::String("Muse".to_string()));
        assert_eq!(result.fields[0].confidence, Some(0.91));
        assert_eq!(result.fields[0].evidences.len(), 1);
        assert_eq!(result.fields[0].evidences[0].page_no, Some(1));
        assert_eq!(
            result.fields[0].evidences[0].source_block_ids,
            vec!["company-block".to_string()]
        );
        assert_eq!(result.fields[1].value, Value::Null);
        assert!(result.fields[1].evidences.is_empty());
    }

    #[cfg(feature = "onnx-runtime")]
    #[test]
    fn normalizes_onnx_fields_array_and_binds_text_evidence() {
        let schema = SchemaSpec {
            name: "demo".to_string(),
            version: "1".to_string(),
            fields: vec![FieldSpec {
                key: "结论".to_string(),
                field_type: FieldType::String,
                required: false,
                multiple: false,
                children: vec![],
                hints: vec![],
            }],
        };
        let doc = sample_document("系统支持 HTTP OCR worker", vec!["系统支持 HTTP OCR worker"]);

        let result = normalize_onnx_json_result(
            &json!({
                "fields": [
                    {
                        "key": "结论",
                        "value": "HTTP OCR",
                        "confidence": "0.87",
                        "evidences": [{ "text": "HTTP OCR worker" }]
                    }
                ]
            }),
            &doc,
            &schema,
        )
        .expect("normalize");

        assert_eq!(
            result.fields[0].value,
            Value::String("HTTP OCR".to_string())
        );
        assert_eq!(result.fields[0].confidence, Some(0.87));
        assert_eq!(result.fields[0].evidences.len(), 1);
        assert_eq!(result.fields[0].evidences[0].page_no, Some(1));
        assert_eq!(
            result.fields[0].evidences[0].text,
            "系统支持 HTTP OCR worker"
        );
        assert_eq!(result.fields[0].evidences[0].source_block_ids.len(), 1);
    }

    #[test]
    fn explicit_missing_onnx_spec_path_returns_error() {
        let model_path =
            std::env::temp_dir().join(format!("muse-onnx-model-{}.onnx", Uuid::new_v4()));
        fs::write(&model_path, b"fake-model").expect("write model");

        let error = resolve_onnx_model_contract(
            &model_path,
            Some("/definitely/missing/spec.json"),
            "text",
            "schema",
            "output",
        )
        .expect_err("missing spec should fail");

        assert!(error.to_string().contains("spec file was not found"));

        let _ = fs::remove_file(model_path);
    }

    fn sample_document(plain_text: &str, lines: Vec<&str>) -> DocumentIr {
        DocumentIr {
            doc_id: Uuid::new_v4().to_string(),
            source_type: SourceType::Text,
            pages: vec![PageIr {
                page_no: 1,
                width: None,
                height: None,
                blocks: lines
                    .into_iter()
                    .map(|line| TextBlock {
                        block_id: Uuid::new_v4().to_string(),
                        page_no: 1,
                        text: line.to_string(),
                        bbox: None,
                        confidence: Some(0.9),
                        source_kind: BlockSourceKind::NativeText,
                    })
                    .collect(),
            }],
            plain_text: plain_text.to_string(),
            metadata: DocumentMetadata::default(),
        }
    }
}
