use crate::domain::{Evidence, ExtractionResult, FieldValue, TaskStatus};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashSet;

#[async_trait]
pub trait Postprocessor: Send + Sync {
    fn name(&self) -> &'static str;
    async fn finalize(&self, result: ExtractionResult) -> anyhow::Result<ExtractionResult>;
}

#[derive(Default)]
pub struct DefaultPostprocessor;

#[async_trait]
impl Postprocessor for DefaultPostprocessor {
    fn name(&self) -> &'static str {
        "default-postprocessor"
    }

    async fn finalize(&self, mut result: ExtractionResult) -> anyhow::Result<ExtractionResult> {
        for field in &mut result.fields {
            normalize_field(field);
        }
        result.status = TaskStatus::Succeeded;
        Ok(result)
    }
}

fn normalize_field(field: &mut FieldValue) {
    normalize_value(&mut field.value);
    field.evidences = dedupe_evidences(std::mem::take(&mut field.evidences));

    if !has_meaningful_value(&field.value) {
        field.confidence = Some(field.confidence.unwrap_or(0.1).min(0.1));
    } else if field.confidence.is_none() {
        field.confidence = Some(0.6);
    }
}

fn normalize_value(value: &mut Value) {
    match value {
        Value::String(text) => {
            *text = text.trim().to_string();
            if text.is_empty() {
                *value = Value::Null;
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                normalize_value(item);
            }

            let mut deduped = Vec::new();
            for item in std::mem::take(items) {
                if !has_meaningful_value(&item) {
                    continue;
                }
                if !deduped.iter().any(|existing| existing == &item) {
                    deduped.push(item);
                }
            }
            *items = deduped;
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                normalize_value(item);
            }
        }
        _ => {}
    }
}

fn dedupe_evidences(evidences: Vec<Evidence>) -> Vec<Evidence> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();

    for mut evidence in evidences {
        evidence.source_block_ids.sort();
        evidence.source_block_ids.dedup();
        evidence.text = evidence.text.trim().to_string();

        let key = format!(
            "{:?}|{}|{:?}|{:?}",
            evidence.page_no, evidence.text, evidence.bbox, evidence.source_block_ids
        );
        if seen.insert(key) {
            deduped.push(evidence);
        }
    }

    deduped.sort_by(|left, right| {
        left.page_no
            .cmp(&right.page_no)
            .then_with(|| left.text.cmp(&right.text))
    });
    deduped
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
    use crate::domain::{BBox, TimingBreakdown};
    use serde_json::json;

    #[tokio::test]
    async fn finalizer_normalizes_arrays_and_evidences() {
        let postprocessor = DefaultPostprocessor;
        let result = ExtractionResult {
            task_id: "task-1".to_string(),
            status: TaskStatus::Postprocessing,
            fields: vec![FieldValue {
                key: "人设要点".to_string(),
                value: json!([" 结构化表达 ", "结构化表达", "", "逻辑清晰"]),
                confidence: Some(0.8),
                evidences: vec![
                    Evidence {
                        page_no: Some(2),
                        text: " 人设要点：逻辑清晰 ".to_string(),
                        bbox: Some(BBox {
                            x1: 1.0,
                            y1: 2.0,
                            x2: 3.0,
                            y2: 4.0,
                        }),
                        source_block_ids: vec![
                            "b2".to_string(),
                            "b1".to_string(),
                            "b2".to_string(),
                        ],
                    },
                    Evidence {
                        page_no: Some(2),
                        text: "人设要点：逻辑清晰".to_string(),
                        bbox: Some(BBox {
                            x1: 1.0,
                            y1: 2.0,
                            x2: 3.0,
                            y2: 4.0,
                        }),
                        source_block_ids: vec!["b1".to_string(), "b2".to_string()],
                    },
                ],
            }],
            raw_text: None,
            timings: TimingBreakdown::default(),
        };

        let finalized = postprocessor.finalize(result).await.expect("finalize");

        assert_eq!(finalized.status, TaskStatus::Succeeded);
        assert_eq!(finalized.fields[0].value, json!(["结构化表达", "逻辑清晰"]));
        assert_eq!(finalized.fields[0].evidences.len(), 1);
        assert_eq!(finalized.fields[0].evidences[0].text, "人设要点：逻辑清晰");
        assert_eq!(
            finalized.fields[0].evidences[0].source_block_ids,
            vec!["b1", "b2"]
        );
    }

    #[tokio::test]
    async fn finalizer_clamps_missing_value_confidence() {
        let postprocessor = DefaultPostprocessor;
        let result = ExtractionResult {
            task_id: "task-2".to_string(),
            status: TaskStatus::Postprocessing,
            fields: vec![FieldValue {
                key: "岗位类型".to_string(),
                value: Value::Null,
                confidence: Some(0.7),
                evidences: vec![],
            }],
            raw_text: None,
            timings: TimingBreakdown::default(),
        };

        let finalized = postprocessor.finalize(result).await.expect("finalize");

        assert_eq!(finalized.fields[0].confidence, Some(0.1));
    }
}
