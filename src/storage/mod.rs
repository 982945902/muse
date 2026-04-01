use crate::domain::{ExtractionResult, TaskRecord};
use async_trait::async_trait;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct CachedExtraction {
    pub cache_key: String,
    pub result: ExtractionResult,
    pub created_at_ms: u64,
    pub last_accessed_at_ms: u64,
    pub hit_count: u64,
}

#[async_trait]
pub trait ExtractionStore: Send + Sync {
    async fn upsert(&self, record: TaskRecord) -> anyhow::Result<()>;
    async fn get(&self, task_id: &str) -> anyhow::Result<Option<TaskRecord>>;
    async fn get_cached(&self, cache_key: &str) -> anyhow::Result<Option<CachedExtraction>>;
    async fn put_cached(
        &self,
        cache_key: &str,
        result: ExtractionResult,
    ) -> anyhow::Result<CachedExtraction>;
}

#[derive(Clone, Default)]
pub struct InMemoryStorage {
    tasks: Arc<RwLock<HashMap<String, TaskRecord>>>,
    cache: Arc<RwLock<HashMap<String, CachedExtraction>>>,
}

#[async_trait]
impl ExtractionStore for InMemoryStorage {
    async fn upsert(&self, record: TaskRecord) -> anyhow::Result<()> {
        self.tasks
            .write()
            .await
            .insert(record.task_id.clone(), record);
        Ok(())
    }

    async fn get(&self, task_id: &str) -> anyhow::Result<Option<TaskRecord>> {
        Ok(self.tasks.read().await.get(task_id).cloned())
    }

    async fn get_cached(&self, cache_key: &str) -> anyhow::Result<Option<CachedExtraction>> {
        let mut cache = self.cache.write().await;
        let now = now_ms();

        if let Some(entry) = cache.get_mut(cache_key) {
            entry.hit_count += 1;
            entry.last_accessed_at_ms = now;
            return Ok(Some(entry.clone()));
        }

        Ok(None)
    }

    async fn put_cached(
        &self,
        cache_key: &str,
        result: ExtractionResult,
    ) -> anyhow::Result<CachedExtraction> {
        let entry = CachedExtraction {
            cache_key: cache_key.to_string(),
            result,
            created_at_ms: now_ms(),
            last_accessed_at_ms: now_ms(),
            hit_count: 0,
        };

        self.cache
            .write()
            .await
            .insert(cache_key.to_string(), entry.clone());

        Ok(entry)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{FieldValue, TaskStatus, TimingBreakdown};
    use serde_json::Value;

    #[tokio::test]
    async fn cache_entry_tracks_hit_count_and_timestamps() {
        let store = InMemoryStorage::default();
        let stored = store
            .put_cached("cache-1", sample_result("task-a"))
            .await
            .expect("put cached");

        assert_eq!(stored.cache_key, "cache-1");
        assert_eq!(stored.hit_count, 0);
        assert!(stored.created_at_ms > 0);

        let first = store
            .get_cached("cache-1")
            .await
            .expect("get cached")
            .expect("cache entry");
        assert_eq!(first.hit_count, 1);
        assert!(first.last_accessed_at_ms >= first.created_at_ms);

        let second = store
            .get_cached("cache-1")
            .await
            .expect("get cached")
            .expect("cache entry");
        assert_eq!(second.hit_count, 2);
        assert!(second.last_accessed_at_ms >= first.last_accessed_at_ms);
    }

    #[tokio::test]
    async fn get_cached_returns_none_for_unknown_key() {
        let store = InMemoryStorage::default();
        let entry = store.get_cached("missing").await.expect("lookup");
        assert!(entry.is_none());
    }

    fn sample_result(task_id: &str) -> ExtractionResult {
        ExtractionResult {
            task_id: task_id.to_string(),
            status: TaskStatus::Succeeded,
            fields: vec![FieldValue {
                key: "岗位类型".to_string(),
                value: Value::String("策略运营".to_string()),
                confidence: Some(0.9),
                evidences: vec![],
            }],
            raw_text: Some("岗位类型：策略运营".to_string()),
            timings: TimingBreakdown::default(),
        }
    }
}
