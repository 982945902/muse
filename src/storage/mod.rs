use crate::domain::{ExtractionResult, TaskRecord};
use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
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
    fn name(&self) -> &'static str;
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
    fn name(&self) -> &'static str {
        "memory"
    }

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

pub struct SqliteStorage {
    connection: Arc<Mutex<Connection>>,
}

impl SqliteStorage {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let connection = Connection::open(path)?;
        initialize_sqlite_schema(&connection)?;

        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    fn lock_connection(&self) -> anyhow::Result<std::sync::MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|_| anyhow::anyhow!("sqlite storage mutex poisoned"))
    }
}

fn initialize_sqlite_schema(connection: &Connection) -> anyhow::Result<()> {
    connection.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            record_json TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            last_accessed_at_ms INTEGER NOT NULL,
            hit_count INTEGER NOT NULL
        );
        ",
    )?;
    Ok(())
}

#[async_trait]
impl ExtractionStore for SqliteStorage {
    fn name(&self) -> &'static str {
        "sqlite"
    }

    async fn upsert(&self, record: TaskRecord) -> anyhow::Result<()> {
        let now = now_ms() as i64;
        let record_json = serde_json::to_string(&record)?;
        let connection = self.lock_connection()?;
        connection.execute(
            "
            INSERT INTO tasks (task_id, record_json, updated_at_ms)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(task_id) DO UPDATE SET
                record_json = excluded.record_json,
                updated_at_ms = excluded.updated_at_ms
            ",
            params![record.task_id, record_json, now],
        )?;
        Ok(())
    }

    async fn get(&self, task_id: &str) -> anyhow::Result<Option<TaskRecord>> {
        let connection = self.lock_connection()?;
        let record_json = connection
            .query_row(
                "SELECT record_json FROM tasks WHERE task_id = ?1",
                params![task_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;

        record_json
            .map(|payload| serde_json::from_str::<TaskRecord>(&payload).map_err(Into::into))
            .transpose()
    }

    async fn get_cached(&self, cache_key: &str) -> anyhow::Result<Option<CachedExtraction>> {
        let now = now_ms() as i64;
        let mut connection = self.lock_connection()?;
        let transaction = connection.transaction()?;
        let cached = transaction
            .query_row(
                "
                SELECT result_json, created_at_ms, last_accessed_at_ms, hit_count
                FROM cache
                WHERE cache_key = ?1
                ",
                params![cache_key],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;

        let Some((result_json, created_at_ms, _last_accessed_at_ms, hit_count)) = cached else {
            return Ok(None);
        };

        let next_hit_count = hit_count + 1;
        transaction.execute(
            "
            UPDATE cache
            SET hit_count = ?2, last_accessed_at_ms = ?3
            WHERE cache_key = ?1
            ",
            params![cache_key, next_hit_count, now],
        )?;
        transaction.commit()?;

        Ok(Some(CachedExtraction {
            cache_key: cache_key.to_string(),
            result: serde_json::from_str::<ExtractionResult>(&result_json)?,
            created_at_ms: created_at_ms as u64,
            last_accessed_at_ms: now as u64,
            hit_count: next_hit_count as u64,
        }))
    }

    async fn put_cached(
        &self,
        cache_key: &str,
        result: ExtractionResult,
    ) -> anyhow::Result<CachedExtraction> {
        let now = now_ms();
        let entry = CachedExtraction {
            cache_key: cache_key.to_string(),
            result,
            created_at_ms: now,
            last_accessed_at_ms: now,
            hit_count: 0,
        };
        let connection = self.lock_connection()?;
        connection.execute(
            "
            INSERT INTO cache (
                cache_key,
                result_json,
                created_at_ms,
                last_accessed_at_ms,
                hit_count
            )
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(cache_key) DO UPDATE SET
                result_json = excluded.result_json,
                created_at_ms = excluded.created_at_ms,
                last_accessed_at_ms = excluded.last_accessed_at_ms,
                hit_count = excluded.hit_count
            ",
            params![
                entry.cache_key,
                serde_json::to_string(&entry.result)?,
                entry.created_at_ms as i64,
                entry.last_accessed_at_ms as i64,
                entry.hit_count as i64
            ],
        )?;

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
    use uuid::Uuid;

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

    #[tokio::test]
    async fn sqlite_storage_persists_task_and_cache_records() {
        let path = std::env::temp_dir().join(format!("muse-storage-{}.sqlite3", Uuid::new_v4()));
        let store = SqliteStorage::new(&path).expect("sqlite storage");

        let task = TaskRecord {
            task_id: "task-sqlite".to_string(),
            status: TaskStatus::Succeeded,
            result: Some(sample_result("task-sqlite")),
            message: Some("ok".to_string()),
        };
        store.upsert(task.clone()).await.expect("upsert task");

        let cached = store
            .put_cached("cache-sqlite", sample_result("task-sqlite"))
            .await
            .expect("put cache");
        assert_eq!(cached.hit_count, 0);

        let loaded_task = store
            .get("task-sqlite")
            .await
            .expect("get task")
            .expect("task record");
        assert_eq!(loaded_task.task_id, task.task_id);
        assert_eq!(loaded_task.message, task.message);

        let first_cache_hit = store
            .get_cached("cache-sqlite")
            .await
            .expect("get cache")
            .expect("cache entry");
        assert_eq!(first_cache_hit.hit_count, 1);

        let second_cache_hit = store
            .get_cached("cache-sqlite")
            .await
            .expect("get cache")
            .expect("cache entry");
        assert_eq!(second_cache_hit.hit_count, 2);

        let _ = std::fs::remove_file(path);
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
