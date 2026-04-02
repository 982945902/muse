use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{RwLock, broadcast};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamEvent {
    pub sequence: u64,
    pub event_type: String,
    pub task_id: String,
    pub created_at_ms: u64,
    pub payload: Value,
}

pub struct EventSubscription {
    pub history: Vec<StreamEvent>,
    pub receiver: broadcast::Receiver<StreamEvent>,
}

#[derive(Clone, Default)]
pub struct EventHub {
    streams: Arc<RwLock<HashMap<String, TaskEventStream>>>,
}

struct TaskEventStream {
    next_sequence: u64,
    history: Vec<StreamEvent>,
    sender: broadcast::Sender<StreamEvent>,
}

impl EventHub {
    pub async fn ensure_task(&self, task_id: &str) {
        let mut streams = self.streams.write().await;
        streams
            .entry(task_id.to_string())
            .or_insert_with(TaskEventStream::new);
    }

    pub async fn exists(&self, task_id: &str) -> bool {
        self.streams.read().await.contains_key(task_id)
    }

    pub async fn publish(
        &self,
        task_id: &str,
        event_type: impl Into<String>,
        payload: Value,
    ) -> StreamEvent {
        let mut streams = self.streams.write().await;
        let stream = streams
            .entry(task_id.to_string())
            .or_insert_with(TaskEventStream::new);

        let event = StreamEvent {
            sequence: stream.next_sequence,
            event_type: event_type.into(),
            task_id: task_id.to_string(),
            created_at_ms: now_ms(),
            payload,
        };
        stream.next_sequence += 1;
        stream.history.push(event.clone());
        let _ = stream.sender.send(event.clone());
        event
    }

    pub async fn subscribe(&self, task_id: &str) -> Option<EventSubscription> {
        let streams = self.streams.read().await;
        let stream = streams.get(task_id)?;

        Some(EventSubscription {
            history: stream.history.clone(),
            receiver: stream.sender.subscribe(),
        })
    }

    #[cfg(test)]
    pub async fn task_ids(&self) -> Vec<String> {
        let mut task_ids = self
            .streams
            .read()
            .await
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        task_ids.sort();
        task_ids
    }
}

impl TaskEventStream {
    fn new() -> Self {
        let (sender, _) = broadcast::channel(256);
        Self {
            next_sequence: 1,
            history: Vec::new(),
            sender,
        }
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
    use serde_json::json;

    #[tokio::test]
    async fn subscribe_receives_history_and_live_events() {
        let hub = EventHub::default();
        hub.ensure_task("task-1").await;
        let first = hub
            .publish("task-1", "task.accepted", json!({"mode": "async"}))
            .await;

        let mut subscription = hub.subscribe("task-1").await.expect("subscription");
        assert_eq!(subscription.history.len(), 1);
        assert_eq!(subscription.history[0].sequence, first.sequence);

        let live = hub
            .publish("task-1", "stage.changed", json!({"stage": "parsing"}))
            .await;
        let received = subscription.receiver.recv().await.expect("event");
        assert_eq!(received.event_type, live.event_type);
        assert_eq!(received.sequence, live.sequence);
    }

    #[tokio::test]
    async fn exists_returns_false_for_unknown_task() {
        let hub = EventHub::default();
        assert!(!hub.exists("missing").await);
    }
}
