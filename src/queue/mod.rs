use async_trait::async_trait;
use std::{future::Future, pin::Pin};

pub type QueueJob = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

#[async_trait]
pub trait TaskQueue: Send + Sync {
    fn name(&self) -> &'static str;
    async fn dispatch(&self, job: QueueJob) -> anyhow::Result<()>;
}

#[derive(Default)]
pub struct InMemoryQueue;

#[async_trait]
impl TaskQueue for InMemoryQueue {
    fn name(&self) -> &'static str {
        "in-memory-queue"
    }

    async fn dispatch(&self, job: QueueJob) -> anyhow::Result<()> {
        tokio::spawn(job);
        Ok(())
    }
}
