use actix::prelude::*;
use std::collections::VecDeque;
use std::ops::Add;
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, BufReader};

const WORKER_COUNT: usize = 5;
const CHUNK_SIZE: usize = 8 * 1024;

struct ProcessChunk(Vec<u8>);
impl Message for ProcessChunk {
    type Result = ();
}

struct WorkerAvailable(Addr<Worker>);

impl Message for WorkerAvailable {
    type Result = ();
}

struct Worker {
    id: usize,
    pipeline: Pipeline,
    manager: Addr<FileReader>,
}

impl Actor for Worker {
    type Context = Context<Self>;
}

impl Handler<ProcessChunk> for Worker {
    type Result = ();

    fn handle(&mut self, msg: ProcessChunk, ctx: &mut Self::Context) -> Self::Result {
        let manager = self.manager.clone();
        let worker_addr = ctx.address().clone();
        let id = self.id.clone();
        let pipeline = self.pipeline.clone();

        tokio::spawn(async move {
            println!("Worker {} processing chunk of size {}", id, msg.0.len());
            //self.pipeline.process_chunk(msg.0);
            manager.send(WorkerAvailable(worker_addr)).await.unwrap();
        });
    }
}

struct FileReader {
    file: File,
    workers: VecDeque<Addr<Worker>>,
}

impl Actor for FileReader {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.read_next_chunk(ctx);
    }
}

impl FileReader {
    async fn read_next_chunk(&mut self, ctx: &mut Context<Self>) {
        let mut chunk = vec![0; CHUNK_SIZE];
        let worker = self.workers.pop_front().unwrap();

        let mut file = self.file.try_clone().await.unwrap();

        let n = file.read(&mut chunk).await.unwrap();
        chunk.truncate(n);
        worker.send(ProcessChunk(chunk)).await.unwrap();
    }
}

impl Handler<WorkerAvailable> for FileReader {
    type Result = ();

    fn handle(&mut self, msg: WorkerAvailable, ctx: &mut Self::Context) -> Self::Result {
        self.workers.push_back(msg.0);
        self.read_next_chunk(ctx);
    }
}

#[derive(Debug, Clone)]
struct Pipeline {
    file: String,
}

impl Pipeline {
    fn new(file: String) -> Self {
        Pipeline { file }
    }

    fn process_chunk(&self, chunk: Vec<u8>) {
        println!("Processing chunk of size {}", chunk.len());
    }
}
