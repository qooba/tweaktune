use libsqlite3_sys as ffi;
use once_cell::sync::Lazy;
use serde_json::Value as JsonValue;
use sqlite_vec::sqlite3_vec_init;
use sqlx::Row;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
    SqlitePool,
};
use std::{path::Path, str::FromStr};

// unsafe {
//     libsqlite3_sys::sqlite3_auto_extension(Some(std::mem::transmute(
//         sqlite_vec::sqlite3_vec_init as *const (),
//     )));
// }

static EXTENSION_REGISTERED: Lazy<()> = Lazy::new(|| unsafe {
    let rc = libsqlite3_sys::sqlite3_auto_extension(Some(std::mem::transmute(
        sqlite3_vec_init as *const (),
    )));

    if rc != ffi::SQLITE_OK {
        panic!("Failed to register sqlite3_vec_init extension: {}", rc);
    }
});

pub async fn open_state_db(db_path: &Path) -> Result<SqlitePool, sqlx::Error> {
    Lazy::force(&EXTENSION_REGISTERED);

    if let Some(dir) = db_path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }

    let opts = SqliteConnectOptions::from_str(&format!("sqlite://{}", db_path.display()))?
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .busy_timeout(std::time::Duration::from_secs(5))
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new().connect_with(opts).await?;

    sqlx::query("PRAGMA synchronous=NORMAL;")
        .execute(&pool)
        .await?;

    sqlx::query("PRAGMA temp_store=MEMORY;")
        .execute(&pool)
        .await?;

    sqlx::query("PRAGMA mmap_size=30000000000;")
        .execute(&pool)
        .await?;

    sqlx::migrate!("db/migrations").run(&pool).await?;

    Ok(pool)
}

#[derive(Clone)]
pub struct State {
    pub db: SqlitePool,
}

impl State {
    pub async fn new(path: &str) -> Result<Self, sqlx::Error> {
        let db_path = &std::path::PathBuf::from(format!("{}/{}", &path, "state.db"));
        let db = open_state_db(db_path).await?;
        Ok(Self { db })
    }

    // Runs
    pub async fn add_run(
        &self,
        run_id: &str,
        log_path: &str,
        metadata: Option<JsonValue>,
    ) -> Result<(), sqlx::Error> {
        let meta = metadata.map(|m| m.to_string());
        sqlx::query(
            "INSERT INTO runs(run_id, log_path, metadata) VALUES (?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET log_path=excluded.log_path, metadata=excluded.metadata",
        )
        .bind(run_id)
        .bind(log_path)
        .bind(meta)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    pub async fn delete_run(&self, run_id: &str) -> Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM runs WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    // Items
    pub async fn add_item(
        &self,
        item_id: &str,
        run_id: &str,
        iter_index: i64,
        metadata: Option<JsonValue>,
    ) -> Result<(), sqlx::Error> {
        let meta = metadata.map(|m| m.to_string());
        sqlx::query(
            "INSERT INTO items(item_id, run_id, iter_index, metadata) VALUES (?, ?, ?, ?) ON CONFLICT(item_id) DO UPDATE SET run_id=excluded.run_id, iter_index=excluded.iter_index, metadata=excluded.metadata",
        )
        .bind(item_id)
        .bind(run_id)
        .bind(iter_index)
        .bind(meta)
        .execute(&self.db)
        .await?;

        Ok(())
    }

    pub async fn delete_item(&self, item_id: &str) -> Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM items WHERE item_id = ?")
            .bind(item_id)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    // Hashes
    pub async fn add_hash(&self, item_id: &str, key: &str, hash: &str) -> Result<(), sqlx::Error> {
        sqlx::query("INSERT INTO hashes(item_id, key, hash) VALUES (?, ?, ?)")
            .bind(item_id)
            .bind(key)
            .bind(hash)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    pub async fn hash_exists(&self, key: &str, hash: &str) -> Result<bool, sqlx::Error> {
        let v: Option<i64> =
            sqlx::query_scalar("SELECT 1 FROM hashes WHERE key = ? AND hash = ? LIMIT 1")
                .bind(key)
                .bind(hash)
                .fetch_optional(&self.db)
                .await?;
        Ok(v.is_some())
    }

    // Simhashes
    pub async fn add_simhash(
        &self,
        item_id: &str,
        key: &str,
        simhash: i64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query("INSERT INTO simhashes(item_id, key, simhash) VALUES (?, ?, ?)")
            .bind(item_id)
            .bind(key)
            .bind(simhash)
            .execute(&self.db)
            .await?;
        Ok(())
    }

    // Embeddings
    pub async fn add_embedding(
        &self,
        item_id: &str,
        key: &str,
        embedding: &[f32],
    ) -> Result<(), sqlx::Error> {
        // Serialize f32 slice to little-endian bytes
        let mut buf = Vec::with_capacity(embedding.len() * 4);
        for v in embedding {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        sqlx::query("INSERT INTO embeddings(item_id, key, embedding) VALUES (?, ?, ?)")
            .bind(item_id)
            .bind(key)
            .bind(buf)
            .execute(&self.db)
            .await?;

        Ok(())
    }

    /// KNN search for embeddings (in Rust): fetch candidates for `key`, deserialize
    /// stored blob embeddings (f32 LE), compute cosine similarity against `query` and
    /// return up to `k` nearest neighbors as tuples (item_id, similarity).
    pub async fn knn_embeddings(
        &self,
        key: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(Option<String>, f32)>, sqlx::Error> {
        let rows = sqlx::query("SELECT embedding, item_id FROM embeddings WHERE key = ?")
            .bind(key)
            .fetch_all(&self.db)
            .await?;

        // Precompute query norm
        let q_norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut candidates: Vec<(Option<String>, f32)> = Vec::new();

        for r in rows.into_iter() {
            let blob: Option<Vec<u8>> = r.get("embedding");
            let item_id: Option<String> = r.get("item_id");

            if let Some(b) = blob {
                if b.len() % 4 != 0 {
                    continue; // invalid blob
                }

                // deserialize to f32 vector (little-endian)
                let mut v = Vec::with_capacity(b.len() / 4);
                for chunk in b.chunks_exact(4) {
                    let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    v.push(f32::from_le_bytes(arr));
                }

                if v.len() != query.len() {
                    // skip mismatched dimensions
                    continue;
                }

                // cosine similarity
                let dot = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum::<f32>();
                let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if q_norm == 0.0 || v_norm == 0.0 {
                    0.0
                } else {
                    dot / (q_norm * v_norm)
                };

                candidates.push((item_id, sim));
            }
        }

        // sort by similarity desc and take k
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        Ok(candidates)
    }

    /// KNN using sqlite-vec SQL functions. This performs the distance computation
    /// inside SQLite using `vec_distance_cosine` and `vec_f32` and returns
    /// (item_id, similarity) ordered by similarity descending.
    pub async fn knn_embeddings_sql(
        &self,
        key: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(Option<String>, f32)>, sqlx::Error> {
        // serialize query as f32 LE BLOB (sqlite-vec accepts vec_f32('[1,2,3]') but
        // we will pass the raw blob using vec_f32(?) by creating the same format
        // as vec_f32: the extension provides vec_f32(text) to create blob from text.
        // Simpler: pass textual representation to vec_f32 in SQL.

        // build JSON-like array string e.g. "[1.0, 2.0, 3.0]"
        let mut s = String::with_capacity(query.len() * 6);
        s.push('[');
        for (i, v) in query.iter().enumerate() {
            if i != 0 {
                s.push_str(", ");
            }
            // Use Debug formatting to ensure decimal point
            s.push_str(&format!("{}", v));
        }
        s.push(']');

        // vec_distance_cosine returns a distance: 1 - cosine; similarity = 1 - distance
        // Order by distance ascending, but return similarity.
        let q = sqlx::query(
            "SELECT item_id, (1.0 - vec_distance_cosine(embedding, vec_f32(?))) as similarity FROM embeddings WHERE key = ? ORDER BY vec_distance_cosine(embedding, vec_f32(?)) ASC LIMIT ?",
        )
        .bind(&s)
        .bind(key)
        .bind(&s)
        .bind(k as i64)
        .fetch_all(&self.db)
        .await?;

        let mut out = Vec::new();
        for row in q {
            let item_id: Option<String> = row.get("item_id");
            let sim: f32 = row.get::<f64, _>("similarity") as f32;
            out.push((item_id, sim));
        }

        Ok(out)
    }

    /// KNN search for simhash: preselect candidates by matching any stored band (b0..b3)
    /// and then compute exact Hamming distance in Rust, returning up to `k` nearest neighbors
    /// as tuples (simhash, distance, item_id).
    pub async fn knn_simhash(
        &self,
        key: &str,
        query_simhash: u64,
        k: usize,
    ) -> Result<Vec<(u64, u32, Option<String>)>, sqlx::Error> {
        // extract 16-bit bands matching the schema
        let b0 = (query_simhash & 0xFF) as i64;
        let b1 = ((query_simhash >> 8) & 0xFF) as i64;
        let b2 = ((query_simhash >> 16) & 0xFF) as i64;
        let b3 = ((query_simhash >> 24) & 0xFF) as i64;
        let b4 = ((query_simhash >> 32) & 0xFF) as i64;
        let b5 = ((query_simhash >> 40) & 0xFF) as i64;
        let b6 = ((query_simhash >> 48) & 0xFF) as i64;
        let b7 = ((query_simhash >> 56) & 0xFF) as i64;
        let s0 = (query_simhash & 0xFFFF) as i64;
        let s1 = ((query_simhash >> 16) & 0xFFFF) as i64;
        let s2 = ((query_simhash >> 32) & 0xFFFF) as i64;
        let s3 = ((query_simhash >> 48) & 0xFFFF) as i64;
        let s4 = ((query_simhash >> 8) & 0xFFFF) as i64;
        let s5 = ((query_simhash >> 24) & 0xFFFF) as i64;
        let s6 = ((query_simhash >> 40) & 0xFFFF) as i64;
        let s7 = ((query_simhash >> 56) & 0xFFFF) as i64;

        // preselect candidates where any band matches. limit to a reasonable number
        let limit = (k.saturating_mul(10)).max(100) as i64;

        let rows = sqlx::query("SELECT simhash, item_id FROM simhashes WHERE key = ? AND (b0 = ? OR b1 = ? OR b2 = ? OR b3 = ? OR b4 = ? OR b5 = ? OR b6 = ? OR b7 = ? OR s0 = ? OR s1 = ? OR s2 = ? OR s3 = ? OR s4 = ? OR s5 = ? OR s6 = ? OR s7 = ?) LIMIT ?")
            .bind(key)
            .bind(b0)
            .bind(b1)
            .bind(b2)
            .bind(b3)
            .bind(b4)
            .bind(b5)
            .bind(b6)
            .bind(b7)
            .bind(s0)
            .bind(s1)
            .bind(s2)
            .bind(s3)
            .bind(s4)
            .bind(s5)
            .bind(s6)
            .bind(s7)
            .bind(limit)
            .fetch_all(&self.db)
            .await?;

        let mut candidates: Vec<(u64, u32, Option<String>)> = rows
            .into_iter()
            .map(|r| {
                let s: i64 = r.get("simhash");
                let sim = s as u64;
                let item_id: Option<String> = r.get("item_id");
                let dist = (sim ^ query_simhash).count_ones();
                (sim, dist, item_id)
            })
            .collect();

        // sort by distance ascending and take k
        candidates.sort_by_key(|t| t.1);
        candidates.truncate(k);
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_callhash_and_simhash_flow() -> Result<(), sqlx::Error> {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let state = State::new(path).await?;

        // add run and item
        state.add_run("run1", "/tmp/log", None).await?;
        state.add_item("item1", "run1", 0, None).await?;

        // hash
        assert!(!state.hash_exists("k1", "h1").await?);
        state.add_hash("item1", "k1", "h1").await?;
        assert!(state.hash_exists("k1", "h1").await?);

        // simhash
        let q: u64 = 0x0123_4567_89AB_CDEF;
        state.add_simhash("item1", "k1", q as i64).await?;
        let res = state.knn_simhash("k1", q, 1).await?;
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, q);

        Ok(())
    }

    #[tokio::test]
    async fn test_knn_distance_ordering() -> Result<(), sqlx::Error> {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let state = State::new(path).await?;

        // create several simhashes around a query
        let q: u64 = 0b1111_0000u64 << 56; // some high bits set
                                           // neighbours with small differences
        let a = q;
        let b = q ^ 0b1;
        let c = q ^ 0b11;
        let d = q ^ 0xFFFF_FFFF;

        state.add_run("run1", "/tmp/log", None).await?;
        state.add_item("item1", "run1", 0, None).await?;
        state.add_simhash("item1", "k2", a as i64).await?;
        state.add_simhash("item1", "k2", b as i64).await?;
        state.add_simhash("item1", "k2", c as i64).await?;
        state.add_simhash("item1", "k2", d as i64).await?;

        let res = state.knn_simhash("k2", q, 3).await?;
        assert_eq!(res.len(), 3);
        // distances should be non-decreasing
        assert!(res[0].1 <= res[1].1 && res[1].1 <= res[2].1);

        Ok(())
    }

    #[tokio::test]
    async fn test_embeddings_flow() -> Result<(), sqlx::Error> {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let state = State::new(path).await?;

        state.add_run("run_emb", "/tmp/log", None).await?;
        state.add_item("item_emb_1", "run_emb", 0, None).await?;
        state.add_item("item_emb_2", "run_emb", 1, None).await?;

        // two simple 3-d vectors: one identical to query, one orthogonal
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let q = vec![1.0f32, 0.0, 0.0];

        state.add_embedding("item_emb_1", "ek", &a).await?;
        state.add_embedding("item_emb_2", "ek", &b).await?;

        let res = state.knn_embeddings("ek", &q, 2).await?;
        assert_eq!(res.len(), 2);

        // first result should be item_emb_1 with similarity 1.0
        assert_eq!(res[0].0.as_deref(), Some("item_emb_1"));
        assert!((res[0].1 - 1.0).abs() < 1e-6);

        // second result should have lower similarity
        assert_eq!(res[1].0.as_deref(), Some("item_emb_2"));
        assert!(res[1].1 < res[0].1);

        Ok(())
    }

    #[tokio::test]
    async fn test_knn_embeddings_sql() -> Result<(), sqlx::Error> {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let state = State::new(path).await?;

        state.add_run("run_emb_sql", "/tmp/log", None).await?;
        state.add_item("item_sql_1", "run_emb_sql", 0, None).await?;
        state.add_item("item_sql_2", "run_emb_sql", 1, None).await?;

        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let q = vec![1.0f32, 0.0, 0.0];

        state.add_embedding("item_sql_1", "sek", &a).await?;
        state.add_embedding("item_sql_2", "sek", &b).await?;

        let res = state.knn_embeddings_sql("sek", &q, 2).await?;
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0.as_deref(), Some("item_sql_1"));
        // similarity for identical vectors should be near 1.0
        assert!((res[0].1 - 1.0).abs() < 1e-5);

        Ok(())
    }
}
