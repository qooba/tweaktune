use serde_json::Value as JsonValue;
use sqlx::Row;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
    SqlitePool,
};
use std::{path::Path, str::FromStr};

pub async fn open_state_db(db_path: &Path) -> Result<SqlitePool, sqlx::Error> {
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
        item_id: Option<&str>,
        key: &str,
        simhash: i64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query("INSERT OR IGNORE INTO simhashes(item_id, key, simhash) VALUES (?, ?, ?)")
            .bind(item_id)
            .bind(key)
            .bind(simhash)
            .execute(&self.db)
            .await?;
        Ok(())
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
        let b0 = (query_simhash & 0xFFFF) as i64;
        let b1 = ((query_simhash >> 16) & 0xFFFF) as i64;
        let b2 = ((query_simhash >> 32) & 0xFFFF) as i64;
        let b3 = ((query_simhash >> 48) & 0xFFFF) as i64;

        // preselect candidates where any band matches. limit to a reasonable number
        let limit = (k.saturating_mul(10)).max(100) as i64;

        let rows = sqlx::query("SELECT simhash, item_id FROM simhashes WHERE key = ? AND (b0 = ? OR b1 = ? OR b2 = ? OR b3 = ?) LIMIT ?")
            .bind(key)
            .bind(b0)
            .bind(b1)
            .bind(b2)
            .bind(b3)
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

        // callhash
        assert!(!state.callhash_exists("k1", "h1").await?);
        state.add_callhash("item1", "k1", "h1").await?;
        assert!(state.callhash_exists("k1", "h1").await?);

        // simhash
        let q: u64 = 0x0123_4567_89AB_CDEF;
        state.add_simhash(Some("item1"), "k1", q as i64).await?;
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

        state.add_simhash(None, "k2", a as i64).await?;
        state.add_simhash(None, "k2", b as i64).await?;
        state.add_simhash(None, "k2", c as i64).await?;
        state.add_simhash(None, "k2", d as i64).await?;

        let res = state.knn_simhash("k2", q, 3).await?;
        assert_eq!(res.len(), 3);
        // distances should be non-decreasing
        assert!(res[0].1 <= res[1].1 && res[1].1 <= res[2].1);

        Ok(())
    }
}
