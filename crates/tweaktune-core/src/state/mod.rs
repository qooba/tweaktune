use std::{path::Path, str::FromStr};

use polars::sql;
use polars_arrow::io::ipc::format;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
    SqlitePool,
};

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

pub async fn begin_run(pool: &SqlitePool, run_name: &str) -> Result<i64, sqlx::Error> {
    todo!()
}
