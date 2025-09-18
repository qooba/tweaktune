PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
	run_id TEXT PRIMARY KEY,
	status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
	created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    log_path TEXT NOT NULL, -- path to log file
	metadata JSON -- optional JSON metadata about the run (parameters, env, etc.)
);

CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
	created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    iter_index INTEGER NOT NULL,
    metadata JSON, -- optional JSON metadata about the item (parameters, env, etc.)
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_items_index ON items(iter_index);

CREATE TABLE IF NOT EXISTS hashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	item_id TEXT,
	key TEXT NOT NULL,
    hash TEXT NOT NULL, -- the actual hash value (e.g., hex string for call_hash, integer string for simhash)
	created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(item_id) REFERENCES items(item_id) ON DELETE CASCADE,
	UNIQUE(key, hash)
);

CREATE INDEX IF NOT EXISTS ix_hashes_item_key ON hashes(item_id, key);
CREATE INDEX IF NOT EXISTS ix_hashes_key_hash ON hashes(key, hash);

CREATE TABLE IF NOT EXISTS simhashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	item_id TEXT,
	key TEXT NOT NULL,
    simhash INTEGER NOT NULL,

    -- 8x8-bit bands
    b0 INTEGER GENERATED ALWAYS AS (simhash & 0xFF) STORED,
    b1 INTEGER GENERATED ALWAYS AS ((simhash >> 8) & 0xFF) STORED,
    b2 INTEGER GENERATED ALWAYS AS ((simhash >> 16) & 0xFF) STORED,
    b3 INTEGER GENERATED ALWAYS AS ((simhash >> 24) & 0xFF) STORED,
    b4 INTEGER GENERATED ALWAYS AS ((simhash >> 32) & 0xFF) STORED,
    b5 INTEGER GENERATED ALWAYS AS ((simhash >> 40) & 0xFF) STORED,
    b6 INTEGER GENERATED ALWAYS AS ((simhash >> 48) & 0xFF) STORED,
    b7 INTEGER GENERATED ALWAYS AS ((simhash >> 56) & 0xFF) STORED,

    -- 8x8-bit bands (offset by 4 bit)
    s0 INTEGER GENERATED ALWAYS AS ((simhash >> 4) & 0xFF) STORED, 
    s1 INTEGER GENERATED ALWAYS AS ((simhash >> 12) & 0xFF) STORED,
    s2 INTEGER GENERATED ALWAYS AS ((simhash >> 20) & 0xFF) STORED,
    s3 INTEGER GENERATED ALWAYS AS ((simhash >> 28) & 0xFF) STORED,
    s4 INTEGER GENERATED ALWAYS AS ((simhash >> 36) & 0xFF) STORED,
    s5 INTEGER GENERATED ALWAYS AS ((simhash >> 44) & 0xFF) STORED,
    s6 INTEGER GENERATED ALWAYS AS ((simhash >> 52) & 0xFF) STORED,
    s7 INTEGER GENERATED ALWAYS AS ((simhash >> 60) & 0xFF) STORED,

	created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(item_id) REFERENCES items(item_id) ON DELETE CASCADE,
	UNIQUE(key, simhash)
);

CREATE INDEX IF NOT EXISTS ix_simhashes_key_simhash ON simhashes(key, simhash);
CREATE INDEX IF NOT EXISTS ix_simhashes_b0 ON simhashes(key,b0);
CREATE INDEX IF NOT EXISTS ix_simhashes_b1 ON simhashes(key,b1);
CREATE INDEX IF NOT EXISTS ix_simhashes_b2 ON simhashes(key,b2);
CREATE INDEX IF NOT EXISTS ix_simhashes_b3 ON simhashes(key,b3);
CREATE INDEX IF NOT EXISTS ix_simhashes_b4 ON simhashes(key,b4);
CREATE INDEX IF NOT EXISTS ix_simhashes_b5 ON simhashes(key,b5);
CREATE INDEX IF NOT EXISTS ix_simhashes_b6 ON simhashes(key,b6);
CREATE INDEX IF NOT EXISTS ix_simhashes_b7 ON simhashes(key,b7);
CREATE INDEX IF NOT EXISTS ix_simhashes_s0 ON simhashes(key,s0);
CREATE INDEX IF NOT EXISTS ix_simhashes_s1 ON simhashes(key,s1);
CREATE INDEX IF NOT EXISTS ix_simhashes_s2 ON simhashes(key,s2);
CREATE INDEX IF NOT EXISTS ix_simhashes_s3 ON simhashes(key,s3);
CREATE INDEX IF NOT EXISTS ix_simhashes_s4 ON simhashes(key,s4);
CREATE INDEX IF NOT EXISTS ix_simhashes_s5 ON simhashes(key,s5);
CREATE INDEX IF NOT EXISTS ix_simhashes_s6 ON simhashes(key,s6);
CREATE INDEX IF NOT EXISTS ix_simhashes_s7 ON simhashes(key,s7);

PRAGMA user_version = 1;

