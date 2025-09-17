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

CREATE TABLE IF NOT EXISTS callhashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	item_id TEXT,
	key TEXT NOT NULL,
    hash TEXT NOT NULL, -- the actual hash value (e.g., hex string for call_hash, integer string for simhash)
	created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(item_id) REFERENCES items(item_id) ON DELETE CASCADE,
	UNIQUE(key, hash)
);

CREATE INDEX IF NOT EXISTS ix_callhashes_item_key ON callhashes(item_id, key);
CREATE INDEX IF NOT EXISTS ix_callhashes_key_hash ON callhashes(key, hash);

CREATE TABLE IF NOT EXISTS simhashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	item_id TEXT,
	key TEXT NOT NULL,
    simhash INTEGER NOT NULL,
    b0 INTEGER GENERATED ALWAYS AS (simhash & 0xFFFF) STORED,
    b1 INTEGER GENERATED ALWAYS AS ((simhash >> 16) & 0xFFFF) STORED,
    b2 INTEGER GENERATED ALWAYS AS ((simhash >> 32) & 0xFFFF) STORED,
    b3 INTEGER GENERATED ALWAYS AS ((simhash >> 48) & 0xFFFF) STORED,
	created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(item_id) REFERENCES items(item_id) ON DELETE CASCADE,
	UNIQUE(key, simhash)
);

CREATE INDEX IF NOT EXISTS ix_simhashes_key_simhash ON simhashes(key, simhash);
CREATE INDEX IF NOT EXISTS ix_simhashes_b0 ON simhashes(b0);
CREATE INDEX IF NOT EXISTS ix_simhashes_b1 ON simhashes(b1);
CREATE INDEX IF NOT EXISTS ix_simhashes_b2 ON simhashes(b2);
CREATE INDEX IF NOT EXISTS ix_simhashes_b3 ON simhashes(b3);

PRAGMA user_version = 1;

