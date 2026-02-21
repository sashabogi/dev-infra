"""SQLite + FTS5 memory store for rescued memories."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Memory:
    category: str  # fact | decision | skill
    subcategory: str
    content: str
    importance: int
    project: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    commit_hash: str = ""
    created_at: str = ""
    accessed_at: str = ""
    access_count: int = 0

    def __post_init__(self):
        if not self.commit_hash:
            self.commit_hash = hashlib.sha256(
                f"{self.category}:{self.subcategory}:{self.content}".encode()
            ).hexdigest()
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.accessed_at:
            self.accessed_at = self.created_at


@dataclass
class RescueRun:
    project: str | None
    session_id: str | None
    context_length: int
    memories_extracted: int
    memories_committed: int
    duration_ms: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _db_path(config: dict | None = None) -> Path:
    if config and "rescue" in config and "db_path" in config["rescue"]:
        p = config["rescue"]["db_path"]
    else:
        p = "~/.dev-infra/memory.db"
    return Path(os.path.expanduser(p))


class MemoryStore:
    def __init__(self, config: dict | None = None):
        self._db_path = _db_path(config)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                subcategory TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER NOT NULL,
                project TEXT,
                session_id TEXT,
                commit_hash TEXT UNIQUE,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_category
                ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_project
                ON memories(project);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_commit_hash
                ON memories(commit_hash);

            CREATE TABLE IF NOT EXISTS rescue_runs (
                id TEXT PRIMARY KEY,
                project TEXT,
                session_id TEXT,
                context_length INTEGER NOT NULL,
                memories_extracted INTEGER NOT NULL,
                memories_committed INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

        # Create FTS5 virtual table (ignore error if already exists)
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, category, subcategory, project,
                    tokenize='porter unicode61'
                )
            """)
        except sqlite3.OperationalError:
            pass  # FTS5 table already exists

        self._conn.commit()

    def deduplicate(self, commit_hash: str) -> bool:
        """Check if a memory with this hash already exists. Returns True if duplicate."""
        row = self._conn.execute(
            "SELECT 1 FROM memories WHERE commit_hash = ?", (commit_hash,)
        ).fetchone()
        return row is not None

    def save_memory(self, memory: Memory) -> bool:
        """Save a memory. Returns False if duplicate (skipped)."""
        if self.deduplicate(memory.commit_hash):
            return False

        self._conn.execute(
            """INSERT INTO memories
               (id, category, subcategory, content, importance, project,
                session_id, commit_hash, created_at, accessed_at,
                access_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.category,
                memory.subcategory,
                memory.content,
                memory.importance,
                memory.project,
                memory.session_id,
                memory.commit_hash,
                memory.created_at,
                memory.accessed_at,
                memory.access_count,
                json.dumps(memory.metadata),
            ),
        )

        # Insert into FTS index
        try:
            self._conn.execute(
                """INSERT INTO memories_fts (rowid, content, category, subcategory, project)
                   VALUES (
                       (SELECT rowid FROM memories WHERE id = ?),
                       ?, ?, ?, ?
                   )""",
                (
                    memory.id,
                    memory.content,
                    memory.category,
                    memory.subcategory,
                    memory.project or "",
                ),
            )
        except sqlite3.OperationalError:
            pass  # FTS insert failed, non-fatal

        self._conn.commit()
        return True

    def save_run(self, run: RescueRun) -> None:
        self._conn.execute(
            """INSERT INTO rescue_runs
               (id, project, session_id, context_length,
                memories_extracted, memories_committed, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.id,
                run.project,
                run.session_id,
                run.context_length,
                run.memories_extracted,
                run.memories_committed,
                run.duration_ms,
                run.created_at,
            ),
        )
        self._conn.commit()

    def search(
        self,
        query: str,
        project: str | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search with BM25 ranking."""
        try:
            sql = """
                SELECT m.id, m.category, m.subcategory, m.content,
                       m.importance, m.project, m.session_id,
                       m.created_at, m.access_count,
                       bm25(memories_fts) as rank
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ?
            """
            params: list[Any] = [query]

            if project:
                sql += " AND m.project = ?"
                params.append(project)
            if category:
                sql += " AND m.category = ?"
                params.append(category)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            rows = self._conn.execute(sql, params).fetchall()

            # Update access counts
            now = datetime.now(timezone.utc).isoformat()
            for row in rows:
                self._conn.execute(
                    """UPDATE memories
                       SET access_count = access_count + 1, accessed_at = ?
                       WHERE id = ?""",
                    (now, row[0]),
                )
            self._conn.commit()

            return [
                {
                    "id": r[0],
                    "category": r[1],
                    "subcategory": r[2],
                    "content": r[3],
                    "importance": r[4],
                    "project": r[5],
                    "session_id": r[6],
                    "created_at": r[7],
                    "access_count": r[8],
                    "rank": r[9],
                }
                for r in rows
            ]

        except sqlite3.OperationalError:
            # FTS might not be populated; fall back to LIKE search
            sql = """
                SELECT id, category, subcategory, content, importance,
                       project, session_id, created_at, access_count
                FROM memories
                WHERE content LIKE ?
            """
            params_like: list[Any] = [f"%{query}%"]

            if project:
                sql += " AND project = ?"
                params_like.append(project)
            if category:
                sql += " AND category = ?"
                params_like.append(category)

            sql += " ORDER BY importance DESC LIMIT ?"
            params_like.append(limit)

            rows = self._conn.execute(sql, params_like).fetchall()
            return [
                {
                    "id": r[0],
                    "category": r[1],
                    "subcategory": r[2],
                    "content": r[3],
                    "importance": r[4],
                    "project": r[5],
                    "session_id": r[6],
                    "created_at": r[7],
                    "access_count": r[8],
                    "rank": 0,
                }
                for r in rows
            ]

    def get_stats(self) -> dict:
        """Get database statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        by_category = {}
        for row in self._conn.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        ).fetchall():
            by_category[row[0]] = row[1]

        by_project = {}
        for row in self._conn.execute(
            """SELECT COALESCE(project, 'unassigned'), COUNT(*)
               FROM memories GROUP BY project ORDER BY COUNT(*) DESC LIMIT 20"""
        ).fetchall():
            by_project[row[0]] = row[1]

        recent_runs = []
        for row in self._conn.execute(
            """SELECT id, project, context_length, memories_extracted,
                      memories_committed, duration_ms, created_at
               FROM rescue_runs ORDER BY created_at DESC LIMIT 10"""
        ).fetchall():
            recent_runs.append(
                {
                    "id": row[0],
                    "project": row[1],
                    "context_length": row[2],
                    "memories_extracted": row[3],
                    "memories_committed": row[4],
                    "duration_ms": row[5],
                    "created_at": row[6],
                }
            )

        return {
            "total_memories": total,
            "by_category": by_category,
            "by_project": by_project,
            "recent_runs": recent_runs,
        }

    def close(self) -> None:
        self._conn.close()
