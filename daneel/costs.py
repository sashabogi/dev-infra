"""Cost tracking with SQLite persistence."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CostRecord:
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    quality_score: float
    latency_ms: int
    status: str
    role: str | None = None


# Estimated Opus 4 pricing per million tokens
OPUS_COST_PER_M_INPUT = 15.0
OPUS_COST_PER_M_OUTPUT = 75.0


def _db_path(config: dict | None = None) -> Path:
    if config and "costs" in config and "db_path" in config["costs"]:
        p = config["costs"]["db_path"]
    else:
        p = "~/.dev-infra/costs.db"
    return Path(os.path.expanduser(p))


def _ensure_db(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS costs (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            quality_score REAL NOT NULL,
            latency_ms INTEGER NOT NULL,
            status TEXT NOT NULL,
            role TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_costs_timestamp ON costs(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_costs_provider ON costs(provider)
    """)
    # Migration: add role column to existing databases
    try:
        conn.execute("ALTER TABLE costs ADD COLUMN role TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_costs_role ON costs(role)
    """)
    conn.commit()
    return conn


class CostTracker:
    def __init__(self, config: dict | None = None):
        self._db_path = _db_path(config)
        self._conn = _ensure_db(self._db_path)

    def record(self, rec: CostRecord) -> None:
        self._conn.execute(
            """INSERT INTO costs
               (id, timestamp, provider, model, tokens_in, tokens_out,
                cost_usd, quality_score, latency_ms, status, role)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                time.time(),
                rec.provider,
                rec.model,
                rec.tokens_in,
                rec.tokens_out,
                rec.cost_usd,
                rec.quality_score,
                rec.latency_ms,
                rec.status,
                rec.role,
            ),
        )
        self._conn.commit()

    def get_rolling_costs(self, hours: int = 24) -> dict:
        cutoff = time.time() - (hours * 3600)
        rows = self._conn.execute(
            """SELECT provider, model,
                      COUNT(*) as requests,
                      SUM(tokens_in) as total_in,
                      SUM(tokens_out) as total_out,
                      SUM(cost_usd) as total_cost,
                      AVG(quality_score) as avg_quality,
                      AVG(latency_ms) as avg_latency,
                      SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                      SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
                      SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejections
               FROM costs
               WHERE timestamp > ?
               GROUP BY provider, model
               ORDER BY total_cost DESC""",
            (cutoff,),
        ).fetchall()

        providers = {}
        total_cost = 0.0
        total_in = 0
        total_out = 0

        for row in rows:
            (
                provider, model, requests, t_in, t_out, cost,
                avg_q, avg_lat, successes, failures, rejections,
            ) = row
            providers[provider] = {
                "model": model,
                "requests": requests,
                "tokens_in": t_in,
                "tokens_out": t_out,
                "cost_usd": round(cost, 6),
                "avg_quality": round(avg_q, 3),
                "avg_latency_ms": round(avg_lat, 1),
                "successes": successes,
                "failures": failures,
                "rejections": rejections,
            }
            total_cost += cost
            total_in += t_in
            total_out += t_out

        opus_cost = (total_in / 1_000_000) * OPUS_COST_PER_M_INPUT + \
                    (total_out / 1_000_000) * OPUS_COST_PER_M_OUTPUT

        # Per-role breakdown (only includes requests with a role set)
        role_rows = self._conn.execute(
            """SELECT role,
                      COUNT(*) as requests,
                      SUM(cost_usd) as total_cost,
                      SUM(tokens_in) as total_in,
                      SUM(tokens_out) as total_out
               FROM costs
               WHERE timestamp > ? AND role IS NOT NULL
               GROUP BY role
               ORDER BY total_cost DESC""",
            (cutoff,),
        ).fetchall()

        roles: dict[str, dict] = {}
        for rrow in role_rows:
            r_name, r_requests, r_cost, r_in, r_out = rrow
            roles[r_name] = {
                "requests": r_requests,
                "cost_usd": round(r_cost, 6),
                "tokens_in": r_in,
                "tokens_out": r_out,
            }

        result = {
            "window_hours": hours,
            "providers": providers,
            "total_cost_usd": round(total_cost, 6),
            "opus_equivalent_usd": round(opus_cost, 4),
            "savings_usd": round(opus_cost - total_cost, 4),
            "savings_pct": round(
                ((opus_cost - total_cost) / opus_cost * 100) if opus_cost > 0 else 0, 1
            ),
        }
        if roles:
            result["roles"] = roles
        return result

    def get_total_saved(self) -> dict:
        row = self._conn.execute(
            """SELECT SUM(tokens_in), SUM(tokens_out), SUM(cost_usd), COUNT(*)
               FROM costs WHERE status = 'success'"""
        ).fetchone()

        total_in = row[0] or 0
        total_out = row[1] or 0
        total_cost = row[2] or 0.0
        total_requests = row[3] or 0

        opus_cost = (total_in / 1_000_000) * OPUS_COST_PER_M_INPUT + \
                    (total_out / 1_000_000) * OPUS_COST_PER_M_OUTPUT

        return {
            "total_requests": total_requests,
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "actual_cost_usd": round(total_cost, 6),
            "opus_equivalent_usd": round(opus_cost, 4),
            "total_saved_usd": round(opus_cost - total_cost, 4),
        }

    def close(self) -> None:
        self._conn.close()
