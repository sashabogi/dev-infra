"""Memory rescue engine — extracts facts, decisions, and skills before context compaction."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from rescue.extractors import (
    DecisionExtractor,
    ExtractionConfig,
    FactExtractor,
    SkillExtractor,
)
from rescue.store import Memory, MemoryStore, RescueRun

logger = logging.getLogger("rescue.engine")


class RescueEngine:
    def __init__(self, config: dict | None = None):
        self._config = config or {}
        rescue_cfg = self._config.get("rescue", {})

        self._importance_threshold: int = rescue_cfg.get("importance_threshold", 7)
        self._max_context_chars: int = rescue_cfg.get("max_context_chars", 100_000)

        # Resolve extraction model from config
        extraction_model_name = rescue_cfg.get("extraction_model", "cerebras")
        providers = self._config.get("providers", {})
        model_id = "llama-3.3-70b"
        if extraction_model_name in providers:
            model_id = providers[extraction_model_name].get("model", model_id)

        ext_config = ExtractionConfig(model=model_id)

        self._fact_extractor = FactExtractor(ext_config)
        self._decision_extractor = DecisionExtractor(ext_config)
        self._skill_extractor = SkillExtractor(ext_config)
        self._store = MemoryStore(self._config)

    async def rescue_context(
        self,
        context_text: str,
        project: str | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        """Main entry point: extract and save memories from context."""
        start = time.monotonic()

        # Truncate if too long
        if len(context_text) > self._max_context_chars:
            logger.info(
                "Context truncated from %d to %d chars",
                len(context_text),
                self._max_context_chars,
            )
            context_text = context_text[: self._max_context_chars]

        # Fan out to 3 parallel extractors
        fact_task = asyncio.create_task(
            self._fact_extractor.extract(context_text, project, session_id)
        )
        decision_task = asyncio.create_task(
            self._decision_extractor.extract(context_text, project, session_id)
        )
        skill_task = asyncio.create_task(
            self._skill_extractor.extract(context_text, project, session_id)
        )

        results = await asyncio.gather(
            fact_task, decision_task, skill_task, return_exceptions=True
        )

        all_memories: list[Memory] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Extractor failed: %s", result)
                continue
            all_memories.extend(result)

        total_extracted = len(all_memories)
        logger.info("Extracted %d raw memories", total_extracted)

        # Filter by importance threshold
        important = [
            m for m in all_memories if m.importance >= self._importance_threshold
        ]
        logger.info(
            "After importance filter (>=%d): %d memories",
            self._importance_threshold,
            len(important),
        )

        # Save with deduplication
        committed: list[Memory] = []
        for memory in important:
            saved = self._store.save_memory(memory)
            if saved:
                committed.append(memory)
            else:
                logger.debug("Duplicate skipped: %s", memory.content[:80])

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Rescue complete: %d extracted, %d committed in %dms",
            total_extracted,
            len(committed),
            duration_ms,
        )

        # Record the run
        self._store.save_run(
            RescueRun(
                project=project,
                session_id=session_id,
                context_length=len(context_text),
                memories_extracted=total_extracted,
                memories_committed=len(committed),
                duration_ms=duration_ms,
            )
        )

        return committed

    def search(
        self,
        query: str,
        project: str | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search rescued memories."""
        return self._store.search(query, project, category, limit)

    def get_stats(self) -> dict:
        """Get memory database statistics."""
        return self._store.get_stats()

    def close(self) -> None:
        self._store.close()
