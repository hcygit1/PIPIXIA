"""Legacy AgentManager session and memory forwarding methods."""

from __future__ import annotations

from typing import Any, AsyncGenerator

from runtime.session_lifecycle import SessionLifecycle


class AgentManagerSessionCompatibilityMixin:
    """Keep legacy session-runtime patch points on AgentManager."""

    async def _ingest_completed_turn(
        self,
        agent_id: str,
        session_id: str,
        user_content: str,
        assistant_content: str,
        turn_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> None:
        await self._incremental_ingest(
            agent_id,
            session_id,
            user_content,
            assistant_content,
            turn_id,
            parent_task_id,
        )

    async def _run_auto_compaction(
        self,
        session_id: str,
        agent_id: str,
        **kwargs: Any,
    ) -> None:
        await self._maybe_auto_compact(
            session_id,
            agent_id,
            **kwargs,
        )

    async def _compress_for_recovery(
        self,
        session_id: str,
        agent_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self.compress_session(
            session_id,
            agent_id,
            **kwargs,
        )

    async def _incremental_ingest(
        self,
        agent_id: str,
        session_id: str,
        user_content: str,
        assistant_content: str,
        turn_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> None:
        await self._memory_runtime.ingest_turn(
            agent_id,
            session_id,
            user_content,
            assistant_content,
            turn_id=turn_id,
            parent_task_id=parent_task_id,
        )

    async def _batch_ingest_messages(
        self,
        agent_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        session_end: bool = False,
        parent_task_id: str | None = None,
    ) -> None:
        await self._memory_runtime.ingest_messages(
            agent_id,
            session_id,
            messages,
            session_end=session_end,
            parent_task_id=parent_task_id,
        )

    async def _maybe_auto_compact(
        self,
        session_id: str,
        agent_id: str,
        overhead_tokens: int = 0,
    ) -> None:
        await self._session_lifecycle.maybe_auto_compact(
            session_id,
            agent_id,
            overhead_tokens=overhead_tokens,
            compress_session=self.compress_session,
        )

    async def _handle_reset(
        self,
        session_id: str,
        agent_id: str,
        model_override: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._session_commands.handle_reset(
            session_id,
            agent_id,
            model_override=model_override,
            get_store=lambda target_agent_id: self.mem_stores.get(
                target_agent_id
            ),
            get_state=self.get_state,
            batch_ingest_messages=self._batch_ingest_messages,
            switch_model=self.switch_model,
        ):
            yield event

    async def _handle_reset_noflush(
        self,
        session_id: str,
        agent_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._session_commands.handle_reset_noflush(
            session_id,
            agent_id,
            get_store=lambda target_agent_id: self.mem_stores.get(
                target_agent_id
            ),
            get_state=self.get_state,
        ):
            yield event

    async def _handle_compact(
        self,
        session_id: str,
        agent_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._session_commands.handle_compact(
            session_id,
            agent_id,
            compress_session=self.compress_session,
            calc_compress_count_by_turns=(
                self._calc_compress_count_by_turns
            ),
        ):
            yield event

    async def _generate_structured_summary(
        self,
        agent_id: str,
        session_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        return await self._session_lifecycle.generate_structured_summary(
            agent_id,
            session_id,
            to_compress,
            text_to_summarize,
            plain_fallback=self._summarize_plain_fallback,
        )

    async def _summarize_plain_fallback(
        self,
        agent_id: str,
        to_compress: list[dict[str, Any]],
        text_to_summarize: str,
    ) -> dict[str, Any]:
        return await self._session_lifecycle.summarize_plain_fallback(
            agent_id,
            to_compress,
            text_to_summarize,
        )

    @staticmethod
    def _calc_compress_count_by_turns(
        messages: list[dict[str, Any]],
        keep_turns: int,
    ) -> int:
        return SessionLifecycle.calc_compress_count_by_turns(
            messages,
            keep_turns,
        )
