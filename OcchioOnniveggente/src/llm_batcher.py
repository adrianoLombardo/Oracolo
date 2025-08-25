from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

from .local_llm import _load_model


@dataclass
class _Request:
    prompt: str
    future: asyncio.Future[str]


class LLMBatcher:
    """Simple asynchronous batcher for local LLM generation.

    Requests are queued and grouped together every ``batch_interval``
    milliseconds (up to ``max_batch_size`` prompts) before invoking the model
    ``generate`` method. Each result is dispatched back to the originating
    caller.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        precision: str,
        batch_interval: int,
        max_batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.batch_interval = batch_interval / 1000.0
        self.max_batch_size = max_batch_size
        self.queue: asyncio.Queue[_Request] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def generate(self, prompt: str) -> str:
        """Enqueue *prompt* and wait for the generated text."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._runner())
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await self.queue.put(_Request(prompt, fut))
        return await fut

    def generate_sync(self, prompt: str) -> str:
        """Synchronous wrapper for :meth:`generate`."""
        return asyncio.run(self.generate(prompt))

    async def _runner(self) -> None:
        tokenizer, model = _load_model(self.model_path, self.device, self.precision)
        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
        while True:
            req = await self.queue.get()
            batch: List[_Request] = [req]
            start = asyncio.get_event_loop().time()
            while len(batch) < self.max_batch_size:
                remaining = self.batch_interval - (asyncio.get_event_loop().time() - start)
                if remaining <= 0:
                    break
                try:
                    next_req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(next_req)
                except asyncio.TimeoutError:
                    break
            prompts = [r.prompt for r in batch]
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                outputs = model.generate(**inputs, pad_token_id=pad_token)
                texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for r, txt in zip(batch, texts):
                    r.future.set_result(txt.strip())
            except Exception as exc:  # pragma: no cover - runtime failures
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
