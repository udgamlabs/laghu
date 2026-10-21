"""Async DeepSeek client with a hard spend cap, retries, a raw-response cache and usage-based cost tracking.

- Endpoint: POST https://api.deepseek.com/chat/completions (OpenAI-compatible), model `deepseek-flash`
  (= DeepSeek-V4.1-Flash-1017, MIT open weights).
- The API key is read at runtime from the DEEPSEEK_API_KEY environment variable. It is never printed, logged or written anywhere; it only goes into the Authorization header of the HTTPS request.
- Cost comes from each response's `usage` (prompt_cache_hit_tokens / prompt_cache_miss_tokens / completion_tokens).
  Prices (USD per 1M tokens, off-peak, valid all weekend 2026-10-17/18): miss 0.15, hit 0.003, output 0.60.
- Spend cap: before a request is sent, spent + reserved(in-flight worst cases) + this request's worst case must stay
  below the cap; otherwise SpendCapReached is raised and no more requests start. The running total is saved to
  spend.json after every response (resumable across runs: the file is loaded at start).
- Cache: every successful response is appended to raw_cache.jsonl keyed by a hash of the request body, so a re-run
  with the same request costs nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import subprocess

import aiohttp

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
GENERATOR = "deepseek-flash (DeepSeek-V4.1-Flash-1017)"
PRICE = {"miss": 0.15e-6, "hit": 0.003e-6, "out": 0.60e-6}     # USD per token, off-peak


class SpendCapReached(Exception):
    pass


def _read_key() -> str:
    k = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not k:
        raise SystemExit("Set DEEPSEEK_API_KEY")
    return k


def cost_of(usage: dict) -> float:
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = int(usage.get("prompt_tokens") or 0) - hit
    out = int(usage.get("completion_tokens") or 0)
    return hit * PRICE["hit"] + int(miss) * PRICE["miss"] + out * PRICE["out"]


def req_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]


class DeepSeek:
    def __init__(self, out_dir: Path, cap_usd: float, concurrency: int = 16, timeout_s: float = 240.0,
                 max_retries: int = 5):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cap = float(cap_usd)
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.max_retries = max_retries
        self.spend_path = self.out_dir / "spend.json"
        self.cache_path = self.out_dir / "raw_cache.jsonl"
        self.spend = {"usd": 0.0, "requests": 0, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
                      "completion_tokens": 0, "reasoning_tokens": 0, "errors": 0, "cache_reuses": 0,
                      "by_slice": {}, "prices_per_1M": {"input_cache_miss": 0.15, "input_cache_hit": 0.003,
                                                         "output": 0.60}, "cap_usd": self.cap}
        if self.spend_path.exists():
            old = json.loads(self.spend_path.read_text())
            for k in ("usd", "requests", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens",
                      "reasoning_tokens", "errors", "cache_reuses"):
                self.spend[k] = old.get(k, 0)
            self.spend["by_slice"] = old.get("by_slice", {})
        self.reserved = 0.0
        self.stopped = False
        # identical requests made concurrently get different replies; keep all of them and serve them in order
        # on replay, so a re-run reproduces the same variety instead of repeating one reply
        self.cache: dict[str, list] = {}
        self._hits: dict[str, int] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                    self.cache.setdefault(r["key"], []).append(r["response"])
                except (json.JSONDecodeError, KeyError):
                    continue
        self.served: list[dict] = []          # every request this run used (fresh or from cache), for accounting
        self._key = _read_key()
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()
        self._save_spend()
        with (self.out_dir / "run_requests.jsonl").open("w", encoding="utf-8") as f:
            for x in self.served:
                f.write(json.dumps(x) + "\n")

    def _save_spend(self):
        tmp = self.spend_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.spend, indent=1))
        tmp.replace(self.spend_path)

    @staticmethod
    def worst_case(body: dict) -> float:
        chars = sum(len(m.get("content") or "") for m in body["messages"])
        return (chars / 3.0) * PRICE["miss"] + int(body.get("max_tokens", 2048)) * PRICE["out"]

    async def chat(self, messages: list[dict], *, slice_name: str, thinking: bool = False, json_mode: bool = False,
                   max_tokens: int = 1200, temperature: float = 0.7, extra: dict | None = None) -> dict:
        """Returns {"content", "reasoning_content", "usage", "cost", "cached", "finish_reason"}."""
        body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "stream": False}
        if not thinking:
            body["thinking"] = {"type": "disabled"}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if extra:
            body.update(extra)
        key = req_key(body)
        if key in self.cache:
            self.spend["cache_reuses"] += 1
            i = self._hits.get(key, 0)
            self._hits[key] = i + 1
            r = dict(self.cache[key][i % len(self.cache[key])])
            self.served.append({"key": key, "slice": slice_name, "cost_when_made": cost_of(r.get("usage") or {}),
                                "cached": True})
            r["cached"] = True
            r["cost"] = 0.0
            return r
        wc = self.worst_case(body)
        async with self.sem:                     # only in-flight requests reserve their worst case
            async with self._lock:
                if self.stopped or self.spend["usd"] + self.reserved + wc > self.cap:
                    self.stopped = True
                    raise SpendCapReached(f"spent ${self.spend['usd']:.4f} + reserved ${self.reserved:.4f}")
                self.reserved += wc
            try:
                data = await self._post(body)
            finally:
                async with self._lock:
                    self.reserved -= wc
        usage = data.get("usage") or {}
        c = cost_of(usage)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        out = {"content": msg.get("content") or "", "reasoning_content": msg.get("reasoning_content") or "",
               "usage": usage, "finish_reason": choice.get("finish_reason"), "cached": False}
        async with self._lock:
            s = self.spend
            s["usd"] += c
            s["requests"] += 1
            s["prompt_cache_hit_tokens"] += int(usage.get("prompt_cache_hit_tokens") or 0)
            s["prompt_cache_miss_tokens"] += int(usage.get("prompt_cache_miss_tokens") or 0)
            s["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            s["reasoning_tokens"] += int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            b = s["by_slice"].setdefault(slice_name, {"usd": 0.0, "requests": 0, "completion_tokens": 0,
                                                      "prompt_tokens": 0})
            b["usd"] += c
            b["requests"] += 1
            b["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            b["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self._save_spend()
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "slice": slice_name, "t": time.time(), "response": out},
                                   ensure_ascii=False) + "\n")
            self.cache.setdefault(key, []).append(out)
            self._hits[key] = self._hits.get(key, 0) + 1
            if s["usd"] >= self.cap:
                self.stopped = True
        self.served.append({"key": key, "slice": slice_name, "cost_when_made": c, "cached": False})
        out = dict(out)
        out["cost"] = c
        return out

    async def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        delay = 2.0
        last = None
        for attempt in range(self.max_retries):
            try:
                async with self._session.post(API_URL, json=body, headers=headers) as r:
                    if r.status == 200:
                        return await r.json()
                    text = (await r.text())[:300]
                    last = f"HTTP {r.status}: {text}"
                    if r.status in (400, 401, 402, 403, 422):
                        self.spend["errors"] += 1
                        raise RuntimeError(last)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last = f"{type(e).__name__}: {e}"
            self.spend["errors"] += 1
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 30)
        raise RuntimeError(f"request failed after {self.max_retries} tries: {last}")
