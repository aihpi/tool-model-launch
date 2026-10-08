#!/usr/bin/env python3
"""sml GPU pool agent.

One OpenAI-compatible HTTP endpoint (the job's framework port) fronting several
vLLM servers that share the job's GPUs through vLLM sleep mode:

* every catalog model is started once at boot and then put to sleep (level 1:
  weights offloaded to host RAM, KV cache dropped, GPU memory released);
* a request for model M wakes M in seconds, after draining and sleeping the
  least-recently-used awake models until M's GPU fraction fits;
* awake models idle for ``sleep_after`` go back to sleep.

Runs inside the vllm-openai container: Python >= 3.11 (tomllib); starlette,
uvicorn and httpx ship with vLLM. Config is TOML, see hpi/pool.toml. The pure
planning functions (plan_wake / plan_sleep) are unit-tested without vLLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import tomllib

COLD, STARTING, AWAKE, ASLEEP = "cold", "starting", "awake", "asleep"
_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}


def parse_duration(value: str | int | float) -> float:
    """'5m', '90s', '2h' or a bare number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    units = {"s": 1, "m": 60, "h": 3600}
    text = value.strip()
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


@dataclass
class ModelSpec:
    served_name: str  # what LiteLLM sends as `model`
    model: str  # HF id or path given to `vllm serve`
    gpus: list[int]  # CUDA_VISIBLE_DEVICES; len == tensor-parallel-size
    gpu_fraction: float  # --gpu-memory-utilization
    args: str = ""


@dataclass
class PoolConfig:
    models: list[ModelSpec]
    sleep_after: float = 300.0
    sleep_level: int = 1
    gpu_headroom: float = 0.05
    drain_timeout: float = 600.0
    start_timeout: float = 1800.0

    @classmethod
    def load(cls, path: str) -> PoolConfig:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PoolConfig:
        models = [
            ModelSpec(
                served_name=m["served_name"],
                model=m.get("model", m["served_name"]),
                gpus=[int(g) for g in m["gpus"]],
                gpu_fraction=float(m["gpu_fraction"]),
                args=m.get("args", ""),
            )
            for m in raw.get("models", [])
        ]
        cfg = cls(
            models=models,
            sleep_after=parse_duration(raw.get("sleep_after", "5m")),
            sleep_level=int(raw.get("sleep_level", 1)),
            gpu_headroom=float(raw.get("gpu_headroom", 0.05)),
            drain_timeout=parse_duration(raw.get("drain_timeout", "10m")),
            start_timeout=parse_duration(raw.get("start_timeout", "30m")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.models:
            raise ValueError("pool config has no [[models]]")
        names = [m.served_name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate served_name in pool config: {names}")
        limit = 1.0 - self.gpu_headroom
        for m in self.models:
            if not m.gpus:
                raise ValueError(f"{m.served_name}: gpus must not be empty")
            if not 0 < m.gpu_fraction <= limit:
                raise ValueError(
                    f"{m.served_name}: gpu_fraction {m.gpu_fraction} must be in (0, {limit:.2f}] (1 - gpu_headroom)"
                )
        # ponytail: level 2 needs reload_weights + reset_prefix_cache after waking; add when RAM is the bottleneck.
        if self.sleep_level != 1:
            raise ValueError("only sleep_level = 1 is supported")


@dataclass
class Entry:
    spec: ModelSpec
    port: int = 0
    state: str = COLD
    in_flight: int = 0
    last_used: float = 0.0
    proc: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.spec.served_name


# ── planning (pure) ───────────────────────────────────────────────────────────


def plan_wake(entries: dict[str, Entry], name: str, headroom: float) -> list[str]:
    """Awake models to put to sleep so ``name`` fits on each of its GPUs, LRU first."""
    target = entries[name]
    victims: list[str] = []
    for gpu in target.spec.gpus:
        awake = [
            e
            for e in entries.values()
            if e is not target and e.state == AWAKE and gpu in e.spec.gpus and e.name not in victims
        ]
        used = sum(e.spec.gpu_fraction for e in awake)
        budget = 1.0 - headroom - target.spec.gpu_fraction
        while used > budget + 1e-9 and awake:
            victim = min(awake, key=lambda e: e.last_used)
            awake.remove(victim)
            victims.append(victim.name)
            used -= victim.spec.gpu_fraction
        if used > budget + 1e-9:  # pragma: no cover - PoolConfig.validate rules this out
            raise RuntimeError(f"{name} cannot fit on GPU {gpu}")
    return victims


def plan_sleep(entries: dict[str, Entry], now: float, sleep_after: float) -> list[str]:
    """Awake, idle (no in-flight request) models unused for at least ``sleep_after``."""
    return [
        e.name for e in entries.values() if e.state == AWAKE and e.in_flight == 0 and now - e.last_used >= sleep_after
    ]


# ── runtime ───────────────────────────────────────────────────────────────────


def _free_port() -> int:
    # Children bind localhost only; on a shared node any fixed scheme can collide
    # with another job, so let the OS pick (the usual tiny bind race is accepted).
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def log(msg: str) -> None:
    print(f"[pool {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Pool:
    def __init__(self, cfg: PoolConfig) -> None:
        self.cfg = cfg
        self.entries = {m.served_name: Entry(m) for m in cfg.models}
        # ponytail: one global lock for all state transitions (they are rare and
        # take seconds once everything is resident); per-model locks + GPU
        # reservations if cold starts blocking unrelated models ever matters.
        self.lock = asyncio.Lock()
        self.ready = False
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))

    def _url(self, e: Entry, path: str) -> str:
        return f"http://127.0.0.1:{e.port}{path}"

    # child control -------------------------------------------------------------

    def _spawn(self, e: Entry) -> None:
        spec = e.spec
        e.port = _free_port()
        cmd = (
            f"vllm serve {spec.model} --served-model-name {spec.served_name} "
            f"--host 127.0.0.1 --port {e.port} --enable-sleep-mode "
            f"--gpu-memory-utilization {spec.gpu_fraction} --tensor-parallel-size {len(spec.gpus)} {spec.args}"
        )
        env = {
            **os.environ,
            "VLLM_SERVER_DEV_MODE": "1",  # exposes /sleep, /wake_up, /is_sleeping
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in spec.gpus),
        }
        log(f"start {e.name}: {cmd}")
        e.proc = subprocess.Popen(shlex.split(cmd), env=env)  # noqa: S603 - our own command line
        e.state = STARTING

    async def _wait_healthy(self, e: Entry) -> None:
        deadline = time.monotonic() + self.cfg.start_timeout
        while time.monotonic() < deadline:
            if e.proc is not None and e.proc.poll() is not None:
                raise RuntimeError(f"{e.name} exited with code {e.proc.returncode} while starting")
            try:
                r = await self.http.get(self._url(e, "/health"), timeout=5.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
        raise TimeoutError(f"{e.name} not healthy after {self.cfg.start_timeout:.0f}s")

    async def _sleep(self, e: Entry) -> None:
        # Sleeping mid-request crashes vLLM with a CUDA illegal memory access, so
        # drain first; the caller holds the lock, which also gates new requests.
        deadline = time.monotonic() + self.cfg.drain_timeout
        while e.in_flight and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if e.in_flight:
            raise TimeoutError(
                f"{e.name} still has {e.in_flight} requests in flight after {self.cfg.drain_timeout:.0f}s"
            )
        r = await self.http.post(self._url(e, "/sleep"), params={"level": self.cfg.sleep_level}, timeout=300.0)
        r.raise_for_status()
        e.state = ASLEEP
        log(f"asleep {e.name}")

    async def _wake(self, e: Entry) -> None:
        r = await self.http.post(self._url(e, "/wake_up"), timeout=600.0)
        r.raise_for_status()
        while (await self.http.get(self._url(e, "/is_sleeping"), timeout=10.0)).json().get("is_sleeping"):
            await asyncio.sleep(0.2)
        e.state = AWAKE
        e.last_used = time.monotonic()
        log(f"awake {e.name}")

    def _refresh(self, e: Entry) -> None:
        if e.proc is not None and e.proc.poll() is not None:
            log(f"{e.name} died with code {e.proc.returncode}; will restart on demand")
            e.proc = None
            e.state = COLD

    # state machine -------------------------------------------------------------

    async def ensure_awake(self, name: str) -> Entry:
        e = self.entries[name]
        async with self.lock:
            self._refresh(e)
            if e.state == AWAKE:
                return e
            for victim in plan_wake(self.entries, name, self.cfg.gpu_headroom):
                await self._sleep(self.entries[victim])
            if e.state == COLD:
                self._spawn(e)
                await self._wait_healthy(e)
                e.state = AWAKE
                e.last_used = time.monotonic()
                log(f"awake {e.name} (fresh start)")
            elif e.state == ASLEEP:
                await self._wake(e)
            return e

    async def boot(self) -> None:
        # Sequential: each vLLM asserts its gpu_fraction is *free* at start, so
        # the previous one must already be asleep. End state: all resident, GPUs empty.
        for name in self.entries:
            try:
                await self.ensure_awake(name)
                async with self.lock:
                    await self._sleep(self.entries[name])
            except Exception as exc:  # keep booting the others; this one restarts on demand
                log(f"boot: {name} failed: {exc}")
        self.ready = True
        log("boot complete: " + ", ".join(f"{e.name}={e.state}" for e in self.entries.values()))

    async def sleeper(self, interval: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval)
            for name in plan_sleep(self.entries, time.monotonic(), self.cfg.sleep_after):
                async with self.lock:
                    e = self.entries[name]
                    if e.state == AWAKE and e.in_flight == 0:
                        try:
                            await self._sleep(e)
                        except Exception as exc:
                            log(f"sleep {name} failed: {exc}")

    def terminate(self) -> None:
        for e in self.entries.values():
            if e.proc is not None and e.proc.poll() is None:
                e.proc.terminate()


# ── HTTP front door ───────────────────────────────────────────────────────────


def build_app(pool: Pool, *, run_background: bool = True) -> Any:
    from contextlib import asynccontextmanager

    from starlette.applications import Starlette
    from starlette.background import BackgroundTask
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    def error(status: int, message: str, **headers: str) -> Response:
        return JSONResponse({"error": {"message": message, "type": "pool_error"}}, status_code=status, headers=headers)

    async def health(request: Request) -> Response:
        # 200 only once every model is resident: sml's health checker (and the
        # consecutive-chain handover) must not consider a still-booting pool healthy.
        return JSONResponse({"status": "ok" if pool.ready else "booting"}, status_code=200 if pool.ready else 503)

    async def models(request: Request) -> Response:
        data = [
            {"id": e.name, "object": "model", "owned_by": "sml-pool", "state": e.state, "gpus": e.spec.gpus}
            for e in pool.entries.values()
        ]
        return JSONResponse({"object": "list", "data": data})

    async def proxy(request: Request) -> Response:
        body = await request.body()
        try:
            name = json.loads(body).get("model") if body else None
        except (ValueError, AttributeError):
            name = None
        if name not in pool.entries:
            return error(404, f"unknown model {name!r}; this pool serves {sorted(pool.entries)}")
        try:
            e = await pool.ensure_awake(name)
        except Exception as exc:
            return error(503, f"{name} is unavailable: {exc}", **{"Retry-After": "30"})
        e.in_flight += 1
        e.last_used = time.monotonic()
        url = pool._url(e, request.url.path) + (f"?{request.url.query}" if request.url.query else "")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        try:
            upstream = await pool.http.send(
                pool.http.build_request(request.method, url, headers=headers, content=body), stream=True
            )
        except httpx.HTTPError as exc:
            e.in_flight -= 1
            return error(502, f"{name}: {exc}")

        async def done() -> None:
            await upstream.aclose()
            e.in_flight -= 1
            e.last_used = time.monotonic()

        async def stream() -> AsyncIterator[bytes]:
            async for chunk in upstream.aiter_raw():
                yield chunk

        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS}
        return StreamingResponse(
            stream(), status_code=upstream.status_code, headers=resp_headers, background=BackgroundTask(done)
        )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        tasks = [asyncio.create_task(pool.boot()), asyncio.create_task(pool.sleeper())] if run_background else []
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            pool.terminate()

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/{path:path}", proxy, methods=["GET", "POST"]),
        ],
        lifespan=lifespan,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="pool TOML (see hpi/pool.toml)")
    ap.add_argument("--port", type=int, required=True, help="front-door port (sml passes the framework port)")
    ap.add_argument("--host", default="0.0.0.0")  # noqa: S104 - the job's health checker probes the node IP
    args = ap.parse_args(argv)
    cfg = PoolConfig.load(os.path.expanduser(args.config))
    import uvicorn

    uvicorn.run(build_app(Pool(cfg)), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
