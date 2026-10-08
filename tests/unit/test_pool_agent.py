"""The GPU pool agent's pure core: config parsing/validation, LRU fit planning,
idle-sleep planning, and the state machine with vLLM stubbed out."""

import asyncio
import importlib.util
import sys
from importlib.resources import files
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "pool_agent", str(files("swiss_ai_model_launch.assets").joinpath("pool_agent.py"))
)
assert _spec is not None and _spec.loader is not None
pool_agent = importlib.util.module_from_spec(_spec)
sys.modules["pool_agent"] = pool_agent  # dataclasses resolve `from __future__` annotations via sys.modules
_spec.loader.exec_module(pool_agent)

AWAKE, ASLEEP, COLD = pool_agent.AWAKE, pool_agent.ASLEEP, pool_agent.COLD


def _spec_(name: str, gpus: list[int], fraction: float) -> "pool_agent.ModelSpec":
    return pool_agent.ModelSpec(served_name=name, model=name, gpus=gpus, gpu_fraction=fraction)


def _entries(*rows: tuple[str, list[int], float, str, float]) -> dict:
    out = {}
    for name, gpus, fraction, state, last_used in rows:
        out[name] = pool_agent.Entry(_spec_(name, gpus, fraction), state=state, last_used=last_used)
    return out


# ── config ────────────────────────────────────────────────────────────────────


def test_parse_duration() -> None:
    assert pool_agent.parse_duration("5m") == 300
    assert pool_agent.parse_duration("90s") == 90
    assert pool_agent.parse_duration("2h") == 7200
    assert pool_agent.parse_duration(42) == 42
    assert pool_agent.parse_duration("7") == 7


def test_config_load_defaults_and_fields(tmp_path: Path) -> None:
    toml = tmp_path / "pool.toml"
    toml.write_text(
        'sleep_after = "10m"\n'
        "[[models]]\n"
        'served_name = "alice/Qwen/Qwen3-0.6B"\n'
        'model = "Qwen/Qwen3-0.6B"\n'
        "gpus = [0]\n"
        "gpu_fraction = 0.45\n"
        'args = "--max-model-len 8192"\n'
        "[[models]]\n"
        'served_name = "alice/Qwen/Qwen3-32B"\n'
        "gpus = [0, 1]\n"
        "gpu_fraction = 0.9\n"
    )
    cfg = pool_agent.PoolConfig.load(str(toml))
    assert cfg.sleep_after == 600 and cfg.sleep_level == 1 and cfg.gpu_headroom == 0.05
    assert cfg.models[0].model == "Qwen/Qwen3-0.6B" and cfg.models[0].args == "--max-model-len 8192"
    assert cfg.models[1].model == "alice/Qwen/Qwen3-32B"  # defaults to served_name
    assert cfg.models[1].gpus == [0, 1]


@pytest.mark.parametrize(
    "raw,match",
    [
        ({}, "no \\[\\[models\\]\\]"),
        ({"models": [{"served_name": "a", "gpus": [0], "gpu_fraction": 0.5}] * 2}, "duplicate"),
        ({"models": [{"served_name": "a", "gpus": [0], "gpu_fraction": 0.97}]}, "gpu_fraction"),
        ({"models": [{"served_name": "a", "gpus": [], "gpu_fraction": 0.5}]}, "gpus"),
        ({"sleep_level": 2, "models": [{"served_name": "a", "gpus": [0], "gpu_fraction": 0.5}]}, "sleep_level"),
    ],
)
def test_config_validation(raw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pool_agent.PoolConfig.from_dict(raw)


# ── planning ──────────────────────────────────────────────────────────────────


def test_plan_wake_nothing_to_do_when_it_fits() -> None:
    entries = _entries(("a", [0], 0.45, AWAKE, 1.0), ("b", [0], 0.45, ASLEEP, 0.0))
    assert pool_agent.plan_wake(entries, "b", 0.05) == []


def test_plan_wake_evicts_lru_first_until_it_fits() -> None:
    entries = _entries(("a", [0], 0.45, AWAKE, 10.0), ("b", [0], 0.45, AWAKE, 20.0), ("c", [0], 0.9, ASLEEP, 0.0))
    assert pool_agent.plan_wake(entries, "c", 0.05) == ["a", "b"]
    # A smaller target only needs the single least-recently-used victim.
    entries["c"].spec.gpu_fraction = 0.45
    entries["c"].state = ASLEEP
    entries["d"] = pool_agent.Entry(_spec_("d", [0], 0.1), state=AWAKE, last_used=5.0)
    # used = 0.45 + 0.45 + 0.1 = 1.0; budget = 0.95 - 0.45 = 0.5 → evict d (5.0) then a (10.0)
    assert pool_agent.plan_wake(entries, "c", 0.05) == ["d", "a"]


def test_plan_wake_tensor_parallel_target_frees_every_gpu_it_spans() -> None:
    entries = _entries(("a", [0], 0.9, AWAKE, 1.0), ("b", [1], 0.9, AWAKE, 2.0), ("tp", [0, 1], 0.9, ASLEEP, 0.0))
    assert pool_agent.plan_wake(entries, "tp", 0.05) == ["a", "b"]


def test_plan_wake_counts_a_multi_gpu_victim_once() -> None:
    entries = _entries(("tp", [0, 1], 0.9, AWAKE, 1.0), ("a", [0], 0.9, ASLEEP, 0.0), ("b", [1], 0.9, ASLEEP, 0.0))
    assert pool_agent.plan_wake(entries, "a", 0.05) == ["tp"]


def test_plan_wake_ignores_other_gpus_and_sleeping_models() -> None:
    entries = _entries(("other", [1], 0.9, AWAKE, 1.0), ("asleep", [0], 0.9, ASLEEP, 1.0), ("a", [0], 0.9, ASLEEP, 0.0))
    assert pool_agent.plan_wake(entries, "a", 0.05) == []


def test_plan_sleep_only_idle_awake_models() -> None:
    entries = _entries(
        ("idle", [0], 0.4, AWAKE, 0.0), ("fresh", [0], 0.4, AWAKE, 290.0), ("asleep", [0], 0.4, ASLEEP, 0.0)
    )
    entries["busy"] = pool_agent.Entry(_spec_("busy", [0], 0.1), state=AWAKE, last_used=0.0, in_flight=1)
    assert pool_agent.plan_sleep(entries, now=300.0, sleep_after=300.0) == ["idle"]


# ── state machine with vLLM stubbed ───────────────────────────────────────────


class _FakeProc:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


def _pool(*rows) -> tuple["pool_agent.Pool", list[str]]:
    cfg = pool_agent.PoolConfig.from_dict(
        {"models": [{"served_name": n, "gpus": g, "gpu_fraction": f} for n, g, f, _, _ in rows]}
    )
    pool = pool_agent.Pool(cfg)
    for name, _, _, state, last_used in rows:
        pool.entries[name].state = state
        pool.entries[name].last_used = last_used
        if state != COLD:
            pool.entries[name].proc = _FakeProc()
    calls: list[str] = []

    def spawn(e):
        calls.append(f"spawn {e.name}")
        e.proc = _FakeProc()
        e.state = pool_agent.STARTING

    async def wait_healthy(e):
        calls.append(f"healthy {e.name}")

    async def sleep(e):
        calls.append(f"sleep {e.name}")
        e.state = ASLEEP

    async def wake(e):
        calls.append(f"wake {e.name}")
        e.state = AWAKE

    pool._spawn, pool._wait_healthy, pool._sleep, pool._wake = spawn, wait_healthy, sleep, wake  # type: ignore[method-assign]
    return pool, calls


def test_ensure_awake_wakes_sleeping_model_after_evicting_lru() -> None:
    pool, calls = _pool(("a", [0], 0.45, AWAKE, 1.0), ("b", [0], 0.45, AWAKE, 2.0), ("c", [0], 0.9, ASLEEP, 0.0))
    e = asyncio.run(pool.ensure_awake("c"))
    assert calls == ["sleep a", "sleep b", "wake c"]
    assert e.state == AWAKE and pool.entries["a"].state == ASLEEP


def test_ensure_awake_is_a_noop_for_an_awake_model() -> None:
    pool, calls = _pool(("a", [0], 0.45, AWAKE, 1.0))
    asyncio.run(pool.ensure_awake("a"))
    assert calls == []


def test_ensure_awake_starts_a_cold_model() -> None:
    pool, calls = _pool(("a", [0], 0.9, COLD, 0.0))
    e = asyncio.run(pool.ensure_awake("a"))
    assert calls == ["spawn a", "healthy a"]
    assert e.state == AWAKE and e.last_used > 0


def test_dead_child_is_treated_as_cold_and_restarted() -> None:
    pool, calls = _pool(("a", [0], 0.9, ASLEEP, 0.0))
    pool.entries["a"].proc = _FakeProc(returncode=137)
    asyncio.run(pool.ensure_awake("a"))
    assert calls == ["spawn a", "healthy a"]


def test_boot_leaves_everything_resident_and_asleep() -> None:
    pool, calls = _pool(("a", [0], 0.9, COLD, 0.0), ("b", [0], 0.9, COLD, 0.0))
    asyncio.run(pool.boot())
    assert calls == ["spawn a", "healthy a", "sleep a", "spawn b", "healthy b", "sleep b"]
    assert pool.ready and {e.state for e in pool.entries.values()} == {ASLEEP}


def test_boot_survives_one_failing_model() -> None:
    pool, calls = _pool(("a", [0], 0.9, COLD, 0.0), ("b", [0], 0.9, COLD, 0.0))

    async def wait_healthy(e):
        if e.name == "a":
            raise TimeoutError("no health")
        calls.append(f"healthy {e.name}")

    pool._wait_healthy = wait_healthy  # type: ignore[method-assign]
    asyncio.run(pool.boot())
    assert pool.ready and pool.entries["b"].state == ASLEEP


# ── HTTP front door (no background tasks, no children) ────────────────────────


def test_front_door_health_models_and_unknown_model() -> None:
    from starlette.testclient import TestClient

    pool, _ = _pool(("alice/m", [0], 0.9, ASLEEP, 0.0))
    client = TestClient(pool_agent.build_app(pool, run_background=False))
    assert client.get("/health").status_code == 503  # not booted yet
    pool.ready = True
    assert client.get("/health").json() == {"status": "ok"}
    listing = client.get("/v1/models").json()
    assert listing["data"][0]["id"] == "alice/m" and listing["data"][0]["state"] == ASLEEP
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404 and "alice/m" in r.json()["error"]["message"]
    r = client.post("/v1/chat/completions", content=b"not json")
    assert r.status_code == 404


def test_front_door_reports_unavailable_model_as_503_with_retry_after() -> None:
    from starlette.testclient import TestClient

    pool, _ = _pool(("alice/m", [0], 0.9, ASLEEP, 0.0))

    async def failing_wake(e):
        raise TimeoutError("still draining")

    pool._wake = failing_wake  # type: ignore[method-assign]
    client = TestClient(pool_agent.build_app(pool, run_background=False))
    r = client.post("/v1/chat/completions", json={"model": "alice/m", "messages": []})
    assert r.status_code == 503 and r.headers["Retry-After"] == "30"
