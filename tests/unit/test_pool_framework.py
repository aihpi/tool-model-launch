# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""`--framework pool`: the agent is materialised with the rank scripts and is the
framework process; single node, no router."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from swiss_ai_model_launch.launchers.framework import render_all, render_rank_scripts
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology

_HAS_SHELLCHECK = shutil.which("shellcheck") is not None


def _make_args(**overrides: Any) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="alice/pool",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="pool",
        framework_args="--config ~/hpi/pool.toml",
        opentela_service_name="pool",
    )
    return LaunchArgs(**{**defaults, **overrides})


def test_pool_renders_head_and_agent() -> None:
    scripts = render_rank_scripts(_make_args())
    assert set(scripts) == {"head.sh", "pool_agent.py"}
    assert scripts["pool_agent.py"].startswith("#!/usr/bin/env python3")
    head = scripts["head.sh"]
    assert "export VLLM_SERVER_DEV_MODE=1" in head
    assert "python3 $HOME/.sml/job-${SLURM_JOB_ID}/pool_agent.py --port 8080 --config ~/hpi/pool.toml" in head
    assert "--service.name pool" in head
    assert "--service.port 8080" in head


def test_master_materialises_the_agent_under_ranks_dir() -> None:
    master = render_all(_make_args())["master.sh"]
    assert "cat > \"$RANKS_DIR/pool_agent.py\" <<'__SML_POOL_AGENT_EOF__'" in master
    assert master.count("__SML_POOL_AGENT_EOF__") == 2


def test_pool_with_auto_port_and_no_opentela() -> None:
    head = render_rank_scripts(_make_args(framework_port="auto", disable_opentela=True))["head.sh"]
    assert "pool_agent.py --port $FRAMEWORK_PORT --config" in head
    assert "$OPENTELA_BIN" not in head


def test_pool_rejects_multi_node_and_router() -> None:
    with pytest.raises(ValueError, match="nodes_per_replica"):
        _make_args(topology=Topology(nodes_per_replica=2))
    with pytest.raises(ValueError, match="router"):
        _make_args(topology=Topology(replicas=2), router="sglang")


def test_pool_replicas_are_independent_pools() -> None:
    master = render_all(_make_args(topology=Topology(replicas=2)))["master.sh"]
    assert master.count('bash "$RANKS_DIR/head.sh"') == 2


@pytest.mark.skipif(not _HAS_SHELLCHECK, reason="shellcheck not installed")
def test_pool_scripts_pass_shellcheck(tmp_path: Path) -> None:
    out = render_all(_make_args(framework_port="auto"))
    for filename in ("master.sh", "head.sh"):
        path = tmp_path / filename
        path.write_text(("#!/bin/bash\n" if filename == "master.sh" else "") + out[filename])
        r = subprocess.run(["shellcheck", "-S", "warning", str(path)], capture_output=True)
        assert r.returncode == 0, f"shellcheck failed for {filename}:\n{r.stdout.decode()}"
