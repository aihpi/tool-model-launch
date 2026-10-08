# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""master.sh ends with `wait -n`, which returns when *any* background job of the
shell finishes. Only the critical srun steps may be visible to it: every other
background command has to be disowned, or its exit ends the job. A background
prune that was not disowned once ended every job two seconds after start."""

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from swiss_ai_model_launch.launchers.framework import render_master
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology

_HEREDOC_START = re.compile(r"<<'(__SML_\w+_EOF__)'")


def _make_args(**overrides: Any) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="alice/vendor/model",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="vllm",
        framework_args="--served-model-name alice/vendor/model",
    )
    return LaunchArgs(**{**defaults, **overrides})


_SHAPES = {
    "default": {},
    "hpi-site": dict(
        enroot_data_path="/share/enroot-data/$USER",
        disable_metrics=True,
        disable_dcgm_exporter=True,
        framework_port="auto",
    ),
    "metrics": dict(enroot_data_path="/share/enroot-data/$USER"),
    "sglang-router": dict(
        framework="sglang", router="sglang", topology=Topology(replicas=2), enroot_data_path="/share/e"
    ),
    "multinode": dict(topology=Topology(replicas=1, nodes_per_replica=2), enroot_data_path="/share/e"),
}


def _top_level_lines(master: str) -> list[str]:
    """master.sh without the bodies of the embedded rank scripts / helpers
    (heredocs) and of quoted scripts run on other nodes (``bash -c "..."``);
    the closing line of such a quoted script (``" &``) is kept."""
    lines, end_marker, in_quoted = [], None, False
    for line in master.splitlines():
        if end_marker:
            if line.strip() == end_marker:
                end_marker = None
            continue
        if in_quoted:
            if line.lstrip().startswith('"'):
                in_quoted = False
                lines.append(line)
            continue
        match = _HEREDOC_START.search(line)
        if match:
            end_marker = match.group(1)
        if line.rstrip().endswith('bash -c "'):
            in_quoted = True
        lines.append(line)
    return lines


@pytest.mark.parametrize("shape", sorted(_SHAPES), ids=str)
def test_every_background_command_is_critical_or_disowned(shape: str) -> None:
    lines = _top_level_lines(render_master(_make_args(**_SHAPES[shape])))
    found = 0
    for i, line in enumerate(lines):
        stripped = line.split("#")[0].rstrip() if not line.lstrip().startswith("#") else ""
        if not stripped.endswith("&") or stripped.endswith("&&"):
            continue
        found += 1
        following = "\n".join(lines[i + 1 : i + 4])
        assert "critical_pids+=($!)" in following or "disown" in following, (
            f"background command not tracked or disowned in shape {shape!r}:\n{line}\n{following}"
        )
    assert found >= 1


def test_full_master_waits_for_the_critical_step(tmp_path: Path) -> None:
    """Replay the whole rendered master.sh with fake srun/scontrol/python3: it has
    to end with the critical step's exit code, after that step ends. The bug
    this guards against ended every job with code 0 within a second."""
    image = tmp_path / "image.sqsh"
    image.touch()
    env_toml = tmp_path / "env.toml"
    env_toml.write_text(f'image = "{image}"\nmounts = []\n')
    master = render_master(
        _make_args(
            environment=str(env_toml),
            container_spec="pyxis",
            enroot_data_path=str(tmp_path / "enroot-data"),
            disable_metrics=True,
            disable_dcgm_exporter=True,
            framework_port="auto",
        )
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fakes = {
        # `srun ... hostname -i` answers an IP; the critical step runs 2 s and exits 7.
        "srun": 'for a in "$@"; do [[ "$a" == hostname ]] && { echo 10.0.0.1; exit 0; }; done\nsleep 2\nexit 7',
        "scontrol": '[[ "$1 $2" == "show hostnames" ]] && { echo node0; exit 0; }\nexit 1',
        "python3": "sleep 30",  # the replica health checker
    }
    for name, body in fakes.items():
        (fake_bin / name).write_text(f"#!/bin/bash\n{body}\n")
        (fake_bin / name).chmod(0o755)
    (tmp_path / "home").mkdir()
    work = tmp_path / "work"
    work.mkdir()
    # Run it as a file, like sbatch does: under `bash -c` bash reaps the
    # background prune earlier and the bug does not show.
    script = work / "master.sh"
    script.write_text("#!/bin/bash\n" + master)
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=work,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "USER": "alice",
            "SLURM_JOB_ID": "4242",
            "SLURM_NODELIST": "node0",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "with code 7" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 7
