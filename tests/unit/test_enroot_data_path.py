# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""--enroot-data-path: master.sh redirects stock pyxis' rootfs directory and prunes
rootfs of jobs Slurm no longer knows. The block is exercised with real bash."""

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.framework import render_master
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs

_PATH = "/shared/enroot-data/$USER"


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


def test_default_leaves_pyxis_alone() -> None:
    assert "sml_enroot" not in render_master(_make_args())


def test_block_rendered_before_the_first_srun_with_user_unexpanded() -> None:
    master = render_master(_make_args(enroot_data_path=_PATH))
    assert f'sml_enroot_data="{_PATH}"' in master
    assert master.index("sml_prune_enroot") < master.index("srun --nodes=1")


def _block(master: str) -> str:
    start = master.index("# Stock pyxis unpacks")
    end = master.index("\n\n", start)
    return master[start:end]


def _run(tmp_path: Path, known_jobs: set[str], home_kind: str, job_id: str = "999") -> tuple[Path, Path, str]:
    """Run the rendered block under a fake $HOME with a fake scontrol.

    home_kind: "missing" | "symlink" | "dir-stale" (holds only a finished job's rootfs)
    | "dir-live" (holds a running job's rootfs).
    """
    home = tmp_path / "home"
    data = tmp_path / "data"
    (home / ".local" / "share").mkdir(parents=True)
    link = home / ".local" / "share" / "enroot"
    if home_kind == "symlink":
        data.mkdir()
        link.symlink_to(data)
    elif home_kind.startswith("dir"):
        (link / ("pyxis_222.0" if home_kind == "dir-live" else "pyxis_111.0")).mkdir(parents=True)
    data.mkdir(exist_ok=True)
    for name in ("pyxis_333.0", "pyxis_222.1", f"pyxis_{job_id}.0", "pyxis_named", "other"):
        (data / name).mkdir(exist_ok=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scontrol = fake_bin / "scontrol"
    scontrol.write_text('#!/bin/bash\ncase " $KNOWN " in *" $3 "*) exit 0 ;; *) exit 1 ;; esac\n')
    scontrol.chmod(0o755)
    master = render_master(_make_args(enroot_data_path=str(data)))
    proc = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _block(master)],
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(home),
            "SLURM_JOB_ID": job_id,
            "KNOWN": " ".join(sorted(known_jobs)),
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return link, data, proc.stdout + proc.stderr


@pytest.mark.parametrize("home_kind", ["missing", "symlink", "dir-stale"])
def test_home_dir_becomes_a_symlink_and_stale_rootfs_are_pruned(tmp_path: Path, home_kind: str) -> None:
    link, data, out = _run(tmp_path, known_jobs={"222"}, home_kind=home_kind)
    assert link.is_symlink() and os.path.realpath(link) == os.path.realpath(data)
    remaining = sorted(p.name for p in data.iterdir())
    # 333 is gone (unknown job); 222 is running; 999 is this job; non-numeric names are not ours
    assert remaining == ["other", "pyxis_222.1", "pyxis_999.0", "pyxis_named"]
    assert "pyxis_333.0" in out
    if home_kind == "dir-stale":
        assert "pyxis_111.0" in out  # pruned before the directory was replaced


def test_home_dir_with_a_live_rootfs_is_left_alone_with_a_warning(tmp_path: Path) -> None:
    link, data, out = _run(tmp_path, known_jobs={"222"}, home_kind="dir-live")
    assert link.is_dir() and not link.is_symlink()
    assert (link / "pyxis_222.0").is_dir()
    assert "warning" in out and "not empty" in out
    assert not (data / "pyxis_333.0").exists()  # pruning of the target still happened


def test_cli_flag_and_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    base = [
        "advanced", "--system", "clariden", "--partition", "normal", "--framework", "vllm",
        "--environment", "/path/to/env.toml", "--framework-args", "--served-model-name vendor/model-abc",
    ]  # fmt: skip
    monkeypatch.delenv("SML_ENROOT_DATA_PATH", raising=False)
    la = build_launch_args_from_advanced(_build_parser().parse_args(base), username="a", account="p", partition="n")
    assert la.enroot_data_path is None
    la = build_launch_args_from_advanced(
        _build_parser().parse_args([*base, "--enroot-data-path", _PATH]), username="a", account="p", partition="n"
    )
    assert la.enroot_data_path == _PATH
    monkeypatch.setenv("SML_ENROOT_DATA_PATH", "/elsewhere/$USER")
    la = build_launch_args_from_advanced(_build_parser().parse_args(base), username="a", account="p", partition="n")
    assert la.enroot_data_path == "/elsewhere/$USER"
