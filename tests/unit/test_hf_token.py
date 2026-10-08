# ruff: noqa: S105, S106, S603, S607  # fake test token; subprocess against controlled paths
"""Hugging Face token for gated models: --hf-token-file renders an HF_TOKEN export
that never reaches the xtrace log, a missing file is a no-op, and `sml init`
writes the file mode 600 only on sites that set SML_HF_TOKEN_FILE."""

import asyncio
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from swiss_ai_model_launch.cli import hf_token
from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.framework import render_all
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs

_SECRET = "hf_s3cr3tT0ken"


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


def _run_setup(head: str, home: Path) -> subprocess.CompletedProcess[str]:
    setup = head[: head.index("vllm serve") if "$OPENTELA_BIN" not in head else head.index("$OPENTELA_BIN start")]
    # Probe with tracing off, or the probe itself would trace the value.
    probe = '_x=$-\n{ set +x; } 2>/dev/null\necho "xtrace=$_x"\necho "HF_TOKEN=${HF_TOKEN:-<unset>}"\n'
    script = setup + "\n" + probe
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", "HOME": str(home), "SLURM_JOB_ID": "2724141"},
        capture_output=True,
        text=True,
    )


def test_default_renders_no_token_handling() -> None:
    for content in render_all(_make_args()).values():
        assert "HF_TOKEN" not in content


def test_token_is_exported_but_never_traced(tmp_path: Path) -> None:
    token_file = tmp_path / "hf-token"
    token_file.write_text(_SECRET + "\n")
    head = render_all(_make_args(hf_token_file=str(token_file)))["head.sh"]
    assert _SECRET not in head
    proc = _run_setup(head, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert f"HF_TOKEN={_SECRET}" in proc.stdout  # exported, trailing newline stripped
    assert _SECRET not in proc.stderr  # the xtrace log
    assert "[[ -r" in proc.stderr  # tracing was on around it
    assert "x" in proc.stdout.split("xtrace=")[1].split()[0]  # and is restored


def test_missing_token_file_is_a_no_op(tmp_path: Path) -> None:
    head = render_all(_make_args(hf_token_file=str(tmp_path / "absent")))["head.sh"]
    proc = _run_setup(head, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "HF_TOKEN=<unset>" in proc.stdout


def test_tilde_means_the_job_users_home(tmp_path: Path) -> None:
    (tmp_path / ".sml").mkdir()
    (tmp_path / ".sml" / "hf token").write_text(_SECRET)
    head = render_all(_make_args(hf_token_file="~/.sml/hf token"))["head.sh"]
    assert "\"$HOME\"/'.sml/hf token'" in head
    proc = _run_setup(head, tmp_path)
    assert f"HF_TOKEN={_SECRET}" in proc.stdout, proc.stderr


def test_cli_flag_defaults_to_the_site_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ["advanced", "--framework", "vllm", "--environment", "e.toml", "--served-model-name", "vendor/m"]
    monkeypatch.delenv("SML_HF_TOKEN_FILE", raising=False)
    args = _build_parser().parse_args(base)
    assert build_launch_args_from_advanced(args, username="alice", account="a", partition="p").hf_token_file is None
    monkeypatch.setenv("SML_HF_TOKEN_FILE", "~/.sml/hf-token")
    args = _build_parser().parse_args(base)
    launch = build_launch_args_from_advanced(args, username="alice", account="a", partition="p")
    assert launch.hf_token_file == "~/.sml/hf-token"
    args = _build_parser().parse_args([*base, "--hf-token-file", "/other"])
    assert build_launch_args_from_advanced(args, username="alice", account="a", partition="p").hf_token_file == "/other"


# ── the `sml init` step ───────────────────────────────────────────────────────


def test_write_token_is_mode_600_even_over_a_readable_file(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "hf-token"
    path.parent.mkdir()
    path.write_text("old")
    path.chmod(0o644)
    hf_token.write_token(path, f"  {_SECRET}\n")
    assert path.read_text() == _SECRET + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_init_step_is_inert_without_the_site_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SML_HF_TOKEN_FILE", raising=False)
    monkeypatch.setenv("SML_HF_TOKEN", _SECRET)
    monkeypatch.setenv("HOME", str(tmp_path))
    asyncio.run(hf_token.configure_from_env())
    assert list(tmp_path.iterdir()) == []


def test_init_step_writes_the_token_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SML_HF_TOKEN_FILE", "~/.sml/hf-token")
    monkeypatch.setenv("SML_HF_TOKEN", _SECRET)
    asyncio.run(hf_token.configure_from_env())
    written = tmp_path / ".sml" / "hf-token"
    assert written.read_text() == _SECRET + "\n"
    assert stat.S_IMODE(written.stat().st_mode) == 0o600


def test_init_step_without_a_token_keeps_the_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "hf-token"
    existing.write_text("kept\n")
    monkeypatch.setenv("SML_HF_TOKEN_FILE", str(existing))
    monkeypatch.delenv("SML_HF_TOKEN", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    asyncio.run(hf_token.configure_from_env())
    assert existing.read_text() == "kept\n"
    assert "Keeping the existing" in capsys.readouterr().out
