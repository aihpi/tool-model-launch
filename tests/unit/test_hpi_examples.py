# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""Render the HPI example scripts (hpi/examples) through the production parser
and check the result is valid bash with the shape hpi/README.md promises."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_recipes import render_hpi

_HAS_SHELLCHECK = shutil.which("shellcheck") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = sorted(str(p.relative_to(_REPO_ROOT)) for p in (_REPO_ROOT / "hpi" / "examples").glob("*.sh"))


_RECIPE_REF = re.compile(r'--recipe "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/\.\./recipes/([^"]+)"')


def _recipe_of(example_path: str) -> Path:
    match = _RECIPE_REF.search((_REPO_ROOT / example_path).read_text())
    assert match, f"{example_path} does not call sml advanced --recipe <hpi/recipes/...>"
    return _REPO_ROOT / "hpi" / "recipes" / match.group(1)


def _render(example_path: str) -> dict[str, str]:
    return render_hpi(["--recipe", str(_recipe_of(example_path))])


@pytest.mark.parametrize("example_path", _EXAMPLES, ids=lambda p: Path(p).stem)
def test_hpi_example_runs_sml_with_its_recipe(tmp_path: Path, example_path: str) -> None:
    """Run the wrapper for real with a fake `sml` that echoes its argv: the recipe
    path must resolve to an existing file and extra flags must pass through."""
    fake = tmp_path / "sml"
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    proc = subprocess.run(
        ["bash", str(_REPO_ROOT / example_path), "--mem", "8G"],
        cwd=cwd,
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    argv = proc.stdout.splitlines()
    assert argv[:2] == ["advanced", "--recipe"]
    assert Path(argv[2]).resolve() == _recipe_of(example_path).resolve()
    assert argv[3:] == ["--mem", "8G"]


def test_hpi_examples_exist() -> None:
    assert "hpi/examples/qwen3-0.6b-vllm.sh" in _EXAMPLES


@pytest.mark.parametrize("example_path", _EXAMPLES, ids=lambda p: Path(p).stem)
def test_hpi_example_has_the_reference_job_shape(example_path: str) -> None:
    out = _render(example_path)
    master, head = out["master.sh"], out["head.sh"]
    assert "#SBATCH --gres=gpu:" in master
    assert "#SBATCH --exclusive" not in master
    assert "#SBATCH --exclude=ga03" in master
    assert "FRAMEWORK_PORT=$((20000 + SLURM_JOB_ID % 10000))" in head
    assert head.index('"$WSTUNNEL_BIN" client') < head.index("$OPENTELA_BIN start")
    assert "--bootstrap.static" in head and "--config-dir" in head and "--seed" in head
    assert "--service.port $FRAMEWORK_PORT" in head and "--port $FRAMEWORK_PORT" in head
    assert 'sml_enroot_data="/sc/projects/sci-aisc/aisc-share/enroot-data/$USER"' in master
    for content in out.values():
        assert "capstor" not in content and "cscs" not in content


@pytest.mark.parametrize("example_path", _EXAMPLES, ids=lambda p: Path(p).stem)
def test_hpi_example_renders_valid_bash(tmp_path: Path, example_path: str) -> None:
    for filename, content in _render(example_path).items():
        path = tmp_path / filename
        path.write_text(content)
        if filename.endswith(".py"):  # the pool agent rides along with the rank scripts
            result = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True)
            assert result.returncode == 0, f"py_compile failed for {filename}:\n{result.stderr.decode()}"
            continue
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True)
        assert result.returncode == 0, f"bash -n failed for {filename}:\n{result.stderr.decode()}"
        if _HAS_SHELLCHECK:
            result = subprocess.run(["shellcheck", "-S", "warning", str(path)], capture_output=True)
            assert result.returncode == 0, f"shellcheck failed for {filename}:\n{result.stdout.decode()}"


def test_hpi_env_carries_no_secrets_and_no_cscs_paths() -> None:
    env = (_REPO_ROOT / "hpi" / "sml.env").read_text()
    toml = (_REPO_ROOT / "hpi" / "envs" / "vllm_hpi.toml").read_text()
    for text in (env, toml):
        assert "capstor" not in text and "cscs" not in text.lower()
    assert "SML_OPENTELA_BOOTSTRAP_ADDR=Qm" in env
    # the token file is referenced in comments only, never as a value
    assert all("otela-tunnel-token" not in part for part in env.split("SML_")[1:])
