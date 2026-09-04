# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""Render the HPI example scripts (hpi/examples) through the production parser
and check the result is valid bash with the shape hpi/README.md promises."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from swiss_ai_model_launch.cli.main import build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.framework import render_master, render_rank_scripts
from swiss_ai_model_launch.launchers.utils import render_sbatch_header
from tests.unit.test_examples import _parse_sml_advanced_script

_HAS_SHELLCHECK = shutil.which("shellcheck") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = sorted(str(p.relative_to(_REPO_ROOT)) for p in (_REPO_ROOT / "hpi" / "examples").glob("*.sh"))


def _render(example_path: str) -> dict[str, str]:
    # The scripts resolve $HPI from their own location; the parser sees plain text.
    content = (_REPO_ROOT / example_path).read_text().replace("$HPI", str(_REPO_ROOT / "hpi"))
    args = _parse_sml_advanced_script(content)
    launch_args = build_launch_args_from_advanced(args, username="alice", account="aisc-staff", partition="aisc-batch")
    out = {"master.sh": render_sbatch_header(launch_args) + render_master(launch_args)}
    out.update(render_rank_scripts(launch_args))
    return out


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
