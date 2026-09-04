"""``container_spec``: how the env toml reaches srun. "edf" (default) must render
exactly what upstream renders; "pyxis" translates the toml into stock pyxis flags
for sites whose pyxis has no ``--environment`` (EDF) support."""

from pathlib import Path
from typing import Any

import pytest

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.framework import render_master
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs

_TOML = """
image = "/share/images/vllm-{arch}.sqsh"
mounts = [
  "/share/otela-share:/opentelabin",
  "/sc/home",
]
workdir = "/opt"

[env]
HF_HOME = "/share/hf-cache"
NCCL_NET = "AWS Libfabric"

[annotations]
com.hooks.cxi.enabled = "true"
"""


def _make_args(env_file: Path, **overrides: Any) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="alice/vendor/model",
        account="proj01",
        partition="normal",
        environment=str(env_file),
        framework="vllm",
        framework_args="--served-model-name alice/vendor/model",
    )
    return LaunchArgs(**{**defaults, **overrides})


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "env.toml"
    path.write_text(_TOML)
    return path


def test_default_is_edf_and_renders_the_upstream_srun(env_file: Path) -> None:
    master = render_master(_make_args(env_file))
    assert '--environment="${SML_ENV_FILE}"' in master
    assert '--container-mounts="$RANKS_DIR:$RANKS_DIR"' in master
    assert "SML_CONTAINER_ARGS" not in master
    assert "--container-image" not in master


def test_pyxis_translates_the_toml_into_container_flags(env_file: Path) -> None:
    master = render_master(_make_args(env_file, container_spec="pyxis"))
    assert "--environment=" not in master
    assert '--container-image="/share/images/vllm-${SML_ARCH}.sqsh"' in master
    assert '--container-mounts="/share/otela-share:/opentelabin,/sc/home:/sc/home,$RANKS_DIR:$RANKS_DIR"' in master
    assert '--container-workdir="/opt"' in master
    assert "--no-container-entrypoint" in master
    assert "export HF_HOME=/share/hf-cache" in master
    assert "export NCCL_NET='AWS Libfabric'" in master
    assert '--container-env="HF_HOME,NCCL_NET"' in master
    assert '"${SML_CONTAINER_ARGS[@]}" \\' in master
    assert "com.hooks" not in master
    # The array is defined after arch detection (it references $SML_ARCH) and
    # before the first srun that expands it.
    assert (
        master.index("SML_ARCH=amd64") < master.index("SML_CONTAINER_ARGS=(") < master.index("${SML_CONTAINER_ARGS[@]}")
    )


def test_pyxis_honours_an_explicit_entrypoint(tmp_path: Path) -> None:
    env_file = tmp_path / "env.toml"
    env_file.write_text('image = "/share/img.sqsh"\nentrypoint = true\n')
    master = render_master(_make_args(env_file, container_spec="pyxis"))
    assert "    --container-entrypoint\n" in master
    assert "--no-container-entrypoint" not in master


def test_pyxis_needs_the_toml_at_render_time(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="container-spec pyxis"):
        render_master(_make_args(tmp_path / "missing.toml", container_spec="pyxis"))


def test_pyxis_router_srun_uses_the_same_flags(env_file: Path) -> None:
    from swiss_ai_model_launch.launchers.topology import Topology

    master = render_master(
        _make_args(env_file, container_spec="pyxis", framework="sglang", router="sglang", topology=Topology(replicas=2))
    )
    # Every containerised srun (heads and the router) expands the same array.
    assert master.count('"${SML_CONTAINER_ARGS[@]}" \\') == master.count("--container-writable") == 3


def test_cli_flag_and_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _build_parser()
    base = ["advanced", "--framework", "vllm", "--environment", "env.toml", "--served-model-name", "vendor/model"]
    args = parser.parse_args([*base, "--container-spec", "pyxis"])
    assert build_launch_args_from_advanced(args, username="alice", account="a", partition="p").container_spec == "pyxis"
    args = parser.parse_args(base)
    assert build_launch_args_from_advanced(args, username="alice", account="a", partition="p").container_spec == "edf"
    monkeypatch.setenv("SML_CONTAINER_SPEC", "pyxis")
    args = _build_parser().parse_args(base)
    assert build_launch_args_from_advanced(args, username="alice", account="a", partition="p").container_spec == "pyxis"
