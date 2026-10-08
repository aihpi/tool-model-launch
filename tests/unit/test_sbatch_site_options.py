"""Site-specific #SBATCH knobs: --gres, --cpus-per-task, --mem, --no-exclusive,
--sbatch-arg. Defaults must reproduce the historical exclusive-node header."""

from typing import Any

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.utils import render_sbatch_header


def _make_args(**overrides: Any) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="vendor/model-abc1",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="vllm",
    )
    return LaunchArgs(**{**defaults, **overrides})


def test_defaults_keep_exclusive_and_no_resource_requests() -> None:
    sbatch = _make_args().to_sbatch_args()
    assert "--exclusive" in sbatch
    assert not any(a.startswith(("--gres", "--cpus-per-task", "--mem")) for a in sbatch)


def test_shared_node_options_rendered() -> None:
    sbatch = _make_args(
        exclusive=False,
        gres="gpu:1",
        cpus_per_task=8,
        mem="48G",
        sbatch_args=["--exclude=ga03", "--constraint=h100"],
    ).to_sbatch_args()
    assert "--exclusive" not in sbatch
    assert "--gres=gpu:1" in sbatch
    assert "--cpus-per-task=8" in sbatch
    assert "--mem=48G" in sbatch
    # passthrough is verbatim and last
    assert sbatch[-2:] == ["--exclude=ga03", "--constraint=h100"]


def test_header_mirrors_to_sbatch_args() -> None:
    args = _make_args(exclusive=False, gres="gpu:2", sbatch_args=["--qos=low"])
    header = render_sbatch_header(args, reservation="res1")
    expected = [f"#SBATCH {a}" for a in args.to_sbatch_args(reservation="res1")]
    assert header.splitlines()[1:] == expected


def test_cli_flags_reach_launch_args() -> None:
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "advanced",
            "--system",
            "clariden",
            "--partition",
            "normal",
            "--framework",
            "vllm",
            "--environment",
            "/path/to/env.toml",
            "--framework-args",
            "--served-model-name vendor/model-abc",
            "--gres",
            "gpu:1",
            "--cpus-per-task",
            "8",
            "--mem",
            "48G",
            "--no-exclusive",
            "--sbatch-arg=--exclude=ga03",
            "--sbatch-arg=--qos=low",
        ]
    )
    la = build_launch_args_from_advanced(ns, username="alice", account="proj01", partition="normal")
    assert (la.exclusive, la.gres, la.cpus_per_task, la.mem) == (False, "gpu:1", 8, "48G")
    assert la.sbatch_args == ["--exclude=ga03", "--qos=low"]


def test_cli_defaults_are_upstream_behaviour() -> None:
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "advanced",
            "--system",
            "clariden",
            "--partition",
            "normal",
            "--framework",
            "vllm",
            "--environment",
            "/path/to/env.toml",
            "--framework-args",
            "--served-model-name vendor/model-abc",
        ]
    )
    la = build_launch_args_from_advanced(ns, username="alice", account="proj01", partition="normal")
    assert la.exclusive is True
    assert (la.gres, la.cpus_per_task, la.mem, la.sbatch_args) == (None, None, None, [])
