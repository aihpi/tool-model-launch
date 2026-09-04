# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""Site-portability options rendered into the scripts: per-job framework port
("auto"), deterministic OpenTela identity + private mesh, the wstunnel to a
non-routable bootstrap peer, a custom OpenTela service name, and the SML_*
environment overrides. Defaults must render exactly what upstream renders."""

import importlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers import launch_args as launch_args_module
from swiss_ai_model_launch.launchers.framework import (
    OPENTELA_BOOTSTRAP_ADDR,
    render_all,
    render_master,
    render_rank_scripts,
)
from swiss_ai_model_launch.launchers.launch_args import (
    FRAMEWORK_PORT_AUTO_EXPR,
    TELEMETRY_ENDPOINT,
    LaunchArgs,
    telemetry_endpoint,
)
from swiss_ai_model_launch.launchers.topology import Topology

_HAS_SHELLCHECK = shutil.which("shellcheck") is not None
_PEER_ID = "QmWBnedcUEawmdTQTXgyY6BAQFDMwiUDndmRqgGkRBY4Qr"
_TUNNEL = dict(
    tunnel_url="wss://gateway.example.org:443",
    tunnel_token_file="~/otela-tunnel-token",  # noqa: S106  (a path, not a secret)
    tunnel_target="otela-head.svc.cluster.local:43905",
)


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


# ── framework port ────────────────────────────────────────────────────────────


def test_fixed_port_renders_literals_and_no_port_variable() -> None:
    out = render_all(_make_args())
    assert "--port 8080" in out["head.sh"]
    assert "--service.port 8080" in out["head.sh"]
    assert "\nFRAMEWORK_PORT=" not in out["head.sh"]
    assert "\nFRAMEWORK_PORT=" not in out["master.sh"]


def test_auto_port_defined_once_per_script_and_used_everywhere() -> None:
    out = render_all(
        _make_args(
            framework_port="auto",
            topology=Topology(replicas=2),
            router="sglang",
            telemetry_endpoint="https://telemetry.example.com/jobs",
        )
    )
    definition = f"FRAMEWORK_PORT={FRAMEWORK_PORT_AUTO_EXPR}"
    rank_scripts = {k: v for k, v in out.items() if k != "master.sh"}
    for name, content in rank_scripts.items():
        assert content.count(definition) == 1, name
    # master.sh defines it once for itself, plus the embedded rank-script heredocs.
    assert out["master.sh"].count(definition) == 1 + len(rank_scripts)
    head = out["head.sh"]
    assert "--port $FRAMEWORK_PORT" in head
    # The definition precedes its first use.
    assert head.index(definition) < head.index("--port $FRAMEWORK_PORT")
    # shlex-quoted label re-opens quoting around the expansion.
    assert "--label 'framework_args=--port '\"$FRAMEWORK_PORT\"' --served-model-name" in head
    router = out["router.sh"]
    assert "http://$ip:$FRAMEWORK_PORT/health" in router
    assert 'worker_urls="$worker_urls http://$ip:$FRAMEWORK_PORT"' in router
    master = out["master.sh"]
    assert "SML_HEALTH_FRAMEWORK_PORT=$FRAMEWORK_PORT" in master
    assert '"framework_port": \'"$FRAMEWORK_PORT"\'' in master
    assert "http://$replica_0_head_ip:$FRAMEWORK_PORT" in master


def test_auto_port_is_the_opentela_service_port_without_router() -> None:
    head = render_rank_scripts(_make_args(framework_port="auto"))["head.sh"]
    assert "--service.port $FRAMEWORK_PORT" in head


def test_explicit_numeric_port() -> None:
    head = render_rank_scripts(_make_args(framework_port=9000))["head.sh"]
    assert "--port 9000" in head
    assert "--service.port 9000" in head


# ── OpenTela identity / private mesh ──────────────────────────────────────────


def test_every_opentela_peer_gets_static_bootstrap_config_dir_and_seed() -> None:
    scripts = render_rank_scripts(_make_args(topology=Topology(replicas=2, nodes_per_replica=2), router="sglang"))
    for name in ("head.sh", "follower.sh", "router.sh"):
        s = scripts[name]
        assert f'--bootstrap.static "{OPENTELA_BOOTSTRAP_ADDR}"' in s, name
        assert '--config-dir "$HOME/.sml/job-${SLURM_JOB_ID}/otela-step-${SLURM_STEP_ID:-0}"' in s, name
        assert "--seed $((SLURM_JOB_ID * 1000 + ${SLURM_STEP_ID:-0}))" in s, name


def test_identity_flags_absent_when_opentela_disabled() -> None:
    head = render_rank_scripts(_make_args(disable_opentela=True))["head.sh"]
    assert "--bootstrap.static" not in head
    assert "--seed" not in head


# ── tunnel ────────────────────────────────────────────────────────────────────


def test_tunnel_runs_before_opentela_and_bare_peer_id_points_at_tunnel() -> None:
    out = render_all(_make_args(opentela_bootstrap_addr=_PEER_ID, **_TUNNEL))
    head = out["head.sh"]
    assert "TUN=$((30000 + SLURM_JOB_ID % 10000))" in head
    assert f'"$WSTUNNEL_BIN" client -L "tcp://127.0.0.1:$TUN:{_TUNNEL["tunnel_target"]}"' in head
    assert '--http-upgrade-path-prefix "otela-$(cat ~/otela-tunnel-token)"' in head
    assert head.index('"$WSTUNNEL_BIN" client') < head.index("$OPENTELA_BIN start")
    expected = f'"/ip4/127.0.0.1/tcp/$TUN/p2p/{_PEER_ID}"'
    assert f"--bootstrap.addr {expected}" in head
    assert f"--bootstrap.static {expected}" in head
    # the binary is resolved next to OPENTELA_BIN, per arch
    assert "export WSTUNNEL_BIN=/opentelabin/prod/wstunnel-amd64" in out["master.sh"]
    assert "export WSTUNNEL_BIN=/opentelabin/prod/wstunnel-arm64" in out["master.sh"]


def test_tunnel_keeps_a_full_multiaddr_untouched() -> None:
    addr = f"/ip4/10.0.0.1/tcp/43905/p2p/{_PEER_ID}"
    head = render_rank_scripts(_make_args(opentela_bootstrap_addr=addr, **_TUNNEL))["head.sh"]
    assert f'--bootstrap.addr "{addr}"' in head


def test_no_tunnel_by_default() -> None:
    out = render_all(_make_args())
    assert "WSTUNNEL" not in out["master.sh"]
    assert "wstunnel" not in out["head.sh"].lower()


def test_tunnel_options_must_come_together() -> None:
    with pytest.raises(ValueError, match="together"):
        _make_args(tunnel_url="wss://gateway.example.org:443")


def test_token_value_never_rendered() -> None:
    # Only the file path is rendered; the job reads it at run time.
    out = render_all(_make_args(**_TUNNEL))
    for content in out.values():
        assert "otela-tunnel-token" not in content or "$(cat ~/otela-tunnel-token)" in content


# ── service name ──────────────────────────────────────────────────────────────


def test_custom_service_name_in_wrap_and_telemetry() -> None:
    out = render_all(_make_args(opentela_service_name="pool", telemetry_endpoint="https://t.example.com/x"))
    assert "--service.name pool" in out["head.sh"]
    assert '"ocf_service_name": "pool"' in out["master.sh"]


def test_default_service_name_is_llm() -> None:
    assert "--service.name llm" in render_rank_scripts(_make_args())["head.sh"]


# ── the full shared-node + tunnel shape still lints ───────────────────────────


@pytest.mark.skipif(not _HAS_SHELLCHECK, reason="shellcheck not installed")
@pytest.mark.parametrize("replicas,router", [(1, "opentela"), (2, "sglang")])
def test_site_shape_passes_shellcheck(tmp_path: Path, replicas: int, router: str) -> None:
    args = _make_args(
        framework="sglang",
        framework_port="auto",
        opentela_bootstrap_addr=_PEER_ID,
        topology=Topology(replicas=replicas),
        router=router,
        telemetry_endpoint="https://telemetry.example.com/jobs",
        **_TUNNEL,
    )
    master_path = tmp_path / "master.sh"
    master_path.write_text("#!/bin/bash\n" + render_master(args))
    result = subprocess.run(["shellcheck", "-S", "warning", str(master_path)], capture_output=True)
    assert result.returncode == 0, f"shellcheck failed for master.sh:\n{result.stdout.decode()}"
    for filename, content in render_rank_scripts(args).items():
        path = tmp_path / filename
        path.write_text(content)
        r = subprocess.run(["shellcheck", "-S", "warning", str(path)], capture_output=True)
        assert r.returncode == 0, f"shellcheck failed for {filename}:\n{r.stdout.decode()}"


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse(*extra: str):
    parser = _build_parser()
    return parser.parse_args(
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
            *extra,
        ]
    )


def _build(ns) -> LaunchArgs:
    return build_launch_args_from_advanced(ns, username="alice", account="proj01", partition="normal")


def test_cli_port_auto_and_numeric_and_invalid() -> None:
    assert _build(_parse("--framework-port", "auto")).framework_port == "auto"
    assert _build(_parse("--framework-port", "9000")).framework_port == 9000
    assert _build(_parse()).framework_port == 8080
    with pytest.raises(SystemExit):
        _parse("--framework-port", "nine")


def test_cli_tunnel_and_service_name_reach_launch_args() -> None:
    la = _build(
        _parse(
            "--tunnel-url",
            _TUNNEL["tunnel_url"],
            "--tunnel-token-file",
            _TUNNEL["tunnel_token_file"],
            "--tunnel-target",
            _TUNNEL["tunnel_target"],
            "--opentela-bootstrap-addr",
            _PEER_ID,
            "--opentela-service-name",
            "pool",
        )
    )
    assert (la.tunnel_url, la.tunnel_token_file, la.tunnel_target) == tuple(_TUNNEL.values())
    assert la.opentela_bootstrap_addr == _PEER_ID
    assert la.opentela_service_name == "pool"


def test_cli_bootstrap_addr_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_OPENTELA_BOOTSTRAP_ADDR", _PEER_ID)
    assert _build(_parse()).opentela_bootstrap_addr == _PEER_ID
    monkeypatch.setenv("SML_OPENTELA_BOOTSTRAP_ADDR", "")
    assert _build(_parse()).opentela_bootstrap_addr is None


# ── environment overrides ─────────────────────────────────────────────────────


def test_telemetry_endpoint_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SML_TELEMETRY_ENDPOINT", raising=False)
    assert telemetry_endpoint() == TELEMETRY_ENDPOINT
    monkeypatch.setenv("SML_TELEMETRY_ENDPOINT", "")
    assert telemetry_endpoint() is None
    monkeypatch.setenv("SML_TELEMETRY_ENDPOINT", "https://t.example.com/launches")
    assert telemetry_endpoint() == "https://t.example.com/launches"


def test_health_check_url_and_model_prefix_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from swiss_ai_model_launch.cli.healthcheck import checker

    monkeypatch.setenv("SML_HEALTH_CHECK_URL", "https://api.example.org/v1/chat/completions")
    monkeypatch.setenv("SML_HEALTH_MODEL_PREFIX", "hosted_vllm/")
    try:
        reloaded = importlib.reload(checker)
        assert reloaded._HEALTH_CHECK_URL == "https://api.example.org/v1/chat/completions"
        assert reloaded._MODEL_PREFIX == "hosted_vllm/"
    finally:
        monkeypatch.delenv("SML_HEALTH_CHECK_URL")
        monkeypatch.delenv("SML_HEALTH_MODEL_PREFIX")
        importlib.reload(checker)
    assert launch_args_module.FRAMEWORK_PORT == 8080  # untouched by any of the above
