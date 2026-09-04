import math
import os
import re
import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from swiss_ai_model_launch.launchers.topology import Topology

# Routing strategy across replicas. OpenTela (default): OpenTela load-balances across
# the replica peers on the mesh. sglang: an in-job SGLang router fronts the replicas
# and becomes the served endpoint.
RouterMode = Literal["opentela", "sglang"]
ROUTER_OPENTELA: RouterMode = "opentela"
ROUTER_SGLANG: RouterMode = "sglang"

# The framework's HTTP server port is hardcoded across the system: it's
# auto-injected as ``--port`` into framework_args, used as OpenTela's
# ``--service.port``, and embedded in the router's worker URLs. Exposing
# it as a knob just creates ways for the three to drift.
FRAMEWORK_PORT = 8080
# LaunchArgs.framework_port == "auto": the scripts derive a per-job port from
# SLURM_JOB_ID at run time (shared-node clusters; two jobs may land on one node).
FRAMEWORK_PORT_AUTO = "auto"
FRAMEWORK_PORT_AUTO_EXPR = "$((20000 + SLURM_JOB_ID % 10000))"

ContainerSpec = Literal["edf", "pyxis"]
CONTAINER_SPEC_EDF: ContainerSpec = "edf"
CONTAINER_SPEC_PYXIS: ContainerSpec = "pyxis"


TELEMETRY_ENDPOINT = "https://sml-dev.swissai.svc.cscs.ch/launches"


def telemetry_endpoint() -> str | None:
    """Launch-telemetry sink. SML_TELEMETRY_ENDPOINT overrides it; empty disables."""
    return os.environ.get("SML_TELEMETRY_ENDPOINT", TELEMETRY_ENDPOINT) or None


# SLURM caps a single job at 12h on the target clusters. A model that needs to
# stay up longer is served by a chain of consecutive jobs (see --consecutive),
# each running for at most this cap.
DEFAULT_MAX_JOB_TIME = "12:00:00"

_PORT_FLAG_RE = re.compile(r"(?:^|\s)--port(?:[\s=])")


def time_str_to_seconds(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_time_str(seconds: int) -> str:
    # SLURM's finest time-limit granularity is a minute, so never emit 00:00:00.
    seconds = max(seconds, 60)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def plan_consecutive_offsets(total_seconds: int, job_seconds: int, handover_seconds: int) -> list[int]:
    """Start offsets (seconds from the chain's base time) for a consecutive chain.

    Each job runs for ``job_seconds`` (the SLURM per-job cap). A successor starts
    ``handover_seconds`` before its predecessor's time limit, so the spacing
    between starts is ``job_seconds - handover_seconds`` and the overlap gives the
    fresh job time to become healthy before the old one expires. The number of
    jobs is the minimum whose continuous coverage —
    ``(n - 1) * interval + job_seconds`` — reaches ``total_seconds``.

    Returns ``[0]`` (a single job) when the requested total fits inside one job.
    """
    if job_seconds <= 0:
        raise ValueError("job time must be positive")
    if handover_seconds < 0 or handover_seconds >= job_seconds:
        raise ValueError("handover time must be in the range [0, job time)")
    if total_seconds <= job_seconds:
        return [0]
    interval = job_seconds - handover_seconds
    n = math.ceil((total_seconds - job_seconds) / interval) + 1
    return [i * interval for i in range(n)]


class LaunchArgs(BaseModel):
    job_name: str
    served_model_name: str
    account: str
    partition: str

    topology: Topology = Field(default_factory=Topology)

    time: str = "02:00:00"
    # Consecutive-chain scheduling. The head job carries an absolute SLURM
    # --begin (its anchor); every successor instead carries a SLURM --dependency
    # of the form "after:<prev>+<minutes>" so it starts a fixed delay after its
    # predecessor *actually* begins — making the chain robust to queue delay
    # rather than pinned to wall-clock times guessed at submission. Both are None
    # for an ordinary single launch. previous_job_id is the predecessor this job
    # cancels from inside once all its replicas are healthy (see the in-job
    # replica health checker).
    begin: str | None = None
    dependency: str | None = None
    previous_job_id: int | None = None
    environment: str

    framework: str
    framework_args: str = ""
    pre_launch_cmds: str = ""
    router: RouterMode = ROUTER_OPENTELA
    router_args: str = ""
    disable_opentela: bool = False
    opentela_bootstrap_addr: str | None = None
    dev: bool = False
    telemetry_endpoint: str | None = None
    metrics_remote_write_url: str = "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    metrics_agent_binary: str = "/capstor/store/cscs/swissai/infra01/opentela-share/vmagent"
    dcgm_exporter_binary: str = "/capstor/store/cscs/swissai/infra01/opentela-share/dcgm-exporter"
    disable_dcgm_exporter: bool = False
    disable_metrics: bool = False
    # Site-specific #SBATCH knobs. The defaults reproduce the historical header
    # (whole exclusive nodes, no GRES/CPU/memory request) for clusters where a
    # job owns its node; shared-node clusters set them per launch. sbatch_args
    # is a verbatim passthrough for anything without its own knob
    # (--exclude, --constraint, --qos, ...).
    exclusive: bool = True
    gres: str | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    sbatch_args: list[str] = Field(default_factory=list)
    # Framework HTTP port: fixed by default (see FRAMEWORK_PORT), "auto" for
    # shared nodes.
    framework_port: int | Literal["auto"] = FRAMEWORK_PORT
    # OpenTela service the replicas (or the router) advertise. `llm` is what the
    # upstream gateway routes to; a multi-model pool registers under its own name.
    opentela_service_name: str = "llm"
    # Optional wstunnel to a bootstrap peer that is not directly routable (e.g.
    # an OpenTela head inside Kubernetes). Set all three or none. The token is
    # read from the file at run time and never appears in scripts, labels or
    # squeue output.
    tunnel_url: str | None = None
    tunnel_token_file: str | None = None
    tunnel_target: str | None = None
    # How the env toml reaches srun. "edf" hands the file to pyxis' CSCS-only
    # `--environment` flag (upstream behaviour); "pyxis" translates it at render
    # time into stock `--container-image/--container-mounts/--container-workdir`
    # flags plus exported env for sites running unmodified pyxis.
    container_spec: ContainerSpec = CONTAINER_SPEC_EDF

    @model_validator(mode="after")
    def _validate(self) -> "LaunchArgs":
        if not self.disable_metrics and not self.metrics_remote_write_url:
            raise ValueError("Metrics require a remote write URL when metrics are enabled.")
        if _PORT_FLAG_RE.search(self.framework_args):
            warnings.warn(
                f"`--port` in framework_args is redundant; the framework port is managed by sml "
                f"({self.framework_port}) and auto-injected. Setting it manually risks desyncing "
                f"the framework, OpenTela, and the router.",
                UserWarning,
                stacklevel=2,
            )
        tunnel = (self.tunnel_url, self.tunnel_token_file, self.tunnel_target)
        if any(tunnel) and not all(tunnel):
            raise ValueError("tunnel_url, tunnel_token_file and tunnel_target must be set together.")
        if self.framework == "pool":
            if self.topology.nodes_per_replica != 1:
                raise ValueError("The pool framework runs one node per replica (nodes_per_replica must be 1).")
            if self.router != ROUTER_OPENTELA:
                raise ValueError("The pool framework dispatches by model name itself; use the default opentela router.")
        return self

    @property
    def framework_port_shell(self) -> str:
        """The port as written into the scripts: a literal, or the FRAMEWORK_PORT
        shell variable the scripts define when the port is "auto"."""
        return "$FRAMEWORK_PORT" if self.framework_port == FRAMEWORK_PORT_AUTO else str(self.framework_port)

    @property
    def total_nodes(self) -> int:
        return self.topology.replicas * self.topology.nodes_per_replica

    def to_sbatch_args(self, *, reservation: str | None = None) -> list[str]:
        args = [
            f"--job-name={self.job_name}",
            f"--account={self.account}",
            f"--time={self.time}",
        ]
        if self.exclusive:
            args.append("--exclusive")
        args += [
            f"--nodes={self.total_nodes}",
            f"--partition={self.partition}",
            "--output=logs/%j/log.out",
            "--error=logs/%j/log.err",
        ]
        if self.gres:
            args.append(f"--gres={self.gres}")
        if self.cpus_per_task:
            args.append(f"--cpus-per-task={self.cpus_per_task}")
        if self.mem:
            args.append(f"--mem={self.mem}")
        if reservation:
            args.append(f"--reservation={reservation}")
        if self.begin:
            args.append(f"--begin={self.begin}")
        if self.dependency:
            args.append(f"--dependency={self.dependency}")
        args.extend(self.sbatch_args)
        return args

    def to_job_env(self) -> dict[str, str]:
        framework_args = f"--port {self.framework_port_shell} {self.framework_args}".strip()
        return {
            "FRAMEWORK": self.framework,
            "SML_ENVIRONMENT": self.environment,
            "FRAMEWORK_ARGS": framework_args,
            "PRE_LAUNCH_CMDS": self.pre_launch_cmds,
            "REPLICAS": str(self.topology.replicas),
            "NODES_PER_REPLICA": str(self.topology.nodes_per_replica),
            "ROUTER": self.router,
            "ROUTER_ENVIRONMENT": self.environment,
            "ROUTER_ARGS": self.router_args,
            "USE_OPENTELA": "false" if self.disable_opentela else "true",
            "SERVED_MODEL_NAME": self.served_model_name,
            "METRICS_REMOTE_WRITE_URL": self.metrics_remote_write_url or "",
            "METRICS_AGENT_BIN": self.metrics_agent_binary,
            "TELEMETRY_ENDPOINT": self.telemetry_endpoint or "",
            "SML_TIME": self.time,
            "SML_PREVIOUS_JOB_ID": str(self.previous_job_id) if self.previous_job_id is not None else "",
        }
