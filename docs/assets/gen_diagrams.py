"""Generate docs/assets/sml-workflow.svg and docs/assets/gpu-pool.svg.
Standalone files (used via <img>), so colours are explicit and read on light and dark grounds."""
from pathlib import Path

INK = "#334155"      # text + default strokes
MUTED = "#64748b"    # arrows, notes
FILL = "#f8fafc"     # box fill
UP = "#94a3b8"       # upstream box border
NEW = "#2563eb"      # new generic flag/option (code in src/)
HPI = "#d97706"      # hpi/ config
FONT = "font-family='ui-monospace, SFMono-Regular, Menlo, monospace'"
SANS = "font-family='ui-sans-serif, system-ui, sans-serif'"


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h, self.label = w, h, label
        self.parts = []

    def rect(self, x, y, w, h, stroke=UP, fill=FILL, dash=None, rx=6, sw=1.5):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{d}/>")

    def text(self, x, y, s, size=12, anchor="middle", color=INK, mono=False, weight="normal"):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        f = FONT if mono else SANS
        self.parts.append(f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' fill='{color}' font-weight='{weight}' {f}>{s}</text>")

    def box(self, x, y, w, h, lines, stroke=UP, size=12, mono_from=1, title_weight="600"):
        """Box with a sans title line then mono detail lines, vertically centred.
        Detail lines are 10px mono; every line must fit the box (checked)."""
        self.rect(x, y, w, h, stroke=stroke)
        n = len(lines)
        lh = size + 3
        y0 = y + h / 2 - (n - 1) * lh / 2 + size / 2 - 1
        for i, line in enumerate(lines):
            mono = i >= mono_from
            sz = size if i == 0 else 10
            est = len(line) * (0.62 if mono else 0.56) * sz
            if est > w - 16:
                print(f"OVERFLOW {est:.0f} > {w - 16}: {line!r}")
            self.text(x + w / 2, y0 + i * lh, line, size=sz, mono=mono,
                      weight=title_weight if i == 0 else "normal", color=INK if i == 0 else "#475569")

    def arrow(self, x1, y1, x2, y2, label=None, dash=None, color=MUTED, lx=None, ly=None, anchor="middle", size=11):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.5' marker-end='url(#arrow)'{d}/>")
        if label:
            self.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6, label, size=size, color=MUTED, anchor=anchor)

    def path(self, d, label=None, dash=None, color=MUTED, lx=0, ly=0, anchor="middle", size=11, head=True):
        dd = f" stroke-dasharray='{dash}'" if dash else ""
        m = " marker-end='url(#arrow)'" if head else ""
        self.parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='1.5'{m}{dd}/>")
        if label:
            self.text(lx, ly, label, size=size, color=MUTED, anchor=anchor)

    def render(self):
        head = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {self.w} {self.h}' width='{self.w}' height='{self.h}' "
                f"role='img' aria-label='{self.label}'>\n"
                "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='8' markerHeight='8' orient='auto-start-reverse'>"
                f"<path d='M 0 0 L 10 5 L 0 10 z' fill='{MUTED}'/></marker></defs>\n"
                f"<rect x='0' y='0' width='{self.w}' height='{self.h}' fill='white'/>\n")
        return head + "\n".join(self.parts) + "\n</svg>\n"


def legend(s, x, y):
    for i, (color, label) in enumerate([(UP, "upstream sml, unchanged"), (NEW, "new generic option in src/ (default = upstream behaviour)"), (HPI, "value from hpi/ (config, never in src/)")]):
        yy = y + i * 18
        s.rect(x, yy - 9, 22, 12, stroke=color, rx=3)
        s.text(x + 30, yy + 1, label, size=11, anchor="start", color=MUTED)


# ───────────────────────────── 1. sml workflow ─────────────────────────────
W, H = 1440, 720
s = Svg(W, H, "How an sml advanced launch becomes a Slurm job whose replica reaches the OpenTela head in Kubernetes through a wstunnel and is served through LiteLLM")
lanes = [
    (20, 340, "laptop / login node rx02", "sml (Python)"),
    (360, 700, "Slurm batch node", "master.sh"),
    (720, 1080, "compute node · pyxis container", "head.sh (one per replica)"),
    (1100, 1420, "gateway + Kubernetes (litellm ns)", "aihpi/litellm-k8s"),
]
for x0, x1, title, sub in lanes:
    s.rect(x0, 20, x1 - x0, H - 100, stroke="#e2e8f0", fill="#ffffff", rx=10, sw=1)
    s.text((x0 + x1) / 2, 44, title, size=14, weight="700")
    s.text((x0 + x1) / 2, 62, sub, size=11, color=MUTED, mono=True)

# lane 1
BW, BH = 296, 52
x = 42
s.box(x, 90, BW, BH, ["sml advanced …", "hpi/examples/*.sh · source hpi/sml.env"], stroke=HPI)
s.box(x, 172, BW, BH, ["build_launch_args_from_advanced", "cli/main.py → LaunchArgs"])
s.box(x, 254, BW, BH, ["render_sbatch_header + render_master", "launchers/utils.py · framework.py"])
s.box(x, 336, BW, 88, ["master.sh", "heredocs: head.sh follower.sh router.sh", "replica_health_checker.py pool_agent.py", "#SBATCH --gres --no-exclusive --sbatch-arg"], stroke=NEW)
s.box(x, 454, BW, BH, ["sbatch (SlurmLauncher)", "from ~/.sml · named job, adopt-before-retry"])
for y in (142, 224, 306):
    s.arrow(x + BW / 2, y, x + BW / 2, y + 30)
s.arrow(x + BW / 2, 424, x + BW / 2, 454)
s.arrow(x + BW, 480, 384, 116, label="job starts", lx=352, ly=470, anchor="middle", size=10)

# lane 2
x = 384
s.box(x, 90, BW, BH, ["arch detection", "export OPENTELA_BIN, WSTUNNEL_BIN"], stroke=NEW)
s.box(x, 172, BW, BH, ["env TOML resolution", "hpi/envs/vllm_hpi.toml: image mounts [env]"], stroke=HPI)
s.box(x, 254, BW, BH, ["self-extract rank scripts", "RANKS_DIR=~/.sml/job-$SLURM_JOB_ID"])
s.box(x, 336, BW, 62, ["srun … --environment=<EDF>", "--container-mounts $RANKS_DIR", "bash $RANKS_DIR/head.sh  (per replica)"])
s.box(x, 428, BW, BH, ["replica_health_checker.py", "GET <replica ip>:$FRAMEWORK_PORT/health"])
s.box(x, 510, BW, BH, ["wait -n on critical srun PIDs", "first exit ends the job"])
for y in (142, 224, 306):
    s.arrow(x + BW / 2, y, x + BW / 2, y + 30)
s.arrow(x + BW / 2, 398, x + BW / 2, 428)
s.arrow(x + BW / 2, 480, x + BW / 2, 510)
s.arrow(x + BW, 362, 744, 116, label="starts", lx=712, ly=250, size=10)
s.path(f"M {x + BW} 454 L 725 454 L 725 415", label="probes /health", dash="4 3", lx=712, ly=475, head=True, size=10)

# lane 3
x = 744
s.box(x, 90, BW, BH, ["env exports + site setup", "FRAMEWORK_PORT=$((20000+SLURM_JOB_ID%10000))"], stroke=NEW)
s.box(x, 172, BW, 62, ["wstunnel client", "-L tcp://127.0.0.1:$TUN:<otela-head>:43905", "prefix otela-$(cat ~/otela-tunnel-token)"], stroke=NEW)
s.box(x, 264, BW, 88, ["otela start", "--bootstrap.static /ip4/127.0.0.1/tcp/$TUN/…", "--config-dir … --seed $((job*1000+step))", "--service.name llm|pool --service.port $PORT"], stroke=NEW)
s.box(x, 382, BW, 62, ["--subprocess: framework", "vllm serve --port $FRAMEWORK_PORT …", "or python3 …/pool_agent.py --port …"])
s.arrow(x + BW / 2, 142, x + BW / 2, 172)
s.arrow(x + BW / 2, 234, x + BW / 2, 264)
s.arrow(x + BW / 2, 352, x + BW / 2, 382)
s.text(x + BW / 2, 466, "OpenTela forwards requests to :$FRAMEWORK_PORT", size=10, color=MUTED)
# otela -> tunnel (loop back up)
s.path(f"M {x + BW} 308 L 1064 308 L 1064 203 L {x + BW} 203", label="p2p via 127.0.0.1:$TUN", lx=1068, ly=258, anchor="start", size=10)
# tunnel -> gateway
s.path(f"M {x + BW / 2} 172 L {x + BW / 2} 156 L 1260 156 L 1260 172", label="wss://api.aisc.hpi.de:443 · path otela-<token>", lx=1100, ly=150, anchor="middle", size=10)

# lane 4
x = 1130
s.box(x, 172, 260, 52, ["Caddy → wstunnel server", "sidecar in litellm-proxy"])
s.box(x, 264, 260, 52, ["otela-head :43905", "p2p mesh · relay · seed 43905"])
s.box(x, 356, 260, 52, ["otela-head :8092 (cluster-local)", "/v1/service/<svc>/v1/chat/completions"])
s.box(x, 448, 260, 52, ["LiteLLM (api.aisc.hpi.de)", "hosted_vllm/<user>/<vendor>/<model>"])
s.box(x, 540, 260, 52, ["user · OpenWebUI · API key", "access group otela-test"])
s.arrow(x + 130, 224, x + 130, 264, label="tcp to head", size=10, lx=x + 200)
s.arrow(x + 130, 356, x + 130, 316, label="picks a registered worker", size=10, lx=x + 130, ly=344)
s.arrow(x + 130, 448, x + 130, 408, label="api_base", size=10, lx=x + 175, ly=432)
s.arrow(x + 130, 540, x + 130, 500, label="POST /v1/chat/completions", size=10, lx=x + 130, ly=528)

legend(s, 30, 660)
s.text(W - 20, 700, "sml-workflow.svg · aihpi/tool-model-launch", size=10, anchor="end", color=MUTED)
Path("docs/assets/sml-workflow.svg").write_text(s.render())

# ───────────────────────────── 2. GPU pool ─────────────────────────────
W, H = 1240, 670
s = Svg(W, H, "One Slurm job holding a GPU runs the pool agent: an OpenAI-compatible front door that dispatches by model name to per-model vLLM children sharing the GPU through sleep mode, waking the requested model and putting least-recently-used ones to sleep")

# inbound
s.box(20, 40, 200, 52, ["LiteLLM", "rows: hosted_vllm/<user>/…"])
s.box(20, 122, 200, 52, ["otela-head", "/v1/service/pool/v1 → worker"])
s.arrow(120, 92, 120, 122, label="api_base", size=10, lx=160)
s.arrow(220, 148, 270, 148, label="mesh", size=10, ly=140)

# job box
JX, JY, JW, JH = 270, 30, 660, 470
s.rect(JX, JY, JW, JH, stroke="#cbd5e1", fill="#ffffff", rx=10, dash="6 4")
s.text(JX + 12, JY + 18, "Slurm job: --framework pool --gres gpu:1 --mem <sum of weights + 20%> --consecutive", size=10, anchor="start", color=MUTED, mono=True)

# front door
s.box(JX + 20, 110, 380, 76, ["pool_agent.py  (front door)", ":$FRAMEWORK_PORT · one OpenTela worker, service pool", "otela start --subprocess python3 …/pool_agent.py", "/health is 200 only once every model is resident"], stroke=NEW)
s.text(JX + 210, 205, "dispatch on body.model", size=11, color=MUTED, mono=True)

# children
CX = JX + 20
cols = [(CX, "A  0.45 GPU", "awake", "#16a34a"), (CX + 220, "B  0.45 GPU", "awake", "#16a34a"), (CX + 440, "C  0.90 GPU", "asleep", MUTED)]
for cx, title, state, col in cols:
    s.rect(cx, 230, 190, 70, stroke=col)
    s.text(cx + 95, 250, f"vllm serve {title[0]}", size=12, weight="600")
    s.text(cx + 95, 267, "--enable-sleep-mode", size=11, mono=True, color="#475569")
    s.text(cx + 95, 283, f"--gpu-memory-utilization {title.split()[1]}", size=11, mono=True, color="#475569")
    s.text(cx + 95, 318, state, size=12, color=col, weight="700")
    s.arrow(JX + 210, 186, cx + 95, 230, dash=None if state == "awake" else "4 3")
s.text(JX + JW - 20, 100, "children: VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES=<gpus>", size=10, color=MUTED, mono=True, anchor="end")

# GPU memory strip
GX, GY, GW, GH = JX + 20, 350, 620, 34
s.text(GX, GY - 8, "H100 · 80 GB", size=11, anchor="start", color=MUTED)
s.rect(GX, GY, GW, GH, stroke=UP, fill="#ffffff", rx=4)
s.rect(GX, GY, GW * 0.45, GH, stroke="#16a34a", fill="#dcfce7", rx=4)
s.text(GX + GW * 0.225, GY + 22, "A  weights + KV cache  (0.45)", size=11)
s.rect(GX + GW * 0.45, GY, GW * 0.45, GH, stroke="#16a34a", fill="#dcfce7", rx=4)
s.text(GX + GW * 0.675, GY + 22, "B  weights + KV cache  (0.45)", size=11)
s.rect(GX + GW * 0.90, GY, GW * 0.10, GH, stroke=UP, fill="#f1f5f9", rx=4)
s.text(GX + GW * 0.95, GY + 22, "free", size=10, color=MUTED)
s.text(GX, GY + GH + 14, "≈0.5 GB CUDA context per sleeping child · Σ awake ≤ 1 − gpu_headroom", size=10, anchor="start", color=MUTED)

# host RAM strip
RY = 420
s.text(GX, RY - 8, "host RAM (--mem)", size=11, anchor="start", color=MUTED)
s.rect(GX, RY, GW, 34, stroke=UP, fill="#ffffff", rx=4)
s.rect(GX + GW * 0.30, RY, GW * 0.60, 34, stroke=MUTED, fill="#f1f5f9", rx=4)
s.text(GX + GW * 0.60, RY + 22, "C  weights offloaded (sleep level 1): wakes in seconds", size=11)
s.arrow(CX + 600, 300, CX + 600, RY, dash="4 3", label="/sleep?level=1", size=10, lx=CX + 600, ly=340)

# footer note inside job
s.text(JX + 12, JY + JH - 12, "sleeper loop, every 30 s: awake, in_flight = 0, idle >= sleep_after -> drain -> /sleep", size=10, anchor="start", color=MUTED, mono=True)

# state machine (right)
SX = 960
s.text(SX + 120, 52, "per-model state machine", size=13, weight="700")
def state(x, y, name, col=UP):
    s.rect(x, y, 120, 36, stroke=col, rx=18)
    s.text(x + 60, y + 23, name, size=12, weight="600")
state(SX + 60, 80, "cold")
state(SX + 60, 170, "starting")
state(SX + 60, 270, "awake", "#16a34a")
state(SX + 60, 400, "asleep", MUTED)
s.arrow(SX + 120, 116, SX + 120, 170, label="vllm serve (request or boot)", size=10, lx=SX + 120, ly=148)
s.arrow(SX + 120, 206, SX + 120, 270, label="/health 200", size=10, lx=SX + 120, ly=243)
s.arrow(SX + 100, 306, SX + 100, 400, label="drain, then /sleep", size=10, lx=SX + 44, ly=350, anchor="middle")
s.text(SX + 44, 363, "idle ≥ sleep_after", size=10, color=MUTED)
s.text(SX + 44, 376, "or LRU eviction", size=10, color=MUTED)
s.arrow(SX + 140, 400, SX + 140, 306, label="/wake_up", size=10, lx=SX + 196, ly=350)
s.text(SX + 196, 363, "on request, after", size=10, color=MUTED)
s.text(SX + 196, 376, "freeing its GPUs", size=10, color=MUTED)
s.path(f"M {SX + 180} 418 L {SX + 230} 418 L {SX + 230} 98 L {SX + 180} 98", label="child died", dash="4 3", lx=SX + 232, ly=260, anchor="start", size=10)
s.text(SX + 120, 470, "boot: start each model in turn, sleep it,", size=10, color=MUTED)
s.text(SX + 120, 483, "then the next — vLLM needs its fraction free at start", size=10, color=MUTED)

s.text(20, 560, "Request for C while A and B are awake: plan_wake picks the least-recently-used awake models on C's GPUs (A, then B), drains and sleeps them,", size=11, anchor="start", color=INK)
s.text(20, 578, "then /wake_up C. Requests for A and B keep flowing while both fit (0.45 + 0.45 ≤ 0.95). Sleeping mid-request would crash vLLM, so drain always comes first.", size=11, anchor="start", color=INK)
for i, (color, label) in enumerate([("#16a34a", "awake: holds its GPU fraction"), (MUTED, "asleep: weights in host RAM, GPU memory released"), (NEW, "new code in this fork (assets/pool_agent.py, --framework pool)")]):
    yy = 605 + i * 16
    s.rect(20, yy - 9, 22, 12, stroke=color, rx=3)
    s.text(50, yy + 1, label, size=11, anchor="start", color=MUTED)
s.text(W - 20, 655, "gpu-pool.svg · aihpi/tool-model-launch", size=10, anchor="end", color=MUTED)
Path("docs/assets/gpu-pool.svg").write_text(s.render())
print("ok")
