# HPI AISC cluster

Everything HPI-specific lives in this folder; `src/` only carries generic options
whose defaults are upstream behaviour, so `git merge upstream/main` stays clean.

| Item | Value |
| --- | --- |
| Login node | `rx02` |
| Slurm | account `aisc-staff`, partition `aisc-batch`, nodes are shared (`--gres`, no `--exclusive`), `ga03` is aarch64 (excluded) |
| OpenTela head | k8s, namespace `litellm`, peer `QmWBnedcUEawmdTQTXgyY6BAQFDMwiUDndmRqgGkRBY4Qr`, reached via wstunnel to `otela-head.litellm.svc.cluster.local:43905` |
| Tunnel | `wss://api.aisc.hpi.de:443`, path prefix `otela-<token>`; token in `~/otela-tunnel-token` (mode 600, **never in git**) |
| Head HTTP API (inside k8s only) | `http://otela-head.litellm.svc.cluster.local:8092` |
| Binaries | OpenTela `v0.2.4` (`otela-amd64`), wstunnel `v10.7.1` |
| Runbook of the underlying setup | `docs/opentela-slurm.md` in `aihpi/litellm-k8s` |

## One-time setup

1. **Shared directory** (absolute path, readable from compute nodes; replace `/PATH/TO/aisc-share` in
   [`envs/vllm_hpi.toml`](envs/vllm_hpi.toml)):

   ```text
   aisc-share/
     otela-share/prod/otela-amd64       # OpenTela v0.2.4
     otela-share/prod/wstunnel-amd64    # wstunnel v10.7.1
     images/vllm-openai.sqsh            # see below
     hf-cache/                          # HF_HOME
     vllm-cache/                        # VLLM_CACHE_ROOT
     enroot-data/<user>/                # pyxis container rootfs (see step 3)
   ```

   The layout mirrors upstream's `/opentelabin/{prod,dev}/otela-<arch>`, which the launch
   scripts resolve at run time.

2. **Container image**: import the vLLM image matching the version you validated with the venv
   (vLLM >= 0.28):

   ```bash
   enroot import -o /PATH/TO/aisc-share/images/vllm-openai.sqsh docker://vllm/vllm-openai:<tag>
   ```

3. **Keep pyxis out of home** (automatic): pyxis unpacks every container rootfs into
   `~/.local/share/enroot` (~30 GB per running vLLM job against the 200 GB home quota) and
   ignores `ENROOT_DATA_PATH` from the job environment. With `SML_ENROOT_DATA_PATH` from
   `sml.env` (or `--enroot-data-path`), every job's `master.sh` turns that directory into a
   symlink to `aisc-share/enroot-data/$USER` and removes rootfs left behind by jobs Slurm no
   longer knows, before the first `srun`. Nothing to do per user; if `~/.local/share/enroot`
   already holds a *running* job's rootfs the job only warns and retries next time.
   The env toml mounts `hf-cache/` and `vllm-cache/` and sets `HF_HOME` / `VLLM_CACHE_ROOT` to
   them, so weights and torch.compile caches stay out of home as well. Those three shared
   directories must be group-writable for everyone who launches (`chmod g+ws`).

4. **Token**: `~/otela-tunnel-token` on the Slurm home, mode 600. The job reads it at run time;
   it never appears in scripts, labels or `squeue`.

5. **sml**: `source hpi/sml.env`, then `sml init` choosing the `slurm` launcher. The
   "research API key" is a LiteLLM key that has the `otela-test` access group.

## Launch

```bash
source hpi/sml.env
bash hpi/examples/qwen3-0.6b-vllm.sh
```

The example is the `sml advanced` form of the reference `~/otela-worker.sbatch`. What the flags do:

| Flag | Why |
| --- | --- |
| `--gres gpu:h100:1 --no-exclusive --cpus-per-task 8 --mem 48G` | shared nodes; `aisc-batch` also holds A30 (`gx17v1`) and L40 (`ga03`) nodes, so name the GPU type |
| `--container-spec pyxis` | the cluster's pyxis has no `--environment` (EDF) flag: the env toml is translated into `--container-image/--container-mounts/--container-workdir/--container-env`, image entrypoint skipped |
| `--sbatch-arg=--exclude=ga03` | no arm64 binaries |
| `--framework-port auto` | two jobs may share a node: port from `SLURM_JOB_ID` |
| `--tunnel-url/--tunnel-token-file/--tunnel-target` | wstunnel to the head; the bootstrap addr from `sml.env` is a bare peer ID reached through it |
| `--disable-metrics --disable-dcgm-exporter` | no Prometheus pipeline here |

Every OpenTela peer also gets `--bootstrap.static` (private mesh: never the public eth-easl
bootstrap servers), a per-step `--config-dir` and a deterministic `--seed`, so `--replicas 2` gives
two distinct peers even on a shared home.

Dry run without submitting: append `--output-script /tmp/check` and read `master.sh` / `head.sh`.

## Register in LiteLLM

One row per served name (replicas are balanced by OpenTela, not LiteLLM), via the admin API, not
`config.yaml`:

```bash
curl -s -X POST https://api.aisc.hpi.de/model/new \
  -H "Authorization: Bearer $LITELLM_ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{
    "model_name": "hosted_vllm/<user>/Qwen/Qwen3-0.6B",
    "litellm_params": {
      "model": "hosted_vllm/<user>/Qwen/Qwen3-0.6B",
      "api_base": "http://otela-head.litellm.svc.cluster.local:8092/v1/service/llm/v1"
    },
    "model_info": { "access_groups": ["otela-test"] }
  }'
```

`<user>` is the Slurm username: sml namespaces every served name as `<user>/<vendor>/<model>`.

## GPU pool: many models on one H100

`hpi/pool.toml` + `hpi/examples/pool.sh` run one job that keeps every listed model resident but asleep and wakes the requested one in seconds ([docs/gpu-pool.md](../docs/gpu-pool.md)). Register each catalog model in LiteLLM with the **same** `api_base`, `http://otela-head.litellm.svc.cluster.local:8092/v1/service/pool/v1`, and `model: hosted_vllm/<served_name>`. Replace `<user>` in `pool.toml` with your Slurm username before submitting.

## Verify

1. Job log: wstunnel connected, `bootstrap_connected=true`, relay reservation on
   `QmWBnedcUEaw…` only, health check passed.
2. From the deploy host: the head's table holds the head plus our replicas, nothing else.

   ```bash
   kubectl -n litellm run curl-$RANDOM --rm -i --restart=Never --image=curlimages/curl:8.10.1 -- \
     -s http://otela-head.litellm.svc:8092/v1/dnt/table | grep -o '"id":"[^"]*"'
   ```

3. A chat completion through LiteLLM answers; the `sml` health panel turns green.
