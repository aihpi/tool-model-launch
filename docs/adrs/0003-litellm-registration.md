# ADR-0003: LiteLLM rows for sml models come from the OpenTela table

**Status**: Proposed (2026-10-08). To be built in `aihpi/litellm-k8s`, not in this repository.

## Context

At HPI an sml job registers its model at the OpenTela head (k8s, namespace `litellm`) through a
wstunnel; LiteLLM then needs one row per served name with
`api_base = http://otela-head.litellm.svc.cluster.local:8092/v1/service/<service>/v1`. Today that
row is added by hand through `/model/new` with the LiteLLM admin key. Users cannot do it, rows
outlive their jobs, and every new model needs an admin.

Measured on 2026-10-08 with two jobs serving different models on one node (jobs 2624157 and
2624158, OpenTela v0.2.4), requests sent through a worker's own OpenTela node:

| Requested model | Requests | Answered by the requested model |
| --- | --- | --- |
| `felix.boelter/Qwen/Qwen3-0.6B` | 40 | 40 |
| `felix.boelter/Qwen/Qwen2.5-0.5B-Instruct` | 40 | 40 |
| a model nobody serves | 3 | `503 No provider found for the requested service.` |

So OpenTela routes a service **by model**, not randomly across its workers: every service entry in
the table carries `identity_group: ["model=<served name>"]`, and requests are matched on the
body's `model`. All single-model jobs can share the service `llm` and one `api_base`.

The table (`GET /v1/dnt/table` on the head, keyed by `/<peer id>`) gives everything a row needs:

| Field | Example | Use |
| --- | --- | --- |
| `service[].name` | `llm`, `pool` | the `api_base` path |
| `service[].identity_group` | `model=felix.boelter/Qwen/Qwen3-0.6B` | the row's model name |
| `labels.launched_by` | `felix.boelter` | owner; must match the served-name prefix |
| `labels.served_model_name` | `felix.boelter/Qwen/Qwen3-0.6B` | cross-check |
| `labels.pool_models` | `alice/Qwen/Qwen3-0.6B,alice/Qwen/Qwen3-8B` | pool catalog (sml ≥ f953e32) |
| `labels.expires_at` | `2026-10-08T12:29:27Z` | the job's planned end |
| `status`, `connected`, `last_seen` | `ready`, `true` | liveness |

## Decision

A **reconciler** runs as a small Deployment next to the head in the `litellm` namespace:

1. Every 30 s it reads `/v1/dnt/table` and builds the desired set of rows: one per distinct model
   name across ready, connected workers (replicas of one model collapse into one row; OpenTela
   balances between them). For a pool worker, one row per `pool_models` entry, or per
   `identity_group` entry if OpenTela lists the pool's models there (to be checked once a pool
   runs on v0.2.4).
2. A row is `model_name = model = hosted_vllm/<served name>`, `api_base = …/v1/service/<service>/v1`,
   `model_info.access_groups = ["otela-test"]` for now, and
   `model_info.managed_by = "sml-reconciler"`.
3. It adds missing rows with `/model/new` and deletes rows it manages whose model has been absent
   for **3 minutes** (restarts and `--consecutive` handovers stay invisible to users). It never
   touches rows without `managed_by = sml-reconciler`.
4. It refuses a model whose served-name prefix differs from `launched_by` (no publishing under
   someone else's name) and skips workers past `expires_at`.
5. Safety: if the table is empty or the head is unreachable, it deletes nothing that cycle.

The admin key lives in a k8s Secret in the `litellm` namespace and never reaches the cluster.

Alternatives considered:

1. **Self-registration from the job** with a scoped LiteLLM key: the key would sit on every compute
   node and in every user's home, and a job killed by Slurm leaves its row behind. Rejected.
2. **One wildcard row** (`hosted_vllm/*` pointing at the `llm` service): works for routing, but
   LiteLLM cannot list or access-control individual models behind a wildcard, and `/v1/models`
   shows nothing useful. Rejected.
3. **Manual rows** (status quo): works, needs an admin for every model and leaves stale rows.

## Consequences

- A model appears in LiteLLM about 30 s after its job is healthy and disappears about 3.5 min
  after the job ends; `sml` users never touch LiteLLM.
- Per-model service names are not needed (the measurement above); sml keeps `llm` and `pool`.
- Access policy is one place to change later: per-user access groups (`sml-user-<user>`) or a
  `public` label are a reconciler change, not a job change.
- The reconciler depends on the table format of OpenTela v0.2.4; an OpenTela upgrade has to be
  checked against it.
- One more component in `litellm-k8s` to run and monitor (a single loop, no state beyond LiteLLM
  itself).
