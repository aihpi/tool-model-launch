# ADR-0002: Many models per GPU through a pool job with vLLM sleep mode

**Status**: Accepted (2026-09-04)

## Context

At HPI, LiteLLM lists many models that are used sporadically. Keeping one Slurm job per model up around the clock wastes GPUs; starting a job per request takes minutes (queue + weight load) and users see failures until it is up. GPUs are H100 80 GB on shared nodes; a single model of interest often needs less than half a GPU.

vLLM's sleep mode (`--enable-sleep-mode`, `POST /sleep?level=1`, `POST /wake_up`) offloads a model's weights to host RAM and frees its GPU memory while the process stays alive; waking takes seconds. Two facts shape the design: a vLLM process asserts that its `gpu_memory_utilization` fraction is free when it starts, and putting a model to sleep while it is processing a request crashes it.

## Decision

A **pool job**: one long-lived Slurm job holding N GPUs runs `pool_agent.py` as its framework process. The agent is an OpenAI-compatible front door that dispatches by model name to per-model vLLM children, keeps every child resident but asleep, wakes on demand, drains before sleeping, and evicts least-recently-used awake models when the requested one does not fit. The pool is one OpenTela worker under its own service name (`pool`); all its LiteLLM rows share one `api_base`.

Alternatives considered:

1. **Per-model Slurm scheduler** ("keeper"): a cron tick reads LiteLLM's Prometheus counters (`litellm_proxy_total_requests_metric{requested_model}` also counts failed requests to cold models, so it doubles as the wake signal), submits a job for wanted models, cancels idle ones, evicts LRU under a GPU budget. Rejected for the first pass: cold starts of minutes on every wake, LiteLLM-side plumbing (`/metrics` exposure and auth), a state file, and a controller that has to keep running somewhere. Kept as the follow-up if holding GPUs around the clock turns out too expensive.
2. **Static co-location** (two vLLM instances at 0.45 each, always awake): no code, but the awake set is fixed at start; it cannot host a third model or a large one.
3. **Idle-exit inside each single-model job**: scale-to-zero but no scale-from-zero without (1).

## Consequences

- Switching between models takes seconds instead of minutes; more models than GPUs can be "online" in LiteLLM at once.
- The pool's GPUs are held for the job's lifetime; `--consecutive` keeps it up, and the successor boots its catalog while the old pool serves.
- Host RAM bounds the catalog (`--mem` ≥ Σ weights); level-2 sleep would lift that at the price of slower wakes and is not implemented.
- `--opentela-service-name` had to become a generic option; the OpenTela head routes a service, not a model name.
- The agent is a new component to maintain (~350 lines, pure planning core unit-tested, HTTP path smoke-tested against a fake `vllm`).
