#!/bin/bash
# Equivalent of the hand-written ~/otela-worker.sbatch: one vLLM replica on one
# shared H100, registered at our OpenTela head through the wstunnel.
# Prerequisites: `source hpi/sml.env`, `sml init` (launcher: slurm), the paths in
# hpi/envs/vllm_hpi.toml, and ~/otela-tunnel-token (mode 600).
# Resolve hpi/ relative to this script so it runs from any directory.
HPI=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sml advanced \
  --framework vllm \
  --environment "$HPI/envs/vllm_hpi.toml" \
  --container-spec pyxis \
  --enroot-data-path '/sc/projects/sci-aisc/aisc-share/enroot-data/$USER' \
  --gres gpu:h100:1 \
  --no-exclusive \
  --cpus-per-task 8 \
  --mem 48G \
  --sbatch-arg=--exclude=ga03 \
  --framework-port auto \
  --disable-metrics \
  --disable-dcgm-exporter \
  --tunnel-url wss://api.aisc.hpi.de:443 \
  --tunnel-token-file ~/otela-tunnel-token \
  --tunnel-target otela-head.litellm.svc.cluster.local:43905 \
  --time 02:00:00 \
  --framework-args "--model Qwen/Qwen3-0.6B \
    --served-model-name Qwen/Qwen3-0.6B \
    --host 0.0.0.0"
