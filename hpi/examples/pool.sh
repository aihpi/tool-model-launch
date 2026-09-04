#!/bin/bash
# One long-lived job holding one H100 and serving every model in hpi/pool.toml
# through vLLM sleep mode. LiteLLM rows for these models all point at
# .../v1/service/pool/v1 (see hpi/README.md). --mem must hold all weights.
# Prerequisites: as in qwen3-0.6b-vllm.sh, plus `sed -i "s|<user>|$USER|" hpi/pool.toml`.
# Resolve hpi/ relative to this script so it runs from any directory.
HPI=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sml advanced \
  --framework pool \
  --environment "$HPI/envs/vllm_hpi.toml" \
  --container-spec pyxis \
  --enroot-data-path '/sc/projects/sci-aisc/aisc-share/enroot-data/$USER' \
  --served-model-name pool \
  --opentela-service-name pool \
  --gres gpu:h100:1 \
  --no-exclusive \
  --cpus-per-task 16 \
  --mem 100G \
  --sbatch-arg=--exclude=ga03 \
  --framework-port auto \
  --disable-metrics \
  --disable-dcgm-exporter \
  --tunnel-url wss://api.aisc.hpi.de:443 \
  --tunnel-token-file ~/otela-tunnel-token \
  --tunnel-target otela-head.litellm.svc.cluster.local:43905 \
  --time 12:00:00 \
  --framework-args "--config $HPI/pool.toml"
