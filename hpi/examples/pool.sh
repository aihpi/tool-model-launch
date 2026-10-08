#!/bin/bash
# The GPU pool: one long-lived job serving every model in ~/.sml/pool.toml from one H100
# through vLLM sleep mode; same as `sml advanced --recipe pool`. LiteLLM rows for the pool
# models all point at .../v1/service/pool/v1 (see hpi/README.md).
# Prerequisites: as in qwen3-0.6b-vllm.sh, plus
#   sed "s|<user>|$USER|" "$(dirname "$0")/../pool.toml" > ~/.sml/pool.toml
exec sml advanced --recipe "$(dirname "${BASH_SOURCE[0]}")/../recipes/pool.args" "$@"
