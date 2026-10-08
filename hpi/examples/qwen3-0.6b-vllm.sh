#!/bin/bash
# The reference job (one vLLM replica of Qwen3-0.6B on one shared H100), kept as a script
# for muscle memory; it is the same as `sml advanced --recipe qwen3-0.6b`.
# Prerequisites: `source /sc/projects/sci-aisc/aisc-share/sml/env.sh` (or hpi/sml.env),
# `sml init`, and ~/otela-tunnel-token (mode 600). Extra flags are passed through.
exec sml advanced --recipe "$(dirname "${BASH_SOURCE[0]}")/../recipes/qwen3-0.6b.args" "$@"
