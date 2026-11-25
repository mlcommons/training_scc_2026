#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/source-or-patches"
exec ./run.sh submission "${1:-10}"
