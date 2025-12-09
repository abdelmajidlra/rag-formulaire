#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/reindex-${TIMESTAMP}.log"

python "${SCRIPT_DIR}/init_indexes.py" >"${LOG_FILE}" 2>&1
