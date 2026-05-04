#!/usr/bin/env bash
set -u

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${PAPERLESS_AI_ENV_FILE:-$PROJECT_DIR/.env.lmstudio}"
RUN_DIR="${PAPERLESS_AI_RUN_DIR:-$PROJECT_DIR/paperless_lmstudio_runs/launchd-local}"
LOCK_DIR="${PAPERLESS_AI_LOCK_DIR:-$RUN_DIR.lock}"
PYTHON_BIN="${PAPERLESS_AI_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
PROVIDER="${PAPERLESS_AI_PROVIDER:-lmstudio}"
LIMIT="${PAPERLESS_AI_LIMIT:-0}"
THRESHOLD="${PAPERLESS_AI_THRESHOLD:-0.86}"
WORKERS="${PAPERLESS_AI_WORKERS:-1}"
TIMEOUT="${PAPERLESS_AI_TIMEOUT:-300}"
EXTRA_ARGS="${PAPERLESS_AI_EXTRA_ARGS:-}"
NOTIFY="${PAPERLESS_AI_NOTIFY:-1}"

mkdir -p "$RUN_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

notify() {
  local title="$1"
  local message="$2"
  if [[ "$NOTIFY" != "1" ]]; then
    return 0
  fi
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$message\" with title \"$title\"" >/dev/null 2>&1 || true
  fi
}

log="$RUN_DIR/run.log"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(timestamp)] another Paperless AI run is active; exiting" | tee -a "$log"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if [[ ! -r "$ENV_FILE" ]]; then
  echo "[$(timestamp)] env file is missing or unreadable: $ENV_FILE" | tee -a "$log"
  notify "Paperless AI failed" "Missing env file"
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[$(timestamp)] python is missing or not executable: $PYTHON_BIN" | tee -a "$log"
  notify "Paperless AI failed" "Missing Python environment"
  exit 2
fi

echo "[$(timestamp)] starting Paperless AI provider=$PROVIDER workers=$WORKERS output=$RUN_DIR" | tee -a "$log"

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

# EXTRA_ARGS is intentionally shell-split for advanced local launchd tuning.
# Keep secrets in the env file, not in EXTRA_ARGS.
set +e
"$PYTHON_BIN" -u "$PROJECT_DIR/paperless_lmstudio_classifier.py" \
  --provider "$PROVIDER" \
  --limit "$LIMIT" \
  --threshold "$THRESHOLD" \
  --rules-first \
  --drop-bulk-unclassified \
  --timeout "$TIMEOUT" \
  --resume \
  --workers "$WORKERS" \
  --output-dir "$RUN_DIR" \
  $EXTRA_ARGS 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
  echo "[$(timestamp)] Paperless AI failed with exit code $status" | tee -a "$log"
  notify "Paperless AI failed" "Exit code $status"
  exit "$status"
fi

echo "[$(timestamp)] Paperless AI completed" | tee -a "$log"
exit 0
