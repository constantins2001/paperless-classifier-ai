#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${PAPERLESS_AI_LAUNCHD_LABEL:-me.constantinschreiber.paperless-ai}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
RUNNER="$PROJECT_DIR/scripts/paperless-ai-launchd-runner.sh"
RUN_DIR="${PAPERLESS_AI_RUN_DIR:-$PROJECT_DIR/paperless_lmstudio_runs/launchd-local}"
ENV_FILE="${PAPERLESS_AI_ENV_FILE:-$PROJECT_DIR/.env.lmstudio}"
PYTHON_BIN="${PAPERLESS_AI_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
PROVIDER="${PAPERLESS_AI_PROVIDER:-lmstudio}"
WORKERS="${PAPERLESS_AI_WORKERS:-1}"
LIMIT="${PAPERLESS_AI_LIMIT:-0}"
THRESHOLD="${PAPERLESS_AI_THRESHOLD:-0.86}"
TIMEOUT="${PAPERLESS_AI_TIMEOUT:-300}"
START_INTERVAL="${PAPERLESS_AI_START_INTERVAL:-300}"
EXTRA_ARGS="${PAPERLESS_AI_EXTRA_ARGS:-}"
NOTIFY="${PAPERLESS_AI_NOTIFY:-1}"

mkdir -p "$HOME/Library/LaunchAgents" "$RUN_DIR"
chmod +x "$RUNNER"

if [[ ! -r "$ENV_FILE" ]]; then
  echo "Env file is missing or unreadable: $ENV_FILE" >&2
  echo "Create it first, for example: cp .env.example .env.lmstudio && chmod 600 .env.lmstudio" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is missing or not executable: $PYTHON_BIN" >&2
  echo "Create the venv first: python3 -m venv .venv && .venv/bin/python -m pip install -e '.[vision]'" >&2
  exit 2
fi

xml_escape() {
  sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g' \
    -e 's/"/\&quot;/g' \
    -e "s/'/\&apos;/g"
}

esc_project_dir="$(printf '%s' "$PROJECT_DIR" | xml_escape)"
esc_env_file="$(printf '%s' "$ENV_FILE" | xml_escape)"
esc_run_dir="$(printf '%s' "$RUN_DIR" | xml_escape)"
esc_python_bin="$(printf '%s' "$PYTHON_BIN" | xml_escape)"
esc_provider="$(printf '%s' "$PROVIDER" | xml_escape)"
esc_workers="$(printf '%s' "$WORKERS" | xml_escape)"
esc_limit="$(printf '%s' "$LIMIT" | xml_escape)"
esc_threshold="$(printf '%s' "$THRESHOLD" | xml_escape)"
esc_timeout="$(printf '%s' "$TIMEOUT" | xml_escape)"
esc_extra_args="$(printf '%s' "$EXTRA_ARGS" | xml_escape)"
esc_notify="$(printf '%s' "$NOTIFY" | xml_escape)"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
      <string>$RUNNER</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
      <key>PROJECT_DIR</key>
      <string>$esc_project_dir</string>
      <key>PAPERLESS_AI_ENV_FILE</key>
      <string>$esc_env_file</string>
      <key>PAPERLESS_AI_RUN_DIR</key>
      <string>$esc_run_dir</string>
      <key>PAPERLESS_AI_PYTHON</key>
      <string>$esc_python_bin</string>
      <key>PAPERLESS_AI_PROVIDER</key>
      <string>$esc_provider</string>
      <key>PAPERLESS_AI_WORKERS</key>
      <string>$esc_workers</string>
      <key>PAPERLESS_AI_LIMIT</key>
      <string>$esc_limit</string>
      <key>PAPERLESS_AI_THRESHOLD</key>
      <string>$esc_threshold</string>
      <key>PAPERLESS_AI_TIMEOUT</key>
      <string>$esc_timeout</string>
      <key>PAPERLESS_AI_EXTRA_ARGS</key>
      <string>$esc_extra_args</string>
      <key>PAPERLESS_AI_NOTIFY</key>
      <string>$esc_notify</string>
    </dict>

    <key>StartInterval</key>
    <integer>$START_INTERVAL</integer>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$esc_run_dir/launchd.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$esc_run_dir/launchd.stderr.log</string>
  </dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"

echo "Installed launchd job: $LABEL"
echo "Plist: $PLIST"
echo "Run dir: $RUN_DIR"
echo "Status: launchctl print gui/$(id -u)/$LABEL"
