# Paperless Classifier AI

Conservative Paperless-ngx Inbox classifier powered by a local LM Studio model.

The tool fetches Inbox documents from Paperless, asks a local OpenAI-compatible
LM Studio server to classify metadata, writes auditable dry-run output, and only
removes the Inbox tag when a classification passes validation.

## Highlights

- Dry-run by default
- No document deletion, ever
- `--apply-audit` applies checked dry-run patches without reclassifying
- Confidence and `needs_review` gates
- JSONL and Markdown audit output for every run
- Resume support for long local-model jobs
- Optional deterministic rules for repetitive receipts
- Optional image thumbnail input for multimodal local models
- macOS battery safety: pauses on battery power by default
- Zero runtime dependencies beyond Python standard library

## Requirements

- Python 3.11+
- Paperless-ngx API token
- LM Studio with an OpenAI-compatible local server
- A local instruction model, for example `gemma-4-31b-it`

## Installation

Clone the repository:

```bash
git clone https://github.com/constantins2001/paperless-classifier-ai.git
cd paperless-classifier-ai
```

Run directly:

```bash
python3 paperless_lmstudio_classifier.py --self-test
```

Or install an editable CLI:

```bash
python3 -m pip install -e .
paperless-classifier-ai --self-test
```

## Configuration

Start LM Studio, load your model, and enable the local server. The default
endpoint is:

```text
http://127.0.0.1:1234/v1
```

Create your environment from the example:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env`:

```bash
PAPERLESS_URL=https://paperless.example.com
PAPERLESS_TOKEN=replace-me
LMSTUDIO_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=gemma-4-31b-it
```

The script automatically loads `.env` from the current working directory and
does not overwrite variables that are already present in the environment. It
intentionally does not store API keys anywhere else.

## Safe Dry Run

```bash
python3 paperless_lmstudio_classifier.py --limit 5
```

This fetches Inbox documents, classifies them, and writes:

- `paperless_lmstudio_runs/<timestamp>/audit.jsonl`
- `paperless_lmstudio_runs/<timestamp>/summary.md`

No Paperless records are changed unless `--apply` or `--apply-audit` is present.

## Recommended Long-Run Workflow

Run a full resumable dry-run:

```bash
python3 -u paperless_lmstudio_classifier.py \
  --limit 0 \
  --threshold 0.86 \
  --rules-first \
  --drop-bulk-unclassified \
  --resume \
  --output-dir paperless_lmstudio_runs/full-dryrun
```

Inspect the summary:

```bash
less paperless_lmstudio_runs/full-dryrun/summary.md
```

Apply the exact audited patches without rerunning the model:

```bash
python3 paperless_lmstudio_classifier.py \
  --apply-audit paperless_lmstudio_runs/full-dryrun/audit.jsonl \
  --threshold 0.86
```

`--apply-audit` only applies records that were `dry_run_ready`, still pass the
confidence and review checks, and still have the Inbox tag.

## Direct Apply Mode

For small daily batches, you can classify and apply in one run:

```bash
python3 paperless_lmstudio_classifier.py \
  --limit 10 \
  --threshold 0.86 \
  --rules-first \
  --drop-bulk-unclassified \
  --apply
```

The script applies only classifications that:

- have valid correspondent, document type, created date, title, and tags
- are not marked `needs_review`
- are not delete candidates
- meet the confidence threshold

When applying, it patches metadata and removes the Inbox tag in the same
Paperless update.

## Useful Modes

Classify specific documents:

```bash
python3 paperless_lmstudio_classifier.py --id 7823 --id 7824
```

Filter Paperless results:

```bash
python3 paperless_lmstudio_classifier.py --query Cloudflare --limit 10
```

Use thumbnails for multimodal local models:

```bash
python3 paperless_lmstudio_classifier.py --vision --limit 5
```

Use JSON-schema constrained output when LM Studio is loaded with a larger
context window:

```bash
python3 paperless_lmstudio_classifier.py \
  --response-format json_schema \
  --content-chars 8000
```

Use deterministic rules before LM Studio for repetitive vendors:

```bash
python3 paperless_lmstudio_classifier.py --rules-first --query REWE --limit 100
```

Allow creation of missing Paperless metadata:

```bash
python3 paperless_lmstudio_classifier.py \
  --create-correspondents \
  --create-document-types \
  --limit 10
```

## Battery Safety

On macOS, the script checks `pmset -g batt` before each document. By default it
pauses while the notebook is on battery power and resumes automatically when AC
power is connected.

Override the safety gate:

```bash
python3 paperless_lmstudio_classifier.py --allow-battery --limit 10
```

Change the wait interval:

```bash
python3 paperless_lmstudio_classifier.py --power-check-interval 60 --limit 10
```

On systems where `pmset` is not available, the power source is treated as
unknown and processing continues.

## Nightly Cron

Cron works best when LM Studio is already running with the model loaded and the
local server enabled.

Create a private environment file outside the repository:

```bash
cat > "$HOME/.paperless-classifier-ai.env" <<'EOF'
export PAPERLESS_URL="https://paperless.example.com"
export PAPERLESS_TOKEN="replace-me"
export LMSTUDIO_URL="http://127.0.0.1:1234/v1"
export LMSTUDIO_MODEL="gemma-4-31b-it"
EOF
chmod 600 "$HOME/.paperless-classifier-ai.env"
```

Edit cron:

```bash
crontab -e
```

Review-first nightly dry run:

```cron
SHELL=/bin/zsh
PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

0 2 * * * cd "$HOME/WebstormProjects/paperless-classifier-ai" && mkdir -p paperless_lmstudio_runs/nightly-dryrun && . "$HOME/.paperless-classifier-ai.env" && python3 -u paperless_lmstudio_classifier.py --limit 0 --threshold 0.86 --rules-first --drop-bulk-unclassified --resume --output-dir paperless_lmstudio_runs/nightly-dryrun >> paperless_lmstudio_runs/nightly-dryrun/cron.log 2>&1
```

After reviewing `paperless_lmstudio_runs/nightly-dryrun/summary.md`, apply the
audited patches:

```bash
cd "$HOME/WebstormProjects/paperless-classifier-ai"
. "$HOME/.paperless-classifier-ai.env"
python3 paperless_lmstudio_classifier.py \
  --apply-audit paperless_lmstudio_runs/nightly-dryrun/audit.jsonl \
  --threshold 0.86
```

Fully automatic nightly apply for small daily batches:

```cron
SHELL=/bin/zsh
PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

0 2 * * * cd "$HOME/WebstormProjects/paperless-classifier-ai" && mkdir -p paperless_lmstudio_runs/nightly-apply && . "$HOME/.paperless-classifier-ai.env" && python3 -u paperless_lmstudio_classifier.py --limit 25 --threshold 0.86 --rules-first --drop-bulk-unclassified --apply --output-dir "paperless_lmstudio_runs/nightly-apply/$(date +\%Y\%m\%d-\%H\%M\%S)" >> paperless_lmstudio_runs/nightly-apply/cron.log 2>&1
```

Use `--allow-battery` in cron only if you explicitly want processing while
unplugged.

## Audit Files

`audit.jsonl` is the machine-readable record of every decision. `summary.md` is
the human review view. Both may contain private document metadata and are
ignored by Git.

Potential delete candidates are listed in the summary only. The script never
deletes Paperless documents.

## Development

Run local checks:

```bash
python3 paperless_lmstudio_classifier.py --self-test
python3 -m py_compile paperless_lmstudio_classifier.py
```

## Tuning Notes

Defaults are tuned for an LM Studio model loaded with a 4096-token context:
compact prompts, `--response-format text`, and `--content-chars 2500`. If you
load the model with a larger context window, raise `--content-chars` and consider
`--response-format json_schema`.
