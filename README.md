# Paperless Classifier AI

Conservative Paperless-ngx Inbox classifier powered by an OpenAI-compatible
vision model.

The tool fetches Inbox documents from Paperless, asks a local OpenAI-compatible
LM Studio server or hosted provider such as OpenRouter to classify metadata,
writes auditable dry-run output, and only removes the Inbox tag when a
classification passes validation.

## Highlights

- Dry-run by default
- No document deletion, ever
- `--apply-audit` applies checked dry-run patches without reclassifying
- Confidence and `needs_review` gates
- JSONL and Markdown audit output for every run
- Resume support for long local-model jobs
- Optional deterministic rules for repetitive receipts
- Vision input is on by default for multimodal local models
- All-page PDF rendering when the context budget allows it
- Image-first policy: when all pages fit, Paperless OCR text is omitted
- macOS battery safety: pauses on battery power by default
- Core uses Python standard library; all-page PDF vision uses optional PyMuPDF

## Requirements

- Python 3.11+
- Paperless-ngx API token
- LM Studio with an OpenAI-compatible local server, or OpenRouter
- A vision-capable instruction model, for example `gemma-4-31b-it` locally or `google/gemma-4-31b-it` on OpenRouter
- PyMuPDF for all-page PDF rendering

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

Install an editable CLI with PDF vision support:

```bash
python3 -m pip install -e ".[vision]"
paperless-classifier-ai --self-test
```

Without the `vision` extra, PDFs fall back to the Paperless thumbnail and are
held for review by default.

## Configuration

For LM Studio, load your model and enable the local server. The default endpoint
is:

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
LLM_PROVIDER=lmstudio
LLM_URL=http://127.0.0.1:1234/v1
LLM_MODEL=gemma-4-31b-it
LLM_CONTEXT_WINDOW=8096
```

The script automatically loads `.env` from the current working directory and
does not overwrite variables that are already present in the environment. It
intentionally does not store API keys anywhere else.

For OpenRouter bulk runs, use a private ignored env file instead of committing
secrets:

```bash
cat > .env.openrouter <<'EOF'
export PAPERLESS_URL="https://paperless.example.com"
export PAPERLESS_TOKEN="replace-me"
export LLM_PROVIDER="openrouter"
export LLM_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="google/gemma-4-31b-it"
export LLM_CONTEXT_WINDOW="8096"
export OPENROUTER_API_KEY="replace-me"
export OPENROUTER_PDF_ENGINE="mistral-ocr"
export OPENROUTER_APP_NAME="paperless-classifier-ai"
EOF
chmod 600 .env.openrouter
```

When `--provider openrouter` is used, PDF inputs default to OpenRouter's `file`
content type with the `file-parser` plugin and `mistral-ocr` engine. That means
PDFs are sent as PDFs, not as local screenshots, and the model receives
OpenRouter's parsed/OCRed document content. Use `--pdf-input rendered-images` if
you want the local page-rendering path instead.

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
  --context-window 8096 \
  --workers 4 \
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
confidence and review checks, and still have the Inbox tag. With default vision
enabled, it also requires the dry-run audit record to contain acceptable vision
evidence; old text-only audits are skipped unless you explicitly pass
`--no-vision`.

`--workers` controls parallelism. Use `1` for strictly serial local runs. For a
hosted provider such as OpenRouter, start with `--workers 3` or `--workers 4`
and increase only if the provider rate limits and Paperless stay comfortable.
Parallel classification cannot be combined with automatic creation of missing
correspondents or document types.

## Parallel Bulk And Metadata Creation

For the initial Inbox cleanup, use parallel dry-run classification with existing
Paperless metadata only:

```bash
python3 -u paperless_lmstudio_classifier.py \
  --provider openrouter \
  --limit 0 \
  --workers 32 \
  --threshold 0.86 \
  --rules-first \
  --drop-bulk-unclassified \
  --resume \
  --output-dir paperless_lmstudio_runs/openrouter-bulk-vision-001
```

That pass will not create correspondents or document types. If the model needs a
new correspondent or document type, the document is held for review because the
resource does not exist yet.

After reviewing the audit, run a serial creation/apply pass. Do not reuse the
same output directory with `--resume`, because the first pass has already
recorded those review records and resume would skip them. Use a new output
directory, or pass explicit `--id` values for the reviewed documents:

```bash
python3 -u paperless_lmstudio_classifier.py \
  --provider openrouter \
  --workers 1 \
  --create-correspondents \
  --create-document-types \
  --threshold 0.86 \
  --rules-first \
  --drop-bulk-unclassified \
  --apply \
  --output-dir paperless_lmstudio_runs/openrouter-serial-create-001
```

The serial creation pass can create missing correspondents/document types and
then apply the document patch. Creation only happens during `--apply`; dry-run
mode never mutates Paperless metadata.

For documents that were already `dry_run_ready` in the bulk audit and do not
need new metadata, use `--apply-audit` instead of reclassifying:

```bash
python3 paperless_lmstudio_classifier.py \
  --apply-audit paperless_lmstudio_runs/openrouter-bulk-vision-001/audit.jsonl \
  --threshold 0.86 \
  --workers 4
```

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

Disable vision for text-only models:

```bash
python3 paperless_lmstudio_classifier.py --no-vision --limit 5
```

Use a larger context window and JSON-schema constrained output:

```bash
python3 paperless_lmstudio_classifier.py \
  --context-window 16384 \
  --response-format json_schema \
  --content-chars 8000
```

Use deterministic rules before LM Studio for repetitive vendors:

```bash
python3 paperless_lmstudio_classifier.py --rules-first --query REWE --limit 100
```

Allow creation of missing Paperless metadata during apply:

```bash
python3 paperless_lmstudio_classifier.py \
  --create-correspondents \
  --create-document-types \
  --apply \
  --limit 10
```

## Vision And OCR Policy

Vision is enabled by default. The script fetches the Paperless preview for each
document. If it is a PDF and PyMuPDF is installed, it renders page images and
sends as many pages as fit inside the configured context window. If every page
fits, the Paperless OCR text is omitted and the model is instructed to read the
images directly. This uses the local model as the effective OCR source for that
classification.

If not every page fits, the script sends representative pages across the
document and includes a Paperless OCR excerpt as fallback context. Partial vision
is held for review by default; use `--allow-partial-vision` only if you accept
applying classifications where not every page image was seen.

The page budget is estimated from:

- `--context-window`
- `--max-tokens`
- `--context-safety-tokens`
- the prompt size
- `--image-token-estimate`
- optional `--max-vision-pages`

Use these controls when you change the LM Studio context window:

```bash
python3 paperless_lmstudio_classifier.py \
  --context-window 8096 \
  --image-token-estimate 768 \
  --context-safety-tokens 512 \
  --limit 10
```

Paperless preview PDFs require PyMuPDF:

```bash
python3 -m pip install -e ".[vision]"
```

## Flag Reference

Connection and model:

- `--paperless-url`: Paperless-ngx base URL. Defaults to `PAPERLESS_URL`.
- `--paperless-token`: Paperless API token. Defaults to `PAPERLESS_TOKEN`.
- `--provider`: LLM provider preset: `lmstudio`, `openrouter`, or `openai-compatible`.
- `--llm-url` / `--lmstudio-url`: OpenAI-compatible API base. Defaults to `LLM_URL`, provider-specific env values, or the local LM Studio URL.
- `--model`: Model name. Defaults to `LLM_MODEL`, provider-specific env values, `gemma-4-31b-it`, or `google/gemma-4-31b-it` for OpenRouter.
- `--llm-api-key`: Bearer key for hosted providers. Defaults to `LLM_API_KEY` or `OPENROUTER_API_KEY`.
- `--openrouter-site-url`: Optional OpenRouter `HTTP-Referer` header.
- `--openrouter-app-name`: Optional OpenRouter `X-Title` header.
- `--context-window`: LLM context window used for page budgeting. Defaults to `LLM_CONTEXT_WINDOW`, `LMSTUDIO_CONTEXT_WINDOW`, or `8096`.

Document selection:

- `--label`: Inbox tag name fallback when Paperless does not expose `is_inbox_tag`.
- `--limit`: Maximum documents to process. `0` means all matching documents.
- `--id`: Process one document ID. Repeat for multiple IDs.
- `--query`: Paperless full-text search filter.
- `--page-size`: Paperless API page size.
- `--ordering`: Paperless document ordering, for example `-created`.
- `--resume`: Skip documents that already have terminal records in the chosen `audit.jsonl`.

Apply behavior:

- `--apply`: Patch Paperless metadata and remove the Inbox tag. Without it, the run is dry-run only.
- `--apply-audit PATH`: Apply exact `dry_run_ready` patches from an existing `audit.jsonl` without rerunning the model.
- `--force`: Override confidence, review, and safety gates. This is intentionally sharp.
- `--threshold`: Minimum confidence required for apply.

Vision and text:

- `--vision` / `--no-vision`: Enable or disable image input. Vision is enabled by default.
- `--vision-dpi`: DPI for rendering Paperless preview PDFs into page images.
- `--max-vision-pages`: Hard cap on rendered pages. `0` means only the context budget decides.
- `--allow-partial-vision`: Permit apply when not all pages were sent as images.
- `--pdf-input auto|rendered-images|openrouter-file`: In `auto`, OpenRouter gets PDFs as PDF file inputs, while local providers get rendered page images.
- `--openrouter-pdf-engine mistral-ocr|cloudflare-ai|native|default`: PDF parser engine for OpenRouter file input. `mistral-ocr` is best for scanned documents; `default` lets OpenRouter choose.
- `--ocr-source auto|always|never`: In `auto`, Paperless OCR is omitted when all pages fit as images and included only as fallback when they do not. `always` includes Paperless OCR. `never` omits it.
- `--content-chars`: Paperless OCR characters sent when OCR fallback is used.
- `--image-token-estimate`: Estimated context tokens consumed by each page image.
- `--context-safety-tokens`: Reserved context tokens for estimator error and protocol overhead.

Classification tuning:

- `--temperature`: LLM sampling temperature.
- `--max-tokens`: Maximum response tokens from the LLM.
- `--response-format text|json_schema`: Use `json_schema` only when your provider, model, and context window handle it reliably.
- `--rules-first`: Use deterministic rules before the LLM. Currently this is useful for repetitive REWE eBon receipts.
- `--replace-tags`: Replace non-Inbox tags with model-selected tags instead of preserving existing non-Inbox tags.
- `--drop-bulk-unclassified`: Remove the `Bulk Unclassified` tag from the final tag set after a successful classification. This is only tag cleanup; it does not delete documents.
- `--create-correspondents`: During `--apply`, create missing Paperless correspondents when the model proposes one. Requires `--workers 1`.
- `--create-document-types`: During `--apply`, create missing Paperless document types when the model proposes one. Requires `--workers 1`.
- `--email-date-drift-review-days`: Hold email terms/conditions documents for review when the chosen date differs too far from the email/document date.

Runtime:

- `--timeout`: HTTP timeout in seconds.
- `--retries`: Retries for transient LLM errors.
- `--retry-sleep`: Base sleep between LLM retries.
- `--workers`: Documents to process concurrently. Defaults to `PAPERLESS_AI_WORKERS` or `1`.
- `--sleep`: Delay between documents.
- `--allow-battery`: Do not pause on macOS battery power.
- `--power-check-interval`: Recheck interval while paused on battery.
- `--output-dir`: Run artifact directory.
- `--self-test`: Run local parser sanity checks.
- `--version`: Print the CLI version.

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
export LMSTUDIO_CONTEXT_WINDOW="8096"
export PAPERLESS_AI_WORKERS="1"
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

Defaults are tuned for an LM Studio model loaded with an `8096` token context:
compact prompts, `--response-format text`, image-first OCR replacement when all
pages fit, and Paperless OCR fallback only when needed. If you load the model
with a larger context window, raise `--context-window` first. Then consider
raising `--content-chars`, lowering `--image-token-estimate` if your backend
accounts images cheaply, or trying `--response-format json_schema`.
