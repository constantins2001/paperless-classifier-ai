# Security

This tool handles private document metadata and sends document text to the
configured LM Studio endpoint. The intended setup is a local LM Studio server on
`127.0.0.1`.

## Secrets

Do not commit Paperless API tokens, `.env` files, run logs, or audit outputs.
The repository ignores those files by default. Use `.env.example` as a template
and keep the real values outside Git.

## Private Documents

Audit files can contain document titles, extracted text snippets, dates,
correspondents, and model decisions. Treat `paperless_lmstudio_runs/` as private.

## Safe Operation

The classifier is dry-run by default. Review `summary.md` before applying
changes, or use `--apply-audit` to apply only previously audited
`dry_run_ready` patches.

The tool never deletes Paperless documents. Potential delete candidates are only
listed for manual review.
