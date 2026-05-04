#!/usr/bin/env python3
"""Classify Paperless Inbox documents with an OpenAI-compatible vision model.

The script is intentionally conservative:
- It defaults to dry-run mode.
- It never deletes documents.
- It removes the Inbox tag only when a valid, confident classification is applied.
- It writes JSONL and Markdown audit files for every run.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


__version__ = "0.5.2"
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234/v1"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "gemma-4-31b-it"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it"
DEFAULT_OUTPUT_ROOT = "paperless_lmstudio_runs"
DEFAULT_THRESHOLD = 0.86
DEFAULT_CONTEXT_WINDOW = 8096
DEFAULT_CONTEXT_SAFETY_TOKENS = 512
DEFAULT_IMAGE_TOKEN_ESTIMATE = 768
DEFAULT_VISION_DPI = 120
DEFAULT_WORKERS = 1
MAX_TITLE_LEN = 128
# Terminal statuses for the local audit ledger. `updated_kept_inbox` deliberately
# stays here: the document still has Inbox in Paperless, but the local run has
# already applied the best available metadata and left it for human review.
RESUME_SKIP_STATUSES = {
    "dry_run_ready",
    "updated",
    "updated_kept_inbox",
    "skipped_already_not_in_inbox",
    "skipped_delete_candidate",
    "skipped_needs_review",
    "skipped_unreadable",
}
TERMINAL_ERROR_MARKERS = (
    "document closed or encrypted",
)


class ClassifierError(RuntimeError):
    pass


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def utc_today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def normalize_name(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = value[: int(limit * 0.72)]
    tail = value[-int(limit * 0.28) :]
    return f"{head}\n\n[... middle truncated by classifier ...]\n\n{tail}"


def data_url(content_type: str, raw: bytes) -> str:
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def estimate_tokens_from_text(value: str) -> int:
    # Good enough for local context budgeting without importing a tokenizer.
    return math.ceil(len(value) / 4)


def positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def select_representative_pages(total_pages: int, max_pages: int) -> list[int]:
    """Return zero-based page indexes, preferring coverage over contiguity."""
    if total_pages <= 0 or max_pages <= 0:
        return []
    if max_pages >= total_pages:
        return list(range(total_pages))
    if max_pages == 1:
        return [0]

    selected = {
        round(position * (total_pages - 1) / (max_pages - 1))
        for position in range(max_pages)
    }
    candidate = 0
    while len(selected) < max_pages and candidate < total_pages:
        selected.add(candidate)
        candidate += 1
    return sorted(selected)[:max_pages]


def vision_page_budget(
    args: argparse.Namespace,
    user_text: str,
    total_pages: int,
) -> tuple[int, dict[str, Any]]:
    prompt_tokens = estimate_tokens_from_text(build_system_prompt()) + estimate_tokens_from_text(user_text)
    available = args.context_window - args.max_tokens - args.context_safety_tokens - prompt_tokens
    by_context = max(0, available // max(1, args.image_token_estimate))
    if total_pages > 0 and by_context == 0:
        by_context = 1
    if args.max_vision_pages:
        by_context = min(by_context, args.max_vision_pages)
    selected_count = max(0, min(total_pages, by_context))
    return selected_count, {
        "context_window": args.context_window,
        "context_safety_tokens": args.context_safety_tokens,
        "max_response_tokens": args.max_tokens,
        "estimated_text_tokens": prompt_tokens,
        "estimated_image_tokens_each": args.image_token_estimate,
        "estimated_available_image_tokens": available,
        "max_pages_by_context": by_context,
        "max_vision_pages": args.max_vision_pages,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating markdown fences and leading commentary."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ClassifierError("LM response did not contain a JSON object")

    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : pos + 1])
                if not isinstance(parsed, dict):
                    raise ClassifierError("LM JSON root was not an object")
                return parsed
    raise ClassifierError("LM response contained incomplete JSON")


def is_date(value: str | None) -> bool:
    if not value:
        return False
    try:
        dt.date.fromisoformat(value)
        return True
    except ValueError:
        return False


def parse_iso_date(value: Any) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def current_power_source() -> tuple[str, str]:
    """Return (state, detail) where state is ac, battery, or unknown."""
    pmset = shutil.which("pmset")
    if not pmset:
        return "unknown", "pmset not found"
    try:
        result = subprocess.run(
            [pmset, "-g", "batt"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "unknown", str(exc)
    output = (result.stdout or "") + (result.stderr or "")
    first_line = output.strip().splitlines()[0] if output.strip() else "empty pmset output"
    if "AC Power" in output:
        return "ac", first_line
    if "Battery Power" in output:
        return "battery", first_line
    return "unknown", first_line


def wait_for_ac_power(args: argparse.Namespace) -> None:
    if args.allow_battery:
        return
    while True:
        state, detail = current_power_source()
        if state == "ac":
            if getattr(args, "_power_pause_logged", False):
                print(f"Power: AC detected, resuming. {detail}", flush=True)
                args._power_pause_logged = False
            return
        if state == "unknown":
            if not getattr(args, "_power_unknown_logged", False):
                print(f"Power: unable to determine AC/battery state, continuing. {detail}", flush=True)
                args._power_unknown_logged = True
            return

        print(
            f"Power: battery detected, pausing. {detail}. "
            f"Rechecking in {args.power_check_interval:.0f}s. "
            "Use --allow-battery to override.",
            flush=True,
        )
        args._power_pause_logged = True
        time.sleep(args.power_check_interval)


def safe_title(value: Any, fallback: str) -> str:
    title = str(value or "").strip()
    if not title:
        title = fallback.strip() or "Untitled document"
    title = re.sub(r"\s+", " ", title)
    return title[:MAX_TITLE_LEN]


def classification_safety_warnings(
    doc: dict[str, Any],
    classification: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    warnings: list[str] = []
    content = str(doc.get("content") or "")
    title = str(classification.get("title") or "")
    correspondent = str(
        classification.get("correspondent_name")
        or (classification.get("correspondent") or {}).get("name")
        or ""
    )

    if re.search(r"From:\s*Clockify\b", content, re.IGNORECASE) and normalize_name(correspondent) not in {
        "clockify",
        "cake.com",
        "cake.com inc",
    }:
        warnings.append(
            "email sender is Clockify but selected correspondent is not Clockify/CAKE.com"
        )

    doc_created = parse_iso_date(doc.get("created"))
    classified_created = parse_iso_date(classification.get("created"))
    if doc.get("mime_type") == "message/rfc822" and doc_created and classified_created:
        drift_days = abs((classified_created - doc_created).days)
        terms_text = f"{title} {content[:2500]}".casefold()
        has_terms_language = any(
            marker in terms_text
            for marker in [
                "agb",
                "bedingungen",
                "datenschutz",
                "reparaturbedingungen",
                "terms",
                "widerruf",
            ]
        )
        if has_terms_language and drift_days > args.email_date_drift_review_days:
            warnings.append(
                "email terms/conditions date differs significantly from the email/document date"
            )

    return warnings


@dataclass(frozen=True)
class Resource:
    id: int
    name: str
    slug: str | None = None
    is_inbox_tag: bool = False


@dataclass(frozen=True)
class VisionImage:
    page_number: int
    data_url: str


@dataclass(frozen=True)
class VisionFile:
    filename: str
    data_url: str


class ResourceCatalog:
    def __init__(
        self,
        correspondents: list[dict[str, Any]],
        document_types: list[dict[str, Any]],
        tags: list[dict[str, Any]],
    ) -> None:
        self.correspondents = self._build(correspondents)
        self.document_types = self._build(document_types)
        self.tags = self._build(tags, include_inbox=True)
        self.correspondents_by_name = self._by_name(self.correspondents)
        self.document_types_by_name = self._by_name(self.document_types)
        self.tags_by_name = self._by_name(self.tags)
        self.tags_by_slug = {
            normalize_name(r.slug or ""): r for r in self.tags.values() if r.slug
        }

    @staticmethod
    def _build(values: list[dict[str, Any]], include_inbox: bool = False) -> dict[int, Resource]:
        out: dict[int, Resource] = {}
        for value in values:
            try:
                rid = int(value["id"])
            except (KeyError, TypeError, ValueError):
                continue
            out[rid] = Resource(
                id=rid,
                name=str(value.get("name") or ""),
                slug=value.get("slug"),
                is_inbox_tag=bool(value.get("is_inbox_tag")) if include_inbox else False,
            )
        return out

    @staticmethod
    def _by_name(values: dict[int, Resource]) -> dict[str, Resource]:
        return {normalize_name(r.name): r for r in values.values()}

    def find_tag(self, name: str) -> Resource | None:
        return self.tags_by_name.get(normalize_name(name)) or self.tags_by_slug.get(
            normalize_name(name)
        )

    def email_attachment_tag_id(self) -> int | None:
        tag = self.find_tag("Email Attachment")
        return tag.id if tag else None

    def bulk_unclassified_tag_id(self) -> int | None:
        tag = self.find_tag("Bulk Unclassified")
        return tag.id if tag else None

    def inbox_tag(self, requested_name: str) -> Resource:
        for tag in self.tags.values():
            if tag.is_inbox_tag:
                return tag
        tag = self.find_tag(requested_name)
        if not tag:
            raise ClassifierError(f"Could not find Inbox tag named {requested_name!r}")
        return tag

    def prompt_payload(self) -> dict[str, Any]:
        def slim(values: dict[int, Resource]) -> list[str]:
            return [f"{r.id}:{r.name}" for r in sorted(values.values(), key=lambda r: r.name.casefold())]

        return {
            "correspondents": slim(self.correspondents),
            "document_types": slim(self.document_types),
            "tags": slim(self.tags),
        }


class JsonHttpClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        body = None
        req_headers = dict(headers or {})
        if data is not None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ClassifierError(f"HTTP {exc.code} for {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ClassifierError(f"Could not reach {url}: {exc.reason}") from exc
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def request_bytes(self, method: str, url: str, headers: dict[str, str] | None = None) -> tuple[bytes, str]:
        req = urllib.request.Request(url, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                return response.read(), content_type
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ClassifierError(f"HTTP {exc.code} for {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ClassifierError(f"Could not reach {url}: {exc.reason}") from exc


class PaperlessClient:
    def __init__(self, base_url: str, token: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Token {token}"}
        self.http = JsonHttpClient(timeout)

    def url(self, path: str, params: dict[str, Any] | None = None) -> str:
        path = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        return url

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.http.request_json("GET", self.url(path, params), self.headers)

    def patch(self, path: str, data: dict[str, Any]) -> Any:
        return self.http.request_json("PATCH", self.url(path), self.headers, data)

    def post(self, path: str, data: dict[str, Any]) -> Any:
        return self.http.request_json("POST", self.url(path), self.headers, data)

    def bytes(self, path: str) -> tuple[bytes, str]:
        return self.http.request_bytes("GET", self.url(path), self.headers)

    def paginated(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = dict(params or {})
        params.setdefault("page_size", 100)
        page = int(params.get("page", 1))
        results: list[dict[str, Any]] = []
        while True:
            params["page"] = page
            payload = self.get(path, params)
            batch = payload.get("results", [])
            results.extend(batch)
            if not payload.get("next") or not batch:
                return results
            page += 1

    def catalog(self) -> ResourceCatalog:
        correspondents = self.paginated("/api/correspondents/", {"page_size": 1000})
        document_types = self.paginated("/api/document_types/", {"page_size": 1000})
        tags = self.paginated("/api/tags/", {"page_size": 1000})
        return ResourceCatalog(correspondents, document_types, tags)

    def inbox_ids(
        self,
        inbox_tag_id: int,
        page_size: int,
        ordering: str,
        query: str | None,
        limit: int | None,
    ) -> list[int]:
        params: dict[str, Any] = {
            "tags__id__all": inbox_tag_id,
            "page_size": page_size,
            "ordering": ordering,
        }
        if query:
            params["query"] = query
        first = self.get("/api/documents/", params)
        if isinstance(first.get("all"), list):
            ids = [int(value) for value in first["all"]]
        else:
            ids = [int(doc["id"]) for doc in first.get("results", [])]
            page = 2
            while first.get("next"):
                params["page"] = page
                first = self.get("/api/documents/", params)
                ids.extend(int(doc["id"]) for doc in first.get("results", []))
                page += 1
        return ids[:limit] if limit else ids

    def document(self, doc_id: int) -> dict[str, Any]:
        return self.get(f"/api/documents/{doc_id}/")

    def thumbnail_data_url(self, doc_id: int) -> str:
        raw, content_type = self.bytes(f"/api/documents/{doc_id}/thumb/")
        return data_url(content_type, raw)

    def preview_bytes(self, doc_id: int) -> tuple[bytes, str]:
        return self.bytes(f"/api/documents/{doc_id}/preview/")


class LLMClient:
    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
        openrouter_site_url: str | None,
        openrouter_app_name: str | None,
        timeout: float,
        temperature: float,
        max_tokens: int,
        response_format: str,
        retries: int,
        retry_sleep: float,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.http = JsonHttpClient(timeout)
        self.headers: dict[str, str] = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if provider == "openrouter":
            if openrouter_site_url:
                self.headers["HTTP-Referer"] = openrouter_site_url
            if openrouter_app_name:
                self.headers["X-Title"] = openrouter_app_name

    def classify(
        self,
        messages: list[dict[str, Any]],
        plugins: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if plugins:
            payload["plugins"] = plugins
        if self.response_format == "json_schema":
            payload["response_format"] = classification_response_format()
        elif self.response_format == "text":
            pass
        else:
            raise ClassifierError(f"Unsupported response format: {self.response_format}")
        last_error: ClassifierError | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self.http.request_json(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    data=payload,
                )
                break
            except ClassifierError as exc:
                last_error = exc
                message = str(exc)
                retryable = any(
                    marker in message
                    for marker in ["Model reloaded", "temporarily unavailable", "Connection refused"]
                )
                if not retryable or attempt >= self.retries:
                    raise
                time.sleep(self.retry_sleep * (attempt + 1))
        else:
            raise last_error or ClassifierError("LLM request failed")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ClassifierError(f"Unexpected LLM response: {data}") from exc
        return parse_json_object(content), content


def classification_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "paperless_classification",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "needs_review": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "correspondent": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": ["integer", "null"]},
                            "name": {"type": ["string", "null"]},
                            "create": {"type": "boolean"},
                        },
                        "required": ["id", "name", "create"],
                    },
                    "document_type": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": ["integer", "null"]},
                            "name": {"type": ["string", "null"]},
                            "create": {"type": "boolean"},
                        },
                        "required": ["id", "name", "create"],
                    },
                    "created": {"type": ["string", "null"]},
                    "title": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                            },
                            "required": ["id", "name"],
                        },
                    },
                    "suggested_new_tags": {"type": "array", "items": {"type": "string"}},
                    "delete_candidate": {"type": "boolean"},
                    "delete_reason": {"type": ["string", "null"]},
                },
                "required": [
                    "confidence",
                    "needs_review",
                    "reason",
                    "correspondent",
                    "document_type",
                    "created",
                    "title",
                    "tags",
                    "suggested_new_tags",
                    "delete_candidate",
                    "delete_reason",
                ],
            },
        },
    }


def build_system_prompt() -> str:
    return """Classify Paperless-ngx documents. Return one strict JSON object and no prose.

Rules: When page images are attached, read the images directly and treat visual text as the source of truth. Paperless OCR content may be omitted, partial, or only a fallback. company=sender/issuer/merchant/court/employer/provider, not recipient or employee. For work timesheets use the employer/client company, not the worker name. created=document issue/signature/submission/transaction/letter date, not import date. If a document covers a period/year, put that period in the title; do not use period end such as Dec 31 as created when a signing/issue/submission date is visible. Use existing IDs when possible. Preserve an existing correspondent/type when it is semantically compatible. Invoice/receipt/eBon/bill => Rechnung if available. Tags must be existing IDs, but never include Inbox. Keep Email Attachment for emails. If no existing company/type fits, id null + create true. Never delete; mark delete_candidate only. needs_review true if weak/ambiguous/missing IDs. Confidence means safe to apply.

Required JSON shape:
{
  "confidence": 0.0,
  "needs_review": true,
  "reason": "short reason",
  "correspondent": {"id": null, "name": "company/person", "create": false},
  "document_type": {"id": null, "name": "type", "create": false},
  "created": "YYYY-MM-DD",
  "title": "archive title, max 128 chars",
  "tags": [{"id": 0, "name": "tag"}],
  "suggested_new_tags": [],
  "delete_candidate": false,
  "delete_reason": null
}"""


def build_user_message(
    doc: dict[str, Any],
    catalog: ResourceCatalog,
    content_chars: int,
    paperless_ocr_content: str | None,
    paperless_ocr_policy: str,
    vision: dict[str, Any] | None = None,
) -> str:
    existing_tag_names = [
        catalog.tags[tag_id].name for tag_id in doc.get("tags", []) if tag_id in catalog.tags
    ]
    existing_correspondent_name = None
    if doc.get("correspondent") in catalog.correspondents:
        existing_correspondent_name = catalog.correspondents[doc["correspondent"]].name
    existing_document_type_name = None
    if doc.get("document_type") in catalog.document_types:
        existing_document_type_name = catalog.document_types[doc["document_type"]].name
    payload = {
        "available_paperless_items": catalog.prompt_payload(),
        "document": {
            "id": doc.get("id"),
            "title": doc.get("title"),
            "created": doc.get("created"),
            "original_file_name": doc.get("original_file_name"),
            "mime_type": doc.get("mime_type"),
            "page_count": doc.get("page_count"),
            "existing_correspondent_id": doc.get("correspondent"),
            "existing_document_type_id": doc.get("document_type"),
            "existing_tag_ids": doc.get("tags", []),
            "existing_tag_names": existing_tag_names,
            "existing_correspondent_name": existing_correspondent_name,
            "existing_document_type_name": existing_document_type_name,
            "paperless_ocr_policy": paperless_ocr_policy,
            "content": truncate_text(paperless_ocr_content or "", content_chars)
            if paperless_ocr_content is not None
            else "",
            "vision": vision or {},
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_messages(
    user_text: str,
    images: list[VisionImage],
    files: list[VisionFile] | None = None,
) -> list[dict[str, Any]]:
    files = files or []
    if images or files:
        content: Any = [{"type": "text", "text": user_text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image.data_url}}
            for image in images
        )
        content.extend(
            {
                "type": "file",
                "file": {
                    "filename": file.filename,
                    "file_data": file.data_url,
                },
            }
            for file in files
        )
    else:
        content = user_text
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": content},
    ]


def should_omit_paperless_ocr(
    doc: dict[str, Any],
    catalog: ResourceCatalog,
    args: argparse.Namespace,
) -> bool:
    if not args.vision:
        return False
    if args.ocr_source == "always":
        return False
    if args.ocr_source == "never":
        return True

    total_pages = positive_int(doc.get("page_count"))
    if not total_pages:
        return False
    metadata_text = build_user_message(
        doc,
        catalog,
        args.content_chars,
        None,
        "omitted_for_context_estimate",
        {"enabled": True, "requested_pages": "all"},
    )
    page_budget, _ = vision_page_budget(args, metadata_text, total_pages)
    return page_budget >= total_pages


def openrouter_pdf_plugins(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.openrouter_pdf_engine == "default":
        return []
    return [
        {
            "id": "file-parser",
            "pdf": {
                "engine": args.openrouter_pdf_engine,
            },
        }
    ]


def build_openrouter_pdf_file_input(
    doc: dict[str, Any],
    paperless: PaperlessClient,
    args: argparse.Namespace,
) -> tuple[list[VisionFile], dict[str, Any], list[dict[str, Any]]]:
    doc_id = int(doc["id"])
    try:
        raw, content_type = paperless.preview_bytes(doc_id)
    except Exception as exc:  # noqa: BLE001 - keep classification auditable.
        return [], {
            "enabled": True,
            "source": "openrouter_pdf_file_unavailable",
            "page_count": positive_int(doc.get("page_count")),
            "included_pages": [],
            "omitted_pages": [],
            "all_pages_included": False,
            "image_count": 0,
            "file_count": 0,
            "warnings": [f"could not fetch Paperless preview for OpenRouter PDF input: {exc}"],
        }, []

    media_type = content_type.split(";", 1)[0].strip().casefold()
    if media_type != "application/pdf" and not raw.startswith(b"%PDF"):
        return [], {
            "enabled": True,
            "source": "openrouter_pdf_file_unavailable",
            "page_count": positive_int(doc.get("page_count")),
            "included_pages": [],
            "omitted_pages": [],
            "all_pages_included": False,
            "image_count": 0,
            "file_count": 0,
            "warnings": [f"Paperless preview content type {content_type!r} is not a PDF"],
        }, []

    page_count = positive_int(doc.get("page_count"))
    included_pages = list(range(1, page_count + 1)) if page_count else []
    filename = str(doc.get("original_file_name") or f"paperless-{doc_id}.pdf")
    if not filename.casefold().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return [VisionFile(filename=filename, data_url=data_url("application/pdf", raw))], {
        "enabled": True,
        "source": "openrouter_pdf_file",
        "renderer": "OpenRouter file-parser",
        "pdf_engine": args.openrouter_pdf_engine,
        "page_count": page_count,
        "included_pages": included_pages,
        "omitted_pages": [],
        "all_pages_included": True,
        "image_count": 0,
        "file_count": 1,
        "warnings": [],
    }, openrouter_pdf_plugins(args)


def render_pdf_vision_images(
    raw: bytes,
    args: argparse.Namespace,
    user_text: str,
) -> tuple[list[VisionImage], dict[str, Any]]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ClassifierError(
            "PDF all-page vision requires PyMuPDF. Install with "
            "`python3 -m pip install '.[vision]'`, or pass --no-vision."
        ) from exc

    try:
        pdf = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - convert renderer failures into audit warnings.
        raise ClassifierError(f"Could not open Paperless preview PDF for vision: {exc}") from exc

    try:
        total_pages = int(pdf.page_count)
        page_budget, budget_info = vision_page_budget(args, user_text, total_pages)
        selected = select_representative_pages(total_pages, page_budget)
        scale = args.vision_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        images: list[VisionImage] = []
        for page_index in selected:
            page = pdf.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(
                VisionImage(
                    page_number=page_index + 1,
                    data_url=data_url("image/png", pixmap.tobytes("png")),
                )
            )
    finally:
        pdf.close()

    included_pages = [image.page_number for image in images]
    warnings: list[str] = []
    if not images:
        warnings.append("vision enabled but context budget did not allow any page images")
    elif len(images) < total_pages:
        warnings.append(
            f"vision included {len(images)} of {total_pages} pages due to context/page budget"
        )

    return images, {
        "enabled": True,
        "source": "paperless_preview_pdf",
        "renderer": "PyMuPDF",
        "dpi": args.vision_dpi,
        "page_count": total_pages,
        "included_pages": included_pages,
        "omitted_pages": [
            page_number
            for page_number in range(1, total_pages + 1)
            if page_number not in set(included_pages)
        ],
        "all_pages_included": len(images) == total_pages,
        "image_count": len(images),
        "budget": budget_info,
        "warnings": warnings,
    }


def thumbnail_fallback_vision(
    paperless: PaperlessClient,
    doc_id: int,
    warning: str,
) -> tuple[list[VisionImage], dict[str, Any]]:
    try:
        thumbnail = paperless.thumbnail_data_url(doc_id)
    except Exception as exc:  # noqa: BLE001 - this becomes an audited review item.
        return [], {
            "enabled": True,
            "source": "unavailable",
            "page_count": None,
            "included_pages": [],
            "omitted_pages": [],
            "all_pages_included": False,
            "image_count": 0,
            "warnings": [warning, f"thumbnail fallback failed: {exc}"],
        }
    return [VisionImage(page_number=1, data_url=thumbnail)], {
        "enabled": True,
        "source": "paperless_thumbnail_fallback",
        "page_count": None,
        "included_pages": [1],
        "omitted_pages": [],
        "all_pages_included": False,
        "image_count": 1,
        "warnings": [warning, "used thumbnail fallback instead of all-page vision"],
    }


def build_vision_inputs(
    doc: dict[str, Any],
    paperless: PaperlessClient,
    args: argparse.Namespace,
    user_text: str,
) -> tuple[list[VisionImage], dict[str, Any]]:
    if not args.vision:
        return [], {"enabled": False, "source": "disabled", "warnings": []}

    doc_id = int(doc["id"])
    try:
        raw, content_type = paperless.preview_bytes(doc_id)
    except Exception as exc:  # noqa: BLE001 - keep classification auditable.
        return thumbnail_fallback_vision(
            paperless,
            doc_id,
            f"could not fetch Paperless preview for all-page vision: {exc}",
        )

    media_type = content_type.split(";", 1)[0].strip().casefold()
    if media_type == "application/pdf" or raw.startswith(b"%PDF"):
        try:
            return render_pdf_vision_images(raw, args, user_text)
        except ClassifierError as exc:
            return thumbnail_fallback_vision(paperless, doc_id, str(exc))

    if media_type.startswith("image/"):
        return [VisionImage(page_number=1, data_url=data_url(media_type, raw))], {
            "enabled": True,
            "source": "paperless_preview_image",
            "page_count": 1,
            "included_pages": [1],
            "omitted_pages": [],
            "all_pages_included": True,
            "image_count": 1,
            "warnings": [],
        }

    return thumbnail_fallback_vision(
        paperless,
        doc_id,
        f"Paperless preview content type {content_type!r} is not directly renderable as page images",
    )


def vision_review_warnings(vision: dict[str, Any] | None, args: argparse.Namespace) -> list[str]:
    if not args.vision:
        return []
    if not isinstance(vision, dict) or not vision.get("enabled"):
        return ["vision enabled but audit/classification has no vision evidence"]
    warnings = list(vision.get("warnings") or [])
    if not vision.get("image_count") and not vision.get("file_count"):
        warnings.append("vision enabled but no images or files were sent")
    if not vision.get("all_pages_included") and not args.allow_partial_vision:
        warnings.append("all-page vision was not available")
    return sorted(set(str(warning) for warning in warnings if warning))


def nested_resource(raw: dict[str, Any], key: str) -> tuple[int | None, str | None, bool]:
    value = raw.get(key)
    if isinstance(value, dict):
        rid = value.get("id")
        name = value.get("name")
        create = bool(value.get("create"))
    else:
        rid = raw.get(f"{key}_id")
        name = raw.get(f"{key}_name")
        create = bool(raw.get(f"create_{key}"))
    try:
        rid_int = int(rid) if rid is not None else None
    except (TypeError, ValueError):
        rid_int = None
    return rid_int, str(name).strip() if name else None, create


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1"}
    return bool(value)


def coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def resolve_resource(
    rid: int | None,
    name: str | None,
    create: bool,
    values: dict[int, Resource],
    by_name: dict[str, Resource],
) -> tuple[int | None, str | None, bool, str | None]:
    if rid in values:
        return rid, values[rid].name, False, None
    if name:
        existing = by_name.get(normalize_name(name))
        if existing:
            return existing.id, existing.name, False, None
    if name and create:
        return None, name, True, None
    if rid is not None:
        return None, name, create, f"unknown id {rid}"
    return None, name, create, "missing resource"


def resolve_tags(raw: Any, catalog: ResourceCatalog) -> tuple[list[int], list[str]]:
    tag_ids: list[int] = []
    warnings: list[str] = []
    if not isinstance(raw, list):
        return tag_ids, ["tags was not a list"]
    for item in raw:
        rid: int | None = None
        name: str | None = None
        if isinstance(item, dict):
            try:
                rid = int(item["id"]) if item.get("id") is not None else None
            except (TypeError, ValueError):
                rid = None
            name = str(item.get("name") or "").strip() or None
        else:
            try:
                rid = int(item)
            except (TypeError, ValueError):
                name = str(item).strip()
        if rid in catalog.tags:
            tag_ids.append(int(rid))
            continue
        if name:
            found = catalog.find_tag(name)
            if found:
                tag_ids.append(found.id)
                continue
        warnings.append(f"unknown tag {item!r}")
    return sorted(set(tag_ids)), warnings


def maybe_rule_classify(doc: dict[str, Any], catalog: ResourceCatalog) -> dict[str, Any] | None:
    """Optional deterministic fast path for repetitive vendors."""
    content = str(doc.get("content") or "")
    title = str(doc.get("title") or "")
    rewe = catalog.correspondents_by_name.get(normalize_name("REWE"))
    invoice = catalog.document_types_by_name.get(normalize_name("Rechnung"))
    if not (rewe and invoice):
        return None
    if "REWE" not in content and "REWE" not in title:
        return None

    date_match = re.search(r"Datum:\s*(\d{2})\.(\d{2})\.(\d{4})", content)
    email_match = re.search(r"Dein REWE eBon vom (\d{2})\.(\d{2})\.(\d{4})", content) or re.search(
        r"Dein REWE eBon vom (\d{2})\.(\d{2})\.(\d{4})", title
    )
    amount_match = re.search(r"SUMME\s+EUR\s+(\d+,\d{2})", content)
    email_amount = re.search(r"H[oö]he von\s+(\d+,\d{2})\s*EUR", content) or re.search(
        r"H[oö]he von\s+(\d+,\d{2})\s*€", content
    )

    if date_match:
        day, month, year = date_match.groups()
        amount = amount_match.group(1) if amount_match else None
        kind = "REWE eBon"
    elif email_match:
        day, month, year = email_match.groups()
        amount = email_amount.group(1) if email_amount else None
        kind = "REWE eBon E-Mail"
    else:
        return None

    title_suffix = f" - {amount} EUR" if amount else ""
    created = f"{year}-{month}-{day}"
    return {
        "confidence": 0.98,
        "needs_review": False,
        "reason": "Deterministic REWE eBon pattern",
        "correspondent": {"id": rewe.id, "name": rewe.name, "create": False},
        "document_type": {"id": invoice.id, "name": invoice.name, "create": False},
        "created": created,
        "title": f"{kind} {day}.{month}.{year}{title_suffix}",
        "tags": [],
        "suggested_new_tags": [],
        "delete_candidate": False,
        "delete_reason": None,
    }


def normalize_classification(
    raw: dict[str, Any],
    doc: dict[str, Any],
    catalog: ResourceCatalog,
    args: argparse.Namespace,
) -> dict[str, Any]:
    warnings: list[str] = []
    confidence = coerce_confidence(raw.get("confidence"))
    needs_review = coerce_bool(raw.get("needs_review"))
    delete_candidate = coerce_bool(raw.get("delete_candidate"))

    corr_id, corr_name, corr_create = nested_resource(raw, "correspondent")
    corr_id, corr_name, corr_create, warning = resolve_resource(
        corr_id, corr_name, corr_create, catalog.correspondents, catalog.correspondents_by_name
    )
    if warning:
        warnings.append(f"correspondent: {warning}")

    dtype_id, dtype_name, dtype_create = nested_resource(raw, "document_type")
    dtype_id, dtype_name, dtype_create, warning = resolve_resource(
        dtype_id, dtype_name, dtype_create, catalog.document_types, catalog.document_types_by_name
    )
    if warning:
        warnings.append(f"document_type: {warning}")

    llm_tag_ids, tag_warnings = resolve_tags(raw.get("tags", []), catalog)
    warnings.extend(tag_warnings)

    existing_non_inbox = [int(t) for t in doc.get("tags", []) if int(t) != args.inbox_tag_id]
    if args.replace_tags:
        final_tags = llm_tag_ids
    else:
        final_tags = sorted(set(existing_non_inbox + llm_tag_ids))

    email_tag_id = catalog.email_attachment_tag_id()
    if email_tag_id and (doc.get("mime_type") == "message/rfc822" or email_tag_id in existing_non_inbox):
        final_tags = sorted(set(final_tags + [email_tag_id]))

    if args.drop_bulk_unclassified:
        bulk_id = catalog.bulk_unclassified_tag_id()
        if bulk_id:
            final_tags = [tag_id for tag_id in final_tags if tag_id != bulk_id]
    final_tags = [tag_id for tag_id in final_tags if tag_id != args.inbox_tag_id]

    created = str(raw.get("created") or doc.get("created") or "").strip()
    if not is_date(created):
        warnings.append("created is missing or invalid")

    title = safe_title(raw.get("title"), str(doc.get("title") or "Untitled document"))
    if title.casefold().startswith(("scan ", "your invoice is attached", "untitled")):
        warnings.append("title is generic")

    warnings.extend(
        classification_safety_warnings(
            doc,
            {
                "correspondent_name": corr_name,
                "created": created,
                "title": title,
            },
            args,
        )
    )

    if corr_create and not args.create_correspondents:
        warnings.append(f"new correspondent proposed but --create-correspondents is off: {corr_name}")
    if dtype_create and not args.create_document_types:
        warnings.append(f"new document type proposed but --create-document-types is off: {dtype_name}")
    if not corr_id and not (corr_create and args.create_correspondents):
        needs_review = True
    if not dtype_id and not (dtype_create and args.create_document_types):
        needs_review = True
    if warnings:
        needs_review = True

    return {
        "confidence": confidence,
        "needs_review": needs_review,
        "warnings": warnings,
        "reason": str(raw.get("reason") or "").strip(),
        "correspondent_id": corr_id,
        "correspondent_name": corr_name,
        "create_correspondent": corr_create,
        "document_type_id": dtype_id,
        "document_type_name": dtype_name,
        "create_document_type": dtype_create,
        "created": created,
        "title": title,
        "tag_ids": final_tags,
        "suggested_new_tags": raw.get("suggested_new_tags") or [],
        "delete_candidate": delete_candidate,
        "delete_reason": raw.get("delete_reason"),
        "raw": raw,
    }


def create_missing_resources(
    normalized: dict[str, Any],
    paperless: PaperlessClient,
    catalog: ResourceCatalog,
    args: argparse.Namespace,
) -> None:
    if normalized["create_correspondent"] and args.create_correspondents:
        created = paperless.post("/api/correspondents/", {"name": normalized["correspondent_name"]})
        normalized["correspondent_id"] = int(created["id"])
        catalog.correspondents[normalized["correspondent_id"]] = Resource(
            normalized["correspondent_id"], created["name"], created.get("slug")
        )
    if normalized["create_document_type"] and args.create_document_types:
        created = paperless.post("/api/document_types/", {"name": normalized["document_type_name"]})
        normalized["document_type_id"] = int(created["id"])
        catalog.document_types[normalized["document_type_id"]] = Resource(
            normalized["document_type_id"], created["name"], created.get("slug")
        )


def build_patch(
    normalized: dict[str, Any],
    *,
    remove_inbox_tags: bool = True,
    inbox_tag_id: int | None = None,
) -> dict[str, Any]:
    tag_ids = sorted(set(int(tag_id) for tag_id in normalized.get("tag_ids", [])))
    if not remove_inbox_tags and inbox_tag_id:
        tag_ids = sorted(set(tag_ids + [inbox_tag_id]))
    patch = {
        "correspondent": normalized["correspondent_id"],
        "document_type": normalized["document_type_id"],
        "created": normalized["created"],
        "title": normalized["title"],
        "tags": tag_ids,
    }
    patch["remove_inbox_tags"] = remove_inbox_tags
    return patch


def has_patchable_metadata(normalized: dict[str, Any]) -> tuple[bool, str]:
    if normalized.get("delete_candidate"):
        return False, "delete candidate"
    if not normalized.get("correspondent_id") or not normalized.get("document_type_id"):
        return False, "missing IDs"
    if not is_date(normalized.get("created")):
        return False, "invalid created date"
    if not normalized.get("title"):
        return False, "missing title"
    tag_ids = normalized.get("tag_ids")
    if not isinstance(tag_ids, list) or any(positive_int(tag_id) is None for tag_id in tag_ids):
        return False, "invalid tags"
    return True, "patchable"


def should_apply(normalized: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if normalized["delete_candidate"]:
        return False, "delete candidate"
    if normalized["needs_review"] and not args.force:
        return False, "needs review"
    if normalized["confidence"] < args.threshold and not args.force:
        return False, f"confidence below threshold {args.threshold}"
    return has_patchable_metadata(normalized)


def status_for_error_message(message: str) -> tuple[str, str]:
    if any(marker in message.casefold() for marker in TERMINAL_ERROR_MARKERS):
        return "skipped_unreadable", message
    return "failed", message


def record_exception(record: dict[str, Any], exc: Exception) -> None:
    status, message = status_for_error_message(str(exc))
    record["status"] = status
    if status.startswith("skipped"):
        record["skip_reason"] = message
    else:
        record["error"] = message


def resume_seed_record(source_record: dict[str, Any], source: Path) -> dict[str, Any] | None:
    status = source_record.get("status")
    if status in RESUME_SKIP_STATUSES:
        seed = dict(source_record)
    elif status == "failed":
        error_status, message = status_for_error_message(str(source_record.get("error") or ""))
        if error_status not in RESUME_SKIP_STATUSES:
            return None
        seed = dict(source_record)
        seed["status"] = error_status
        seed["skip_reason"] = message
    else:
        return None
    seed["seeded_from_audit"] = str(source)
    return seed


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ClassifierError(f"Invalid JSONL in {path} line {line_number}: {exc}") from exc
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def append_record_and_print(
    records: list[dict[str, Any]],
    audit_jsonl: Path,
    record: dict[str, Any],
    completed: int,
    total: int,
) -> None:
    records.append(record)
    write_jsonl(audit_jsonl, record)
    title = record.get("classification", {}).get("title") or record.get("original_title")
    queued = record.get("index")
    queued_suffix = f" queued #{queued}" if queued and queued != completed else ""
    print(
        f"[{completed}/{total}]{queued_suffix} {record['document_id']}: {record['status']} - {title}",
        flush=True,
    )


def latest_by_document(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            doc_id = int(record["document_id"])
        except (KeyError, TypeError, ValueError):
            continue
        latest[doc_id] = record
    return latest


def markdown_summary(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    records = list(latest_by_document(records).values())
    updated = [r for r in records if r["status"] == "updated"]
    updated_kept_inbox = [r for r in records if r["status"] == "updated_kept_inbox"]
    dry_ready = [r for r in records if r["status"] == "dry_run_ready"]
    skipped = [r for r in records if r["status"].startswith("skipped")]
    failed = [r for r in records if r["status"] == "failed"]
    deletes = [r for r in records if r.get("classification", {}).get("delete_candidate")]
    vision_incomplete = [
        r
        for r in records
        if r.get("classification", {}).get("vision", {}).get("enabled")
        and not r.get("classification", {}).get("vision", {}).get("all_pages_included")
    ]

    lines = [
        "# Paperless AI Classification Run",
        "",
        f"- Mode: {'apply' if args.apply else 'dry-run'}",
        f"- Provider: `{args.provider}`",
        f"- Model: `{args.model}`",
        f"- Threshold: `{args.threshold}`",
        f"- Documents considered: `{len(records)}`",
        f"- Updated: `{len(updated)}`",
        f"- Updated, kept Inbox: `{len(updated_kept_inbox)}`",
        f"- Dry-run ready: `{len(dry_ready)}`",
        f"- Skipped: `{len(skipped)}`",
        f"- Failed: `{len(failed)}`",
        f"- Vision incomplete: `{len(vision_incomplete)}`",
        "",
    ]
    if deletes:
        lines.extend(["## Delete candidates", ""])
        for record in deletes:
            c = record.get("classification", {})
            lines.append(
                f"- `{record['document_id']}`: {record.get('original_title')} - {c.get('delete_reason') or 'No reason'}"
            )
        lines.append("")
    if skipped or failed:
        lines.extend(["## Review needed", ""])
        for record in skipped + failed:
            c = record.get("classification", {})
            reason = record.get("skip_reason") or record.get("error") or c.get("reason") or "Unknown"
            warnings = "; ".join(c.get("warnings") or [])
            suffix = f" ({warnings})" if warnings else ""
            lines.append(f"- `{record['document_id']}`: {record.get('original_title')} - {reason}{suffix}")
        lines.append("")
    lines.extend(["## Successful classifications", ""])
    for record in updated + updated_kept_inbox + dry_ready:
        c = record.get("classification", {})
        vision = c.get("vision") or {}
        vision_suffix = ""
        if vision.get("enabled"):
            if vision.get("file_count"):
                vision_suffix = (
                    f" | vision `PDF file, {vision.get('page_count') or '?'} pages`"
                )
            else:
                vision_suffix = (
                    f" | vision `{vision.get('image_count', 0)}/{vision.get('page_count') or '?'} pages`"
                )
        lines.append(
            f"- `{record['document_id']}`: {c.get('title')} | company `{c.get('correspondent_name')}` | type `{c.get('document_type_name')}` | date `{c.get('created')}` | confidence `{c.get('confidence')}`{vision_suffix}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify_one(
    doc: dict[str, Any],
    paperless: PaperlessClient,
    llm: LLMClient,
    catalog: ResourceCatalog,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.rules_first:
        raw = maybe_rule_classify(doc, catalog)
        raw_source = "rule"
    else:
        raw = None
        raw_source = args.provider

    raw_text = None
    if raw is None:
        images: list[VisionImage] = []
        files: list[VisionFile] = []
        plugins: list[dict[str, Any]] = []
        use_openrouter_pdf = (
            args.vision
            and args.provider == "openrouter"
            and args.pdf_input in {"auto", "openrouter-file"}
        )
        if use_openrouter_pdf:
            files, vision, plugins = build_openrouter_pdf_file_input(doc, paperless, args)
            if files:
                paperless_ocr = None if args.ocr_source != "always" else str(doc.get("content") or "")
                ocr_policy = (
                    "included_because_ocr_source_always"
                    if paperless_ocr is not None
                    else "omitted_because_openrouter_pdf_input_includes_all_pages"
                )
            else:
                paperless_ocr = str(doc.get("content") or "")
                ocr_policy = "included_because_openrouter_pdf_input_was_not_available"
        else:
            vision = {"enabled": args.vision, "requested_pages": "all", "warnings": []}
            paperless_ocr = None
            ocr_policy = "pending"

        if not files:
            omit_paperless_ocr = should_omit_paperless_ocr(doc, catalog, args)
            paperless_ocr = None if omit_paperless_ocr else str(doc.get("content") or "")
            ocr_policy = (
                "omitted_because_all_pages_fit_in_vision_context"
                if omit_paperless_ocr
                else "included_as_fallback_or_partial_vision_context"
            )
            preliminary_user_text = build_user_message(
                doc,
                catalog,
                args.content_chars,
                paperless_ocr,
                ocr_policy,
                {"enabled": args.vision, "requested_pages": "all"},
            )
            images, vision = build_vision_inputs(doc, paperless, args, preliminary_user_text)
            if (
                omit_paperless_ocr
                and args.ocr_source == "auto"
                and args.vision
                and not vision.get("all_pages_included")
            ):
                paperless_ocr = str(doc.get("content") or "")
                ocr_policy = "included_because_all_page_vision_was_not_available"
                preliminary_user_text = build_user_message(
                    doc,
                    catalog,
                    args.content_chars,
                    paperless_ocr,
                    ocr_policy,
                    {"enabled": args.vision, "requested_pages": "all"},
                )
                images, vision = build_vision_inputs(doc, paperless, args, preliminary_user_text)
        user_text = build_user_message(
            doc,
            catalog,
            args.content_chars,
            paperless_ocr,
            ocr_policy,
            {
                key: value
                for key, value in vision.items()
                if key not in {"warnings", "budget"}
            },
        )
        messages = build_messages(user_text, images, files)
        raw, raw_text = llm.classify(messages, plugins)
        raw_source = args.provider
    else:
        vision = {"enabled": False, "source": "rule", "warnings": []}

    normalized = normalize_classification(raw, doc, catalog, args)
    normalized["source"] = raw_source
    normalized["vision"] = vision
    vision_warnings = [] if raw_source == "rule" else vision_review_warnings(vision, args)
    if vision_warnings:
        normalized["warnings"] = sorted(set(normalized.get("warnings", []) + vision_warnings))
        normalized["needs_review"] = True
    if raw_text:
        normalized["raw_text"] = raw_text
    return normalized


def run(args: argparse.Namespace) -> int:
    if not args.paperless_url:
        raise ClassifierError("Set PAPERLESS_URL or pass --paperless-url")
    if not args.paperless_token:
        raise ClassifierError("Set PAPERLESS_TOKEN or pass --paperless-token")
    if args.provider == "openrouter" and not args.llm_api_key:
        raise ClassifierError("Set OPENROUTER_API_KEY or pass --llm-api-key for OpenRouter")
    if args.workers > 1 and (args.create_correspondents or args.create_document_types):
        raise ClassifierError(
            "--workers > 1 cannot be combined with --create-correspondents or "
            "--create-document-types because resource creation must be serialized"
        )

    output_dir = Path(args.output_dir or Path(DEFAULT_OUTPUT_ROOT) / now_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_jsonl = output_dir / "audit.jsonl"
    audit_md = output_dir / "summary.md"

    paperless = PaperlessClient(args.paperless_url, args.paperless_token, args.timeout)
    llm = LLMClient(
        args.provider,
        args.llm_url,
        args.model,
        args.llm_api_key,
        args.openrouter_site_url,
        args.openrouter_app_name,
        args.timeout,
        args.temperature,
        args.max_tokens,
        args.response_format,
        args.retries,
        args.retry_sleep,
    )

    catalog = paperless.catalog()
    inbox = catalog.inbox_tag(args.label)
    args.inbox_tag_id = inbox.id

    if args.ids:
        ids = [int(value) for value in args.ids]
    else:
        ids = paperless.inbox_ids(inbox.id, args.page_size, args.ordering, args.query, args.limit)

    existing_records: list[dict[str, Any]] = []
    resume_skipped = 0
    if args.resume:
        existing_records = read_jsonl(audit_jsonl)
        latest = latest_by_document(existing_records)
        skip_ids = {
            doc_id
            for doc_id, record in latest.items()
            if record.get("status") in RESUME_SKIP_STATUSES
        }
        resume_skipped = sum(1 for doc_id in ids if doc_id in skip_ids)
        ids = [doc_id for doc_id in ids if doc_id not in skip_ids]

    print(f"Run directory: {output_dir}", flush=True)
    if args.resume:
        print(f"Resume: {len(existing_records)} existing audit records", flush=True)
        print(f"Resume skipped: {resume_skipped} terminal documents", flush=True)
    print(f"Documents queued: {len(ids)}", flush=True)
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    def process_doc(index: int, doc_id: int) -> dict[str, Any]:
        record: dict[str, Any] = {"document_id": doc_id, "index": index}
        try:
            wait_for_ac_power(args)
            doc = paperless.document(doc_id)
            record["original_title"] = doc.get("title")
            record["original_tags"] = doc.get("tags", [])
            normalized = classify_one(doc, paperless, llm, catalog, args)
            record["classification"] = normalized

            if normalized["delete_candidate"]:
                record["status"] = "skipped_delete_candidate"
                record["skip_reason"] = normalized.get("delete_reason") or "delete candidate"
            else:
                if args.apply:
                    create_missing_resources(normalized, paperless, catalog, args)
                ready, reason = should_apply(normalized, args)
                patchable, patchable_reason = has_patchable_metadata(normalized)
                record["patch"] = build_patch(normalized) if ready else None
                record["review_patch"] = (
                    build_patch(
                        normalized,
                        remove_inbox_tags=False,
                        inbox_tag_id=args.inbox_tag_id,
                    )
                    if not ready and patchable
                    else None
                )
                if not ready:
                    if args.apply and args.apply_review_metadata and patchable:
                        result = paperless.patch(f"/api/documents/{doc_id}/", record["review_patch"])
                        record["status"] = "updated_kept_inbox"
                        record["skip_reason"] = reason
                        record["updated_title"] = result.get("title")
                    else:
                        record["status"] = "skipped_needs_review"
                        record["skip_reason"] = reason if patchable else patchable_reason
                elif args.apply:
                    result = paperless.patch(f"/api/documents/{doc_id}/", record["patch"])
                    record["status"] = "updated"
                    record["updated_title"] = result.get("title")
                else:
                    record["status"] = "dry_run_ready"
                    record["skip_reason"] = "dry-run"
        except Exception as exc:  # noqa: BLE001 - audit should capture any per-document failure.
            record_exception(record, exc)

        if args.sleep:
            time.sleep(args.sleep)
        return record

    records: list[dict[str, Any]] = list(existing_records)
    total = len(ids)
    if args.workers == 1:
        for index, doc_id in enumerate(ids, 1):
            record = process_doc(index, doc_id)
            append_record_and_print(records, audit_jsonl, record, index, total)
    else:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(process_doc, index, doc_id): (index, doc_id)
                for index, doc_id in enumerate(ids, 1)
            }
            for completed, future in enumerate(futures.as_completed(future_map), 1):
                index, doc_id = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - executor-level failure should be audited.
                    record = {
                        "document_id": doc_id,
                        "index": index,
                    }
                    record_exception(record, exc)
                append_record_and_print(records, audit_jsonl, record, completed, total)

    markdown_summary(audit_md, records, args)
    print(f"Audit JSONL: {audit_jsonl}", flush=True)
    print(f"Summary: {audit_md}", flush=True)
    return 0


def apply_from_audit(args: argparse.Namespace) -> int:
    if not args.paperless_url:
        raise ClassifierError("Set PAPERLESS_URL or pass --paperless-url")
    if not args.paperless_token:
        raise ClassifierError("Set PAPERLESS_TOKEN or pass --paperless-token")

    source = Path(args.apply_audit)
    source_records = latest_by_document(read_jsonl(source))
    output_dir = Path(args.output_dir or Path(DEFAULT_OUTPUT_ROOT) / f"apply-{now_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_jsonl = output_dir / "audit.jsonl"
    audit_md = output_dir / "summary.md"

    paperless = PaperlessClient(args.paperless_url, args.paperless_token, args.timeout)
    catalog = paperless.catalog()
    inbox = catalog.inbox_tag(args.label)
    args.inbox_tag_id = inbox.id

    candidates = []
    for record in source_records.values():
        if record.get("status") == "dry_run_ready" and isinstance(record.get("patch"), dict):
            candidates.append(record)
        elif (
            args.apply_review_metadata
            and record.get("status") == "skipped_needs_review"
            and isinstance(record.get("review_patch"), dict)
        ):
            candidates.append(record)
        elif args.apply_review_metadata and record.get("status") == "skipped_needs_review":
            classification = record.get("classification") or {}
            patchable, _ = has_patchable_metadata(classification)
            if patchable:
                record = dict(record)
                record["review_patch"] = build_patch(
                    classification,
                    remove_inbox_tags=False,
                    inbox_tag_id=args.inbox_tag_id,
                )
                candidates.append(record)
    candidates.sort(key=lambda record: int(record["document_id"]))

    existing_records: list[dict[str, Any]] = []
    resume_skipped = 0
    resume_seeded = 0
    if args.resume:
        existing_records = read_jsonl(audit_jsonl)
        latest = latest_by_document(existing_records)
        candidate_ids = {int(record["document_id"]) for record in candidates}
        current_inbox_ids = set(
            paperless.inbox_ids(inbox.id, args.page_size, args.ordering, args.query, None)
        )
        for doc_id, source_record in sorted(source_records.items()):
            if doc_id in candidate_ids or doc_id in latest or doc_id not in current_inbox_ids:
                continue
            seed = resume_seed_record(source_record, source)
            if not seed:
                continue
            write_jsonl(audit_jsonl, seed)
            existing_records.append(seed)
            latest[doc_id] = seed
            resume_seeded += 1
        skip_ids = {
            doc_id
            for doc_id, record in latest.items()
            if record.get("status") in RESUME_SKIP_STATUSES
        }
        resume_skipped = sum(1 for record in candidates if int(record["document_id"]) in skip_ids)
        candidates = [
            record for record in candidates if int(record["document_id"]) not in skip_ids
        ]
    if args.limit:
        candidates = candidates[: args.limit]

    print(f"Applying audited patches from: {source}", flush=True)
    print(f"Run directory: {output_dir}", flush=True)
    if args.resume:
        print(f"Resume: {len(existing_records)} existing audit records", flush=True)
        print(f"Resume seeded: {resume_seeded} terminal source records", flush=True)
        print(f"Resume skipped: {resume_skipped} terminal documents", flush=True)
    print(f"Audited patches queued: {len(candidates)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    def process_candidate(index: int, source_record: dict[str, Any]) -> dict[str, Any]:
        doc_id = int(source_record["document_id"])
        record = {
            "document_id": doc_id,
            "index": index,
            "original_title": source_record.get("original_title"),
            "classification": source_record.get("classification"),
            "patch": source_record.get("patch")
            if source_record.get("status") == "dry_run_ready"
            else source_record.get("review_patch"),
            "review_metadata": source_record.get("status") == "skipped_needs_review",
        }
        try:
            wait_for_ac_power(args)
            classification = record.get("classification") or {}
            if classification.get("delete_candidate"):
                record["status"] = "skipped_delete_candidate"
                record["skip_reason"] = classification.get("delete_reason") or "delete candidate"
            elif record["review_metadata"]:
                current = paperless.document(doc_id)
                record["original_title"] = current.get("title")
                record["original_tags"] = current.get("tags", [])
                if inbox.id not in current.get("tags", []):
                    record["status"] = "skipped_already_not_in_inbox"
                    record["skip_reason"] = "Inbox tag already absent"
                else:
                    result = paperless.patch(f"/api/documents/{doc_id}/", record["patch"])
                    record["status"] = "updated_kept_inbox"
                    record["updated_title"] = result.get("title")
                    record["skip_reason"] = source_record.get("skip_reason") or "review metadata applied"
            elif float(classification.get("confidence") or 0) < args.threshold and not args.force:
                record["status"] = "skipped_needs_review"
                record["skip_reason"] = f"confidence below threshold {args.threshold}"
            elif classification.get("needs_review") and not args.force:
                record["status"] = "skipped_needs_review"
                record["skip_reason"] = "needs review"
            else:
                current = paperless.document(doc_id)
                record["original_title"] = current.get("title")
                record["original_tags"] = current.get("tags", [])
                safety_warnings = classification_safety_warnings(current, classification, args)
                if classification.get("source") != "rule":
                    safety_warnings.extend(
                        vision_review_warnings(classification.get("vision"), args)
                    )
                if safety_warnings and not args.force:
                    classification["needs_review"] = True
                    classification["warnings"] = sorted(
                        set((classification.get("warnings") or []) + safety_warnings)
                    )
                    record["status"] = "skipped_needs_review"
                    record["skip_reason"] = "; ".join(safety_warnings)
                    record["patch"] = None
                elif inbox.id not in current.get("tags", []):
                    record["status"] = "skipped_already_not_in_inbox"
                    record["skip_reason"] = "Inbox tag already absent"
                else:
                    result = paperless.patch(f"/api/documents/{doc_id}/", record["patch"])
                    record["status"] = "updated"
                    record["updated_title"] = result.get("title")
        except Exception as exc:  # noqa: BLE001 - audit should capture any per-document failure.
            record_exception(record, exc)

        if args.sleep:
            time.sleep(args.sleep)
        return record

    records: list[dict[str, Any]] = list(existing_records)
    total = len(candidates)
    if args.workers == 1:
        for index, source_record in enumerate(candidates, 1):
            record = process_candidate(index, source_record)
            append_record_and_print(records, audit_jsonl, record, index, total)
    else:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(process_candidate, index, source_record): (
                    index,
                    int(source_record["document_id"]),
                )
                for index, source_record in enumerate(candidates, 1)
            }
            for completed, future in enumerate(futures.as_completed(future_map), 1):
                index, doc_id = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - executor-level failure should be audited.
                    record = {
                        "document_id": doc_id,
                        "index": index,
                    }
                    record_exception(record, exc)
                append_record_and_print(records, audit_jsonl, record, completed, total)

    markdown_summary(audit_md, records, args)
    print(f"Audit JSONL: {audit_jsonl}", flush=True)
    print(f"Summary: {audit_md}", flush=True)
    return 0


def self_test() -> int:
    samples = [
        '```json\n{"confidence":0.91,"needs_review":false,"correspondent":{"id":68,"name":"REWE","create":false},"document_type":{"id":1,"name":"Rechnung","create":false},"created":"2026-05-02","title":"REWE eBon 02.05.2026 - 23,79 EUR","tags":[{"id":11,"name":"Email Attachment"}],"delete_candidate":false}\n```',
        'Some text {"confidence":"0.5","needs_review":"yes","correspondent_id":null,"document_type_id":1,"created":"bad","title":"Scan 1","tags":["Email Attachment"],"delete_candidate":true,"delete_reason":"duplicate"} trailing',
    ]
    for sample in samples:
        parsed = parse_json_object(sample)
        assert isinstance(parsed, dict), parsed
    assert is_date("2026-05-03")
    assert not is_date("03.05.2026")
    assert safe_title("", "Fallback") == "Fallback"
    assert select_representative_pages(5, 3) == [0, 2, 4]
    assert select_representative_pages(2, 10) == [0, 1]
    sample_normalized = {
        "correspondent_id": 1,
        "document_type_id": 2,
        "created": "2026-05-02",
        "title": "Sample",
        "tag_ids": [3],
        "delete_candidate": False,
    }
    assert has_patchable_metadata(sample_normalized) == (True, "patchable")
    assert has_patchable_metadata({**sample_normalized, "tag_ids": ["bad"]}) == (
        False,
        "invalid tags",
    )
    assert status_for_error_message("document closed or encrypted") == (
        "skipped_unreadable",
        "document closed or encrypted",
    )
    assert resume_seed_record(
        {"document_id": 1, "status": "failed", "error": "document closed or encrypted"},
        Path("audit.jsonl"),
    )["status"] == "skipped_unreadable"
    assert build_patch(sample_normalized)["remove_inbox_tags"] is True
    review_patch = build_patch(sample_normalized, remove_inbox_tags=False, inbox_tag_id=9)
    assert review_patch["remove_inbox_tags"] is False
    assert review_patch["tags"] == [3, 9]
    print("Self-test passed", flush=True)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    load_env_file(Path(".env"))
    provider_default = os.getenv("LLM_PROVIDER", "lmstudio")
    llm_url_env = os.getenv("LLM_URL") or os.getenv("OPENROUTER_URL") or os.getenv("LMSTUDIO_URL")
    model_env = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or os.getenv("LMSTUDIO_MODEL")
    llm_api_key_default = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if provider_default == "openrouter":
        llm_url_default = llm_url_env or DEFAULT_OPENROUTER_URL
        model_default = model_env or DEFAULT_OPENROUTER_MODEL
    else:
        llm_url_default = llm_url_env or DEFAULT_LMSTUDIO_URL
        model_default = model_env or DEFAULT_MODEL

    parser = argparse.ArgumentParser(
        description="Classify Paperless Inbox documents with an OpenAI-compatible LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--paperless-url", default=os.getenv("PAPERLESS_URL"), help="Paperless-ngx base URL")
    parser.add_argument("--paperless-token", default=os.getenv("PAPERLESS_TOKEN"), help="Paperless API token")
    parser.add_argument(
        "--provider",
        choices=["lmstudio", "openrouter", "openai-compatible"],
        default=provider_default,
        help="LLM provider preset",
    )
    parser.add_argument(
        "--llm-url",
        "--lmstudio-url",
        dest="llm_url",
        default=llm_url_default,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--model", default=model_default, help="LLM model name")
    parser.add_argument(
        "--llm-api-key",
        default=llm_api_key_default,
        help="Bearer API key for hosted OpenAI-compatible providers",
    )
    parser.add_argument(
        "--openrouter-site-url",
        default=os.getenv("OPENROUTER_SITE_URL"),
        help="Optional OpenRouter HTTP-Referer header",
    )
    parser.add_argument(
        "--openrouter-app-name",
        default=os.getenv("OPENROUTER_APP_NAME", "paperless-classifier-ai"),
        help="Optional OpenRouter X-Title header",
    )
    parser.add_argument("--label", default="Inbox", help="Inbox tag name if no is_inbox_tag exists")
    parser.add_argument("--limit", type=int, default=10, help="Maximum documents to process; 0 means all")
    parser.add_argument("--id", dest="ids", action="append", help="Classify a specific document ID")
    parser.add_argument("--query", help="Optional Paperless full-text query filter")
    parser.add_argument("--page-size", type=int, default=100, help="Paperless API page size")
    parser.add_argument("--ordering", default="-created", help="Paperless document ordering")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum confidence for apply")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum LLM response tokens")
    parser.add_argument(
        "--context-window",
        type=int,
        default=int(os.getenv("LLM_CONTEXT_WINDOW") or os.getenv("LMSTUDIO_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW)),
        help="LLM context window used for image page budgeting",
    )
    parser.add_argument(
        "--context-safety-tokens",
        type=int,
        default=DEFAULT_CONTEXT_SAFETY_TOKENS,
        help="Reserved context tokens for schema overhead and estimator error",
    )
    parser.add_argument(
        "--image-token-estimate",
        type=int,
        default=DEFAULT_IMAGE_TOKEN_ESTIMATE,
        help="Estimated context tokens consumed by each rendered page image",
    )
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "text"],
        default="text",
        help="Use json_schema only when your provider/model handles it reliably",
    )
    parser.add_argument("--content-chars", type=int, default=2500, help="Paperless OCR characters sent when OCR fallback is used")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient LLM errors")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Base sleep between LLM retries")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("PAPERLESS_AI_WORKERS", DEFAULT_WORKERS)),
        help="Documents to process concurrently",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between documents")
    parser.add_argument("--allow-battery", action="store_true", help="Do not pause when macOS reports Battery Power")
    parser.add_argument("--power-check-interval", type=float, default=300.0, help="Seconds between AC-power checks while paused")
    parser.add_argument(
        "--email-date-drift-review-days",
        type=int,
        default=180,
        help="Hold email terms/conditions for review when chosen date differs from email date by this many days",
    )
    parser.add_argument("--output-dir", help="Run artifact directory")
    parser.add_argument("--resume", action="store_true", help="Skip already audited documents in output-dir/audit.jsonl")
    parser.add_argument("--apply-audit", help="Apply dry-run-ready patches from an audit.jsonl without reclassifying")
    parser.add_argument("--apply", action="store_true", help="Patch Paperless records. Omit for dry-run.")
    parser.add_argument(
        "--apply-review-metadata",
        action="store_true",
        help="In apply mode, patch review-required metadata but keep the Inbox tag",
    )
    parser.add_argument("--force", action="store_true", help="Apply even when needs_review/confidence gate fails")
    parser.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send rendered Paperless page images to the LLM; use --no-vision to disable",
    )
    parser.add_argument(
        "--vision-dpi",
        type=int,
        default=DEFAULT_VISION_DPI,
        help="DPI used when rendering Paperless preview PDFs into page images",
    )
    parser.add_argument(
        "--max-vision-pages",
        type=int,
        default=0,
        help="Hard cap on rendered vision pages; 0 means only the context budget decides",
    )
    parser.add_argument(
        "--allow-partial-vision",
        action="store_true",
        help="Allow apply when only a representative subset of pages was sent",
    )
    parser.add_argument(
        "--pdf-input",
        choices=["auto", "rendered-images", "openrouter-file"],
        default="auto",
        help="PDF input strategy: OpenRouter file input in auto/openrouter-file, rendered page images otherwise",
    )
    parser.add_argument(
        "--openrouter-pdf-engine",
        choices=["mistral-ocr", "cloudflare-ai", "native", "default"],
        default=os.getenv("OPENROUTER_PDF_ENGINE", "mistral-ocr"),
        help="OpenRouter file-parser PDF engine when PDF file input is used",
    )
    parser.add_argument(
        "--ocr-source",
        choices=["auto", "always", "never"],
        default="auto",
        help="Use Paperless OCR always, never, or only when not all pages fit as images",
    )
    parser.add_argument("--rules-first", action="store_true", help="Use deterministic vendor rules before the LLM")
    parser.add_argument("--replace-tags", action="store_true", help="Replace non-Inbox tags instead of preserving them")
    parser.add_argument("--drop-bulk-unclassified", action="store_true", help="Drop Bulk Unclassified after classification")
    parser.add_argument("--create-correspondents", action="store_true", help="Create missing correspondent resources")
    parser.add_argument("--create-document-types", action="store_true", help="Create missing document type resources")
    parser.add_argument("--self-test", action="store_true", help="Run local parser sanity checks")
    args = parser.parse_args(argv)
    if args.provider == "openrouter":
        if not llm_url_env and args.llm_url == DEFAULT_LMSTUDIO_URL:
            args.llm_url = DEFAULT_OPENROUTER_URL
        if not model_env and args.model == DEFAULT_MODEL:
            args.model = DEFAULT_OPENROUTER_MODEL
    if args.limit is not None and args.limit < 1:
        args.limit = None
    if args.context_window < 1:
        parser.error("--context-window must be positive")
    if args.context_safety_tokens < 0:
        parser.error("--context-safety-tokens must not be negative")
    if args.image_token_estimate < 1:
        parser.error("--image-token-estimate must be positive")
    if args.vision_dpi < 1:
        parser.error("--vision-dpi must be positive")
    if args.max_vision_pages < 0:
        parser.error("--max-vision-pages must not be negative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.pdf_input == "openrouter-file" and args.provider != "openrouter":
        parser.error("--pdf-input openrouter-file requires --provider openrouter")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.self_test:
            return self_test()
        if args.apply_audit:
            return apply_from_audit(args)
        return run(args)
    except ClassifierError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
