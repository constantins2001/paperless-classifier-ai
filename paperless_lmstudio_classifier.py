#!/usr/bin/env python3
"""Classify Paperless Inbox documents with a local LM Studio model.

The script is intentionally conservative:
- It defaults to dry-run mode.
- It never deletes documents.
- It removes the Inbox tag only when a valid, confident classification is applied.
- It writes JSONL and Markdown audit files for every run.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
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


__version__ = "0.1.0"
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "gemma-4-31b-it"
DEFAULT_OUTPUT_ROOT = "paperless_lmstudio_runs"
DEFAULT_THRESHOLD = 0.86
MAX_TITLE_LEN = 128
RESUME_SKIP_STATUSES = {
    "dry_run_ready",
    "updated",
    "skipped_delete_candidate",
    "skipped_needs_review",
}


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
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{content_type};base64,{encoded}"


class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
        response_format: str,
        retries: int,
        retry_sleep: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.http = JsonHttpClient(timeout)

    def classify(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.response_format == "json_schema":
            payload["response_format"] = classification_response_format()
        elif self.response_format == "text":
            pass
        else:
            raise ClassifierError(f"Unsupported response format: {self.response_format}")
        last_error: ClassifierError | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self.http.request_json("POST", f"{self.base_url}/chat/completions", data=payload)
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
            raise last_error or ClassifierError("LM Studio request failed")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ClassifierError(f"Unexpected LM Studio response: {data}") from exc
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

Rules: company=sender/issuer/merchant/court/employer/provider, not recipient or employee. For work timesheets use the employer/client company, not the worker name. created=document issue/signature/submission/transaction/letter date, not import date. If a document covers a period/year, put that period in the title; do not use period end such as Dec 31 as created when a signing/issue/submission date is visible. Use existing IDs when possible. Preserve an existing correspondent/type when it is semantically compatible. Invoice/receipt/eBon/bill => Rechnung if available. Tags must be existing IDs, but never include Inbox. Keep Email Attachment for emails. If no existing company/type fits, id null + create true. Never delete; mark delete_candidate only. needs_review true if weak/ambiguous/missing IDs. Confidence means safe to apply.

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
            "content": truncate_text(str(doc.get("content") or ""), content_chars),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_messages(
    doc: dict[str, Any],
    catalog: ResourceCatalog,
    content_chars: int,
    image_data_url: str | None,
) -> list[dict[str, Any]]:
    user_text = build_user_message(doc, catalog, content_chars)
    if image_data_url:
        content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        content = user_text
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": content},
    ]


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


def build_patch(normalized: dict[str, Any]) -> dict[str, Any]:
    return {
        "correspondent": normalized["correspondent_id"],
        "document_type": normalized["document_type_id"],
        "created": normalized["created"],
        "title": normalized["title"],
        "tags": normalized["tag_ids"],
        "remove_inbox_tags": True,
    }


def should_apply(normalized: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if normalized["delete_candidate"]:
        return False, "delete candidate"
    if normalized["needs_review"] and not args.force:
        return False, "needs review"
    if normalized["confidence"] < args.threshold and not args.force:
        return False, f"confidence below threshold {args.threshold}"
    if not normalized["correspondent_id"] or not normalized["document_type_id"]:
        return False, "missing IDs"
    if not is_date(normalized["created"]):
        return False, "invalid created date"
    if not normalized["title"]:
        return False, "missing title"
    return True, "ready"


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
    dry_ready = [r for r in records if r["status"] == "dry_run_ready"]
    skipped = [r for r in records if r["status"].startswith("skipped")]
    failed = [r for r in records if r["status"] == "failed"]
    deletes = [r for r in records if r.get("classification", {}).get("delete_candidate")]

    lines = [
        "# Paperless LM Studio Classification Run",
        "",
        f"- Mode: {'apply' if args.apply else 'dry-run'}",
        f"- Model: `{args.model}`",
        f"- Threshold: `{args.threshold}`",
        f"- Documents considered: `{len(records)}`",
        f"- Updated: `{len(updated)}`",
        f"- Dry-run ready: `{len(dry_ready)}`",
        f"- Skipped: `{len(skipped)}`",
        f"- Failed: `{len(failed)}`",
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
    for record in updated + dry_ready:
        c = record.get("classification", {})
        lines.append(
            f"- `{record['document_id']}`: {c.get('title')} | company `{c.get('correspondent_name')}` | type `{c.get('document_type_name')}` | date `{c.get('created')}` | confidence `{c.get('confidence')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify_one(
    doc: dict[str, Any],
    paperless: PaperlessClient,
    lmstudio: LMStudioClient,
    catalog: ResourceCatalog,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.rules_first:
        raw = maybe_rule_classify(doc, catalog)
        raw_source = "rule"
    else:
        raw = None
        raw_source = "lmstudio"

    raw_text = None
    if raw is None:
        image_data_url = None
        if args.vision:
            image_data_url = paperless.thumbnail_data_url(int(doc["id"]))
        messages = build_messages(doc, catalog, args.content_chars, image_data_url)
        raw, raw_text = lmstudio.classify(messages)
        raw_source = "lmstudio"

    normalized = normalize_classification(raw, doc, catalog, args)
    normalized["source"] = raw_source
    if raw_text:
        normalized["raw_text"] = raw_text
    return normalized


def run(args: argparse.Namespace) -> int:
    if not args.paperless_url:
        raise ClassifierError("Set PAPERLESS_URL or pass --paperless-url")
    if not args.paperless_token:
        raise ClassifierError("Set PAPERLESS_TOKEN or pass --paperless-token")

    output_dir = Path(args.output_dir or Path(DEFAULT_OUTPUT_ROOT) / now_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_jsonl = output_dir / "audit.jsonl"
    audit_md = output_dir / "summary.md"

    paperless = PaperlessClient(args.paperless_url, args.paperless_token, args.timeout)
    lmstudio = LMStudioClient(
        args.lmstudio_url,
        args.model,
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
    if args.resume:
        existing_records = read_jsonl(audit_jsonl)
        latest = latest_by_document(existing_records)
        skip_ids = {
            doc_id
            for doc_id, record in latest.items()
            if record.get("status") in RESUME_SKIP_STATUSES
        }
        ids = [doc_id for doc_id in ids if doc_id not in skip_ids]

    print(f"Run directory: {output_dir}", flush=True)
    if args.resume:
        print(f"Resume: {len(existing_records)} existing audit records", flush=True)
    print(f"Documents queued: {len(ids)}", flush=True)
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}", flush=True)

    records: list[dict[str, Any]] = list(existing_records)
    for index, doc_id in enumerate(ids, 1):
        record: dict[str, Any] = {"document_id": doc_id, "index": index}
        try:
            wait_for_ac_power(args)
            doc = paperless.document(doc_id)
            record["original_title"] = doc.get("title")
            record["original_tags"] = doc.get("tags", [])
            normalized = classify_one(doc, paperless, lmstudio, catalog, args)
            record["classification"] = normalized

            if normalized["delete_candidate"]:
                record["status"] = "skipped_delete_candidate"
                record["skip_reason"] = normalized.get("delete_reason") or "delete candidate"
            else:
                create_missing_resources(normalized, paperless, catalog, args)
                ready, reason = should_apply(normalized, args)
                record["patch"] = build_patch(normalized) if ready else None
                if not ready:
                    record["status"] = "skipped_needs_review"
                    record["skip_reason"] = reason
                elif args.apply:
                    result = paperless.patch(f"/api/documents/{doc_id}/", record["patch"])
                    record["status"] = "updated"
                    record["updated_title"] = result.get("title")
                else:
                    record["status"] = "dry_run_ready"
                    record["skip_reason"] = "dry-run"
        except Exception as exc:  # noqa: BLE001 - audit should capture any per-document failure.
            record["status"] = "failed"
            record["error"] = str(exc)

        records.append(record)
        write_jsonl(audit_jsonl, record)
        status = record["status"]
        title = record.get("classification", {}).get("title") or record.get("original_title")
        print(f"[{index}/{len(ids)}] {doc_id}: {status} - {title}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

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

    candidates = [
        record
        for record in source_records.values()
        if record.get("status") == "dry_run_ready" and isinstance(record.get("patch"), dict)
    ]
    candidates.sort(key=lambda record: int(record["document_id"]))
    if args.limit:
        candidates = candidates[: args.limit]

    print(f"Applying audited patches from: {source}", flush=True)
    print(f"Run directory: {output_dir}", flush=True)
    print(f"Audited patches queued: {len(candidates)}", flush=True)

    records: list[dict[str, Any]] = []
    for index, source_record in enumerate(candidates, 1):
        doc_id = int(source_record["document_id"])
        record = {
            "document_id": doc_id,
            "index": index,
            "original_title": source_record.get("original_title"),
            "classification": source_record.get("classification"),
            "patch": source_record.get("patch"),
        }
        try:
            wait_for_ac_power(args)
            classification = record.get("classification") or {}
            if classification.get("delete_candidate"):
                record["status"] = "skipped_delete_candidate"
                record["skip_reason"] = classification.get("delete_reason") or "delete candidate"
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
            record["status"] = "failed"
            record["error"] = str(exc)

        records.append(record)
        write_jsonl(audit_jsonl, record)
        title = record.get("classification", {}).get("title") or record.get("original_title")
        print(f"[{index}/{len(candidates)}] {doc_id}: {record['status']} - {title}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

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
    print("Self-test passed", flush=True)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    load_env_file(Path(".env"))
    parser = argparse.ArgumentParser(
        description="Classify Paperless Inbox documents with LM Studio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--paperless-url", default=os.getenv("PAPERLESS_URL"), help="Paperless-ngx base URL")
    parser.add_argument("--paperless-token", default=os.getenv("PAPERLESS_TOKEN"), help="Paperless API token")
    parser.add_argument(
        "--lmstudio-url",
        default=os.getenv("LMSTUDIO_URL", DEFAULT_LMSTUDIO_URL),
        help="OpenAI-compatible LM Studio API base URL",
    )
    parser.add_argument("--model", default=os.getenv("LMSTUDIO_MODEL", DEFAULT_MODEL), help="LM Studio model name")
    parser.add_argument("--label", default="Inbox", help="Inbox tag name if no is_inbox_tag exists")
    parser.add_argument("--limit", type=int, default=10, help="Maximum documents to process; 0 means all")
    parser.add_argument("--id", dest="ids", action="append", help="Classify a specific document ID")
    parser.add_argument("--query", help="Optional Paperless full-text query filter")
    parser.add_argument("--page-size", type=int, default=100, help="Paperless API page size")
    parser.add_argument("--ordering", default="-created", help="Paperless document ordering")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum confidence for apply")
    parser.add_argument("--temperature", type=float, default=0.1, help="LM Studio sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum LM Studio response tokens")
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "text"],
        default="text",
        help="Use json_schema only when LM Studio is loaded with enough context",
    )
    parser.add_argument("--content-chars", type=int, default=2500, help="Document text characters sent to the model")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient LM Studio errors")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Base sleep between LM Studio retries")
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
    parser.add_argument("--force", action="store_true", help="Apply even when needs_review/confidence gate fails")
    parser.add_argument("--vision", action="store_true", help="Send Paperless thumbnail as an image_url to LM Studio")
    parser.add_argument("--rules-first", action="store_true", help="Use deterministic vendor rules before LM Studio")
    parser.add_argument("--replace-tags", action="store_true", help="Replace non-Inbox tags instead of preserving them")
    parser.add_argument("--drop-bulk-unclassified", action="store_true", help="Drop Bulk Unclassified after classification")
    parser.add_argument("--create-correspondents", action="store_true", help="Create missing correspondent resources")
    parser.add_argument("--create-document-types", action="store_true", help="Create missing document type resources")
    parser.add_argument("--self-test", action="store_true", help="Run local parser sanity checks")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        args.limit = None
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
