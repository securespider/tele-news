"""Cyber-security primitives: SSRF/DNS-rebinding defense, prompt-injection sanitization,
HMAC callback signing, invite tokens, rate limiting, and log redaction.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import os
import re
import secrets
import socket
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Coroutine
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx
from telegram.error import RetryAfter
from telegram.ext import BaseRateLimiter

logger = logging.getLogger(__name__)

_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_UNSAFE_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
_ALLOWED_REDIRECT_CODES = {301, 302, 303, 307, 308}

_DELIMITER_TAG_RE = re.compile(
    r"</?(?:rss_candidate|rss_candidates|article_content|hn_comments|paper_content|"
    r"papers_corpus|user_query|system_prompt|instructions)\b[^>]*>",
    re.IGNORECASE,
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SUSPICIOUS_INJECTION_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
    r"reveal\s+(?:your\s+)?system\s+prompt|"
    r"disregard\s+(?:all\s+)?(?:above|previous)\s+directives)",
    re.IGNORECASE,
)

_TG_TOKEN_RE = re.compile(r"(?:(?<=bot)|\b)\d{5,}:[A-Za-z0-9_-]{25,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_INVITE_CODE_RE = re.compile(r"(/(?:join|invite)\s+)[A-Za-z0-9_-]{12,}", re.IGNORECASE)


class SSRFBlockedError(ValueError):
    """Raised when a URL, redirect, or resolved IP violates SSRF policy."""


class PayloadTooLargeError(ValueError):
    """Raised when an HTTP response exceeds the streaming byte cap."""


def _unwrap_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Return the IP and any embedded IPv4 addresses (mapped, 6to4, Teredo)."""
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            candidates.append(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            candidates.append(ip.sixtofour)
        if ip.teredo is not None:
            candidates.extend(ip.teredo)
    return candidates


def is_safe_ip(ip_str: str) -> bool:
    """Return True only if ip_str is a publicly routable unicast IP address."""
    cleaned = (ip_str or "").strip().split("%", 1)[0]
    try:
        parsed = ipaddress.ip_address(cleaned)
    except ValueError:
        return False

    for candidate in _unwrap_ip(parsed):
        if (
            candidate.is_private
            or candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_multicast
            or candidate.is_reserved
            or candidate.is_unspecified
        ):
            return False
        if isinstance(candidate, ipaddress.IPv4Address) and candidate in _CGNAT_NETWORK:
            return False
    return True


def is_safe_public_url(url: str, *, allow_ports: set[int] | None = None) -> bool:
    """Verify URL uses http/https, has no embedded credentials, allowed port, and public host/IP."""
    allowed_ports = allow_ports if allow_ports is not None else {80, 443}
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return False

    if parts.scheme.lower() not in {"http", "https"}:
        return False

    if parts.username is not None or parts.password is not None:
        return False

    try:
        port = parts.port
    except ValueError:
        return False
    if port is not None and port not in allowed_ports:
        return False

    host = parts.hostname
    if not host:
        return False
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(_UNSAFE_SUFFIXES):
        return False

    try:
        ipaddress.ip_address(lowered)
    except ValueError:
        return True

    return is_safe_ip(lowered)


class SafeNetworkBackend(httpcore.AnyIOBackend):
    """httpcore network backend that resolves DNS upfront, blocks private IPs, and pins IP."""

    def __init__(self, *, allow_ports: set[int] | None = None) -> None:
        super().__init__()
        self._allow_ports = allow_ports if allow_ports is not None else {80, 443}

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        if port not in self._allow_ports:
            raise SSRFBlockedError(f"Blocked non-standard port {port} for host {host}")

        cleaned_host = (host or "").strip().rstrip(".")
        if not cleaned_host or cleaned_host.lower() == "localhost" or cleaned_host.lower().endswith(_UNSAFE_SUFFIXES):
            raise SSRFBlockedError(f"Blocked unsafe hostname: {host}")

        try:
            ipaddress.ip_address(cleaned_host)
            resolved_ips = [cleaned_host]
        except ValueError:
            loop = asyncio.get_running_loop()
            try:
                addr_info = await loop.getaddrinfo(
                    cleaned_host, port, type=socket.SOCK_STREAM
                )
            except socket.gaierror as exc:
                raise SSRFBlockedError(f"DNS resolution failed for {host}: {exc}") from exc

            resolved_ips = []
            for info in addr_info:
                sockaddr = info[4]
                if sockaddr and isinstance(sockaddr[0], str):
                    ip_val = sockaddr[0]
                    if ip_val not in resolved_ips:
                        resolved_ips.append(ip_val)

        if not resolved_ips:
            raise SSRFBlockedError(f"No IP addresses resolved for {host}")

        for ip_val in resolved_ips:
            if not is_safe_ip(ip_val):
                raise SSRFBlockedError(f"Blocked unsafe resolved IP {ip_val} for {host}")

        pinned_ip = resolved_ips[0]
        return await super().connect_tcp(
            pinned_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


def create_safe_async_client(
    *,
    timeout: float = 15.0,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with DNS-rebinding SafeNetworkBackend and disabled redirects."""
    transport = httpx.AsyncHTTPTransport()
    pool = getattr(transport, "_pool", None)
    if pool is not None and hasattr(pool, "_network_backend"):
        pool._network_backend = SafeNetworkBackend()
    return httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        follow_redirects=False,
        headers=headers or {},
    )


async def _read_stream_capped(response: httpx.Response, max_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise PayloadTooLargeError(
                    f"Response Content-Length {content_length} exceeds {max_bytes}"
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(f"Streamed response exceeded {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def safe_fetch_bytes(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_redirects: int = 5,
    max_bytes: int = 5 * 1024 * 1024,
) -> tuple[int, str, httpx.Headers, bytes]:
    """Perform GET with pre-hop SSRF validation, manual redirect chasing, and byte caps."""
    current_url = (url or "").strip()
    req_headers = dict(headers or {})

    for hop in range(max_redirects + 1):
        if not is_safe_public_url(current_url):
            raise SSRFBlockedError(f"Blocked unsafe URL: {current_url}")

        async with client.stream("GET", current_url, headers=req_headers) as response:
            if not is_safe_public_url(str(response.url)):
                raise SSRFBlockedError(f"Unsafe response URL rejected: {response.url}")

            if response.status_code in _ALLOWED_REDIRECT_CODES:
                location = response.headers.get("Location")
                if not location:
                    response.raise_for_status()
                if hop >= max_redirects:
                    raise SSRFBlockedError(f"Exceeded max redirects ({max_redirects}) for {url}")
                next_url = urljoin(str(response.url), location.strip())
                if not is_safe_public_url(next_url):
                    raise SSRFBlockedError(f"Unsafe redirect target rejected: {next_url}")
                current_url = next_url
                continue

            if response.status_code == 304:
                return (304, str(response.url), response.headers, b"")

            response.raise_for_status()
            body = await _read_stream_capped(response, max_bytes)
            return (response.status_code, str(response.url), response.headers, body)

    raise SSRFBlockedError(f"Exceeded max redirects ({max_redirects}) for {url}")


async def safe_fetch_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_redirects: int = 5,
    max_bytes: int = 5 * 1024 * 1024,
) -> tuple[int, str, httpx.Headers, str]:
    """Wrapper over safe_fetch_bytes that decodes body safely using response encoding."""
    current_url = (url or "").strip()
    req_headers = dict(headers or {})

    for hop in range(max_redirects + 1):
        if not is_safe_public_url(current_url):
            raise SSRFBlockedError(f"Blocked unsafe URL: {current_url}")

        async with client.stream("GET", current_url, headers=req_headers) as response:
            if not is_safe_public_url(str(response.url)):
                raise SSRFBlockedError(f"Unsafe response URL rejected: {response.url}")

            if response.status_code in _ALLOWED_REDIRECT_CODES:
                location = response.headers.get("Location")
                if not location:
                    response.raise_for_status()
                if hop >= max_redirects:
                    raise SSRFBlockedError(f"Exceeded max redirects ({max_redirects}) for {url}")
                next_url = urljoin(str(response.url), location.strip())
                if not is_safe_public_url(next_url):
                    raise SSRFBlockedError(f"Unsafe redirect target rejected: {next_url}")
                current_url = next_url
                continue

            if response.status_code == 304:
                return (304, str(response.url), response.headers, "")

            response.raise_for_status()
            body_bytes = await _read_stream_capped(response, max_bytes)
            encoding = response.encoding or "utf-8"
            return (
                response.status_code,
                str(response.url),
                response.headers,
                body_bytes.decode(encoding, errors="replace"),
            )

    raise SSRFBlockedError(f"Exceeded max redirects ({max_redirects}) for {url}")


def sanitize_untrusted_text(text: str | None, max_len: int = 10_000) -> str:
    """Strip control bytes and neutralize prompt-delimiter tags in untrusted text."""
    if not text:
        return ""
    cleaned = _CONTROL_CHARS_RE.sub("", str(text))
    cleaned = _DELIMITER_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        cleaned,
    )
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned).strip()
    return cleaned[:max_len]


def detect_suspicious_injection(text: str | None) -> str | None:
    """Detect common prompt-injection phrases for security audit logging."""
    if not text:
        return None
    match = _SUSPICIOUS_INJECTION_RE.search(text)
    return match.group(0) if match else None


def generate_invite_token() -> str:
    """Generate a high-entropy URL-safe invite token."""
    return secrets.token_urlsafe(18)


def hash_invite_token(token: str) -> str:
    """Hash an invite token with SHA-256 before storing or looking up in SQLite."""
    return hashlib.sha256((token or "").strip().encode("utf-8")).hexdigest()


class CallbackSigner:
    """HMAC-SHA256 signer for Telegram inline callback_data (<= 64 bytes)."""

    def __init__(self, secret_material: str) -> None:
        self._key = hmac.new(
            (secret_material or "fallback-secret").encode("utf-8"),
            b"dailyai-cb-v1",
            hashlib.sha256,
        ).digest()

    def _compute_sig(self, user_id: int, action: str, ref: str) -> str:
        msg = f"{user_id}:{action}:{ref}".encode("utf-8")
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()[:12]

    def sign(self, user_id: int, action: str, ref: str) -> str:
        """Produce user-bound signed callback string within Telegram's 64-byte limit."""
        clean_action = action.strip()
        clean_ref = ref.strip()
        if ":" in clean_action or ":" in clean_ref:
            raise ValueError("Callback action and ref cannot contain ':'")
        sig = self._compute_sig(user_id, clean_action, clean_ref)
        payload = f"{clean_action}:{clean_ref}:{sig}"
        if len(payload.encode("utf-8")) > 64:
            raise ValueError(f"Callback payload exceeds 64 bytes: {len(payload)}")
        return payload

    def verify(self, user_id: int, data: str | None) -> tuple[str, str] | None:
        """Verify HMAC signature and user binding; return (action, ref) or None."""
        if not data:
            return None
        parts = data.split(":")
        if len(parts) != 3:
            return None
        action, ref, provided_sig = parts
        if not action or not ref or len(provided_sig) != 12:
            return None
        expected_sig = self._compute_sig(user_id, action, ref)
        if not hmac.compare_digest(provided_sig, expected_sig):
            return None
        return (action, ref)


class SlidingWindowRateLimiter:
    """In-memory per-key sliding-window rate limiter with bounded memory cleanup."""

    def __init__(self, max_keys: int = 2000) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._max_keys = max_keys

    def check(
        self,
        key: str,
        max_requests: int,
        window_seconds: float,
        *,
        now: float | None = None,
    ) -> tuple[bool, float]:
        """Record request if under limit; return (allowed, retry_after_seconds)."""
        current = now if now is not None else time.monotonic()
        cutoff = current - window_seconds
        bucket = self._buckets[key]
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= max_requests:
            retry_after = max(0.1, bucket[0] + window_seconds - current)
            return (False, round(retry_after, 1))

        bucket.append(current)
        if len(self._buckets) > self._max_keys:
            self._prune_empty(cutoff)
        return (True, 0.0)

    def _prune_empty(self, cutoff: float) -> None:
        stale_keys = [k for k, q in self._buckets.items() if not q or q[-1] <= cutoff]
        for k in stale_keys:
            self._buckets.pop(k, None)


class TelegramRateLimiter(BaseRateLimiter[Any]):
    """Zero-dependency python-telegram-bot BaseRateLimiter with flood control & RetryAfter backoff."""

    def __init__(
        self,
        *,
        overall_max_rate: float = 25.0,
        per_chat_interval: float = 1.0,
        max_retries: int = 2,
    ) -> None:
        self._overall_interval = 1.0 / max(1.0, overall_max_rate)
        self._per_chat_interval = max(0.0, per_chat_interval)
        self._max_retries = max(0, max_retries)
        self._lock: asyncio.Lock | None = None
        self._last_global_ts: float = 0.0
        self._last_chat_ts: dict[str, float] = {}

    async def initialize(self) -> None:
        self._lock = asyncio.Lock()

    async def shutdown(self) -> None:
        self._last_chat_ts.clear()

    async def _acquire_slot(self, chat_key: str | None) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            wait_global = max(0.0, self._last_global_ts + self._overall_interval - now)
            wait_chat = 0.0
            if chat_key is not None and self._per_chat_interval > 0:
                last_chat = self._last_chat_ts.get(chat_key, 0.0)
                wait_chat = max(0.0, last_chat + self._per_chat_interval - now)
            delay = max(wait_global, wait_chat)
            if delay > 0:
                await asyncio.sleep(delay)
            ts = time.monotonic()
            self._last_global_ts = ts
            if chat_key is not None:
                self._last_chat_ts[chat_key] = ts
                if len(self._last_chat_ts) > 1000:
                    cutoff = ts - 600.0
                    self._last_chat_ts = {
                        k: v for k, v in self._last_chat_ts.items() if v > cutoff
                    }

    async def process_request(
        self,
        callback: Callable[..., Coroutine[Any, Any, Any]],
        args: Any,
        kwargs: dict[str, Any],
        endpoint: str,
        data: dict[str, Any],
        rate_limit_args: Any | None,
    ) -> Any:
        raw_chat_id = data.get("chat_id")
        chat_key = str(raw_chat_id) if raw_chat_id is not None else None
        for attempt in range(self._max_retries + 1):
            await self._acquire_slot(chat_key)
            try:
                return await callback(*args, **kwargs)
            except RetryAfter as exc:
                if attempt >= self._max_retries:
                    raise
                raw_retry = exc.retry_after
                retry_secs = (
                    raw_retry.total_seconds()
                    if hasattr(raw_retry, "total_seconds")
                    else float(raw_retry)
                )
                sleep_for = retry_secs + 0.2
                logger.warning(
                    "Telegram RetryAfter on %s (attempt %d); sleeping %.1fs",
                    endpoint,
                    attempt + 1,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)


def redact_secrets(text: str) -> str:
    """Scrub Telegram tokens, API keys, Bearer headers, and invite codes from log text."""
    if not text:
        return text
    scrubbed = _TG_TOKEN_RE.sub("<REDACTED_TG_TOKEN>", text)
    scrubbed = _OPENAI_KEY_RE.sub("<REDACTED_API_KEY>", scrubbed)
    scrubbed = _BEARER_RE.sub(r"\1<REDACTED>", scrubbed)
    scrubbed = _INVITE_CODE_RE.sub(r"\1<REDACTED_INVITE>", scrubbed)
    return scrubbed


class SecretRedactingFilter(logging.Filter):
    """Logging filter that scrubs sensitive tokens before records hit handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
            scrubbed = redact_secrets(rendered)
            if scrubbed != rendered:
                record.msg = scrubbed
                record.args = ()
        except Exception:
            pass
        return True


def harden_file_permissions(path: Path) -> None:
    """Restrict sensitive local files (SQLite DB, .env) to owner read/write (0o600)."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except OSError:
        pass
