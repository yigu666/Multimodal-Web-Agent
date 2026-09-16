from __future__ import annotations

import ipaddress
import socket
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..schemas import SearchBackendError


def canonical_url(url: str) -> str:
    parsed = urlsplit(str(url).strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SearchBackendError("BAD_URL", "only http/https URLs are allowed")
    if parsed.username or parsed.password:
        raise SearchBackendError("BAD_URL", "URL credentials are forbidden")
    host = (parsed.hostname or "").rstrip(".").casefold()
    if not host:
        raise SearchBackendError("BAD_URL", "URL host is missing")
    port = parsed.port
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = "[%s]" % host
    if port is not None and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        netloc += ":%d" % port
    path = parsed.path or "/"
    tracking_names = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_") and key.casefold() not in tracking_names
    ]
    return urlunsplit((parsed.scheme.lower(), netloc, path, urlencode(query_items, doseq=True), ""))


def _is_forbidden_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_public_url(url: str, *, resolver=socket.getaddrinfo) -> str:
    canonical = canonical_url(url)
    host = urlsplit(canonical).hostname or ""
    if host in {"localhost", "localhost.localdomain"}:
        raise SearchBackendError("BAD_URL", "local hostnames are forbidden")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_forbidden_ip(str(literal)):
            raise SearchBackendError("BAD_URL", "non-public IP address is forbidden")
        return canonical
    try:
        resolved = {row[4][0] for row in resolver(host, None)}
    except OSError as exc:
        raise SearchBackendError("PAGE_FETCH_ERROR", "DNS resolution failed") from exc
    if not resolved:
        raise SearchBackendError("PAGE_FETCH_ERROR", "DNS returned no addresses")
    if any(_is_forbidden_ip(address) for address in resolved):
        raise SearchBackendError("BAD_URL", "host resolves to a non-public address")
    return canonical
