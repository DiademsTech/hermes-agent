"""Logins a member approved from their PC password manager, for ``browser_vault`` (Diadems).

Registers the ``pcpass`` login backend (handles ``pcpass:<request_id>``). It lists only the logins
a member approved for the CURRENT conversation (Diadems route ``/v1/capsule/pc-passwords/list``) and
resolves a password or one-time code at fill time: it makes a one-off RSA-2048 key pair, asks
Diadems to have the member's PC encrypt the value for that key
(``/v1/capsule/pc-passwords/release``) and decrypts the envelope in memory. Hermes checks the page
origin against ``allowed_origins`` before resolving, wraps the backend (``PluginLoginBackend``) and
fills the page itself; the value never reaches the model or the transcript.

Bundled so ``ctx.register_login_backend`` accepts it (user plugins are refused). Off unless the
profile lists ``diadems-pc-passwords`` in ``plugins.enabled`` and sets ``vault.pcpass.enabled:
true``. The Diadems addresses come from ``$HERMES_HOME/.diadems/pc-passwords.json`` (non-secret,
read at call time); without that file the backend lists nothing. The agent key is the profile's
``DIADEMS_API_KEY``.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request

from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta

logger = logging.getLogger(__name__)

SETTINGS_FILE = (".diadems", "pc-passwords.json")
_PREFIX = "pcpass:"
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_LIST_TIMEOUT = 8
_RELEASE_TIMEOUT = 35
_CACHE_SECONDS = 30
_MAX_BODY = 262_144
_MAX_SETTINGS = 4096
_ALG = "RSA-OAEP-256+A256GCM"
FAILED = "The PC password could not be released."
# Codes Diadems may answer; anything else is logged as "error" (never server text).
_CODES = frozenset({
    "approval_not_found", "approval_expired", "approval_used", "not_member_turn",
    "pc_offline", "pc_timeout", "login_changed", "manager_locked", "otp_unavailable",
})


def _home():
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _key():
    """The profile's agent key; never another profile's (a scope error means no key)."""
    try:
        from agent.secret_scope import get_secret
    except ImportError:
        return os.environ.get("DIADEMS_API_KEY", "")
    try:
        return get_secret("DIADEMS_API_KEY") or ""
    except Exception:  # UnscopedSecretError and the like: fail closed
        return ""


def _session_id():
    """The conversation of the running turn (the ContextVar first, as Hermes reads it)."""
    try:
        from gateway.session_context import get_session_env

        value = get_session_env("HERMES_SESSION_ID", "")
    except Exception:  # outside the gateway there is no portal conversation
        value = ""
    value = value if isinstance(value, str) else ""
    return value if value.startswith("dia-") and not value.startswith("dia-email-") else ""


def _url(name):
    """One Diadems address from the rendered settings; raises when absent or malformed."""
    path = _home().joinpath(*SETTINGS_FILE)
    with open(path, "rb") as handle:
        raw = handle.read(_MAX_SETTINGS + 1)
    if len(raw) > _MAX_SETTINGS:
        raise ValueError("settings too large")
    settings = json.loads(raw)
    value = settings.get(name) if isinstance(settings, dict) else None
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        raise ValueError("no Diadems address")
    return value


def _post(url, body, timeout):
    """(status, JSON object); an HTTP error answers its JSON body too. Raises on transport failure."""
    key = _key()
    if not key:
        raise ValueError("no agent key")
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # fixed https/http URL
            status, raw = response.status, response.read(_MAX_BODY + 1)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(_MAX_BODY + 1)
    if len(raw) > _MAX_BODY:
        raise ValueError("answer too large")
    value = json.loads(raw or b"{}")
    return status, value if isinstance(value, dict) else {}


def _text(value, limit=512):
    return value[:limit] if isinstance(value, str) and value else None


def meta_for(login, now):
    """A VaultItemMeta for one approved login, or None when Diadems' entry is malformed."""
    if not isinstance(login, dict):
        return None
    handle = login.get("handle")
    if (not isinstance(handle, str) or not handle.startswith(_PREFIX)
            or not _REQUEST_ID.fullmatch(handle[len(_PREFIX):])):
        return None
    origin = _text(login.get("origin"))
    raw = login.get("allowed_origins")
    origins = [o for o in (raw if isinstance(raw, list) else []) if isinstance(o, str) and o.startswith("https://")]
    if origin and origin not in origins and origin.startswith("https://"):
        origins.insert(0, origin)
    if not origins:
        return None  # Hermes would refuse to fill an item bound to no origin anyway
    label = _text(login.get("label"), 200) or "Login"
    provider = _text(login.get("provider"), 60)
    identifier = _text(login.get("identifier"), 320)
    return VaultItemMeta(
        id=handle, kind="login", label=f"{label} ({provider})" if provider else label,
        origin=origin if origin in origins else origins[0], created_at=now, identifier=identifier,
        identifier_type=("email" if "@" in identifier else "username") if identifier else None,
        has_otp=login.get("has_otp") is True, allowed_origins=tuple(origins[:20]),
    )


def decrypt_envelope(private_key, envelope, aad):
    """The value of an ``RSA-OAEP-256+A256GCM`` envelope bound to *aad*; raises on any mismatch."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not isinstance(envelope, dict) or envelope.get("alg") != _ALG:
        raise ValueError("unknown envelope")
    wrapped, iv, data = (base64.b64decode(envelope[k], validate=True) for k in ("key", "iv", "data"))
    aes_key = private_key.decrypt(wrapped, padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    if len(aes_key) != 32 or len(iv) != 12:
        raise ValueError("bad envelope")
    value = json.loads(AESGCM(aes_key).decrypt(iv, data, aad.encode("utf-8"))).get("value")
    if not isinstance(value, str) or not value:
        raise ValueError("empty value")
    return value


def _release(handle, field):
    """("ok", value) or (code, None); never logs the value."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    session = _session_id()
    if not session or not isinstance(handle, str) or not handle.startswith(_PREFIX):
        return "no_session", None
    request_id = handle[len(_PREFIX):]
    if not _REQUEST_ID.fullmatch(request_id):
        return "bad_handle", None
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")
    status, answer = _post(_url("url_release"), {
        "session_id": session, "handle": handle, "field": field, "public_key": public_pem,
    }, _RELEASE_TIMEOUT)
    if status != 200 or answer.get("ok") is not True:
        code = answer.get("code")
        return (code if code in _CODES else "error"), None
    return "ok", decrypt_envelope(private_key, answer.get("envelope"), f"{request_id}:{field}")


class PcPasswordBackend(LoginBackend):
    name = "pcpass"
    prefix = "pcpass:"
    display_name = "PC password manager"
    needs_unlock = False

    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}  # session id -> (monotonic time, {handle: VaultItemMeta})

    def list_items(self):
        session = _session_id()
        if not session:
            return []
        try:
            status, answer = _post(_url("url_list"), {"session_id": session}, _LIST_TIMEOUT)
            logins = answer.get("logins") if status == 200 and answer.get("ok") is True else None
            if not isinstance(logins, list):
                return []
            now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            items = [m for m in (meta_for(login, now) for login in logins[:50]) if m is not None]
        except FileNotFoundError:
            return []  # Diadems did not render the relay for this profile
        except Exception as error:  # a listing failure only hides these logins
            logger.info("PC passwords not listed: %s", type(error).__name__)
            return []
        now_at = time.monotonic()
        with self._lock:
            self._cache = {s: entry for s, entry in self._cache.items() if now_at - entry[0] < _CACHE_SECONDS}
            self._cache[session] = (now_at, {m.id: m for m in items})
        return items

    def get_meta(self, handle):
        session = _session_id()
        if not session or not isinstance(handle, str) or not handle.startswith(_PREFIX):
            return None
        with self._lock:
            at, items = self._cache.get(session, (0.0, {}))
        if handle in items and time.monotonic() - at < _CACHE_SECONDS:
            return items[handle]
        return next((m for m in self.list_items() if m.id == handle), None)

    def _resolve(self, handle, field):
        try:
            outcome, value = _release(handle, field)
        except Exception as error:  # the error text is never passed on
            outcome, value = type(error).__name__, None
        if outcome != "ok":
            logger.info("PC password %s not released: %s", field, outcome)
        return outcome, value

    def resolve_password(self, handle):
        _outcome, value = self._resolve(handle, "password")
        if value is None:
            raise RuntimeError(FAILED)
        return value

    def resolve_otp(self, handle):
        outcome, value = self._resolve(handle, "otp")
        if outcome == "otp_unavailable":
            return None
        if value is None:
            raise RuntimeError(FAILED)
        return value


def register(ctx):
    ctx.register_login_backend(PcPasswordBackend())
