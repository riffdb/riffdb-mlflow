"""Strict, secret-free MLflow tracking URI parsing."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


class InvalidStoreURI(ValueError):
    """Raised before transport work when an MLflow RiffDB URI is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """The non-secret configuration carried by an MLflow tracking URI."""

    database: str

    @classmethod
    def from_uri(cls, store_uri: str) -> StoreConfig:
        parsed = urlsplit(store_uri)
        if parsed.scheme != "riffdb":
            raise InvalidStoreURI("tracking store URI must use the riffdb scheme")
        if parsed.username is not None or parsed.password is not None:
            raise InvalidStoreURI("tracking store URI must not contain credentials")
        if parsed.port is not None:
            raise InvalidStoreURI("tracking store URI must not contain a port")
        if not parsed.hostname:
            raise InvalidStoreURI("tracking store URI must name one database")
        if parsed.path not in ("", "/"):
            raise InvalidStoreURI("tracking store URI must not contain a path")
        if parsed.query:
            raise InvalidStoreURI("tracking store URI must not contain query parameters")
        if parsed.fragment:
            raise InvalidStoreURI("tracking store URI must not contain a fragment")
        if parsed.hostname != parsed.netloc:
            raise InvalidStoreURI("tracking store URI contains unsupported authority syntax")
        return cls(database=parsed.hostname)
