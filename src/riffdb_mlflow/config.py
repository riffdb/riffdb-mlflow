"""Strict, secret-free MLflow tracking URI parsing and connection pre-flight validation."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_REQUIRED_ENV_VARS = (
    "RIFFDB_MLFLOW_ENDPOINT",
    "RIFFDB_MLFLOW_TRUST_ROOT",
    "RIFFDB_MLFLOW_SERVER_NAME",
    "RIFFDB_MLFLOW_CREDENTIAL_FILE",
)


class InvalidStoreURI(ValueError):
    """Raised before transport work when an MLflow RiffDB URI is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """The non-secret configuration carried by an MLflow tracking or registry URI."""

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


@dataclass(frozen=True, slots=True)
class ConnectionConfig:
    """Validated RiffDB TLS or driver-socket and bearer-credential settings from environment variables."""

    endpoint: str
    trust_root: str
    server_name: str
    credential_file: str
    driver_socket: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ConnectionConfig:
        source = os.environ if env is None else env
        driver_socket = source.get("RIFFDB_MLFLOW_DRIVER_SOCKET") or source.get(
            "RIFFDB_DRIVER_SOCKET"
        )
        if driver_socket:
            credential_file = source.get("RIFFDB_MLFLOW_CREDENTIAL_FILE") or source.get(
                "RIFFDB_CREDENTIAL_FILE"
            )
            if not credential_file:
                raise MlflowException(
                    "Missing required RiffDB MLflow connection configuration: "
                    "RIFFDB_MLFLOW_CREDENTIAL_FILE",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            cls._validate_driver_socket(str(driver_socket))
            cls._validate_credential_file(str(credential_file))
            return cls(
                endpoint=str(source.get("RIFFDB_MLFLOW_ENDPOINT", "")),
                trust_root=str(source.get("RIFFDB_MLFLOW_TRUST_ROOT", "")),
                server_name=str(source.get("RIFFDB_MLFLOW_SERVER_NAME", "")),
                credential_file=str(credential_file),
                driver_socket=str(driver_socket),
            )

        required = {name: source.get(name) for name in _REQUIRED_ENV_VARS}
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise MlflowException(
                "Missing required RiffDB MLflow connection configuration: " + ", ".join(missing),
                error_code=INVALID_PARAMETER_VALUE,
            )

        endpoint = str(required["RIFFDB_MLFLOW_ENDPOINT"])
        trust_root = str(required["RIFFDB_MLFLOW_TRUST_ROOT"])
        server_name = str(required["RIFFDB_MLFLOW_SERVER_NAME"])
        credential_file = str(required["RIFFDB_MLFLOW_CREDENTIAL_FILE"])

        parsed_endpoint = cls._validate_endpoint(endpoint)
        if server_name != parsed_endpoint:
            raise MlflowException(
                f"Invalid RIFFDB_MLFLOW_SERVER_NAME: {server_name!r} must equal the host "
                f"of RIFFDB_MLFLOW_ENDPOINT ({parsed_endpoint!r})",
                error_code=INVALID_PARAMETER_VALUE,
            )

        cls._validate_trust_root(trust_root)
        cls._validate_credential_file(credential_file)

        return cls(
            endpoint=endpoint,
            trust_root=trust_root,
            server_name=server_name,
            credential_file=credential_file,
            driver_socket=None,
        )

    @staticmethod
    def _validate_driver_socket(driver_socket: str) -> None:
        if not os.path.isabs(driver_socket):
            raise MlflowException(
                "Invalid RIFFDB_DRIVER_SOCKET: must be an absolute Unix socket path",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @staticmethod
    def _validate_endpoint(endpoint: str) -> str:
        try:
            parsed = urlsplit(endpoint)
            port = parsed.port
        except ValueError as error:
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_ENDPOINT: must be an https://<host>:<port> URL",
                error_code=INVALID_PARAMETER_VALUE,
            ) from error

        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_ENDPOINT: must be an https://<host>:<port> URL "
                "without credentials, path, query, or fragment",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return parsed.hostname

    @staticmethod
    def _validate_trust_root(trust_root: str) -> None:
        if not os.path.isabs(trust_root):
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_TRUST_ROOT: must be an absolute file path",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if os.path.islink(trust_root):
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_TRUST_ROOT: must not be a symbolic link",
                error_code=INVALID_PARAMETER_VALUE,
            )
        try:
            st = os.lstat(trust_root)
        except OSError as error:
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_TRUST_ROOT: file does not exist or is unreadable "
                f"({trust_root})",
                error_code=INVALID_PARAMETER_VALUE,
            ) from error
        if not stat.S_ISREG(st.st_mode):
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_TRUST_ROOT: must be a regular file",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @staticmethod
    def _validate_credential_file(credential_file: str) -> None:
        if not os.path.isabs(credential_file):
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: must be an absolute file path",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if os.path.islink(credential_file):
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: must not be a symbolic link",
                error_code=INVALID_PARAMETER_VALUE,
            )
        try:
            st = os.lstat(credential_file)
        except OSError as error:
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: file does not exist or is unreadable",
                error_code=INVALID_PARAMETER_VALUE,
            ) from error
        if not stat.S_ISREG(st.st_mode):
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: must be a regular file",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: must be owned by the current user",
                error_code=INVALID_PARAMETER_VALUE,
            )
        mode = st.st_mode & 0o777
        if mode not in (0o600, 0o400):
            raise MlflowException(
                f"Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: file permissions must be 0600 or 0400 "
                f"(got {mode:04o})",
                error_code=INVALID_PARAMETER_VALUE,
            )
        try:
            with open(credential_file, "rb") as handle:
                content = handle.read().strip()
        except OSError as error:
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: file does not exist or is unreadable",
                error_code=INVALID_PARAMETER_VALUE,
            ) from error
        if st.st_size == 0 or not content:
            raise MlflowException(
                "Invalid RIFFDB_MLFLOW_CREDENTIAL_FILE: credential file must not be empty",
                error_code=INVALID_PARAMETER_VALUE,
            )
