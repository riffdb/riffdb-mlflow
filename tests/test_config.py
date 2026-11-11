from __future__ import annotations

import os
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

from riffdb_mlflow.config import ConnectionConfig, InvalidStoreURI, StoreConfig


def test_accepts_only_database_authority() -> None:
    assert StoreConfig.from_uri("riffdb://mlflow-development") == StoreConfig(
        database="mlflow-development"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "postgresql://mlflow",
        "riffdb://",
        "riffdb://user@mlflow",
        "riffdb://user:secret@mlflow",
        "riffdb://mlflow:7443",
        "riffdb://mlflow/profile",
        "riffdb://mlflow?role=admin",
        "riffdb://mlflow#fragment",
    ],
)
def test_rejects_noncanonical_or_secret_bearing_uri(uri: str) -> None:
    with pytest.raises(InvalidStoreURI):
        StoreConfig.from_uri(uri)


def test_connection_config_pre_flight_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    cred_file = tmp_path / "credential.token"
    cred_file.write_text("secret-token\n", encoding="utf-8")
    os.chmod(cred_file, 0o600)

    # 1. Missing environment variables
    for name in (
        "RIFFDB_MLFLOW_ENDPOINT",
        "RIFFDB_MLFLOW_TRUST_ROOT",
        "RIFFDB_MLFLOW_SERVER_NAME",
        "RIFFDB_MLFLOW_CREDENTIAL_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MlflowException, match="Missing required RiffDB MLflow"):
        ConnectionConfig.from_env()

    monkeypatch.setenv("RIFFDB_MLFLOW_ENDPOINT", "https://localhost:7443")
    monkeypatch.setenv("RIFFDB_MLFLOW_SERVER_NAME", "localhost")
    monkeypatch.setenv("RIFFDB_MLFLOW_TRUST_ROOT", str(ca_file))
    monkeypatch.setenv("RIFFDB_MLFLOW_CREDENTIAL_FILE", str(cred_file))

    valid = ConnectionConfig.from_env()
    assert valid.endpoint == "https://localhost:7443"
    assert valid.server_name == "localhost"
    assert valid.trust_root == str(ca_file)
    assert valid.credential_file == str(cred_file)

    # 2. Invalid endpoint scheme
    monkeypatch.setenv("RIFFDB_MLFLOW_ENDPOINT", "http://localhost:7443")
    with pytest.raises(MlflowException, match="RIFFDB_MLFLOW_ENDPOINT"):
        ConnectionConfig.from_env()
    monkeypatch.setenv("RIFFDB_MLFLOW_ENDPOINT", "https://localhost:7443")

    # 3. Server name mismatch with endpoint host
    monkeypatch.setenv("RIFFDB_MLFLOW_SERVER_NAME", "other-host.local")
    with pytest.raises(MlflowException, match="RIFFDB_MLFLOW_SERVER_NAME"):
        ConnectionConfig.from_env()
    monkeypatch.setenv("RIFFDB_MLFLOW_SERVER_NAME", "localhost")

    # 4. Relative trust root path
    monkeypatch.setenv("RIFFDB_MLFLOW_TRUST_ROOT", "relative/ca.pem")
    with pytest.raises(MlflowException, match="RIFFDB_MLFLOW_TRUST_ROOT"):
        ConnectionConfig.from_env()

    # 5. Symlink trust root path
    ca_symlink = tmp_path / "ca-link.pem"
    ca_symlink.symlink_to(ca_file)
    monkeypatch.setenv("RIFFDB_MLFLOW_TRUST_ROOT", str(ca_symlink))
    with pytest.raises(MlflowException, match="RIFFDB_MLFLOW_TRUST_ROOT"):
        ConnectionConfig.from_env()
    monkeypatch.setenv("RIFFDB_MLFLOW_TRUST_ROOT", str(ca_file))

    # 6. Permissive credential file mode (0644 instead of 0600)
    os.chmod(cred_file, 0o644)
    with pytest.raises(MlflowException, match="RIFFDB_MLFLOW_CREDENTIAL_FILE.*0600"):
        ConnectionConfig.from_env()
    os.chmod(cred_file, 0o600)

    # 7. Empty credential file
    empty_cred = tmp_path / "empty.token"
    empty_cred.write_text("   \n", encoding="utf-8")
    os.chmod(empty_cred, 0o600)
    monkeypatch.setenv("RIFFDB_MLFLOW_CREDENTIAL_FILE", str(empty_cred))
    with pytest.raises(MlflowException, match="RIFFDB_MLFLOW_CREDENTIAL_FILE.*empty"):
        ConnectionConfig.from_env()

    # 8. Driver socket mode (RIFFDB_DRIVER_SOCKET)
    monkeypatch.setenv("RIFFDB_MLFLOW_CREDENTIAL_FILE", str(cred_file))
    for name in (
        "RIFFDB_MLFLOW_ENDPOINT",
        "RIFFDB_MLFLOW_TRUST_ROOT",
        "RIFFDB_MLFLOW_SERVER_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RIFFDB_DRIVER_SOCKET", str(tmp_path / "driverd.sock"))
    socket_cfg = ConnectionConfig.from_env()
    assert socket_cfg.driver_socket == str(tmp_path / "driverd.sock")
    assert socket_cfg.credential_file == str(cred_file)

    monkeypatch.setenv("RIFFDB_DRIVER_SOCKET", "relative/driverd.sock")
    with pytest.raises(MlflowException, match="RIFFDB_DRIVER_SOCKET"):
        ConnectionConfig.from_env()
