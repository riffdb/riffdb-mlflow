from __future__ import annotations

import pytest

from riffdb_mlflow.config import InvalidStoreURI, StoreConfig


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
