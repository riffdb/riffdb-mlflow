from __future__ import annotations

import inspect

import pytest
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.abstract_store import AbstractStore

from riffdb_mlflow.store import RiffDBTrackingStore

EXPECTED_ABSTRACT_METHODS = {
    "_search_runs",
    "create_experiment",
    "create_run",
    "delete_experiment",
    "delete_run",
    "get_experiment",
    "get_metric_history",
    "get_run",
    "link_traces_to_run",
    "log_batch",
    "log_inputs",
    "rename_experiment",
    "restore_experiment",
    "restore_run",
    "search_experiments",
    "update_run_info",
}


def test_pinned_mlflow_abstract_surface_is_reviewed() -> None:
    declared = {
        name
        for name, member in inspect.getmembers(AbstractStore, predicate=inspect.isfunction)
        if getattr(member, "__isabstractmethod__", False)
    }
    assert declared == EXPECTED_ABSTRACT_METHODS


def test_store_constructs_from_official_entrypoint_shape_and_requires_secure_config() -> None:
    store = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    assert store.database == "mlflow-development"
    assert store.artifact_uri == "s3://artifacts"
    with pytest.raises(MlflowException, match="Missing required RiffDB MLflow"):
        store.get_experiment("1")
