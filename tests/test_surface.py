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


def test_extended_mlflow_3_15_tracking_surface_is_implemented() -> None:
    expected_extended_methods = {
        "start_trace",
        "get_trace_info",
        "batch_get_traces",
        "log_spans",
        "log_spans_async",
        "ingest_spans_out_of_order",
        "search_traces",
        "delete_traces",
        "set_trace_tag",
        "delete_trace_tag",
        "link_traces_to_run",
        "unlink_traces_from_run",
        "link_prompts_to_trace",
        "get_assessment",
        "create_assessment",
        "update_assessment",
        "delete_assessment",
        "query_trace_metrics",
        "get_span_model_rollups",
        "get_columnar_span_model_rollups",
        "create_logged_model",
        "get_logged_model",
        "delete_logged_model",
        "finalize_logged_model",
        "set_logged_model_tags",
        "delete_logged_model_tag",
        "log_logged_model_params",
        "search_logged_models",
        "record_logged_model",
        "log_inputs",
        "log_outputs",
    }
    for method_name in expected_extended_methods:
        method = getattr(RiffDBTrackingStore, method_name, None)
        assert callable(method), f"Missing {method_name} on RiffDBTrackingStore"
        base_method = getattr(AbstractStore, method_name, None)
        if base_method is not None:
            assert method is not base_method, f"{method_name} must override AbstractStore"
