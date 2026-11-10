from __future__ import annotations

import inspect

import pytest
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.abstract_store import AbstractStore as ModelRegistryAbstractStore
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.gateway.abstract_mixin import GatewayStoreMixin
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import MCPServerRegistryMixin

from riffdb_mlflow.registry_store import RiffDBModelRegistryStore
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

EXPECTED_REGISTRY_ABSTRACT_METHODS = {
    "create_model_version",
    "create_registered_model",
    "delete_model_version",
    "delete_model_version_tag",
    "delete_registered_model",
    "delete_registered_model_alias",
    "delete_registered_model_tag",
    "get_latest_versions",
    "get_model_version",
    "get_model_version_by_alias",
    "get_model_version_download_uri",
    "get_registered_model",
    "rename_registered_model",
    "search_model_versions",
    "search_registered_models",
    "set_model_version_tag",
    "set_registered_model_alias",
    "set_registered_model_tag",
    "transition_model_version_stage",
    "update_model_version",
    "update_registered_model",
}


def test_pinned_mlflow_abstract_surface_is_reviewed() -> None:
    declared = {
        name
        for name, member in inspect.getmembers(AbstractStore, predicate=inspect.isfunction)
        if getattr(member, "__isabstractmethod__", False)
    }
    assert declared == EXPECTED_ABSTRACT_METHODS


def test_pinned_mlflow_model_registry_surface_is_implemented() -> None:
    declared = {
        name
        for name, member in inspect.getmembers(
            ModelRegistryAbstractStore, predicate=inspect.isfunction
        )
        if getattr(member, "__isabstractmethod__", False)
    }
    assert declared == EXPECTED_REGISTRY_ABSTRACT_METHODS
    store = RiffDBModelRegistryStore("riffdb://mlflow-development")
    assert store.database == "mlflow-development"
    for webhook_method in (
        "create_webhook",
        "get_webhook",
        "list_webhooks",
        "list_webhooks_by_event",
        "update_webhook",
        "delete_webhook",
    ):
        method = getattr(RiffDBModelRegistryStore, webhook_method, None)
        assert callable(method), f"Missing {webhook_method} on RiffDBModelRegistryStore"
        base_method = getattr(ModelRegistryAbstractStore, webhook_method, None)
        assert method is not base_method


def test_store_constructs_from_official_entrypoint_shape_and_requires_secure_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "RIFFDB_MLFLOW_ENDPOINT",
        "RIFFDB_MLFLOW_TRUST_ROOT",
        "RIFFDB_MLFLOW_SERVER_NAME",
        "RIFFDB_MLFLOW_CREDENTIAL_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
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
        "create_mcp_server",
        "get_mcp_server",
        "search_mcp_servers",
        "update_mcp_server",
        "delete_mcp_server",
        "create_mcp_server_version",
        "get_mcp_server_version",
        "get_mcp_server_version_by_alias",
        "get_latest_mcp_server_version",
        "search_mcp_server_versions",
        "update_mcp_server_version",
        "delete_mcp_server_version",
        "create_mcp_access_endpoint",
        "get_mcp_access_endpoint",
        "search_mcp_access_endpoints",
        "update_mcp_access_endpoint",
        "delete_mcp_access_endpoint",
        "set_mcp_server_tag",
        "delete_mcp_server_tag",
        "set_mcp_server_version_tag",
        "delete_mcp_server_version_tag",
        "set_mcp_server_alias",
        "delete_mcp_server_alias",
    }
    for method_name in expected_extended_methods:
        method = getattr(RiffDBTrackingStore, method_name, None)
        assert callable(method), f"Missing {method_name} on RiffDBTrackingStore"
        base_method = getattr(AbstractStore, method_name, None)
        if base_method is not None:
            assert method is not base_method, f"{method_name} must override AbstractStore"
        mcp_base = getattr(MCPServerRegistryMixin, method_name, None)
        if mcp_base is not None:
            assert method is not mcp_base, f"{method_name} must override MCPServerRegistryMixin"


def test_gateway_mixin_methods_raise_explicit_mlflow_exception() -> None:
    store = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    for name, _member in inspect.getmembers(GatewayStoreMixin, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        method = getattr(store, name)
        with pytest.raises(MlflowException, match="SQLAlchemy tracking store"):
            method()
