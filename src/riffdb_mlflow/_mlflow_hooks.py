"""Lightweight import hooks integrating RiffDB stores with MLflow's AI Gateway and Job Runner."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any

_TARGET_MODULES = frozenset(
    {
        "mlflow.server.gateway_api",
        "mlflow.store.tracking.gateway.config_resolver",
        "mlflow.server.jobs.utils",
        "mlflow.server.handlers",
    }
)


def _patch_gateway_api(module: ModuleType) -> None:
    if getattr(module, "_riffdb_patched", False):
        return
    orig_validate = getattr(module, "_validate_store", None)
    if callable(orig_validate):

        def _patched_validate_store(store: Any) -> None:
            if hasattr(store, "get_endpoint_config") or (
                type(store).__name__ == "RiffDBTrackingStore"
            ):
                return
            orig_validate(store)

        module._validate_store = _patched_validate_store  # type: ignore[attr-defined]
    module._riffdb_patched = True  # type: ignore[attr-defined]


def _patch_config_resolver(module: ModuleType) -> None:
    if getattr(module, "_riffdb_patched", False):
        return
    orig_get_ep = getattr(module, "get_endpoint_config", None)
    orig_get_res = getattr(module, "get_resource_endpoint_configs", None)

    if callable(orig_get_ep):

        def _patched_get_endpoint_config(endpoint_name: str, store: Any = None) -> Any:
            resolved_store = store if store is not None else module._get_store()
            if hasattr(resolved_store, "get_endpoint_config"):
                return resolved_store.get_endpoint_config(endpoint_name)
            return orig_get_ep(endpoint_name, store=resolved_store)

        module.get_endpoint_config = _patched_get_endpoint_config  # type: ignore[attr-defined]

    if callable(orig_get_res):

        def _patched_get_resource_endpoint_configs(
            resource_type: str, resource_id: str, store: Any = None
        ) -> Any:
            resolved_store = store if store is not None else module._get_store()
            if hasattr(resolved_store, "get_resource_endpoint_configs"):
                return resolved_store.get_resource_endpoint_configs(resource_type, resource_id)
            return orig_get_res(resource_type, resource_id, store=resolved_store)

        module.get_resource_endpoint_configs = (  # type: ignore[attr-defined]
            _patched_get_resource_endpoint_configs
        )

    module._riffdb_patched = True  # type: ignore[attr-defined]


def _patch_jobs_utils(module: ModuleType) -> None:
    if getattr(module, "_riffdb_patched", False):
        return
    orig_check = getattr(module, "_check_requirements", None)
    if callable(orig_check):

        def _patched_check_requirements(backend_store_uri: str | None = None) -> None:
            from mlflow.exceptions import MlflowException
            from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR

            if os.name == "nt":
                raise MlflowException("MLflow job backend does not support Windows system.")
            resolved_uri = backend_store_uri or os.environ.get(BACKEND_STORE_URI_ENV_VAR)
            if isinstance(resolved_uri, str) and resolved_uri.startswith("riffdb://"):
                from riffdb_mlflow.config import StoreConfig

                StoreConfig.from_uri(resolved_uri)
                return
            orig_check(backend_store_uri)

        module._check_requirements = _patched_check_requirements  # type: ignore[attr-defined]
    module._riffdb_patched = True  # type: ignore[attr-defined]


def _patch_server_handlers(module: ModuleType) -> None:
    if getattr(module, "_riffdb_patched", False):
        return
    orig_get_job_store = getattr(module, "_get_job_store", None)
    if callable(orig_get_job_store):

        def _patched_get_job_store(backend_store_uri: str | None = None) -> Any:
            from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR

            existing = getattr(module, "_job_store", None)
            if existing is not None:
                return existing
            store_uri = backend_store_uri or os.environ.get(BACKEND_STORE_URI_ENV_VAR, None)
            if isinstance(store_uri, str) and store_uri.startswith("riffdb://"):
                from riffdb_mlflow.job_store import RiffDBJobStore

                job_store = RiffDBJobStore(store_uri)
                module._job_store = job_store  # type: ignore[attr-defined]
                return job_store
            return orig_get_job_store(backend_store_uri)

        module._get_job_store = _patched_get_job_store  # type: ignore[attr-defined]
    module._riffdb_patched = True  # type: ignore[attr-defined]


def _apply_patch(fullname: str, module: ModuleType) -> None:
    if fullname == "mlflow.server.gateway_api":
        _patch_gateway_api(module)
    elif fullname == "mlflow.store.tracking.gateway.config_resolver":
        _patch_config_resolver(module)
    elif fullname == "mlflow.server.jobs.utils":
        _patch_jobs_utils(module)
    elif fullname == "mlflow.server.handlers":
        _patch_server_handlers(module)


class _RiffDBPostImportLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader, fullname: str) -> None:
        self._wrapped = wrapped_loader
        self._fullname = fullname

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._wrapped.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _apply_patch(self._fullname, module)


class _RiffDBMetaPathFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._in_progress: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _TARGET_MODULES or fullname in self._in_progress:
            return None
        self._in_progress.add(fullname)
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _RiffDBPostImportLoader(spec.loader, fullname)
                    return spec
            return None
        finally:
            self._in_progress.discard(fullname)


_FINDER_INSTALLED = False


def install_mlflow_hooks() -> None:
    """Install post-import hooks for MLflow AI Gateway and Job Runner integration."""
    global _FINDER_INSTALLED
    if not _FINDER_INSTALLED:
        if not any(isinstance(f, _RiffDBMetaPathFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _RiffDBMetaPathFinder())
        _FINDER_INSTALLED = True

    for mod_name in _TARGET_MODULES:
        loaded = sys.modules.get(mod_name)
        if loaded is not None:
            _apply_patch(mod_name, loaded)


install_mlflow_hooks()
