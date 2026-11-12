"""RiffDB tracking-store integration for MLflow.

The package initializer deliberately does not import the store module. MLflow
discovers tracking-store entry points while its own package is initializing;
an eager import here creates a circular, partially initialized plugin module.
"""

from __future__ import annotations

import sys

if "mlflow" not in sys.modules:
    import mlflow as _mlflow  # noqa: F401

from riffdb_mlflow._mlflow_hooks import install_mlflow_hooks

install_mlflow_hooks()

__version__ = "0.2.5"
