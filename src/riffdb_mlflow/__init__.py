"""RiffDB tracking-store integration for MLflow.

The package initializer deliberately does not import the store module. MLflow
discovers tracking-store entry points while its own package is initializing;
an eager import here creates a circular, partially initialized plugin module.
"""

__version__ = "0.1.0.dev1"
