# riffdb-mlflow

`riffdb-mlflow` is a compiled RiffDB tracking-store plugin for MLflow. It is an
out-of-tree integration: no MLflow-specific code or vocabulary belongs in the
RiffDB repository.

The project currently targets MLflow 3.15.1 and is under active development.
It is not yet usable as a backend store.

The intended server configuration is:

```bash
mlflow server \
  --backend-store-uri riffdb://mlflow-development \
  --default-artifact-root s3://example/mlflow-artifacts
```

RiffDB TLS and credential material is supplied through protected environment
configuration described in [DESIGN.md](DESIGN.md); it is never embedded in the
tracking URI.

Development uses the local RiffDB Python registry:

```bash
uv sync --index riffdb=http://127.0.0.1:4874/python/simple/
make check
```

Public publishing is intentionally out of scope until real RiffDB loopback and
upstream MLflow semantics are green.

