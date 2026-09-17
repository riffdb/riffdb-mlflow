# riffdb-mlflow

`riffdb-mlflow` is a compiled RiffDB tracking-store plugin for MLflow. It is an
out-of-tree integration: no MLflow-specific code or vocabulary belongs in the
RiffDB repository.

The project currently targets MLflow 3.15.1 and is under active development.
It is not yet usable as a backend store.

The current validated development slice includes atomic experiment creation
with tags, experiment lifecycle and tag updates, experiment exact/LIKE/ILIKE
search, atomic run creation with tags, run get/update/delete/restore, and
lifecycle-aware globally ordered run search across a bounded experiment set.
Exact generation, static checks, unit tests, and disposable real-RiffDB
loopback cover that slice.

Run predicates beyond lifecycle and orders beyond `start_time DESC, run_id ASC`
remain unimplemented, as do inputs and the upstream tracking-store conformance
suite. The development tree now also implements bounded atomic batch logging
of metrics, parameters, and tags, plus metric history and latest-metric reads;
this addition passes static checks, 21 unit tests, and application compilation,
but is not yet qualified by the upstream tracking-store suite. Unsupported search shapes fail
before contacting RiffDB; the adapter does not repair them with client-side
filtering or sorting.

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

Source is public on GitHub. Package releases remain out of scope until real
RiffDB loopback and upstream MLflow semantics are green.


## RiffDB compatibility CI

This adapter owns its RiffDB compatibility checks. Pull requests compile the
application sources and verify existing locks and generated artifacts against
the full commit in `.github/riffdb-revision`. Scheduled checks use RiffDB
`main`; manual dispatch accepts a full commit or `main`. Failures are reported
here and do not block RiffDB itself. Framework conformance remains separate.

To upgrade, update the pin, migrate sources as needed, review the result of
`riffdb application lock --write`, and verify with `riffdb application check`.
Commit the pin and migration together in this repository.
