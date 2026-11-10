# Implementation plan

## WP-001 — Repository and upstream surface

Status: complete (`v0.2.1`).

- Pin MLflow `3.15.1` and RiffDB `0.2.1` (`.github/riffdb-revision`).
- Register the `riffdb` URI scheme for both `mlflow.tracking_store` and
  `mlflow.model_registry_store`.
- Lock the sixteen-method tracking abstract surface, twenty-one-method model
  registry abstract surface, and concrete 3.15 extended surfaces in
  `tests/test_surface.py`.
- Establish package, pre-flight connection configuration validation
  (`src/riffdb_mlflow/config.py`), optional extras (`gateway`, `genai`,
  `extras`, `aws`), redaction, and stop boundaries.

## WP-002 — Core contract and generated Python boundary

Status: complete (`v0.2.1`).

- Model experiments, runs, metrics, latest metrics, parameters, tags, inputs,
  traces, spans, assessments, columnar projections, logged models, registered
  models, model versions, registry webhooks, and MCP server registry entities
  with explicit bounds and indexes in `riffdb/contract.riff`.
- Compile every mutation as a named command (36 commands) and every read as a
  named RiffQL query (64 queries).
- Generate and verify `riffdb.application.json`, `riffdb.application.lock.json`,
  and `src/riffdb_mlflow/generated/client.py`.

## WP-003 — Experiments and runs

Status: complete (`v0.2.1`).

- Implement create/get/rename/delete/restore for experiments with default
  artifact root fallback (`_MLFLOW_SERVER_ARTIFACT_ROOT` or `./mlruns`).
- Implement create/get/update/delete/restore for runs.
- Preserve unique-name, lifecycle, random numeric ID, timestamp, and
  public-error semantics.

## WP-004 — Batches, run data, traces, and logged models

Status: complete (`v0.2.1`).

- Implement atomic `log_batch` for metrics, parameters, and tags.
- Implement latest metric selection and paged metric history with exact
  IEEE-754 binary64 encoding.
- Implement dataset/model inputs, outputs, and `LoggedModel` operations.
- Implement `TraceInfo`, out-of-order `Span` ingestion, `Assessment` CRUD, and
  columnar `query_trace_metrics` rollups (`TraceAnalytics`, `SpanAnalytics`,
  `AssessmentAnalytics`, `SpanModelRollup`).

## WP-005 — Search and pagination

Status: complete (`v0.2.1`).

- Support MLflow experiment search for `=`, `!=`, `LIKE`, `ILIKE`, `IS NULL`,
  and `IS NOT NULL` on `name` and `tags.*`, including 2-predicate and
  3-predicate MLflow 3.15 UI filter combinations (`mlflow.experiment.isGateway`).
- Support lifecycle-aware run, trace, logged-model, registered-model,
  model-version, prompt, and MCP server search and bounded pagination.
- Bind continuation tokens to operation, normalized filter, order, authority,
  and underlying RiffDB cursor.

## WP-006 — Model Registry, Prompt Registry, Webhooks, and MCP Server Registry

Status: complete (`v0.2.1`).

- Implement `RiffDBModelRegistryStore` covering all 21 abstract model registry
  methods, unlocking all 16 concrete Prompt Registry methods, plus all 6
  Registry Webhook methods.
- Implement `RiffDBMCPServerRegistryMixin` on `RiffDBTrackingStore` covering
  MCP servers, versions, aliases, tags, and access endpoints.
- Override `GatewayStoreMixin` methods on `RiffDBTrackingStore` to fail with a
  clear `MlflowException` explaining upstream MLflow's `SqlAlchemyStore` gate.

## WP-007 — Live RiffDB 0.2.1 conformance and release verification

Status: complete (`v0.2.1`).

- Verify cold columnar projection readiness and full tracking, tracing,
  analytics, logged-model, experiment UI filter, MCP server registry, model
  registry, prompt registry, and webhook workflows over real RiffDB `0.2.1`
  loopback via `make check` and `make conformance`.
