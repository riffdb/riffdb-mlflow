# Design

## Compatibility target

`riffdb-mlflow` (`0.2.2`) pins **MLflow `3.15.1`** and **RiffDB `0.2.2`** and
registers the `riffdb` URI scheme in both official MLflow entry-point groups:

- `mlflow.tracking_store` → `riffdb_mlflow.store:RiffDBTrackingStore`
- `mlflow.model_registry_store` → `riffdb_mlflow.registry_store:RiffDBModelRegistryStore`

The package implements the tracking, tracing, evaluation assessment, columnar
trace metrics, logged-model, MCP server registry, UI-managed AI Gateway,
gateway guardrails, evaluation scorers & online scoring, model registry, prompt
registry, registry webhook, and server-side job store (`RiffDBJobStore`)
backends. Artifact payloads remain in the artifact repository named by MLflow's
`artifact_uri` argument (or default to `_MLFLOW_SERVER_ARTIFACT_ROOT` /
`./mlruns` when `artifact_uri` is omitted).

MLflow 3.15.1's tracking `AbstractStore` has sixteen abstract methods, its
model registry `AbstractStore` has twenty-one abstract methods, and its
`AbstractJobStore` has twelve abstract methods. The compatibility gate in
`tests/test_surface.py` records all three abstract surfaces as well as the
concrete extended surfaces (`TraceInfo`/`Span`, `Assessment`, `LoggedModel`,
`MCPServerRegistryMixin`, `GatewayStoreMixin`, `Scorer`, `Prompt`, and
`Webhook`) and fails when a pinned upstream upgrade changes the surface.

Because upstream MLflow 3.15 hardcodes `isinstance(store, SqlAlchemyStore)` or
SQL URI scheme checks in four places (`mlflow.server.gateway_api._validate_store`,
`mlflow.store.tracking.gateway.config_resolver`, `mlflow.server.jobs.utils._check_requirements`,
and `mlflow.server.handlers._get_job_store`), `riffdb-mlflow` installs a
lightweight `sys.meta_path` import hook (`riffdb_mlflow._mlflow_hooks`,
autoloaded via `riffdb_mlflow_autoload.pth` on Python startup in both the
`mlflow server` parent process and spawned Huey job-runner child processes)
that delegates `riffdb://` URIs and `RiffDBTrackingStore` instances to
`RiffDBGatewayAndScorerMixin` and `RiffDBJobStore` while leaving all non-RiffDB
stores untouched.

The Python floor for processes loading `riffdb-mlflow` directly is **3.13**,
matching the published `cp313-abi3` RiffDB Python driver. HTTP clients talking
to `mlflow server` over REST are unaffected and support Python 3.10–3.13.

## Core behavior

| Domain | Required behavior |
|---|---|
| Experiments | Unique names, create/get/search (`=`, `!=`, `LIKE`, `ILIKE`, `IS NULL`, `IS NOT NULL`, and 2-/3-predicate UI combinations including `mlflow.experiment.isGateway`), rename, soft delete/restore, tags, stable pages |
| Runs | Create/get/search, lifecycle updates, soft delete/restore, exact latest data |
| Metrics | Atomic bounded batches, full paged history, exact IEEE-754 values, MLflow latest-value tie-breaking |
| Parameters | Immutable per-run key/value binding and duplicate-conflict behavior |
| Tags | Last-write-wins set and exact deletion semantics |
| Inputs & Outputs | Atomic dataset/model input and output associations with deterministic bounded encoding |
| Traces & Spans | Single-trace retrieval (`get_trace`, `get_trace_info`, `batch_get_traces`, `batch_get_trace_infos`), out-of-order span ingestion (`init_or_mutate`), trace tags, run/prompt links, search, filter correlation, and assessments |
| Evaluation Datasets, Label Schemas, Review Queues & Issues | Managed evaluation dataset CRUD and record upserts (`create_dataset`, `upsert_dataset_records`, `search_datasets`), run dataset summaries (`_search_datasets`), label schemas with default `"feedback"` question seeding, human review queues (`get_or_create_user_queue`, item status transitions), and automated issue discovery (`create_issue`, `search_issues`, `update_issue`) |
| Trace Analytics | Contract-declared scalar columnar projections (`TraceAnalytics`, `SpanAnalytics`, `AssessmentAnalytics`) and event-driven rollups (`SpanModelRollup`). Cold projections report ready prior to first demand and return empty series when zero rows match in RiffDB `0.2.2+` |
| Logged Models | Full `LoggedModel` lifecycle, params, tags, and search |
| Model & Prompt Registry | Registered models, versions, stage transitions, aliases, tags, prompt registry (`create_prompt`, `create_prompt_version`, `search_prompts`), and registry webhooks |
| MCP Server Registry | MCP servers, versions, aliases, tags, and access endpoints via `RiffDBMCPServerRegistryMixin` |
| AI Gateway & Guardrails | Encrypted secrets (`KEKManager` + `SecretCache`), model definitions, endpoints, mappings, tags, resource bindings, budget policies, budget window spend, guardrails, and live invocation config resolution via `RiffDBGatewayAndScorerMixin` |
| Scorers & Online Scoring | Versioned scorer CRUD, gateway endpoint binding resolution, online scoring configs, active online scorers, and completed session discovery |
| Server Job Store | Atomic job lifecycle transitions (`PENDING` → `RUNNING` → `SUCCEEDED`/`FAILED`/`TIMEOUT`/`CANCELED`), retry/reset, filtering, and deletion via `RiffDBJobStore(AbstractJobStore)` |
| Authorization | Role-based contract policy through the compiled `MLflowTrackingServer` profile, never Python filtering |

The SQL store is the behavioral reference. RiffDB is not treated as SQL and
the adapter does not emulate SQL transactions; it exposes equivalent MLflow
semantics through 58 compiled RiffDB commands and 86 named RiffQL queries.

## Configuration

MLflow constructs the tracking plugin with `(store_uri, artifact_uri)` and the
model registry plugin with `(store_uri, tracking_uri)`. The canonical URI is
`riffdb://<database>`. It contains no credential or TLS material. Connection
authority comes from protected process configuration validated by
`ConnectionConfig.from_env()` before any transport initialization:

- `RIFFDB_MLFLOW_ENDPOINT`: `https://<host>[:<port>]` without path, query, fragment, or userinfo.
- `RIFFDB_MLFLOW_SERVER_NAME`: hostname or IP matching the endpoint host.
- `RIFFDB_MLFLOW_TRUST_ROOT`: absolute path to a readable PEM certificate file (`-----BEGIN CERTIFICATE-----`).
- `RIFFDB_MLFLOW_CREDENTIAL_FILE`: absolute path to a non-empty regular file (not a symlink), owned by the current user, with mode `0600` or `0400`.

The credential file is consumed through RiffDB's nonextractable protected-file
API. The database in the URI becomes `CallMetadata`'s database alias. Unknown
URI components, user information, ports, paths, query parameters, fragments, or
invalid `RIFFDB_MLFLOW_*` settings fail immediately with
`MlflowException(error_code=INVALID_PARAMETER_VALUE)` naming the offending
setting.

## IDs and time

MLflow-facing experiment and run IDs remain strings with the behavior required
by upstream tests. MLflow's FileStore establishes that experiment IDs may be
random fixed-width integers rather than SQL sequences, so the adapter generates
a 9-digit numeric ID (fitting within MLflow's 32-bit `ProtoScorer.experiment_id`
field) and submits it to one atomic create command. Run IDs use
MLflow's UUID-shaped convention. Millisecond values supplied by MLflow remain
exact signed integers. Adapter wall-clock time never establishes durable
ordering, state transitions, or lease validity.

## Metric representation

An MLflow metric value is an IEEE-754 binary64, not a business decimal. The
adapter preserves its canonical 64-bit payload, including signed zero, NaN, and
infinities. It also derives a bytewise sortable key whose ordering is proven
against MLflow's numeric comparison behavior. The contract stores both values;
neither is rounded through decimal or JSON. Latest metrics are selected by
MLflow's `(step, timestamp, value)` precedence in a named query.

## Query compilation boundary

MLflow accepts a textual search grammar. Python parses that grammar using
MLflow's own parser and normalizes it into a closed internal shape. The shape
selects a generated named query; it never supplies field, operator, index,
order, or raw predicate text to RiffDB. Unsupported conjunctions, fields, or
orders fail with MLflow's invalid-parameter error before contacting RiffDB.

Search result hydration is bounded in the query or command result. N+1 reads,
fetch-and-filter, adapter sorting, and offset simulation are prohibited.

## Error mapping

RiffDB application outcomes map to MLflow's documented error codes for missing
resources, duplicates, invalid states, invalid parameters, concurrency
conflicts, cancellation, deadline, and temporary unavailability. Unknown
transport or validation failures become a stable redacted internal error.
