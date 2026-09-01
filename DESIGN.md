# Design

## Compatibility target

The first release pins MLflow `3.15.1` and registers the `riffdb` URI scheme in
the official `mlflow.tracking_store` entry-point group. The package implements
the tracking backend used for experiments, runs, metrics, parameters, tags,
inputs, and search. Artifact payloads remain in the artifact repository named
by MLflow's `artifact_uri` argument.

MLflow 3.15.1's tracking `AbstractStore` has sixteen abstract methods. The
compatibility gate records those methods and fails when a pinned upstream
upgrade changes the abstract surface. Concrete optional domains added by
MLflow—traces, assessments, evaluation datasets, scorers, review queues, and
similar GenAI metadata—remain explicitly unsupported until a profile and its
upstream tests are added. They must never silently fall back to another store.

The Python floor is 3.13, matching the published RiffDB Python driver rather
than MLflow's broader interpreter matrix.

## Core behavior

| Domain | Required behavior |
|---|---|
| Experiments | Unique names, create/get/search, rename, soft delete/restore, tags, stable pages |
| Runs | Create/get/search, lifecycle updates, soft delete/restore, exact latest data |
| Metrics | Atomic bounded batches, full paged history, exact IEEE-754 values, MLflow latest-value tie-breaking |
| Parameters | Immutable per-run key/value binding and duplicate-conflict behavior |
| Tags | Last-write-wins set and exact deletion semantics |
| Inputs | Atomic dataset/model input associations with deterministic bounded encoding |
| Authorization | Per-experiment and per-run row policy through the compiled profile, never Python filtering |

The SQL store is the behavioral reference. RiffDB is not treated as SQL and
the adapter does not emulate SQL transactions; it exposes equivalent MLflow
semantics through compiled commands and named queries.

## Configuration

MLflow constructs the plugin with `(store_uri, artifact_uri)`. The canonical
tracking URI is `riffdb://<database>`. It contains no credential or TLS
material. Connection authority comes from protected process configuration:

- `RIFFDB_MLFLOW_ENDPOINT`
- `RIFFDB_MLFLOW_TRUST_ROOT`
- `RIFFDB_MLFLOW_SERVER_NAME`
- `RIFFDB_MLFLOW_CREDENTIAL_FILE`

The credential file is consumed through RiffDB's nonextractable protected-file
API. The database in the URI becomes `CallMetadata`'s database alias. Unknown
URI components, user information, ports, paths, query parameters, and fragments
fail before any transport work.

## IDs and time

MLflow-facing experiment and run IDs remain strings with the behavior required
by upstream tests. MLflow's FileStore establishes that experiment IDs may be
random fixed-width integers rather than SQL sequences, so the adapter generates
an 18-digit numeric ID and submits it to one atomic create command. Run IDs use
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
orders fail with MLflow's invalid-parameter error.

Search result hydration must be bounded in the query or command result. N+1
reads, fetch-and-filter, adapter sorting, and offset simulation are prohibited.

## Error mapping

RiffDB application outcomes map to MLflow's documented error codes for missing
resources, duplicates, invalid states, invalid parameters, concurrency
conflicts, cancellation, deadline, and temporary unavailability. Unknown
transport or validation failures become a stable redacted internal error.
