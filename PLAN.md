# Implementation plan

## WP-001 — Repository and upstream surface

Status: in progress.

- Pin MLflow `3.15.1` and record its exact source revision.
- Register the `riffdb` tracking-store URI scheme.
- Lock the sixteen-method abstract surface and inventory the SQL-store tests.
- Establish package, configuration, redaction, and stop boundaries.

Exit: a clean wheel discovers the plugin and upstream surface drift fails CI.

## WP-002 — Core contract and generated Python boundary

Status: pending.

- Model experiments, runs, metrics, latest metrics, parameters, tags, and
  inputs with explicit bounds and indexes.
- Compile every mutation as a named command and every read as a named query.
- Declare application source V7 with only the Python generated surface.
- Generate and check the exact manifest, V8 lock, and Python client.

Exit: source/lock checks are exact and all core data paths are generated.

## WP-003 — Experiments and runs

Status: pending.

- Implement create/get/rename/delete/restore for experiments.
- Implement create/get/update/delete/restore for runs.
- Preserve unique-name, lifecycle, random numeric ID, timestamp, and
  public-error semantics.
- Add focused concurrency and retry tests over real loopback.

## WP-004 — Batches and run data

Status: pending.

- Implement atomic `log_batch` for metrics, parameters, and tags.
- Implement latest metric selection and paged metric history.
- Implement experiment/run tags and dataset/model inputs.
- Prove maximum legal batches, duplicate parameters, IEEE-754 edge cases, and
  idempotent uncertainty recovery.

## WP-005 — Search and pagination

Status: pending.

- Inventory the MLflow experiment/run search grammar and order matrix.
- Map accepted normalized shapes to finite bounded named queries.
- Bind continuation tokens to operation, normalized filter, order, authority,
  and underlying RiffDB cursor.
- Prove pages without client filtering, sorting, or offset repair.

## WP-006 — Upstream and real-driver conformance

Status: pending.

- Run the applicable upstream SQL tracking-store semantics against RiffDB.
- Run an MLflow tracking server and public `MlflowClient` loopback suite.
- Compare correctness and cold/warm timing with PostgreSQL.
- Prove clean installation from an immutable local Python publication.

## WP-007 — Optional MLflow domains and release readiness

Status: deferred until the core tracking backend is green.

- Materialize separate reviewed profiles for optional newer MLflow domains.
- Add compatibility policy, CI, package provenance, and release metadata.
- Do not publish publicly until explicitly authorized.
