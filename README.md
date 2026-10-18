# riffdb-mlflow

`riffdb-mlflow` is a compiled RiffDB tracking-store plugin for MLflow. It is an
out-of-tree integration: no MLflow-specific code or vocabulary belongs in the
RiffDB repository.

The project targets **MLflow 3.15.1** and implements the complete `AbstractStore`
tracking, tracing, evaluation assessment, trace analytics, logged-model, and
run-lineage surface:

- **Experiments & Runs**: atomic experiment creation with tags, lifecycle and
  tag updates, `exact`/`LIKE`/`ILIKE` experiment search, atomic run creation
  with tags, run get/update/delete/restore, bounded atomic `log_batch`
  (metrics, parameters, tags), `get_metric_history`, `log_inputs`, `log_outputs`,
  and `record_logged_model`.
- **Traces & Out-of-Order Spans (`TraceInfo`, `Span`)**: `start_trace`,
  `get_trace_info`, `batch_get_traces`, `log_spans`, `log_spans_async`,
  `ingest_spans_out_of_order`, `search_traces`, `delete_traces`,
  `set_trace_tag`, `delete_trace_tag`, `link_traces_to_run`,
  `unlink_traces_from_run`, and `link_prompts_to_trace`. `IngestTraceSpans`
  uses `init_or_mutate` across `Trace` and `TraceSpan` in one partition commit,
  allowing child spans to arrive before parent/root spans and even before
  `start_trace()` without foreign-key failures or read-time `FINAL` merges.
- **Assessments (`Feedback` & `Expectation`)**: `create_assessment`,
  `get_assessment`, `update_assessment`, and `delete_assessment`.
- **Trace Metrics & Analytical Rollups (`query_trace_metrics`, `ADR-0315`)**:
  backed by contract-declared scalar columnar projections (`TraceAnalytics`,
  `SpanAnalytics`, `AssessmentAnalytics`) and event-driven rollups
  (`SpanModelRollup` with `min`/`max`/`mean`/`sum`/`count` and `hour(...)`
  temporal bucket keys) queried via `source projected` in `.riffq`.
- **Logged Models (`LoggedModel`)**: `create_logged_model`, `get_logged_model`,
  `delete_logged_model`, `finalize_logged_model`, `set_logged_model_tags`,
  `delete_logged_model_tag`, `log_logged_model_params`, and
  `search_logged_models`.

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
