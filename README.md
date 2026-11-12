# riffdb-mlflow

`riffdb-mlflow` (`0.2.4`) is a compiled RiffDB tracking-store, model-registry,
and server job-store plugin for **MLflow 3.15.x** (`mlflow~=3.15.1`, including `3.15.1` and `3.15.2` patch releases) targeting **RiffDB 0.2.3**.
It is an out-of-tree integration: no MLflow-specific code, vocabulary, routes,
or schemas belong in the RiffDB repository.

## Supported MLflow 3.15.x Surface

The package registers the `riffdb` scheme in both official MLflow plugin groups:

- `mlflow.tracking_store` → `riffdb_mlflow.store:RiffDBTrackingStore`
- `mlflow.model_registry_store` → `riffdb_mlflow.registry_store:RiffDBModelRegistryStore`

| MLflow 3.15.1 Domain | Status | Notes |
|---|---|---|
| **Experiments & Runs** (`AbstractStore`) | Supported | Atomic experiment/run creation with tags, lifecycle transitions, `=`, `!=`, `LIKE`, `ILIKE`, `IS NULL`, and `IS NOT NULL` experiment search (including MLflow 3.15 UI gateway-tag filters), bounded atomic `log_batch` (IEEE-754 metrics, parameters, tags), `get_metric_history`, bulk metric history intervals, `log_inputs`, `log_outputs`, `_search_datasets`, and `record_logged_model`. |
| **Traces & Out-of-Order Spans** (`TraceInfo`, `Span`) | Supported | `start_trace`, `get_trace`, `get_trace_info`, `batch_get_traces`, `batch_get_trace_infos`, `log_spans`, `log_spans_async`, `ingest_spans_out_of_order`, `search_traces`, `delete_traces`, `set_trace_tag`, `delete_trace_tag`, `link_traces_to_run`, `unlink_traces_from_run`, `link_prompts_to_trace`, and `calculate_trace_filter_correlation`. |
| **Assessments, Label Schemas & Review Queues** (`Feedback`, `Expectation`, `LabelSchema`, `ReviewQueue`) | Supported | `create_assessment`, `get_assessment`, `update_assessment`, `delete_assessment`, `create_label_schema`, `list_label_schemas` (with default `"feedback"` seeding), `create_review_queue`, `get_or_create_user_queue`, `list_review_queues`, `add_items_to_review_queue`, `list_review_queue_items`, and `set_review_queue_item_status`. |
| **Evaluation Datasets & Issues** (`EvaluationDataset`, `Issue`) | Supported | `create_dataset`, `get_dataset`, `search_datasets`, `delete_dataset`, `upsert_dataset_records`, `_load_dataset_records`, `delete_dataset_records`, dataset tags/experiment links, `create_issue`, `get_issue`, `search_issues`, and `update_issue`. |
| **Columnar Trace Metrics & Rollups** (`query_trace_metrics`) | Supported | Contract-declared scalar columnar projections (`TraceAnalytics`, `SpanAnalytics`, `AssessmentAnalytics`) and event-driven rollups (`SpanModelRollup`) queried via `source projected` in RiffQL. Cold or zero-row projections report ready and return empty metric series without error on RiffDB `0.2.2+`. |
| **Logged Models** (`LoggedModel`) | Supported | `create_logged_model`, `get_logged_model`, `delete_logged_model`, `finalize_logged_model`, `set_logged_model_tags`, `delete_logged_model_tag`, `log_logged_model_params`, and `search_logged_models`. |
| **Model Registry** (`mlflow.model_registry_store`) | Supported | All 21 abstract methods on `mlflow.store.model_registry.AbstractStore`: registered models, model versions, stages, aliases, tags, and download URIs. Works automatically when `--registry-store-uri` defaults to `--backend-store-uri riffdb://<db>`. |
| **Prompt Registry** (`create_prompt`, `search_prompts`, etc.) | Supported | All 16 concrete Prompt Registry methods on `mlflow.store.model_registry.AbstractStore` backed by `RiffDBModelRegistryStore` registered-model and model-version primitives. |
| **Registry Webhooks** (`create_webhook`, `list_webhooks`, etc.) | Supported | All 6 webhook lifecycle and event-filtering methods on `RiffDBModelRegistryStore`. |
| **MCP Server Registry** (`MCPServerRegistryMixin`) | Supported | Implemented on `RiffDBTrackingStore` (`create_mcp_server`, `search_mcp_servers`, versions, aliases, tags, and access endpoints), backing `/ajax-api/3.0/mlflow/mcp-servers` and the MLflow 3.15 MCP Registry UI. |
| **UI-Managed AI Gateway & Guardrails** (`GatewayStoreMixin`) | Supported | All 36 `GatewayStoreMixin` methods (secrets with envelope KEK encryption and `SecretCache`, model definitions, endpoints, model mappings, fallback/routing configs, endpoint tags, resource bindings, budget policies, budget window spend, and gateway guardrails) plus live `/gateway/<endpoint>/mlflow/invocations` `config_resolver` dispatch. |
| **Scorers & Online Scoring** (`register_scorer`, `OnlineScoringConfig`) | Supported | Versioned scorer registration with automatic gateway endpoint-name-to-ID binding, `get_scorer`, `list_scorers`, `list_scorer_versions`, `delete_scorer`, `upsert_online_scoring_config`, `get_active_online_scorers`, and `find_completed_sessions`. |
| **Server-Side Job Runner** (`RiffDBJobStore` / `mlflow.server.jobs`) | Supported | Full 12-method `AbstractJobStore` implementation (`create_job`, `start_job`, `reset_job`, `finish_job`, `mark_job_timed_out`, `fail_job`, `retry_or_fail_job`, `list_jobs`, `get_job`, `cancel_job`, `update_status_details`, `delete_jobs`) automatically wired into `mlflow server` and Huey job workers when `--backend-store-uri` is `riffdb://<db>`. |

## Python Version Boundary & Optional Extras

- **Tracking Server / Direct `riffdb://` Store Process (`Python 3.11+`)**: The
  compiled RiffDB driver wheel targets `cp311-abi3`, so any process that loads
  `riffdb-mlflow` directly (such as `mlflow server` or a client using
  `MlflowClient(tracking_uri="riffdb://<db>")`) supports **Python 3.11 or newer** (`3.11`, `3.12`, `3.13`, `3.14`).
- **HTTP Tracking Clients (`Python 3.10+`)**: Standard MLflow training and
  serving workloads that connect to the tracking server over HTTP/HTTPS
  (`MLFLOW_TRACKING_URI=http://...`) do **not** install `riffdb-mlflow` and can
  run on any Python version supported by MLflow (`3.10+`).

Optional MLflow feature dependencies can be layered into the locked environment
using package extras:

```bash
# Install with standalone AI Gateway, GenAI, extras, or AWS artifact support:
uv sync --extra gateway
# or via pip:
pip install "riffdb-mlflow[gateway,genai,extras,aws]"
```

## Deploying the Application & Provisioning Credentials

Deploy the compiled application lock to a RiffDB `0.2.3+` server and provision
the `MLflowTrackingServer` role. When running from a read-only checkout or
installed package environment, pass `--deployment-dir` and `--credential-output`
so deployment state and mode-`0600` role credentials are written to an external
path rather than `.riffdb/deployments/` inside the working directory:

```bash
riffdb application deploy \
  --endpoint https://riffdb.internal:8443 \
  --database mlflow \
  --provision-role MLflowTrackingServer \
  --tls-trust-root /etc/riffdb/ca.pem \
  --tls-server-name riffdb.internal \
  --credential-file /etc/riffdb/bootstrap-admin.credential \
  --deployment-dir /var/lib/riffdb-mlflow/deployments \
  --credential-output /etc/riffdb-mlflow/tracking-server.credential
```

## Connection Configuration (`RIFFDB_MLFLOW_*`)

The tracking and registry URI is `riffdb://<database>`. It never contains
credentials or TLS parameters. All connection settings are read from environment
variables and validated at store startup before opening a RiffDB transport:

| Environment Variable | Validation Rules |
|---|---|
| `RIFFDB_MLFLOW_ENDPOINT` | Required. Must be `https://<host>[:<port>]` with an explicit host and no path, query, fragment, or userinfo. |
| `RIFFDB_MLFLOW_SERVER_NAME` | Required. Must be a valid hostname or IP that **exactly matches** the host in `RIFFDB_MLFLOW_ENDPOINT`. |
| `RIFFDB_MLFLOW_TRUST_ROOT` | Required. Must be an **absolute path** to an existing, readable regular PEM file containing `-----BEGIN CERTIFICATE-----`. |
| `RIFFDB_MLFLOW_CREDENTIAL_FILE` | Required. Must be an **absolute path** to a non-empty regular file (not a symlink), owned by the current OS user, with permissions `0600` or `0400`. |

If any variable is missing or violates these rules, `RiffDBTrackingStore` and
`RiffDBModelRegistryStore` immediately raise an `MlflowException`
(`INVALID_PARAMETER_VALUE`) naming the exact setting and rule.

## Running the MLflow Server & Clients

```bash
export RIFFDB_MLFLOW_ENDPOINT="https://riffdb.internal:8443"
export RIFFDB_MLFLOW_SERVER_NAME="riffdb.internal"
export RIFFDB_MLFLOW_TRUST_ROOT="/etc/riffdb/ca.pem"
export RIFFDB_MLFLOW_CREDENTIAL_FILE="/etc/riffdb-mlflow/tracking-server.credential"

mlflow server \
  --backend-store-uri riffdb://mlflow \
  --default-artifact-root s3://example/mlflow-artifacts
```

When using `MlflowClient(tracking_uri="riffdb://mlflow")` directly without
specifying an artifact root, the store defaults the artifact root to
`_MLFLOW_SERVER_ARTIFACT_ROOT` if set, or `./mlruns` in the current working
directory (matching MLflow's SQLite/FileStore fallback).

## Artifact Storage & Backup Coordination

By design, `riffdb-mlflow` stores tracking, tracing, and registry **metadata**
in RiffDB while artifact bytes (logged models, plots, files) remain in MLflow's
configured artifact repository (`s3://`, `gs://`, `abfss://`, or local
filesystem). Consistent disaster-recovery backups require coordinating RiffDB
database backups with backups of the underlying artifact storage.

## Development & Verification

```bash
uv sync --locked
make check
make conformance
```

## RiffDB Compatibility CI

This adapter owns its RiffDB compatibility checks. Pull requests compile the
application sources and verify existing locks and generated artifacts against
the full commit in [`.github/riffdb-revision`](.github/riffdb-revision). To
upgrade, update the pin, run `riffdb application lock --write` and
`riffdb generate`, and verify with `make check` and `make conformance`.
