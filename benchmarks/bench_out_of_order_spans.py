from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.entities.trace_state import TraceState
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import (
    SpanAttributeKey,
    SpanMetricKey,
    TokenUsageKey,
    TraceMetricKey,
)
from riffdb_application import (
    AttemptBudget,
    BearerCredential,
    CallMetadata,
    DatabaseAlias,
    SyncApplicationTransport,
)

from riffdb_mlflow.generated.client import RiffDBMLflowClient
from riffdb_mlflow.store import RiffDBTrackingStore


def _make_live_span(
    *,
    trace_id: str,
    span_id_hex: str,
    parent_span_id_hex: str | None,
    name: str,
    span_type: str,
    start_time_ns: int,
    end_time_ns: int,
    model_name: str = "gpt-4o",
    input_tokens: int = 120,
    output_tokens: int = 60,
    cost_usd: float = 0.0015,
) -> Span:
    raw_attrs: dict[str, Any] = {
        SpanAttributeKey.REQUEST_ID: trace_id,
        SpanAttributeKey.SPAN_TYPE: span_type,
        SpanAttributeKey.MODEL: model_name,
        SpanAttributeKey.MODEL_PROVIDER: "openai",
        SpanAttributeKey.CHAT_USAGE: {
            TokenUsageKey.INPUT_TOKENS: input_tokens,
            TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
        },
        SpanAttributeKey.LLM_COST: {
            "input_cost": cost_usd * 0.6,
            "output_cost": cost_usd * 0.4,
            "total_cost": cost_usd,
        },
        SpanAttributeKey.INPUTS: {"prompt": f"benchmark-{name}"},
        SpanAttributeKey.OUTPUTS: {"response": f"ok-{name}"},
    }
    attrs = {k: json.dumps(v) for k, v in raw_attrs.items()}
    return Span.from_dict(
        {
            "trace_id": trace_id,
            "span_id": span_id_hex,
            "parent_span_id": parent_span_id_hex,
            "name": name,
            "start_time_unix_nano": start_time_ns,
            "end_time_unix_nano": end_time_ns,
            "status": {"code": "STATUS_CODE_OK", "message": ""},
            "attributes": attrs,
            "events": [],
        }
    )


def _run_store_workload(
    store: Any,
    store_label: str,
    experiment_id: str,
    num_traces: int,
    spans_per_trace: int,
    *,
    test_spans_before_trace: bool,
) -> dict[str, Any]:
    exp_int = int(experiment_id) if experiment_id.isdigit() else 1
    base_ns = 1_700_000_000_000_000_000
    trace_ids: list[str] = []

    # Prepare out-of-order span batches: child spans first, root span last
    t0 = time.perf_counter()
    for i in range(num_traces):
        tid = f"tr-{exp_int:016x}{i + 1:016x}"
        trace_ids.append(tid)
        t_start_ns = base_ns + i * 1_000_000_000
        root_hex = f"{(i + 1) * 1000 + 1:016x}"

        # Create child spans (out of order: deepest child first, then siblings, then root)
        ooo_spans: list[Span] = []
        for s_idx in range(spans_per_trace - 1, 0, -1):
            span_hex = f"{(i + 1) * 1000 + 1 + s_idx:016x}"
            parent_hex = f"{(i + 1) * 1000 + s_idx:016x}" if s_idx > 1 else root_hex
            s_start = t_start_ns + s_idx * 10_000_000
            s_end = s_start + (25 + (s_idx * 7) % 90) * 1_000_000
            stype = SpanType.LLM if s_idx % 2 == 1 else SpanType.TOOL
            model = "gpt-4o" if s_idx % 3 != 0 else "claude-3-5-sonnet"
            ooo_spans.append(
                _make_live_span(
                    trace_id=tid,
                    span_id_hex=span_hex,
                    parent_span_id_hex=parent_hex,
                    name=f"step_{s_idx}",
                    span_type=stype,
                    start_time_ns=s_start,
                    end_time_ns=s_end,
                    model_name=model,
                    input_tokens=100 + s_idx * 10,
                    output_tokens=40 + s_idx * 5,
                    cost_usd=0.001 * (s_idx + 1),
                )
            )
        root_span = _make_live_span(
            trace_id=tid,
            span_id_hex=root_hex,
            parent_span_id_hex=None,
            name="agent_root",
            span_type=SpanType.AGENT,
            start_time_ns=t_start_ns,
            end_time_ns=t_start_ns + spans_per_trace * 40_000_000,
            model_name="gpt-4o",
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.0005,
        )

        if not test_spans_before_trace:
            store.start_trace(
                TraceInfo(
                    trace_id=tid,
                    trace_location=TraceLocation.from_experiment_id(experiment_id),
                    request_time=t_start_ns // 1_000_000,
                    execution_duration=spans_per_trace * 40,
                    state=TraceState.OK,
                    trace_metadata={"mlflow.trace.session": f"sess-{i % 4}"},
                    tags={"mlflow.traceName": "agent_root"},
                )
            )
            store.log_spans(experiment_id, [*ooo_spans, root_span])
        else:
            # Child spans logged BEFORE root span AND BEFORE start_trace()
            store.log_spans(experiment_id, ooo_spans)
            store.log_spans(experiment_id, [root_span])
            store.start_trace(
                TraceInfo(
                    trace_id=tid,
                    trace_location=TraceLocation.from_experiment_id(experiment_id),
                    request_time=t_start_ns // 1_000_000,
                    execution_duration=spans_per_trace * 40,
                    state=TraceState.OK,
                    trace_metadata={"mlflow.trace.session": f"sess-{i % 4}"},
                    tags={"mlflow.traceName": "agent_root"},
                )
            )

        store.create_assessment(
            Feedback(
                name="relevance",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE,
                    source_id="judge-1",
                ),
                trace_id=tid,
                value=0.85 if i % 2 == 0 else 0.95,
            )
        )
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    fetched = store.batch_get_traces(trace_ids)
    batch_get_ms = (time.perf_counter() - t1) * 1000.0
    assert len(fetched) == num_traces
    assert all(len(t.data.spans) == spans_per_trace for t in fetched)

    t_cold = time.perf_counter()
    _ = store.query_trace_metrics(
        experiment_ids=[experiment_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.LATENCY,
        aggregations=[MetricAggregation(AggregationType.AVG)],
        dimensions=["trace_name"],
    )
    _ = store.query_trace_metrics(
        experiment_ids=[experiment_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[MetricAggregation(AggregationType.AVG)],
        dimensions=["span_name", "span_status"],
    )
    cold_metrics_query_ms = (time.perf_counter() - t_cold) * 1000.0

    t2 = time.perf_counter()
    trace_metrics = store.query_trace_metrics(
        experiment_ids=[experiment_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.LATENCY,
        aggregations=[
            MetricAggregation(AggregationType.AVG),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=95.0),
        ],
        dimensions=["trace_name"],
    )
    span_metrics = store.query_trace_metrics(
        experiment_ids=[experiment_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.LATENCY,
        aggregations=[
            MetricAggregation(AggregationType.AVG),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=95.0),
        ],
        dimensions=["span_name", "span_status"],
    )
    metrics_query_ms = (time.perf_counter() - t2) * 1000.0

    result: dict[str, Any] = {
        "store": store_label,
        "traces": num_traces,
        "spans_per_trace": spans_per_trace,
        "total_spans": num_traces * spans_per_trace,
        "spans_before_start_trace": test_spans_before_trace,
        "ingest_total_ms": round(ingest_ms, 2),
        "batch_get_traces_ms": round(batch_get_ms, 2),
        "cold_demand_activation_metrics_ms": round(cold_metrics_query_ms, 2),
        "query_trace_metrics_ms": round(metrics_query_ms, 2),
        "trace_metric_points": len(trace_metrics),
        "span_metric_points": len(span_metrics),
    }

    if isinstance(store, RiffDBTrackingStore):
        t3 = time.perf_counter()
        ev_rollups = store.get_span_model_rollups(
            experiment_id, span_type="LLM", model_name="gpt-4o", status="OK"
        )
        col_rollups = store.get_columnar_span_model_rollups(experiment_id)
        rollup_ms = (time.perf_counter() - t3) * 1000.0
        result["adr_0315_projected_rollups_ms"] = round(rollup_ms, 2)
        result["event_rollup_groups"] = len(ev_rollups)
        result["columnar_rollup_groups"] = len(col_rollups)

    return result


def _run_benchmark_in_configured_env(num_traces: int, spans_per_trace: int) -> dict[str, Any]:
    endpoint = os.environ["RIFFDB_ENDPOINT"]
    cred_file = os.environ["RIFFDB_CREDENTIAL_FILE"]
    database = os.environ["RIFFDB_DATABASE"]
    if os.environ.get("RIFFDB_TENANT") == "":
        os.environ.pop("RIFFDB_TENANT", None)
    credential = BearerCredential.from_protected_file(cred_file)
    metadata = CallMetadata.authenticated(credential).with_database(DatabaseAlias(database))

    with (
        tempfile.TemporaryDirectory(prefix="riffdb-mlflow-bench-") as tmp_dir,
        SyncApplicationTransport.connect_uri(endpoint, metadata) as transport,
    ):
        riff_store = RiffDBTrackingStore(
            f"riffdb://{database}",
            f"{tmp_dir}/riff-artifacts",
        )
        riff_store._transport = transport
        riff_store._client = RiffDBMLflowClient(transport, AttemptBudget(3))
        riff_exp_id = riff_store.create_experiment("bench-riffdb")
        riff_result = _run_store_workload(
            riff_store,
            "RiffDBTrackingStore",
            riff_exp_id,
            num_traces=num_traces,
            spans_per_trace=spans_per_trace,
            test_spans_before_trace=True,
        )

        sqlite_uri = f"sqlite:///{tmp_dir}/mlflow.db"
        sql_store_cls: Any = SqlAlchemyStore
        sql_store: Any = sql_store_cls(sqlite_uri, f"{tmp_dir}/sql-artifacts")
        sql_exp_id = str(sql_store.create_experiment("bench-sqlalchemy"))

        sql_result = _run_store_workload(
            sql_store,
            "SqlAlchemyStore (SQLite)",
            sql_exp_id,
            num_traces=num_traces,
            spans_per_trace=spans_per_trace,
            test_spans_before_trace=True,
        )

        return {
            "benchmark": "riffdb-mlflow-out-of-order-spans/v1",
            "riffdb": riff_result,
            "sqlalchemy_sqlite": sql_result,
        }


def _find_riffdb_binary() -> str:
    env_bin = os.environ.get("RIFFDB")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    sibling_bin = Path(__file__).resolve().parents[2] / "riffdb" / "target" / "release" / "riffdb"
    if sibling_bin.is_file():
        return str(sibling_bin)
    found = shutil.which("riffdb")
    if found:
        return found
    raise RuntimeError("Could not locate riffdb binary; set RIFFDB=/path/to/riffdb")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark out-of-order LLM span ingestion.")
    parser.add_argument("--traces", type=int, default=20, help="Number of traces")
    parser.add_argument("--spans-per-trace", type=int, default=5, help="Spans per trace")
    parser.add_argument(
        "--inner",
        action="store_true",
        help="Run inside an active `riffdb dev --exec` environment",
    )
    args = parser.parse_args()

    if args.inner or "RIFFDB_ENDPOINT" in os.environ:
        summary = _run_benchmark_in_configured_env(args.traces, args.spans_per_trace)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    riffdb_bin = _find_riffdb_binary()
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    cmd = (
        f"{sys.executable} {script_path} --inner "
        f"--traces {args.traces} --spans-per-trace {args.spans_per_trace}"
    )
    proc = subprocess.run(
        [riffdb_bin, "dev", "--reset", "--exec", cmd],
        cwd=repo_root,
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
