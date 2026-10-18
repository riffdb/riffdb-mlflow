from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlflow.entities import (
    Dataset,
    DatasetInput,
    InputTag,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelParameter,
    LoggedModelStatus,
    LoggedModelTag,
    Metric,
)
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import SpanType
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import (
    AssessmentMetricKey,
    SpanAttributeKey,
    SpanMetricKey,
    TokenUsageKey,
    TraceMetricKey,
)
from riffdb_application import Timestamp

from riffdb_mlflow.generated.client import (
    AggregateSpanAnalyticsByModelFound,
    BatchGetTraceAssessmentsFound,
    BatchGetTracesFound,
    BatchGetTraceSpansFound,
    CreateExperimentExperimentCreated,
    CreateRunWithTagsRunCreated,
    DeleteLoggedModelEntityLoggedModelDeleted,
    DeleteTraceAssessmentTraceAssessmentDeleted,
    DeleteTraceSpanTraceSpanDeleted,
    DeleteTraceTraceDeleted,
    Experiment,
    GetExperimentFound,
    GetExperimentFoundExperiment,
    GetLoggedModelFound,
    GetLoggedModelMissing,
    GetRunFound,
    GetRunFoundRun,
    GetRunLineageFound,
    GetRunLineageMissing,
    GetTraceFound,
    GetTraceMissing,
    IngestTraceSpansTraceSpansIngested,
    LifecycleStage,
    LogBatchBatchLogged,
    QueryAssessmentAnalyticsFound,
    QuerySpanAnalyticsFound,
    QuerySpanModelRollupFound,
    QueryTraceAnalyticsFound,
    Run,
    RunDataKind,
    RunDatum,
    SearchLoggedModelsFound,
    SearchTracesFound,
    UpsertLoggedModelLoggedModelUpserted,
    UpsertRunLineageRunLineageUpserted,
    UpsertTraceAssessmentTraceAssessmentUpserted,
    UpsertTraceTraceUpserted,
)
from riffdb_mlflow.generated.client import (
    RunStatus as RiffDBRunStatus,
)
from riffdb_mlflow.store import RiffDBTrackingStore


@dataclass
class _FakeResult:
    outcome: Any = None
    value: Any = None
    commit_sequence: int = 1
    application_head: int = 1
    next_cursor: str | None = None


class FakeExtendedRiffDBClient:
    """In-memory fake for RiffDB generated client covering runs, traces, spans, assessments,
    projections, and logged models."""

    def __init__(self, experiment_id: int = 1) -> None:
        self.commit_seq = 1
        self.experiments: dict[int, Experiment] = {
            experiment_id: Experiment(
                scope="global",
                experiment_id=experiment_id,
                name="default",
                name_digest=RiffDBTrackingStore._digest("default"),
                artifact_location=f"s3://artifacts/{experiment_id}",
                lifecycle_stage=LifecycleStage.ACTIVE,
                creation_time=Timestamp(seconds=1_700_000_000, nanos=0),
                last_update_time=Timestamp(seconds=1_700_000_000, nanos=0),
                packed_tags=b"{}",
                revision=0,
            )
        }
        self.runs: dict[tuple[int, str], Run] = {}
        self.run_data: dict[tuple[int, str, RunDataKind, str, str], RunDatum] = {}
        self.run_lineages: dict[tuple[int, str], SimpleNamespace] = {}
        self.traces: dict[tuple[int, str], SimpleNamespace] = {}
        self.spans: dict[tuple[int, str, str], SimpleNamespace] = {}
        self.assessments: dict[tuple[int, str, str], SimpleNamespace] = {}
        self.models: dict[tuple[int, str], SimpleNamespace] = {}
        self.span_events: list[dict[str, Any]] = []

    def _next_commit(self) -> int:
        self.commit_seq += 1
        return self.commit_seq

    def get_experiment(self, params: Any) -> _FakeResult:
        exp = self.experiments.get(int(params.experiment_id))
        if exp is None:
            return _FakeResult(value=SimpleNamespace())
        return _FakeResult(
            value=GetExperimentFound(
                experiment=GetExperimentFoundExperiment(
                    experiment_id=exp.experiment_id,
                    name=exp.name,
                    artifact_location=exp.artifact_location,
                    lifecycle_stage=exp.lifecycle_stage,
                    creation_time=exp.creation_time,
                    last_update_time=exp.last_update_time,
                    packed_tags=exp.packed_tags,
                    revision=exp.revision,
                )
            ),
            application_head=self.commit_seq,
        )

    def create_experiment(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        exp = Experiment(
            scope=cmd.scope,
            experiment_id=cmd.experiment_id,
            name=cmd.name,
            name_digest=cmd.name_digest,
            artifact_location=cmd.artifact_location,
            lifecycle_stage=LifecycleStage.ACTIVE,
            creation_time=cmd.creation_time,
            last_update_time=cmd.creation_time,
            packed_tags=cmd.packed_tags,
            revision=0,
        )
        self.experiments[int(cmd.experiment_id)] = exp
        return _FakeResult(
            outcome=CreateExperimentExperimentCreated(experiment=exp),
            commit_sequence=seq,
        )

    def create_run_with_tags(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        run = Run(
            experiment_id=cmd.experiment_id,
            run_id=cmd.run_id,
            user_id=cmd.user_id,
            status=RiffDBRunStatus.RUNNING,
            start_time=cmd.start_time,
            end_time=None,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_uri=cmd.artifact_uri,
            run_name=cmd.run_name,
            packed_tags=cmd.packed_tags,
            packed_params=cmd.packed_params,
            packed_latest_metrics=cmd.packed_latest_metrics,
            revision=0,
            initialized=True,
        )
        self.runs[(int(cmd.experiment_id), cmd.run_id)] = run
        return _FakeResult(
            outcome=CreateRunWithTagsRunCreated(run=run),
            commit_sequence=seq,
        )

    def get_run(self, params: Any) -> _FakeResult:
        run = self.runs.get((int(params.experiment_id), params.run_id))
        if run is None:
            return _FakeResult(value=SimpleNamespace())
        return _FakeResult(
            value=GetRunFound(
                run=GetRunFoundRun(
                    experiment_id=run.experiment_id,
                    run_id=run.run_id,
                    user_id=run.user_id,
                    status=run.status,
                    start_time=run.start_time,
                    end_time=run.end_time,
                    lifecycle_stage=run.lifecycle_stage,
                    artifact_uri=run.artifact_uri,
                    run_name=run.run_name,
                    packed_tags=run.packed_tags,
                    packed_params=run.packed_params,
                    packed_latest_metrics=run.packed_latest_metrics,
                    revision=run.revision,
                    initialized=run.initialized,
                )
            ),
            application_head=self.commit_seq,
        )

    def log_batch(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        run = self.runs[(int(cmd.experiment_id), cmd.run_id)]
        updated = replace(
            run,
            packed_tags=cmd.packed_tags,
            packed_params=cmd.packed_params,
            packed_latest_metrics=cmd.packed_latest_metrics,
            revision=run.revision + 1,
        )
        self.runs[(int(cmd.experiment_id), cmd.run_id)] = updated
        return _FakeResult(outcome=LogBatchBatchLogged(run=updated), commit_sequence=seq)

    def get_run_lineage(self, params: Any) -> _FakeResult:
        lin = self.run_lineages.get((int(params.experiment_id), params.run_id))
        if lin is None:
            return _FakeResult(value=GetRunLineageMissing(), application_head=self.commit_seq)
        return _FakeResult(
            value=GetRunLineageFound(lineage=lin),  # type: ignore[arg-type]
            application_head=self.commit_seq,
        )

    def upsert_run_lineage(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        lin = SimpleNamespace(
            experiment_id=cmd.experiment_id,
            run_id=cmd.run_id,
            packed_datasets=cmd.packed_datasets,
            packed_model_inputs=cmd.packed_model_inputs,
            packed_model_outputs=cmd.packed_model_outputs,
        )
        self.run_lineages[(int(cmd.experiment_id), cmd.run_id)] = lin
        return _FakeResult(
            outcome=UpsertRunLineageRunLineageUpserted(lineage=lin),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def get_trace(self, params: Any) -> _FakeResult:
        exp_id = int(params.experiment_id)
        t = self.traces.get((exp_id, params.trace_id))
        if t is None:
            return _FakeResult(value=GetTraceMissing(), application_head=self.commit_seq)
        spans = [
            s for (e, tid, _), s in self.spans.items() if e == exp_id and tid == params.trace_id
        ]
        spans.sort(key=lambda r: (r.start_time_ns, r.span_id))
        assessments = [
            a
            for (e, tid, _), a in self.assessments.items()
            if e == exp_id and tid == params.trace_id
        ]
        assessments.sort(key=lambda r: (r.created_timestamp_ms, r.assessment_id))
        return _FakeResult(
            value=GetTraceFound(
                trace=t,  # type: ignore[arg-type]
                spans=tuple(spans),  # type: ignore[arg-type]
                assessments=tuple(assessments),  # type: ignore[arg-type]
            ),
            application_head=self.commit_seq,
        )

    def batch_get_traces(self, params: Any) -> _FakeResult:
        exp_id = int(params.experiment_id)
        want = set(params.trace_ids)
        rows = [t for (e, tid), t in self.traces.items() if e == exp_id and tid in want]
        rows.sort(key=lambda r: r.trace_id)
        a_rows = [a for (e, tid, _), a in self.assessments.items() if e == exp_id and tid in want]
        a_rows.sort(key=lambda r: (r.trace_id, r.created_timestamp_ms, r.assessment_id))
        return _FakeResult(
            value=BatchGetTracesFound(
                traces=tuple(rows),  # type: ignore[arg-type]
                assessments=tuple(a_rows[:51]),  # type: ignore[arg-type]
            ),
            application_head=self.commit_seq,
        )

    def batch_get_trace_spans(self, params: Any) -> _FakeResult:
        exp_id = int(params.experiment_id)
        want = set(params.trace_ids)
        rows = [s for (e, tid, _), s in self.spans.items() if e == exp_id and tid in want]
        rows.sort(key=lambda r: (r.trace_id, r.start_time_ns, r.span_id))
        return _FakeResult(
            value=BatchGetTraceSpansFound(spans=tuple(rows)),  # type: ignore[arg-type]
            application_head=self.commit_seq,
        )

    def batch_get_trace_assessments(self, params: Any) -> _FakeResult:
        exp_id = int(params.experiment_id)
        want = set(params.trace_ids)
        rows = [a for (e, tid, _), a in self.assessments.items() if e == exp_id and tid in want]
        rows.sort(key=lambda r: (r.trace_id, r.created_timestamp_ms, r.assessment_id))
        return _FakeResult(
            value=BatchGetTraceAssessmentsFound(assessments=tuple(rows)),  # type: ignore[arg-type]
            application_head=self.commit_seq,
        )

    def upsert_trace(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        t = SimpleNamespace(
            experiment_id=cmd.experiment_id,
            trace_id=cmd.trace_id,
            request_time_ms=cmd.request_time_ms,
            execution_duration_ms=cmd.execution_duration_ms,
            state=cmd.state,
            status_code=cmd.status_code,
            trace_name=cmd.trace_name,
            client_request_id=cmd.client_request_id,
            request_preview=cmd.request_preview,
            response_preview=cmd.response_preview,
            source_run_id=cmd.source_run_id,
            span_count=cmd.span_count,
            error_span_count=cmd.error_span_count,
            total_input_tokens=cmd.total_input_tokens,
            total_output_tokens=cmd.total_output_tokens,
            total_tokens=cmd.total_tokens,
            cache_read_input_tokens=cmd.cache_read_input_tokens,
            cache_creation_input_tokens=cmd.cache_creation_input_tokens,
            total_cost_micros=cmd.total_cost_micros,
            session_id=cmd.session_id,
            packed_trace_metadata=cmd.packed_trace_metadata,
            packed_tags=cmd.packed_tags,
            packed_run_ids=cmd.packed_run_ids,
            packed_prompt_versions=cmd.packed_prompt_versions,
            requested_at=cmd.requested_at,
            initialized=cmd.initialized,
        )
        self.traces[(int(cmd.experiment_id), cmd.trace_id)] = t
        return _FakeResult(
            outcome=UpsertTraceTraceUpserted(trace=t),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def ingest_trace_spans(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        t = SimpleNamespace(
            experiment_id=cmd.experiment_id,
            trace_id=cmd.trace_id,
            request_time_ms=cmd.request_time_ms,
            execution_duration_ms=cmd.execution_duration_ms,
            state=cmd.state,
            status_code=cmd.status_code,
            trace_name=cmd.trace_name,
            client_request_id=cmd.client_request_id,
            request_preview=cmd.request_preview,
            response_preview=cmd.response_preview,
            source_run_id=cmd.source_run_id,
            span_count=cmd.span_count,
            error_span_count=cmd.error_span_count,
            total_input_tokens=cmd.total_input_tokens,
            total_output_tokens=cmd.total_output_tokens,
            total_tokens=cmd.total_tokens,
            cache_read_input_tokens=cmd.cache_read_input_tokens,
            cache_creation_input_tokens=cmd.cache_creation_input_tokens,
            total_cost_micros=cmd.total_cost_micros,
            session_id=cmd.session_id,
            packed_trace_metadata=cmd.packed_trace_metadata,
            packed_tags=cmd.packed_tags,
            packed_run_ids=cmd.packed_run_ids,
            packed_prompt_versions=cmd.packed_prompt_versions,
            requested_at=cmd.requested_at,
            initialized=cmd.initialized,
        )
        self.traces[(int(cmd.experiment_id), cmd.trace_id)] = t
        for s in cmd.spans:
            span_key = (int(cmd.experiment_id), cmd.trace_id, s.span_id)
            already_existed = span_key in self.spans
            self.spans[span_key] = s
            if not already_existed:
                self.span_events.append(
                    {
                        "experiment_id": int(cmd.experiment_id),
                        "span_type": s.span_type,
                        "model_name": s.model_name,
                        "status": s.status,
                        "observed_at": s.started_at,
                        "duration_ms": int(s.duration_ms),
                        "input_tokens": int(s.input_tokens),
                        "output_tokens": int(s.output_tokens),
                        "total_tokens": int(s.total_tokens),
                        "total_cost_micros": int(s.total_cost_micros),
                    }
                )
        return _FakeResult(
            outcome=IngestTraceSpansTraceSpansIngested(trace=t),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def search_traces(self, params: Any) -> _FakeResult:
        rows = [t for (exp_id, _), t in self.traces.items() if exp_id == int(params.experiment_id)]
        rows.sort(key=lambda r: (-r.request_time_ms, r.trace_id))
        return _FakeResult(
            value=SearchTracesFound(traces=tuple(rows)),  # type: ignore[arg-type]
            application_head=self.commit_seq,
        )

    def delete_trace(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        removed = self.traces.pop((int(cmd.experiment_id), cmd.trace_id), SimpleNamespace())
        return _FakeResult(
            outcome=DeleteTraceTraceDeleted(trace=removed),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def delete_trace_span(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        removed = self.spans.pop(
            (int(cmd.experiment_id), cmd.trace_id, cmd.span_id), SimpleNamespace()
        )
        return _FakeResult(
            outcome=DeleteTraceSpanTraceSpanDeleted(span=removed),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def upsert_trace_assessment(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        a = SimpleNamespace(
            experiment_id=cmd.experiment_id,
            trace_id=cmd.trace_id,
            assessment_id=cmd.assessment_id,
            name=cmd.name,
            assessment_type=cmd.assessment_type,
            value_text=cmd.value_text,
            numeric_value_milli=cmd.numeric_value_milli,
            created_timestamp_ms=cmd.created_timestamp_ms,
            last_updated_timestamp_ms=cmd.last_updated_timestamp_ms,
            span_id=cmd.span_id,
            packed_assessment_json=cmd.packed_assessment_json,
        )
        self.assessments[(int(cmd.experiment_id), cmd.trace_id, cmd.assessment_id)] = a
        return _FakeResult(
            outcome=UpsertTraceAssessmentTraceAssessmentUpserted(assessment=a),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def delete_trace_assessment(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        removed = self.assessments.pop(
            (int(cmd.experiment_id), cmd.trace_id, cmd.assessment_id), SimpleNamespace()
        )
        return _FakeResult(
            outcome=DeleteTraceAssessmentTraceAssessmentDeleted(assessment=removed),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def query_trace_analytics_after_commit(self, params: Any, _commit: int) -> _FakeResult:
        rows = [t for (exp_id, _), t in self.traces.items() if exp_id == int(params.experiment_id)]
        return _FakeResult(
            value=QueryTraceAnalyticsFound(traces=tuple(rows)),  # type: ignore[arg-type]
        )

    def query_span_analytics_after_commit(self, params: Any, _commit: int) -> _FakeResult:
        rows = [
            s for (exp_id, _, _), s in self.spans.items() if exp_id == int(params.experiment_id)
        ]
        return _FakeResult(
            value=QuerySpanAnalyticsFound(spans=tuple(rows)),  # type: ignore[arg-type]
        )

    def query_assessment_analytics_after_commit(self, params: Any, _commit: int) -> _FakeResult:
        rows = [
            a
            for (exp_id, _, _), a in self.assessments.items()
            if exp_id == int(params.experiment_id)
        ]
        return _FakeResult(
            value=QueryAssessmentAnalyticsFound(assessments=tuple(rows)),  # type: ignore[arg-type]
        )

    def query_span_model_rollup_after_commit(self, params: Any, _commit: int) -> _FakeResult:
        groups: dict[tuple[int, str, str, str, int], list[dict[str, Any]]] = {}
        for ev in self.span_events:
            if (
                ev["experiment_id"] == int(params.experiment_id)
                and ev["span_type"] == params.span_type
                and ev["model_name"] == params.model_name
                and ev["status"] == params.status
            ):
                bucket_sec = (ev["observed_at"].seconds // 3600) * 3600
                key = (
                    ev["experiment_id"],
                    ev["span_type"],
                    ev["model_name"],
                    ev["status"],
                    bucket_sec,
                )
                groups.setdefault(key, []).append(ev)
        rollups = []
        for (exp_id, stype, mname, status, bucket_sec), items in groups.items():
            durs = [x["duration_ms"] for x in items]
            rollups.append(
                SimpleNamespace(
                    experiment_id=exp_id,
                    span_type=stype,
                    model_name=mname,
                    status=status,
                    observed_at=Timestamp(seconds=bucket_sec, nanos=0),
                    span_count=len(items),
                    total_duration_ms=sum(durs),
                    min_duration_ms=min(durs),
                    max_duration_ms=max(durs),
                    avg_duration_ms=sum(durs) // len(durs),
                    sum_input_tokens=sum(x["input_tokens"] for x in items),
                    sum_output_tokens=sum(x["output_tokens"] for x in items),
                    sum_total_tokens=sum(x["total_tokens"] for x in items),
                    sum_cost_micros=sum(x["total_cost_micros"] for x in items),
                )
            )
        return _FakeResult(
            value=QuerySpanModelRollupFound(rollups=tuple(rollups)),  # type: ignore[arg-type]
        )

    def aggregate_span_analytics_by_model_after_commit(
        self, params: Any, _commit: int
    ) -> _FakeResult:
        groups: dict[tuple[str, str, str, int], list[SimpleNamespace]] = {}
        for (exp_id, _, _), s in self.spans.items():
            if exp_id == int(params.experiment_id):
                bucket_sec = (s.started_at.seconds // 3600) * 3600
                key = (s.model_name, s.span_type, s.status, bucket_sec)
                groups.setdefault(key, []).append(s)
        by_model = []
        for (mname, stype, status, bucket_sec), items in groups.items():
            durs = sorted(int(s.duration_ms) for s in items)
            by_model.append(
                SimpleNamespace(
                    model_name=mname,
                    span_type=stype,
                    status=status,
                    hour_bucket=Timestamp(seconds=bucket_sec, nanos=0),
                    span_count=len(items),
                    sum_duration_ms=Decimal(sum(durs)),
                    min_duration_ms=min(durs),
                    max_duration_ms=max(durs),
                    p50_duration_ms=durs[len(durs) // 2],
                    p95_duration_ms=durs[min(len(durs) - 1, int(len(durs) * 0.95))],
                    p99_duration_ms=durs[min(len(durs) - 1, int(len(durs) * 0.99))],
                    sum_total_tokens=Decimal(sum(int(s.total_tokens) for s in items)),
                    sum_total_cost_micros=Decimal(sum(int(s.total_cost_micros) for s in items)),
                )
            )
        return _FakeResult(
            value=AggregateSpanAnalyticsByModelFound(by_model=tuple(by_model)),  # type: ignore[arg-type]
        )

    def get_logged_model(self, params: Any) -> _FakeResult:
        m = self.models.get((int(params.experiment_id), params.model_id))
        if m is None:
            return _FakeResult(value=GetLoggedModelMissing())
        return _FakeResult(
            value=GetLoggedModelFound(model=m),  # type: ignore[arg-type]
        )

    def search_logged_models(self, params: Any) -> _FakeResult:
        rows = [m for (exp_id, _), m in self.models.items() if exp_id == int(params.experiment_id)]
        return _FakeResult(
            value=SearchLoggedModelsFound(models=tuple(rows)),  # type: ignore[arg-type]
        )

    def upsert_logged_model(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        m = SimpleNamespace(
            experiment_id=cmd.experiment_id,
            model_id=cmd.model_id,
            name=cmd.name,
            artifact_location=cmd.artifact_location,
            creation_timestamp_ms=cmd.creation_timestamp_ms,
            last_updated_timestamp_ms=cmd.last_updated_timestamp_ms,
            model_type=cmd.model_type,
            source_run_id=cmd.source_run_id,
            status=cmd.status,
            status_message=cmd.status_message,
            lifecycle_stage=cmd.lifecycle_stage,
            packed_tags=cmd.packed_tags,
            packed_params=cmd.packed_params,
            packed_metrics=cmd.packed_metrics,
        )
        self.models[(int(cmd.experiment_id), cmd.model_id)] = m
        return _FakeResult(
            outcome=UpsertLoggedModelLoggedModelUpserted(model=m),  # type: ignore[arg-type]
            commit_sequence=seq,
        )

    def delete_logged_model_entity(self, cmd: Any) -> _FakeResult:
        seq = self._next_commit()
        removed = self.models.pop((int(cmd.experiment_id), cmd.model_id), SimpleNamespace())
        return _FakeResult(
            outcome=DeleteLoggedModelEntityLoggedModelDeleted(model=removed),  # type: ignore[arg-type]
            commit_sequence=seq,
        )


def _make_store(experiment_id: int = 1) -> tuple[RiffDBTrackingStore, FakeExtendedRiffDBClient]:
    store = RiffDBTrackingStore("riffdb://mlflow-test", "s3://artifacts")
    fake = FakeExtendedRiffDBClient(experiment_id=experiment_id)
    store._application = lambda: fake  # type: ignore[method-assign]
    return store, fake


def _build_mock_otel_span(
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    span_type: str,
    start_time_ns: int,
    end_time_ns: int,
    status_code: SpanStatusCode = SpanStatusCode.OK,
    status_description: str = "",
    attributes: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> Any:
    from mlflow.entities.span import Span

    attrs: dict[str, Any] = {
        SpanAttributeKey.REQUEST_ID: trace_id,
        SpanAttributeKey.SPAN_TYPE: span_type,
    }
    for k, v in (attributes or {}).items():
        attrs[k] = v
    return Span.from_dict(
        {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": name,
            "start_time_unix_nano": start_time_ns,
            "end_time_unix_nano": end_time_ns,
            "status": {
                "code": f"STATUS_CODE_{status_code.value}",
                "message": status_description,
            },
            "attributes": attrs,
            "events": [
                {
                    "name": ev["name"],
                    "time_unix_nano": ev["timestamp"],
                    "attributes": dict(ev.get("attributes") or {}),
                }
                for ev in (events or [])
            ],
        }
    )


def test_out_of_order_span_ingestion_before_start_trace_and_parent_reconstruction() -> None:
    store, _ = _make_store(experiment_id=1)
    trace_id = "tr-0000000000000001deadbeefcafe0001"

    # 1. Log child LLM span BEFORE parent span and BEFORE start_trace()
    child_llm = _build_mock_otel_span(
        trace_id=trace_id,
        span_id="00000000000000b2",
        parent_span_id="00000000000000a1",
        name="openai.chat.completions",
        span_type=SpanType.LLM,
        start_time_ns=1_700_000_000_100_000_000,
        end_time_ns=1_700_000_000_450_000_000,
        attributes={
            SpanAttributeKey.MODEL: "gpt-4o",
            SpanAttributeKey.MODEL_PROVIDER: "openai",
            SpanAttributeKey.CHAT_USAGE: {
                TokenUsageKey.INPUT_TOKENS: 150,
                TokenUsageKey.OUTPUT_TOKENS: 50,
                TokenUsageKey.TOTAL_TOKENS: 200,
            },
            SpanAttributeKey.LLM_COST: {
                "input_cost": 0.00075,
                "output_cost": 0.00050,
                "total_cost": 0.00125,
            },
            SpanAttributeKey.INPUTS: {"messages": [{"role": "user", "content": "hi"}]},
            SpanAttributeKey.OUTPUTS: {"choices": [{"message": {"content": "hello"}}]},
        },
        events=[{"name": "first_token", "timestamp": 1_700_000_000_200_000_000}],
    )
    logged = store.ingest_spans_out_of_order("1", [child_llm])
    assert len(logged) == 1

    # Materialized provisional TraceInfo already reflects token and cost aggregations
    provisional_info = store.get_trace_info(trace_id)
    assert provisional_info.trace_id == trace_id
    assert json.loads(provisional_info.trace_metadata["mlflow.trace.tokenUsage"]) == {
        TokenUsageKey.INPUT_TOKENS: 150,
        TokenUsageKey.OUTPUT_TOKENS: 50,
        TokenUsageKey.TOTAL_TOKENS: 200,
    }

    # 2. Log tool sibling span via log_spans_async and root span out of order
    tool_span = _build_mock_otel_span(
        trace_id=trace_id,
        span_id="00000000000000c3",
        parent_span_id="00000000000000a1",
        name="vector_lookup",
        span_type=SpanType.TOOL,
        start_time_ns=1_700_000_000_020_000_000,
        end_time_ns=1_700_000_000_090_000_000,
    )
    asyncio.run(store.log_spans_async("1", [tool_span]))

    root_span = _build_mock_otel_span(
        trace_id=trace_id,
        span_id="00000000000000a1",
        parent_span_id=None,
        name="agent_pipeline",
        span_type=SpanType.AGENT,
        start_time_ns=1_700_000_000_000_000_000,
        end_time_ns=1_700_000_000_500_000_000,
    )
    store.log_spans("1", [root_span])

    # 3. Late start_trace arrives and merges metadata/tags without losing span aggregates
    started = store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id("1"),
            request_time=1_700_000_000_000,
            execution_duration=500,
            state=TraceState.OK,
            trace_metadata={"mlflow.trace.session": "sess-xyz", "custom_meta": "val"},
            tags={"mlflow.traceName": "agent_pipeline", "env": "staging"},
        )
    )
    assert started.trace_metadata["mlflow.trace.session"] == "sess-xyz"
    assert started.tags["env"] == "staging"

    # 4. batch_get_traces reconstructs full Trace with ordered spans, inputs, outputs, and events
    full_traces = store.batch_get_traces([trace_id])
    assert len(full_traces) == 1
    full_trace = full_traces[0]
    assert [s.span_id for s in full_trace.data.spans] == [
        "00000000000000a1",
        "00000000000000c3",
        "00000000000000b2",
    ]
    llm_reconstructed = full_trace.data.spans[2]
    assert llm_reconstructed.parent_id == "00000000000000a1"
    assert llm_reconstructed.span_type == SpanType.LLM
    assert llm_reconstructed.inputs == {"messages": [{"role": "user", "content": "hi"}]}
    assert llm_reconstructed.outputs == {"choices": [{"message": {"content": "hello"}}]}
    assert len(llm_reconstructed.events) == 1
    assert llm_reconstructed.events[0].name == "first_token"


def test_trace_tags_links_search_and_delete() -> None:
    store, _ = _make_store(experiment_id=1)
    run = store.create_run("1", "alice", 1_700_000_000_000, [], "eval-run")

    t1 = store.start_trace(
        TraceInfo(
            trace_id="tr-00000000000000010000000000000101",
            trace_location=TraceLocation.from_experiment_id("1"),
            request_time=1_700_000_001_000,
            execution_duration=200,
            state=TraceState.OK,
            trace_metadata={},
            tags={"mlflow.traceName": "chat_ok", "team": "nlp"},
        )
    )
    t2 = store.start_trace(
        TraceInfo(
            trace_id="tr-00000000000000010000000000000102",
            trace_location=TraceLocation.from_experiment_id("1"),
            request_time=1_700_000_002_000,
            execution_duration=800,
            state=TraceState.ERROR,
            trace_metadata={},
            tags={"mlflow.traceName": "chat_err", "team": "vision"},
        )
    )

    store.set_trace_tag(t1.trace_id, "priority", "high")
    assert store.get_trace_info(t1.trace_id).tags["priority"] == "high"
    store.delete_trace_tag(t1.trace_id, "priority")
    assert "priority" not in store.get_trace_info(t1.trace_id).tags

    store.link_traces_to_run([t1.trace_id, t2.trace_id], run.info.run_id)
    linked_t1 = store.get_trace_info(t1.trace_id)
    assert linked_t1.trace_metadata["mlflow.sourceRun"] == run.info.run_id

    store.link_prompts_to_trace(
        [{"name": "qa_prompt", "version": "2"}],
        t1.trace_id,
    )
    assert json.loads(store.get_trace_info(t1.trace_id).tags["mlflow.linkedPrompts"]) == [
        {"name": "qa_prompt", "version": "2"}
    ]

    # Search traces with filters, run_id, and pagination
    by_run, token = store.search_traces(
        experiment_ids=["1"],
        run_id=run.info.run_id,
        max_results=1,
        order_by=["timestamp_ms ASC"],
    )
    assert [t.trace_id for t in by_run] == [t1.trace_id]
    assert token is not None
    page2, token2 = store.search_traces(
        experiment_ids=["1"],
        run_id=run.info.run_id,
        max_results=1,
        order_by=["timestamp_ms ASC"],
        page_token=token,
    )
    assert [t.trace_id for t in page2] == [t2.trace_id]
    assert token2 is None

    err_only, _ = store.search_traces(
        experiment_ids=["1"],
        filter_string="status = 'ERROR' AND tag.team = 'vision'",
    )
    assert [t.trace_id for t in err_only] == [t2.trace_id]

    store.unlink_traces_from_run([t2.trace_id], run.info.run_id)
    after_unlink, _ = store.search_traces(experiment_ids=["1"], run_id=run.info.run_id)
    assert [t.trace_id for t in after_unlink] == [t1.trace_id]

    deleted = store.delete_traces("1", trace_ids=[t2.trace_id])
    assert deleted == 1
    remaining, _ = store.search_traces(experiment_ids=["1"])
    assert [t.trace_id for t in remaining] == [t1.trace_id]


def test_assessments_crud_and_trace_hydration() -> None:
    store, _ = _make_store(experiment_id=1)
    trace_id = "tr-00000000000000010000000000000201"
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id("1"),
            request_time=1_700_000_010_000,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={},
            tags={},
        )
    )

    fb = store.create_assessment(
        Feedback(
            name="faithfulness",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="judge-v1",
            ),
            trace_id=trace_id,
            value=0.9,
            rationale="Grounded in retrieved context",
            span_id="00000000000000a1",
            metadata={"rubric": "v2"},
        )
    )
    assert fb.assessment_id is not None

    exp = store.create_assessment(
        Expectation(
            name="expected_answer",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="annotator-1",
            ),
            trace_id=trace_id,
            value={"label": "approve"},
        )
    )
    assert exp.assessment_id is not None

    updated_fb = store.update_assessment(
        trace_id=trace_id,
        assessment_id=fb.assessment_id,
        name="faithfulness_v2",
        feedback=Feedback(name="faithfulness_v2", value=0.95),
        rationale="Verified all citations",
        metadata={"rubric": "v3"},
    )
    assert updated_fb.name == "faithfulness_v2"
    assert updated_fb.feedback is not None
    assert updated_fb.feedback.value == 0.95
    assert updated_fb.rationale == "Verified all citations"

    fetched_exp = store.get_assessment(trace_id, exp.assessment_id)
    assert fetched_exp.expectation is not None
    assert fetched_exp.expectation.value == {"label": "approve"}

    hydrated = store.get_trace_info(trace_id)
    assert len(hydrated.assessments) == 2

    store.delete_assessment(trace_id, exp.assessment_id)
    hydrated_after = store.get_trace_info(trace_id)
    assert len(hydrated_after.assessments) == 1
    with pytest.raises(MlflowException):
        store.get_assessment(trace_id, exp.assessment_id)


def test_query_trace_metrics_and_adr_0315_rollups() -> None:
    store, _ = _make_store(experiment_id=1)
    t_id = "tr-00000000000000010000000000000301"
    s1 = _build_mock_otel_span(
        trace_id=t_id,
        span_id="0000000000000001",
        parent_span_id=None,
        name="root_llm",
        span_type=SpanType.LLM,
        start_time_ns=1_700_000_000_000_000_000,
        end_time_ns=1_700_000_000_300_000_000,
        attributes={
            SpanAttributeKey.MODEL: "gpt-4o",
            SpanAttributeKey.MODEL_PROVIDER: "openai",
            SpanAttributeKey.CHAT_USAGE: {
                TokenUsageKey.INPUT_TOKENS: 100,
                TokenUsageKey.OUTPUT_TOKENS: 50,
                TokenUsageKey.TOTAL_TOKENS: 150,
            },
            SpanAttributeKey.LLM_COST: {
                "input_cost": 0.001,
                "output_cost": 0.002,
                "total_cost": 0.003,
            },
        },
    )
    store.log_spans("1", [s1])
    store.create_assessment(
        Feedback(
            name="quality",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="u1"),
            trace_id=t_id,
            value=0.8,
        )
    )

    trace_pts = store.query_trace_metrics(
        experiment_ids=["1"],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TOTAL_TOKENS,
        aggregations=[
            MetricAggregation(AggregationType.SUM),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=95.0),
        ],
        dimensions=["trace_name"],
        time_interval_seconds=3600,
        start_time_ms=1_699_999_000_000,
        end_time_ms=1_700_001_000_000,
    )
    assert len(trace_pts) == 1
    assert trace_pts[0].values["SUM"] == 150.0
    assert trace_pts[0].values["P95.0"] == 150.0
    assert trace_pts[0].dimensions["trace_name"] == "root_llm"

    span_pts = store.query_trace_metrics(
        experiment_ids=["1"],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.TOTAL_COST,
        aggregations=[
            MetricAggregation(AggregationType.SUM),
            MetricAggregation(AggregationType.AVG),
        ],
        dimensions=["span_model_name", "span_model_provider"],
    )
    assert len(span_pts) == 1
    assert pytest.approx(span_pts[0].values["SUM"]) == 0.003
    assert span_pts[0].dimensions["span_model_name"] == "gpt-4o"

    asmt_pts = store.query_trace_metrics(
        experiment_ids=["1"],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[MetricAggregation(AggregationType.AVG)],
        dimensions=["assessment_name"],
    )
    assert len(asmt_pts) == 1
    assert pytest.approx(asmt_pts[0].values["AVG"]) == 0.8

    ev_rollups = store.get_span_model_rollups(
        "1", span_type="LLM", model_name="gpt-4o", status="OK"
    )
    assert len(ev_rollups) == 1
    assert ev_rollups[0]["span_count"] == 1
    assert ev_rollups[0]["sum_total_tokens"] == 150

    col_rollups = store.get_columnar_span_model_rollups("1")
    assert len(col_rollups) == 1
    assert col_rollups[0]["p95_duration_ms"] == 300


def test_logged_models_and_run_lineage() -> None:
    store, _ = _make_store(experiment_id=1)
    run = store.create_run("1", "alice", 1_700_000_000_000, [], "train-run")
    store.log_batch(
        run.info.run_id,
        metrics=[Metric("loss", 0.125, 1_700_000_001_000, 1)],
        params=[],
        tags=[],
    )

    m = store.create_logged_model(
        experiment_id="1",
        name="classifier-v1",
        source_run_id=run.info.run_id,
        tags=[LoggedModelTag("framework", "pytorch")],  # type: ignore[no-untyped-call]
        params=[LoggedModelParameter("lr", "0.001")],  # type: ignore[no-untyped-call]
        model_type="classifier",
    )
    assert m.status == LoggedModelStatus.PENDING

    m_ready = store.finalize_logged_model(m.model_id, LoggedModelStatus.READY)
    assert m_ready.status == LoggedModelStatus.READY

    store.set_logged_model_tags(
        m.model_id,
        [LoggedModelTag("stage", "prod")],  # type: ignore[no-untyped-call]
    )
    store.log_logged_model_params(
        m.model_id,
        [LoggedModelParameter("epochs", "10")],  # type: ignore[no-untyped-call]
    )
    assert store.get_logged_model(m.model_id).tags["stage"] == "prod"
    assert store.get_logged_model(m.model_id).params["epochs"] == "10"

    store.delete_logged_model_tag(m.model_id, "framework")
    assert "framework" not in store.get_logged_model(m.model_id).tags

    store.log_inputs(
        run.info.run_id,
        datasets=[
            DatasetInput(
                dataset=Dataset(
                    name="imagenet_sub",
                    digest="abc12345",
                    source_type="s3",
                    source="s3://bucket/imagenet",
                ),
                tags=[InputTag(key="mlflow.data.context", value="train")],
            )
        ],
        models=[LoggedModelInput(model_id=m.model_id)],
    )
    store.log_outputs(run.info.run_id, [LoggedModelOutput(model_id=m.model_id, step=1)])
    store.record_logged_model(
        run.info.run_id, {"model_id": m.model_id, "flavors": {"python_function": {}}}
    )

    hydrated_run = store.get_run(run.info.run_id)
    assert len(hydrated_run.inputs.dataset_inputs) == 1
    assert hydrated_run.inputs.dataset_inputs[0].dataset.name == "imagenet_sub"
    assert [mi.model_id for mi in hydrated_run.inputs.model_inputs] == [m.model_id]
    assert [mo.model_id for mo in hydrated_run.outputs.model_outputs] == [m.model_id]

    searched = store.search_logged_models(
        experiment_ids=["1"],
        filter_string="name = 'classifier-v1' AND params.epochs = '10'",
        datasets=[{"dataset_name": "imagenet_sub", "dataset_digest": "abc12345"}],
    )
    assert [x.model_id for x in searched] == [m.model_id]

    store.delete_logged_model(m.model_id)
    with pytest.raises(MlflowException):
        store.get_logged_model(m.model_id)


def _find_riffdb_binary() -> str | None:
    env_bin = os.environ.get("RIFFDB")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    sibling_bin = Path(__file__).resolve().parents[2] / "riffdb" / "target" / "release" / "riffdb"
    if sibling_bin.is_file():
        return str(sibling_bin)
    return shutil.which("riffdb")


def test_live_riffdb_server_conformance() -> None:
    riffdb_bin = _find_riffdb_binary()
    if not riffdb_bin:
        pytest.skip("riffdb binary not available for live integration test")
    repo_root = Path(__file__).resolve().parents[1]
    python_bin = Path(sys.executable)
    exec_cmd = (
        f"{python_bin} -m riffdb_mlflow "
        '"$RIFFDB_ENDPOINT" "$RIFFDB_CREDENTIAL_FILE" "$RIFFDB_DATABASE"'
    )
    proc = subprocess.run(
        [riffdb_bin, "dev", "--reset", "--exec", exec_cmd],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Live conformance failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert '"schema":"riffdb-mlflow-loopback/v1"' in proc.stdout
