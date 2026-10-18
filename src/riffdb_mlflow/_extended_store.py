from __future__ import annotations

import json
import math
import re
import secrets
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from mlflow.entities import (
    Assessment,
    Dataset,
    DatasetInput,
    Expectation,
    Feedback,
    InputTag,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelParameter,
    LoggedModelStatus,
    LoggedModelTag,
    Metric,
    RunInputs,
    RunOutputs,
    RunTag,
    Span,
    Trace,
    TraceData,
    TraceInfo,
    ViewType,
)
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import (
    AssessmentMetricDimensionKey,
    AssessmentMetricKey,
    SpanMetricDimensionKey,
    SpanMetricKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
)
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_MODELS
from mlflow.utils.uri import append_to_uri_path
from riffdb_application import Timestamp

from riffdb_mlflow.generated.client import (
    AggregateSpanAnalyticsByModelFound,
    AggregateSpanAnalyticsByModelParams,
    DeleteLoggedModelEntityInput,
    DeleteLoggedModelEntityLoggedModelDeleted,
    DeleteLoggedModelEntityLoggedModelMissing,
    DeleteTraceAssessmentInput,
    DeleteTraceAssessmentTraceAssessmentDeleted,
    DeleteTraceAssessmentTraceAssessmentMissing,
    DeleteTraceInput,
    DeleteTraceSpanInput,
    DeleteTraceTraceDeleted,
    DeleteTraceTraceMissing,
    GetLoggedModelFound,
    GetLoggedModelMissing,
    GetLoggedModelParams,
    GetRunLineageFound,
    GetRunLineageParams,
    GetTraceFound,
    GetTraceMissing,
    GetTraceParams,
    IngestTraceSpansInput,
    IngestTraceSpansTraceSpansIngested,
    LifecycleStage,
    QueryAssessmentAnalyticsFound,
    QueryAssessmentAnalyticsParams,
    QuerySpanAnalyticsFound,
    QuerySpanAnalyticsParams,
    QuerySpanModelRollupFound,
    QuerySpanModelRollupParams,
    QueryTraceAnalyticsFound,
    QueryTraceAnalyticsParams,
    SearchLoggedModelsFound,
    SearchLoggedModelsParams,
    SearchTracesFound,
    SearchTracesParams,
    TraceSpan,
    UpsertLoggedModelInput,
    UpsertLoggedModelLoggedModelUpserted,
    UpsertRunLineageInput,
    UpsertRunLineageRunInactive,
    UpsertRunLineageRunLineageUpserted,
    UpsertRunLineageRunMissing,
    UpsertTraceAssessmentInput,
    UpsertTraceAssessmentTraceAssessmentUpserted,
    UpsertTraceAssessmentTraceMissing,
    UpsertTraceInput,
    UpsertTraceTraceUpserted,
)

TIME_BUCKET_LABEL = "time_bucket"

_TRACE_ID_EXPERIMENT_RE = re.compile(r"^tr-([0-9a-f]{16})[0-9a-f]{8,}$")
_MODEL_ID_EXPERIMENT_RE = re.compile(r"^m-([0-9a-f]{16})[0-9a-f]{8,}$")
_FILTER_CLAUSE_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\"`']+)\s*(=|!=|>=|<=|>|<|LIKE|ILIKE)\s*(.+?)\s*$",
    re.IGNORECASE,
)

_VIEW_CONFIGS: dict[MetricViewType, dict[str, tuple[set[AggregationType], set[str]]]] = {
    MetricViewType.TRACES: {
        TraceMetricKey.TRACE_COUNT: (
            {AggregationType.COUNT},
            {TraceMetricDimensionKey.TRACE_NAME, TraceMetricDimensionKey.TRACE_STATUS},
        ),
        TraceMetricKey.SESSION_COUNT: ({AggregationType.COUNT}, set()),
        TraceMetricKey.LATENCY: (
            {
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {TraceMetricDimensionKey.TRACE_NAME, TraceMetricDimensionKey.TRACE_STATUS},
        ),
        TraceMetricKey.INPUT_TOKENS: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {TraceMetricDimensionKey.TRACE_NAME},
        ),
        TraceMetricKey.OUTPUT_TOKENS: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {TraceMetricDimensionKey.TRACE_NAME},
        ),
        TraceMetricKey.TOTAL_TOKENS: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {TraceMetricDimensionKey.TRACE_NAME},
        ),
        TraceMetricKey.CACHE_READ_INPUT_TOKENS: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {TraceMetricDimensionKey.TRACE_NAME},
        ),
        TraceMetricKey.CACHE_CREATION_INPUT_TOKENS: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {TraceMetricDimensionKey.TRACE_NAME},
        ),
    },
    MetricViewType.SPANS: {
        SpanMetricKey.SPAN_COUNT: (
            {AggregationType.COUNT},
            {
                SpanMetricDimensionKey.SPAN_NAME,
                SpanMetricDimensionKey.SPAN_TYPE,
                SpanMetricDimensionKey.SPAN_STATUS,
                SpanMetricDimensionKey.SPAN_MODEL_NAME,
                SpanMetricDimensionKey.SPAN_MODEL_PROVIDER,
            },
        ),
        SpanMetricKey.LATENCY: (
            {
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {SpanMetricDimensionKey.SPAN_NAME, SpanMetricDimensionKey.SPAN_STATUS},
        ),
        SpanMetricKey.INPUT_COST: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {
                SpanMetricDimensionKey.SPAN_MODEL_NAME,
                SpanMetricDimensionKey.SPAN_MODEL_PROVIDER,
            },
        ),
        SpanMetricKey.OUTPUT_COST: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {
                SpanMetricDimensionKey.SPAN_MODEL_NAME,
                SpanMetricDimensionKey.SPAN_MODEL_PROVIDER,
            },
        ),
        SpanMetricKey.TOTAL_COST: (
            {
                AggregationType.SUM,
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {
                SpanMetricDimensionKey.SPAN_MODEL_NAME,
                SpanMetricDimensionKey.SPAN_MODEL_PROVIDER,
            },
        ),
    },
    MetricViewType.ASSESSMENTS: {
        AssessmentMetricKey.ASSESSMENT_COUNT: (
            {AggregationType.COUNT},
            {
                AssessmentMetricDimensionKey.ASSESSMENT_NAME,
                AssessmentMetricDimensionKey.ASSESSMENT_VALUE,
            },
        ),
        AssessmentMetricKey.ASSESSMENT_VALUE: (
            {
                AggregationType.AVG,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.PERCENTILE,
            },
            {AssessmentMetricDimensionKey.ASSESSMENT_NAME},
        ),
    },
}


def _validate_trace_metrics_params(
    view_type: MetricViewType,
    metric_name: str,
    aggregations: list[MetricAggregation],
    dimensions: list[str] | None,
) -> None:
    if view_type not in _VIEW_CONFIGS:
        supported_vt = [vt.value for vt in _VIEW_CONFIGS]
        raise MlflowException.invalid_parameter_value(
            f"view_type must be one of {supported_vt}, got '{view_type}'"
        )
    vt_cfg = _VIEW_CONFIGS[view_type]
    if metric_name not in vt_cfg:
        raise MlflowException.invalid_parameter_value(
            f"metric_name must be one of {list(vt_cfg.keys())}, got '{metric_name}'"
        )
    allowed_aggs, allowed_dims = vt_cfg[metric_name]
    agg_types = {agg.aggregation_type for agg in aggregations}
    if invalid_aggs := (agg_types - allowed_aggs):
        raise MlflowException.invalid_parameter_value(
            f"Found invalid aggregation_type(s): {sorted(a.value for a in invalid_aggs)}. "
            f"Supported aggregation types: {sorted(a.value for a in allowed_aggs)}"
        )
    dims_set = set(dimensions or [])
    if invalid_dims := (dims_set - allowed_dims):
        raise MlflowException.invalid_parameter_value(
            f"Found invalid dimension(s): {sorted(invalid_dims)}. "
            f"Supported dimensions: {sorted(allowed_dims)}"
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ts_from_ms(ms: int) -> Timestamp:
    safe_ms = max(0, int(ms))
    return Timestamp(seconds=safe_ms // 1000, nanos=(safe_ms % 1000) * 1_000_000)


def _ts_from_ns(ns: int) -> Timestamp:
    safe_ns = max(0, int(ns))
    return Timestamp(seconds=safe_ns // 1_000_000_000, nanos=safe_ns % 1_000_000_000)


def _ts_to_ms(ts: Timestamp) -> int:
    return int(ts.seconds) * 1000 + int(ts.nanos) // 1_000_000


def _try_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _linear_percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = max(0.0, min(1.0, float(percentile) / 100.0))
    pos = q * (len(sorted_values) - 1)
    lower_idx = math.floor(pos)
    upper_idx = math.ceil(pos)
    weight = pos - lower_idx
    return float(sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight)


class RiffDBExtendedStoreMixin(AbstractStore):
    _trace_experiment_map: dict[str, int]
    _model_experiment_map: dict[str, int]

    def _ensure_extended_indexes(self) -> None:
        if not hasattr(self, "_trace_experiment_map"):
            self._trace_experiment_map = {}
        if not hasattr(self, "_model_experiment_map"):
            self._model_experiment_map = {}

    def _causal_commit_seq(self, client: Any) -> int:
        seq = getattr(client, "_read_after_commit", None)
        if isinstance(seq, int) and seq >= 1:
            return seq
        return 1

    # ------------------------------------------------------------------
    # Run Lineage (Dataset inputs, LoggedModel inputs/outputs)
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_dataset_inputs(datasets: Sequence[DatasetInput]) -> bytes:
        payload = [
            {
                "dataset": {
                    "name": ds.dataset.name,
                    "digest": ds.dataset.digest,
                    "source_type": ds.dataset.source_type,
                    "source": ds.dataset.source,
                    "schema": ds.dataset.schema,
                    "profile": ds.dataset.profile,
                },
                "tags": [{"key": t.key, "value": t.value} for t in (ds.tags or [])],
            }
            for ds in datasets
        ]
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @staticmethod
    def _decode_dataset_inputs(raw: bytes) -> list[DatasetInput]:
        if not raw:
            return []
        items = json.loads(raw.decode("utf-8"))
        out: list[DatasetInput] = []
        for item in items:
            ds_dict = item["dataset"]
            dataset = Dataset(
                name=ds_dict["name"],
                digest=ds_dict["digest"],
                source_type=ds_dict["source_type"],
                source=ds_dict["source"],
                schema=ds_dict.get("schema"),
                profile=ds_dict.get("profile"),
            )
            tags = [InputTag(key=t["key"], value=t["value"]) for t in item.get("tags", [])]
            out.append(DatasetInput(dataset=dataset, tags=tags))
        return out

    @staticmethod
    def _encode_model_inputs(models: Sequence[LoggedModelInput]) -> bytes:
        payload = [{"model_id": m.model_id} for m in models]
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @staticmethod
    def _decode_model_inputs(raw: bytes) -> list[LoggedModelInput]:
        if not raw:
            return []
        items = json.loads(raw.decode("utf-8"))
        return [LoggedModelInput(model_id=item["model_id"]) for item in items]

    @staticmethod
    def _encode_model_outputs(models: Sequence[LoggedModelOutput]) -> bytes:
        payload = [{"model_id": m.model_id, "step": int(m.step)} for m in models]
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @staticmethod
    def _decode_model_outputs(raw: bytes) -> list[LoggedModelOutput]:
        if not raw:
            return []
        items = json.loads(raw.decode("utf-8"))
        return [
            LoggedModelOutput(model_id=item["model_id"], step=int(item.get("step", 0)))
            for item in items
        ]

    def _load_run_lineage(
        self, client: Any, experiment_id: int, run_id: str
    ) -> tuple[RunInputs, RunOutputs]:
        if not hasattr(client, "get_run_lineage"):
            return RunInputs(dataset_inputs=[]), RunOutputs(model_outputs=[])
        res = client.get_run_lineage(
            GetRunLineageParams(experiment_id=experiment_id, run_id=run_id)
        )
        if not isinstance(res.value, GetRunLineageFound):
            return RunInputs(dataset_inputs=[]), RunOutputs(model_outputs=[])
        lineage = res.value.lineage
        dataset_inputs = self._decode_dataset_inputs(lineage.packed_datasets)
        model_inputs = self._decode_model_inputs(lineage.packed_model_inputs)
        model_outputs = self._decode_model_outputs(lineage.packed_model_outputs)
        return (
            RunInputs(dataset_inputs=dataset_inputs, model_inputs=model_inputs),
            RunOutputs(model_outputs=model_outputs),
        )

    def log_inputs(
        self,
        run_id: str,
        datasets: Sequence[DatasetInput] | None = None,
        models: Sequence[LoggedModelInput] | None = None,
    ) -> None:
        normalized_id, experiment_id = self._run_experiment_id(run_id)  # type: ignore[attr-defined]
        client = self._application()  # type: ignore[attr-defined]
        existing_inputs, existing_outputs = self._load_run_lineage(
            client, experiment_id, normalized_id
        )
        by_dataset_key: dict[tuple[str, str], DatasetInput] = {
            (d.dataset.name, d.dataset.digest): d for d in (existing_inputs.dataset_inputs or [])
        }
        for ds in datasets or []:
            if not isinstance(ds, DatasetInput):
                raise MlflowException.invalid_parameter_value(
                    f"Expected DatasetInput instance, got {type(ds).__name__}"
                )
            by_dataset_key[(ds.dataset.name, ds.dataset.digest)] = ds

        by_model_id: dict[str, LoggedModelInput] = {
            m.model_id: m for m in (existing_inputs.model_inputs or [])
        }
        for model in models or []:
            if not isinstance(model, LoggedModelInput):
                raise MlflowException.invalid_parameter_value(
                    f"Expected LoggedModelInput instance, got {type(model).__name__}"
                )
            by_model_id[model.model_id] = model

        result = client.upsert_run_lineage(
            UpsertRunLineageInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=normalized_id,
                packed_datasets=self._encode_dataset_inputs(list(by_dataset_key.values())),
                packed_model_inputs=self._encode_model_inputs(list(by_model_id.values())),
                packed_model_outputs=self._encode_model_outputs(
                    existing_outputs.model_outputs or []
                ),
            )
        )
        if isinstance(result.outcome, UpsertRunLineageRunMissing):
            self._missing_run(run_id)  # type: ignore[attr-defined]
        if isinstance(result.outcome, UpsertRunLineageRunInactive):
            raise MlflowException(
                f"The run {run_id} must be in the active state.",
                error_code=INVALID_STATE,
            )
        if not isinstance(result.outcome, UpsertRunLineageRunLineageUpserted):
            raise MlflowException("Invalid UpsertRunLineage response from RiffDB")

    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]) -> None:
        normalized_id, experiment_id = self._run_experiment_id(run_id)  # type: ignore[attr-defined]
        if not isinstance(models, Sequence):
            raise MlflowException.invalid_parameter_value("models must be a sequence")
        client = self._application()  # type: ignore[attr-defined]
        existing_inputs, existing_outputs = self._load_run_lineage(
            client, experiment_id, normalized_id
        )
        by_model_id: dict[str, LoggedModelOutput] = {
            m.model_id: m for m in (existing_outputs.model_outputs or [])
        }
        for model in models:
            if not isinstance(model, LoggedModelOutput):
                raise MlflowException.invalid_parameter_value(
                    f"Expected LoggedModelOutput instance, got {type(model).__name__}"
                )
            by_model_id[model.model_id] = model

        result = client.upsert_run_lineage(
            UpsertRunLineageInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=normalized_id,
                packed_datasets=self._encode_dataset_inputs(existing_inputs.dataset_inputs or []),
                packed_model_inputs=self._encode_model_inputs(existing_inputs.model_inputs or []),
                packed_model_outputs=self._encode_model_outputs(list(by_model_id.values())),
            )
        )
        if isinstance(result.outcome, UpsertRunLineageRunMissing):
            self._missing_run(run_id)  # type: ignore[attr-defined]
        if isinstance(result.outcome, UpsertRunLineageRunInactive):
            raise MlflowException(
                f"The run {run_id} must be in the active state.",
                error_code=INVALID_STATE,
            )
        if not isinstance(result.outcome, UpsertRunLineageRunLineageUpserted):
            raise MlflowException("Invalid UpsertRunLineage response from RiffDB")

    def record_logged_model(self, run_id: str, mlflow_model: Any) -> None:
        if mlflow_model is None:
            raise MlflowException.invalid_parameter_value("mlflow_model cannot be None")
        model_dict = (
            mlflow_model.to_dict() if hasattr(mlflow_model, "to_dict") else dict(mlflow_model)
        )
        run = self.get_run(run_id)  # type: ignore[no-untyped-call]
        existing_raw = run.data.tags.get(MLFLOW_LOGGED_MODELS)
        history: list[Any] = json.loads(existing_raw) if existing_raw else []
        if model_dict not in history:
            history.append(model_dict)
        self.set_tag(
            run_id,
            RunTag(MLFLOW_LOGGED_MODELS, json.dumps(history)),  # type: ignore[no-untyped-call]
        )

    # ------------------------------------------------------------------
    # Traces & Spans (including Out-of-Order Span Ingestion)
    # ------------------------------------------------------------------

    def _require_active_experiment(self, experiment_id: int) -> Any:
        exp = self.get_experiment(str(experiment_id))  # type: ignore[no-untyped-call]
        if exp.lifecycle_stage != "active":
            raise MlflowException(
                f"Experiment {experiment_id} must be active.",
                error_code=INVALID_STATE,
            )
        return exp

    def _new_trace_id(self, experiment_id: int) -> str:
        return f"tr-{experiment_id:016x}{secrets.token_hex(8)}"

    def _resolve_trace_experiment_id(self, trace_id: str, location: str | None = None) -> int:
        self._ensure_extended_indexes()
        if not isinstance(trace_id, str) or not trace_id or len(trace_id.encode("utf-8")) > 64:
            raise MlflowException.invalid_parameter_value(f"Invalid trace_id: {trace_id!r}")
        if location is not None:
            exp_id = int(self._numeric_experiment_id(location))  # type: ignore[attr-defined]
            self._trace_experiment_map[trace_id] = exp_id
            return exp_id
        cached = self._trace_experiment_map.get(trace_id)
        if cached is not None:
            return cached
        matched = _TRACE_ID_EXPERIMENT_RE.match(trace_id)
        if matched is not None:
            exp_id = int(matched.group(1), 16)
            self._trace_experiment_map[trace_id] = exp_id
            return exp_id
        client = self._application()  # type: ignore[attr-defined]
        for exp in self.search_experiments(ViewType.ALL):  # type: ignore[no-untyped-call]
            exp_id = int(exp.experiment_id)
            res = client.get_trace(GetTraceParams(experiment_id=exp_id, trace_id=trace_id))
            if isinstance(res.value, GetTraceFound):
                self._trace_experiment_map[trace_id] = exp_id
                return exp_id
        raise MlflowException(
            f"Trace with ID '{trace_id}' does not exist.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    @staticmethod
    def _encode_json_dict(mapping: dict[str, Any]) -> bytes:
        return json.dumps(mapping, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @staticmethod
    def _decode_json_dict(raw: bytes) -> dict[str, str]:
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        return {str(k): str(v) for k, v in data.items()}

    @staticmethod
    def _encode_json_list(items: Sequence[Any]) -> bytes:
        return json.dumps(list(items), separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _decode_json_list(raw: bytes) -> list[Any]:
        if not raw:
            return []
        return list(json.loads(raw.decode("utf-8")))

    @classmethod
    def _assessment_from_row(cls, row: Any) -> Assessment:
        payload = json.loads(row.packed_assessment_json.decode("utf-8"))
        if row.assessment_type == "EXPECTATION" or "expectation" in payload:
            assessment: Assessment = Expectation.from_dictionary(payload)
        else:
            assessment = Feedback.from_dictionary(payload)
        assessment.assessment_id = row.assessment_id
        assessment.trace_id = row.trace_id
        assessment.create_time_ms = int(row.created_timestamp_ms)
        assessment.last_update_time_ms = int(row.last_updated_timestamp_ms)
        return assessment

    @classmethod
    def _to_mlflow_trace_info(
        cls,
        row: Any,
        assessments: Sequence[Any] = (),
    ) -> TraceInfo:
        metadata = cls._decode_json_dict(row.packed_trace_metadata)
        tags = cls._decode_json_dict(row.packed_tags)
        if row.trace_name and "mlflow.traceName" not in tags:
            tags["mlflow.traceName"] = row.trace_name
        if row.source_run_id and "mlflow.sourceRun" not in metadata:
            metadata["mlflow.sourceRun"] = row.source_run_id
        if row.session_id and "mlflow.trace.session" not in metadata:
            metadata["mlflow.trace.session"] = row.session_id
        if (
            row.total_tokens > 0
            or row.total_input_tokens > 0
            or row.total_output_tokens > 0
            or row.cache_read_input_tokens > 0
            or row.cache_creation_input_tokens > 0
        ) and "mlflow.trace.tokenUsage" not in metadata:
            usage_payload: dict[str, int] = {
                "input_tokens": int(row.total_input_tokens),
                "output_tokens": int(row.total_output_tokens),
                "total_tokens": int(row.total_tokens),
            }
            if row.cache_read_input_tokens > 0:
                usage_payload["cache_read_input_tokens"] = int(row.cache_read_input_tokens)
            if row.cache_creation_input_tokens > 0:
                usage_payload["cache_creation_input_tokens"] = int(row.cache_creation_input_tokens)
            metadata["mlflow.trace.tokenUsage"] = json.dumps(usage_payload)

        state_str = row.state or TraceState.STATE_UNSPECIFIED.value
        try:
            state = TraceState(state_str)
        except ValueError:
            state = TraceState.STATE_UNSPECIFIED

        hydrated_assessments = [cls._assessment_from_row(a) for a in assessments]
        return TraceInfo(
            trace_id=row.trace_id,
            trace_location=TraceLocation.from_experiment_id(str(row.experiment_id)),
            request_time=int(row.request_time_ms),
            execution_duration=(
                int(row.execution_duration_ms) if row.execution_duration_ms >= 0 else None
            ),
            state=state,
            trace_metadata=metadata,
            tags=tags,
            assessments=hydrated_assessments,
            client_request_id=row.client_request_id,
            request_preview=row.request_preview,
            response_preview=row.response_preview,
        )

    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        self._ensure_extended_indexes()
        if not isinstance(trace_info, TraceInfo):
            raise MlflowException.invalid_parameter_value("trace_info must be a TraceInfo")
        exp_id_str = trace_info.experiment_id
        if not exp_id_str:
            raise MlflowException.invalid_parameter_value("TraceInfo must have an experiment_id")
        experiment_id = int(self._numeric_experiment_id(exp_id_str))  # type: ignore[attr-defined]
        experiment = self._require_active_experiment(experiment_id)

        trace_id = trace_info.trace_id or self._new_trace_id(experiment_id)
        self._trace_experiment_map[trace_id] = experiment_id

        client = self._application()  # type: ignore[attr-defined]
        existing_res = client.get_trace(
            GetTraceParams(experiment_id=experiment_id, trace_id=trace_id)
        )
        existing_row = (
            existing_res.value.trace if isinstance(existing_res.value, GetTraceFound) else None
        )
        existing_assessments = (
            existing_res.value.assessments if isinstance(existing_res.value, GetTraceFound) else ()
        )

        metadata = dict(trace_info.trace_metadata or {})
        tags = dict(trace_info.tags or {})
        if "mlflow.artifactLocation" not in tags and experiment.artifact_location:
            tags["mlflow.artifactLocation"] = str(
                append_to_uri_path(  # type: ignore[no-untyped-call]
                    experiment.artifact_location, "traces", trace_id, "artifacts"
                )
            )

        request_time_ms = int(trace_info.request_time or _now_ms())
        execution_duration_ms = (
            int(trace_info.execution_duration) if trace_info.execution_duration is not None else 0
        )
        state_val = (
            trace_info.state.value
            if isinstance(trace_info.state, TraceState)
            else str(trace_info.state or TraceState.IN_PROGRESS.value)
        )
        trace_name = tags.get("mlflow.traceName", "")
        source_run_id = metadata.get("mlflow.sourceRun")
        session_id = metadata.get("mlflow.trace.session")

        usage_raw = metadata.get("mlflow.trace.tokenUsage")
        usage_dict = _try_parse_json(usage_raw) if usage_raw else {}
        if not isinstance(usage_dict, dict):
            usage_dict = {}
        input_tokens = int(usage_dict.get("input_tokens", 0))
        output_tokens = int(usage_dict.get("output_tokens", 0))
        total_tokens = int(usage_dict.get("total_tokens", input_tokens + output_tokens))
        cache_read_tokens = int(usage_dict.get("cache_read_input_tokens", 0))
        cache_creation_tokens = int(usage_dict.get("cache_creation_input_tokens", 0))

        cost_raw = metadata.get("mlflow.trace.cost")
        cost_dict = _try_parse_json(cost_raw) if cost_raw else {}
        if not isinstance(cost_dict, dict):
            cost_dict = {}
        total_cost_micros = round(float(cost_dict.get("total_cost", 0.0)) * 1_000_000)

        span_count = 0
        error_span_count = 0
        run_ids: list[str] = [source_run_id] if source_run_id else []
        prompt_versions: list[dict[str, Any]] = []

        if existing_row is not None:
            existing_meta = self._decode_json_dict(existing_row.packed_trace_metadata)
            existing_tags = self._decode_json_dict(existing_row.packed_tags)
            metadata = {**existing_meta, **metadata}
            tags = {**existing_tags, **tags}
            span_count = int(existing_row.span_count)
            error_span_count = int(existing_row.error_span_count)
            if execution_duration_ms == 0 and existing_row.execution_duration_ms > 0:
                execution_duration_ms = int(existing_row.execution_duration_ms)
            if state_val in (
                TraceState.IN_PROGRESS.value,
                TraceState.STATE_UNSPECIFIED.value,
            ) and existing_row.state in (TraceState.OK.value, TraceState.ERROR.value):
                state_val = existing_row.state
            if not trace_name and existing_row.trace_name:
                trace_name = existing_row.trace_name
                tags["mlflow.traceName"] = trace_name
            if total_tokens == 0 and existing_row.total_tokens > 0:
                input_tokens = int(existing_row.total_input_tokens)
                output_tokens = int(existing_row.total_output_tokens)
                total_tokens = int(existing_row.total_tokens)
                cache_read_tokens = int(existing_row.cache_read_input_tokens)
                cache_creation_tokens = int(existing_row.cache_creation_input_tokens)
            if total_cost_micros == 0 and existing_row.total_cost_micros > 0:
                total_cost_micros = int(existing_row.total_cost_micros)
            if not session_id and existing_row.session_id:
                session_id = existing_row.session_id
            run_ids = list(
                dict.fromkeys(self._decode_json_list(existing_row.packed_run_ids) + run_ids)
            )
            prompt_versions = self._decode_json_list(existing_row.packed_prompt_versions)

        result = client.upsert_trace(
            UpsertTraceInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                trace_id=trace_id,
                request_time_ms=request_time_ms,
                execution_duration_ms=execution_duration_ms,
                state=state_val,
                status_code=state_val,
                trace_name=trace_name[:256],
                client_request_id=trace_info.client_request_id,
                request_preview=(
                    trace_info.request_preview[:2000]
                    if trace_info.request_preview
                    else (existing_row.request_preview if existing_row else None)
                ),
                response_preview=(
                    trace_info.response_preview[:2000]
                    if trace_info.response_preview
                    else (existing_row.response_preview if existing_row else None)
                ),
                source_run_id=source_run_id
                or (existing_row.source_run_id if existing_row else None),
                span_count=span_count,
                error_span_count=error_span_count,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_input_tokens=cache_read_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
                total_cost_micros=total_cost_micros,
                session_id=session_id,
                packed_trace_metadata=self._encode_json_dict(metadata),
                packed_tags=self._encode_json_dict(tags),
                packed_run_ids=self._encode_json_list(run_ids),
                packed_prompt_versions=self._encode_json_list(prompt_versions),
                requested_at=_ts_from_ms(request_time_ms),
                initialized=True,
            )
        )
        if not isinstance(result.outcome, UpsertTraceTraceUpserted):
            raise MlflowException("Invalid UpsertTrace response from RiffDB")
        return self._to_mlflow_trace_info(result.outcome.trace, existing_assessments)

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        experiment_id = self._resolve_trace_experiment_id(trace_id)
        client = self._application()  # type: ignore[attr-defined]
        res = client.get_trace(GetTraceParams(experiment_id=experiment_id, trace_id=trace_id))
        if isinstance(res.value, GetTraceMissing):
            raise MlflowException(
                f"Trace with ID '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.value, GetTraceFound):
            raise MlflowException("Invalid GetTrace response from RiffDB")
        return self._to_mlflow_trace_info(res.value.trace, res.value.assessments)

    def batch_get_traces(self, trace_ids: list[str], location: str | None = None) -> list[Trace]:
        if not trace_ids:
            return []
        client = self._application()  # type: ignore[attr-defined]
        traces: list[Trace] = []
        for trace_id in trace_ids:
            experiment_id = self._resolve_trace_experiment_id(trace_id, location=location)
            res = client.get_trace(GetTraceParams(experiment_id=experiment_id, trace_id=trace_id))
            if isinstance(res.value, GetTraceMissing):
                raise MlflowException(
                    f"Trace with ID '{trace_id}' does not exist.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            if not isinstance(res.value, GetTraceFound):
                raise MlflowException("Invalid GetTrace response from RiffDB")
            info = self._to_mlflow_trace_info(res.value.trace, res.value.assessments)
            spans: list[Span] = []
            for span_row in res.value.spans:
                span_dict = json.loads(span_row.packed_span_json.decode("utf-8"))
                spans.append(Span.from_dict(span_dict))
            traces.append(Trace(info=info, data=TraceData(spans=spans)))
        return traces

    @staticmethod
    def _extract_span_telemetry(span: Span) -> dict[str, Any]:
        span_dict = span.to_dict()
        attrs = span.attributes or {}
        raw_attrs = span_dict.get("attributes", {}) or {}

        def _attr(key: str, default: Any = None) -> Any:
            if key in attrs:
                return _try_parse_json(attrs[key])
            if key in raw_attrs:
                return _try_parse_json(raw_attrs[key])
            return default

        span_type = str(
            _attr("mlflow.spanType", getattr(span, "span_type", "UNKNOWN")) or "UNKNOWN"
        )
        status_obj = getattr(span, "status", None)
        status_code_raw: Any = getattr(status_obj, "status_code", status_obj)
        status_str = (
            str(status_code_raw.name)
            if status_code_raw is not None and hasattr(status_code_raw, "name")
            else str(status_code_raw or "OK")
        )
        if status_str.startswith("SpanStatusCode."):
            status_str = status_str.split(".", 1)[1]

        start_ns = int(getattr(span, "start_time_ns", 0) or 0)
        end_ns = int(getattr(span, "end_time_ns", start_ns) or start_ns)
        duration_ms = max(0, (end_ns - start_ns) // 1_000_000)

        model_name = str(
            _attr("mlflow.llm.model")
            or _attr("mlflow.chat.model")
            or _attr("gen_ai.request.model")
            or _attr("model")
            or _attr("llm.model_name")
            or ""
        )
        model_provider = str(
            _attr("mlflow.llm.provider")
            or _attr("mlflow.chat.provider")
            or _attr("gen_ai.system")
            or _attr("model_provider")
            or ""
        )

        usage = _attr("mlflow.chat.tokenUsage") or {}
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = int(
            usage.get("input_tokens")
            or _attr("gen_ai.usage.input_tokens")
            or _attr("input_tokens")
            or 0
        )
        output_tokens = int(
            usage.get("output_tokens")
            or _attr("gen_ai.usage.output_tokens")
            or _attr("output_tokens")
            or 0
        )
        total_tokens = int(
            usage.get("total_tokens") or _attr("total_tokens") or (input_tokens + output_tokens)
        )
        cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
        cache_creation_tokens = int(usage.get("cache_creation_input_tokens") or 0)

        cost = _attr("mlflow.llm.cost") or {}
        if not isinstance(cost, dict):
            cost = {}
        in_cost_usd = float(cost.get("input_cost") or _attr("input_cost") or 0.0)
        out_cost_usd = float(cost.get("output_cost") or _attr("output_cost") or 0.0)
        tot_cost_usd = float(
            cost.get("total_cost") or _attr("total_cost") or (in_cost_usd + out_cost_usd)
        )
        input_cost_micros = round(in_cost_usd * 1_000_000)
        output_cost_micros = round(out_cost_usd * 1_000_000)
        total_cost_micros = round(tot_cost_usd * 1_000_000)

        session_id = (
            _attr("session.id") or _attr("gen_ai.conversation.id") or _attr("mlflow.trace.session")
        )
        inputs_preview = _attr("mlflow.spanInputs")
        outputs_preview = _attr("mlflow.spanOutputs")

        packed_json = json.dumps(span_dict, cls=TraceJSONEncoder, separators=(",", ":")).encode(
            "utf-8"
        )
        return {
            "span_id": str(span.span_id),
            "trace_id": str(span.trace_id),
            "parent_id": str(span.parent_id) if span.parent_id else None,
            "name": str(span.name or "")[:256],
            "span_type": span_type[:64],
            "status": status_str[:32],
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "duration_ms": duration_ms,
            "model_name": model_name[:128],
            "model_provider": model_provider[:64],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_creation_tokens,
            "input_cost_micros": input_cost_micros,
            "output_cost_micros": output_cost_micros,
            "total_cost_micros": total_cost_micros,
            "session_id": str(session_id)[:128] if session_id else None,
            "inputs_preview": (
                json.dumps(inputs_preview)[:2000] if inputs_preview is not None else None
            ),
            "outputs_preview": (
                json.dumps(outputs_preview)[:2000] if outputs_preview is not None else None
            ),
            "packed_span_json": packed_json,
        }

    def log_spans(
        self,
        location: str,
        spans: list[Span],
        tracking_uri: str | None = None,
    ) -> list[Span]:
        self._ensure_extended_indexes()
        if not spans:
            return []
        experiment_id = int(self._numeric_experiment_id(location))  # type: ignore[attr-defined]
        experiment = self._require_active_experiment(experiment_id)
        client = self._application()  # type: ignore[attr-defined]

        spans_by_trace: dict[str, list[Span]] = defaultdict(list)
        for span in spans:
            spans_by_trace[span.trace_id].append(span)
            self._trace_experiment_map[span.trace_id] = experiment_id

        for trace_id, trace_spans in spans_by_trace.items():
            existing_res = client.get_trace(
                GetTraceParams(experiment_id=experiment_id, trace_id=trace_id)
            )
            existing_trace = (
                existing_res.value.trace if isinstance(existing_res.value, GetTraceFound) else None
            )
            existing_spans_by_id: dict[str, Any] = {}
            if isinstance(existing_res.value, GetTraceFound):
                for s_row in existing_res.value.spans:
                    existing_spans_by_id[s_row.span_id] = {
                        "span_id": s_row.span_id,
                        "parent_id": s_row.parent_id,
                        "name": s_row.name,
                        "span_type": s_row.span_type,
                        "status": s_row.status,
                        "start_time_ns": int(s_row.start_time_ns),
                        "end_time_ns": int(s_row.end_time_ns),
                        "duration_ms": int(s_row.duration_ms),
                        "model_name": s_row.model_name,
                        "model_provider": s_row.model_provider,
                        "input_tokens": int(s_row.input_tokens),
                        "output_tokens": int(s_row.output_tokens),
                        "total_tokens": int(s_row.total_tokens),
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "input_cost_micros": int(s_row.input_cost_micros),
                        "output_cost_micros": int(s_row.output_cost_micros),
                        "total_cost_micros": int(s_row.total_cost_micros),
                        "session_id": None,
                        "inputs_preview": None,
                        "outputs_preview": None,
                    }

            batch_telemetry = [self._extract_span_telemetry(s) for s in trace_spans]
            for item in batch_telemetry:
                existing_spans_by_id[item["span_id"]] = item

            all_spans = list(existing_spans_by_id.values())
            min_start_ns = min(s["start_time_ns"] for s in all_spans)
            max_end_ns = max(s["end_time_ns"] for s in all_spans)
            inferred_request_ms = min_start_ns // 1_000_000
            inferred_duration_ms = max(0, (max_end_ns - min_start_ns) // 1_000_000)

            root_span = next((s for s in all_spans if s["parent_id"] is None), None)
            span_count = len(all_spans)
            error_span_count = sum(1 for s in all_spans if s["status"] == "ERROR")
            sum_input_tokens = sum(s["input_tokens"] for s in all_spans)
            sum_output_tokens = sum(s["output_tokens"] for s in all_spans)
            sum_total_tokens = sum(s["total_tokens"] for s in all_spans)
            sum_cache_read = sum(s.get("cache_read_input_tokens", 0) for s in all_spans)
            sum_cache_creation = sum(s.get("cache_creation_input_tokens", 0) for s in all_spans)
            sum_cost_micros = sum(s["total_cost_micros"] for s in all_spans)

            metadata: dict[str, str] = (
                self._decode_json_dict(existing_trace.packed_trace_metadata)
                if existing_trace is not None
                else {}
            )
            tags: dict[str, str] = (
                self._decode_json_dict(existing_trace.packed_tags)
                if existing_trace is not None
                else {}
            )
            if "mlflow.artifactLocation" not in tags and experiment.artifact_location:
                tags["mlflow.artifactLocation"] = str(
                    append_to_uri_path(  # type: ignore[no-untyped-call]
                        experiment.artifact_location, "traces", trace_id, "artifacts"
                    )
                )
            tags["mlflow.trace.spansLocation"] = "TRACKING_STORE"

            trace_name = (
                existing_trace.trace_name
                if existing_trace is not None and existing_trace.trace_name
                else (root_span["name"] if root_span is not None else trace_spans[0].name)
            )
            if trace_name and "mlflow.traceName" not in tags:
                tags["mlflow.traceName"] = trace_name

            if root_span is not None:
                root_status = root_span["status"]
                if root_status == "ERROR":
                    state_val = TraceState.ERROR.value
                elif root_status in ("OK", "UNSET"):
                    state_val = TraceState.OK.value
                else:
                    state_val = TraceState.IN_PROGRESS.value
            elif existing_trace is not None and existing_trace.initialized:
                state_val = existing_trace.state
            else:
                state_val = (
                    TraceState.ERROR.value if error_span_count > 0 else TraceState.IN_PROGRESS.value
                )

            request_time_ms = (
                int(existing_trace.request_time_ms)
                if existing_trace is not None and existing_trace.initialized
                else inferred_request_ms
            )
            execution_duration_ms = (
                int(existing_trace.execution_duration_ms)
                if existing_trace is not None
                and existing_trace.initialized
                and existing_trace.execution_duration_ms > 0
                else inferred_duration_ms
            )

            session_id = next(
                (s["session_id"] for s in batch_telemetry if s.get("session_id")),
                existing_trace.session_id if existing_trace is not None else None,
            )
            if session_id:
                metadata["mlflow.trace.session"] = session_id

            if sum_total_tokens > 0 or sum_input_tokens > 0 or sum_output_tokens > 0:
                usage_obj: dict[str, int] = {
                    "input_tokens": sum_input_tokens,
                    "output_tokens": sum_output_tokens,
                    "total_tokens": sum_total_tokens,
                }
                if sum_cache_read > 0:
                    usage_obj["cache_read_input_tokens"] = sum_cache_read
                if sum_cache_creation > 0:
                    usage_obj["cache_creation_input_tokens"] = sum_cache_creation
                metadata["mlflow.trace.tokenUsage"] = json.dumps(usage_obj)

            if sum_cost_micros > 0:
                sum_in_cost = sum(s["input_cost_micros"] for s in all_spans) / 1_000_000.0
                sum_out_cost = sum(s["output_cost_micros"] for s in all_spans) / 1_000_000.0
                metadata["mlflow.trace.cost"] = json.dumps(
                    {
                        "input_cost": sum_in_cost,
                        "output_cost": sum_out_cost,
                        "total_cost": sum_cost_micros / 1_000_000.0,
                    }
                )

            request_preview = (
                existing_trace.request_preview
                if existing_trace is not None and existing_trace.request_preview
                else (root_span.get("inputs_preview") if root_span else None)
            )
            response_preview = (
                existing_trace.response_preview
                if existing_trace is not None and existing_trace.response_preview
                else (root_span.get("outputs_preview") if root_span else None)
            )

            for chunk_start in range(0, len(batch_telemetry), 64):
                chunk = batch_telemetry[chunk_start : chunk_start + 64]
                span_records = tuple(
                    TraceSpan(
                        experiment_id=experiment_id,
                        trace_id=trace_id,
                        span_id=item["span_id"],
                        parent_id=item["parent_id"],
                        name=item["name"],
                        span_type=item["span_type"],
                        status=item["status"],
                        start_time_ns=item["start_time_ns"],
                        end_time_ns=item["end_time_ns"],
                        duration_ms=item["duration_ms"],
                        model_name=item["model_name"],
                        model_provider=item["model_provider"],
                        input_tokens=item["input_tokens"],
                        output_tokens=item["output_tokens"],
                        total_tokens=item["total_tokens"],
                        input_cost_micros=item["input_cost_micros"],
                        output_cost_micros=item["output_cost_micros"],
                        total_cost_micros=item["total_cost_micros"],
                        started_at=_ts_from_ns(item["start_time_ns"]),
                        packed_span_json=item["packed_span_json"],
                    )
                    for item in chunk
                )
                res = client.ingest_trace_spans(
                    IngestTraceSpansInput(
                        request_id=str(uuid4()),
                        experiment_id=experiment_id,
                        trace_id=trace_id,
                        request_time_ms=request_time_ms,
                        execution_duration_ms=execution_duration_ms,
                        state=state_val,
                        status_code=state_val,
                        trace_name=trace_name[:256],
                        client_request_id=(
                            existing_trace.client_request_id if existing_trace else None
                        ),
                        request_preview=request_preview,
                        response_preview=response_preview,
                        source_run_id=(existing_trace.source_run_id if existing_trace else None),
                        span_count=span_count,
                        error_span_count=error_span_count,
                        total_input_tokens=sum_input_tokens,
                        total_output_tokens=sum_output_tokens,
                        total_tokens=sum_total_tokens,
                        cache_read_input_tokens=sum_cache_read,
                        cache_creation_input_tokens=sum_cache_creation,
                        total_cost_micros=sum_cost_micros,
                        session_id=session_id,
                        packed_trace_metadata=self._encode_json_dict(metadata),
                        packed_tags=self._encode_json_dict(tags),
                        packed_run_ids=(
                            existing_trace.packed_run_ids if existing_trace is not None else b"[]"
                        ),
                        packed_prompt_versions=(
                            existing_trace.packed_prompt_versions
                            if existing_trace is not None
                            else b"[]"
                        ),
                        requested_at=_ts_from_ms(request_time_ms),
                        initialized=bool(
                            existing_trace.initialized
                            if existing_trace is not None
                            else (root_span is not None)
                        ),
                        spans=span_records,
                    )
                )
                if not isinstance(res.outcome, IngestTraceSpansTraceSpansIngested):
                    raise MlflowException("Invalid IngestTraceSpans response from RiffDB")
        return list(spans)

    async def log_spans_async(self, location: str, spans: list[Span]) -> list[Span]:
        return self.log_spans(location, spans)

    def ingest_spans_out_of_order(self, location: str, spans: Sequence[Span]) -> list[Span]:
        """Ingests LLM spans that may arrive out of order."""
        return self.log_spans(location, list(spans))

    @staticmethod
    def _matches_trace_filter(info: TraceInfo, row: Any, filter_string: str | None) -> bool:
        if not filter_string or not filter_string.strip():
            return True
        clauses = [c.strip() for c in re.split(r"\bAND\b", filter_string, flags=re.IGNORECASE)]
        run_ids = set(json.loads(row.packed_run_ids.decode("utf-8")) if row.packed_run_ids else [])
        if row.source_run_id:
            run_ids.add(row.source_run_id)

        for clause in clauses:
            if not clause:
                continue
            m = _FILTER_CLAUSE_RE.match(clause)
            if not m:
                raise MlflowException.invalid_parameter_value(
                    f"Unsupported trace filter clause: {clause!r}"
                )
            raw_lhs, op, raw_rhs = m.group(1), m.group(2).upper(), m.group(3).strip()
            lhs = raw_lhs.strip("`'\"")
            if (raw_rhs.startswith("'") and raw_rhs.endswith("'")) or (
                raw_rhs.startswith('"') and raw_rhs.endswith('"')
            ):
                rhs_val: Any = raw_rhs[1:-1]
            else:
                try:
                    rhs_val = int(raw_rhs)
                except ValueError:
                    try:
                        rhs_val = float(raw_rhs)
                    except ValueError:
                        rhs_val = raw_rhs

            actual: Any = None
            if lhs in ("status", "trace.status", "attributes.status"):
                actual = info.state.value if isinstance(info.state, TraceState) else str(info.state)
            elif lhs in ("name", "trace.name", "attributes.name"):
                actual = info.tags.get("mlflow.traceName", row.trace_name)
            elif lhs in (
                "timestamp_ms",
                "request_time",
                "trace.timestamp_ms",
                "attributes.timestamp_ms",
            ):
                actual = int(info.request_time)
            elif lhs in (
                "execution_time_ms",
                "execution_duration",
                "attributes.execution_time_ms",
            ):
                actual = int(info.execution_duration or 0)
            elif lhs in ("run_id", "trace.run_id", "metadata.mlflow.sourceRun"):
                if op == "=":
                    if str(rhs_val) not in run_ids and info.trace_metadata.get(
                        "mlflow.sourceRun"
                    ) != str(rhs_val):
                        return False
                    continue
                actual = info.trace_metadata.get("mlflow.sourceRun", "")
            elif lhs.startswith("tag.") or lhs.startswith("tags."):
                tag_key = lhs.split(".", 1)[1].strip("`'\"")
                actual = info.tags.get(tag_key)
            elif lhs.startswith("metadata.") or lhs.startswith("request_metadata."):
                meta_key = lhs.split(".", 1)[1].strip("`'\"")
                actual = info.trace_metadata.get(meta_key)
            else:
                raise MlflowException.invalid_parameter_value(
                    f"Unsupported trace filter field: {lhs!r}"
                )

            if not RiffDBExtendedStoreMixin._compare_filter_value(actual, op, rhs_val):
                return False
        return True

    @staticmethod
    def _compare_filter_value(actual: Any, op: str, expected: Any) -> bool:
        if actual is None:
            return op == "!="
        if op == "=":
            return bool(actual == expected or str(actual) == str(expected))
        if op == "!=":
            return bool(actual != expected and str(actual) != str(expected))
        if op in (">", ">=", "<", "<="):
            try:
                lhs_num = float(actual)
                rhs_num = float(expected)
            except (TypeError, ValueError):
                return False
            if op == ">":
                return lhs_num > rhs_num
            if op == ">=":
                return lhs_num >= rhs_num
            if op == "<":
                return lhs_num < rhs_num
            return lhs_num <= rhs_num
        if op in ("LIKE", "ILIKE"):
            pattern = "^" + re.escape(str(expected)).replace(r"%", ".*").replace(r"_", ".") + "$"
            flags = re.IGNORECASE if op == "ILIKE" else 0
            return re.match(pattern, str(actual), flags=flags) is not None
        return False

    def search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 100,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,
        locations: list[str] | None = None,
        sql_warehouse_id: str | None = None,
        include_spans: bool = True,
        run_id: str | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        self._ensure_extended_indexes()
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException.invalid_parameter_value("max_results must be a positive integer")
        target_experiments = list(experiment_ids or locations or [])
        client = self._application()  # type: ignore[attr-defined]
        matched: list[tuple[TraceInfo, Any]] = []
        for exp_id_str in target_experiments:
            exp_id = int(self._numeric_experiment_id(exp_id_str))  # type: ignore[attr-defined]
            cursor: str | None = None
            while True:
                res = client.search_traces(
                    SearchTracesParams(experiment_id=exp_id, limit=50, after=cursor)
                )
                if not isinstance(res.value, SearchTracesFound):
                    raise MlflowException("Invalid SearchTraces response from RiffDB")
                for row in res.value.traces:
                    self._trace_experiment_map[row.trace_id] = exp_id
                    full = client.get_trace(
                        GetTraceParams(experiment_id=exp_id, trace_id=row.trace_id)
                    )
                    assessments = (
                        full.value.assessments if isinstance(full.value, GetTraceFound) else ()
                    )
                    info = self._to_mlflow_trace_info(row, assessments)
                    if run_id is not None:
                        run_ids = self._decode_json_list(row.packed_run_ids)
                        if (
                            run_id not in run_ids
                            and info.trace_metadata.get("mlflow.sourceRun") != run_id
                        ):
                            continue
                    if self._matches_trace_filter(info, row, filter_string):
                        matched.append((info, row))
                cursor = res.next_cursor
                if not cursor or not res.value.traces:
                    break

        if order_by:
            for spec in reversed(order_by):
                parts = spec.strip().split()
                field = parts[0].strip("`'\"")
                desc = len(parts) > 1 and parts[1].upper() == "DESC"
                if field in ("timestamp_ms", "request_time", "trace.timestamp_ms"):
                    matched.sort(key=lambda item: int(item[0].request_time), reverse=desc)
                elif field in ("execution_time_ms", "execution_duration"):
                    matched.sort(
                        key=lambda item: int(item[0].execution_duration or 0),
                        reverse=desc,
                    )
                else:
                    matched.sort(key=lambda item: item[0].trace_id, reverse=desc)
        else:
            matched.sort(key=lambda item: (-int(item[0].request_time), item[0].trace_id))

        offset = int(page_token) if page_token is not None else 0
        slice_items = [item[0] for item in matched[offset : offset + max_results]]
        next_token = str(offset + max_results) if offset + max_results < len(matched) else None
        return slice_items, next_token

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        self._ensure_extended_indexes()
        if trace_ids is None and max_timestamp_millis is None:
            raise MlflowException.invalid_parameter_value(
                "Either trace_ids or max_timestamp_millis must be specified"
            )
        if trace_ids is not None and max_timestamp_millis is not None:
            raise MlflowException.invalid_parameter_value(
                "Cannot specify both trace_ids and max_timestamp_millis"
            )
        if max_traces is not None and max_traces < 1:
            raise MlflowException.invalid_parameter_value("max_traces must be a positive integer")

        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        client = self._application()  # type: ignore[attr-defined]
        target_ids: list[str] = []

        if trace_ids is not None:
            target_ids = list(trace_ids)
        else:
            cursor: str | None = None
            candidates: list[tuple[int, str]] = []
            while True:
                res = client.search_traces(
                    SearchTracesParams(experiment_id=exp_id, limit=50, after=cursor)
                )
                if not isinstance(res.value, SearchTracesFound):
                    break
                for row in res.value.traces:
                    if int(row.request_time_ms) <= int(max_timestamp_millis or 0):
                        candidates.append((int(row.request_time_ms), row.trace_id))
                cursor = res.next_cursor
                if not cursor or not res.value.traces:
                    break
            candidates.sort()
            if max_traces is not None:
                candidates = candidates[:max_traces]
            target_ids = [tid for _, tid in candidates]

        deleted_count = 0
        for tid in target_ids:
            full = client.get_trace(GetTraceParams(experiment_id=exp_id, trace_id=tid))
            if not isinstance(full.value, GetTraceFound):
                continue
            for span_row in full.value.spans:
                client.delete_trace_span(
                    DeleteTraceSpanInput(
                        request_id=str(uuid4()),
                        experiment_id=exp_id,
                        trace_id=tid,
                        span_id=span_row.span_id,
                    )
                )
            for assess_row in full.value.assessments:
                client.delete_trace_assessment(
                    DeleteTraceAssessmentInput(
                        request_id=str(uuid4()),
                        experiment_id=exp_id,
                        trace_id=tid,
                        assessment_id=assess_row.assessment_id,
                    )
                )
            del_res = client.delete_trace(
                DeleteTraceInput(
                    request_id=str(uuid4()),
                    experiment_id=exp_id,
                    trace_id=tid,
                )
            )
            if isinstance(del_res.outcome, DeleteTraceTraceDeleted):
                deleted_count += 1
                self._trace_experiment_map.pop(tid, None)
            elif not isinstance(del_res.outcome, DeleteTraceTraceMissing):
                raise MlflowException("Invalid DeleteTrace response from RiffDB")
        return deleted_count

    def _mutate_trace_row(
        self,
        trace_id: str,
        mutator: Any,
    ) -> Any:
        experiment_id = self._resolve_trace_experiment_id(trace_id)
        client = self._application()  # type: ignore[attr-defined]
        res = client.get_trace(GetTraceParams(experiment_id=experiment_id, trace_id=trace_id))
        if isinstance(res.value, GetTraceMissing) or not isinstance(res.value, GetTraceFound):
            raise MlflowException(
                f"Trace with ID '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.trace
        metadata = self._decode_json_dict(row.packed_trace_metadata)
        tags = self._decode_json_dict(row.packed_tags)
        run_ids = self._decode_json_list(row.packed_run_ids)
        prompt_versions = self._decode_json_list(row.packed_prompt_versions)
        state = {
            "trace_name": row.trace_name,
            "source_run_id": row.source_run_id,
            "metadata": metadata,
            "tags": tags,
            "run_ids": run_ids,
            "prompt_versions": prompt_versions,
        }
        mutator(state)
        up_res = client.upsert_trace(
            UpsertTraceInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                trace_id=trace_id,
                request_time_ms=int(row.request_time_ms),
                execution_duration_ms=int(row.execution_duration_ms),
                state=row.state,
                status_code=row.status_code,
                trace_name=str(state["trace_name"])[:256],
                client_request_id=row.client_request_id,
                request_preview=row.request_preview,
                response_preview=row.response_preview,
                source_run_id=state["source_run_id"],
                span_count=int(row.span_count),
                error_span_count=int(row.error_span_count),
                total_input_tokens=int(row.total_input_tokens),
                total_output_tokens=int(row.total_output_tokens),
                total_tokens=int(row.total_tokens),
                cache_read_input_tokens=int(row.cache_read_input_tokens),
                cache_creation_input_tokens=int(row.cache_creation_input_tokens),
                total_cost_micros=int(row.total_cost_micros),
                session_id=row.session_id,
                packed_trace_metadata=self._encode_json_dict(state["metadata"]),
                packed_tags=self._encode_json_dict(state["tags"]),
                packed_run_ids=self._encode_json_list(state["run_ids"]),
                packed_prompt_versions=self._encode_json_list(state["prompt_versions"]),
                requested_at=row.requested_at,
                initialized=bool(row.initialized),
            )
        )
        if not isinstance(up_res.outcome, UpsertTraceTraceUpserted):
            raise MlflowException("Invalid UpsertTrace response from RiffDB")
        return up_res.outcome.trace

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        if not isinstance(key, str) or not key:
            raise MlflowException.invalid_parameter_value("Tag key must be a non-empty string")
        str_value = str(value)

        def _apply(state: dict[str, Any]) -> None:
            state["tags"][key] = str_value
            if key == "mlflow.traceName":
                state["trace_name"] = str_value

        self._mutate_trace_row(trace_id, _apply)

    def delete_trace_tag(self, trace_id: str, key: str) -> None:
        def _apply(state: dict[str, Any]) -> None:
            state["tags"].pop(key, None)
            if key == "mlflow.traceName":
                state["trace_name"] = ""

        self._mutate_trace_row(trace_id, _apply)

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        normalized_run_id, _ = self._run_experiment_id(run_id)  # type: ignore[attr-defined]
        for trace_id in trace_ids or []:

            def _apply(state: dict[str, Any]) -> None:
                if normalized_run_id not in state["run_ids"]:
                    state["run_ids"].append(normalized_run_id)
                if not state["source_run_id"]:
                    state["source_run_id"] = normalized_run_id
                state["metadata"]["mlflow.sourceRun"] = normalized_run_id

            self._mutate_trace_row(trace_id, _apply)

    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        normalized_run_id, _ = self._run_experiment_id(run_id)  # type: ignore[attr-defined]
        for trace_id in trace_ids or []:

            def _apply(state: dict[str, Any]) -> None:
                state["run_ids"] = [r for r in state["run_ids"] if r != normalized_run_id]
                if state["source_run_id"] == normalized_run_id:
                    state["source_run_id"] = state["run_ids"][0] if state["run_ids"] else None
                    if state["source_run_id"]:
                        state["metadata"]["mlflow.sourceRun"] = state["source_run_id"]
                    else:
                        state["metadata"].pop("mlflow.sourceRun", None)

            self._mutate_trace_row(trace_id, _apply)

    def link_prompts_to_trace(self, trace_id: Any, prompt_versions: Any) -> None:
        # Support both (trace_id, prompt_versions) and legacy (prompt_versions, trace_id)
        if isinstance(trace_id, str) and not isinstance(prompt_versions, str):
            resolved_trace_id = trace_id
            resolved_prompts = prompt_versions
        else:
            resolved_trace_id = str(prompt_versions)
            resolved_prompts = trace_id

        serialized: list[dict[str, Any]] = []
        for pv in resolved_prompts or []:
            if hasattr(pv, "name") and hasattr(pv, "version"):
                serialized.append({"name": pv.name, "version": str(pv.version)})
            elif isinstance(pv, dict):
                serialized.append({"name": str(pv["name"]), "version": str(pv["version"])})
            else:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid prompt version entry: {pv!r}"
                )

        def _apply(state: dict[str, Any]) -> None:
            existing = state["prompt_versions"]
            for entry in serialized:
                if entry not in existing:
                    existing.append(entry)
            state["tags"]["mlflow.linkedPrompts"] = json.dumps(existing)

        self._mutate_trace_row(resolved_trace_id, _apply)

    # ------------------------------------------------------------------
    # Assessments (Feedback & Expectation)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_assessment_value_fields(
        assessment: Assessment,
    ) -> tuple[str, str, int | None]:
        exp_obj = getattr(assessment, "expectation", None)
        fb_obj = getattr(assessment, "feedback", None)
        if isinstance(assessment, Expectation) or exp_obj is not None:
            assessment_type = "EXPECTATION"
            raw_val = exp_obj.value if exp_obj is not None else getattr(assessment, "value", None)
        else:
            assessment_type = "FEEDBACK"
            raw_val = fb_obj.value if fb_obj is not None else getattr(assessment, "value", None)

        value_text = json.dumps(raw_val)
        numeric_milli: int | None = None
        if isinstance(raw_val, bool):
            numeric_milli = 1000 if raw_val else 0
        elif isinstance(raw_val, int | float) and not math.isnan(float(raw_val)):
            numeric_milli = round(float(raw_val) * 1000.0)
        elif isinstance(raw_val, str) and raw_val in ("yes", "no"):
            numeric_milli = 1000 if raw_val == "yes" else 0
        return assessment_type, value_text[:2000], numeric_milli

    def create_assessment(self, assessment: Assessment) -> Assessment:
        if not isinstance(assessment, Assessment):
            raise MlflowException.invalid_parameter_value(
                "assessment must be an Assessment instance"
            )
        trace_id = assessment.trace_id
        if not trace_id:
            raise MlflowException.invalid_parameter_value("Assessment must specify a trace_id")
        experiment_id = self._resolve_trace_experiment_id(trace_id)
        now_ms = _now_ms()
        assessment_id = assessment.assessment_id or f"a-{uuid4().hex[:16]}"
        create_ms = int(assessment.create_time_ms or now_ms)
        update_ms = int(assessment.last_update_time_ms or create_ms)
        assessment.assessment_id = assessment_id
        assessment.create_time_ms = create_ms
        assessment.last_update_time_ms = update_ms

        assessment_type, value_text, numeric_milli = self._extract_assessment_value_fields(
            assessment
        )
        packed = json.dumps(
            assessment.to_dictionary(),  # type: ignore[no-untyped-call]
            separators=(",", ":"),
        ).encode("utf-8")
        client = self._application()  # type: ignore[attr-defined]
        res = client.upsert_trace_assessment(
            UpsertTraceAssessmentInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                trace_id=trace_id,
                assessment_id=assessment_id,
                name=assessment.name[:250],
                assessment_type=assessment_type,
                value_text=value_text,
                numeric_value_milli=numeric_milli,
                created_timestamp_ms=create_ms,
                last_updated_timestamp_ms=update_ms,
                span_id=assessment.span_id,
                packed_assessment_json=packed,
            )
        )
        if isinstance(res.outcome, UpsertTraceAssessmentTraceMissing):
            raise MlflowException(
                f"Trace with ID '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, UpsertTraceAssessmentTraceAssessmentUpserted):
            raise MlflowException("Invalid UpsertTraceAssessment response from RiffDB")
        return self._assessment_from_row(res.outcome.assessment)

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        experiment_id = self._resolve_trace_experiment_id(trace_id)
        client = self._application()  # type: ignore[attr-defined]
        res = client.get_trace(GetTraceParams(experiment_id=experiment_id, trace_id=trace_id))
        if isinstance(res.value, GetTraceMissing) or not isinstance(res.value, GetTraceFound):
            raise MlflowException(
                f"Trace with ID '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        for row in res.value.assessments:
            if row.assessment_id == assessment_id:
                return self._assessment_from_row(row)
        raise MlflowException(
            f"Assessment with ID '{assessment_id}' not found on trace '{trace_id}'.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: str | None = None,
        expectation: Any = None,
        feedback: Any = None,
        rationale: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Assessment:
        if expectation is not None and feedback is not None:
            raise MlflowException.invalid_parameter_value(
                "Cannot specify both `expectation` and `feedback` parameters."
            )
        existing = self.get_assessment(trace_id, assessment_id)
        if expectation is not None and not isinstance(existing, Expectation):
            raise MlflowException.invalid_parameter_value(
                "Cannot update expectation value on a Feedback assessment."
            )
        if feedback is not None and not isinstance(existing, Feedback):
            raise MlflowException.invalid_parameter_value(
                "Cannot update feedback value on an Expectation assessment."
            )

        merged_metadata: dict[str, str] | None = None
        if existing.metadata or metadata:
            merged_metadata = dict(existing.metadata or {})
            if metadata:
                merged_metadata.update(metadata)

        updated_ms = _now_ms()
        create_ms = int(existing.create_time_ms or updated_ms)
        if isinstance(existing, Expectation):
            new_val = (
                expectation.value if expectation is not None else getattr(existing, "value", None)
            )
            updated: Assessment = Expectation(
                name=name if name is not None else existing.name,
                value=new_val,
                source=existing.source,
                trace_id=trace_id,
                metadata=merged_metadata,
                span_id=existing.span_id,
                create_time_ms=create_ms,
                last_update_time_ms=updated_ms,
            )
        else:
            existing_fb = cast(Feedback, existing)
            new_val = feedback.value if feedback is not None else existing_fb.value
            new_err = feedback.error if feedback is not None else existing_fb.error
            updated = Feedback(
                name=name if name is not None else existing_fb.name,
                value=new_val,
                error=new_err,
                source=existing_fb.source,
                trace_id=trace_id,
                metadata=merged_metadata,
                span_id=existing_fb.span_id,
                create_time_ms=create_ms,
                last_update_time_ms=updated_ms,
                rationale=rationale if rationale is not None else existing_fb.rationale,
            )
        updated.assessment_id = existing.assessment_id
        updated.valid = existing.valid
        updated.overrides = existing.overrides

        experiment_id = self._resolve_trace_experiment_id(trace_id)
        assessment_type, value_text, numeric_milli = self._extract_assessment_value_fields(updated)
        packed = json.dumps(
            updated.to_dictionary(),  # type: ignore[no-untyped-call]
            separators=(",", ":"),
        ).encode("utf-8")
        client = self._application()  # type: ignore[attr-defined]
        res = client.upsert_trace_assessment(
            UpsertTraceAssessmentInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                trace_id=trace_id,
                assessment_id=assessment_id,
                name=updated.name[:250],
                assessment_type=assessment_type,
                value_text=value_text,
                numeric_value_milli=numeric_milli,
                created_timestamp_ms=create_ms,
                last_updated_timestamp_ms=updated_ms,
                span_id=updated.span_id,
                packed_assessment_json=packed,
            )
        )
        if not isinstance(res.outcome, UpsertTraceAssessmentTraceAssessmentUpserted):
            raise MlflowException("Invalid UpsertTraceAssessment response from RiffDB")
        return self._assessment_from_row(res.outcome.assessment)

    def delete_assessment(self, trace_id: str, assessment_id: str) -> None:
        experiment_id = self._resolve_trace_experiment_id(trace_id)
        client = self._application()  # type: ignore[attr-defined]
        res = client.delete_trace_assessment(
            DeleteTraceAssessmentInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                trace_id=trace_id,
                assessment_id=assessment_id,
            )
        )
        if isinstance(res.outcome, DeleteTraceAssessmentTraceAssessmentMissing):
            raise MlflowException(
                f"Assessment with ID '{assessment_id}' not found on trace '{trace_id}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteTraceAssessmentTraceAssessmentDeleted):
            raise MlflowException("Invalid DeleteTraceAssessment response from RiffDB")

    # ------------------------------------------------------------------
    # ADR-0315 Trace Metrics & Analytical Rollups
    # ------------------------------------------------------------------

    def query_trace_metrics(
        self,
        experiment_ids: list[str],
        view_type: MetricViewType,
        metric_name: str,
        aggregations: list[MetricAggregation],
        dimensions: list[str] | None = None,
        filters: list[str] | None = None,
        time_interval_seconds: int | None = None,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        max_results: int = 1000,
        page_token: str | None = None,
    ) -> PagedList[Any]:
        _validate_trace_metrics_params(view_type, metric_name, aggregations, dimensions)
        if time_interval_seconds and (start_time_ms is None or end_time_ms is None):
            raise MlflowException.invalid_parameter_value(
                "start_time_ms and end_time_ms are required if time_interval_seconds is set"
            )
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException.invalid_parameter_value("max_results must be a positive integer")

        client = self._application()  # type: ignore[attr-defined]
        commit_seq = self._causal_commit_seq(client)
        dims_list = list(dimensions or [])
        bucket_size_ms = (
            int(time_interval_seconds) * 1000
            if time_interval_seconds is not None and time_interval_seconds > 0
            else None
        )

        has_trace_filter = any(
            not clause.strip().lower().startswith(("span.", "assessment."))
            for f in (filters or [])
            for clause in re.split(r"\bAND\b", f, flags=re.IGNORECASE)
            if clause.strip()
        )
        need_trace_rows = (
            view_type == MetricViewType.TRACES
            or has_trace_filter
            or start_time_ms is not None
            or end_time_ms is not None
        )

        records: list[dict[str, Any]] = []
        for exp_id_str in experiment_ids:
            exp_id = int(self._numeric_experiment_id(exp_id_str))  # type: ignore[attr-defined]
            trace_rows_by_id: dict[str, Any] = {}
            if need_trace_rows:
                t_cursor: str | None = None
                while True:
                    t_res = client.query_trace_analytics_after_commit(
                        QueryTraceAnalyticsParams(experiment_id=exp_id, limit=500, after=t_cursor),
                        commit_seq,
                    )
                    if not isinstance(t_res.value, QueryTraceAnalyticsFound):
                        raise MlflowException("Invalid QueryTraceAnalytics response from RiffDB")
                    for trow in t_res.value.traces:
                        if start_time_ms is not None and int(trow.request_time_ms) < int(
                            start_time_ms
                        ):
                            continue
                        if end_time_ms is not None and int(trow.request_time_ms) > int(end_time_ms):
                            continue
                        if not self._matches_projected_filters(trow, None, None, filters):
                            continue
                        trace_rows_by_id[trow.trace_id] = trow
                    t_cursor = t_res.next_cursor
                    if not t_cursor or not t_res.value.traces:
                        break

            if view_type == MetricViewType.TRACES:
                for trow in trace_rows_by_id.values():
                    val = self._extract_trace_metric_value(trow, metric_name)
                    if val is None:
                        continue
                    dim_map: dict[str, Any] = {}
                    if bucket_size_ms is not None:
                        b_ms = (int(trow.request_time_ms) // bucket_size_ms) * bucket_size_ms
                        dim_map[TIME_BUCKET_LABEL] = b_ms
                    skip = False
                    for d in dims_list:
                        if d == TraceMetricDimensionKey.TRACE_NAME:
                            d_val = trow.trace_name or None
                        elif d == TraceMetricDimensionKey.TRACE_STATUS:
                            d_val = trow.state or None
                        else:
                            d_val = None
                        if d_val is None:
                            skip = True
                            break
                        dim_map[d] = d_val
                    if not skip:
                        records.append({"dims": dim_map, "val": val})

            elif view_type == MetricViewType.SPANS:
                s_cursor: str | None = None
                while True:
                    s_res = client.query_span_analytics_after_commit(
                        QuerySpanAnalyticsParams(experiment_id=exp_id, limit=500, after=s_cursor),
                        commit_seq,
                    )
                    if not isinstance(s_res.value, QuerySpanAnalyticsFound):
                        raise MlflowException("Invalid QuerySpanAnalytics response from RiffDB")
                    for srow in s_res.value.spans:
                        parent_trace = trace_rows_by_id.get(srow.trace_id)
                        if need_trace_rows and parent_trace is None:
                            continue
                        if not self._matches_projected_filters(parent_trace, srow, None, filters):
                            continue
                        val = self._extract_span_metric_value(srow, metric_name)
                        if val is None:
                            continue
                        dim_map = {}
                        if bucket_size_ms is not None:
                            span_ms = int(srow.start_time_ns) // 1_000_000
                            b_ms = (span_ms // bucket_size_ms) * bucket_size_ms
                            dim_map[TIME_BUCKET_LABEL] = b_ms
                        skip = False
                        for d in dims_list:
                            if d == SpanMetricDimensionKey.SPAN_NAME:
                                d_val = srow.name or None
                            elif d == SpanMetricDimensionKey.SPAN_TYPE:
                                d_val = srow.span_type or None
                            elif d == SpanMetricDimensionKey.SPAN_STATUS:
                                d_val = srow.status or None
                            elif d == SpanMetricDimensionKey.SPAN_MODEL_NAME:
                                d_val = srow.model_name or None
                            elif d == SpanMetricDimensionKey.SPAN_MODEL_PROVIDER:
                                d_val = srow.model_provider or None
                            else:
                                d_val = None
                            if d_val is None:
                                skip = True
                                break
                            dim_map[d] = d_val
                        if not skip:
                            records.append({"dims": dim_map, "val": val})
                    s_cursor = s_res.next_cursor
                    if not s_cursor or not s_res.value.spans:
                        break

            elif view_type == MetricViewType.ASSESSMENTS:
                a_cursor: str | None = None
                while True:
                    a_res = client.query_assessment_analytics_after_commit(
                        QueryAssessmentAnalyticsParams(
                            experiment_id=exp_id, limit=500, after=a_cursor
                        ),
                        commit_seq,
                    )
                    if not isinstance(a_res.value, QueryAssessmentAnalyticsFound):
                        raise MlflowException(
                            "Invalid QueryAssessmentAnalytics response from RiffDB"
                        )
                    for arow in a_res.value.assessments:
                        parent_trace = trace_rows_by_id.get(arow.trace_id)
                        if need_trace_rows and parent_trace is None:
                            continue
                        if not self._matches_projected_filters(parent_trace, None, arow, filters):
                            continue
                        val = self._extract_assessment_metric_value(arow, metric_name)
                        if val is None:
                            continue
                        dim_map = {}
                        if bucket_size_ms is not None:
                            b_ms = (
                                int(arow.created_timestamp_ms) // bucket_size_ms
                            ) * bucket_size_ms
                            dim_map[TIME_BUCKET_LABEL] = b_ms
                        skip = False
                        for d in dims_list:
                            if d == AssessmentMetricDimensionKey.ASSESSMENT_NAME:
                                d_val = arow.name or None
                            elif d == AssessmentMetricDimensionKey.ASSESSMENT_VALUE:
                                d_val = arow.value_text or None
                            else:
                                d_val = None
                            if d_val is None:
                                skip = True
                                break
                            dim_map[d] = d_val
                        if not skip:
                            records.append({"dims": dim_map, "val": val})
                    a_cursor = a_res.next_cursor
                    if not a_cursor or not a_res.value.assessments:
                        break

        dim_order = ([TIME_BUCKET_LABEL] if bucket_size_ms is not None else []) + dims_list
        groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
        for rec in records:
            key = tuple(rec["dims"][k] for k in dim_order)
            groups[key].append(rec["val"])

        if (
            not groups
            and not dim_order
            and any(a.aggregation_type == AggregationType.COUNT for a in aggregations)
        ):
            groups[()] = []

        data_points: list[MetricDataPoint] = []
        for key in sorted(groups.keys()):
            raw_vals = groups[key]
            dims_out: dict[str, Any] = {}
            for idx, col_name in enumerate(dim_order):
                if col_name == TIME_BUCKET_LABEL:
                    dt = datetime.fromtimestamp(float(key[idx]) / 1000.0, tz=timezone.utc)
                    dims_out[TIME_BUCKET_LABEL] = dt.isoformat()
                else:
                    dims_out[col_name] = key[idx]

            values_out: dict[str, float] = {}
            for agg in aggregations:
                agg_label = str(agg)
                if agg.aggregation_type == AggregationType.COUNT:
                    if metric_name == TraceMetricKey.SESSION_COUNT:
                        values_out[agg_label] = float(len({v for v in raw_vals if v}))
                    else:
                        values_out[agg_label] = float(len(raw_vals))
                else:
                    num_vals = sorted(float(v) for v in raw_vals if v is not None)
                    if not num_vals:
                        continue
                    if agg.aggregation_type == AggregationType.SUM:
                        values_out[agg_label] = float(sum(num_vals))
                    elif agg.aggregation_type == AggregationType.AVG:
                        values_out[agg_label] = float(sum(num_vals) / len(num_vals))
                    elif agg.aggregation_type == AggregationType.MIN:
                        values_out[agg_label] = float(num_vals[0])
                    elif agg.aggregation_type == AggregationType.MAX:
                        values_out[agg_label] = float(num_vals[-1])
                    elif agg.aggregation_type == AggregationType.PERCENTILE:
                        pct = _linear_percentile(num_vals, float(agg.percentile_value or 50.0))
                        if pct is not None:
                            values_out[agg_label] = pct

            if values_out:
                data_points.append(
                    MetricDataPoint(
                        metric_name=metric_name,
                        dimensions=dims_out,
                        values=values_out,
                    )
                )

        offset = int(page_token) if page_token is not None else 0
        sliced = data_points[offset : offset + max_results]
        next_tok = str(offset + max_results) if offset + max_results < len(data_points) else None
        return PagedList(sliced, next_tok)

    @staticmethod
    def _extract_trace_metric_value(trow: Any, metric_name: str) -> Any:
        if metric_name == TraceMetricKey.TRACE_COUNT:
            return trow.trace_id
        if metric_name == TraceMetricKey.SESSION_COUNT:
            return trow.session_id
        if metric_name == TraceMetricKey.LATENCY:
            return float(trow.execution_duration_ms)
        if metric_name == TraceMetricKey.INPUT_TOKENS:
            return (
                float(trow.total_input_tokens)
                if trow.total_tokens > 0 or trow.total_input_tokens > 0
                else None
            )
        if metric_name == TraceMetricKey.OUTPUT_TOKENS:
            return (
                float(trow.total_output_tokens)
                if trow.total_tokens > 0 or trow.total_output_tokens > 0
                else None
            )
        if metric_name == TraceMetricKey.TOTAL_TOKENS:
            return float(trow.total_tokens) if trow.total_tokens > 0 else None
        if metric_name == TraceMetricKey.CACHE_READ_INPUT_TOKENS:
            return float(trow.cache_read_input_tokens) if trow.cache_read_input_tokens > 0 else None
        if metric_name == TraceMetricKey.CACHE_CREATION_INPUT_TOKENS:
            return (
                float(trow.cache_creation_input_tokens)
                if trow.cache_creation_input_tokens > 0
                else None
            )
        return None

    @staticmethod
    def _extract_span_metric_value(srow: Any, metric_name: str) -> Any:
        if metric_name == SpanMetricKey.SPAN_COUNT:
            return srow.span_id
        if metric_name == SpanMetricKey.LATENCY:
            return float(srow.duration_ms)
        if metric_name == SpanMetricKey.INPUT_COST:
            return (
                float(srow.input_cost_micros) / 1_000_000.0
                if srow.total_cost_micros > 0 or srow.input_cost_micros > 0
                else None
            )
        if metric_name == SpanMetricKey.OUTPUT_COST:
            return (
                float(srow.output_cost_micros) / 1_000_000.0
                if srow.total_cost_micros > 0 or srow.output_cost_micros > 0
                else None
            )
        if metric_name == SpanMetricKey.TOTAL_COST:
            return (
                float(srow.total_cost_micros) / 1_000_000.0 if srow.total_cost_micros > 0 else None
            )
        return None

    @staticmethod
    def _extract_assessment_metric_value(arow: Any, metric_name: str) -> Any:
        if metric_name == AssessmentMetricKey.ASSESSMENT_COUNT:
            return arow.assessment_id
        if metric_name == AssessmentMetricKey.ASSESSMENT_VALUE:
            if arow.numeric_value_milli is None:
                return None
            return float(arow.numeric_value_milli) / 1000.0
        return None

    @classmethod
    def _matches_projected_filters(
        cls,
        trow: Any,
        srow: Any | None,
        arow: Any | None,
        filters: list[str] | None,
    ) -> bool:
        for filter_str in filters or []:
            clauses = [
                c.strip()
                for c in re.split(r"\bAND\b", filter_str, flags=re.IGNORECASE)
                if c.strip()
            ]
            for clause in clauses:
                m = _FILTER_CLAUSE_RE.match(clause)
                if not m:
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid filter clause: {clause!r}"
                    )
                lhs = m.group(1).strip("`'\"").lower()
                op = m.group(2).upper()
                rhs_raw = m.group(3).strip().strip("'\"")
                if lhs.startswith("span."):
                    if srow is None:
                        continue
                    field = lhs.split(".", 1)[1]
                    actual = {
                        "name": srow.name,
                        "type": srow.span_type,
                        "span_type": srow.span_type,
                        "status": srow.status,
                        "model_name": srow.model_name,
                        "model_provider": srow.model_provider,
                    }.get(field)
                    if not cls._compare_filter_value(actual, op, rhs_raw):
                        return False
                elif lhs.startswith("assessment."):
                    if arow is None:
                        continue
                    field = lhs.split(".", 1)[1]
                    actual = {
                        "name": arow.name,
                        "assessment_name": arow.name,
                        "type": arow.assessment_type,
                        "assessment_type": arow.assessment_type,
                        "value": arow.value_text,
                    }.get(field)
                    if not cls._compare_filter_value(actual, op, rhs_raw):
                        return False
                else:
                    field = lhs.split(".", 1)[1] if lhs.startswith("trace.") else lhs
                    actual = {
                        "status": trow.state,
                        "name": trow.trace_name,
                        "trace_name": trow.trace_name,
                        "session_id": trow.session_id,
                    }.get(field)
                    if actual is not None and not cls._compare_filter_value(actual, op, rhs_raw):
                        return False
        return True

    def get_span_model_rollups(
        self,
        experiment_id: str,
        span_type: str = "LLM",
        model_name: str = "gpt-4o",
        status: str = "OK",
        start_time_ms: int = 0,
        end_time_ms: int = 4_102_444_800_000,
    ) -> list[dict[str, Any]]:
        """Queries the ADR-0315 `SpanModelRollup` event projection."""
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        client = self._application()  # type: ignore[attr-defined]
        commit_seq = self._causal_commit_seq(client)
        res = client.query_span_model_rollup_after_commit(
            QuerySpanModelRollupParams(
                experiment_id=exp_id,
                span_type=span_type,
                model_name=model_name,
                status=status,
                start_bucket=_ts_from_ms(start_time_ms),
                end_bucket=_ts_from_ms(end_time_ms),
            ),
            commit_seq,
        )
        if not isinstance(res.value, QuerySpanModelRollupFound):
            raise MlflowException("Invalid QuerySpanModelRollup response from RiffDB")
        return [
            {
                "experiment_id": int(r.experiment_id),
                "span_type": r.span_type,
                "model_name": r.model_name,
                "status": r.status,
                "observed_at_ms": _ts_to_ms(r.observed_at),
                "span_count": int(r.span_count),
                "total_duration_ms": int(r.total_duration_ms),
                "min_duration_ms": int(r.min_duration_ms),
                "max_duration_ms": int(r.max_duration_ms),
                "avg_duration_ms": float(r.avg_duration_ms),
                "sum_input_tokens": int(r.sum_input_tokens),
                "sum_output_tokens": int(r.sum_output_tokens),
                "sum_total_tokens": int(r.sum_total_tokens),
                "sum_cost_micros": int(r.sum_cost_micros),
            }
            for r in res.value.rollups
        ]

    def get_columnar_span_model_rollups(
        self,
        experiment_id: str,
        start_time_ms: int = 0,
        end_time_ms: int = 4_102_444_800_000,
    ) -> list[dict[str, Any]]:
        """Queries the ADR-0315 `SpanAnalytics` columnar projection with p50/p95/p99."""
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        client = self._application()  # type: ignore[attr-defined]
        commit_seq = self._causal_commit_seq(client)
        res = client.aggregate_span_analytics_by_model_after_commit(
            AggregateSpanAnalyticsByModelParams(
                experiment_id=exp_id,
                start_at=_ts_from_ms(start_time_ms),
                end_at=_ts_from_ms(end_time_ms),
            ),
            commit_seq,
        )
        if not isinstance(res.value, AggregateSpanAnalyticsByModelFound):
            raise MlflowException("Invalid AggregateSpanAnalyticsByModel response from RiffDB")
        return [
            {
                "model_name": r.model_name,
                "span_type": r.span_type,
                "status": r.status,
                "hour_bucket_ms": _ts_to_ms(r.hour_bucket),
                "span_count": int(r.span_count),
                "sum_duration_ms": int(r.sum_duration_ms),
                "avg_duration_ms": (
                    float(r.sum_duration_ms) / int(r.span_count) if int(r.span_count) > 0 else 0.0
                ),
                "min_duration_ms": int(r.min_duration_ms or 0),
                "max_duration_ms": int(r.max_duration_ms or 0),
                "p50_duration_ms": int(r.p50_duration_ms or 0),
                "p95_duration_ms": int(r.p95_duration_ms or 0),
                "p99_duration_ms": int(r.p99_duration_ms or 0),
                "sum_total_tokens": int(r.sum_total_tokens),
                "sum_total_cost_micros": int(r.sum_total_cost_micros),
            }
            for r in res.value.by_model
        ]

    # ------------------------------------------------------------------
    # Logged Models (MLflow 3.x)
    # ------------------------------------------------------------------

    def _new_model_id(self, experiment_id: int) -> str:
        return f"m-{experiment_id:016x}{secrets.token_hex(8)}"

    def _resolve_model_experiment_id(self, model_id: str) -> int:
        self._ensure_extended_indexes()
        if not isinstance(model_id, str) or not model_id or len(model_id.encode("utf-8")) > 64:
            raise MlflowException.invalid_parameter_value(f"Invalid model_id: {model_id!r}")
        cached = self._model_experiment_map.get(model_id)
        if cached is not None:
            return cached
        matched = _MODEL_ID_EXPERIMENT_RE.match(model_id)
        if matched is not None:
            exp_id = int(matched.group(1), 16)
            self._model_experiment_map[model_id] = exp_id
            return exp_id
        client = self._application()  # type: ignore[attr-defined]
        for exp in self.search_experiments(ViewType.ALL):  # type: ignore[no-untyped-call]
            exp_id = int(exp.experiment_id)
            res = client.get_logged_model(
                GetLoggedModelParams(experiment_id=exp_id, model_id=model_id)
            )
            if isinstance(res.value, GetLoggedModelFound):
                self._model_experiment_map[model_id] = exp_id
                return exp_id
        raise MlflowException(
            f"LoggedModel with ID '{model_id}' does not exist.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    @classmethod
    def _to_mlflow_logged_model(cls, row: Any) -> LoggedModel:
        tags_dict = cls._decode_json_dict(row.packed_tags)
        params_dict = cls._decode_json_dict(row.packed_params)
        metrics_list = cls._decode_json_list(row.packed_metrics)
        try:
            status = LoggedModelStatus(row.status)
        except ValueError:
            status = LoggedModelStatus.UNSPECIFIED
        return LoggedModel(
            experiment_id=str(row.experiment_id),
            model_id=row.model_id,
            name=row.name,
            artifact_location=row.artifact_location,
            creation_timestamp=int(row.creation_timestamp_ms),
            last_updated_timestamp=int(row.last_updated_timestamp_ms),
            model_type=row.model_type,
            source_run_id=row.source_run_id,
            status=status,
            status_message=row.status_message,
            tags=[
                LoggedModelTag(k, v)  # type: ignore[no-untyped-call]
                for k, v in tags_dict.items()
            ],
            params=[
                LoggedModelParameter(k, v)  # type: ignore[no-untyped-call]
                for k, v in params_dict.items()
            ],
            metrics=[
                Metric(
                    key=m["key"],
                    value=float(m["value"]),
                    timestamp=int(m.get("timestamp", 0)),
                    step=int(m.get("step", 0)),
                )
                for m in metrics_list
            ],
        )

    def create_logged_model(
        self,
        experiment_id: str,
        name: str | None = None,
        source_run_id: str | None = None,
        tags: Sequence[LoggedModelTag] | None = None,
        params: Sequence[LoggedModelParameter] | None = None,
        model_type: str | None = None,
    ) -> LoggedModel:
        self._ensure_extended_indexes()
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        experiment = self._require_active_experiment(exp_id)
        model_id = self._new_model_id(exp_id)
        self._model_experiment_map[model_id] = exp_id
        resolved_name = (name or f"model-{model_id[-8:]}").strip()
        if not resolved_name or len(resolved_name.encode("utf-8")) > 500:
            raise MlflowException.invalid_parameter_value("Invalid logged model name")

        artifact_location = str(
            append_to_uri_path(  # type: ignore[no-untyped-call]
                experiment.artifact_location or self.artifact_uri,  # type: ignore[attr-defined]
                "models",
                model_id,
                "artifacts",
            )
        )
        now_ms = _now_ms()
        tags_map = {t.key: str(t.value) for t in (tags or [])}
        params_map = {p.key: str(p.value) for p in (params or [])}

        client = self._application()  # type: ignore[attr-defined]
        res = client.upsert_logged_model(
            UpsertLoggedModelInput(
                request_id=str(uuid4()),
                experiment_id=exp_id,
                model_id=model_id,
                name=resolved_name,
                artifact_location=artifact_location,
                creation_timestamp_ms=now_ms,
                last_updated_timestamp_ms=now_ms,
                status=LoggedModelStatus.PENDING.value,
                lifecycle_stage=LifecycleStage.ACTIVE,
                status_message=None,
                source_run_id=source_run_id,
                model_type=model_type,
                packed_tags=self._encode_json_dict(tags_map),
                packed_params=self._encode_json_dict(params_map),
                packed_metrics=b"[]",
            )
        )
        if not isinstance(res.outcome, UpsertLoggedModelLoggedModelUpserted):
            raise MlflowException("Invalid UpsertLoggedModel response from RiffDB")
        return self._to_mlflow_logged_model(res.outcome.model)

    def _get_logged_model_row(self, model_id: str, allow_deleted: bool = False) -> Any:
        exp_id = self._resolve_model_experiment_id(model_id)
        client = self._application()  # type: ignore[attr-defined]
        res = client.get_logged_model(GetLoggedModelParams(experiment_id=exp_id, model_id=model_id))
        if isinstance(res.value, GetLoggedModelMissing) or not isinstance(
            res.value, GetLoggedModelFound
        ):
            raise MlflowException(
                f"LoggedModel with ID '{model_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not allow_deleted and res.value.model.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                f"LoggedModel with ID '{model_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return res.value.model

    def get_logged_model(self, model_id: str, allow_deleted: bool = False) -> LoggedModel:
        row = self._get_logged_model_row(model_id, allow_deleted=allow_deleted)
        return self._to_mlflow_logged_model(row)

    def delete_logged_model(self, model_id: str) -> None:
        row = self._get_logged_model_row(model_id)
        client = self._application()  # type: ignore[attr-defined]
        res = client.delete_logged_model_entity(
            DeleteLoggedModelEntityInput(
                request_id=str(uuid4()),
                experiment_id=int(row.experiment_id),
                model_id=model_id,
            )
        )
        if isinstance(res.outcome, DeleteLoggedModelEntityLoggedModelMissing):
            raise MlflowException(
                f"LoggedModel with ID '{model_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteLoggedModelEntityLoggedModelDeleted):
            raise MlflowException("Invalid DeleteLoggedModelEntity response from RiffDB")

    def finalize_logged_model(self, model_id: str, status: LoggedModelStatus | str) -> LoggedModel:
        row = self._get_logged_model_row(model_id)
        status_str = status.value if isinstance(status, LoggedModelStatus) else str(status)
        client = self._application()  # type: ignore[attr-defined]
        res = client.upsert_logged_model(
            UpsertLoggedModelInput(
                request_id=str(uuid4()),
                experiment_id=int(row.experiment_id),
                model_id=model_id,
                name=row.name,
                artifact_location=row.artifact_location,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=_now_ms(),
                status=status_str,
                lifecycle_stage=row.lifecycle_stage,
                status_message=row.status_message,
                source_run_id=row.source_run_id,
                model_type=row.model_type,
                packed_tags=row.packed_tags,
                packed_params=row.packed_params,
                packed_metrics=row.packed_metrics,
            )
        )
        if not isinstance(res.outcome, UpsertLoggedModelLoggedModelUpserted):
            raise MlflowException("Invalid UpsertLoggedModel response from RiffDB")
        return self._to_mlflow_logged_model(res.outcome.model)

    def set_logged_model_tags(self, model_id: str, tags: Sequence[LoggedModelTag]) -> None:
        row = self._get_logged_model_row(model_id)
        tags_map = self._decode_json_dict(row.packed_tags)
        for tag in tags or []:
            tags_map[str(tag.key)] = str(tag.value)
        client = self._application()  # type: ignore[attr-defined]
        res = client.upsert_logged_model(
            UpsertLoggedModelInput(
                request_id=str(uuid4()),
                experiment_id=int(row.experiment_id),
                model_id=model_id,
                name=row.name,
                artifact_location=row.artifact_location,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=_now_ms(),
                status=row.status,
                lifecycle_stage=row.lifecycle_stage,
                status_message=row.status_message,
                source_run_id=row.source_run_id,
                model_type=row.model_type,
                packed_tags=self._encode_json_dict(tags_map),
                packed_params=row.packed_params,
                packed_metrics=row.packed_metrics,
            )
        )
        if not isinstance(res.outcome, UpsertLoggedModelLoggedModelUpserted):
            raise MlflowException("Invalid UpsertLoggedModel response from RiffDB")

    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        row = self._get_logged_model_row(model_id)
        tags_map = self._decode_json_dict(row.packed_tags)
        tags_map.pop(key, None)
        client = self._application()  # type: ignore[attr-defined]
        res = client.upsert_logged_model(
            UpsertLoggedModelInput(
                request_id=str(uuid4()),
                experiment_id=int(row.experiment_id),
                model_id=model_id,
                name=row.name,
                artifact_location=row.artifact_location,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=_now_ms(),
                status=row.status,
                lifecycle_stage=row.lifecycle_stage,
                status_message=row.status_message,
                source_run_id=row.source_run_id,
                model_type=row.model_type,
                packed_tags=self._encode_json_dict(tags_map),
                packed_params=row.packed_params,
                packed_metrics=row.packed_metrics,
            )
        )
        if not isinstance(res.outcome, UpsertLoggedModelLoggedModelUpserted):
            raise MlflowException("Invalid UpsertLoggedModel response from RiffDB")

    def log_logged_model_params(
        self, model_id: str, params: Sequence[LoggedModelParameter]
    ) -> None:
        row = self._get_logged_model_row(model_id)
        params_map = self._decode_json_dict(row.packed_params)
        for param in params or []:
            key = str(param.key)
            val = str(param.value)
            if key in params_map and params_map[key] != val:
                raise MlflowException(
                    f"Changing param values is not allowed for logged model '{model_id}': "
                    f"{key!r} had {params_map[key]!r}, got {val!r}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            params_map[key] = val
        client = self._application()  # type: ignore[attr-defined]
        res = client.upsert_logged_model(
            UpsertLoggedModelInput(
                request_id=str(uuid4()),
                experiment_id=int(row.experiment_id),
                model_id=model_id,
                name=row.name,
                artifact_location=row.artifact_location,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=_now_ms(),
                status=row.status,
                lifecycle_stage=row.lifecycle_stage,
                status_message=row.status_message,
                source_run_id=row.source_run_id,
                model_type=row.model_type,
                packed_tags=row.packed_tags,
                packed_params=self._encode_json_dict(params_map),
                packed_metrics=row.packed_metrics,
            )
        )
        if not isinstance(res.outcome, UpsertLoggedModelLoggedModelUpserted):
            raise MlflowException("Invalid UpsertLoggedModel response from RiffDB")

    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: str | None = None,
        datasets: list[dict[str, Any]] | None = None,
        max_results: int | None = None,
        order_by: list[dict[str, Any]] | None = None,
        page_token: str | None = None,
    ) -> PagedList[LoggedModel]:
        self._ensure_extended_indexes()
        requested = 100 if max_results is None else int(max_results)
        if requested < 1:
            raise MlflowException.invalid_parameter_value("max_results must be a positive integer")
        client = self._application()  # type: ignore[attr-defined]
        models: list[LoggedModel] = []
        for exp_id_str in experiment_ids:
            exp_id = int(self._numeric_experiment_id(exp_id_str))  # type: ignore[attr-defined]
            cursor: str | None = None
            while True:
                res = client.search_logged_models(
                    SearchLoggedModelsParams(experiment_id=exp_id, limit=50, after=cursor)
                )
                if not isinstance(res.value, SearchLoggedModelsFound):
                    raise MlflowException("Invalid SearchLoggedModels response from RiffDB")
                for row in res.value.models:
                    self._model_experiment_map[row.model_id] = exp_id
                    if row.lifecycle_stage != LifecycleStage.ACTIVE:
                        continue
                    model = self._to_mlflow_logged_model(row)
                    if self._matches_logged_model_filter(model, filter_string):
                        models.append(model)
                cursor = res.next_cursor
                if not cursor or not res.value.models:
                    break

        if order_by:
            for spec in reversed(order_by):
                field_name = str(spec.get("field_name", "creation_timestamp"))
                ascending = bool(spec.get("ascending", False))
                if field_name in ("creation_timestamp", "creation_time"):
                    models.sort(key=lambda m: int(m.creation_timestamp), reverse=not ascending)
                elif field_name in ("last_updated_timestamp", "last_update_time"):
                    models.sort(
                        key=lambda m: int(m.last_updated_timestamp),
                        reverse=not ascending,
                    )
                elif field_name == "name":
                    models.sort(key=lambda m: m.name, reverse=not ascending)
        else:
            models.sort(key=lambda m: (-int(m.creation_timestamp), m.model_id))

        offset = int(page_token) if page_token is not None else 0
        sliced = models[offset : offset + requested]
        next_tok = str(offset + requested) if offset + requested < len(models) else None
        return PagedList(sliced, next_tok)

    @classmethod
    def _matches_logged_model_filter(cls, model: LoggedModel, filter_string: str | None) -> bool:
        if not filter_string or not filter_string.strip():
            return True
        clauses = [
            c.strip() for c in re.split(r"\bAND\b", filter_string, flags=re.IGNORECASE) if c.strip()
        ]
        tags_map = dict(model.tags or {})
        params_map = dict(model.params or {})
        for clause in clauses:
            m = _FILTER_CLAUSE_RE.match(clause)
            if not m:
                raise MlflowException.invalid_parameter_value(
                    f"Unsupported logged model filter clause: {clause!r}"
                )
            lhs = m.group(1).strip("`'\"")
            op = m.group(2).upper()
            raw_rhs = m.group(3).strip().strip("'\"")
            if lhs in ("name", "attributes.name"):
                actual: Any = model.name
            elif lhs in ("status", "attributes.status"):
                actual = (
                    model.status.value
                    if isinstance(model.status, LoggedModelStatus)
                    else str(model.status)
                )
            elif lhs in ("model_type", "attributes.model_type"):
                actual = model.model_type
            elif lhs in ("source_run_id", "attributes.source_run_id"):
                actual = model.source_run_id
            elif lhs in ("creation_timestamp", "creation_time"):
                actual = int(model.creation_timestamp)
            elif lhs.startswith("tags.") or lhs.startswith("tag."):
                actual = tags_map.get(lhs.split(".", 1)[1].strip("`'\""))
            elif lhs.startswith("params.") or lhs.startswith("param."):
                actual = params_map.get(lhs.split(".", 1)[1].strip("`'\""))
            else:
                raise MlflowException.invalid_parameter_value(
                    f"Unsupported logged model filter field: {lhs!r}"
                )
            if not cls._compare_filter_value(actual, op, raw_rhs):
                return False
        return True
