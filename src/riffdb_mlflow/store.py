"""MLflow tracking-store implementation backed by a compiled RiffDB profile."""

from __future__ import annotations

import json
import math
import os
import secrets
import struct
import sys
from hashlib import sha256
from typing import Any, NoReturn, cast
from uuid import uuid4

from mlflow.entities import Experiment as MLflowExperiment
from mlflow.entities import (
    ExperimentTag,
    Metric,
    Param,
    RunData,
    RunInfo,
    RunInputs,
    RunOutputs,
    RunTag,
    ViewType,
)
from mlflow.entities import LifecycleStage as MLflowLifecycleStage
from mlflow.entities import Run as MLflowRun
from mlflow.entities import RunStatus as MLflowRunStatus
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.protos.databricks_pb2 import (
    ABORTED,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.validation import (
    _validate_batch_log_data,
    _validate_batch_log_limits,
    _validate_experiment_artifact_location_length,
    _validate_experiment_id,
    _validate_experiment_id_type,
    _validate_experiment_name,
    _validate_experiment_tag,
    _validate_param_keys_unique,
    _validate_run_id,
    _validate_tag,
)
from riffdb_application import (
    AttemptBudget,
    BearerCredential,
    CallMetadata,
    DatabaseAlias,
    SyncApplicationTransport,
    VerifiedTlsConfig,
)

from riffdb_mlflow.config import StoreConfig
from riffdb_mlflow.generated.client import (
    CreateExperimentExperimentCreated,
    CreateExperimentExperimentExists,
    CreateExperimentInput,
    CreateExperimentWithTagsExperimentCreated,
    CreateExperimentWithTagsExperimentExists,
    CreateExperimentWithTagsInput,
    CreateRunWithTagsInput,
    CreateRunWithTagsRunCreated,
    CreateRunWithTagsRunExists,
    CreateRunWithTagsRunTagExists,
    DeleteExperimentExperimentDeleted,
    DeleteExperimentExperimentMissing,
    DeleteExperimentInput,
    DeleteExperimentTagExperimentInactive,
    DeleteExperimentTagExperimentMissing,
    DeleteExperimentTagExperimentStale,
    DeleteExperimentTagExperimentTagDeleted,
    DeleteExperimentTagExperimentTagMissing,
    DeleteExperimentTagInput,
    DeleteRunInput,
    DeleteRunRunDeleted,
    DeleteRunRunMissing,
    ExperimentOrder,
    GetExperimentByNameFound,
    GetExperimentByNameMissing,
    GetExperimentByNameParams,
    GetExperimentFound,
    GetExperimentMissing,
    GetExperimentParams,
    GetMetricHistoryFound,
    GetMetricHistoryParams,
    GetRunFound,
    GetRunMissing,
    GetRunParams,
    LifecycleStage,
    LogBatchBatchLogged,
    LogBatchInput,
    LogBatchParamConflict,
    LogBatchRunInactive,
    LogBatchRunMissing,
    LogBatchRunStale,
    RenameExperimentExperimentMissing,
    RenameExperimentExperimentRenamed,
    RenameExperimentInput,
    RestoreExperimentExperimentMissing,
    RestoreExperimentExperimentRestored,
    RestoreExperimentInput,
    RestoreRunInput,
    RestoreRunRunMissing,
    RestoreRunRunRestored,
    RiffDBMLflowClient,
    RunDataKind,
    RunDatum,
    SearchExperimentsByNameILikeParams,
    SearchExperimentsByNameLikeParams,
    SearchExperimentsByNameParams,
    SearchExperimentsByTagILikeParams,
    SearchExperimentsByTagLikeParams,
    SearchExperimentsByTagParams,
    SearchExperimentsByTwoTagsParams,
    SearchExperimentsParams,
    SearchRunsByLifecycleParams,
    SearchRunsParams,
    SetExperimentTagExperimentInactive,
    SetExperimentTagExperimentMissing,
    SetExperimentTagExperimentStale,
    SetExperimentTagExperimentTagSet,
    SetExperimentTagInput,
    UpdateRunInfoInput,
    UpdateRunInfoRunInactive,
    UpdateRunInfoRunMissing,
    UpdateRunInfoRunStale,
    UpdateRunInfoRunUpdated,
    ValidateRunActiveInput,
    ValidateRunActiveRunActive,
    ValidateRunActiveRunInactive,
    ValidateRunActiveRunMissing,
)
from riffdb_mlflow.generated.client import (
    ExperimentTag as RiffDBExperimentTag,
)
from riffdb_mlflow.generated.client import RunStatus as RiffDBRunStatus

_SCOPE = "global"
_DEFAULT_EXPERIMENT_ID = 0
_DEFAULT_EXPERIMENT_NAME = "Default"
_PACKED_TAGS_MAX_BYTES = 900_000
_PACKED_PARAMS_MAX_BYTES = 700_000
_PACKED_LATEST_METRICS_MAX_BYTES = 400_000
_COMPILED_SEARCH_PAGE_SIZE = 4
_COMPILED_RUN_SEARCH_PAGE_SIZE = 2
_COMPILED_METRIC_HISTORY_PAGE_SIZE = 499
_MAX_METRIC_HISTORY_RESULTS = 10_000
_MAX_RUN_SEARCH_EXPERIMENTS = 1_000
_MAX_RUN_TAGS = 100
_MAX_RUN_NAME_LENGTH = 250
_MAX_USER_ID_LENGTH = 256
_MIN_RANDOM_EXPERIMENT_ID = 100_000_000_000_000_000
_RANDOM_EXPERIMENT_ID_SPAN = 900_000_000_000_000_000


class RiffDBTrackingStore(AbstractStore):
    """Pinned MLflow tracking store backed by one compiled application profile."""

    DEFAULT_EXPERIMENT_ID = "0"

    def __init__(self, store_uri: str, artifact_uri: str | None = None) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self._config = StoreConfig.from_uri(store_uri)
        self._artifact_uri = artifact_uri
        self._transport: SyncApplicationTransport | None = None
        self._client: Any | None = None
        self._default_ready = False

    @property
    def database(self) -> str:
        return self._config.database

    @property
    def artifact_uri(self) -> str | None:
        return self._artifact_uri

    @staticmethod
    def _pending(operation: str) -> NoReturn:
        raise MlflowNotImplementedException(  # type: ignore[no-untyped-call]
            f"riffdb-mlflow core operation is not implemented yet: {operation}"
        )

    def _application(self) -> Any:
        if self._client is None:
            required = {
                name: os.environ.get(name)
                for name in (
                    "RIFFDB_MLFLOW_ENDPOINT",
                    "RIFFDB_MLFLOW_TRUST_ROOT",
                    "RIFFDB_MLFLOW_SERVER_NAME",
                    "RIFFDB_MLFLOW_CREDENTIAL_FILE",
                )
            }
            missing = sorted(name for name, value in required.items() if not value)
            if missing:
                raise MlflowException(
                    "Missing required RiffDB MLflow connection configuration: "
                    + ", ".join(missing),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            credential = BearerCredential.from_protected_file(
                required["RIFFDB_MLFLOW_CREDENTIAL_FILE"]  # type: ignore[arg-type]
            )
            if os.environ.get("RIFFDB_TENANT") == "":
                os.environ.pop("RIFFDB_TENANT", None)
            metadata = CallMetadata.authenticated(credential).with_database(
                DatabaseAlias(self.database)
            )
            tls = VerifiedTlsConfig(
                endpoint=required["RIFFDB_MLFLOW_ENDPOINT"],  # type: ignore[arg-type]
                trust_root=required["RIFFDB_MLFLOW_TRUST_ROOT"],  # type: ignore[arg-type]
                server_name=required["RIFFDB_MLFLOW_SERVER_NAME"],  # type: ignore[arg-type]
            )
            self._transport = SyncApplicationTransport.connect_verified_tls(tls, metadata)
            self._client = RiffDBMLflowClient(self._transport, AttemptBudget(3))
        if not self._default_ready:
            self._ensure_default_experiment(self._client)
            self._default_ready = True
        return self._client

    def _ensure_default_experiment(self, client: Any) -> None:
        artifact_location = self._resolved_artifact_location("0", None)
        result = client.create_experiment(
            CreateExperimentInput(
                request_id=str(uuid4()),
                experiment_id=_DEFAULT_EXPERIMENT_ID,
                name=_DEFAULT_EXPERIMENT_NAME,
                name_digest=self._digest(_DEFAULT_EXPERIMENT_NAME),
                artifact_location=artifact_location,
                packed_tags=self._encode_tags(None),
            )
        )
        if not isinstance(
            result.outcome,
            CreateExperimentExperimentCreated | CreateExperimentExperimentExists,
        ):
            raise MlflowException("Unable to initialize the default experiment")

    def _resolved_artifact_location(self, experiment_id: str, artifact_location: str | None) -> str:
        if artifact_location:
            _validate_experiment_artifact_location_length(artifact_location)
            return artifact_location
        if not self._artifact_uri:
            raise MlflowException(
                "A default artifact root is required for the RiffDB tracking store",
                error_code=INVALID_PARAMETER_VALUE,
            )
        resolved = append_to_uri_path(  # type: ignore[no-untyped-call]
            self._artifact_uri, experiment_id
        )
        return str(resolved)

    @staticmethod
    def _digest(value: str) -> bytes:
        return sha256(value.encode("utf-8")).digest()

    @staticmethod
    def _encode_tags(tags: Any) -> bytes:
        packed: dict[str, str] = {}
        for tag in tags or ():
            _validate_experiment_tag(tag.key, tag.value)  # type: ignore[no-untyped-call]
            if tag.key in packed:
                raise MlflowException(
                    f"Duplicate experiment tag key: {tag.key!r}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            packed[tag.key] = tag.value
        encoded = json.dumps(
            packed, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > _PACKED_TAGS_MAX_BYTES:
            raise MlflowException(
                "Experiment tags exceed the RiffDB profile's encoded byte limit",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return encoded

    @classmethod
    def _tag_rows(cls, experiment_id: int, tags: Any) -> tuple[RiffDBExperimentTag, ...]:
        rows: list[RiffDBExperimentTag] = []
        seen: set[str] = set()
        for tag in tags or ():
            _validate_experiment_tag(tag.key, tag.value)  # type: ignore[no-untyped-call]
            if tag.key in seen:
                raise MlflowException(
                    f"Duplicate experiment tag key: {tag.key!r}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            seen.add(tag.key)
            rows.append(
                RiffDBExperimentTag(
                    scope=_SCOPE,
                    experiment_id=experiment_id,
                    tag_key=tag.key,
                    value=tag.value,
                    value_digest=cls._digest(tag.value),
                )
            )
        return tuple(rows)

    @staticmethod
    def _decode_tags(encoded: bytes) -> dict[str, str]:
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MlflowException("Stored experiment tags are corrupt") from error
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise MlflowException("Stored experiment tags have an invalid shape")
        return value

    @staticmethod
    def _encode_run_tags(tags: Any) -> bytes:
        packed: dict[str, str] = {}
        for index, tag in enumerate(tags or ()):
            normalized = _validate_tag(  # type: ignore[no-untyped-call]
                tag.key, tag.value, path=f"tags[{index}]"
            )
            if normalized.key in packed:
                raise MlflowException(
                    f"Duplicate run tag key: {normalized.key!r}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            packed[normalized.key] = normalized.value
        if len(packed) > _MAX_RUN_TAGS:
            raise MlflowException.invalid_parameter_value(
                f"A run may contain at most {_MAX_RUN_TAGS} tags"
            )
        encoded = json.dumps(
            packed, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > _PACKED_TAGS_MAX_BYTES:
            raise MlflowException.invalid_parameter_value(
                "Run tags exceed the RiffDB profile's encoded byte limit"
            )
        return encoded

    @staticmethod
    def _latest_item_key(
        kind: RunDataKind, key: str, entry_id: str, *, promote_latest: bool
    ) -> str:
        if kind == RunDataKind.METRIC and promote_latest:
            return key
        digest = sha256(f"{kind.value}:{key}:{entry_id}".encode()).hexdigest()
        return f"~{digest}"

    @classmethod
    def _run_tag_rows(cls, experiment_id: int, run_id: str, tags: Any) -> tuple[RunDatum, ...]:
        packed = cls._decode_tags(cls._encode_run_tags(tags))
        return tuple(
            RunDatum(
                experiment_id=experiment_id,
                run_id=run_id,
                kind=RunDataKind.TAG,
                item_key=key,
                entry_id="",
                logical_key=key,
                latest_item_key=cls._latest_item_key(
                    RunDataKind.TAG, key, "", promote_latest=False
                ),
                sort_key="",
                latest_sort_key="",
                text_value=value,
                value_digest=cls._digest(value),
                value_bits=0,
                value_identity=0,
                value_order=0,
                metric_timestamp=0,
                metric_step=0,
                is_nan=False,
                promote_latest=False,
            )
            for key, value in packed.items()
        )

    @staticmethod
    def _encode_params(params: dict[str, str]) -> bytes:
        encoded = json.dumps(
            params, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > _PACKED_PARAMS_MAX_BYTES:
            raise MlflowException.invalid_parameter_value(
                "Run parameters exceed the RiffDB profile's encoded byte limit"
            )
        return encoded

    @staticmethod
    def _decode_params(encoded: bytes) -> dict[str, str]:
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MlflowException("Stored run parameters are corrupt") from error
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise MlflowException("Stored run parameters have an invalid shape")
        return value

    @staticmethod
    def _float_bits(value: float) -> int:
        return cast(int, struct.unpack(">Q", struct.pack(">d", value))[0])

    @staticmethod
    def _bits_float(value: int) -> float:
        return cast(float, struct.unpack(">d", struct.pack(">Q", value))[0])

    @classmethod
    def _metric_encoding(cls, value: Any) -> tuple[int, int, int, bool, float]:
        raw = float(value)
        raw_bits = cls._float_bits(raw)
        is_nan = math.isnan(raw)
        if is_nan:
            sanitized = 0.0
        elif math.isinf(raw):
            sanitized = sys.float_info.max if raw > 0 else -sys.float_info.max
        else:
            sanitized = raw
        if sanitized == 0.0:
            sanitized = 0.0
        identity = cls._float_bits(sanitized)
        if identity & (1 << 63):
            order = (~identity) & ((1 << 64) - 1)
        else:
            order = identity ^ (1 << 63)
        return raw_bits, identity, order, is_nan, sanitized

    @staticmethod
    def _signed_sort(value: int) -> str:
        if value < -(1 << 63) or value > (1 << 63) - 1:
            raise MlflowException.invalid_parameter_value(
                "Metric timestamp and step must be signed 64-bit integers"
            )
        return f"{(value + (1 << 63)) & ((1 << 64) - 1):016x}"

    @classmethod
    def _metric_identity(cls, timestamp: int, step: int, identity: int, is_nan: bool) -> str:
        return (
            cls._signed_sort(timestamp)
            + cls._signed_sort(step)
            + f"{identity:016x}"
            + ("1" if is_nan else "0")
        )

    @classmethod
    def _metric_history_sort(cls, timestamp: int, step: int, order: int, is_nan: bool) -> str:
        return (
            cls._signed_sort(timestamp)
            + cls._signed_sort(step)
            + f"{order:016x}"
            + ("1" if is_nan else "0")
        )

    @staticmethod
    def _metric_item_key(run_id: str, metric_key: str) -> str:
        return f"{run_id}:{metric_key}"

    @staticmethod
    def _encode_latest_metrics(values: dict[str, tuple[int, int, int]]) -> bytes:
        payload = {
            key: [f"{bits:016x}", timestamp, step]
            for key, (bits, timestamp, step) in values.items()
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > _PACKED_LATEST_METRICS_MAX_BYTES:
            raise MlflowException.invalid_parameter_value(
                "Latest metrics exceed the RiffDB profile's encoded byte limit"
            )
        return encoded

    @staticmethod
    def _decode_latest_metrics(encoded: bytes) -> dict[str, tuple[int, int, int]]:
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MlflowException("Stored latest metrics are corrupt") from error
        if not isinstance(value, dict):
            raise MlflowException("Stored latest metrics have an invalid shape")
        decoded: dict[str, tuple[int, int, int]] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not isinstance(item, list)
                or len(item) != 3
                or not isinstance(item[0], str)
                or not isinstance(item[1], int)
                or not isinstance(item[2], int)
            ):
                raise MlflowException("Stored latest metrics have an invalid shape")
            try:
                bits = int(item[0], 16)
            except ValueError as error:
                raise MlflowException("Stored latest metrics are corrupt") from error
            if bits < 0 or bits > (1 << 64) - 1:
                raise MlflowException("Stored latest metrics are corrupt")
            decoded[key] = (bits, item[1], item[2])
        return decoded

    @staticmethod
    def _new_run_id(experiment_id: int) -> str:
        return f"{experiment_id:016x}{secrets.token_hex(8)}"

    @staticmethod
    def _run_experiment_id(run_id: Any) -> tuple[str, int]:
        _validate_run_id(run_id)  # type: ignore[no-untyped-call]
        value = str(run_id)
        if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
            raise MlflowException.invalid_parameter_value(f"Invalid run ID: {run_id!r}")
        return value, int(value[:16], 16)

    @staticmethod
    def _missing_run(run_id: Any) -> NoReturn:
        raise MlflowException(
            f"Run with id={run_id} not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    @staticmethod
    def _validate_run_name(run_name: Any) -> str:
        if not isinstance(run_name, str) or not run_name or len(run_name) > _MAX_RUN_NAME_LENGTH:
            raise MlflowException.invalid_parameter_value(
                f"Run name must be a non-empty string of at most {_MAX_RUN_NAME_LENGTH} characters"
            )
        return run_name

    @staticmethod
    def _to_riffdb_run_status(status: Any) -> RiffDBRunStatus:
        try:
            name = MLflowRunStatus.to_string(status)  # type: ignore[no-untyped-call]
            return RiffDBRunStatus[name]
        except (KeyError, TypeError, ValueError) as error:
            raise MlflowException.invalid_parameter_value(
                f"Invalid run status: {status!r}"
            ) from error

    @classmethod
    def _to_mlflow_run(cls, row: Any) -> MLflowRun:
        lifecycle = (
            MLflowLifecycleStage.ACTIVE
            if row.lifecycle_stage == LifecycleStage.ACTIVE
            else MLflowLifecycleStage.DELETED
        )
        info = RunInfo(  # type: ignore[no-untyped-call]
            row.run_id,
            str(row.experiment_id),
            row.user_id,
            row.status.name,
            row.start_time,
            row.end_time,
            lifecycle,
            row.artifact_uri,
            row.run_name,
        )
        data = RunData(  # type: ignore[no-untyped-call]
            metrics=[
                Metric(
                    key,
                    cls._bits_float(bits),
                    timestamp,
                    step,
                )
                for key, (bits, timestamp, step) in cls._decode_latest_metrics(
                    row.packed_latest_metrics
                ).items()
            ],
            params=[
                Param(key, value)  # type: ignore[no-untyped-call]
                for key, value in cls._decode_params(row.packed_params).items()
            ],
            tags=[
                RunTag(key, value)  # type: ignore[no-untyped-call]
                for key, value in cls._decode_tags(row.packed_tags).items()
            ],
        )
        return MLflowRun(info, data, RunInputs(dataset_inputs=[]), RunOutputs(model_outputs=[]))

    @staticmethod
    def _numeric_experiment_id(experiment_id: Any) -> int:
        _validate_experiment_id_type(experiment_id)  # type: ignore[no-untyped-call]
        text = str(experiment_id)
        _validate_experiment_id(text)  # type: ignore[no-untyped-call]
        try:
            value = int(text)
        except (TypeError, ValueError) as error:
            raise MlflowException(
                f"Invalid experiment ID '{experiment_id}'. Experiment ID must be a valid integer.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from error
        if value < 0 or value > 18_446_744_073_709_551_615:
            raise MlflowException(
                f"Invalid experiment ID: '{experiment_id}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return value

    @staticmethod
    def _to_mlflow_experiment(row: Any) -> MLflowExperiment:
        lifecycle = (
            MLflowLifecycleStage.ACTIVE
            if row.lifecycle_stage == LifecycleStage.ACTIVE
            else MLflowLifecycleStage.DELETED
        )
        creation_time = row.creation_time.seconds * 1000 + row.creation_time.nanos // 1_000_000
        last_update_time = (
            row.last_update_time.seconds * 1000 + row.last_update_time.nanos // 1_000_000
        )
        return MLflowExperiment(  # type: ignore[no-untyped-call]
            experiment_id=str(row.experiment_id),
            name=row.name,
            artifact_location=row.artifact_location,
            lifecycle_stage=lifecycle,
            tags=[
                ExperimentTag(key, value)  # type: ignore[no-untyped-call]
                for key, value in RiffDBTrackingStore._decode_tags(row.packed_tags).items()
            ],
            creation_time=creation_time,
            last_update_time=last_update_time,
        )

    @staticmethod
    def _missing_experiment(experiment_id: Any) -> NoReturn:
        raise MlflowException(
            f"No Experiment with id={experiment_id} exists",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    @staticmethod
    def _search_lifecycle_stages(view_type: Any) -> tuple[LifecycleStage, ...]:
        if view_type is None:
            view_type = ViewType.ACTIVE_ONLY
        if view_type not in (ViewType.ACTIVE_ONLY, ViewType.DELETED_ONLY, ViewType.ALL):
            raise MlflowException.invalid_parameter_value(f"Invalid view type: {view_type}")
        stages = MLflowLifecycleStage.view_type_to_stages(  # type: ignore[no-untyped-call]
            view_type
        )
        return tuple(
            LifecycleStage.ACTIVE
            if stage == MLflowLifecycleStage.ACTIVE
            else LifecycleStage.DELETED
            for stage in stages
        )

    @staticmethod
    def _search_order(order_by: Any) -> ExperimentOrder:
        parsed = [
            SearchExperimentsUtils.parse_order_by_for_search_experiments(  # type: ignore[no-untyped-call]
                item
            )
            for item in (order_by or ["creation_time DESC", "experiment_id ASC"])
        ]
        if not parsed or len(parsed) > 2 or any(type_ != "attribute" for type_, _, _ in parsed):
            raise MlflowException.invalid_parameter_value(
                "RiffDB experiment search accepts one ordered attribute and an optional "
                "experiment_id tie-breaker"
            )
        primary_type, primary, ascending = parsed[0]
        del primary_type
        if primary == "experiment_id":
            if len(parsed) != 1:
                raise MlflowException.invalid_parameter_value(
                    "experiment_id must be the only experiment order when it is primary"
                )
            return ExperimentOrder.ID_ASC if ascending else ExperimentOrder.ID_DESC
        if primary not in {"name", "creation_time", "last_update_time"}:
            raise MlflowException.invalid_parameter_value(
                f"Unsupported experiment order attribute: {primary}"
            )
        tie_ascending = False
        if len(parsed) == 2:
            _, tie_key, tie_ascending = parsed[1]
            if tie_key != "experiment_id":
                raise MlflowException.invalid_parameter_value(
                    "The secondary experiment order must be experiment_id"
                )
        prefix = {
            "name": "NAME",
            "creation_time": "CREATION",
            "last_update_time": "UPDATE",
        }[primary]
        direction = "ASC" if ascending else "DESC"
        tie = "ASC" if tie_ascending else "DESC"
        return ExperimentOrder[f"{prefix}_{direction}_ID_{tie}"]

    @classmethod
    def _dispatch_experiment_search(
        cls,
        client: Any,
        filters: list[dict[str, Any]],
        lifecycle_stages: tuple[LifecycleStage, ...],
        order: ExperimentOrder,
        limit: int,
        cursor: str | None,
    ) -> Any:
        if not filters:
            return client.search_experiments(
                SearchExperimentsParams(
                    scope=_SCOPE,
                    lifecycle_stages=lifecycle_stages,
                    order=order,
                    limit=limit,
                    after=cursor,
                )
            )
        if len(filters) == 2:
            first, second = filters
            if not all(
                predicate["type"] == "tag" and predicate["comparator"] == "="
                for predicate in (first, second)
            ):
                raise MlflowException.invalid_parameter_value(
                    "RiffDB two-predicate experiment search currently requires two exact tags"
                )
            return client.search_experiments_by_two_tags(
                SearchExperimentsByTwoTagsParams(
                    scope=_SCOPE,
                    first_tag_key=first["key"],
                    first_value_digest=cls._digest(first["value"]),
                    second_tag_key=second["key"],
                    second_value_digest=cls._digest(second["value"]),
                    lifecycle_stages=lifecycle_stages,
                    order=order,
                    limit=limit,
                    after=cursor,
                )
            )
        predicate = filters[0]
        kind = predicate["type"]
        key = predicate["key"]
        comparator = predicate["comparator"]
        value = predicate["value"]
        if kind == "attribute" and key == "name":
            if comparator == "=":
                return client.search_experiments_by_name(
                    SearchExperimentsByNameParams(
                        scope=_SCOPE,
                        name_digest=cls._digest(value),
                        lifecycle_stages=lifecycle_stages,
                        order=order,
                        limit=limit,
                        after=cursor,
                    )
                )
            if comparator == "LIKE":
                return client.search_experiments_by_name_like(
                    SearchExperimentsByNameLikeParams(
                        scope=_SCOPE,
                        pattern=value,
                        lifecycle_stages=lifecycle_stages,
                        order=order,
                        limit=limit,
                        after=cursor,
                    )
                )
            if comparator == "ILIKE":
                return client.search_experiments_by_name_ilike(
                    SearchExperimentsByNameILikeParams(
                        scope=_SCOPE,
                        pattern=value,
                        lifecycle_stages=lifecycle_stages,
                        order=order,
                        limit=limit,
                        after=cursor,
                    )
                )
        if kind == "tag":
            if comparator == "=":
                return client.search_experiments_by_tag(
                    SearchExperimentsByTagParams(
                        scope=_SCOPE,
                        tag_key=key,
                        value_digest=cls._digest(value),
                        lifecycle_stages=lifecycle_stages,
                        order=order,
                        limit=limit,
                        after=cursor,
                    )
                )
            if comparator == "LIKE":
                return client.search_experiments_by_tag_like(
                    SearchExperimentsByTagLikeParams(
                        scope=_SCOPE,
                        tag_key=key,
                        pattern=value,
                        lifecycle_stages=lifecycle_stages,
                        order=order,
                        limit=limit,
                        after=cursor,
                    )
                )
            if comparator == "ILIKE":
                return client.search_experiments_by_tag_ilike(
                    SearchExperimentsByTagILikeParams(
                        scope=_SCOPE,
                        tag_key=key,
                        pattern=value,
                        lifecycle_stages=lifecycle_stages,
                        order=order,
                        limit=limit,
                        after=cursor,
                    )
                )
        raise MlflowException.invalid_parameter_value(
            f"Unsupported RiffDB experiment filter: {kind}.{key} {comparator}"
        )

    def search_experiments(
        self,
        view_type: Any = None,
        max_results: Any = None,
        filter_string: Any = None,
        order_by: Any = None,
        page_token: Any = None,
    ) -> PagedList[MLflowExperiment]:
        if max_results is None:
            max_results = SEARCH_MAX_RESULTS_DEFAULT
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results <= 0
            or max_results > SEARCH_MAX_RESULTS_THRESHOLD
        ):
            raise MlflowException.invalid_parameter_value(
                f"Invalid value for max_results. It must be an integer between 1 and "
                f"{SEARCH_MAX_RESULTS_THRESHOLD}, but got {max_results}"
            )

        lifecycle_stages = self._search_lifecycle_stages(view_type)
        order = self._search_order(order_by)
        filters = SearchExperimentsUtils.parse_search_filter(  # type: ignore[no-untyped-call]
            filter_string
        )
        if len(filters) > 2:
            raise MlflowException.invalid_parameter_value(
                "RiffDB experiment search currently accepts at most two filter predicates"
            )

        client = self._application()
        cursor = page_token
        experiments: list[MLflowExperiment] = []
        while len(experiments) < max_results:
            limit = min(_COMPILED_SEARCH_PAGE_SIZE, max_results - len(experiments))
            result = self._dispatch_experiment_search(
                client, filters, lifecycle_stages, order, limit, cursor
            )
            value = result.value
            rows = getattr(value, "experiments", None)
            if rows is None:
                raise MlflowException("RiffDB experiment search integrity failure")
            experiments.extend(self._to_mlflow_experiment(row) for row in rows)
            cursor = result.next_cursor
            if cursor is None:
                break
            if not rows:
                raise MlflowException("RiffDB returned a non-advancing experiment cursor")
        return PagedList(experiments, cursor)

    def create_experiment(self, name: Any, artifact_location: Any = None, tags: Any = None) -> str:
        _validate_experiment_name(name)  # type: ignore[no-untyped-call]
        client = self._application()
        for _ in range(3):
            experiment_id = _MIN_RANDOM_EXPERIMENT_ID + secrets.randbelow(
                _RANDOM_EXPERIMENT_ID_SPAN
            )
            packed_tags = self._encode_tags(tags)
            tag_rows = self._tag_rows(experiment_id, tags)
            common = {
                "request_id": str(uuid4()),
                "experiment_id": experiment_id,
                "name": name,
                "name_digest": self._digest(name),
                "artifact_location": self._resolved_artifact_location(
                    str(experiment_id), artifact_location
                ),
                "packed_tags": packed_tags,
            }
            if tag_rows:
                result = client.create_experiment_with_tags(
                    CreateExperimentWithTagsInput(tags=tag_rows, **common)
                )
            else:
                result = client.create_experiment(CreateExperimentInput(**common))
            if isinstance(
                result.outcome,
                CreateExperimentExperimentCreated | CreateExperimentWithTagsExperimentCreated,
            ):
                return str(experiment_id)
            if isinstance(
                result.outcome,
                CreateExperimentExperimentExists | CreateExperimentWithTagsExperimentExists,
            ):
                continue
        raise MlflowException(
            f"Experiment '{name}' already exists.",
            error_code=RESOURCE_ALREADY_EXISTS,
        )

    def get_experiment(self, experiment_id: Any) -> MLflowExperiment:
        numeric_id = self._numeric_experiment_id(experiment_id)
        result = self._application().get_experiment(
            GetExperimentParams(scope=_SCOPE, experiment_id=numeric_id)
        )
        if isinstance(result.value, GetExperimentMissing):
            self._missing_experiment(experiment_id)
        if not isinstance(result.value, GetExperimentFound):
            raise MlflowException("Invalid GetExperiment response from RiffDB")
        return self._to_mlflow_experiment(result.value.experiment)

    def get_experiment_by_name(self, experiment_name: Any) -> MLflowExperiment | None:
        _validate_experiment_name(experiment_name)  # type: ignore[no-untyped-call]
        result = self._application().get_experiment_by_name(
            GetExperimentByNameParams(scope=_SCOPE, name=experiment_name)
        )
        if isinstance(result.value, GetExperimentByNameMissing):
            return None
        if not isinstance(result.value, GetExperimentByNameFound):
            raise MlflowException("Invalid GetExperimentByName response from RiffDB")
        return self._to_mlflow_experiment(result.value.experiment)

    def delete_experiment(self, experiment_id: Any) -> None:
        numeric_id = self._numeric_experiment_id(experiment_id)
        result = self._application().delete_experiment(
            DeleteExperimentInput(experiment_id=numeric_id)
        )
        if isinstance(result.outcome, DeleteExperimentExperimentMissing):
            self._missing_experiment(experiment_id)
        if not isinstance(result.outcome, DeleteExperimentExperimentDeleted):
            raise MlflowException("Invalid DeleteExperiment response from RiffDB")

    def restore_experiment(self, experiment_id: Any) -> None:
        numeric_id = self._numeric_experiment_id(experiment_id)
        result = self._application().restore_experiment(
            RestoreExperimentInput(experiment_id=numeric_id)
        )
        if isinstance(result.outcome, RestoreExperimentExperimentMissing):
            self._missing_experiment(experiment_id)
        if not isinstance(result.outcome, RestoreExperimentExperimentRestored):
            raise MlflowException("Invalid RestoreExperiment response from RiffDB")

    def rename_experiment(self, experiment_id: Any, new_name: Any) -> None:
        numeric_id = self._numeric_experiment_id(experiment_id)
        _validate_experiment_name(new_name)  # type: ignore[no-untyped-call]
        result = self._application().rename_experiment(
            RenameExperimentInput(
                experiment_id=numeric_id,
                name=new_name,
                name_digest=self._digest(new_name),
            )
        )
        if isinstance(result.outcome, RenameExperimentExperimentMissing):
            self._missing_experiment(experiment_id)
        if not isinstance(result.outcome, RenameExperimentExperimentRenamed):
            raise MlflowException("Invalid RenameExperiment response from RiffDB")

    def set_experiment_tag(self, experiment_id: Any, tag: Any) -> None:
        numeric_id = self._numeric_experiment_id(experiment_id)
        _validate_experiment_tag(tag.key, tag.value)  # type: ignore[no-untyped-call]
        client = self._application()
        for _ in range(3):
            current = client.get_experiment(
                GetExperimentParams(scope=_SCOPE, experiment_id=numeric_id)
            )
            if isinstance(current.value, GetExperimentMissing):
                self._missing_experiment(experiment_id)
            if not isinstance(current.value, GetExperimentFound):
                raise MlflowException("Invalid GetExperiment response from RiffDB")
            row = current.value.experiment
            packed = self._decode_tags(row.packed_tags)
            packed[tag.key] = tag.value
            result = client.set_experiment_tag(
                SetExperimentTagInput(
                    request_id=str(uuid4()),
                    experiment_id=numeric_id,
                    tag_key=tag.key,
                    value=tag.value,
                    value_digest=self._digest(tag.value),
                    packed_tags=self._encode_tags(
                        ExperimentTag(key, value)  # type: ignore[no-untyped-call]
                        for key, value in packed.items()
                    ),
                    expected_revision=row.revision,
                )
            )
            if isinstance(result.outcome, SetExperimentTagExperimentTagSet):
                return
            if isinstance(result.outcome, SetExperimentTagExperimentStale):
                continue
            if isinstance(result.outcome, SetExperimentTagExperimentMissing):
                self._missing_experiment(experiment_id)
            if isinstance(result.outcome, SetExperimentTagExperimentInactive):
                raise MlflowException(
                    f"Cannot set a tag on deleted experiment {experiment_id}",
                    error_code=INVALID_STATE,
                )
            raise MlflowException("Invalid SetExperimentTag response from RiffDB")
        raise MlflowException(
            "Experiment tag update conflicted with another write", error_code=ABORTED
        )

    def delete_experiment_tag(self, experiment_id: Any, key: Any) -> None:
        numeric_id = self._numeric_experiment_id(experiment_id)
        client = self._application()
        for _ in range(3):
            current = client.get_experiment(
                GetExperimentParams(scope=_SCOPE, experiment_id=numeric_id)
            )
            if isinstance(current.value, GetExperimentMissing):
                self._missing_experiment(experiment_id)
            if not isinstance(current.value, GetExperimentFound):
                raise MlflowException("Invalid GetExperiment response from RiffDB")
            row = current.value.experiment
            packed = self._decode_tags(row.packed_tags)
            packed.pop(key, None)
            result = client.delete_experiment_tag(
                DeleteExperimentTagInput(
                    request_id=str(uuid4()),
                    experiment_id=numeric_id,
                    tag_key=key,
                    packed_tags=self._encode_tags(
                        ExperimentTag(tag_key, value)  # type: ignore[no-untyped-call]
                        for tag_key, value in packed.items()
                    ),
                    expected_revision=row.revision,
                )
            )
            if isinstance(result.outcome, DeleteExperimentTagExperimentTagDeleted):
                return
            if isinstance(result.outcome, DeleteExperimentTagExperimentStale):
                continue
            if isinstance(result.outcome, DeleteExperimentTagExperimentMissing):
                self._missing_experiment(experiment_id)
            if isinstance(result.outcome, DeleteExperimentTagExperimentInactive):
                raise MlflowException(
                    f"Cannot delete a tag on deleted experiment {experiment_id}",
                    error_code=INVALID_STATE,
                )
            if isinstance(result.outcome, DeleteExperimentTagExperimentTagMissing):
                raise MlflowException(
                    f"No experiment tag with key {key!r} exists",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            raise MlflowException("Invalid DeleteExperimentTag response from RiffDB")
        raise MlflowException(
            "Experiment tag deletion conflicted with another write", error_code=ABORTED
        )

    def get_run(self, run_id: Any) -> MLflowRun:
        normalized_id, experiment_id = self._run_experiment_id(run_id)
        result = self._application().get_run(
            GetRunParams(experiment_id=experiment_id, run_id=normalized_id)
        )
        if isinstance(result.value, GetRunMissing):
            self._missing_run(run_id)
        if not isinstance(result.value, GetRunFound):
            raise MlflowException("Invalid GetRun response from RiffDB")
        return self._to_mlflow_run(result.value.run)

    def update_run_info(
        self, run_id: Any, run_status: Any, end_time: Any, run_name: Any
    ) -> RunInfo:
        normalized_id, experiment_id = self._run_experiment_id(run_id)
        client = self._application()
        for _ in range(3):
            current = client.get_run(
                GetRunParams(experiment_id=experiment_id, run_id=normalized_id)
            )
            if isinstance(current.value, GetRunMissing):
                self._missing_run(run_id)
            if not isinstance(current.value, GetRunFound):
                raise MlflowException("Invalid GetRun response from RiffDB")
            row = current.value.run
            next_status = (
                row.status if run_status is None else self._to_riffdb_run_status(run_status)
            )
            next_end_time = row.end_time if end_time is None else end_time
            next_name = row.run_name if not run_name else self._validate_run_name(run_name)
            tags = self._decode_tags(row.packed_tags)
            tags[MLFLOW_RUN_NAME] = next_name
            packed_tags = self._encode_run_tags(
                RunTag(key, value)  # type: ignore[no-untyped-call]
                for key, value in tags.items()
            )
            result = client.update_run_info(
                UpdateRunInfoInput(
                    request_id=str(uuid4()),
                    experiment_id=experiment_id,
                    run_id=normalized_id,
                    status=next_status,
                    end_time=next_end_time,
                    run_name=next_name,
                    run_name_tag_value=next_name,
                    run_name_digest=self._digest(next_name),
                    packed_tags=packed_tags,
                    expected_revision=row.revision,
                )
            )
            if isinstance(result.outcome, UpdateRunInfoRunUpdated):
                return self._to_mlflow_run(result.outcome.run).info
            if isinstance(result.outcome, UpdateRunInfoRunStale):
                continue
            if isinstance(result.outcome, UpdateRunInfoRunMissing):
                self._missing_run(run_id)
            if isinstance(result.outcome, UpdateRunInfoRunInactive):
                raise MlflowException(
                    f"The run {run_id} must be in the active state.",
                    error_code=INVALID_STATE,
                )
            raise MlflowException("Invalid UpdateRunInfo response from RiffDB")
        raise MlflowException("Run update conflicted with another write", error_code=ABORTED)

    def create_run(
        self,
        experiment_id: Any,
        user_id: Any,
        start_time: Any,
        tags: Any,
        run_name: Any,
    ) -> MLflowRun:
        numeric_experiment_id = self._numeric_experiment_id(experiment_id)
        experiment = self.get_experiment(str(numeric_experiment_id))
        if experiment.lifecycle_stage != MLflowLifecycleStage.ACTIVE:
            raise MlflowException(
                f"The experiment {experiment_id} must be in the active state.",
                error_code=INVALID_STATE,
            )
        if not isinstance(user_id, str) or len(user_id) > _MAX_USER_ID_LENGTH:
            raise MlflowException.invalid_parameter_value(
                f"user_id must be a string of at most {_MAX_USER_ID_LENGTH} characters"
            )
        if isinstance(start_time, bool) or not isinstance(start_time, int):
            raise MlflowException.invalid_parameter_value("start_time must be an integer")

        normalized_tags = [
            _validate_tag(  # type: ignore[no-untyped-call]
                tag.key, tag.value, path=f"tags[{index}]"
            )
            for index, tag in enumerate(list(tags or ()))
        ]
        run_name_tag = next(
            (tag.value for tag in normalized_tags if tag.key == MLFLOW_RUN_NAME), None
        )
        if run_name and run_name_tag and run_name != run_name_tag:
            raise MlflowException(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', mlflow.runName='{run_name_tag}').",
                error_code=INVALID_PARAMETER_VALUE,
            )
        resolved_name = self._validate_run_name(
            run_name or run_name_tag or _generate_random_name()  # type: ignore[no-untyped-call]
        )
        if run_name_tag is None:
            normalized_tags.append(
                RunTag(MLFLOW_RUN_NAME, resolved_name)  # type: ignore[no-untyped-call]
            )
        packed_tags = self._encode_run_tags(normalized_tags)

        client = self._application()
        for _ in range(3):
            normalized_id = self._new_run_id(numeric_experiment_id)
            artifact_uri = str(
                append_to_uri_path(  # type: ignore[no-untyped-call]
                    experiment.artifact_location,
                    normalized_id,
                    "artifacts",
                )
            )
            tag_rows = self._run_tag_rows(numeric_experiment_id, normalized_id, normalized_tags)
            result = client.create_run_with_tags(
                CreateRunWithTagsInput(
                    request_id=str(uuid4()),
                    experiment_id=numeric_experiment_id,
                    run_id=normalized_id,
                    user_id=user_id,
                    start_time=start_time,
                    artifact_uri=artifact_uri,
                    run_name=resolved_name,
                    packed_tags=packed_tags,
                    packed_params=self._encode_params({}),
                    packed_latest_metrics=self._encode_latest_metrics({}),
                    tags=tag_rows,
                )
            )
            if isinstance(result.outcome, CreateRunWithTagsRunCreated):
                return self._to_mlflow_run(result.outcome.run)
            if isinstance(result.outcome, CreateRunWithTagsRunExists):
                continue
            if isinstance(result.outcome, CreateRunWithTagsRunTagExists):
                raise MlflowException("Duplicate run tag key", error_code=INVALID_PARAMETER_VALUE)
            raise MlflowException("Invalid CreateRun response from RiffDB")
        raise MlflowException("Unable to allocate a unique run ID", error_code=ABORTED)

    def delete_run(self, run_id: Any) -> None:
        normalized_id, experiment_id = self._run_experiment_id(run_id)
        result = self._application().delete_run(
            DeleteRunInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=normalized_id,
            )
        )
        if isinstance(result.outcome, DeleteRunRunMissing):
            self._missing_run(run_id)
        if not isinstance(result.outcome, DeleteRunRunDeleted):
            raise MlflowException("Invalid DeleteRun response from RiffDB")

    def restore_run(self, run_id: Any) -> None:
        normalized_id, experiment_id = self._run_experiment_id(run_id)
        result = self._application().restore_run(
            RestoreRunInput(
                request_id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=normalized_id,
            )
        )
        if isinstance(result.outcome, RestoreRunRunMissing):
            self._missing_run(run_id)
        if not isinstance(result.outcome, RestoreRunRunRestored):
            raise MlflowException("Invalid RestoreRun response from RiffDB")

    def get_metric_history(
        self,
        run_id: Any,
        metric_key: Any,
        max_results: Any = None,
        page_token: Any = None,
    ) -> PagedList[Metric]:
        normalized_id, experiment_id = self._run_experiment_id(run_id)
        if not isinstance(metric_key, str) or not metric_key or len(metric_key) > 250:
            raise MlflowException.invalid_parameter_value("Invalid metric key")
        if isinstance(max_results, bool) or (
            max_results is not None and (not isinstance(max_results, int) or max_results < 1)
        ):
            raise MlflowException.invalid_parameter_value(
                "max_results must be a positive integer or None"
            )
        requested = _MAX_METRIC_HISTORY_RESULTS if max_results is None else max_results
        if requested > _MAX_METRIC_HISTORY_RESULTS:
            raise MlflowException.invalid_parameter_value(
                f"Metric history requests are bounded to {_MAX_METRIC_HISTORY_RESULTS} rows"
            )

        client = self._application()
        current = client.get_run(GetRunParams(experiment_id=experiment_id, run_id=normalized_id))
        if isinstance(current.value, GetRunMissing):
            self._missing_run(run_id)
        if not isinstance(current.value, GetRunFound):
            raise MlflowException("Invalid GetRun response from RiffDB")

        cursor = page_token
        metrics: list[Metric] = []
        while len(metrics) < requested:
            limit = min(_COMPILED_METRIC_HISTORY_PAGE_SIZE, requested - len(metrics))
            result = client.get_metric_history(
                GetMetricHistoryParams(
                    experiment_id=experiment_id,
                    item_key=self._metric_item_key(normalized_id, metric_key),
                    kind=RunDataKind.METRIC,
                    value_digest=bytes(32),
                    limit=limit,
                    after=cursor,
                )
            )
            if not isinstance(result.value, GetMetricHistoryFound):
                raise MlflowException("RiffDB metric history integrity failure")
            rows = result.value.metrics
            metrics.extend(
                Metric(
                    row.logical_key,
                    self._bits_float(row.value_bits),
                    row.metric_timestamp,
                    row.metric_step,
                )
                for row in rows
            )
            cursor = result.next_cursor
            if cursor is None:
                break
            if not rows:
                raise MlflowException("RiffDB returned a non-advancing metric cursor")

        if max_results is None and cursor is not None:
            raise MlflowException.invalid_parameter_value(
                f"Metric history exceeds the profile limit of {_MAX_METRIC_HISTORY_RESULTS} rows; "
                "request explicit pages"
            )
        return PagedList(metrics, cursor if max_results is not None else None)

    def _search_runs(
        self,
        experiment_ids: Any,
        filter_string: Any,
        run_view_type: Any,
        max_results: Any,
        order_by: Any,
        page_token: Any,
    ) -> tuple[list[MLflowRun], str | None]:
        if isinstance(max_results, bool) or (
            max_results is not None and (not isinstance(max_results, int) or max_results < 1)
        ):
            raise MlflowException.invalid_parameter_value(
                "max_results must be a positive integer or None"
            )
        if max_results is not None and max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException.invalid_parameter_value(
                f"max_results must be at most {SEARCH_MAX_RESULTS_THRESHOLD}"
            )
        requested = SEARCH_MAX_RESULTS_DEFAULT if max_results is None else max_results
        if len(experiment_ids) > _MAX_RUN_SEARCH_EXPERIMENTS:
            raise MlflowException.invalid_parameter_value(
                f"Run search accepts at most {_MAX_RUN_SEARCH_EXPERIMENTS} experiment IDs"
            )
        numeric_ids = tuple(self._numeric_experiment_id(value) for value in experiment_ids)
        filters = SearchUtils.parse_search_filter(  # type: ignore[no-untyped-call]
            filter_string
        )
        if filters:
            raise MlflowException.invalid_parameter_value(
                "RiffDB run search filters are not implemented yet"
            )
        parsed_order = [
            SearchUtils.parse_order_by_for_search_runs(  # type: ignore[no-untyped-call]
                item
            )
            for item in (order_by or ())
        ]
        natural_order = [
            ("attribute", "start_time", False),
            ("attribute", "run_id", True),
        ]
        if parsed_order not in ([], natural_order[:1], natural_order):
            raise MlflowException.invalid_parameter_value(
                "RiffDB run search currently supports start_time DESC, run_id ASC"
            )
        if run_view_type not in (ViewType.ACTIVE_ONLY, ViewType.DELETED_ONLY, ViewType.ALL):
            raise MlflowException.invalid_parameter_value(f"Invalid run view type: {run_view_type}")

        client = self._application()
        cursor = page_token
        runs: list[MLflowRun] = []
        while len(runs) < requested:
            limit = min(_COMPILED_RUN_SEARCH_PAGE_SIZE, requested - len(runs))
            if run_view_type == ViewType.ALL:
                result = client.search_runs(
                    SearchRunsParams(
                        experiment_ids=numeric_ids,
                        limit=limit,
                        after=cursor,
                    )
                )
            else:
                lifecycle = (
                    LifecycleStage.ACTIVE
                    if run_view_type == ViewType.ACTIVE_ONLY
                    else LifecycleStage.DELETED
                )
                result = client.search_runs_by_lifecycle(
                    SearchRunsByLifecycleParams(
                        experiment_ids=numeric_ids,
                        lifecycle_stage=lifecycle,
                        limit=limit,
                        after=cursor,
                    )
                )
            rows = getattr(result.value, "runs", None)
            if rows is None:
                raise MlflowException("RiffDB run search integrity failure")
            runs.extend(self._to_mlflow_run(row) for row in rows)
            cursor = result.next_cursor
            if cursor is None:
                break
            if not rows:
                raise MlflowException("RiffDB returned a non-advancing run cursor")
        return runs, cursor

    def log_batch(self, run_id: Any, metrics: Any, params: Any, tags: Any) -> None:
        normalized_id, experiment_id = self._run_experiment_id(run_id)
        normalized_metrics, normalized_params, normalized_tags = _validate_batch_log_data(  # type: ignore[no-untyped-call]
            list(metrics or ()), list(params or ()), list(tags or ())
        )
        _validate_batch_log_limits(  # type: ignore[no-untyped-call]
            normalized_metrics, normalized_params, normalized_tags
        )
        _validate_param_keys_unique(normalized_params)  # type: ignore[no-untyped-call]

        client = self._application()
        if not normalized_metrics and not normalized_params and not normalized_tags:
            result = client.validate_run_active(
                ValidateRunActiveInput(
                    request_id=str(uuid4()),
                    experiment_id=experiment_id,
                    run_id=normalized_id,
                )
            )
            if isinstance(result.outcome, ValidateRunActiveRunActive):
                return
            if isinstance(result.outcome, ValidateRunActiveRunMissing):
                self._missing_run(run_id)
            if isinstance(result.outcome, ValidateRunActiveRunInactive):
                raise MlflowException(
                    f"The run {run_id} must be in the active state.",
                    error_code=INVALID_STATE,
                )
            raise MlflowException("Invalid ValidateRunActive response from RiffDB")

        metric_records: list[tuple[Any, int, int, int, bool, float, str, str]] = []
        seen_metrics: set[tuple[str, int, int, int, bool]] = set()
        for metric in normalized_metrics:
            timestamp = int(metric.timestamp)
            step = int(metric.step)
            bits, identity, order, is_nan, sanitized = self._metric_encoding(metric.value)
            deduplication_key = (metric.key, timestamp, step, identity, is_nan)
            if deduplication_key in seen_metrics:
                continue
            seen_metrics.add(deduplication_key)
            entry_id = self._metric_identity(timestamp, step, identity, is_nan)
            metric_records.append(
                (
                    metric,
                    bits,
                    identity,
                    order,
                    is_nan,
                    sanitized,
                    entry_id,
                    self._metric_history_sort(timestamp, step, order, is_nan),
                )
            )

        collapsed_tags: dict[str, str] = {}
        for tag in normalized_tags:
            collapsed_tags[tag.key] = tag.value

        for _ in range(3):
            current = client.get_run(
                GetRunParams(experiment_id=experiment_id, run_id=normalized_id)
            )
            if isinstance(current.value, GetRunMissing):
                self._missing_run(run_id)
            if not isinstance(current.value, GetRunFound):
                raise MlflowException("Invalid GetRun response from RiffDB")
            row = current.value.run

            packed_params = self._decode_params(row.packed_params)
            for param in normalized_params:
                packed_params[param.key] = param.value

            packed_tags = self._decode_tags(row.packed_tags)
            packed_tags.update(collapsed_tags)

            latest = self._decode_latest_metrics(row.packed_latest_metrics)
            promoted: dict[str, str] = {}
            latest_scores: dict[str, tuple[int, int, float]] = {}
            for key, (bits, timestamp, step) in latest.items():
                _, _, _, _, sanitized = self._metric_encoding(self._bits_float(bits))
                latest_scores[key] = (step, timestamp, sanitized)
            for (
                metric,
                bits,
                _identity,
                _order,
                _is_nan,
                sanitized,
                entry_id,
                _history_sort,
            ) in metric_records:
                score = (int(metric.step), int(metric.timestamp), sanitized)
                if metric.key not in latest_scores or score > latest_scores[metric.key]:
                    latest_scores[metric.key] = score
                    latest[metric.key] = (bits, int(metric.timestamp), int(metric.step))
                    promoted[metric.key] = entry_id

            items: list[RunDatum] = []
            for (
                metric,
                bits,
                identity,
                order,
                is_nan,
                _sanitized,
                entry_id,
                history_sort,
            ) in metric_records:
                promote_latest = promoted.get(metric.key) == entry_id
                items.append(
                    RunDatum(
                        experiment_id=experiment_id,
                        run_id=normalized_id,
                        kind=RunDataKind.METRIC,
                        item_key=self._metric_item_key(normalized_id, metric.key),
                        entry_id=entry_id,
                        logical_key=metric.key,
                        latest_item_key=self._latest_item_key(
                            RunDataKind.METRIC,
                            metric.key,
                            entry_id,
                            promote_latest=promote_latest,
                        ),
                        sort_key=history_sort,
                        latest_sort_key=f"{order:016x}",
                        text_value="",
                        value_digest=bytes(32),
                        value_bits=bits,
                        value_identity=identity,
                        value_order=order,
                        metric_timestamp=int(metric.timestamp),
                        metric_step=int(metric.step),
                        is_nan=is_nan,
                        promote_latest=promote_latest,
                    )
                )
            items.extend(
                RunDatum(
                    experiment_id=experiment_id,
                    run_id=normalized_id,
                    kind=RunDataKind.PARAM,
                    item_key=param.key,
                    entry_id="",
                    logical_key=param.key,
                    latest_item_key=self._latest_item_key(
                        RunDataKind.PARAM, param.key, "", promote_latest=False
                    ),
                    sort_key="",
                    latest_sort_key="",
                    text_value=param.value,
                    value_digest=self._digest(param.value),
                    value_bits=0,
                    value_identity=0,
                    value_order=0,
                    metric_timestamp=0,
                    metric_step=0,
                    is_nan=False,
                    promote_latest=False,
                )
                for param in normalized_params
            )
            items.extend(
                RunDatum(
                    experiment_id=experiment_id,
                    run_id=normalized_id,
                    kind=RunDataKind.TAG,
                    item_key=key,
                    entry_id="",
                    logical_key=key,
                    latest_item_key=self._latest_item_key(
                        RunDataKind.TAG, key, "", promote_latest=False
                    ),
                    sort_key="",
                    latest_sort_key="",
                    text_value=value,
                    value_digest=self._digest(value),
                    value_bits=0,
                    value_identity=0,
                    value_order=0,
                    metric_timestamp=0,
                    metric_step=0,
                    is_nan=False,
                    promote_latest=False,
                )
                for key, value in collapsed_tags.items()
            )

            result = client.log_batch(
                LogBatchInput(
                    request_id=str(uuid4()),
                    experiment_id=experiment_id,
                    run_id=normalized_id,
                    expected_revision=row.revision,
                    packed_tags=self._encode_run_tags(
                        RunTag(key, value)  # type: ignore[no-untyped-call]
                        for key, value in packed_tags.items()
                    ),
                    packed_params=self._encode_params(packed_params),
                    packed_latest_metrics=self._encode_latest_metrics(latest),
                    items=tuple(items),
                )
            )
            if isinstance(result.outcome, LogBatchBatchLogged):
                return
            if isinstance(result.outcome, LogBatchRunStale):
                continue
            if isinstance(result.outcome, LogBatchRunMissing):
                self._missing_run(run_id)
            if isinstance(result.outcome, LogBatchRunInactive):
                raise MlflowException(
                    f"The run {run_id} must be in the active state.",
                    error_code=INVALID_STATE,
                )
            if isinstance(result.outcome, LogBatchParamConflict):
                old_value = self._decode_params(row.packed_params).get(result.outcome.param_key)
                new_value = next(
                    (
                        param.value
                        for param in normalized_params
                        if param.key == result.outcome.param_key
                    ),
                    None,
                )
                raise MlflowException(
                    "Changing param values is not allowed. Param with "
                    f"key='{result.outcome.param_key}' was already logged with "
                    f"value='{old_value}' for run ID='{run_id}'. Attempted logging "
                    f"new value '{new_value}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            raise MlflowException("Invalid LogBatch response from RiffDB")
        raise MlflowException("Run batch conflicted with another write", error_code=ABORTED)

    def log_inputs(self, run_id: Any, datasets: Any = None, models: Any = None) -> NoReturn:
        self._pending("log_inputs")

    def link_traces_to_run(self, trace_ids: Any, run_id: Any) -> NoReturn:
        self._pending("link_traces_to_run")
