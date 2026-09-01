"""MLflow tracking-store implementation backed by a compiled RiffDB profile."""

from __future__ import annotations

import json
import os
import secrets
from hashlib import sha256
from typing import Any, NoReturn
from uuid import uuid4

from mlflow.entities import Experiment as MLflowExperiment
from mlflow.entities import ExperimentTag, ViewType
from mlflow.entities import LifecycleStage as MLflowLifecycleStage
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
from mlflow.utils.search_utils import SearchExperimentsUtils
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.validation import (
    _validate_experiment_artifact_location_length,
    _validate_experiment_id,
    _validate_experiment_id_type,
    _validate_experiment_name,
    _validate_experiment_tag,
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
    DeleteExperimentExperimentDeleted,
    DeleteExperimentExperimentMissing,
    DeleteExperimentInput,
    DeleteExperimentTagExperimentInactive,
    DeleteExperimentTagExperimentMissing,
    DeleteExperimentTagExperimentStale,
    DeleteExperimentTagExperimentTagDeleted,
    DeleteExperimentTagExperimentTagMissing,
    DeleteExperimentTagInput,
    ExperimentOrder,
    GetExperimentByNameFound,
    GetExperimentByNameParams,
    GetExperimentFound,
    GetExperimentMissing,
    GetExperimentParams,
    LifecycleStage,
    RenameExperimentExperimentMissing,
    RenameExperimentExperimentRenamed,
    RenameExperimentInput,
    RestoreExperimentExperimentMissing,
    RestoreExperimentExperimentRestored,
    RestoreExperimentInput,
    RiffDBMLflowClient,
    SearchExperimentsByNameILikeParams,
    SearchExperimentsByNameLikeParams,
    SearchExperimentsByNameParams,
    SearchExperimentsByTagILikeParams,
    SearchExperimentsByTagLikeParams,
    SearchExperimentsByTagParams,
    SearchExperimentsByTwoTagsParams,
    SearchExperimentsParams,
    SetExperimentTagExperimentInactive,
    SetExperimentTagExperimentMissing,
    SetExperimentTagExperimentStale,
    SetExperimentTagExperimentTagSet,
    SetExperimentTagInput,
)
from riffdb_mlflow.generated.client import (
    ExperimentTag as RiffDBExperimentTag,
)

_SCOPE = "global"
_DEFAULT_EXPERIMENT_ID = 0
_DEFAULT_EXPERIMENT_NAME = "Default"
_PACKED_TAGS_MAX_BYTES = 900_000
_COMPILED_SEARCH_PAGE_SIZE = 50
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
        if not isinstance(result.value, GetExperimentByNameFound):
            raise MlflowException("Invalid GetExperimentByName response from RiffDB")
        if not result.value.experiments:
            return None
        return self._to_mlflow_experiment(result.value.experiments[0])

    def delete_experiment(self, experiment_id: Any) -> None:
        numeric_id = self._numeric_experiment_id(experiment_id)
        result = self._application().delete_experiment(
            DeleteExperimentInput(request_id=str(uuid4()), experiment_id=numeric_id)
        )
        if isinstance(result.outcome, DeleteExperimentExperimentMissing):
            self._missing_experiment(experiment_id)
        if not isinstance(result.outcome, DeleteExperimentExperimentDeleted):
            raise MlflowException("Invalid DeleteExperiment response from RiffDB")

    def restore_experiment(self, experiment_id: Any) -> None:
        numeric_id = self._numeric_experiment_id(experiment_id)
        result = self._application().restore_experiment(
            RestoreExperimentInput(request_id=str(uuid4()), experiment_id=numeric_id)
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
                request_id=str(uuid4()),
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

    def get_run(self, run_id: Any) -> NoReturn:
        self._pending("get_run")

    def update_run_info(
        self, run_id: Any, run_status: Any, end_time: Any, run_name: Any
    ) -> NoReturn:
        self._pending("update_run_info")

    def create_run(
        self,
        experiment_id: Any,
        user_id: Any,
        start_time: Any,
        tags: Any,
        run_name: Any,
    ) -> NoReturn:
        self._pending("create_run")

    def delete_run(self, run_id: Any) -> NoReturn:
        self._pending("delete_run")

    def restore_run(self, run_id: Any) -> NoReturn:
        self._pending("restore_run")

    def get_metric_history(
        self,
        run_id: Any,
        metric_key: Any,
        max_results: Any = None,
        page_token: Any = None,
    ) -> NoReturn:
        self._pending("get_metric_history")

    def _search_runs(
        self,
        experiment_ids: Any,
        filter_string: Any,
        run_view_type: Any,
        max_results: Any,
        order_by: Any,
        page_token: Any,
    ) -> NoReturn:
        self._pending("_search_runs")

    def log_batch(self, run_id: Any, metrics: Any, params: Any, tags: Any) -> NoReturn:
        self._pending("log_batch")

    def log_inputs(self, run_id: Any, datasets: Any = None, models: Any = None) -> NoReturn:
        self._pending("log_inputs")

    def link_traces_to_run(self, trace_ids: Any, run_id: Any) -> NoReturn:
        self._pending("link_traces_to_run")
