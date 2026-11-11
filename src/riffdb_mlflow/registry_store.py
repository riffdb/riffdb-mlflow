"""MLflow model-registry store implementation backed by a compiled RiffDB profile."""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any
from uuid import uuid4

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_stages import (
    ALL_STAGES,
    DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS,
    STAGE_ARCHIVED,
    STAGE_DELETED_INTERNAL,
    STAGE_NONE,
    get_canonical_stage,
)
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.webhook import (
    Webhook,
    WebhookAction,
    WebhookEntity,
    WebhookEvent,
    WebhookStatus,
)
from mlflow.exceptions import MlflowException
from mlflow.prompt.registry_utils import (
    add_prompt_filter_string,
    handle_resource_already_exist_error,
    has_prompt_tag,
)
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.tracking.client import MlflowClient
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    _REGISTERED_MODEL_ALIAS_LATEST,
    _validate_model_alias_name,
    _validate_model_alias_name_reserved,
    _validate_model_name,
    _validate_model_renaming,
    _validate_model_version,
    _validate_model_version_tag,
    _validate_registered_model_tag,
    _validate_tag_name,
    _validate_webhook_events,
    _validate_webhook_name,
    _validate_webhook_url,
)
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME
from riffdb_application import (
    AttemptBudget,
    BearerCredential,
    CallMetadata,
    DatabaseAlias,
    SyncApplicationTransport,
    VerifiedTlsConfig,
)

from riffdb_mlflow.config import ConnectionConfig, StoreConfig
from riffdb_mlflow.generated.client import (
    CreateRegisteredModelEntityInput,
    CreateRegisteredModelEntityRegisteredModelCreated,
    CreateRegisteredModelEntityRegisteredModelExists,
    DeleteModelVersionEntityInput,
    DeleteRegisteredModelEntityInput,
    DeleteRegisteredModelEntityRegisteredModelDeleted,
    DeleteRegisteredModelEntityRegisteredModelMissing,
    GetModelVersionFound,
    GetModelVersionMissing,
    GetModelVersionParams,
    GetRegisteredModelFound,
    GetRegisteredModelMissing,
    GetRegisteredModelParams,
    GetRegistryWebhookFound,
    GetRegistryWebhookMissing,
    GetRegistryWebhookParams,
    ListModelVersionsByModelFound,
    ListModelVersionsByModelParams,
    ListRegistryWebhooksFound,
    ListRegistryWebhooksParams,
    RiffDBMLflowClient,
    SearchModelVersionsFound,
    SearchModelVersionsParams,
    SearchRegisteredModelsFound,
    SearchRegisteredModelsParams,
    UpsertModelVersionEntityInput,
    UpsertModelVersionEntityModelVersionUpserted,
    UpsertRegisteredModelEntityInput,
    UpsertRegisteredModelEntityRegisteredModelUpserted,
    UpsertRegistryWebhookEntityInput,
    UpsertRegistryWebhookEntityRegistryWebhookUpserted,
)

_SCOPE = "global"
_PAGE_BATCH_SIZE = 10
_INTERNAL_DEPLOYMENT_JOB_KEY = "__riffdb_deployment_job_id__"
_INTERNAL_MODEL_ID_KEY = "__riffdb_model_id__"


class RiffDBModelRegistryStore(AbstractStore):
    """Pinned MLflow model registry store backed by one compiled RiffDB application profile."""

    def __init__(self, store_uri: str, tracking_uri: str | None = None) -> None:
        super().__init__(store_uri=store_uri, tracking_uri=tracking_uri)  # type: ignore[no-untyped-call]
        self._config = StoreConfig.from_uri(store_uri)
        self._tracking_uri = tracking_uri or store_uri
        self._transport: SyncApplicationTransport | None = None
        self._client: Any | None = None

    @property
    def database(self) -> str:
        return self._config.database

    def _application(self) -> Any:
        if self._client is None:
            conn = ConnectionConfig.from_env()
            credential = BearerCredential.from_protected_file(conn.credential_file)
            if os.environ.get("RIFFDB_TENANT") == "":
                os.environ.pop("RIFFDB_TENANT", None)
            metadata = CallMetadata.authenticated(credential).with_database(
                DatabaseAlias(self.database)
            )
            if conn.driver_socket is not None:
                self._transport = SyncApplicationTransport.connect_driver_socket(
                    conn.driver_socket, metadata
                )
            else:
                tls = VerifiedTlsConfig(
                    endpoint=conn.endpoint,
                    trust_root=conn.trust_root,
                    server_name=conn.server_name,
                )
                self._transport = SyncApplicationTransport.connect_verified_tls(tls, metadata)
            self._client = RiffDBMLflowClient(self._transport, AttemptBudget(3))
        return self._client

    @staticmethod
    def _encode_json_dict(mapping: dict[str, Any]) -> bytes:
        return json.dumps(
            mapping, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")

    @staticmethod
    def _decode_json_dict(raw: bytes) -> dict[str, Any]:
        if not raw:
            return {}
        val = json.loads(raw.decode("utf-8"))
        return dict(val) if isinstance(val, dict) else {}

    @staticmethod
    def _encode_json_list(items: list[Any]) -> bytes:
        return json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _decode_json_list(raw: bytes) -> list[Any]:
        if not raw:
            return []
        val = json.loads(raw.decode("utf-8"))
        return list(val) if isinstance(val, list) else []

    def _list_version_rows_for_model(
        self, client: Any, name: str, include_deleted: bool = False
    ) -> list[Any]:
        rows: list[Any] = []
        cursor: str | None = None
        while True:
            res = client.list_model_versions_by_model(
                ListModelVersionsByModelParams(
                    scope=_SCOPE,
                    name=name,
                    limit=_PAGE_BATCH_SIZE,
                    after=cursor,
                )
            )
            if not isinstance(res.value, ListModelVersionsByModelFound):
                raise MlflowException("Invalid ListModelVersionsByModel response from RiffDB")
            for row in res.value.versions:
                if include_deleted or row.current_stage != STAGE_DELETED_INTERNAL:
                    rows.append(row)
            cursor = res.next_cursor
            if not cursor or not res.value.versions:
                break
        return rows

    def _to_mlflow_model_version(
        self, row: Any, aliases_map: dict[str, int] | None = None
    ) -> ModelVersion:
        tags_dict = self._decode_json_dict(row.packed_tags)
        model_id = tags_dict.pop(_INTERNAL_MODEL_ID_KEY, None)
        tags = [
            ModelVersionTag(key=str(k), value=str(v))  # type: ignore[no-untyped-call]
            for k, v in tags_dict.items()
        ]
        version_num = int(row.version_num)
        aliases: list[str] = []
        if aliases_map:
            aliases = [
                str(alias)
                for alias, target_ver in aliases_map.items()
                if int(target_ver) == version_num
            ]
        return ModelVersion(
            name=row.name,
            version=version_num,  # type: ignore[arg-type]
            creation_timestamp=int(row.creation_timestamp_ms),
            last_updated_timestamp=int(row.last_updated_timestamp_ms),
            description=row.description,
            user_id=row.user_id,
            current_stage=row.current_stage,
            source=row.source_path,
            run_id=row.run_id,
            status=row.status,
            status_message=row.status_message,
            tags=tags,
            run_link=row.run_link,
            aliases=aliases,
            workspace=DEFAULT_WORKSPACE_NAME,
            model_id=model_id,
        )

    def _to_mlflow_registered_model(
        self, client: Any, row: Any, version_rows: list[Any] | None = None
    ) -> RegisteredModel:
        if version_rows is None:
            version_rows = self._list_version_rows_for_model(client, row.name)
        aliases_map = self._decode_json_dict(row.packed_aliases)
        latest_by_stage: dict[str, Any] = {}
        for v_row in version_rows:
            stage = v_row.current_stage
            if stage != STAGE_DELETED_INTERNAL and (
                stage not in latest_by_stage
                or int(latest_by_stage[stage].version_num) < int(v_row.version_num)
            ):
                latest_by_stage[stage] = v_row
        latest_versions = [
            self._to_mlflow_model_version(v_row, aliases_map=aliases_map)
            for v_row in latest_by_stage.values()
        ]
        tags_dict = self._decode_json_dict(row.packed_tags)
        deployment_job_id = tags_dict.pop(_INTERNAL_DEPLOYMENT_JOB_KEY, None)
        tags = [
            RegisteredModelTag(key=str(k), value=str(v))  # type: ignore[no-untyped-call]
            for k, v in tags_dict.items()
        ]
        aliases = [
            RegisteredModelAlias(alias=str(alias), version=int(ver))  # type: ignore[no-untyped-call]
            for alias, ver in aliases_map.items()
        ]
        return RegisteredModel(
            row.name,
            int(row.creation_timestamp_ms),
            int(row.last_updated_timestamp_ms),
            row.description,
            latest_versions,
            tags,
            aliases,
            deployment_job_id,
            workspace=DEFAULT_WORKSPACE_NAME,
        )

    def _get_registered_model_row(self, client: Any, name: str) -> Any:
        _validate_model_name(name)
        res = client.get_registered_model(GetRegisteredModelParams(scope=_SCOPE, name=name))
        if isinstance(res.value, GetRegisteredModelMissing) or not isinstance(
            res.value, GetRegisteredModelFound
        ):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return res.value.model

    def _get_model_version_row(
        self, client: Any, name: str, version: Any, include_deleted: bool = False
    ) -> tuple[Any, int]:
        _validate_model_name(name)
        _validate_model_version(version)  # type: ignore[no-untyped-call]
        version_num = int(version)
        res = client.get_model_version(
            GetModelVersionParams(scope=_SCOPE, name=name, version_num=version_num)
        )
        if isinstance(res.value, GetModelVersionMissing) or not isinstance(
            res.value, GetModelVersionFound
        ):
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.model_version
        if not include_deleted and row.current_stage == STAGE_DELETED_INTERNAL:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return row, version_num

    # ------------------------------------------------------------------
    # RegisteredModel CRUD
    # ------------------------------------------------------------------

    def create_registered_model(
        self,
        name: Any,
        tags: Any = None,
        description: Any = None,
        deployment_job_id: Any = None,
    ) -> RegisteredModel:
        _validate_model_name(name)
        tags_dict: dict[str, str] = {}
        for tag in tags or []:
            _validate_registered_model_tag(tag.key, tag.value)  # type: ignore[no-untyped-call]
            tags_dict[str(tag.key)] = str(tag.value)
        if deployment_job_id is not None:
            tags_dict[_INTERNAL_DEPLOYMENT_JOB_KEY] = str(deployment_job_id)

        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        client = self._application()
        res = client.create_registered_model_entity(
            CreateRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=name,
                creation_timestamp_ms=now_ms,
                last_updated_timestamp_ms=now_ms,
                description=description,
                packed_tags=self._encode_json_dict(tags_dict),
                packed_aliases=self._encode_json_dict({}),
                latest_version=0,
            )
        )
        if isinstance(res.outcome, CreateRegisteredModelEntityRegisteredModelExists):
            existing_model = self.get_registered_model(name)
            handle_resource_already_exist_error(
                name,
                has_prompt_tag(existing_model._tags),
                has_prompt_tag(tags),
            )
        if not isinstance(res.outcome, CreateRegisteredModelEntityRegisteredModelCreated):
            raise MlflowException("Invalid CreateRegisteredModelEntity response from RiffDB")
        return self._to_mlflow_registered_model(client, res.outcome.model, version_rows=[])

    def update_registered_model(
        self, name: Any, description: Any, deployment_job_id: Any = None
    ) -> RegisteredModel:
        client = self._application()
        row = self._get_registered_model_row(client, name)
        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        tags_dict = self._decode_json_dict(row.packed_tags)
        if deployment_job_id is not None:
            if deployment_job_id == "":
                tags_dict.pop(_INTERNAL_DEPLOYMENT_JOB_KEY, None)
            else:
                tags_dict[_INTERNAL_DEPLOYMENT_JOB_KEY] = str(deployment_job_id)
        res = client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=row.name,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=description,
                packed_tags=self._encode_json_dict(tags_dict),
                packed_aliases=row.packed_aliases,
                latest_version=int(row.latest_version),
            )
        )
        if not isinstance(res.outcome, UpsertRegisteredModelEntityRegisteredModelUpserted):
            raise MlflowException("Invalid UpsertRegisteredModelEntity response from RiffDB")
        return self._to_mlflow_registered_model(client, res.outcome.model)

    def rename_registered_model(self, name: Any, new_name: Any) -> RegisteredModel:
        _validate_model_renaming(new_name)
        client = self._application()
        row = self._get_registered_model_row(client, name)
        if name == new_name:
            return self._to_mlflow_registered_model(client, row)

        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        create_res = client.create_registered_model_entity(
            CreateRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=new_name,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=row.description,
                packed_tags=row.packed_tags,
                packed_aliases=row.packed_aliases,
                latest_version=int(row.latest_version),
            )
        )
        if isinstance(create_res.outcome, CreateRegisteredModelEntityRegisteredModelExists):
            raise MlflowException(
                f"Registered Model (name={new_name}) already exists.",
                RESOURCE_ALREADY_EXISTS,
            )
        if not isinstance(create_res.outcome, CreateRegisteredModelEntityRegisteredModelCreated):
            raise MlflowException("Invalid CreateRegisteredModelEntity response from RiffDB")

        version_rows = self._list_version_rows_for_model(client, name, include_deleted=True)
        new_version_rows: list[Any] = []
        for v_row in version_rows:
            up_res = client.upsert_model_version_entity(
                UpsertModelVersionEntityInput(
                    request_id=str(uuid4()),
                    name=new_name,
                    version_num=int(v_row.version_num),
                    creation_timestamp_ms=int(v_row.creation_timestamp_ms),
                    last_updated_timestamp_ms=now_ms,
                    description=v_row.description,
                    user_id=v_row.user_id,
                    current_stage=v_row.current_stage,
                    source_path=v_row.source_path,
                    storage_location=v_row.storage_location,
                    run_id=v_row.run_id,
                    run_link=v_row.run_link,
                    status=v_row.status,
                    status_message=v_row.status_message,
                    packed_tags=v_row.packed_tags,
                )
            )
            if not isinstance(up_res.outcome, UpsertModelVersionEntityModelVersionUpserted):
                raise MlflowException("Invalid UpsertModelVersionEntity response from RiffDB")
            if up_res.outcome.model_version.current_stage != STAGE_DELETED_INTERNAL:
                new_version_rows.append(up_res.outcome.model_version)
            client.delete_model_version_entity(
                DeleteModelVersionEntityInput(
                    request_id=str(uuid4()),
                    name=name,
                    version_num=int(v_row.version_num),
                )
            )

        client.delete_registered_model_entity(
            DeleteRegisteredModelEntityInput(request_id=str(uuid4()), name=name)
        )
        return self._to_mlflow_registered_model(
            client, create_res.outcome.model, version_rows=new_version_rows
        )

    def delete_registered_model(self, name: Any) -> None:
        client = self._application()
        self._get_registered_model_row(client, name)
        version_rows = self._list_version_rows_for_model(client, name, include_deleted=True)
        for v_row in version_rows:
            client.delete_model_version_entity(
                DeleteModelVersionEntityInput(
                    request_id=str(uuid4()),
                    name=name,
                    version_num=int(v_row.version_num),
                )
            )
        res = client.delete_registered_model_entity(
            DeleteRegisteredModelEntityInput(request_id=str(uuid4()), name=name)
        )
        if isinstance(res.outcome, DeleteRegisteredModelEntityRegisteredModelMissing):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteRegisteredModelEntityRegisteredModelDeleted):
            raise MlflowException("Invalid DeleteRegisteredModelEntity response from RiffDB")

    def get_registered_model(self, name: Any) -> RegisteredModel:
        client = self._application()
        row = self._get_registered_model_row(client, name)
        return self._to_mlflow_registered_model(client, row)

    def search_registered_models(
        self,
        filter_string: Any = None,
        max_results: Any = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        order_by: Any = None,
        page_token: Any = None,
    ) -> PagedList[RegisteredModel]:
        if max_results is None:
            max_results = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer, "
                f"but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        effective_filter = add_prompt_filter_string(filter_string, is_prompt=False)
        SearchModelUtils.parse_search_filter(effective_filter)  # type: ignore[no-untyped-call]
        SearchModelUtils._get_sort_key(order_by or [])  # type: ignore[no-untyped-call]
        start_offset = SearchUtils.parse_start_offset_from_page_token(  # type: ignore[no-untyped-call]
            page_token
        )

        client = self._application()
        models: list[RegisteredModel] = []
        cursor: str | None = None
        while True:
            res = client.search_registered_models(
                SearchRegisteredModelsParams(
                    scope=_SCOPE,
                    limit=_PAGE_BATCH_SIZE,
                    after=cursor,
                )
            )
            if not isinstance(res.value, SearchRegisteredModelsFound):
                raise MlflowException("Invalid SearchRegisteredModels response from RiffDB")
            for row in res.value.models:
                models.append(self._to_mlflow_registered_model(client, row))
            cursor = res.next_cursor
            if not cursor or not res.value.models:
                break

        filtered = SearchModelUtils.filter(models, effective_filter)  # type: ignore[no-untyped-call]
        sorted_rms = SearchModelUtils.sort(filtered, order_by)  # type: ignore[no-untyped-call]
        final_offset = start_offset + max_results
        paginated = sorted_rms[start_offset:final_offset]
        next_token = (
            SearchUtils.create_page_token(final_offset)  # type: ignore[no-untyped-call]
            if final_offset < len(sorted_rms)
            else None
        )
        return PagedList(paginated, next_token)

    def get_latest_versions(self, name: Any, stages: Any = None) -> list[ModelVersion]:
        rm = self.get_registered_model(name)
        if stages is None or len(stages) == 0:
            expected_stages = {get_canonical_stage(stage) for stage in ALL_STAGES}  # type: ignore[no-untyped-call]
        else:
            expected_stages = {get_canonical_stage(stage) for stage in stages}  # type: ignore[no-untyped-call]
        return [mv for mv in rm.latest_versions if mv.current_stage in expected_stages]

    def set_registered_model_tag(self, name: Any, tag: Any) -> None:
        _validate_model_name(name)
        _validate_registered_model_tag(tag.key, tag.value)  # type: ignore[no-untyped-call]
        client = self._application()
        row = self._get_registered_model_row(client, name)
        tags_dict = self._decode_json_dict(row.packed_tags)
        tags_dict[str(tag.key)] = str(tag.value)
        res = client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=row.name,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=int(row.last_updated_timestamp_ms),
                description=row.description,
                packed_tags=self._encode_json_dict(tags_dict),
                packed_aliases=row.packed_aliases,
                latest_version=int(row.latest_version),
            )
        )
        if not isinstance(res.outcome, UpsertRegisteredModelEntityRegisteredModelUpserted):
            raise MlflowException("Invalid UpsertRegisteredModelEntity response from RiffDB")

    def delete_registered_model_tag(self, name: Any, key: Any) -> None:
        _validate_model_name(name)
        _validate_tag_name(key)  # type: ignore[no-untyped-call]
        client = self._application()
        row = self._get_registered_model_row(client, name)
        tags_dict = self._decode_json_dict(row.packed_tags)
        tags_dict.pop(str(key), None)
        res = client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=row.name,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=int(row.last_updated_timestamp_ms),
                description=row.description,
                packed_tags=self._encode_json_dict(tags_dict),
                packed_aliases=row.packed_aliases,
                latest_version=int(row.latest_version),
            )
        )
        if not isinstance(res.outcome, UpsertRegisteredModelEntityRegisteredModelUpserted):
            raise MlflowException("Invalid UpsertRegisteredModelEntity response from RiffDB")

    # ------------------------------------------------------------------
    # ModelVersion CRUD
    # ------------------------------------------------------------------

    def create_model_version(
        self,
        name: Any,
        source: Any,
        run_id: Any = None,
        tags: Any = None,
        run_link: Any = None,
        description: Any = None,
        local_model_path: Any = None,
        model_id: str | None = None,
    ) -> ModelVersion:
        del local_model_path
        _validate_model_name(name)
        tags_dict: dict[str, str] = {}
        for tag in tags or []:
            _validate_model_version_tag(tag.key, tag.value)  # type: ignore[no-untyped-call]
            tags_dict[str(tag.key)] = str(tag.value)
        if model_id is not None:
            tags_dict[_INTERNAL_MODEL_ID_KEY] = str(model_id)

        storage_location = str(source or "")
        if source and urllib.parse.urlparse(str(source)).scheme == "models":
            parsed_model_uri = _parse_model_uri(str(source))
            try:
                if parsed_model_uri.model_id is not None:
                    logged_model = MlflowClient(tracking_uri=self._tracking_uri).get_logged_model(
                        parsed_model_uri.model_id
                    )
                    storage_location = logged_model.artifact_location
                    run_id = run_id or logged_model.source_run_id
                else:
                    storage_location = self.get_model_version_download_uri(
                        parsed_model_uri.name, parsed_model_uri.version
                    )
            except Exception as error:
                raise MlflowException(
                    f"Unable to fetch model from model URI source artifact location '{source}'. "
                    f"Error: {error}"
                ) from error

        if not run_id and model_id:
            try:
                logged_model = MlflowClient(tracking_uri=self._tracking_uri).get_logged_model(
                    model_id
                )
                run_id = logged_model.source_run_id
            except Exception:
                pass

        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        next_version = int(rm_row.latest_version) + 1
        status_str = ModelVersionStatus.to_string(ModelVersionStatus.READY)  # type: ignore[no-untyped-call]

        up_rm = client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=rm_row.name,
                creation_timestamp_ms=int(rm_row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=rm_row.description,
                packed_tags=rm_row.packed_tags,
                packed_aliases=rm_row.packed_aliases,
                latest_version=next_version,
            )
        )
        if not isinstance(up_rm.outcome, UpsertRegisteredModelEntityRegisteredModelUpserted):
            raise MlflowException("Invalid UpsertRegisteredModelEntity response from RiffDB")

        up_mv = client.upsert_model_version_entity(
            UpsertModelVersionEntityInput(
                request_id=str(uuid4()),
                name=name,
                version_num=next_version,
                creation_timestamp_ms=now_ms,
                last_updated_timestamp_ms=now_ms,
                description=description,
                user_id=None,
                current_stage=STAGE_NONE,
                source_path=str(source or ""),
                storage_location=storage_location,
                run_id=run_id,
                run_link=run_link,
                status=status_str,
                status_message=None,
                packed_tags=self._encode_json_dict(tags_dict),
            )
        )
        if not isinstance(up_mv.outcome, UpsertModelVersionEntityModelVersionUpserted):
            raise MlflowException("Invalid UpsertModelVersionEntity response from RiffDB")

        aliases_map = self._decode_json_dict(rm_row.packed_aliases)
        return self._to_mlflow_model_version(up_mv.outcome.model_version, aliases_map=aliases_map)

    def update_model_version(
        self, name: Any, version: Any, description: Any = None
    ) -> ModelVersion:
        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        v_row, version_num = self._get_model_version_row(client, name, version)
        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        res = client.upsert_model_version_entity(
            UpsertModelVersionEntityInput(
                request_id=str(uuid4()),
                name=name,
                version_num=version_num,
                creation_timestamp_ms=int(v_row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=description,
                user_id=v_row.user_id,
                current_stage=v_row.current_stage,
                source_path=v_row.source_path,
                storage_location=v_row.storage_location,
                run_id=v_row.run_id,
                run_link=v_row.run_link,
                status=v_row.status,
                status_message=v_row.status_message,
                packed_tags=v_row.packed_tags,
            )
        )
        if not isinstance(res.outcome, UpsertModelVersionEntityModelVersionUpserted):
            raise MlflowException("Invalid UpsertModelVersionEntity response from RiffDB")
        aliases_map = self._decode_json_dict(rm_row.packed_aliases)
        return self._to_mlflow_model_version(res.outcome.model_version, aliases_map=aliases_map)

    def transition_model_version_stage(
        self, name: Any, version: Any, stage: Any, archive_existing_versions: Any
    ) -> ModelVersion:
        canonical_stage = get_canonical_stage(stage)  # type: ignore[no-untyped-call]
        is_active_stage = canonical_stage in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        if archive_existing_versions and not is_active_stage:
            raise MlflowException(
                f"Model version transition cannot archive existing model versions "
                f"because '{stage}' is not an Active stage. Valid stages are "
                f"{DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS}"
            )

        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        target_row, version_num = self._get_model_version_row(client, name, version)
        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]

        if archive_existing_versions:
            for existing in self._list_version_rows_for_model(client, name):
                if (
                    int(existing.version_num) != version_num
                    and existing.current_stage == canonical_stage
                ):
                    client.upsert_model_version_entity(
                        UpsertModelVersionEntityInput(
                            request_id=str(uuid4()),
                            name=name,
                            version_num=int(existing.version_num),
                            creation_timestamp_ms=int(existing.creation_timestamp_ms),
                            last_updated_timestamp_ms=now_ms,
                            description=existing.description,
                            user_id=existing.user_id,
                            current_stage=STAGE_ARCHIVED,
                            source_path=existing.source_path,
                            storage_location=existing.storage_location,
                            run_id=existing.run_id,
                            run_link=existing.run_link,
                            status=existing.status,
                            status_message=existing.status_message,
                            packed_tags=existing.packed_tags,
                        )
                    )

        up_mv = client.upsert_model_version_entity(
            UpsertModelVersionEntityInput(
                request_id=str(uuid4()),
                name=name,
                version_num=version_num,
                creation_timestamp_ms=int(target_row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=target_row.description,
                user_id=target_row.user_id,
                current_stage=canonical_stage,
                source_path=target_row.source_path,
                storage_location=target_row.storage_location,
                run_id=target_row.run_id,
                run_link=target_row.run_link,
                status=target_row.status,
                status_message=target_row.status_message,
                packed_tags=target_row.packed_tags,
            )
        )
        if not isinstance(up_mv.outcome, UpsertModelVersionEntityModelVersionUpserted):
            raise MlflowException("Invalid UpsertModelVersionEntity response from RiffDB")

        client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=rm_row.name,
                creation_timestamp_ms=int(rm_row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=rm_row.description,
                packed_tags=rm_row.packed_tags,
                packed_aliases=rm_row.packed_aliases,
                latest_version=int(rm_row.latest_version),
            )
        )
        aliases_map = self._decode_json_dict(rm_row.packed_aliases)
        return self._to_mlflow_model_version(up_mv.outcome.model_version, aliases_map=aliases_map)

    def delete_model_version(self, name: Any, version: Any) -> None:
        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        v_row, version_num = self._get_model_version_row(client, name, version)
        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]

        aliases_map = {
            k: v
            for k, v in self._decode_json_dict(rm_row.packed_aliases).items()
            if int(v) != version_num
        }
        client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=rm_row.name,
                creation_timestamp_ms=int(rm_row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=rm_row.description,
                packed_tags=rm_row.packed_tags,
                packed_aliases=self._encode_json_dict(aliases_map),
                latest_version=int(rm_row.latest_version),
            )
        )
        res = client.upsert_model_version_entity(
            UpsertModelVersionEntityInput(
                request_id=str(uuid4()),
                name=name,
                version_num=version_num,
                creation_timestamp_ms=int(v_row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                description=None,
                user_id=None,
                current_stage=STAGE_DELETED_INTERNAL,
                source_path="REDACTED-SOURCE-PATH",
                storage_location=v_row.storage_location,
                run_id="REDACTED-RUN-ID",
                run_link="REDACTED-RUN-LINK",
                status=v_row.status,
                status_message=None,
                packed_tags=v_row.packed_tags,
            )
        )
        if not isinstance(res.outcome, UpsertModelVersionEntityModelVersionUpserted):
            raise MlflowException("Invalid UpsertModelVersionEntity response from RiffDB")

    def get_model_version(self, name: Any, version: Any) -> ModelVersion:
        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        v_row, _ = self._get_model_version_row(client, name, version)
        aliases_map = self._decode_json_dict(rm_row.packed_aliases)
        return self._to_mlflow_model_version(v_row, aliases_map=aliases_map)

    def get_model_version_download_uri(self, name: Any, version: Any) -> str:
        from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

        client = self._application()
        v_row, _ = self._get_model_version_row(client, name, version)
        uri = str(v_row.storage_location or v_row.source_path)
        if RunsArtifactRepository.is_runs_uri(uri):  # type: ignore[no-untyped-call]
            try:
                return str(RunsArtifactRepository.get_underlying_uri(uri))
            except Exception:
                return uri
        return uri

    def search_model_versions(
        self,
        filter_string: Any = None,
        max_results: Any = SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        order_by: Any = None,
        page_token: Any = None,
    ) -> PagedList[ModelVersion]:
        if max_results is None:
            max_results = SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer, "
                f"but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        effective_filter = add_prompt_filter_string(filter_string, is_prompt=False)
        effective_order_by = order_by or [
            "last_updated_timestamp DESC",
            "name ASC",
            "version_number DESC",
        ]
        SearchModelVersionUtils.parse_search_filter(effective_filter)  # type: ignore[no-untyped-call]
        SearchModelVersionUtils._get_sort_key(effective_order_by)  # type: ignore[no-untyped-call]
        start_offset = SearchUtils.parse_start_offset_from_page_token(  # type: ignore[no-untyped-call]
            page_token
        )

        client = self._application()
        versions: list[ModelVersion] = []
        cursor: str | None = None
        while True:
            res = client.search_model_versions(
                SearchModelVersionsParams(
                    scope=_SCOPE,
                    limit=_PAGE_BATCH_SIZE,
                    after=cursor,
                )
            )
            if not isinstance(res.value, SearchModelVersionsFound):
                raise MlflowException("Invalid SearchModelVersions response from RiffDB")
            for row in res.value.versions:
                if row.current_stage != STAGE_DELETED_INTERNAL:
                    versions.append(self._to_mlflow_model_version(row))
            cursor = res.next_cursor
            if not cursor or not res.value.versions:
                break

        filtered = SearchModelVersionUtils.filter(versions, effective_filter)  # type: ignore[no-untyped-call]
        sorted_mvs = SearchModelVersionUtils.sort(  # type: ignore[no-untyped-call]
            filtered, effective_order_by
        )
        final_offset = start_offset + max_results
        paginated = sorted_mvs[start_offset:final_offset]
        next_token = (
            SearchUtils.create_page_token(final_offset)  # type: ignore[no-untyped-call]
            if final_offset < len(sorted_mvs)
            else None
        )
        return PagedList(paginated, next_token)

    def set_model_version_tag(self, name: Any, version: Any, tag: Any) -> None:
        _validate_model_name(name)
        _validate_model_version(version)  # type: ignore[no-untyped-call]
        _validate_model_version_tag(tag.key, tag.value)  # type: ignore[no-untyped-call]
        client = self._application()
        v_row, version_num = self._get_model_version_row(client, name, version)
        tags_dict = self._decode_json_dict(v_row.packed_tags)
        tags_dict[str(tag.key)] = str(tag.value)
        res = client.upsert_model_version_entity(
            UpsertModelVersionEntityInput(
                request_id=str(uuid4()),
                name=name,
                version_num=version_num,
                creation_timestamp_ms=int(v_row.creation_timestamp_ms),
                last_updated_timestamp_ms=int(v_row.last_updated_timestamp_ms),
                description=v_row.description,
                user_id=v_row.user_id,
                current_stage=v_row.current_stage,
                source_path=v_row.source_path,
                storage_location=v_row.storage_location,
                run_id=v_row.run_id,
                run_link=v_row.run_link,
                status=v_row.status,
                status_message=v_row.status_message,
                packed_tags=self._encode_json_dict(tags_dict),
            )
        )
        if not isinstance(res.outcome, UpsertModelVersionEntityModelVersionUpserted):
            raise MlflowException("Invalid UpsertModelVersionEntity response from RiffDB")

    def delete_model_version_tag(self, name: Any, version: Any, key: Any) -> None:
        _validate_model_name(name)
        _validate_model_version(version)  # type: ignore[no-untyped-call]
        _validate_tag_name(key)  # type: ignore[no-untyped-call]
        client = self._application()
        v_row, version_num = self._get_model_version_row(client, name, version)
        tags_dict = self._decode_json_dict(v_row.packed_tags)
        tags_dict.pop(str(key), None)
        res = client.upsert_model_version_entity(
            UpsertModelVersionEntityInput(
                request_id=str(uuid4()),
                name=name,
                version_num=version_num,
                creation_timestamp_ms=int(v_row.creation_timestamp_ms),
                last_updated_timestamp_ms=int(v_row.last_updated_timestamp_ms),
                description=v_row.description,
                user_id=v_row.user_id,
                current_stage=v_row.current_stage,
                source_path=v_row.source_path,
                storage_location=v_row.storage_location,
                run_id=v_row.run_id,
                run_link=v_row.run_link,
                status=v_row.status,
                status_message=v_row.status_message,
                packed_tags=self._encode_json_dict(tags_dict),
            )
        )
        if not isinstance(res.outcome, UpsertModelVersionEntityModelVersionUpserted):
            raise MlflowException("Invalid UpsertModelVersionEntity response from RiffDB")

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------

    def set_registered_model_alias(self, name: Any, alias: Any, version: Any) -> None:
        _validate_model_name(name)
        _validate_model_alias_name(alias)  # type: ignore[no-untyped-call]
        _validate_model_alias_name_reserved(alias)  # type: ignore[no-untyped-call]
        _validate_model_version(version)  # type: ignore[no-untyped-call]
        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        _, version_num = self._get_model_version_row(client, name, version)
        aliases_map = self._decode_json_dict(rm_row.packed_aliases)
        aliases_map[str(alias)] = version_num
        res = client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=rm_row.name,
                creation_timestamp_ms=int(rm_row.creation_timestamp_ms),
                last_updated_timestamp_ms=int(rm_row.last_updated_timestamp_ms),
                description=rm_row.description,
                packed_tags=rm_row.packed_tags,
                packed_aliases=self._encode_json_dict(aliases_map),
                latest_version=int(rm_row.latest_version),
            )
        )
        if not isinstance(res.outcome, UpsertRegisteredModelEntityRegisteredModelUpserted):
            raise MlflowException("Invalid UpsertRegisteredModelEntity response from RiffDB")

    def delete_registered_model_alias(self, name: Any, alias: Any) -> None:
        _validate_model_name(name)
        _validate_model_alias_name(alias)  # type: ignore[no-untyped-call]
        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        aliases_map = self._decode_json_dict(rm_row.packed_aliases)
        aliases_map.pop(str(alias), None)
        res = client.upsert_registered_model_entity(
            UpsertRegisteredModelEntityInput(
                request_id=str(uuid4()),
                name=rm_row.name,
                creation_timestamp_ms=int(rm_row.creation_timestamp_ms),
                last_updated_timestamp_ms=int(rm_row.last_updated_timestamp_ms),
                description=rm_row.description,
                packed_tags=rm_row.packed_tags,
                packed_aliases=self._encode_json_dict(aliases_map),
                latest_version=int(rm_row.latest_version),
            )
        )
        if not isinstance(res.outcome, UpsertRegisteredModelEntityRegisteredModelUpserted):
            raise MlflowException("Invalid UpsertRegisteredModelEntity response from RiffDB")

    def get_model_version_by_alias(self, name: Any, alias: Any) -> ModelVersion:
        _validate_model_name(name)
        _validate_model_alias_name(alias)  # type: ignore[no-untyped-call]
        if str(alias).lower() == _REGISTERED_MODEL_ALIAS_LATEST:
            if versions := self.get_latest_versions(name):
                return versions[0]
            raise MlflowException(
                f"Latest version not found for model {name}.",
                RESOURCE_DOES_NOT_EXIST,
            )

        client = self._application()
        rm_row = self._get_registered_model_row(client, name)
        aliases_map = self._decode_json_dict(rm_row.packed_aliases)
        if str(alias) not in aliases_map:
            raise MlflowException(
                f"Registered model alias {alias} not found.",
                INVALID_PARAMETER_VALUE,
            )
        v_row, _ = self._get_model_version_row(client, name, int(aliases_map[str(alias)]))
        return self._to_mlflow_model_version(v_row, aliases_map=aliases_map)

    def _await_model_version_creation(self, mv: Any, await_creation_for: Any) -> None:
        del mv, await_creation_for

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------

    def _to_mlflow_webhook(self, row: Any) -> Webhook:
        raw_events = self._decode_json_list(row.packed_events)
        events = [
            WebhookEvent(
                entity=WebhookEntity(str(item["entity"])),
                action=WebhookAction(str(item["action"])),
            )
            for item in raw_events
        ]
        return Webhook(
            webhook_id=row.webhook_id,
            name=row.name,
            url=row.url,
            events=events,
            creation_timestamp=int(row.creation_timestamp_ms),
            last_updated_timestamp=int(row.last_updated_timestamp_ms),
            workspace=DEFAULT_WORKSPACE_NAME,
            description=row.description,
            status=WebhookStatus(row.status),
            secret=row.secret,
        )

    def _get_webhook_row(self, client: Any, webhook_id: str) -> Any:
        res = client.get_registry_webhook(
            GetRegistryWebhookParams(scope=_SCOPE, webhook_id=webhook_id)
        )
        if isinstance(res.value, GetRegistryWebhookMissing) or not isinstance(
            res.value, GetRegistryWebhookFound
        ):
            raise MlflowException(
                f"Webhook with ID {webhook_id} not found.",
                RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.webhook
        if row.deleted_timestamp_ms is not None:
            raise MlflowException(
                f"Webhook with ID {webhook_id} not found.",
                RESOURCE_DOES_NOT_EXIST,
            )
        return row

    def create_webhook(
        self,
        name: str,
        url: str,
        events: list[WebhookEvent],
        description: str | None = None,
        secret: str | None = None,
        status: WebhookStatus | None = None,
    ) -> Webhook:
        _validate_webhook_name(name)
        _validate_webhook_url(url)
        _validate_webhook_events(events)

        webhook_id = str(uuid4())
        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        status_val = (status or WebhookStatus.ACTIVE).value
        packed_events = self._encode_json_list(
            [{"entity": e.entity.value, "action": e.action.value} for e in events]
        )
        client = self._application()
        res = client.upsert_registry_webhook_entity(
            UpsertRegistryWebhookEntityInput(
                request_id=str(uuid4()),
                webhook_id=webhook_id,
                name=name,
                description=description,
                url=url,
                status=status_val,
                secret=secret,
                creation_timestamp_ms=now_ms,
                last_updated_timestamp_ms=now_ms,
                deleted_timestamp_ms=None,
                packed_events=packed_events,
            )
        )
        if not isinstance(res.outcome, UpsertRegistryWebhookEntityRegistryWebhookUpserted):
            raise MlflowException("Invalid UpsertRegistryWebhookEntity response from RiffDB")
        return self._to_mlflow_webhook(res.outcome.webhook)

    def get_webhook(self, webhook_id: str) -> Webhook:
        client = self._application()
        row = self._get_webhook_row(client, webhook_id)
        return self._to_mlflow_webhook(row)

    def _list_active_webhooks(self, client: Any) -> list[Webhook]:
        webhooks: list[Webhook] = []
        cursor: str | None = None
        while True:
            res = client.list_registry_webhooks(
                ListRegistryWebhooksParams(
                    scope=_SCOPE,
                    limit=_PAGE_BATCH_SIZE,
                    after=cursor,
                )
            )
            if not isinstance(res.value, ListRegistryWebhooksFound):
                raise MlflowException("Invalid ListRegistryWebhooks response from RiffDB")
            for row in res.value.webhooks:
                if row.deleted_timestamp_ms is None:
                    webhooks.append(self._to_mlflow_webhook(row))
            cursor = res.next_cursor
            if not cursor or not res.value.webhooks:
                break
        return webhooks

    def list_webhooks(
        self,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Webhook]:
        resolved_max = 100 if max_results is None else max_results
        if (
            isinstance(resolved_max, bool)
            or not isinstance(resolved_max, int)
            or resolved_max < 1
            or resolved_max > 1000
        ):
            raise MlflowException(
                "max_results must be between 1 and 1000.",
                INVALID_PARAMETER_VALUE,
            )
        offset = SearchUtils.parse_start_offset_from_page_token(  # type: ignore[no-untyped-call]
            page_token
        )
        client = self._application()
        all_webhooks = self._list_active_webhooks(client)
        sliced = all_webhooks[offset : offset + resolved_max]
        next_token = (
            SearchUtils.create_page_token(offset + resolved_max)  # type: ignore[no-untyped-call]
            if offset + resolved_max < len(all_webhooks)
            else None
        )
        return PagedList(sliced, next_token)

    def list_webhooks_by_event(
        self,
        event: WebhookEvent,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Webhook]:
        resolved_max = 100 if max_results is None else max_results
        if (
            isinstance(resolved_max, bool)
            or not isinstance(resolved_max, int)
            or resolved_max < 1
            or resolved_max > 1000
        ):
            raise MlflowException(
                "max_results must be between 1 and 1000.",
                INVALID_PARAMETER_VALUE,
            )
        offset = SearchUtils.parse_start_offset_from_page_token(  # type: ignore[no-untyped-call]
            page_token
        )
        client = self._application()
        matching = [
            w
            for w in self._list_active_webhooks(client)
            if any(
                we.entity.value == event.entity.value and we.action.value == event.action.value
                for we in w.events
            )
        ]
        sliced = matching[offset : offset + resolved_max]
        next_token = (
            SearchUtils.create_page_token(offset + resolved_max)  # type: ignore[no-untyped-call]
            if offset + resolved_max < len(matching)
            else None
        )
        return PagedList(sliced, next_token)

    def update_webhook(
        self,
        webhook_id: str,
        name: str | None = None,
        description: str | None = None,
        url: str | None = None,
        events: list[WebhookEvent] | None = None,
        secret: str | None = None,
        status: WebhookStatus | None = None,
    ) -> Webhook:
        client = self._application()
        row = self._get_webhook_row(client, webhook_id)
        if name is not None:
            _validate_webhook_name(name)
        if url is not None:
            _validate_webhook_url(url)
        if events is not None:
            _validate_webhook_events(events)
            packed_events = self._encode_json_list(
                [{"entity": e.entity.value, "action": e.action.value} for e in events]
            )
        else:
            packed_events = row.packed_events

        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        res = client.upsert_registry_webhook_entity(
            UpsertRegistryWebhookEntityInput(
                request_id=str(uuid4()),
                webhook_id=webhook_id,
                name=name if name is not None else row.name,
                description=description if description is not None else row.description,
                url=url if url is not None else row.url,
                status=status.value if status is not None else row.status,
                secret=secret if secret is not None else row.secret,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                deleted_timestamp_ms=None,
                packed_events=packed_events,
            )
        )
        if not isinstance(res.outcome, UpsertRegistryWebhookEntityRegistryWebhookUpserted):
            raise MlflowException("Invalid UpsertRegistryWebhookEntity response from RiffDB")
        return self._to_mlflow_webhook(res.outcome.webhook)

    def delete_webhook(self, webhook_id: str) -> None:
        client = self._application()
        row = self._get_webhook_row(client, webhook_id)
        now_ms = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        res = client.upsert_registry_webhook_entity(
            UpsertRegistryWebhookEntityInput(
                request_id=str(uuid4()),
                webhook_id=webhook_id,
                name=row.name,
                description=row.description,
                url=row.url,
                status=row.status,
                secret=row.secret,
                creation_timestamp_ms=int(row.creation_timestamp_ms),
                last_updated_timestamp_ms=now_ms,
                deleted_timestamp_ms=now_ms,
                packed_events=row.packed_events,
            )
        )
        if not isinstance(res.outcome, UpsertRegistryWebhookEntityRegistryWebhookUpserted):
            raise MlflowException("Invalid UpsertRegistryWebhookEntity response from RiffDB")
