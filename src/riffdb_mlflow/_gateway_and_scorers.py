from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.genai.scorers.online.entities import (
        CompletedSession,
        OnlineScorer,
        OnlineScoringConfig,
    )

from mlflow.entities import (
    DatasetRecord,
    DatasetRecordSource,
    EvaluationDataset,
    ExperimentTag,
    FallbackConfig,
    FallbackStrategy,
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelConfig,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelDefinition,
    GatewaySecretInfo,
    Issue,
    IssueSeverity,
    IssueStatus,
    RoutingStrategy,
    ScorerVersion,
    ViewType,
)
from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)
from mlflow.entities.gateway_endpoint import GatewayModelLinkageType, GatewayResourceType
from mlflow.entities.gateway_guardrail import (
    GatewayGuardrail,
    GatewayGuardrailConfig,
    GuardrailAction,
    GuardrailStage,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking._secret_cache import (
    _DEFAULT_CACHE_MAX_SIZE,
    _DEFAULT_CACHE_TTL,
    SECRETS_CACHE_MAX_SIZE_ENV_VAR,
    SECRETS_CACHE_TTL_ENV_VAR,
    SecretCache,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig, GatewayModelConfig
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.crypto import (
    KEKManager,
    _decrypt_secret,
    _encrypt_secret,
    _mask_secret_value,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_EXPERIMENT_IS_GATEWAY,
    MLFLOW_EXPERIMENT_SOURCE_ID,
    MLFLOW_EXPERIMENT_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.search_utils import (
    SearchEvaluationDatasetsUtils,
    SearchIssuesUtils,
    SearchTraceUtils,
    SearchUtils,
)
from mlflow.utils.workspace_context import get_request_workspace

from riffdb_mlflow.generated.client import (
    DeleteEvaluationDatasetEntityEvaluationDatasetDeleted,
    DeleteEvaluationDatasetEntityEvaluationDatasetMissing,
    DeleteEvaluationDatasetEntityInput,
    DeleteGatewayBudgetPolicyEntityGatewayBudgetPolicyDeleted,
    DeleteGatewayBudgetPolicyEntityGatewayBudgetPolicyMissing,
    DeleteGatewayBudgetPolicyEntityInput,
    DeleteGatewayEndpointEntityGatewayEndpointDeleted,
    DeleteGatewayEndpointEntityGatewayEndpointMissing,
    DeleteGatewayEndpointEntityInput,
    DeleteGatewayGuardrailEntityGatewayGuardrailDeleted,
    DeleteGatewayGuardrailEntityGatewayGuardrailMissing,
    DeleteGatewayGuardrailEntityInput,
    DeleteGatewayModelDefinitionEntityGatewayModelDefinitionDeleted,
    DeleteGatewayModelDefinitionEntityGatewayModelDefinitionMissing,
    DeleteGatewayModelDefinitionEntityInput,
    DeleteGatewaySecretEntityGatewaySecretDeleted,
    DeleteGatewaySecretEntityGatewaySecretMissing,
    DeleteGatewaySecretEntityInput,
    DeleteLabelSchemaEntityInput,
    DeleteLabelSchemaEntityLabelSchemaDeleted,
    DeleteLabelSchemaEntityLabelSchemaMissing,
    DeleteReviewQueueEntityInput,
    DeleteReviewQueueEntityReviewQueueDeleted,
    DeleteReviewQueueEntityReviewQueueMissing,
    DeleteScorerEntityInput,
    DeleteScorerEntityScorerDeleted,
    DeleteScorerEntityScorerMissing,
    EvaluationDatasetEntity,
    GatewayBudgetPolicyEntity,
    GatewayEndpointEntity,
    GatewayGuardrailEntity,
    GatewayModelDefinitionEntity,
    GatewaySecretEntity,
    GetEvaluationDatasetFound,
    GetEvaluationDatasetMissing,
    GetEvaluationDatasetParams,
    GetGatewayBudgetPolicyFound,
    GetGatewayBudgetPolicyMissing,
    GetGatewayBudgetPolicyParams,
    GetGatewayEndpointFound,
    GetGatewayEndpointMissing,
    GetGatewayEndpointParams,
    GetGatewayGuardrailFound,
    GetGatewayGuardrailMissing,
    GetGatewayGuardrailParams,
    GetGatewayModelDefinitionFound,
    GetGatewayModelDefinitionMissing,
    GetGatewayModelDefinitionParams,
    GetGatewaySecretFound,
    GetGatewaySecretMissing,
    GetGatewaySecretParams,
    GetIssueFound,
    GetIssueMissing,
    GetIssueParams,
    GetLabelSchemaFound,
    GetLabelSchemaMissing,
    GetLabelSchemaParams,
    GetReviewQueueFound,
    GetReviewQueueMissing,
    GetReviewQueueParams,
    GetScorerFound,
    GetScorerMissing,
    GetScorerParams,
    IssueEntity,
    LabelSchemaEntity,
    ListEvaluationDatasetsFound,
    ListEvaluationDatasetsParams,
    ListGatewayBudgetPoliciesFound,
    ListGatewayBudgetPoliciesParams,
    ListGatewayEndpointsFound,
    ListGatewayEndpointsParams,
    ListGatewayGuardrailsFound,
    ListGatewayGuardrailsParams,
    ListGatewayModelDefinitionsFound,
    ListGatewayModelDefinitionsParams,
    ListGatewaySecretsFound,
    ListGatewaySecretsParams,
    ListIssuesFound,
    ListIssuesParams,
    ListLabelSchemasFound,
    ListLabelSchemasParams,
    ListReviewQueuesFound,
    ListReviewQueuesParams,
    ListScorersFound,
    ListScorersParams,
    ReviewQueueEntity,
    ScorerEntity,
    SearchTracesFound,
    SearchTracesParams,
    UpsertEvaluationDatasetEntityEvaluationDatasetUpserted,
    UpsertEvaluationDatasetEntityInput,
    UpsertGatewayBudgetPolicyEntityGatewayBudgetPolicyUpserted,
    UpsertGatewayBudgetPolicyEntityInput,
    UpsertGatewayEndpointEntityGatewayEndpointUpserted,
    UpsertGatewayEndpointEntityInput,
    UpsertGatewayGuardrailEntityGatewayGuardrailUpserted,
    UpsertGatewayGuardrailEntityInput,
    UpsertGatewayModelDefinitionEntityGatewayModelDefinitionUpserted,
    UpsertGatewayModelDefinitionEntityInput,
    UpsertGatewaySecretEntityGatewaySecretUpserted,
    UpsertGatewaySecretEntityInput,
    UpsertIssueEntityInput,
    UpsertIssueEntityIssueUpserted,
    UpsertLabelSchemaEntityInput,
    UpsertLabelSchemaEntityLabelSchemaUpserted,
    UpsertReviewQueueEntityInput,
    UpsertReviewQueueEntityReviewQueueUpserted,
    UpsertScorerEntityInput,
    UpsertScorerEntityScorerUpserted,
)

_SCOPE = "global"
_TARGETED_BUDGET_SCOPES = (BudgetTargetScope.ENDPOINT.value,)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _pack_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _unpack_list(payload: bytes) -> list[dict[str, Any]]:
    if not payload:
        return []
    loaded = json.loads(payload.decode("utf-8"))
    if isinstance(loaded, list):
        return [dict(item) for item in loaded if isinstance(item, dict)]
    return []


def _unpack_dict(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload.decode("utf-8"))
    if isinstance(loaded, dict):
        return loaded
    return {}


def _validate_one_of(
    param1_name: str, param1_value: Any, param2_name: str, param2_value: Any
) -> None:
    if (param1_value is None) == (param2_value is None):
        raise MlflowException(
            f"Exactly one of {param1_name} or {param2_name} must be provided",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _normalize_budget_target_value(
    target_scope: str | None, target_value: str | None
) -> str | None:
    if target_scope in _TARGETED_BUDGET_SCOPES:
        if not target_value:
            raise MlflowException(
                message=f"target_value is required when target_scope is {target_scope}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return target_value
    return None


class RiffDBGatewayAndScorerMixin(AbstractStore):
    """Compiled RiffDB implementation of MLflow 3.15 GatewayStoreMixin and Scorers."""

    _secret_cache: SecretCache | None = None

    def _gw_client(self) -> Any:
        return self._application()  # type: ignore[attr-defined]

    @property
    def secret_cache(self) -> SecretCache:
        if self._secret_cache is None:
            ttl = int(os.environ.get(SECRETS_CACHE_TTL_ENV_VAR, _DEFAULT_CACHE_TTL))
            max_size = int(os.environ.get(SECRETS_CACHE_MAX_SIZE_ENV_VAR, _DEFAULT_CACHE_MAX_SIZE))
            self._secret_cache = SecretCache(ttl_seconds=ttl, max_size=max_size)
        return self._secret_cache

    def _invalidate_secret_cache(self) -> None:
        if self._secret_cache is not None:
            self._secret_cache.clear()

    def _get_or_create_gateway_experiment_id(
        self, experiment_name: str, tags: list[ExperimentTag] | None = None
    ) -> str:
        try:
            exp_id: str = self.create_experiment(  # type: ignore[no-untyped-call]
                experiment_name, artifact_location=None, tags=tags
            )
            return exp_id
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                exp = self.get_experiment_by_name(experiment_name)  # type: ignore[no-untyped-call]
                if exp is not None:
                    return str(exp.experiment_id)
            raise

    @staticmethod
    def _validate_gateway_max_results(max_results: int) -> None:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                f"Invalid value for max_results. It must be a positive integer, got {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    # ------------------------------------------------------------------
    # Gateway Secrets
    # ------------------------------------------------------------------

    def _fetch_gateway_secret_entity(self, secret_id: str) -> GatewaySecretEntity:
        client = self._gw_client()
        res = client.get_gateway_secret(GetGatewaySecretParams(scope=_SCOPE, secret_id=secret_id))
        if isinstance(res.value, GetGatewaySecretMissing) or not isinstance(
            res.value, GetGatewaySecretFound
        ):
            raise MlflowException(
                f"GatewaySecret not found (secret_id='{secret_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.secret
        return GatewaySecretEntity(
            scope=_SCOPE,
            secret_id=row.secret_id,
            secret_name=row.secret_name,
            encrypted_value=row.encrypted_value,
            wrapped_dek=row.wrapped_dek,
            masked_value_json=row.masked_value_json,
            kek_version=int(row.kek_version),
            provider=row.provider,
            auth_config_json=row.auth_config_json,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
        )

    def _list_gateway_secret_entities(self) -> list[GatewaySecretEntity]:
        client = self._gw_client()
        entities: list[GatewaySecretEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_gateway_secrets(
                ListGatewaySecretsParams(scope=_SCOPE, limit=10, after=cursor)
            )
            if not isinstance(res.value, ListGatewaySecretsFound):
                raise MlflowException("Invalid ListGatewaySecrets response from RiffDB")
            for row in res.value.secrets:
                entities.append(
                    GatewaySecretEntity(
                        scope=_SCOPE,
                        secret_id=row.secret_id,
                        secret_name=row.secret_name,
                        encrypted_value=row.encrypted_value,
                        wrapped_dek=row.wrapped_dek,
                        masked_value_json=row.masked_value_json,
                        kek_version=int(row.kek_version),
                        provider=row.provider,
                        auth_config_json=row.auth_config_json,
                        created_at_ms=int(row.created_at_ms),
                        last_updated_at_ms=int(row.last_updated_at_ms),
                        created_by=row.created_by,
                        last_updated_by=row.last_updated_by,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.secrets:
                break
        return entities

    @staticmethod
    def _secret_entity_to_mlflow(entity: GatewaySecretEntity) -> GatewaySecretInfo:
        try:
            masked_value = json.loads(entity.masked_value_json)
        except (json.JSONDecodeError, TypeError):
            masked_value = {"value": "***"}
        return GatewaySecretInfo(
            secret_id=entity.secret_id,
            secret_name=entity.secret_name,
            masked_values=masked_value,
            created_at=int(entity.created_at_ms),
            last_updated_at=int(entity.last_updated_at_ms),
            provider=entity.provider,
            auth_config=json.loads(entity.auth_config_json) if entity.auth_config_json else None,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
        )

    def _save_gateway_secret_entity(self, entity: GatewaySecretEntity) -> GatewaySecretEntity:
        client = self._gw_client()
        res = client.upsert_gateway_secret_entity(
            UpsertGatewaySecretEntityInput(
                request_id=str(uuid.uuid4()),
                secret_id=entity.secret_id,
                secret_name=entity.secret_name,
                encrypted_value=entity.encrypted_value,
                wrapped_dek=entity.wrapped_dek,
                masked_value_json=entity.masked_value_json,
                kek_version=int(entity.kek_version),
                provider=entity.provider,
                auth_config_json=entity.auth_config_json,
                created_at_ms=int(entity.created_at_ms),
                last_updated_at_ms=int(entity.last_updated_at_ms),
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
            )
        )
        if not isinstance(res.outcome, UpsertGatewaySecretEntityGatewaySecretUpserted):
            raise MlflowException("Invalid UpsertGatewaySecretEntity response from RiffDB")
        return res.outcome.secret

    def create_gateway_secret(
        self,
        secret_name: str,
        secret_value: dict[str, str],
        provider: str | None = None,
        auth_config: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> GatewaySecretInfo:
        for existing in self._list_gateway_secret_entities():
            if existing.secret_name == secret_name:
                raise MlflowException(
                    f"Secret with name '{secret_name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
        secret_id = f"s-{uuid.uuid4().hex}"
        now = _now_ms()
        value_to_encrypt = json.dumps(secret_value)
        masked_value = _mask_secret_value(secret_value)
        kek_manager = KEKManager()
        encrypted = _encrypt_secret(
            secret_value=value_to_encrypt,
            kek_manager=kek_manager,
            secret_id=secret_id,
            secret_name=secret_name,
        )
        saved = self._save_gateway_secret_entity(
            GatewaySecretEntity(
                scope=_SCOPE,
                secret_id=secret_id,
                secret_name=secret_name,
                encrypted_value=encrypted.encrypted_value,
                wrapped_dek=encrypted.wrapped_dek,
                masked_value_json=json.dumps(masked_value),
                kek_version=int(encrypted.kek_version),
                provider=provider,
                auth_config_json=json.dumps(auth_config) if auth_config else None,
                created_at_ms=now,
                last_updated_at_ms=now,
                created_by=created_by,
                last_updated_by=created_by,
            )
        )
        return self._secret_entity_to_mlflow(saved)

    def get_secret_info(
        self, secret_id: str | None = None, secret_name: str | None = None
    ) -> GatewaySecretInfo:
        _validate_one_of("secret_id", secret_id, "secret_name", secret_name)
        if secret_id is not None:
            return self._secret_entity_to_mlflow(self._fetch_gateway_secret_entity(secret_id))
        for item in self._list_gateway_secret_entities():
            if item.secret_name == secret_name:
                return self._secret_entity_to_mlflow(item)
        raise MlflowException(
            f"GatewaySecret not found (secret_name='{secret_name}')",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def update_gateway_secret(
        self,
        secret_id: str,
        secret_value: dict[str, str] | None = None,
        auth_config: dict[str, Any] | None = None,
        updated_by: str | None = None,
    ) -> GatewaySecretInfo:
        existing = self._fetch_gateway_secret_entity(secret_id)
        encrypted_value = existing.encrypted_value
        wrapped_dek = existing.wrapped_dek
        kek_version = int(existing.kek_version)
        masked_value_json = existing.masked_value_json

        if secret_value is not None:
            value_to_encrypt = json.dumps(secret_value)
            masked_value = _mask_secret_value(secret_value)
            kek_manager = KEKManager()
            encrypted = _encrypt_secret(
                secret_value=value_to_encrypt,
                kek_manager=kek_manager,
                secret_id=existing.secret_id,
                secret_name=existing.secret_name,
            )
            encrypted_value = encrypted.encrypted_value
            wrapped_dek = encrypted.wrapped_dek
            kek_version = int(encrypted.kek_version)
            masked_value_json = json.dumps(masked_value)

        auth_config_json = existing.auth_config_json
        if auth_config is not None:
            auth_config_json = json.dumps(auth_config) if auth_config else None

        saved = self._save_gateway_secret_entity(
            GatewaySecretEntity(
                scope=_SCOPE,
                secret_id=existing.secret_id,
                secret_name=existing.secret_name,
                encrypted_value=encrypted_value,
                wrapped_dek=wrapped_dek,
                masked_value_json=masked_value_json,
                kek_version=kek_version,
                provider=existing.provider,
                auth_config_json=auth_config_json,
                created_at_ms=int(existing.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=existing.created_by,
                last_updated_by=updated_by,
            )
        )
        self._invalidate_secret_cache()
        return self._secret_entity_to_mlflow(saved)

    def delete_gateway_secret(self, secret_id: str) -> None:
        self._fetch_gateway_secret_entity(secret_id)
        # Orphan any model definitions referencing this secret (SET NULL semantics)
        for model_def in self._list_gateway_model_definition_entities():
            if model_def.secret_id == secret_id:
                self._save_gateway_model_definition_entity(
                    GatewayModelDefinitionEntity(
                        scope=_SCOPE,
                        model_definition_id=model_def.model_definition_id,
                        name=model_def.name,
                        secret_id=None,
                        provider=model_def.provider,
                        model_name=model_def.model_name,
                        created_at_ms=int(model_def.created_at_ms),
                        last_updated_at_ms=int(model_def.last_updated_at_ms),
                        created_by=model_def.created_by,
                        last_updated_by=model_def.last_updated_by,
                    )
                )
        client = self._gw_client()
        res = client.delete_gateway_secret_entity(
            DeleteGatewaySecretEntityInput(request_id=str(uuid.uuid4()), secret_id=secret_id)
        )
        if isinstance(res.outcome, DeleteGatewaySecretEntityGatewaySecretMissing):
            raise MlflowException(
                f"GatewaySecret not found (secret_id='{secret_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteGatewaySecretEntityGatewaySecretDeleted):
            raise MlflowException("Invalid DeleteGatewaySecretEntity response from RiffDB")
        self._invalidate_secret_cache()

    def list_secret_infos(self, provider: str | None = None) -> list[GatewaySecretInfo]:
        entities = self._list_gateway_secret_entities()
        if provider is not None:
            entities = [e for e in entities if e.provider == provider]
        return [self._secret_entity_to_mlflow(e) for e in entities]

    # ------------------------------------------------------------------
    # Gateway Model Definitions
    # ------------------------------------------------------------------

    def _fetch_gateway_model_definition_entity(
        self, model_definition_id: str
    ) -> GatewayModelDefinitionEntity:
        client = self._gw_client()
        res = client.get_gateway_model_definition(
            GetGatewayModelDefinitionParams(scope=_SCOPE, model_definition_id=model_definition_id)
        )
        if isinstance(res.value, GetGatewayModelDefinitionMissing) or not isinstance(
            res.value, GetGatewayModelDefinitionFound
        ):
            raise MlflowException(
                f"GatewayModelDefinition not found (model_definition_id='{model_definition_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.model_def
        return GatewayModelDefinitionEntity(
            scope=_SCOPE,
            model_definition_id=row.model_definition_id,
            name=row.name,
            secret_id=row.secret_id,
            provider=row.provider,
            model_name=row.model_name,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
        )

    def _list_gateway_model_definition_entities(self) -> list[GatewayModelDefinitionEntity]:
        client = self._gw_client()
        entities: list[GatewayModelDefinitionEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_gateway_model_definitions(
                ListGatewayModelDefinitionsParams(scope=_SCOPE, limit=50, after=cursor)
            )
            if not isinstance(res.value, ListGatewayModelDefinitionsFound):
                raise MlflowException("Invalid ListGatewayModelDefinitions response from RiffDB")
            for row in res.value.model_defs:
                entities.append(
                    GatewayModelDefinitionEntity(
                        scope=_SCOPE,
                        model_definition_id=row.model_definition_id,
                        name=row.name,
                        secret_id=row.secret_id,
                        provider=row.provider,
                        model_name=row.model_name,
                        created_at_ms=int(row.created_at_ms),
                        last_updated_at_ms=int(row.last_updated_at_ms),
                        created_by=row.created_by,
                        last_updated_by=row.last_updated_by,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.model_defs:
                break
        return entities

    def _save_gateway_model_definition_entity(
        self, entity: GatewayModelDefinitionEntity
    ) -> GatewayModelDefinitionEntity:
        client = self._gw_client()
        res = client.upsert_gateway_model_definition_entity(
            UpsertGatewayModelDefinitionEntityInput(
                request_id=str(uuid.uuid4()),
                model_definition_id=entity.model_definition_id,
                name=entity.name,
                secret_id=entity.secret_id,
                provider=entity.provider,
                model_name=entity.model_name,
                created_at_ms=int(entity.created_at_ms),
                last_updated_at_ms=int(entity.last_updated_at_ms),
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
            )
        )
        if not isinstance(
            res.outcome, UpsertGatewayModelDefinitionEntityGatewayModelDefinitionUpserted
        ):
            raise MlflowException("Invalid UpsertGatewayModelDefinitionEntity response from RiffDB")
        return res.outcome.model_def

    def _model_def_entity_to_mlflow(
        self,
        entity: GatewayModelDefinitionEntity,
        secrets_by_id: dict[str, GatewaySecretEntity] | None = None,
    ) -> GatewayModelDefinition:
        secret_name: str | None = None
        if entity.secret_id is not None:
            if secrets_by_id is not None and entity.secret_id in secrets_by_id:
                secret_name = secrets_by_id[entity.secret_id].secret_name
            else:
                try:
                    secret_name = self._fetch_gateway_secret_entity(entity.secret_id).secret_name
                except MlflowException:
                    secret_name = None
        return GatewayModelDefinition(
            model_definition_id=entity.model_definition_id,
            name=entity.name,
            secret_id=entity.secret_id,
            secret_name=secret_name,
            provider=entity.provider,
            model_name=entity.model_name,
            created_at=int(entity.created_at_ms),
            last_updated_at=int(entity.last_updated_at_ms),
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
        )

    def create_gateway_model_definition(
        self,
        name: str,
        secret_id: str,
        provider: str,
        model_name: str,
        created_by: str | None = None,
    ) -> GatewayModelDefinition:
        secret_entity = self._fetch_gateway_secret_entity(secret_id)
        for existing in self._list_gateway_model_definition_entities():
            if existing.name == name:
                raise MlflowException(
                    f"Model definition with name '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
        model_definition_id = f"d-{uuid.uuid4().hex}"
        now = _now_ms()
        saved = self._save_gateway_model_definition_entity(
            GatewayModelDefinitionEntity(
                scope=_SCOPE,
                model_definition_id=model_definition_id,
                name=name,
                secret_id=secret_id,
                provider=provider,
                model_name=model_name,
                created_at_ms=now,
                last_updated_at_ms=now,
                created_by=created_by,
                last_updated_by=created_by,
            )
        )
        return self._model_def_entity_to_mlflow(saved, {secret_id: secret_entity})

    def get_gateway_model_definition(
        self, model_definition_id: str | None = None, name: str | None = None
    ) -> GatewayModelDefinition:
        _validate_one_of("model_definition_id", model_definition_id, "name", name)
        if model_definition_id is not None:
            return self._model_def_entity_to_mlflow(
                self._fetch_gateway_model_definition_entity(model_definition_id)
            )
        for item in self._list_gateway_model_definition_entities():
            if item.name == name:
                return self._model_def_entity_to_mlflow(item)
        raise MlflowException(
            f"GatewayModelDefinition not found (name='{name}')",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def list_gateway_model_definitions(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayModelDefinition]:
        entities = self._list_gateway_model_definition_entities()
        if provider is not None:
            entities = [e for e in entities if e.provider == provider]
        if secret_id is not None:
            entities = [e for e in entities if e.secret_id == secret_id]
        secrets_by_id = {s.secret_id: s for s in self._list_gateway_secret_entities()}
        return [self._model_def_entity_to_mlflow(e, secrets_by_id) for e in entities]

    def update_gateway_model_definition(
        self,
        model_definition_id: str,
        name: str | None = None,
        secret_id: str | None = None,
        model_name: str | None = None,
        updated_by: str | None = None,
        provider: str | None = None,
    ) -> GatewayModelDefinition:
        existing = self._fetch_gateway_model_definition_entity(model_definition_id)
        if name is not None and name != existing.name:
            for item in self._list_gateway_model_definition_entities():
                if item.name == name and item.model_definition_id != model_definition_id:
                    raise MlflowException(
                        f"Model definition with name '{name}' already exists",
                        error_code=RESOURCE_ALREADY_EXISTS,
                    )
        resolved_secret_id = existing.secret_id
        if secret_id is not None:
            self._fetch_gateway_secret_entity(secret_id)
            resolved_secret_id = secret_id

        saved = self._save_gateway_model_definition_entity(
            GatewayModelDefinitionEntity(
                scope=_SCOPE,
                model_definition_id=existing.model_definition_id,
                name=name if name is not None else existing.name,
                secret_id=resolved_secret_id,
                provider=provider if provider is not None else existing.provider,
                model_name=model_name if model_name is not None else existing.model_name,
                created_at_ms=int(existing.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=existing.created_by,
                last_updated_by=updated_by if updated_by else existing.last_updated_by,
            )
        )
        self._invalidate_secret_cache()
        return self._model_def_entity_to_mlflow(saved)

    def delete_gateway_model_definition(self, model_definition_id: str) -> None:
        self._fetch_gateway_model_definition_entity(model_definition_id)
        for endpoint in self._list_gateway_endpoint_entities():
            for mapping in _unpack_list(endpoint.packed_model_mappings):
                if mapping.get("model_definition_id") == model_definition_id:
                    raise MlflowException(
                        "Cannot delete model definition that is currently in use by endpoints. "
                        "Detach it from all endpoints first.",
                        error_code=INVALID_STATE,
                    )
        client = self._gw_client()
        res = client.delete_gateway_model_definition_entity(
            DeleteGatewayModelDefinitionEntityInput(
                request_id=str(uuid.uuid4()), model_definition_id=model_definition_id
            )
        )
        if isinstance(res.outcome, DeleteGatewayModelDefinitionEntityGatewayModelDefinitionMissing):
            raise MlflowException(
                f"GatewayModelDefinition not found (model_definition_id='{model_definition_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(
            res.outcome, DeleteGatewayModelDefinitionEntityGatewayModelDefinitionDeleted
        ):
            raise MlflowException("Invalid DeleteGatewayModelDefinitionEntity response from RiffDB")
        self._invalidate_secret_cache()

    # ------------------------------------------------------------------
    # Gateway Endpoints, Model Mappings, Bindings, Tags
    # ------------------------------------------------------------------

    def _fetch_gateway_endpoint_entity(self, endpoint_id: str) -> GatewayEndpointEntity:
        client = self._gw_client()
        res = client.get_gateway_endpoint(
            GetGatewayEndpointParams(scope=_SCOPE, endpoint_id=endpoint_id)
        )
        if isinstance(res.value, GetGatewayEndpointMissing) or not isinstance(
            res.value, GetGatewayEndpointFound
        ):
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.endpoint
        return GatewayEndpointEntity(
            scope=_SCOPE,
            endpoint_id=row.endpoint_id,
            name=row.name,
            routing_strategy=row.routing_strategy,
            fallback_config_json=row.fallback_config_json,
            experiment_id=int(row.experiment_id) if row.experiment_id is not None else None,
            usage_tracking=bool(row.usage_tracking),
            packed_model_mappings=row.packed_model_mappings,
            packed_tags=row.packed_tags,
            packed_bindings=row.packed_bindings,
            packed_guardrail_configs=row.packed_guardrail_configs,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
        )

    def _list_gateway_endpoint_entities(self) -> list[GatewayEndpointEntity]:
        client = self._gw_client()
        entities: list[GatewayEndpointEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_gateway_endpoints(
                ListGatewayEndpointsParams(scope=_SCOPE, limit=10, after=cursor)
            )
            if not isinstance(res.value, ListGatewayEndpointsFound):
                raise MlflowException("Invalid ListGatewayEndpoints response from RiffDB")
            for row in res.value.endpoints:
                entities.append(
                    GatewayEndpointEntity(
                        scope=_SCOPE,
                        endpoint_id=row.endpoint_id,
                        name=row.name,
                        routing_strategy=row.routing_strategy,
                        fallback_config_json=row.fallback_config_json,
                        experiment_id=int(row.experiment_id)
                        if row.experiment_id is not None
                        else None,
                        usage_tracking=bool(row.usage_tracking),
                        packed_model_mappings=row.packed_model_mappings,
                        packed_tags=row.packed_tags,
                        packed_bindings=row.packed_bindings,
                        packed_guardrail_configs=row.packed_guardrail_configs,
                        created_at_ms=int(row.created_at_ms),
                        last_updated_at_ms=int(row.last_updated_at_ms),
                        created_by=row.created_by,
                        last_updated_by=row.last_updated_by,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.endpoints:
                break
        return entities

    def _save_gateway_endpoint_entity(self, entity: GatewayEndpointEntity) -> GatewayEndpointEntity:
        client = self._gw_client()
        res = client.upsert_gateway_endpoint_entity(
            UpsertGatewayEndpointEntityInput(
                request_id=str(uuid.uuid4()),
                endpoint_id=entity.endpoint_id,
                name=entity.name,
                routing_strategy=entity.routing_strategy,
                fallback_config_json=entity.fallback_config_json,
                experiment_id=int(entity.experiment_id)
                if entity.experiment_id is not None
                else None,
                usage_tracking=bool(entity.usage_tracking),
                packed_model_mappings=entity.packed_model_mappings,
                packed_tags=entity.packed_tags,
                packed_bindings=entity.packed_bindings,
                packed_guardrail_configs=entity.packed_guardrail_configs,
                created_at_ms=int(entity.created_at_ms),
                last_updated_at_ms=int(entity.last_updated_at_ms),
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
            )
        )
        if not isinstance(res.outcome, UpsertGatewayEndpointEntityGatewayEndpointUpserted):
            raise MlflowException("Invalid UpsertGatewayEndpointEntity response from RiffDB")
        return res.outcome.endpoint

    def _mapping_dict_to_mlflow(
        self,
        endpoint_id: str,
        m_dict: dict[str, Any],
        model_defs_by_id: dict[str, GatewayModelDefinitionEntity] | None = None,
        secrets_by_id: dict[str, GatewaySecretEntity] | None = None,
    ) -> GatewayEndpointModelMapping:
        model_def_id = str(m_dict["model_definition_id"])
        model_def_entity: GatewayModelDefinitionEntity | None = None
        if model_defs_by_id is not None and model_def_id in model_defs_by_id:
            model_def_entity = model_defs_by_id[model_def_id]
        else:
            try:
                model_def_entity = self._fetch_gateway_model_definition_entity(model_def_id)
            except MlflowException:
                model_def_entity = None
        model_def_mlflow = (
            self._model_def_entity_to_mlflow(model_def_entity, secrets_by_id)
            if model_def_entity is not None
            else None
        )
        return GatewayEndpointModelMapping(
            mapping_id=str(m_dict["mapping_id"]),
            endpoint_id=endpoint_id,
            model_definition_id=model_def_id,
            model_definition=model_def_mlflow,
            weight=float(m_dict.get("weight", 1.0)),
            linkage_type=GatewayModelLinkageType(
                str(m_dict.get("linkage_type", GatewayModelLinkageType.PRIMARY.value))
            ),
            fallback_order=m_dict.get("fallback_order"),
            created_at=int(m_dict.get("created_at", 0)),
            created_by=m_dict.get("created_by"),
        )

    def _gw_endpoint_entity_to_mlflow(
        self,
        entity: GatewayEndpointEntity,
        model_defs_by_id: dict[str, GatewayModelDefinitionEntity] | None = None,
        secrets_by_id: dict[str, GatewaySecretEntity] | None = None,
    ) -> GatewayEndpoint:
        fallback_config: FallbackConfig | None = None
        if entity.fallback_config_json:
            try:
                fc_dict = json.loads(entity.fallback_config_json)
                fallback_config = FallbackConfig(
                    strategy=FallbackStrategy(fc_dict["strategy"])
                    if fc_dict.get("strategy")
                    else None,
                    max_attempts=fc_dict.get("max_attempts"),
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                fallback_config = None
        routing_strategy = (
            RoutingStrategy(entity.routing_strategy) if entity.routing_strategy else None
        )
        mappings = [
            self._mapping_dict_to_mlflow(
                entity.endpoint_id,
                m,
                model_defs_by_id=model_defs_by_id,
                secrets_by_id=secrets_by_id,
            )
            for m in _unpack_list(entity.packed_model_mappings)
        ]
        tags_map = _unpack_dict(entity.packed_tags)
        tags = [GatewayEndpointTag(key=str(k), value=str(v)) for k, v in tags_map.items()]
        return GatewayEndpoint(
            endpoint_id=entity.endpoint_id,
            name=entity.name,
            model_mappings=mappings,
            tags=tags,
            created_at=int(entity.created_at_ms),
            last_updated_at=int(entity.last_updated_at_ms),
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            routing_strategy=routing_strategy,
            fallback_config=fallback_config,
            experiment_id=str(entity.experiment_id) if entity.experiment_id is not None else None,
            usage_tracking=bool(entity.usage_tracking),
        )

    def create_gateway_endpoint(
        self,
        name: str,
        model_configs: list[GatewayEndpointModelConfig],
        created_by: str | None = None,
        routing_strategy: RoutingStrategy | None = None,
        fallback_config: FallbackConfig | None = None,
        experiment_id: str | None = None,
        usage_tracking: bool = True,
    ) -> GatewayEndpoint:
        if not model_configs:
            raise MlflowException(
                "Endpoint must have at least one model configuration",
                error_code=INVALID_PARAMETER_VALUE,
            )
        for existing in self._list_gateway_endpoint_entities():
            if existing.name == name:
                raise MlflowException(
                    f"Endpoint with name '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )

        model_defs_by_id = {
            m.model_definition_id: m for m in self._list_gateway_model_definition_entities()
        }
        all_model_def_ids = {config.model_definition_id for config in model_configs}
        if missing := all_model_def_ids - set(model_defs_by_id.keys()):
            raise MlflowException(
                f"Model definitions not found: {', '.join(sorted(missing))}",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        endpoint_id = f"e-{uuid.uuid4().hex}"
        now = _now_ms()

        resolved_exp_id = experiment_id
        if usage_tracking and resolved_exp_id is None:
            resolved_exp_id = self._get_or_create_gateway_experiment_id(
                f"gateway/{name}",
                tags=[
                    ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_TYPE, "GATEWAY"),  # type: ignore[no-untyped-call]
                    ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_ID, endpoint_id),  # type: ignore[no-untyped-call]
                    ExperimentTag(MLFLOW_EXPERIMENT_IS_GATEWAY, "true"),  # type: ignore[no-untyped-call]
                ],
            )

        fallback_model_def_ids = [
            config.model_definition_id
            for config in model_configs
            if config.linkage_type == GatewayModelLinkageType.FALLBACK
        ]
        fallback_config_json: str | None = None
        if fallback_config or fallback_model_def_ids:
            fallback_config_json = json.dumps(
                {
                    "strategy": (
                        fallback_config.strategy.value
                        if fallback_config and fallback_config.strategy
                        else None
                    ),
                    "max_attempts": fallback_config.max_attempts if fallback_config else None,
                    "model_definition_ids": fallback_model_def_ids,
                }
            )

        mappings_payload = [
            {
                "mapping_id": f"m-{uuid.uuid4().hex}",
                "endpoint_id": endpoint_id,
                "model_definition_id": config.model_definition_id,
                "weight": config.weight,
                "linkage_type": config.linkage_type.value,
                "fallback_order": config.fallback_order,
                "created_at": now,
                "created_by": created_by,
            }
            for config in model_configs
        ]

        saved = self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=endpoint_id,
                name=name,
                routing_strategy=routing_strategy.value if routing_strategy else None,
                fallback_config_json=fallback_config_json,
                experiment_id=int(resolved_exp_id) if resolved_exp_id is not None else None,
                usage_tracking=bool(usage_tracking),
                packed_model_mappings=_pack_json(mappings_payload),
                packed_tags=_pack_json({}),
                packed_bindings=_pack_json([]),
                packed_guardrail_configs=_pack_json([]),
                created_at_ms=now,
                last_updated_at_ms=now,
                created_by=created_by,
                last_updated_by=created_by,
            )
        )
        return self._gw_endpoint_entity_to_mlflow(saved, model_defs_by_id=model_defs_by_id)

    def get_gateway_endpoint(
        self, endpoint_id: str | None = None, name: str | None = None
    ) -> GatewayEndpoint:
        _validate_one_of("endpoint_id", endpoint_id, "name", name)
        if endpoint_id is not None:
            return self._gw_endpoint_entity_to_mlflow(
                self._fetch_gateway_endpoint_entity(endpoint_id)
            )
        for item in self._list_gateway_endpoint_entities():
            if item.name == name:
                return self._gw_endpoint_entity_to_mlflow(item)
        raise MlflowException(
            f"GatewayEndpoint not found (name='{name}')",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def update_gateway_endpoint(
        self,
        endpoint_id: str,
        name: str | None = None,
        updated_by: str | None = None,
        routing_strategy: RoutingStrategy | None = None,
        fallback_config: FallbackConfig | None = None,
        model_configs: list[GatewayEndpointModelConfig] | None = None,
        experiment_id: str | None = None,
        usage_tracking: bool | None = None,
    ) -> GatewayEndpoint:
        existing = self._fetch_gateway_endpoint_entity(endpoint_id)
        if name is not None and name != existing.name:
            for item in self._list_gateway_endpoint_entities():
                if item.name == name and item.endpoint_id != endpoint_id:
                    raise MlflowException(
                        f"Endpoint with name '{name}' already exists",
                        error_code=RESOURCE_ALREADY_EXISTS,
                    )

        resolved_name = name if name is not None else existing.name
        resolved_usage_tracking = (
            usage_tracking if usage_tracking is not None else bool(existing.usage_tracking)
        )
        resolved_exp_id: int | None = existing.experiment_id
        if usage_tracking and experiment_id is None and existing.experiment_id is None:
            created_exp_id = self._get_or_create_gateway_experiment_id(
                f"gateway/{resolved_name}",
                tags=[
                    ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_TYPE, "GATEWAY"),  # type: ignore[no-untyped-call]
                    ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_ID, endpoint_id),  # type: ignore[no-untyped-call]
                    ExperimentTag(MLFLOW_EXPERIMENT_IS_GATEWAY, "true"),  # type: ignore[no-untyped-call]
                ],
            )
            resolved_exp_id = int(created_exp_id)
        if experiment_id is not None:
            resolved_exp_id = int(experiment_id)

        resolved_routing = (
            routing_strategy.value if routing_strategy is not None else existing.routing_strategy
        )
        resolved_fallback_json = existing.fallback_config_json
        resolved_mappings_bytes = existing.packed_model_mappings
        now = _now_ms()

        if model_configs is not None:
            for config in model_configs:
                self._fetch_gateway_model_definition_entity(config.model_definition_id)
            mappings_payload = [
                {
                    "mapping_id": f"m-{uuid.uuid4().hex}",
                    "endpoint_id": endpoint_id,
                    "model_definition_id": config.model_definition_id,
                    "weight": config.weight,
                    "linkage_type": config.linkage_type.value,
                    "fallback_order": config.fallback_order,
                    "created_at": now,
                    "created_by": updated_by,
                }
                for config in model_configs
            ]
            resolved_mappings_bytes = _pack_json(mappings_payload)
            fallback_model_def_ids = [
                config.model_definition_id
                for config in model_configs
                if config.linkage_type == GatewayModelLinkageType.FALLBACK
            ]
            resolved_fallback_json = json.dumps(
                {
                    "strategy": (
                        fallback_config.strategy.value
                        if fallback_config and fallback_config.strategy
                        else None
                    ),
                    "max_attempts": fallback_config.max_attempts if fallback_config else None,
                    "model_definition_ids": fallback_model_def_ids,
                }
            )
        elif fallback_config is not None:
            existing_fc = (
                json.loads(existing.fallback_config_json) if existing.fallback_config_json else {}
            )
            resolved_fallback_json = json.dumps(
                {
                    "strategy": (
                        fallback_config.strategy.value if fallback_config.strategy else None
                    ),
                    "max_attempts": fallback_config.max_attempts,
                    "model_definition_ids": existing_fc.get("model_definition_ids", []),
                }
            )

        saved = self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=endpoint_id,
                name=resolved_name,
                routing_strategy=resolved_routing,
                fallback_config_json=resolved_fallback_json,
                experiment_id=resolved_exp_id,
                usage_tracking=resolved_usage_tracking,
                packed_model_mappings=resolved_mappings_bytes,
                packed_tags=existing.packed_tags,
                packed_bindings=existing.packed_bindings,
                packed_guardrail_configs=existing.packed_guardrail_configs,
                created_at_ms=int(existing.created_at_ms),
                last_updated_at_ms=now,
                created_by=existing.created_by,
                last_updated_by=updated_by if updated_by else existing.last_updated_by,
            )
        )
        self._invalidate_secret_cache()
        return self._gw_endpoint_entity_to_mlflow(saved)

    def delete_gateway_endpoint(self, endpoint_id: str) -> None:
        self._fetch_gateway_endpoint_entity(endpoint_id)
        client = self._gw_client()
        res = client.delete_gateway_endpoint_entity(
            DeleteGatewayEndpointEntityInput(request_id=str(uuid.uuid4()), endpoint_id=endpoint_id)
        )
        if isinstance(res.outcome, DeleteGatewayEndpointEntityGatewayEndpointMissing):
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteGatewayEndpointEntityGatewayEndpointDeleted):
            raise MlflowException("Invalid DeleteGatewayEndpointEntity response from RiffDB")
        self._invalidate_secret_cache()

    def list_gateway_endpoints(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayEndpoint]:
        endpoints = self._list_gateway_endpoint_entities()
        model_defs_by_id = {
            m.model_definition_id: m for m in self._list_gateway_model_definition_entities()
        }
        secrets_by_id = {s.secret_id: s for s in self._list_gateway_secret_entities()}
        results: list[GatewayEndpoint] = []
        for ep in endpoints:
            mappings = _unpack_list(ep.packed_model_mappings)
            if not mappings:
                continue
            if provider is not None or secret_id is not None:
                matched = False
                for m in mappings:
                    m_def = model_defs_by_id.get(str(m.get("model_definition_id", "")))
                    if m_def is None:
                        continue
                    if provider is not None and m_def.provider != provider:
                        continue
                    if secret_id is not None and m_def.secret_id != secret_id:
                        continue
                    matched = True
                    break
                if not matched:
                    continue
            results.append(
                self._gw_endpoint_entity_to_mlflow(
                    ep, model_defs_by_id=model_defs_by_id, secrets_by_id=secrets_by_id
                )
            )
        return results

    def attach_model_to_endpoint(
        self,
        endpoint_id: str,
        model_config: GatewayEndpointModelConfig,
        created_by: str | None = None,
    ) -> GatewayEndpointModelMapping:
        ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        model_def = self._fetch_gateway_model_definition_entity(model_config.model_definition_id)
        mappings = _unpack_list(ep.packed_model_mappings)
        for m in mappings:
            if (
                m.get("model_definition_id") == model_config.model_definition_id
                and m.get("linkage_type") == model_config.linkage_type.value
            ):
                raise MlflowException(
                    f"Model definition '{model_config.model_definition_id}' is already attached to "
                    f"endpoint '{endpoint_id}'",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
        now = _now_ms()
        new_mapping = {
            "mapping_id": f"m-{uuid.uuid4().hex}",
            "endpoint_id": endpoint_id,
            "model_definition_id": model_config.model_definition_id,
            "weight": model_config.weight,
            "linkage_type": model_config.linkage_type.value,
            "fallback_order": model_config.fallback_order,
            "created_at": now,
            "created_by": created_by,
        }
        mappings.append(new_mapping)
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=_pack_json(mappings),
                packed_tags=ep.packed_tags,
                packed_bindings=ep.packed_bindings,
                packed_guardrail_configs=ep.packed_guardrail_configs,
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=now,
                created_by=ep.created_by,
                last_updated_by=created_by if created_by else ep.last_updated_by,
            )
        )
        self._invalidate_secret_cache()
        return self._mapping_dict_to_mlflow(
            endpoint_id, new_mapping, model_defs_by_id={model_def.model_definition_id: model_def}
        )

    def detach_model_from_endpoint(
        self,
        endpoint_id: str,
        model_definition_id: str,
        linkage_type: str | None = None,
    ) -> None:
        ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        mappings = _unpack_list(ep.packed_model_mappings)
        kept: list[dict[str, Any]] = []
        removed = False
        for m in mappings:
            if m.get("model_definition_id") == model_definition_id and (
                linkage_type is None or m.get("linkage_type") == linkage_type
            ):
                removed = True
                continue
            kept.append(m)
        if not removed:
            linkage_str = f" with linkage type '{linkage_type}'" if linkage_type else ""
            raise MlflowException(
                f"Model definition '{model_definition_id}' is not attached to "
                f"endpoint '{endpoint_id}'{linkage_str}",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=_pack_json(kept),
                packed_tags=ep.packed_tags,
                packed_bindings=ep.packed_bindings,
                packed_guardrail_configs=ep.packed_guardrail_configs,
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )
        self._invalidate_secret_cache()

    def create_endpoint_binding(
        self,
        endpoint_id: str,
        resource_type: str,
        resource_id: str,
        created_by: str | None = None,
        display_name: str | None = None,
    ) -> GatewayEndpointBinding:
        ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        bindings = _unpack_list(ep.packed_bindings)
        now = _now_ms()
        new_binding = {
            "endpoint_id": endpoint_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "display_name": display_name,
            "created_at": now,
            "last_updated_at": now,
            "created_by": created_by,
            "last_updated_by": created_by,
        }
        bindings = [
            b
            for b in bindings
            if not (b.get("resource_type") == resource_type and b.get("resource_id") == resource_id)
        ]
        bindings.append(new_binding)
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=ep.packed_model_mappings,
                packed_tags=ep.packed_tags,
                packed_bindings=_pack_json(bindings),
                packed_guardrail_configs=ep.packed_guardrail_configs,
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=now,
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )
        self._invalidate_secret_cache()
        return GatewayEndpointBinding(
            endpoint_id=endpoint_id,
            resource_type=GatewayResourceType(resource_type),
            resource_id=resource_id,
            created_at=now,
            last_updated_at=now,
            created_by=created_by,
            last_updated_by=created_by,
            display_name=display_name,
        )

    def delete_endpoint_binding(
        self, endpoint_id: str, resource_type: str, resource_id: str
    ) -> None:
        try:
            ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        except MlflowException as e:
            raise MlflowException(
                f"GatewayEndpointBinding not found (endpoint_id='{endpoint_id}', "
                f"resource_type='{resource_type}', resource_id='{resource_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e
        bindings = _unpack_list(ep.packed_bindings)
        kept: list[dict[str, Any]] = []
        removed = False
        for b in bindings:
            if b.get("resource_type") == resource_type and b.get("resource_id") == resource_id:
                removed = True
                continue
            kept.append(b)
        if not removed:
            raise MlflowException(
                f"GatewayEndpointBinding not found (endpoint_id='{endpoint_id}', "
                f"resource_type='{resource_type}', resource_id='{resource_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=ep.packed_model_mappings,
                packed_tags=ep.packed_tags,
                packed_bindings=_pack_json(kept),
                packed_guardrail_configs=ep.packed_guardrail_configs,
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )
        self._invalidate_secret_cache()

    def list_endpoint_bindings(
        self,
        endpoint_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> list[GatewayEndpointBinding]:
        endpoints = (
            [self._fetch_gateway_endpoint_entity(endpoint_id)]
            if endpoint_id is not None
            else self._list_gateway_endpoint_entities()
        )
        results: list[GatewayEndpointBinding] = []
        for ep in endpoints:
            for b in _unpack_list(ep.packed_bindings):
                if resource_type is not None and b.get("resource_type") != resource_type:
                    continue
                if resource_id is not None and b.get("resource_id") != resource_id:
                    continue
                results.append(
                    GatewayEndpointBinding(
                        endpoint_id=ep.endpoint_id,
                        resource_type=GatewayResourceType(str(b["resource_type"])),
                        resource_id=str(b["resource_id"]),
                        created_at=int(b.get("created_at", 0)),
                        last_updated_at=int(b.get("last_updated_at", 0)),
                        created_by=b.get("created_by"),
                        last_updated_by=b.get("last_updated_by"),
                        display_name=b.get("display_name"),
                    )
                )
        return results

    def set_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        tag: GatewayEndpointTag,
    ) -> None:
        ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        tags_map = _unpack_dict(ep.packed_tags)
        tags_map[str(tag.key)] = str(tag.value)
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=ep.packed_model_mappings,
                packed_tags=_pack_json(tags_map),
                packed_bindings=ep.packed_bindings,
                packed_guardrail_configs=ep.packed_guardrail_configs,
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )

    def delete_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        key: str,
    ) -> None:
        ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        tags_map = _unpack_dict(ep.packed_tags)
        tags_map.pop(key, None)
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=ep.packed_model_mappings,
                packed_tags=_pack_json(tags_map),
                packed_bindings=ep.packed_bindings,
                packed_guardrail_configs=ep.packed_guardrail_configs,
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )

    # ------------------------------------------------------------------
    # Server-Side Gateway Config Resolver (Decrypted Model Configs)
    # ------------------------------------------------------------------

    def _build_endpoint_config(self, ep: GatewayEndpointEntity) -> GatewayEndpointConfig:
        kek_manager = KEKManager()
        model_configs: list[GatewayModelConfig] = []
        for m in _unpack_list(ep.packed_model_mappings):
            model_def = self._fetch_gateway_model_definition_entity(str(m["model_definition_id"]))
            if model_def.secret_id is None:
                continue
            secret = self._fetch_gateway_secret_entity(model_def.secret_id)
            decrypted_raw = _decrypt_secret(
                encrypted_value=secret.encrypted_value,
                wrapped_dek=secret.wrapped_dek,
                kek_manager=kek_manager,
                secret_id=secret.secret_id,
                secret_name=secret.secret_name,
            )
            decrypted_value = (
                decrypted_raw if isinstance(decrypted_raw, dict) else json.loads(str(decrypted_raw))
            )
            model_configs.append(
                GatewayModelConfig(
                    model_definition_id=model_def.model_definition_id,
                    provider=model_def.provider,
                    model_name=model_def.model_name,
                    secret_value=decrypted_value,
                    auth_config=json.loads(secret.auth_config_json)
                    if secret.auth_config_json
                    else None,
                    weight=float(m.get("weight", 1.0)),
                    linkage_type=GatewayModelLinkageType(
                        str(m.get("linkage_type", GatewayModelLinkageType.PRIMARY.value))
                    ),
                    fallback_order=m.get("fallback_order"),
                )
            )
        ep_mlflow = self._gw_endpoint_entity_to_mlflow(ep)
        return GatewayEndpointConfig(
            endpoint_id=ep.endpoint_id,
            endpoint_name=ep.name,
            models=model_configs,
            routing_strategy=ep_mlflow.routing_strategy,
            fallback_config=ep_mlflow.fallback_config,
            experiment_id=ep_mlflow.experiment_id,
            usage_tracking=bool(ep_mlflow.usage_tracking and ep_mlflow.experiment_id is not None),
        )

    def get_endpoint_config(self, endpoint_name: str) -> GatewayEndpointConfig:
        workspace = get_request_workspace() or ""
        cache_key = f"endpoint_config:{workspace}:{endpoint_name}"
        if cached := self.secret_cache.get(cache_key):
            cached_dict = cached if isinstance(cached, dict) else json.loads(str(cached))
            return GatewayEndpointConfig.from_dict(cached_dict)
        target_ep: GatewayEndpointEntity | None = None
        for ep in self._list_gateway_endpoint_entities():
            if ep.name == endpoint_name:
                target_ep = ep
                break
        if target_ep is None:
            raise MlflowException(
                f"GatewayEndpoint not found (name='{endpoint_name}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        result = self._build_endpoint_config(target_ep)
        self.secret_cache.set(cache_key, result.to_dict())
        return result

    def get_resource_endpoint_configs(
        self, resource_type: str, resource_id: str
    ) -> list[GatewayEndpointConfig]:
        configs: list[GatewayEndpointConfig] = []
        for ep in self._list_gateway_endpoint_entities():
            for b in _unpack_list(ep.packed_bindings):
                if b.get("resource_type") == resource_type and b.get("resource_id") == resource_id:
                    cfg = self._build_endpoint_config(ep)
                    configs.append(
                        GatewayEndpointConfig(
                            endpoint_id=cfg.endpoint_id,
                            endpoint_name=cfg.endpoint_name,
                            models=cfg.models,
                            routing_strategy=cfg.routing_strategy,
                            fallback_config=cfg.fallback_config,
                        )
                    )
        return configs

    # ------------------------------------------------------------------
    # Gateway Budget Policies & Trace Cost
    # ------------------------------------------------------------------

    def _fetch_budget_policy_entity(self, budget_policy_id: str) -> GatewayBudgetPolicyEntity:
        client = self._gw_client()
        res = client.get_gateway_budget_policy(
            GetGatewayBudgetPolicyParams(scope=_SCOPE, budget_policy_id=budget_policy_id)
        )
        if isinstance(res.value, GetGatewayBudgetPolicyMissing) or not isinstance(
            res.value, GetGatewayBudgetPolicyFound
        ):
            raise MlflowException(
                f"BudgetPolicy not found (budget_policy_id='{budget_policy_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.policy
        return GatewayBudgetPolicyEntity(
            scope=_SCOPE,
            budget_policy_id=row.budget_policy_id,
            budget_unit=row.budget_unit,
            budget_amount_json=row.budget_amount_json,
            duration_unit=row.duration_unit,
            duration_value=int(row.duration_value),
            target_scope=row.target_scope,
            target_value=row.target_value,
            budget_action=row.budget_action,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
        )

    def _list_budget_policy_entities(self) -> list[GatewayBudgetPolicyEntity]:
        client = self._gw_client()
        entities: list[GatewayBudgetPolicyEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_gateway_budget_policies(
                ListGatewayBudgetPoliciesParams(scope=_SCOPE, limit=50, after=cursor)
            )
            if not isinstance(res.value, ListGatewayBudgetPoliciesFound):
                raise MlflowException("Invalid ListGatewayBudgetPolicies response from RiffDB")
            for row in res.value.policies:
                entities.append(
                    GatewayBudgetPolicyEntity(
                        scope=_SCOPE,
                        budget_policy_id=row.budget_policy_id,
                        budget_unit=row.budget_unit,
                        budget_amount_json=row.budget_amount_json,
                        duration_unit=row.duration_unit,
                        duration_value=int(row.duration_value),
                        target_scope=row.target_scope,
                        target_value=row.target_value,
                        budget_action=row.budget_action,
                        created_at_ms=int(row.created_at_ms),
                        last_updated_at_ms=int(row.last_updated_at_ms),
                        created_by=row.created_by,
                        last_updated_by=row.last_updated_by,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.policies:
                break
        return entities

    @staticmethod
    def _budget_policy_entity_to_mlflow(entity: GatewayBudgetPolicyEntity) -> GatewayBudgetPolicy:
        return GatewayBudgetPolicy(
            budget_policy_id=entity.budget_policy_id,
            budget_unit=BudgetUnit(entity.budget_unit),
            budget_amount=float(json.loads(entity.budget_amount_json)),
            duration=BudgetDuration(
                unit=BudgetDurationUnit(entity.duration_unit),
                value=int(entity.duration_value),
            ),
            target_scope=BudgetTargetScope(entity.target_scope),
            budget_action=BudgetAction(entity.budget_action),
            created_at=int(entity.created_at_ms),
            last_updated_at=int(entity.last_updated_at_ms),
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            target_value=entity.target_value,
        )

    def _save_budget_policy_entity(
        self, entity: GatewayBudgetPolicyEntity
    ) -> GatewayBudgetPolicyEntity:
        client = self._gw_client()
        res = client.upsert_gateway_budget_policy_entity(
            UpsertGatewayBudgetPolicyEntityInput(
                request_id=str(uuid.uuid4()),
                budget_policy_id=entity.budget_policy_id,
                budget_unit=entity.budget_unit,
                budget_amount_json=entity.budget_amount_json,
                duration_unit=entity.duration_unit,
                duration_value=int(entity.duration_value),
                target_scope=entity.target_scope,
                target_value=entity.target_value,
                budget_action=entity.budget_action,
                created_at_ms=int(entity.created_at_ms),
                last_updated_at_ms=int(entity.last_updated_at_ms),
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
            )
        )
        if not isinstance(res.outcome, UpsertGatewayBudgetPolicyEntityGatewayBudgetPolicyUpserted):
            raise MlflowException("Invalid UpsertGatewayBudgetPolicyEntity response from RiffDB")
        return res.outcome.policy

    def create_budget_policy(
        self,
        budget_unit: BudgetUnit,
        budget_amount: float,
        duration: BudgetDuration,
        target_scope: BudgetTargetScope,
        budget_action: BudgetAction,
        created_by: str | None = None,
        target_value: str | None = None,
    ) -> GatewayBudgetPolicy:
        scope_value = (
            target_scope.value if isinstance(target_scope, BudgetTargetScope) else str(target_scope)
        )
        normalized_target = _normalize_budget_target_value(scope_value, target_value)
        if scope_value == BudgetTargetScope.ENDPOINT.value and normalized_target is not None:
            self._fetch_gateway_endpoint_entity(normalized_target)

        budget_policy_id = f"bp-{uuid.uuid4().hex}"
        now = _now_ms()
        saved = self._save_budget_policy_entity(
            GatewayBudgetPolicyEntity(
                scope=_SCOPE,
                budget_policy_id=budget_policy_id,
                budget_unit=budget_unit.value
                if isinstance(budget_unit, BudgetUnit)
                else str(budget_unit),
                budget_amount_json=json.dumps(float(budget_amount)),
                duration_unit=duration.unit.value,
                duration_value=int(duration.value),
                target_scope=scope_value,
                target_value=normalized_target,
                budget_action=budget_action.value
                if isinstance(budget_action, BudgetAction)
                else str(budget_action),
                created_at_ms=now,
                last_updated_at_ms=now,
                created_by=created_by,
                last_updated_by=created_by,
            )
        )
        return self._budget_policy_entity_to_mlflow(saved)

    def get_budget_policy(self, budget_policy_id: str) -> GatewayBudgetPolicy:
        return self._budget_policy_entity_to_mlflow(
            self._fetch_budget_policy_entity(budget_policy_id)
        )

    def update_budget_policy(
        self,
        budget_policy_id: str,
        budget_unit: BudgetUnit | None = None,
        budget_amount: float | None = None,
        duration: BudgetDuration | None = None,
        target_scope: BudgetTargetScope | None = None,
        budget_action: BudgetAction | None = None,
        updated_by: str | None = None,
        target_value: str | None = None,
    ) -> GatewayBudgetPolicy:
        existing = self._fetch_budget_policy_entity(budget_policy_id)
        resolved_unit = (
            (budget_unit.value if isinstance(budget_unit, BudgetUnit) else str(budget_unit))
            if budget_unit is not None
            else existing.budget_unit
        )
        resolved_amount_json = (
            json.dumps(float(budget_amount))
            if budget_amount is not None
            else existing.budget_amount_json
        )
        resolved_dur_unit = duration.unit.value if duration is not None else existing.duration_unit
        resolved_dur_val = (
            int(duration.value) if duration is not None else int(existing.duration_value)
        )

        resolved_scope = existing.target_scope
        resolved_target = existing.target_value
        if target_scope is not None:
            scope_val = (
                target_scope.value
                if isinstance(target_scope, BudgetTargetScope)
                else str(target_scope)
            )
            if scope_val != existing.target_scope:
                resolved_target = None
            resolved_scope = scope_val
        if target_value is not None:
            resolved_target = target_value
        resolved_target = _normalize_budget_target_value(resolved_scope, resolved_target)
        if (
            target_value is not None
            and resolved_scope == BudgetTargetScope.ENDPOINT.value
            and resolved_target is not None
        ):
            self._fetch_gateway_endpoint_entity(resolved_target)

        resolved_action = (
            (budget_action.value if isinstance(budget_action, BudgetAction) else str(budget_action))
            if budget_action is not None
            else existing.budget_action
        )
        saved = self._save_budget_policy_entity(
            GatewayBudgetPolicyEntity(
                scope=_SCOPE,
                budget_policy_id=budget_policy_id,
                budget_unit=resolved_unit,
                budget_amount_json=resolved_amount_json,
                duration_unit=resolved_dur_unit,
                duration_value=resolved_dur_val,
                target_scope=resolved_scope,
                target_value=resolved_target,
                budget_action=resolved_action,
                created_at_ms=int(existing.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=existing.created_by,
                last_updated_by=updated_by if updated_by is not None else existing.last_updated_by,
            )
        )
        return self._budget_policy_entity_to_mlflow(saved)

    def delete_budget_policy(self, budget_policy_id: str) -> None:
        self._fetch_budget_policy_entity(budget_policy_id)
        client = self._gw_client()
        res = client.delete_gateway_budget_policy_entity(
            DeleteGatewayBudgetPolicyEntityInput(
                request_id=str(uuid.uuid4()), budget_policy_id=budget_policy_id
            )
        )
        if isinstance(res.outcome, DeleteGatewayBudgetPolicyEntityGatewayBudgetPolicyMissing):
            raise MlflowException(
                f"BudgetPolicy not found (budget_policy_id='{budget_policy_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteGatewayBudgetPolicyEntityGatewayBudgetPolicyDeleted):
            raise MlflowException("Invalid DeleteGatewayBudgetPolicyEntity response from RiffDB")

    def list_budget_policies(
        self,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        page_token: str | None = None,
    ) -> PagedList[GatewayBudgetPolicy]:
        self._validate_gateway_max_results(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)  # type: ignore[no-untyped-call]
        policies = [
            self._budget_policy_entity_to_mlflow(e) for e in self._list_budget_policy_entities()
        ]
        policies.sort(key=lambda p: p.budget_policy_id)
        sliced = policies[offset : offset + max_results + 1]
        next_token = None
        if len(sliced) > max_results:
            next_token = SearchUtils.create_page_token(offset + max_results)  # type: ignore[no-untyped-call]
        return PagedList(sliced[:max_results], next_token)

    def sum_gateway_trace_cost(
        self,
        start_time_ms: int,
        end_time_ms: int,
        workspace: str | None = None,
        endpoint_id: str | None = None,
    ) -> float:
        client = self._gw_client()
        total_micros = 0
        for exp in self.search_experiments(ViewType.ALL):  # type: ignore[no-untyped-call]
            exp_id = int(exp.experiment_id)
            cursor: str | None = None
            while True:
                res = client.search_traces(
                    SearchTracesParams(experiment_id=exp_id, limit=25, after=cursor)
                )
                if not isinstance(res.value, SearchTracesFound):
                    raise MlflowException("Invalid SearchTraces response from RiffDB")
                for row in res.value.traces:
                    if not row.initialized:
                        continue
                    req_ms = int(row.request_time_ms)
                    if req_ms < start_time_ms or req_ms >= end_time_ms:
                        continue
                    meta = _unpack_dict(row.packed_trace_metadata)
                    ep_val = meta.get(TraceMetadataKey.GATEWAY_ENDPOINT_ID)
                    if ep_val is None:
                        continue
                    if endpoint_id is not None and ep_val != endpoint_id:
                        continue
                    total_micros += int(row.total_cost_micros)
                cursor = res.next_cursor
                if not cursor or not res.value.traces:
                    break
        return float(total_micros) / 1_000_000.0

    # ------------------------------------------------------------------
    # Gateway Guardrails & Endpoint Guardrail Configs
    # ------------------------------------------------------------------

    def _fetch_gateway_guardrail_entity(self, guardrail_id: str) -> GatewayGuardrailEntity:
        client = self._gw_client()
        res = client.get_gateway_guardrail(
            GetGatewayGuardrailParams(scope=_SCOPE, guardrail_id=guardrail_id)
        )
        if isinstance(res.value, GetGatewayGuardrailMissing) or not isinstance(
            res.value, GetGatewayGuardrailFound
        ):
            raise MlflowException(
                f"Guardrail not found (guardrail_id='{guardrail_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.guardrail
        return GatewayGuardrailEntity(
            scope=_SCOPE,
            guardrail_id=row.guardrail_id,
            name=row.name,
            scorer_id=row.scorer_id,
            scorer_version=int(row.scorer_version),
            stage=row.stage,
            action=row.action,
            action_endpoint_id=row.action_endpoint_id,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
        )

    def _list_gateway_guardrail_entities(self) -> list[GatewayGuardrailEntity]:
        client = self._gw_client()
        entities: list[GatewayGuardrailEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_gateway_guardrails(
                ListGatewayGuardrailsParams(scope=_SCOPE, limit=50, after=cursor)
            )
            if not isinstance(res.value, ListGatewayGuardrailsFound):
                raise MlflowException("Invalid ListGatewayGuardrails response from RiffDB")
            for row in res.value.guardrails:
                entities.append(
                    GatewayGuardrailEntity(
                        scope=_SCOPE,
                        guardrail_id=row.guardrail_id,
                        name=row.name,
                        scorer_id=row.scorer_id,
                        scorer_version=int(row.scorer_version),
                        stage=row.stage,
                        action=row.action,
                        action_endpoint_id=row.action_endpoint_id,
                        created_at_ms=int(row.created_at_ms),
                        last_updated_at_ms=int(row.last_updated_at_ms),
                        created_by=row.created_by,
                        last_updated_by=row.last_updated_by,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.guardrails:
                break
        return entities

    def _save_gateway_guardrail_entity(
        self, entity: GatewayGuardrailEntity
    ) -> GatewayGuardrailEntity:
        client = self._gw_client()
        res = client.upsert_gateway_guardrail_entity(
            UpsertGatewayGuardrailEntityInput(
                request_id=str(uuid.uuid4()),
                guardrail_id=entity.guardrail_id,
                name=entity.name,
                scorer_id=entity.scorer_id,
                scorer_version=int(entity.scorer_version),
                stage=entity.stage,
                action=entity.action,
                action_endpoint_id=entity.action_endpoint_id,
                created_at_ms=int(entity.created_at_ms),
                last_updated_at_ms=int(entity.last_updated_at_ms),
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
            )
        )
        if not isinstance(res.outcome, UpsertGatewayGuardrailEntityGatewayGuardrailUpserted):
            raise MlflowException("Invalid UpsertGatewayGuardrailEntity response from RiffDB")
        return res.outcome.guardrail

    def _guardrail_entity_to_mlflow(self, entity: GatewayGuardrailEntity) -> GatewayGuardrail:
        scorer_version = self._get_scorer_version_by_id(
            entity.scorer_id, int(entity.scorer_version)
        )
        action_endpoint_name: str | None = None
        if entity.action_endpoint_id is not None:
            try:
                action_endpoint_name = self._fetch_gateway_endpoint_entity(
                    entity.action_endpoint_id
                ).name
            except MlflowException:
                action_endpoint_name = None
        return GatewayGuardrail(
            guardrail_id=entity.guardrail_id,
            name=entity.name,
            scorer=scorer_version,
            stage=GuardrailStage(entity.stage),
            action=GuardrailAction(entity.action),
            action_endpoint_name=action_endpoint_name,
            created_at=int(entity.created_at_ms),
            last_updated_at=int(entity.last_updated_at_ms),
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
        )

    def create_gateway_guardrail(
        self,
        name: str,
        scorer_id: str,
        scorer_version: int,
        stage: GuardrailStage,
        action: GuardrailAction,
        action_endpoint_id: str | None = None,
        created_by: str | None = None,
    ) -> GatewayGuardrail:
        self._get_scorer_version_by_id(scorer_id, int(scorer_version))
        guardrail_id = f"gr-{uuid.uuid4().hex}"
        now = _now_ms()
        saved = self._save_gateway_guardrail_entity(
            GatewayGuardrailEntity(
                scope=_SCOPE,
                guardrail_id=guardrail_id,
                name=name,
                scorer_id=scorer_id,
                scorer_version=int(scorer_version),
                stage=stage.value if isinstance(stage, GuardrailStage) else str(stage),
                action=action.value if isinstance(action, GuardrailAction) else str(action),
                action_endpoint_id=action_endpoint_id,
                created_at_ms=now,
                last_updated_at_ms=now,
                created_by=created_by,
                last_updated_by=created_by,
            )
        )
        return self._guardrail_entity_to_mlflow(saved)

    def get_gateway_guardrail(self, guardrail_id: str) -> GatewayGuardrail:
        return self._guardrail_entity_to_mlflow(self._fetch_gateway_guardrail_entity(guardrail_id))

    def delete_gateway_guardrail(self, guardrail_id: str) -> None:
        self._fetch_gateway_guardrail_entity(guardrail_id)
        for ep in self._list_gateway_endpoint_entities():
            configs = _unpack_list(ep.packed_guardrail_configs)
            kept = [c for c in configs if c.get("guardrail_id") != guardrail_id]
            if len(kept) != len(configs):
                self._save_gateway_endpoint_entity(
                    GatewayEndpointEntity(
                        scope=_SCOPE,
                        endpoint_id=ep.endpoint_id,
                        name=ep.name,
                        routing_strategy=ep.routing_strategy,
                        fallback_config_json=ep.fallback_config_json,
                        experiment_id=ep.experiment_id,
                        usage_tracking=ep.usage_tracking,
                        packed_model_mappings=ep.packed_model_mappings,
                        packed_tags=ep.packed_tags,
                        packed_bindings=ep.packed_bindings,
                        packed_guardrail_configs=_pack_json(kept),
                        created_at_ms=int(ep.created_at_ms),
                        last_updated_at_ms=_now_ms(),
                        created_by=ep.created_by,
                        last_updated_by=ep.last_updated_by,
                    )
                )
        client = self._gw_client()
        res = client.delete_gateway_guardrail_entity(
            DeleteGatewayGuardrailEntityInput(
                request_id=str(uuid.uuid4()), guardrail_id=guardrail_id
            )
        )
        if isinstance(res.outcome, DeleteGatewayGuardrailEntityGatewayGuardrailMissing):
            raise MlflowException(
                f"Guardrail not found (guardrail_id='{guardrail_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteGatewayGuardrailEntityGatewayGuardrailDeleted):
            raise MlflowException("Invalid DeleteGatewayGuardrailEntity response from RiffDB")

    def list_gateway_guardrails(
        self,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        page_token: str | None = None,
    ) -> PagedList[GatewayGuardrail]:
        self._validate_gateway_max_results(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)  # type: ignore[no-untyped-call]
        guardrails = [
            self._guardrail_entity_to_mlflow(g) for g in self._list_gateway_guardrail_entities()
        ]
        guardrails.sort(key=lambda g: g.guardrail_id)
        sliced = guardrails[offset : offset + max_results + 1]
        next_token = None
        if len(sliced) > max_results:
            next_token = SearchUtils.create_page_token(offset + max_results)  # type: ignore[no-untyped-call]
        return PagedList(sliced[:max_results], next_token)

    def _guardrail_config_dict_to_mlflow(
        self, endpoint_id: str, cfg: dict[str, Any]
    ) -> GatewayGuardrailConfig:
        guardrail_id = str(cfg["guardrail_id"])
        guardrail_mlflow: GatewayGuardrail | None = None
        try:
            guardrail_mlflow = self.get_gateway_guardrail(guardrail_id)
        except MlflowException:
            guardrail_mlflow = None
        return GatewayGuardrailConfig(
            endpoint_id=endpoint_id,
            guardrail_id=guardrail_id,
            execution_order=cfg.get("execution_order"),
            created_at=int(cfg.get("created_at", 0)),
            guardrail=guardrail_mlflow,
            created_by=cfg.get("created_by"),
        )

    def add_guardrail_to_endpoint(
        self,
        endpoint_id: str,
        guardrail_id: str,
        execution_order: int | None = None,
        created_by: str | None = None,
    ) -> GatewayGuardrailConfig:
        ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        self._fetch_gateway_guardrail_entity(guardrail_id)
        configs = _unpack_list(ep.packed_guardrail_configs)
        for c in configs:
            if c.get("guardrail_id") == guardrail_id:
                raise MlflowException(
                    f"Guardrail '{guardrail_id}' is already added to endpoint '{endpoint_id}'",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
        now = _now_ms()
        new_cfg = {
            "endpoint_id": endpoint_id,
            "guardrail_id": guardrail_id,
            "execution_order": execution_order,
            "created_by": created_by,
            "created_at": now,
        }
        configs.append(new_cfg)
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=ep.packed_model_mappings,
                packed_tags=ep.packed_tags,
                packed_bindings=ep.packed_bindings,
                packed_guardrail_configs=_pack_json(configs),
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=now,
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )
        return self._guardrail_config_dict_to_mlflow(endpoint_id, new_cfg)

    def update_endpoint_guardrail_config(
        self,
        endpoint_id: str,
        guardrail_id: str,
        execution_order: int | None = None,
    ) -> GatewayGuardrailConfig:
        try:
            ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        except MlflowException as e:
            raise MlflowException(
                f"GuardrailConfig not found (endpoint_id='{endpoint_id}', "
                f"guardrail_id='{guardrail_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e
        configs = _unpack_list(ep.packed_guardrail_configs)
        updated_cfg: dict[str, Any] | None = None
        for c in configs:
            if c.get("guardrail_id") == guardrail_id:
                c["execution_order"] = execution_order
                updated_cfg = c
                break
        if updated_cfg is None:
            raise MlflowException(
                f"GuardrailConfig not found (endpoint_id='{endpoint_id}', "
                f"guardrail_id='{guardrail_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=ep.packed_model_mappings,
                packed_tags=ep.packed_tags,
                packed_bindings=ep.packed_bindings,
                packed_guardrail_configs=_pack_json(configs),
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )
        return self._guardrail_config_dict_to_mlflow(endpoint_id, updated_cfg)

    def remove_guardrail_from_endpoint(
        self,
        endpoint_id: str,
        guardrail_id: str,
    ) -> None:
        try:
            ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        except MlflowException as e:
            raise MlflowException(
                f"GuardrailConfig not found (endpoint_id='{endpoint_id}', "
                f"guardrail_id='{guardrail_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e
        configs = _unpack_list(ep.packed_guardrail_configs)
        kept = [c for c in configs if c.get("guardrail_id") != guardrail_id]
        if len(kept) == len(configs):
            raise MlflowException(
                f"GuardrailConfig not found (endpoint_id='{endpoint_id}', "
                f"guardrail_id='{guardrail_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        self._save_gateway_endpoint_entity(
            GatewayEndpointEntity(
                scope=_SCOPE,
                endpoint_id=ep.endpoint_id,
                name=ep.name,
                routing_strategy=ep.routing_strategy,
                fallback_config_json=ep.fallback_config_json,
                experiment_id=ep.experiment_id,
                usage_tracking=ep.usage_tracking,
                packed_model_mappings=ep.packed_model_mappings,
                packed_tags=ep.packed_tags,
                packed_bindings=ep.packed_bindings,
                packed_guardrail_configs=_pack_json(kept),
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=_now_ms(),
                created_by=ep.created_by,
                last_updated_by=ep.last_updated_by,
            )
        )

    def list_endpoint_guardrail_configs(
        self,
        endpoint_id: str,
    ) -> list[GatewayGuardrailConfig]:
        ep = self._fetch_gateway_endpoint_entity(endpoint_id)
        configs = _unpack_list(ep.packed_guardrail_configs)
        configs.sort(
            key=lambda c: (
                1 if c.get("execution_order") is None else 0,
                int(c.get("execution_order") or 0),
                int(c.get("created_at") or 0),
            )
        )
        return [self._guardrail_config_dict_to_mlflow(endpoint_id, c) for c in configs]

    # ------------------------------------------------------------------
    # Scorers & Online Scoring
    # ------------------------------------------------------------------

    def _fetch_scorer_entity(self, scorer_id: str) -> ScorerEntity:
        client = self._gw_client()
        res = client.get_scorer(GetScorerParams(scope=_SCOPE, scorer_id=scorer_id))
        if isinstance(res.value, GetScorerMissing) or not isinstance(res.value, GetScorerFound):
            raise MlflowException(
                f"Scorer with ID '{scorer_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.scorer
        return ScorerEntity(
            scope=_SCOPE,
            scorer_id=row.scorer_id,
            experiment_id=int(row.experiment_id),
            scorer_name=row.scorer_name,
            latest_version=int(row.latest_version),
            packed_versions=row.packed_versions,
            packed_online_config=row.packed_online_config,
        )

    def _list_scorer_entities(self) -> list[ScorerEntity]:
        client = self._gw_client()
        entities: list[ScorerEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_scorers(ListScorersParams(scope=_SCOPE, limit=10, after=cursor))
            if not isinstance(res.value, ListScorersFound):
                raise MlflowException("Invalid ListScorers response from RiffDB")
            for row in res.value.scorers:
                entities.append(
                    ScorerEntity(
                        scope=_SCOPE,
                        scorer_id=row.scorer_id,
                        experiment_id=int(row.experiment_id),
                        scorer_name=row.scorer_name,
                        latest_version=int(row.latest_version),
                        packed_versions=row.packed_versions,
                        packed_online_config=row.packed_online_config,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.scorers:
                break
        return entities

    def _save_scorer_entity(self, entity: ScorerEntity) -> ScorerEntity:
        client = self._gw_client()
        res = client.upsert_scorer_entity(
            UpsertScorerEntityInput(
                request_id=str(uuid.uuid4()),
                scorer_id=entity.scorer_id,
                experiment_id=int(entity.experiment_id),
                scorer_name=entity.scorer_name,
                latest_version=int(entity.latest_version),
                packed_versions=entity.packed_versions,
                packed_online_config=entity.packed_online_config,
            )
        )
        if not isinstance(res.outcome, UpsertScorerEntityScorerUpserted):
            raise MlflowException("Invalid UpsertScorerEntity response from RiffDB")
        return res.outcome.scorer

    def _find_scorer_by_experiment_and_name(
        self, experiment_id: int, name: str
    ) -> ScorerEntity | None:
        for entity in self._list_scorer_entities():
            if int(entity.experiment_id) == experiment_id and entity.scorer_name == name:
                return entity
        return None

    def _get_scorer_version_by_id(
        self, scorer_id: str, version: int | None = None
    ) -> ScorerVersion:
        scorer = self._fetch_scorer_entity(scorer_id)
        versions = _unpack_list(scorer.packed_versions)
        if not versions:
            raise MlflowException(
                f"Scorer with ID '{scorer_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if version is not None:
            for v in versions:
                if int(v.get("scorer_version", 0)) == int(version):
                    return ScorerVersion(
                        experiment_id=str(scorer.experiment_id),
                        scorer_name=scorer.scorer_name,
                        scorer_version=int(v["scorer_version"]),
                        serialized_scorer=str(v["serialized_scorer"]),
                        creation_time=int(v.get("creation_time", 0)),
                        scorer_id=scorer.scorer_id,
                    )
            raise MlflowException(
                f"Scorer with ID '{scorer_id}' and version {version} not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        latest = max(versions, key=lambda v: int(v.get("scorer_version", 0)))
        return ScorerVersion(
            experiment_id=str(scorer.experiment_id),
            scorer_name=scorer.scorer_name,
            scorer_version=int(latest["scorer_version"]),
            serialized_scorer=str(latest["serialized_scorer"]),
            creation_time=int(latest.get("creation_time", 0)),
            scorer_id=scorer.scorer_id,
        )

    def resolve_endpoint_in_scorer(self, scorer_version: ScorerVersion) -> ScorerVersion:
        from mlflow.genai.scorers.scorer_utils import (
            build_gateway_model,
            extract_endpoint_ref,
            extract_model_from_serialized_scorer,
            is_gateway_model,
            update_model_in_serialized_scorer,
        )

        serialized_data = json.loads(scorer_version._serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)
        if not is_gateway_model(model) or model is None:
            return scorer_version
        endpoint_ref = extract_endpoint_ref(model)
        try:
            endpoint = self.get_gateway_endpoint(endpoint_id=endpoint_ref)
            new_model: str | None = build_gateway_model(endpoint.name) if endpoint.name else None
        except MlflowException:
            new_model = None
        serialized_data = update_model_in_serialized_scorer(serialized_data, new_model)
        return ScorerVersion(
            experiment_id=scorer_version.experiment_id,
            scorer_id=scorer_version.scorer_id,
            scorer_version=scorer_version.scorer_version,
            scorer_name=scorer_version.scorer_name,
            serialized_scorer=json.dumps(serialized_data),
            creation_time=scorer_version.creation_time,
        )

    def _resolve_endpoint_in_serialized_scorer(self, serialized_scorer: str) -> str:
        from mlflow.genai.scorers.scorer_utils import (
            build_gateway_model,
            extract_endpoint_ref,
            extract_model_from_serialized_scorer,
            is_gateway_model,
            update_model_in_serialized_scorer,
        )

        serialized_data = json.loads(serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)
        if is_gateway_model(model) and model is not None:
            endpoint_id = extract_endpoint_ref(model)
            try:
                endpoint = self.get_gateway_endpoint(endpoint_id=endpoint_id)
                if endpoint.name:
                    new_model = build_gateway_model(endpoint.name)
                    serialized_data = update_model_in_serialized_scorer(serialized_data, new_model)
            except MlflowException:
                pass
        return json.dumps(serialized_data)

    def _batch_resolve_endpoint_in_serialized_scorers(
        self, serialized_scorers: list[str]
    ) -> list[str]:
        from mlflow.genai.scorers.scorer_utils import (
            build_gateway_model,
            extract_endpoint_ref,
            extract_model_from_serialized_scorer,
            is_gateway_model,
            update_model_in_serialized_scorer,
        )

        resolved: list[str] = []
        id_to_name: dict[str, str] = {}
        for serialized in serialized_scorers:
            serialized_data = json.loads(serialized)
            model = extract_model_from_serialized_scorer(serialized_data)
            if is_gateway_model(model) and model is not None:
                if not id_to_name:
                    all_endpoints = self._list_gateway_endpoint_entities()
                    id_to_name = {ep.endpoint_id: ep.name for ep in all_endpoints}
                endpoint_id = extract_endpoint_ref(model)
                endpoint_name = id_to_name.get(endpoint_id) if endpoint_id is not None else None
                new_model = build_gateway_model(endpoint_name) if endpoint_name else None
                serialized_data = update_model_in_serialized_scorer(serialized_data, new_model)
            resolved.append(json.dumps(serialized_data))
        return resolved

    def _remove_scorer_bindings(self, scorer_id: str) -> None:
        for ep in self._list_gateway_endpoint_entities():
            bindings = _unpack_list(ep.packed_bindings)
            kept = [
                b
                for b in bindings
                if not (
                    b.get("resource_type") == GatewayResourceType.SCORER.value
                    and b.get("resource_id") == scorer_id
                )
            ]
            if len(kept) != len(bindings):
                self._save_gateway_endpoint_entity(
                    GatewayEndpointEntity(
                        scope=_SCOPE,
                        endpoint_id=ep.endpoint_id,
                        name=ep.name,
                        routing_strategy=ep.routing_strategy,
                        fallback_config_json=ep.fallback_config_json,
                        experiment_id=ep.experiment_id,
                        usage_tracking=ep.usage_tracking,
                        packed_model_mappings=ep.packed_model_mappings,
                        packed_tags=ep.packed_tags,
                        packed_bindings=_pack_json(kept),
                        packed_guardrail_configs=ep.packed_guardrail_configs,
                        created_at_ms=int(ep.created_at_ms),
                        last_updated_at_ms=_now_ms(),
                        created_by=ep.created_by,
                        last_updated_by=ep.last_updated_by,
                    )
                )

    def register_scorer(
        self, experiment_id: str, name: str, serialized_scorer: str
    ) -> ScorerVersion:
        from mlflow.genai.scorers.scorer_utils import (
            build_gateway_model,
            extract_endpoint_ref,
            extract_model_from_serialized_scorer,
            is_gateway_model,
            update_model_in_serialized_scorer,
            validate_scorer_model,
            validate_scorer_name,
        )

        validate_scorer_name(name)
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        self._require_active_experiment(exp_id)  # type: ignore[attr-defined]

        serialized_data = json.loads(serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)
        validate_scorer_model(model)

        endpoint_id: str | None = None
        if is_gateway_model(model) and model is not None:
            endpoint_name = extract_endpoint_ref(model)
            endpoint = self.get_gateway_endpoint(name=endpoint_name)
            endpoint_id = endpoint.endpoint_id
            serialized_data = update_model_in_serialized_scorer(
                serialized_data, build_gateway_model(endpoint.endpoint_id)
            )
            serialized_scorer = json.dumps(serialized_data)

        existing = self._find_scorer_by_experiment_and_name(exp_id, name)
        now = _now_ms()
        if existing is None:
            scorer_id = str(uuid.uuid4())
            versions: list[dict[str, Any]] = []
            new_version = 1
            online_cfg_bytes = b""
        else:
            scorer_id = existing.scorer_id
            versions = _unpack_list(existing.packed_versions)
            max_ver = max((int(v.get("scorer_version", 0)) for v in versions), default=0)
            new_version = max(max_ver, int(existing.latest_version)) + 1
            online_cfg_bytes = existing.packed_online_config

        version_entry = {
            "scorer_version": new_version,
            "serialized_scorer": serialized_scorer,
            "creation_time": now,
        }
        versions.append(version_entry)
        self._save_scorer_entity(
            ScorerEntity(
                scope=_SCOPE,
                scorer_id=scorer_id,
                experiment_id=exp_id,
                scorer_name=name,
                latest_version=new_version,
                packed_versions=_pack_json(versions),
                packed_online_config=online_cfg_bytes,
            )
        )

        if endpoint_id is not None:
            self._remove_scorer_bindings(scorer_id)
            self.create_endpoint_binding(
                endpoint_id=endpoint_id,
                resource_type=GatewayResourceType.SCORER.value,
                resource_id=scorer_id,
                display_name=name,
            )

        sv = ScorerVersion(
            experiment_id=str(exp_id),
            scorer_name=name,
            scorer_version=new_version,
            serialized_scorer=serialized_scorer,
            creation_time=now,
            scorer_id=scorer_id,
        )
        return self.resolve_endpoint_in_scorer(sv)

    def list_scorers(self, experiment_id: str) -> list[ScorerVersion]:
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        self._require_active_experiment(exp_id)  # type: ignore[attr-defined]
        return self.list_scorers_across_experiments([str(exp_id)])

    def list_scorers_across_experiments(self, experiment_ids: list[str]) -> list[ScorerVersion]:
        if not experiment_ids:
            return []
        target_exp_ids = {int(e) for e in experiment_ids}
        candidates: list[tuple[ScorerEntity, dict[str, Any]]] = []
        for scorer in self._list_scorer_entities():
            if int(scorer.experiment_id) not in target_exp_ids:
                continue
            versions = _unpack_list(scorer.packed_versions)
            if not versions:
                continue
            latest = max(versions, key=lambda v: int(v.get("scorer_version", 0)))
            candidates.append((scorer, latest))

        candidates.sort(key=lambda pair: (int(pair[0].experiment_id), pair[0].scorer_name))
        resolved_serialized = self._batch_resolve_endpoint_in_serialized_scorers(
            [str(pair[1]["serialized_scorer"]) for pair in candidates]
        )
        return [
            ScorerVersion(
                experiment_id=str(pair[0].experiment_id),
                scorer_id=pair[0].scorer_id,
                scorer_version=int(pair[1]["scorer_version"]),
                scorer_name=pair[0].scorer_name,
                serialized_scorer=resolved_serialized[idx],
                creation_time=int(pair[1].get("creation_time", 0)),
            )
            for idx, pair in enumerate(candidates)
        ]

    def get_scorer(
        self, experiment_id: str, name: str, version: int | None = None
    ) -> ScorerVersion:
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        self._require_active_experiment(exp_id)  # type: ignore[attr-defined]
        scorer = self._find_scorer_by_experiment_and_name(exp_id, name)
        if scorer is None:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        try:
            sv = self._get_scorer_version_by_id(
                scorer.scorer_id, int(version) if version is not None else None
            )
        except MlflowException as e:
            if version is not None:
                raise MlflowException(
                    f"Scorer with name '{name}' and version {version} not found for "
                    f"experiment {experiment_id}.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from e
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e
        return self.resolve_endpoint_in_scorer(sv)

    def delete_scorer(self, experiment_id: str, name: str, version: int | None = None) -> None:
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        self._require_active_experiment(exp_id)  # type: ignore[attr-defined]
        scorer = self._find_scorer_by_experiment_and_name(exp_id, name)
        if scorer is None:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        versions = _unpack_list(scorer.packed_versions)
        if version is not None:
            kept = [v for v in versions if int(v.get("scorer_version", 0)) != int(version)]
            if len(kept) == len(versions):
                raise MlflowException(
                    f"Scorer with name '{name}' and version {version} not found for"
                    f" experiment {experiment_id}.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            self._save_scorer_entity(
                ScorerEntity(
                    scope=_SCOPE,
                    scorer_id=scorer.scorer_id,
                    experiment_id=int(scorer.experiment_id),
                    scorer_name=scorer.scorer_name,
                    latest_version=int(scorer.latest_version),
                    packed_versions=_pack_json(kept),
                    packed_online_config=scorer.packed_online_config,
                )
            )
            return

        if not versions:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        self._remove_scorer_bindings(scorer.scorer_id)
        client = self._gw_client()
        res = client.delete_scorer_entity(
            DeleteScorerEntityInput(request_id=str(uuid.uuid4()), scorer_id=scorer.scorer_id)
        )
        if isinstance(res.outcome, DeleteScorerEntityScorerMissing):
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteScorerEntityScorerDeleted):
            raise MlflowException("Invalid DeleteScorerEntity response from RiffDB")

    def list_scorer_versions(self, experiment_id: str, name: str) -> list[ScorerVersion]:
        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        self._require_active_experiment(exp_id)  # type: ignore[attr-defined]
        scorer = self._find_scorer_by_experiment_and_name(exp_id, name)
        if scorer is None:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        versions = _unpack_list(scorer.packed_versions)
        if not versions:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        versions.sort(key=lambda v: int(v.get("scorer_version", 0)))
        resolved_serialized = self._batch_resolve_endpoint_in_serialized_scorers(
            [str(v["serialized_scorer"]) for v in versions]
        )
        return [
            ScorerVersion(
                experiment_id=str(scorer.experiment_id),
                scorer_id=scorer.scorer_id,
                scorer_version=int(v["scorer_version"]),
                scorer_name=scorer.scorer_name,
                serialized_scorer=resolved_serialized[idx],
                creation_time=int(v.get("creation_time", 0)),
            )
            for idx, v in enumerate(versions)
        ]

    def get_online_scoring_configs(self, scorer_ids: list[str]) -> list[OnlineScoringConfig]:
        from mlflow.genai.scorers.online.entities import OnlineScoringConfig

        if not scorer_ids:
            return []
        target_ids = set(scorer_ids)
        results: list[OnlineScoringConfig] = []
        for scorer in self._list_scorer_entities():
            if scorer.scorer_id not in target_ids:
                continue
            cfg_dict = _unpack_dict(scorer.packed_online_config)
            if not cfg_dict or "online_scoring_config_id" not in cfg_dict:
                continue
            results.append(
                OnlineScoringConfig(
                    online_scoring_config_id=str(cfg_dict["online_scoring_config_id"]),
                    scorer_id=scorer.scorer_id,
                    sample_rate=float(cfg_dict["sample_rate"]),
                    experiment_id=str(cfg_dict.get("experiment_id", scorer.experiment_id)),
                    filter_string=cfg_dict.get("filter_string"),
                )
            )
        return results

    def upsert_online_scoring_config(
        self,
        experiment_id: str,
        scorer_name: str,
        sample_rate: float,
        filter_string: str | None = None,
    ) -> OnlineScoringConfig:
        from mlflow.genai.scorers.online.entities import OnlineScoringConfig
        from mlflow.genai.scorers.scorer_utils import (
            extract_model_from_serialized_scorer,
            is_gateway_model,
        )

        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int | float):
            raise MlflowException.invalid_parameter_value(
                f"sample_rate must be a number, got {type(sample_rate).__name__}"
            )
        if not 0.0 <= float(sample_rate) <= 1.0:
            raise MlflowException.invalid_parameter_value(
                f"sample_rate must be between 0.0 and 1.0, got {sample_rate}"
            )
        if filter_string:
            if not isinstance(filter_string, str):
                raise MlflowException.invalid_parameter_value(
                    f"filter_string must be a string, got {type(filter_string).__name__}"
                )
            SearchTraceUtils.parse_search_filter_for_search_traces(filter_string)  # type: ignore[no-untyped-call]

        exp_id = int(self._numeric_experiment_id(experiment_id))  # type: ignore[attr-defined]
        self._require_active_experiment(exp_id)  # type: ignore[attr-defined]
        scorer = self._find_scorer_by_experiment_and_name(exp_id, scorer_name)
        if scorer is None:
            raise MlflowException(
                f"Scorer with name '{scorer_name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        versions = _unpack_list(scorer.packed_versions)
        if versions and sample_rate > 0:
            latest = max(versions, key=lambda v: int(v.get("scorer_version", 0)))
            serialized_str = str(latest["serialized_scorer"])
            serialized_data = json.loads(serialized_str)
            model = extract_model_from_serialized_scorer(serialized_data)
            if not is_gateway_model(model):
                raise MlflowException(
                    f"Scorer '{scorer_name}' does not use a gateway model. "
                    "Automatic evaluation is only supported for scorers that use "
                    "gateway models.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            try:
                from mlflow.genai.judges.instructions_judge import (
                    EXPECTATIONS_FIELD,
                    InstructionsJudge,
                )
                from mlflow.genai.scorers.base import Scorer

                scorer_obj = Scorer.model_validate_json(serialized_str)
            except Exception:
                scorer_obj = None
            if (
                scorer_obj is not None
                and isinstance(scorer_obj, InstructionsJudge)
                and EXPECTATIONS_FIELD in scorer_obj.get_input_fields()
            ):
                raise MlflowException(
                    f"Scorer '{scorer_name}' requires expectations, but scorers with "
                    "expectations are not currently supported for automatic evaluation.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        config_id = uuid.uuid4().hex
        cfg_payload = {
            "online_scoring_config_id": config_id,
            "scorer_id": scorer.scorer_id,
            "sample_rate": float(sample_rate),
            "experiment_id": str(exp_id),
            "filter_string": filter_string,
        }
        self._save_scorer_entity(
            ScorerEntity(
                scope=_SCOPE,
                scorer_id=scorer.scorer_id,
                experiment_id=int(scorer.experiment_id),
                scorer_name=scorer.scorer_name,
                latest_version=int(scorer.latest_version),
                packed_versions=scorer.packed_versions,
                packed_online_config=_pack_json(cfg_payload),
            )
        )
        return OnlineScoringConfig(
            online_scoring_config_id=config_id,
            scorer_id=scorer.scorer_id,
            sample_rate=float(sample_rate),
            experiment_id=str(exp_id),
            filter_string=filter_string,
        )

    def get_active_online_scorers(self) -> list[OnlineScorer]:
        from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
        from mlflow.genai.scorers.scorer_utils import (
            extract_model_from_serialized_scorer,
            is_gateway_model,
        )

        results: list[OnlineScorer] = []
        for scorer in self._list_scorer_entities():
            cfg_dict = _unpack_dict(scorer.packed_online_config)
            if not cfg_dict or "online_scoring_config_id" not in cfg_dict:
                continue
            sample_rate = float(cfg_dict.get("sample_rate", 0.0))
            if sample_rate <= 0.0:
                continue
            versions = _unpack_list(scorer.packed_versions)
            if not versions:
                continue
            latest = max(versions, key=lambda v: int(v.get("scorer_version", 0)))
            serialized_str = str(latest["serialized_scorer"])
            serialized_data = json.loads(serialized_str)
            model = extract_model_from_serialized_scorer(serialized_data)
            if not is_gateway_model(model):
                continue
            online_cfg = OnlineScoringConfig(
                online_scoring_config_id=str(cfg_dict["online_scoring_config_id"]),
                scorer_id=scorer.scorer_id,
                sample_rate=sample_rate,
                experiment_id=str(cfg_dict.get("experiment_id", scorer.experiment_id)),
                filter_string=cfg_dict.get("filter_string"),
            )
            results.append(
                OnlineScorer(
                    name=scorer.scorer_name,
                    serialized_scorer=self._resolve_endpoint_in_serialized_scorer(serialized_str),
                    online_config=online_cfg,
                )
            )
        return results

    def find_completed_sessions(
        self,
        experiment_id: str,
        min_last_trace_timestamp_ms: int,
        max_last_trace_timestamp_ms: int,
        max_results: int | None = None,
        filter_string: str | None = None,
    ) -> list[CompletedSession]:
        from mlflow.genai.scorers.online.entities import CompletedSession

        try:
            exp_id = int(experiment_id)
        except (ValueError, TypeError) as err:
            raise MlflowException(
                f"Invalid experiment ID '{experiment_id}'. Experiment ID must be a valid integer.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from err

        traces_list, _ = self.search_traces(
            experiment_ids=[str(exp_id)],
            filter_string=None,
            max_results=10000,
        )
        session_traces: dict[str, list[Any]] = {}
        for info in traces_list:
            sess_id = info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
            if not sess_id:
                continue
            session_traces.setdefault(sess_id, []).append(info)

        matching_first_trace_ids: set[str] | None = None
        if filter_string:
            filtered_list, _ = self.search_traces(
                experiment_ids=[str(exp_id)],
                filter_string=filter_string,
                max_results=10000,
            )
            matching_first_trace_ids = {t.trace_id for t in filtered_list}

        completed: list[CompletedSession] = []
        for sess_id, traces in session_traces.items():
            traces.sort(key=lambda t: (int(t.request_time), str(t.trace_id)))
            first_ts = int(traces[0].request_time)
            last_ts = int(traces[-1].request_time)
            if last_ts < min_last_trace_timestamp_ms or last_ts > max_last_trace_timestamp_ms:
                continue
            if (
                matching_first_trace_ids is not None
                and traces[0].trace_id not in matching_first_trace_ids
            ):
                continue
            completed.append(
                CompletedSession(
                    session_id=sess_id,
                    first_trace_timestamp_ms=first_ts,
                    last_trace_timestamp_ms=last_ts,
                )
            )

        completed.sort(key=lambda s: (int(s.last_trace_timestamp_ms), str(s.session_id)))
        if max_results is not None:
            completed = completed[: int(max_results)]
        return completed

    # ------------------------------------------------------------------
    # Evaluation Datasets (MLflow 3.x)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dataset_digest(name: str, last_update_time: int) -> str:
        digest_input = f"{name}:{last_update_time}".encode()
        return hashlib.sha256(digest_input).hexdigest()[:8]

    @staticmethod
    def _infer_field_type(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"

    def _update_dataset_schema(
        self, existing_schema_json: str | None, record_dicts: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not record_dicts and not existing_schema_json:
            return None
        schema: dict[str, Any] = (
            json.loads(existing_schema_json)
            if existing_schema_json
            else {"inputs": {}, "outputs": {}, "expectations": {}, "version": "1.0"}
        )
        for record in record_dicts:
            if inputs := record.get("inputs"):
                for key, value in inputs.items():
                    if key not in schema["inputs"]:
                        schema["inputs"][key] = self._infer_field_type(value)
            if (outputs := record.get("outputs")) is not None:
                if isinstance(outputs, dict):
                    if not isinstance(schema.get("outputs"), dict):
                        schema["outputs"] = {}
                    for key, value in outputs.items():
                        if key not in schema["outputs"]:
                            schema["outputs"][key] = self._infer_field_type(value)
                elif not schema.get("outputs"):
                    schema["outputs"] = self._infer_field_type(outputs)
            if expectations := record.get("expectations"):
                for key, value in expectations.items():
                    if key not in schema["expectations"]:
                        schema["expectations"][key] = self._infer_field_type(value)
        return schema

    def _fetch_evaluation_dataset_entity(self, dataset_id: str) -> EvaluationDatasetEntity:
        client = self._gw_client()
        res = client.get_evaluation_dataset(
            GetEvaluationDatasetParams(scope=_SCOPE, dataset_id=dataset_id)
        )
        if isinstance(res.value, GetEvaluationDatasetMissing) or not isinstance(
            res.value, GetEvaluationDatasetFound
        ):
            raise MlflowException(
                f"Evaluation dataset with id '{dataset_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.dataset
        return EvaluationDatasetEntity(
            scope=_SCOPE,
            dataset_id=row.dataset_id,
            name=row.name,
            digest=row.digest,
            created_time_ms=int(row.created_time_ms),
            last_update_time_ms=int(row.last_update_time_ms),
            schema_json=row.schema_json,
            profile_json=row.profile_json,
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
            packed_tags=row.packed_tags,
            packed_experiment_ids=row.packed_experiment_ids,
            packed_records=row.packed_records,
        )

    def _list_evaluation_dataset_entities(self) -> list[EvaluationDatasetEntity]:
        client = self._gw_client()
        entities: list[EvaluationDatasetEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_evaluation_datasets(
                ListEvaluationDatasetsParams(scope=_SCOPE, limit=5, after=cursor)
            )
            if not isinstance(res.value, ListEvaluationDatasetsFound):
                break
            for row in res.value.datasets:
                entities.append(
                    EvaluationDatasetEntity(
                        scope=_SCOPE,
                        dataset_id=row.dataset_id,
                        name=row.name,
                        digest=row.digest,
                        created_time_ms=int(row.created_time_ms),
                        last_update_time_ms=int(row.last_update_time_ms),
                        schema_json=row.schema_json,
                        profile_json=row.profile_json,
                        created_by=row.created_by,
                        last_updated_by=row.last_updated_by,
                        packed_tags=row.packed_tags,
                        packed_experiment_ids=row.packed_experiment_ids,
                        packed_records=row.packed_records,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.datasets:
                break
        return entities

    def _save_evaluation_dataset_entity(
        self, entity: EvaluationDatasetEntity
    ) -> EvaluationDatasetEntity:
        client = self._gw_client()
        res = client.upsert_evaluation_dataset_entity(
            UpsertEvaluationDatasetEntityInput(
                request_id=str(uuid.uuid4()),
                dataset_id=entity.dataset_id,
                name=entity.name,
                digest=entity.digest,
                created_time_ms=int(entity.created_time_ms),
                last_update_time_ms=int(entity.last_update_time_ms),
                schema_json=entity.schema_json,
                profile_json=entity.profile_json,
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
                packed_tags=entity.packed_tags,
                packed_experiment_ids=entity.packed_experiment_ids,
                packed_records=entity.packed_records,
            )
        )
        if not isinstance(res.outcome, UpsertEvaluationDatasetEntityEvaluationDatasetUpserted):
            raise MlflowException("Invalid UpsertEvaluationDatasetEntity response from RiffDB")
        return res.outcome.dataset

    @staticmethod
    def _unpack_dataset_records(dataset_id: str, packed_records: bytes) -> list[DatasetRecord]:
        stored_records = _unpack_list(packed_records)
        stored_records.sort(
            key=lambda r: (int(r.get("created_time", 0)), str(r.get("dataset_record_id", "")))
        )
        out: list[DatasetRecord] = []
        for item in stored_records:
            src_dict = item.get("source")
            src = DatasetRecordSource.from_dict(src_dict) if isinstance(src_dict, dict) else None
            out.append(
                DatasetRecord(
                    dataset_record_id=str(item["dataset_record_id"]),
                    dataset_id=dataset_id,
                    inputs=dict(item.get("inputs") or {}),
                    outputs=item.get("outputs"),
                    expectations=item.get("expectations"),
                    tags=item.get("tags"),
                    source=src,
                    source_id=item.get("source_id"),
                    created_time=int(item.get("created_time", 0)),
                    last_update_time=int(item.get("last_update_time", 0)),
                    created_by=item.get("created_by"),
                    last_updated_by=item.get("last_updated_by"),
                )
            )
        return out

    def _evaluation_dataset_to_mlflow(self, entity: EvaluationDatasetEntity) -> EvaluationDataset:
        tags_dict = {str(k): str(v) for k, v in _unpack_dict(entity.packed_tags).items()}
        exp_ids_raw = json.loads(entity.packed_experiment_ids.decode("utf-8"))
        exp_ids = [str(e) for e in exp_ids_raw] if isinstance(exp_ids_raw, list) else []
        ds = EvaluationDataset(
            dataset_id=entity.dataset_id,
            name=entity.name,
            tags=tags_dict,
            schema=entity.schema_json,
            profile=entity.profile_json,
            digest=entity.digest,
            created_time=int(entity.created_time_ms),
            last_update_time=int(entity.last_update_time_ms),
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
        )
        ds.experiment_ids = exp_ids
        ds._records = self._unpack_dataset_records(entity.dataset_id, entity.packed_records)
        ds._tracking_store = self
        return ds

    def create_dataset(
        self,
        name: str,
        tags: dict[str, str] | None = None,
        experiment_ids: list[str] | None = None,
    ) -> EvaluationDataset:
        now = _now_ms()
        dataset_id = f"d-{uuid.uuid4().hex}"
        digest = self._compute_dataset_digest(name, now)
        tags_map = {str(k): str(v) for k, v in (tags or {}).items() if v is not None}
        user_id = tags_map.get(MLFLOW_USER)
        exp_ids = [str(e) for e in (experiment_ids or [])]
        saved = self._save_evaluation_dataset_entity(
            EvaluationDatasetEntity(
                scope=_SCOPE,
                dataset_id=dataset_id,
                name=name,
                digest=digest,
                created_time_ms=now,
                last_update_time_ms=now,
                schema_json=None,
                profile_json=None,
                created_by=user_id,
                last_updated_by=user_id,
                packed_tags=_pack_json(tags_map),
                packed_experiment_ids=_pack_json(exp_ids),
                packed_records=b"[]",
            )
        )
        return self._evaluation_dataset_to_mlflow(saved)

    def get_dataset(self, dataset_id: str) -> EvaluationDataset:
        return self._evaluation_dataset_to_mlflow(self._fetch_evaluation_dataset_entity(dataset_id))

    def delete_dataset(self, dataset_id: str) -> None:
        client = self._gw_client()
        res = client.delete_evaluation_dataset_entity(
            DeleteEvaluationDatasetEntityInput(request_id=str(uuid.uuid4()), dataset_id=dataset_id)
        )
        if not isinstance(
            res.outcome,
            DeleteEvaluationDatasetEntityEvaluationDatasetDeleted
            | DeleteEvaluationDatasetEntityEvaluationDatasetMissing,
        ):
            raise MlflowException("Invalid DeleteEvaluationDatasetEntity response from RiffDB")

    def search_datasets(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 1000,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[EvaluationDataset]:
        self._validate_gateway_max_results(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)  # type: ignore[no-untyped-call]
        parsed_filters = (
            SearchEvaluationDatasetsUtils.parse_search_filter(filter_string)  # type: ignore[no-untyped-call]
            if filter_string
            else []
        )
        order_specs: list[tuple[str, bool]] = []
        if order_by:
            for clause in order_by:
                _, key, is_asc = (
                    SearchEvaluationDatasetsUtils.parse_order_by_for_search_evaluation_datasets(  # type: ignore[no-untyped-call]
                        clause
                    )
                )
                order_specs.append((str(key), bool(is_asc)))

        target_exps = {str(e) for e in experiment_ids} if experiment_ids else None
        matched: list[EvaluationDataset] = []
        for entity in self._list_evaluation_dataset_entities():
            ds = self._evaluation_dataset_to_mlflow(entity)
            if target_exps is not None and not (set(ds.experiment_ids) & target_exps):
                continue
            if not self._matches_evaluation_dataset_filters(ds, parsed_filters):
                continue
            matched.append(ds)

        if order_specs:
            for key, is_asc in reversed(order_specs):
                if key == "name":
                    matched.sort(key=lambda d: d.name or "", reverse=not is_asc)
                elif key == "created_time":
                    matched.sort(key=lambda d: int(d.created_time or 0), reverse=not is_asc)
                elif key == "last_update_time":
                    matched.sort(key=lambda d: int(d.last_update_time or 0), reverse=not is_asc)
        else:
            matched.sort(key=lambda d: (-int(d.created_time or 0), d.dataset_id))

        sliced = matched[offset : offset + max_results + 1]
        next_token = None
        if len(sliced) > max_results:
            next_token = SearchUtils.create_page_token(offset + max_results)  # type: ignore[no-untyped-call]
        return PagedList(sliced[:max_results], next_token)

    @staticmethod
    def _matches_evaluation_dataset_filters(
        ds: EvaluationDataset, parsed_filters: list[dict[str, Any]]
    ) -> bool:
        for flt in parsed_filters:
            ftype = flt.get("type")
            key = str(flt.get("key"))
            comp = str(flt.get("comparator")).upper()
            expected = flt.get("value")
            if ftype == "attribute":
                actual: Any = getattr(ds, key, None)
            else:
                actual = (ds.tags or {}).get(key)
            if actual is None:
                if comp != "!=":
                    return False
                continue
            if key in ("created_time", "last_update_time"):
                if expected is None:
                    return False
                lhs_num = float(actual)
                rhs_num = float(expected)
                if comp == "=" and lhs_num != rhs_num:
                    return False
                if comp == "!=" and lhs_num == rhs_num:
                    return False
                if comp == ">" and not (lhs_num > rhs_num):
                    return False
                if comp == ">=" and not (lhs_num >= rhs_num):
                    return False
                if comp == "<" and not (lhs_num < rhs_num):
                    return False
                if comp == "<=" and not (lhs_num <= rhs_num):
                    return False
            else:
                lhs_str = str(actual)
                rhs_str = str(expected)
                if comp == "=" and lhs_str != rhs_str:
                    return False
                if comp == "!=" and lhs_str == rhs_str:
                    return False
                if comp in ("LIKE", "ILIKE"):
                    pat = "^" + re.escape(rhs_str).replace(r"%", ".*").replace(r"_", ".") + "$"
                    flags = re.IGNORECASE if comp == "ILIKE" else 0
                    if re.match(pat, lhs_str, flags=flags) is None:
                        return False
        return True

    def set_dataset_tags(self, dataset_id: str, tags: dict[str, Any]) -> None:
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        tags_map = {str(k): str(v) for k, v in _unpack_dict(existing.packed_tags).items()}
        for k, v in (tags or {}).items():
            if v is not None:
                tags_map[str(k)] = str(v)
        now = _now_ms()
        updated_by = tags_map.get(MLFLOW_USER, existing.last_updated_by)
        self._save_evaluation_dataset_entity(
            EvaluationDatasetEntity(
                scope=_SCOPE,
                dataset_id=existing.dataset_id,
                name=existing.name,
                digest=existing.digest,
                created_time_ms=int(existing.created_time_ms),
                last_update_time_ms=now,
                schema_json=existing.schema_json,
                profile_json=existing.profile_json,
                created_by=existing.created_by,
                last_updated_by=updated_by,
                packed_tags=_pack_json(tags_map),
                packed_experiment_ids=existing.packed_experiment_ids,
                packed_records=existing.packed_records,
            )
        )

    def delete_dataset_tag(self, dataset_id: str, key: str) -> None:
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        tags_map = {str(k): str(v) for k, v in _unpack_dict(existing.packed_tags).items()}
        tags_map.pop(key, None)
        self._save_evaluation_dataset_entity(
            EvaluationDatasetEntity(
                scope=_SCOPE,
                dataset_id=existing.dataset_id,
                name=existing.name,
                digest=existing.digest,
                created_time_ms=int(existing.created_time_ms),
                last_update_time_ms=_now_ms(),
                schema_json=existing.schema_json,
                profile_json=existing.profile_json,
                created_by=existing.created_by,
                last_updated_by=existing.last_updated_by,
                packed_tags=_pack_json(tags_map),
                packed_experiment_ids=existing.packed_experiment_ids,
                packed_records=existing.packed_records,
            )
        )

    def upsert_dataset_records(
        self, dataset_id: str, records: list[dict[str, Any]]
    ) -> dict[str, int]:
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        stored_records = _unpack_list(existing.packed_records)
        by_hash: dict[str, dict[str, Any]] = {
            str(r["input_hash"]): r for r in stored_records if "input_hash" in r
        }
        inserted_count = 0
        updated_count = 0
        now = _now_ms()
        updated_by: str | None = existing.last_updated_by

        for record_dict in records or []:
            inputs_obj = record_dict.get("inputs", {})
            inputs_json = json.dumps(inputs_obj, sort_keys=True)
            input_hash = hashlib.sha256(inputs_json.encode("utf-8")).hexdigest()
            tags = record_dict.get("tags")
            actor = tags.get(MLFLOW_USER) if isinstance(tags, dict) else None
            if actor:
                updated_by = str(actor)

            source_raw = record_dict.get("source")
            if isinstance(source_raw, DatasetRecordSource):
                source_dict: dict[str, Any] | None = source_raw.to_dict()
            elif isinstance(source_raw, dict):
                source_dict = DatasetRecordSource.from_dict(source_raw).to_dict()
            else:
                source_dict = None

            if input_hash in by_hash:
                cur = by_hash[input_hash]
                if "outputs" in record_dict and record_dict["outputs"] is not None:
                    cur["outputs"] = record_dict["outputs"]
                if "expectations" in record_dict and record_dict["expectations"] is not None:
                    merged_exp = dict(cur.get("expectations") or {})
                    merged_exp.update(record_dict["expectations"])
                    cur["expectations"] = merged_exp
                if isinstance(tags, dict):
                    merged_tags = dict(cur.get("tags") or {})
                    merged_tags.update(tags)
                    cur["tags"] = merged_tags
                if source_dict is not None:
                    cur["source"] = source_dict
                cur["last_update_time"] = now
                if actor:
                    cur["last_updated_by"] = str(actor)
                updated_count += 1
            else:
                rec_id = record_dict.get("dataset_record_id") or f"dr-{uuid.uuid4().hex}"
                entry = {
                    "dataset_record_id": str(rec_id),
                    "dataset_id": dataset_id,
                    "input_hash": input_hash,
                    "inputs": inputs_obj,
                    "outputs": record_dict.get("outputs"),
                    "expectations": record_dict.get("expectations"),
                    "tags": dict(tags) if isinstance(tags, dict) else None,
                    "source": source_dict,
                    "source_id": record_dict.get("source_id"),
                    "created_time": now,
                    "last_update_time": now,
                    "created_by": str(actor) if actor else None,
                    "last_updated_by": str(actor) if actor else None,
                }
                by_hash[input_hash] = entry
                inserted_count += 1

        final_records = list(by_hash.values())
        final_records.sort(
            key=lambda r: (int(r.get("created_time", 0)), str(r.get("dataset_record_id", "")))
        )
        updated_schema = self._update_dataset_schema(existing.schema_json, records or [])
        profile_json = json.dumps({"num_records": len(final_records)}) if final_records else None
        new_digest = self._compute_dataset_digest(existing.name, now)

        self._save_evaluation_dataset_entity(
            EvaluationDatasetEntity(
                scope=_SCOPE,
                dataset_id=existing.dataset_id,
                name=existing.name,
                digest=new_digest,
                created_time_ms=int(existing.created_time_ms),
                last_update_time_ms=now,
                schema_json=(
                    json.dumps(updated_schema)
                    if updated_schema is not None
                    else existing.schema_json
                ),
                profile_json=profile_json,
                created_by=existing.created_by,
                last_updated_by=updated_by,
                packed_tags=existing.packed_tags,
                packed_experiment_ids=existing.packed_experiment_ids,
                packed_records=_pack_json(final_records),
            )
        )
        return {"inserted": inserted_count, "updated": updated_count}

    def _load_dataset_records(
        self,
        dataset_id: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> tuple[list[DatasetRecord], str | None]:
        from mlflow.store.tracking import LOAD_DATASET_RECORDS_MAX_RESULTS

        effective_max = (
            None if max_results is None else (int(max_results) or LOAD_DATASET_RECORDS_MAX_RESULTS)
        )
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        stored_records = _unpack_list(existing.packed_records)
        stored_records.sort(
            key=lambda r: (int(r.get("created_time", 0)), str(r.get("dataset_record_id", "")))
        )
        if page_token:
            try:
                decoded = base64.b64decode(page_token.encode("utf-8")).decode("utf-8")
                last_created_str, last_rec_id = decoded.split(":", 1)
                last_created = int(last_created_str)
                stored_records = [
                    r
                    for r in stored_records
                    if (int(r.get("created_time", 0)), str(r.get("dataset_record_id", "")))
                    > (last_created, last_rec_id)
                ]
            except (ValueError, AttributeError):
                offset = int(page_token)
                stored_records = stored_records[offset:]

        next_page_token: str | None = None
        if effective_max is not None:
            if len(stored_records) > effective_max:
                stored_records = stored_records[:effective_max]
                last = stored_records[-1]
                cursor = f"{last.get('created_time', 0)}:{last.get('dataset_record_id', '')}"
                next_page_token = base64.b64encode(cursor.encode("utf-8")).decode("utf-8")

        out: list[DatasetRecord] = []
        for item in stored_records:
            src_dict = item.get("source")
            src = DatasetRecordSource.from_dict(src_dict) if isinstance(src_dict, dict) else None
            out.append(
                DatasetRecord(
                    dataset_record_id=str(item["dataset_record_id"]),
                    dataset_id=dataset_id,
                    inputs=dict(item.get("inputs") or {}),
                    outputs=item.get("outputs"),
                    expectations=item.get("expectations"),
                    tags=item.get("tags"),
                    source=src,
                    source_id=item.get("source_id"),
                    created_time=int(item.get("created_time", 0)),
                    last_update_time=int(item.get("last_update_time", 0)),
                    created_by=item.get("created_by"),
                    last_updated_by=item.get("last_updated_by"),
                )
            )
        return out, next_page_token

    def delete_dataset_records(self, dataset_id: str, dataset_record_ids: list[str]) -> int:
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        stored_records = _unpack_list(existing.packed_records)
        target_ids = set(dataset_record_ids or [])
        remaining = [r for r in stored_records if str(r.get("dataset_record_id")) not in target_ids]
        deleted_count = len(stored_records) - len(remaining)
        if deleted_count > 0:
            now = _now_ms()
            profile_json = json.dumps({"num_records": len(remaining)}) if remaining else None
            self._save_evaluation_dataset_entity(
                EvaluationDatasetEntity(
                    scope=_SCOPE,
                    dataset_id=existing.dataset_id,
                    name=existing.name,
                    digest=self._compute_dataset_digest(existing.name, now),
                    created_time_ms=int(existing.created_time_ms),
                    last_update_time_ms=now,
                    schema_json=existing.schema_json,
                    profile_json=profile_json,
                    created_by=existing.created_by,
                    last_updated_by=existing.last_updated_by,
                    packed_tags=existing.packed_tags,
                    packed_experiment_ids=existing.packed_experiment_ids,
                    packed_records=_pack_json(remaining),
                )
            )
        return deleted_count

    def get_dataset_experiment_ids(self, dataset_id: str) -> list[str]:
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        raw = json.loads(existing.packed_experiment_ids.decode("utf-8"))
        return [str(e) for e in raw] if isinstance(raw, list) else []

    def add_dataset_to_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> EvaluationDataset:
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        for exp_id in experiment_ids or []:
            self.get_experiment(str(exp_id))  # type: ignore[no-untyped-call]
        cur_ids = self.get_dataset_experiment_ids(dataset_id)
        for exp_id in experiment_ids or []:
            s_id = str(exp_id)
            if s_id not in cur_ids:
                cur_ids.append(s_id)
        saved = self._save_evaluation_dataset_entity(
            EvaluationDatasetEntity(
                scope=_SCOPE,
                dataset_id=existing.dataset_id,
                name=existing.name,
                digest=existing.digest,
                created_time_ms=int(existing.created_time_ms),
                last_update_time_ms=_now_ms(),
                schema_json=existing.schema_json,
                profile_json=existing.profile_json,
                created_by=existing.created_by,
                last_updated_by=existing.last_updated_by,
                packed_tags=existing.packed_tags,
                packed_experiment_ids=_pack_json(cur_ids),
                packed_records=existing.packed_records,
            )
        )
        return self._evaluation_dataset_to_mlflow(saved)

    def remove_dataset_from_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> EvaluationDataset:
        existing = self._fetch_evaluation_dataset_entity(dataset_id)
        remove_set = {str(e) for e in (experiment_ids or [])}
        cur_ids = [e for e in self.get_dataset_experiment_ids(dataset_id) if e not in remove_set]
        saved = self._save_evaluation_dataset_entity(
            EvaluationDatasetEntity(
                scope=_SCOPE,
                dataset_id=existing.dataset_id,
                name=existing.name,
                digest=existing.digest,
                created_time_ms=int(existing.created_time_ms),
                last_update_time_ms=_now_ms(),
                schema_json=existing.schema_json,
                profile_json=existing.profile_json,
                created_by=existing.created_by,
                last_updated_by=existing.last_updated_by,
                packed_tags=existing.packed_tags,
                packed_experiment_ids=_pack_json(cur_ids),
                packed_records=existing.packed_records,
            )
        )
        return self._evaluation_dataset_to_mlflow(saved)

    # ------------------------------------------------------------------
    # Label Schemas (MLflow 3.15+)
    # ------------------------------------------------------------------

    def _validate_exp_exists(self, experiment_id: str) -> int:
        exp_id = int(self._numeric_experiment_id(str(experiment_id)))  # type: ignore[attr-defined]
        exp = self.get_experiment(str(exp_id))  # type: ignore[no-untyped-call]
        if exp is None:
            raise MlflowException(
                f"Experiment '{experiment_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return exp_id

    def _fetch_label_schema_entity(self, schema_id: str) -> LabelSchemaEntity:
        client = self._gw_client()
        res = client.get_label_schema(GetLabelSchemaParams(scope=_SCOPE, schema_id=schema_id))
        if isinstance(res.value, GetLabelSchemaMissing) or not isinstance(
            res.value, GetLabelSchemaFound
        ):
            raise MlflowException(
                f"Label schema with id '{schema_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.label_schema
        return LabelSchemaEntity(
            scope=_SCOPE,
            schema_id=row.schema_id,
            experiment_id=int(row.experiment_id),
            name=row.name,
            schema_type=row.schema_type,
            instruction=row.instruction,
            enable_comment=bool(row.enable_comment),
            input_type=row.input_type,
            input_config_json=row.input_config_json,
            created_by=row.created_by,
            created_time_ms=int(row.created_time_ms),
            last_update_time_ms=int(row.last_update_time_ms),
            is_default=bool(row.is_default),
        )

    def _list_label_schema_entities(self, experiment_id: int) -> list[LabelSchemaEntity]:
        client = self._gw_client()
        entities: list[LabelSchemaEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_label_schemas(
                ListLabelSchemasParams(scope=_SCOPE, limit=25, after=cursor)
            )
            if not isinstance(res.value, ListLabelSchemasFound):
                break
            for row in res.value.label_schemas:
                if int(row.experiment_id) != experiment_id:
                    continue
                entities.append(
                    LabelSchemaEntity(
                        scope=_SCOPE,
                        schema_id=row.schema_id,
                        experiment_id=int(row.experiment_id),
                        name=row.name,
                        schema_type=row.schema_type,
                        instruction=row.instruction,
                        enable_comment=bool(row.enable_comment),
                        input_type=row.input_type,
                        input_config_json=row.input_config_json,
                        created_by=row.created_by,
                        created_time_ms=int(row.created_time_ms),
                        last_update_time_ms=int(row.last_update_time_ms),
                        is_default=bool(row.is_default),
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.label_schemas:
                break
        return entities

    def _save_label_schema_entity(self, entity: LabelSchemaEntity) -> LabelSchemaEntity:
        client = self._gw_client()
        res = client.upsert_label_schema_entity(
            UpsertLabelSchemaEntityInput(
                request_id=str(uuid.uuid4()),
                schema_id=entity.schema_id,
                experiment_id=int(entity.experiment_id),
                name=entity.name,
                schema_type=entity.schema_type,
                instruction=entity.instruction,
                enable_comment=bool(entity.enable_comment),
                input_type=entity.input_type,
                input_config_json=entity.input_config_json,
                created_by=entity.created_by,
                created_time_ms=int(entity.created_time_ms),
                last_update_time_ms=int(entity.last_update_time_ms),
                is_default=bool(entity.is_default),
            )
        )
        if not isinstance(res.outcome, UpsertLabelSchemaEntityLabelSchemaUpserted):
            raise MlflowException("Invalid UpsertLabelSchemaEntity response from RiffDB")
        return res.outcome.label_schema

    @staticmethod
    def _label_schema_to_mlflow(entity: LabelSchemaEntity) -> Any:
        from mlflow.genai.label_schemas.label_schemas import LabelSchema, LabelSchemaType
        from mlflow.store.tracking.dbmodels.models import _input_from_dict

        return LabelSchema(
            name=entity.name,
            type=LabelSchemaType(entity.schema_type),
            input=_input_from_dict(entity.input_type, json.loads(entity.input_config_json)),
            instruction=entity.instruction,
            enable_comment=bool(entity.enable_comment),
            schema_id=entity.schema_id,
            experiment_id=str(entity.experiment_id),
            created_by=entity.created_by,
            created_at=int(entity.created_time_ms),
            updated_at=int(entity.last_update_time_ms),
            is_default=bool(entity.is_default),
        )

    def _ensure_default_label_schema(self, exp_id: int) -> None:
        from mlflow.genai.label_schemas.label_schemas import InputText, LabelSchemaType
        from mlflow.genai.label_schemas.validation import (
            DEFAULT_LABEL_SCHEMA_INSTRUCTION,
            DEFAULT_LABEL_SCHEMA_NAME,
        )
        from mlflow.store.tracking.dbmodels.models import _input_to_dict

        existing = self._list_label_schema_entities(exp_id)
        if any(e.is_default or e.name == DEFAULT_LABEL_SCHEMA_NAME for e in existing):
            return
        input_type, input_config_json = _input_to_dict(InputText())
        now = _now_ms()
        self._save_label_schema_entity(
            LabelSchemaEntity(
                scope=_SCOPE,
                schema_id=f"ls-{uuid.uuid4().hex}",
                experiment_id=exp_id,
                name=DEFAULT_LABEL_SCHEMA_NAME,
                schema_type=str(LabelSchemaType.FEEDBACK),
                instruction=DEFAULT_LABEL_SCHEMA_INSTRUCTION,
                enable_comment=False,
                input_type=input_type,
                input_config_json=input_config_json,
                created_by=None,
                created_time_ms=now,
                last_update_time_ms=now,
                is_default=True,
            )
        )

    def create_label_schema(
        self,
        experiment_id: str,
        *,
        name: str,
        type: Any,
        input: Any,
        instruction: str | None = None,
        enable_comment: bool = False,
    ) -> Any:
        from mlflow.genai.label_schemas.label_schemas import LabelSchemaType
        from mlflow.genai.label_schemas.validation import validate_schema_for_create
        from mlflow.store.tracking.dbmodels.models import _input_to_dict

        validate_schema_for_create(
            name=name,
            type=type,
            input=input,
            instruction=instruction,
            enable_comment=enable_comment,
        )
        exp_id = self._validate_exp_exists(experiment_id)
        for existing in self._list_label_schema_entities(exp_id):
            if existing.name == name:
                raise MlflowException(
                    f"Label schema with name '{name}' already exists for experiment "
                    f"'{experiment_id}'.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
        input_type, input_config_json = _input_to_dict(input)
        now = _now_ms()
        saved = self._save_label_schema_entity(
            LabelSchemaEntity(
                scope=_SCOPE,
                schema_id=f"ls-{uuid.uuid4().hex}",
                experiment_id=exp_id,
                name=name,
                schema_type=str(LabelSchemaType(str(type))),
                instruction=instruction,
                enable_comment=bool(enable_comment),
                input_type=input_type,
                input_config_json=input_config_json,
                created_by=None,
                created_time_ms=now,
                last_update_time_ms=now,
                is_default=False,
            )
        )
        return self._label_schema_to_mlflow(saved)

    def get_label_schema(self, schema_id: str) -> Any:
        return self._label_schema_to_mlflow(self._fetch_label_schema_entity(schema_id))

    def get_label_schema_by_name(self, experiment_id: str, name: str) -> Any:
        exp_id = self._validate_exp_exists(experiment_id)
        for entity in self._list_label_schema_entities(exp_id):
            if entity.name == name:
                return self._label_schema_to_mlflow(entity)
        raise MlflowException(
            f"Label schema with name '{name}' not found for experiment '{experiment_id}'.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def list_label_schemas(
        self,
        experiment_id: str,
        max_results: int = 100,
        page_token: str | None = None,
    ) -> PagedList[Any]:
        self._validate_gateway_max_results(max_results)
        offset = (
            SearchUtils.parse_start_offset_from_page_token(page_token)  # type: ignore[no-untyped-call]
            if page_token
            else 0
        )
        exp_id = self._validate_exp_exists(experiment_id)
        self._ensure_default_label_schema(exp_id)
        entities = self._list_label_schema_entities(exp_id)
        entities.sort(key=lambda e: (-int(e.created_time_ms), e.schema_id))
        sliced = entities[offset : offset + max_results + 1]
        next_token = None
        if len(sliced) > max_results:
            next_token = SearchUtils.create_page_token(offset + max_results)  # type: ignore[no-untyped-call]
        return PagedList(
            [self._label_schema_to_mlflow(e) for e in sliced[:max_results]], next_token
        )

    def update_label_schema(
        self,
        schema_id: str,
        *,
        name: str | None = None,
        instruction: str | None = None,
        enable_comment: bool | None = None,
        input: Any = None,
    ) -> Any:
        from mlflow.genai.label_schemas.validation import validate_schema_for_update
        from mlflow.store.tracking.dbmodels.models import _input_to_dict

        existing = self._fetch_label_schema_entity(schema_id)
        existing_entity = self._label_schema_to_mlflow(existing)
        validate_schema_for_update(
            existing=existing_entity,
            name=name,
            instruction=instruction,
            enable_comment=enable_comment,
            input=input,
        )
        resolved_name = existing.name
        if name is not None and name != existing.name:
            for other in self._list_label_schema_entities(int(existing.experiment_id)):
                if other.schema_id != schema_id and other.name == name:
                    raise MlflowException(
                        f"Label schema with name '{name}' already exists for experiment "
                        f"'{existing.experiment_id}'.",
                        error_code=RESOURCE_ALREADY_EXISTS,
                    )
            resolved_name = name
        resolved_instruction = instruction if instruction is not None else existing.instruction
        resolved_comment = (
            bool(enable_comment) if enable_comment is not None else bool(existing.enable_comment)
        )
        if input is not None:
            resolved_input_type, resolved_input_cfg = _input_to_dict(input)
        else:
            resolved_input_type = existing.input_type
            resolved_input_cfg = existing.input_config_json

        saved = self._save_label_schema_entity(
            LabelSchemaEntity(
                scope=_SCOPE,
                schema_id=existing.schema_id,
                experiment_id=int(existing.experiment_id),
                name=resolved_name,
                schema_type=existing.schema_type,
                instruction=resolved_instruction,
                enable_comment=resolved_comment,
                input_type=resolved_input_type,
                input_config_json=resolved_input_cfg,
                created_by=existing.created_by,
                created_time_ms=int(existing.created_time_ms),
                last_update_time_ms=_now_ms(),
                is_default=bool(existing.is_default),
            )
        )
        return self._label_schema_to_mlflow(saved)

    def delete_label_schema(self, schema_id: str) -> None:
        try:
            existing = self._fetch_label_schema_entity(schema_id)
        except MlflowException:
            return
        if existing.is_default:
            raise MlflowException(
                "The experiment's default question cannot be deleted.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        client = self._gw_client()
        res = client.delete_label_schema_entity(
            DeleteLabelSchemaEntityInput(request_id=str(uuid.uuid4()), schema_id=schema_id)
        )
        if not isinstance(
            res.outcome,
            DeleteLabelSchemaEntityLabelSchemaDeleted | DeleteLabelSchemaEntityLabelSchemaMissing,
        ):
            raise MlflowException("Invalid DeleteLabelSchemaEntity response from RiffDB")

    # ------------------------------------------------------------------
    # Human Review Queues (MLflow 3.15+)
    # ------------------------------------------------------------------

    def _fetch_review_queue_entity(self, queue_id: str) -> ReviewQueueEntity:
        client = self._gw_client()
        res = client.get_review_queue(GetReviewQueueParams(scope=_SCOPE, queue_id=queue_id))
        if isinstance(res.value, GetReviewQueueMissing) or not isinstance(
            res.value, GetReviewQueueFound
        ):
            raise MlflowException(
                f"Review queue with id '{queue_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.review_queue
        return ReviewQueueEntity(
            scope=_SCOPE,
            queue_id=row.queue_id,
            experiment_id=int(row.experiment_id),
            name=row.name,
            name_key=row.name_key,
            queue_type=row.queue_type,
            created_by=row.created_by,
            creation_time_ms=int(row.creation_time_ms),
            last_update_time_ms=int(row.last_update_time_ms),
            packed_users=row.packed_users,
            packed_schema_ids=row.packed_schema_ids,
            packed_items=row.packed_items,
        )

    def _list_review_queue_entities(self, experiment_id: int) -> list[ReviewQueueEntity]:
        client = self._gw_client()
        entities: list[ReviewQueueEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_review_queues(
                ListReviewQueuesParams(scope=_SCOPE, limit=10, after=cursor)
            )
            if not isinstance(res.value, ListReviewQueuesFound):
                break
            for row in res.value.review_queues:
                if int(row.experiment_id) != experiment_id:
                    continue
                entities.append(
                    ReviewQueueEntity(
                        scope=_SCOPE,
                        queue_id=row.queue_id,
                        experiment_id=int(row.experiment_id),
                        name=row.name,
                        name_key=row.name_key,
                        queue_type=row.queue_type,
                        created_by=row.created_by,
                        creation_time_ms=int(row.creation_time_ms),
                        last_update_time_ms=int(row.last_update_time_ms),
                        packed_users=row.packed_users,
                        packed_schema_ids=row.packed_schema_ids,
                        packed_items=row.packed_items,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.review_queues:
                break
        return entities

    def _save_review_queue_entity(self, entity: ReviewQueueEntity) -> ReviewQueueEntity:
        client = self._gw_client()
        res = client.upsert_review_queue_entity(
            UpsertReviewQueueEntityInput(
                request_id=str(uuid.uuid4()),
                queue_id=entity.queue_id,
                experiment_id=int(entity.experiment_id),
                name=entity.name,
                name_key=entity.name_key,
                queue_type=entity.queue_type,
                created_by=entity.created_by,
                creation_time_ms=int(entity.creation_time_ms),
                last_update_time_ms=int(entity.last_update_time_ms),
                packed_users=entity.packed_users,
                packed_schema_ids=entity.packed_schema_ids,
                packed_items=entity.packed_items,
            )
        )
        if not isinstance(res.outcome, UpsertReviewQueueEntityReviewQueueUpserted):
            raise MlflowException("Invalid UpsertReviewQueueEntity response from RiffDB")
        return res.outcome.review_queue

    @staticmethod
    def _review_queue_to_mlflow(entity: ReviewQueueEntity) -> Any:
        from mlflow.genai.review_queues import ReviewQueue, ReviewQueueType

        users_raw = json.loads(entity.packed_users.decode("utf-8"))
        schemas_raw = json.loads(entity.packed_schema_ids.decode("utf-8"))
        users = [str(u) for u in users_raw] if isinstance(users_raw, list) else []
        schema_ids = [str(s) for s in schemas_raw] if isinstance(schemas_raw, list) else []
        return ReviewQueue(
            queue_id=entity.queue_id,
            experiment_id=str(entity.experiment_id),
            name=entity.name,
            queue_type=ReviewQueueType(entity.queue_type),
            created_by=entity.created_by,
            creation_time_ms=int(entity.creation_time_ms),
            last_update_time_ms=int(entity.last_update_time_ms),
            users=users,
            schema_ids=schema_ids,
        )

    @staticmethod
    def _review_queue_item_to_mlflow(queue_id: str, item_dict: dict[str, Any]) -> Any:
        from mlflow.genai.review_queues import (
            ReviewItemType,
            ReviewQueueItem,
            ReviewStatus,
        )

        return ReviewQueueItem(
            queue_id=queue_id,
            item_type=ReviewItemType(str(item_dict.get("item_type", "trace"))),
            item_id=str(item_dict["item_id"]),
            status=ReviewStatus(str(item_dict.get("status", "pending"))),
            creation_time_ms=int(item_dict.get("creation_time_ms", 0)),
            last_update_time_ms=int(item_dict.get("last_update_time_ms", 0)),
            completed_by=item_dict.get("completed_by"),
            completed_time_ms=item_dict.get("completed_time_ms"),
        )

    def _validate_review_queue_schema_ids(self, exp_id: int, schema_ids: list[str]) -> None:
        if not schema_ids:
            return
        existing_schemas = {e.schema_id: e for e in self._list_label_schema_entities(exp_id)}
        for sid in schema_ids:
            if sid not in existing_schemas:
                raise MlflowException(
                    f"Label schema '{sid}' does not exist in experiment '{exp_id}'.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

    def create_review_queue(
        self,
        experiment_id: str,
        *,
        name: str,
        queue_type: Any,
        created_by: str | None = None,
        users: list[str] | None = None,
        schema_ids: list[str] | None = None,
    ) -> Any:
        from mlflow.genai.review_queues.validation import validate_queue_for_create

        validated = validate_queue_for_create(
            name=name,
            queue_type=queue_type,
            users=users,
            schema_ids=schema_ids,
        )
        exp_id = self._validate_exp_exists(experiment_id)
        self._validate_review_queue_schema_ids(exp_id, list(validated.schema_ids))
        name_key = validated.name.lower()
        for existing in self._list_review_queue_entities(exp_id):
            if existing.name_key == name_key:
                raise MlflowException(
                    f"Review queue with name '{validated.name}' already exists "
                    "(names are case-insensitive).",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
        now = _now_ms()
        saved = self._save_review_queue_entity(
            ReviewQueueEntity(
                scope=_SCOPE,
                queue_id=f"rq-{uuid.uuid4().hex}",
                experiment_id=exp_id,
                name=validated.name,
                name_key=name_key,
                queue_type=str(validated.queue_type),
                created_by=created_by,
                creation_time_ms=now,
                last_update_time_ms=now,
                packed_users=_pack_json(list(validated.users)),
                packed_schema_ids=_pack_json(list(validated.schema_ids)),
                packed_items=b"[]",
            )
        )
        return self._review_queue_to_mlflow(saved)

    def get_review_queue(self, queue_id: str) -> Any:
        return self._review_queue_to_mlflow(self._fetch_review_queue_entity(queue_id))

    def get_review_queue_by_name(self, experiment_id: str, *, name: str) -> Any:
        exp_id = self._validate_exp_exists(experiment_id)
        target_key = name.lower()
        for entity in self._list_review_queue_entities(exp_id):
            if entity.name_key == target_key:
                return self._review_queue_to_mlflow(entity)
        raise MlflowException(
            f"Review queue with name '{name}' not found for experiment '{experiment_id}'.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def get_or_create_user_queue(self, experiment_id: str, *, user: str) -> Any:
        from mlflow.genai.review_queues import ReviewQueueType
        from mlflow.genai.review_queues.validation import normalize_user

        name = normalize_user(user)
        try:
            return self.create_review_queue(
                experiment_id,
                name=name,
                queue_type="user",
                created_by=name,
            )
        except MlflowException as e:
            if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise
            existing = self.get_review_queue_by_name(experiment_id, name=name)
            if ReviewQueueType(existing.queue_type) != ReviewQueueType.USER:
                raise MlflowException(
                    f"A non-user queue named '{name}' already exists; cannot get-or-create "
                    f"a user queue with that name.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            return existing

    def list_review_queues(
        self,
        experiment_id: str,
        *,
        user: str | None = None,
        item_id: str | None = None,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Any]:
        from mlflow.genai.review_queues.validation import normalize_user

        if max_results is None:
            resolved_max = SEARCH_MAX_RESULTS_DEFAULT
        else:
            self._validate_gateway_max_results(max_results)
            resolved_max = int(max_results)
        offset = (
            SearchUtils.parse_start_offset_from_page_token(page_token)  # type: ignore[no-untyped-call]
            if page_token
            else 0
        )
        exp_id = self._validate_exp_exists(experiment_id)
        norm_user = normalize_user(user) if user is not None else None
        matched: list[ReviewQueueEntity] = []
        for entity in self._list_review_queue_entities(exp_id):
            if norm_user is not None:
                users_raw = json.loads(entity.packed_users.decode("utf-8"))
                users_set = {str(u) for u in users_raw} if isinstance(users_raw, list) else set()
                if norm_user not in users_set:
                    continue
            if item_id is not None:
                items = _unpack_list(entity.packed_items)
                if not any(str(it.get("item_id")) == item_id for it in items):
                    continue
            matched.append(entity)

        matched.sort(key=lambda e: (-int(e.creation_time_ms), e.queue_id))
        sliced = matched[offset : offset + resolved_max + 1]
        next_token = None
        if len(sliced) > resolved_max:
            next_token = SearchUtils.create_page_token(offset + resolved_max)  # type: ignore[no-untyped-call]
        return PagedList(
            [self._review_queue_to_mlflow(e) for e in sliced[:resolved_max]], next_token
        )

    def update_review_queue(
        self,
        queue_id: str,
        *,
        users: list[str] | None = None,
        schema_ids: list[str] | None = None,
        name: str | None = None,
        new_owner: str | None = None,
    ) -> Any:
        from mlflow.genai.review_queues import ReviewQueueType
        from mlflow.genai.review_queues.validation import (
            normalize_schema_ids,
            normalize_users,
            validate_custom_queue_name,
            validate_queue_owner,
        )

        existing = self._fetch_review_queue_entity(queue_id)
        if ReviewQueueType(existing.queue_type) == ReviewQueueType.USER:
            raise MlflowException(
                "A user queue's name, assigned user, schemas, and owner are fixed "
                "and cannot be updated.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if users is None and schema_ids is None and name is None and new_owner is None:
            return self._review_queue_to_mlflow(existing)

        resolved_owner = (
            validate_queue_owner(new_owner) if new_owner is not None else existing.created_by
        )
        resolved_name = existing.name
        resolved_name_key = existing.name_key
        if name is not None:
            new_name = validate_custom_queue_name(name)
            new_key = new_name.lower()
            if new_key != existing.name_key:
                for other in self._list_review_queue_entities(int(existing.experiment_id)):
                    if other.queue_id != queue_id and other.name_key == new_key:
                        raise MlflowException(
                            f"Review queue with name '{new_name}' already exists "
                            "(names are case-insensitive).",
                            error_code=RESOURCE_ALREADY_EXISTS,
                        )
            resolved_name = new_name
            resolved_name_key = new_key

        packed_users = (
            _pack_json(list(normalize_users(users))) if users is not None else existing.packed_users
        )
        packed_schema_ids = existing.packed_schema_ids
        if schema_ids is not None:
            cur_items = _unpack_list(existing.packed_items)
            if cur_items:
                raise MlflowException(
                    "A review queue's questions are locked once items are assigned to it. "
                    "Remove the items before changing its questions.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            norm_schemas = list(normalize_schema_ids(schema_ids))
            self._validate_review_queue_schema_ids(int(existing.experiment_id), norm_schemas)
            packed_schema_ids = _pack_json(norm_schemas)

        saved = self._save_review_queue_entity(
            ReviewQueueEntity(
                scope=_SCOPE,
                queue_id=existing.queue_id,
                experiment_id=int(existing.experiment_id),
                name=resolved_name,
                name_key=resolved_name_key,
                queue_type=existing.queue_type,
                created_by=resolved_owner,
                creation_time_ms=int(existing.creation_time_ms),
                last_update_time_ms=_now_ms(),
                packed_users=packed_users,
                packed_schema_ids=packed_schema_ids,
                packed_items=existing.packed_items,
            )
        )
        return self._review_queue_to_mlflow(saved)

    def delete_review_queue(self, queue_id: str) -> None:
        client = self._gw_client()
        res = client.delete_review_queue_entity(
            DeleteReviewQueueEntityInput(request_id=str(uuid.uuid4()), queue_id=queue_id)
        )
        if not isinstance(
            res.outcome,
            DeleteReviewQueueEntityReviewQueueDeleted | DeleteReviewQueueEntityReviewQueueMissing,
        ):
            raise MlflowException("Invalid DeleteReviewQueueEntity response from RiffDB")

    def add_items_to_review_queue(
        self, queue_id: str, *, item_ids: list[str], item_type: str = "trace"
    ) -> list[Any]:
        from mlflow.genai.review_queues import ReviewStatus
        from mlflow.genai.review_queues.validation import (
            coerce_item_type,
            validate_item_ids_for_attach,
        )

        coerced_type = str(coerce_item_type(item_type))
        norm_ids = list(validate_item_ids_for_attach(item_ids))
        existing = self._fetch_review_queue_entity(queue_id)
        items = _unpack_list(existing.packed_items)
        by_id: dict[str, dict[str, Any]] = {str(it["item_id"]): it for it in items}
        now = _now_ms()
        changed = False
        for iid in norm_ids:
            if iid not in by_id:
                by_id[iid] = {
                    "item_id": iid,
                    "item_type": coerced_type,
                    "status": str(ReviewStatus.PENDING),
                    "creation_time_ms": now,
                    "last_update_time_ms": now,
                    "completed_by": None,
                    "completed_time_ms": None,
                }
                changed = True
        if changed:
            self._save_review_queue_entity(
                ReviewQueueEntity(
                    scope=_SCOPE,
                    queue_id=existing.queue_id,
                    experiment_id=int(existing.experiment_id),
                    name=existing.name,
                    name_key=existing.name_key,
                    queue_type=existing.queue_type,
                    created_by=existing.created_by,
                    creation_time_ms=int(existing.creation_time_ms),
                    last_update_time_ms=int(existing.last_update_time_ms),
                    packed_users=existing.packed_users,
                    packed_schema_ids=existing.packed_schema_ids,
                    packed_items=_pack_json(list(by_id.values())),
                )
            )
        return [
            self._review_queue_item_to_mlflow(queue_id, by_id[iid])
            for iid in norm_ids
            if iid in by_id
        ]

    def list_review_queue_items(
        self,
        queue_id: str,
        *,
        status: Any = None,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Any]:
        from mlflow.genai.review_queues.validation import coerce_status

        if max_results is None:
            resolved_max = SEARCH_MAX_RESULTS_DEFAULT
        else:
            self._validate_gateway_max_results(max_results)
            resolved_max = int(max_results)
        offset = (
            SearchUtils.parse_start_offset_from_page_token(page_token)  # type: ignore[no-untyped-call]
            if page_token
            else 0
        )
        coerced_status = str(coerce_status(status)) if status is not None else None
        existing = self._fetch_review_queue_entity(queue_id)
        items = _unpack_list(existing.packed_items)
        if coerced_status is not None:
            items = [it for it in items if str(it.get("status")) == coerced_status]
        items.sort(key=lambda it: (-int(it.get("creation_time_ms", 0)), str(it.get("item_id", ""))))
        sliced = items[offset : offset + resolved_max + 1]
        next_token = None
        if len(sliced) > resolved_max:
            next_token = SearchUtils.create_page_token(offset + resolved_max)  # type: ignore[no-untyped-call]
        return PagedList(
            [self._review_queue_item_to_mlflow(queue_id, it) for it in sliced[:resolved_max]],
            next_token,
        )

    def set_review_queue_item_status(
        self,
        queue_id: str,
        *,
        item_id: str,
        status: Any,
        completed_by: str | None = None,
    ) -> Any:
        from mlflow.genai.review_queues import ReviewStatus
        from mlflow.genai.review_queues.validation import (
            USER_MAX_LENGTH,
            coerce_status,
            normalize_item_id,
            normalize_user,
        )

        new_status = coerce_status(status)
        norm_item_id = normalize_item_id(item_id)
        norm_completed_by = normalize_user(completed_by) if completed_by is not None else None
        if new_status == ReviewStatus.PENDING:
            if norm_completed_by is not None:
                raise MlflowException(
                    "`completed_by` must not be set when reopening an item to `pending`.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        elif not norm_completed_by:
            raise MlflowException(
                f"`completed_by` is required when setting status to `{new_status}`.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if norm_completed_by is not None and len(norm_completed_by) > USER_MAX_LENGTH:
            raise MlflowException(
                f"`completed_by` must be at most {USER_MAX_LENGTH} characters; "
                f"got {len(norm_completed_by)}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        existing = self._fetch_review_queue_entity(queue_id)
        items = _unpack_list(existing.packed_items)
        by_id: dict[str, dict[str, Any]] = {str(it["item_id"]): it for it in items}
        if norm_item_id not in by_id:
            raise MlflowException(
                f"Item '{norm_item_id}' is not attached to review queue '{queue_id}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        target = by_id[norm_item_id]
        if str(target.get("status")) == str(new_status):
            return self._review_queue_item_to_mlflow(queue_id, target)

        now = _now_ms()
        target["status"] = str(new_status)
        target["last_update_time_ms"] = now
        if new_status == ReviewStatus.PENDING:
            target["completed_by"] = None
            target["completed_time_ms"] = None
        else:
            target["completed_by"] = norm_completed_by
            target["completed_time_ms"] = now

        self._save_review_queue_entity(
            ReviewQueueEntity(
                scope=_SCOPE,
                queue_id=existing.queue_id,
                experiment_id=int(existing.experiment_id),
                name=existing.name,
                name_key=existing.name_key,
                queue_type=existing.queue_type,
                created_by=existing.created_by,
                creation_time_ms=int(existing.creation_time_ms),
                last_update_time_ms=int(existing.last_update_time_ms),
                packed_users=existing.packed_users,
                packed_schema_ids=existing.packed_schema_ids,
                packed_items=_pack_json(list(by_id.values())),
            )
        )
        return self._review_queue_item_to_mlflow(queue_id, target)

    def remove_items_from_review_queue(self, queue_id: str, *, item_ids: list[str]) -> None:
        from mlflow.genai.review_queues.validation import validate_item_ids_for_attach

        norm_ids = set(validate_item_ids_for_attach(item_ids))
        existing = self._fetch_review_queue_entity(queue_id)
        items = _unpack_list(existing.packed_items)
        remaining = [it for it in items if str(it.get("item_id")) not in norm_ids]
        if len(remaining) != len(items):
            self._save_review_queue_entity(
                ReviewQueueEntity(
                    scope=_SCOPE,
                    queue_id=existing.queue_id,
                    experiment_id=int(existing.experiment_id),
                    name=existing.name,
                    name_key=existing.name_key,
                    queue_type=existing.queue_type,
                    created_by=existing.created_by,
                    creation_time_ms=int(existing.creation_time_ms),
                    last_update_time_ms=int(existing.last_update_time_ms),
                    packed_users=existing.packed_users,
                    packed_schema_ids=existing.packed_schema_ids,
                    packed_items=_pack_json(remaining),
                )
            )

    def _delete_review_queue_items_for_traces(self, trace_ids: list[str]) -> None:
        target_set = set(trace_ids or [])
        if not target_set:
            return
        for exp in self.search_experiments(ViewType.ALL):  # type: ignore[no-untyped-call]
            for queue in self._list_review_queue_entities(int(exp.experiment_id)):
                items = _unpack_list(queue.packed_items)
                remaining = [it for it in items if str(it.get("item_id")) not in target_set]
                if len(remaining) != len(items):
                    self._save_review_queue_entity(
                        ReviewQueueEntity(
                            scope=_SCOPE,
                            queue_id=queue.queue_id,
                            experiment_id=int(queue.experiment_id),
                            name=queue.name,
                            name_key=queue.name_key,
                            queue_type=queue.queue_type,
                            created_by=queue.created_by,
                            creation_time_ms=int(queue.creation_time_ms),
                            last_update_time_ms=int(queue.last_update_time_ms),
                            packed_users=queue.packed_users,
                            packed_schema_ids=queue.packed_schema_ids,
                            packed_items=_pack_json(remaining),
                        )
                    )

    # ------------------------------------------------------------------
    # Issues (MLflow 3.15+)
    # ------------------------------------------------------------------

    def _fetch_issue_entity(self, issue_id: str) -> IssueEntity:
        client = self._gw_client()
        res = client.get_issue(GetIssueParams(scope=_SCOPE, issue_id=issue_id))
        if isinstance(res.value, GetIssueMissing) or not isinstance(res.value, GetIssueFound):
            raise MlflowException(
                f"Issue with ID '{issue_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.issue
        return IssueEntity(
            scope=_SCOPE,
            issue_id=row.issue_id,
            experiment_id=int(row.experiment_id),
            name=row.name,
            description=row.description,
            status=row.status,
            severity=row.severity,
            root_causes_json=row.root_causes_json,
            source_run_id=row.source_run_id,
            categories_json=row.categories_json,
            created_timestamp_ms=int(row.created_timestamp_ms),
            last_updated_timestamp_ms=int(row.last_updated_timestamp_ms),
            created_by=row.created_by,
        )

    def _list_issue_entities(self) -> list[IssueEntity]:
        client = self._gw_client()
        entities: list[IssueEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_issues(ListIssuesParams(scope=_SCOPE, limit=25, after=cursor))
            if not isinstance(res.value, ListIssuesFound):
                break
            for row in res.value.issues:
                entities.append(
                    IssueEntity(
                        scope=_SCOPE,
                        issue_id=row.issue_id,
                        experiment_id=int(row.experiment_id),
                        name=row.name,
                        description=row.description,
                        status=row.status,
                        severity=row.severity,
                        root_causes_json=row.root_causes_json,
                        source_run_id=row.source_run_id,
                        categories_json=row.categories_json,
                        created_timestamp_ms=int(row.created_timestamp_ms),
                        last_updated_timestamp_ms=int(row.last_updated_timestamp_ms),
                        created_by=row.created_by,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.issues:
                break
        return entities

    def _save_issue_entity(self, entity: IssueEntity) -> IssueEntity:
        client = self._gw_client()
        res = client.upsert_issue_entity(
            UpsertIssueEntityInput(
                request_id=str(uuid.uuid4()),
                issue_id=entity.issue_id,
                experiment_id=int(entity.experiment_id),
                name=entity.name,
                description=entity.description,
                status=entity.status,
                severity=entity.severity,
                root_causes_json=entity.root_causes_json,
                source_run_id=entity.source_run_id,
                categories_json=entity.categories_json,
                created_timestamp_ms=int(entity.created_timestamp_ms),
                last_updated_timestamp_ms=int(entity.last_updated_timestamp_ms),
                created_by=entity.created_by,
            )
        )
        if not isinstance(res.outcome, UpsertIssueEntityIssueUpserted):
            raise MlflowException("Invalid UpsertIssueEntity response from RiffDB")
        return res.outcome.issue

    @staticmethod
    def _issue_to_mlflow(entity: IssueEntity, trace_count: int | None = None) -> Issue:
        return Issue(
            issue_id=entity.issue_id,
            experiment_id=str(entity.experiment_id),
            name=entity.name,
            description=entity.description,
            status=IssueStatus(entity.status),
            severity=IssueSeverity(entity.severity) if entity.severity else None,
            root_causes=(json.loads(entity.root_causes_json) if entity.root_causes_json else None),
            source_run_id=entity.source_run_id,
            categories=(json.loads(entity.categories_json) if entity.categories_json else None),
            created_timestamp=int(entity.created_timestamp_ms),
            last_updated_timestamp=int(entity.last_updated_timestamp_ms),
            created_by=entity.created_by,
            trace_count=trace_count,
        )

    def create_issue(
        self,
        experiment_id: str,
        name: str,
        description: str,
        status: IssueStatus = IssueStatus.PENDING,
        severity: IssueSeverity | None = None,
        root_causes: list[str] | None = None,
        source_run_id: str | None = None,
        categories: list[str] | None = None,
        created_by: str | None = None,
    ) -> Issue:
        exp_id = int(self._numeric_experiment_id(str(experiment_id)))  # type: ignore[attr-defined]
        self._require_active_experiment(exp_id)  # type: ignore[attr-defined]
        now = _now_ms()
        status_val = status.value if isinstance(status, IssueStatus) else str(status)
        sev_val = (
            (severity.value if isinstance(severity, IssueSeverity) else str(severity))
            if severity is not None
            else None
        )
        saved = self._save_issue_entity(
            IssueEntity(
                scope=_SCOPE,
                issue_id=f"iss-{uuid.uuid4().hex}",
                experiment_id=exp_id,
                name=name,
                description=description,
                status=status_val,
                severity=sev_val,
                root_causes_json=(json.dumps(root_causes) if root_causes is not None else None),
                source_run_id=source_run_id,
                categories_json=(json.dumps(categories) if categories is not None else None),
                created_timestamp_ms=now,
                last_updated_timestamp_ms=now,
                created_by=created_by,
            )
        )
        return self._issue_to_mlflow(saved)

    def get_issue(self, issue_id: str) -> Issue:
        return self._issue_to_mlflow(self._fetch_issue_entity(issue_id))

    def search_issues(
        self,
        experiment_id: str | None = None,
        filter_string: str | None = None,
        max_results: int | None = None,
        page_token: str | None = None,
        include_trace_count: bool = False,
    ) -> PagedList[Issue]:
        if max_results is not None:
            self._validate_gateway_max_results(max_results)
            resolved_max = int(max_results)
        else:
            resolved_max = 100
        offset = SearchTraceUtils.parse_start_offset_from_page_token(page_token)  # type: ignore[no-untyped-call]
        parsed_filters = (
            SearchIssuesUtils.parse_search_filter(filter_string)  # type: ignore[no-untyped-call]
            if filter_string
            else []
        )
        target_exp = int(experiment_id) if experiment_id is not None else None
        matched: list[IssueEntity] = []
        for entity in self._list_issue_entities():
            if target_exp is not None and int(entity.experiment_id) != target_exp:
                continue
            ok = True
            for flt in parsed_filters:
                key = str(flt.get("key"))
                comp = str(flt.get("comparator")).upper()
                val = str(flt.get("value"))
                actual = getattr(entity, key, None)
                if comp == "=" and actual != val:
                    ok = False
                    break
                if comp == "!=" and actual == val:
                    ok = False
                    break
            if ok:
                matched.append(entity)

        severity_ranks = {s.value: s._rank for s in IssueSeverity}
        matched.sort(
            key=lambda e: (
                -severity_ranks.get(e.severity or "", -1),
                -int(e.created_timestamp_ms),
                e.issue_id,
            )
        )
        sliced = matched[offset : offset + resolved_max + 1]
        next_token = None
        if len(sliced) > resolved_max:
            next_token = SearchTraceUtils.create_page_token(offset + resolved_max)  # type: ignore[no-untyped-call]
        t_cnt = 0 if include_trace_count else None
        return PagedList(
            [self._issue_to_mlflow(e, trace_count=t_cnt) for e in sliced[:resolved_max]],
            next_token,
        )

    def update_issue(
        self,
        issue_id: str,
        status: IssueStatus | None = None,
        name: str | None = None,
        description: str | None = None,
        severity: IssueSeverity | None = None,
    ) -> Issue:
        existing = self._fetch_issue_entity(issue_id)
        resolved_status = (
            (status.value if isinstance(status, IssueStatus) else str(status))
            if status is not None
            else existing.status
        )
        resolved_name = name if name is not None else existing.name
        resolved_desc = description if description is not None else existing.description
        resolved_sev = (
            (severity.value if isinstance(severity, IssueSeverity) else str(severity))
            if severity is not None
            else existing.severity
        )
        saved = self._save_issue_entity(
            IssueEntity(
                scope=_SCOPE,
                issue_id=existing.issue_id,
                experiment_id=int(existing.experiment_id),
                name=resolved_name,
                description=resolved_desc,
                status=resolved_status,
                severity=resolved_sev,
                root_causes_json=existing.root_causes_json,
                source_run_id=existing.source_run_id,
                categories_json=existing.categories_json,
                created_timestamp_ms=int(existing.created_timestamp_ms),
                last_updated_timestamp_ms=_now_ms(),
                created_by=existing.created_by,
            )
        )
        return self._issue_to_mlflow(saved)
