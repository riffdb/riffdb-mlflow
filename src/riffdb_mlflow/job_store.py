"""MLflow server job store implementation backed by a compiled RiffDB profile."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.jobs.abstract_store import AbstractJobStore
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
    DeleteServerJobEntityInput,
    DeleteServerJobEntityServerJobDeleted,
    DeleteServerJobEntityServerJobMissing,
    GetServerJobFound,
    GetServerJobMissing,
    GetServerJobParams,
    ListServerJobsFound,
    ListServerJobsParams,
    RiffDBMLflowClient,
    ServerJobEntity,
    UpsertServerJobEntityInput,
    UpsertServerJobEntityServerJobUpserted,
)

_SCOPE = "global"


def _now_ms() -> int:
    return int(time.time() * 1000)


class RiffDBJobStore(AbstractJobStore):
    """Pinned MLflow server job store backed by one compiled RiffDB application profile."""

    def __init__(self, store_uri: str) -> None:
        super().__init__()
        self._config = StoreConfig.from_uri(store_uri)
        self._store_uri = store_uri
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
            tls = VerifiedTlsConfig(
                endpoint=conn.endpoint,
                trust_root=conn.trust_root,
                server_name=conn.server_name,
            )
            self._transport = SyncApplicationTransport.connect_verified_tls(tls, metadata)
            self._client = RiffDBMLflowClient(self._transport, AttemptBudget(3))
        return self._client

    def _fetch_job_entity(self, job_id: str) -> ServerJobEntity:
        client = self._application()
        res = client.get_server_job(GetServerJobParams(scope=_SCOPE, job_id=job_id))
        if isinstance(res.value, GetServerJobMissing) or not isinstance(
            res.value, GetServerJobFound
        ):
            raise MlflowException(
                f"Job with ID {job_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.job
        return ServerJobEntity(
            scope=_SCOPE,
            job_id=row.job_id,
            creation_time_ms=int(row.creation_time_ms),
            job_name=row.job_name,
            params_json=row.params_json,
            timeout_json=row.timeout_json,
            status_code=int(row.status_code),
            result_text=row.result_text,
            retry_count=int(row.retry_count),
            last_update_time_ms=int(row.last_update_time_ms),
            status_details_json=row.status_details_json,
        )

    def _list_job_entities(self) -> list[ServerJobEntity]:
        client = self._application()
        entities: list[ServerJobEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_server_jobs(
                ListServerJobsParams(scope=_SCOPE, limit=10, after=cursor)
            )
            if not isinstance(res.value, ListServerJobsFound):
                raise MlflowException("Invalid ListServerJobs response from RiffDB")
            for row in res.value.jobs:
                entities.append(
                    ServerJobEntity(
                        scope=_SCOPE,
                        job_id=row.job_id,
                        creation_time_ms=int(row.creation_time_ms),
                        job_name=row.job_name,
                        params_json=row.params_json,
                        timeout_json=row.timeout_json,
                        status_code=int(row.status_code),
                        result_text=row.result_text,
                        retry_count=int(row.retry_count),
                        last_update_time_ms=int(row.last_update_time_ms),
                        status_details_json=row.status_details_json,
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.jobs:
                break
        return entities

    def _save_job_entity(self, entity: ServerJobEntity) -> ServerJobEntity:
        client = self._application()
        res = client.upsert_server_job_entity(
            UpsertServerJobEntityInput(
                request_id=str(uuid.uuid4()),
                job_id=entity.job_id,
                creation_time_ms=int(entity.creation_time_ms),
                job_name=entity.job_name,
                params_json=entity.params_json,
                timeout_json=entity.timeout_json,
                status_code=int(entity.status_code),
                result_text=entity.result_text,
                retry_count=int(entity.retry_count),
                last_update_time_ms=int(entity.last_update_time_ms),
                status_details_json=entity.status_details_json,
            )
        )
        if not isinstance(res.outcome, UpsertServerJobEntityServerJobUpserted):
            raise MlflowException("Invalid UpsertServerJobEntity response from RiffDB")
        return res.outcome.job

    @staticmethod
    def _entity_to_mlflow(entity: ServerJobEntity) -> Job:
        timeout = float(json.loads(entity.timeout_json)) if entity.timeout_json else None
        status_details = (
            json.loads(entity.status_details_json) if entity.status_details_json else None
        )
        return Job(  # type: ignore[abstract]
            job_id=entity.job_id,
            creation_time=int(entity.creation_time_ms),
            job_name=entity.job_name,
            params=entity.params_json,
            timeout=timeout,
            status=JobStatus.from_int(int(entity.status_code)),
            result=entity.result_text,
            retry_count=int(entity.retry_count),
            last_update_time=int(entity.last_update_time_ms),
            status_details=status_details,
        )

    def create_job(self, job_name: str, params: str, timeout: float | None = None) -> Job:
        job_id = str(uuid.uuid4())
        now = _now_ms()
        saved = self._save_job_entity(
            ServerJobEntity(
                scope=_SCOPE,
                job_id=job_id,
                creation_time_ms=now,
                job_name=job_name,
                params_json=params,
                timeout_json=json.dumps(float(timeout)) if timeout is not None else None,
                status_code=int(JobStatus.PENDING.to_int()),
                result_text=None,
                retry_count=0,
                last_update_time_ms=now,
                status_details_json=None,
            )
        )
        return self._entity_to_mlflow(saved)

    def _update_job(self, job_id: str, new_status: JobStatus, result: str | None = None) -> Job:
        existing = self._fetch_job_entity(job_id)
        current_status = JobStatus.from_int(int(existing.status_code))
        if JobStatus.is_finalized(current_status):
            raise MlflowException(
                f"The Job {job_id} is already finalized with status: {current_status}, "
                "it can't be updated."
            )
        saved = self._save_job_entity(
            ServerJobEntity(
                scope=_SCOPE,
                job_id=existing.job_id,
                creation_time_ms=int(existing.creation_time_ms),
                job_name=existing.job_name,
                params_json=existing.params_json,
                timeout_json=existing.timeout_json,
                status_code=int(new_status.to_int()),
                result_text=result if result is not None else existing.result_text,
                retry_count=int(existing.retry_count),
                last_update_time_ms=_now_ms(),
                status_details_json=existing.status_details_json,
            )
        )
        return self._entity_to_mlflow(saved)

    def start_job(self, job_id: str) -> None:
        existing = self._fetch_job_entity(job_id)
        if int(existing.status_code) != int(JobStatus.PENDING.to_int()):
            raise MlflowException(
                f"Job {job_id} is in {JobStatus.from_int(int(existing.status_code))} state, "
                "cannot start (must be PENDING)"
            )
        self._save_job_entity(
            ServerJobEntity(
                scope=_SCOPE,
                job_id=existing.job_id,
                creation_time_ms=int(existing.creation_time_ms),
                job_name=existing.job_name,
                params_json=existing.params_json,
                timeout_json=existing.timeout_json,
                status_code=int(JobStatus.RUNNING.to_int()),
                result_text=existing.result_text,
                retry_count=int(existing.retry_count),
                last_update_time_ms=_now_ms(),
                status_details_json=existing.status_details_json,
            )
        )

    def reset_job(self, job_id: str) -> None:
        self._update_job(job_id, JobStatus.PENDING)

    def finish_job(self, job_id: str, result: str) -> None:
        self._update_job(job_id, JobStatus.SUCCEEDED, result)

    def fail_job(self, job_id: str, error: str) -> None:
        self._update_job(job_id, JobStatus.FAILED, error)

    def mark_job_timed_out(self, job_id: str) -> None:
        self._update_job(job_id, JobStatus.TIMEOUT)

    def retry_or_fail_job(self, job_id: str, error: str) -> int | None:
        max_retries = int(MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES.get())  # type: ignore[no-untyped-call]
        existing = self._fetch_job_entity(job_id)
        if int(existing.retry_count) >= max_retries:
            self._save_job_entity(
                ServerJobEntity(
                    scope=_SCOPE,
                    job_id=existing.job_id,
                    creation_time_ms=int(existing.creation_time_ms),
                    job_name=existing.job_name,
                    params_json=existing.params_json,
                    timeout_json=existing.timeout_json,
                    status_code=int(JobStatus.FAILED.to_int()),
                    result_text=error,
                    retry_count=int(existing.retry_count),
                    last_update_time_ms=_now_ms(),
                    status_details_json=existing.status_details_json,
                )
            )
            return None
        new_retry = int(existing.retry_count) + 1
        self._save_job_entity(
            ServerJobEntity(
                scope=_SCOPE,
                job_id=existing.job_id,
                creation_time_ms=int(existing.creation_time_ms),
                job_name=existing.job_name,
                params_json=existing.params_json,
                timeout_json=existing.timeout_json,
                status_code=int(JobStatus.PENDING.to_int()),
                result_text=existing.result_text,
                retry_count=new_retry,
                last_update_time_ms=_now_ms(),
                status_details_json=existing.status_details_json,
            )
        )
        return new_retry

    def list_jobs(
        self,
        job_name: str | None = None,
        statuses: list[JobStatus] | None = None,
        begin_timestamp: int | None = None,
        end_timestamp: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[Job]:
        status_ints = {int(s.to_int()) for s in statuses} if statuses else None
        entities = self._list_job_entities()
        entities.sort(key=lambda e: (int(e.creation_time_ms), e.job_id))
        for entity in entities:
            if job_name is not None and entity.job_name != job_name:
                continue
            if status_ints is not None and int(entity.status_code) not in status_ints:
                continue
            if begin_timestamp is not None and int(entity.creation_time_ms) < begin_timestamp:
                continue
            if end_timestamp is not None and int(entity.creation_time_ms) > end_timestamp:
                continue
            if params:
                try:
                    job_params = json.loads(entity.params_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(job_params, dict):
                    continue
                if any(job_params.get(k) != v or k not in job_params for k, v in params.items()):
                    continue
            yield self._entity_to_mlflow(entity)

    def get_job(self, job_id: str) -> Job:
        return self._entity_to_mlflow(self._fetch_job_entity(job_id))

    def cancel_job(self, job_id: str) -> Job:
        return self._update_job(job_id, JobStatus.CANCELED)

    def update_status_details(self, job_id: str, status_details: dict[str, Any]) -> None:
        existing = self._fetch_job_entity(job_id)
        current_details: dict[str, Any] = {}
        if existing.status_details_json:
            try:
                loaded = json.loads(existing.status_details_json)
                if isinstance(loaded, dict):
                    current_details = loaded
            except (json.JSONDecodeError, TypeError):
                current_details = {}
        current_details.update(status_details)
        self._save_job_entity(
            ServerJobEntity(
                scope=_SCOPE,
                job_id=existing.job_id,
                creation_time_ms=int(existing.creation_time_ms),
                job_name=existing.job_name,
                params_json=existing.params_json,
                timeout_json=existing.timeout_json,
                status_code=int(existing.status_code),
                result_text=existing.result_text,
                retry_count=int(existing.retry_count),
                last_update_time_ms=_now_ms(),
                status_details_json=json.dumps(current_details),
            )
        )

    def delete_jobs(self, older_than: int = 0, job_ids: list[str] | None = None) -> list[str]:
        now = _now_ms()
        time_threshold = now - older_than
        finalized_statuses = {
            int(JobStatus.SUCCEEDED.to_int()),
            int(JobStatus.FAILED.to_int()),
            int(JobStatus.TIMEOUT.to_int()),
            int(JobStatus.CANCELED.to_int()),
        }
        job_id_filter = set(job_ids) if job_ids else None
        deleted_ids: list[str] = []
        client = self._application()
        for entity in self._list_job_entities():
            if int(entity.status_code) not in finalized_statuses:
                continue
            if job_id_filter is not None and entity.job_id not in job_id_filter:
                continue
            if older_than > 0 and int(entity.creation_time_ms) >= time_threshold:
                continue
            res = client.delete_server_job_entity(
                DeleteServerJobEntityInput(
                    request_id=str(uuid.uuid4()),
                    job_id=entity.job_id,
                )
            )
            if isinstance(
                res.outcome,
                DeleteServerJobEntityServerJobDeleted | DeleteServerJobEntityServerJobMissing,
            ):
                deleted_ids.append(entity.job_id)
        return deleted_ids
