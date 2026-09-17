from __future__ import annotations

import math
import struct
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from mlflow.entities import LifecycleStage as MLflowLifecycleStage
from mlflow.entities import Metric, Param, RunStatus, RunTag, ViewType
from mlflow.exceptions import MlflowException
from riffdb_application import Timestamp

from riffdb_mlflow.generated.client import (
    CreateRunWithTagsRunCreated,
    DeleteRunRunDeleted,
    DeleteRunRunMissing,
    Experiment,
    GetExperimentFound,
    GetExperimentFoundExperiment,
    GetMetricHistoryFound,
    GetMetricHistoryFoundMetrics,
    GetRunFound,
    GetRunFoundRun,
    GetRunMissing,
    LifecycleStage,
    LogBatchBatchLogged,
    LogBatchParamConflict,
    LogBatchRunInactive,
    LogBatchRunMissing,
    LogBatchRunStale,
    RestoreRunRunMissing,
    RestoreRunRunRestored,
    Run,
    RunDataKind,
    RunDatum,
    UpdateRunInfoRunInactive,
    UpdateRunInfoRunMissing,
    UpdateRunInfoRunStale,
    UpdateRunInfoRunUpdated,
    ValidateRunActiveRunActive,
    ValidateRunActiveRunInactive,
    ValidateRunActiveRunMissing,
)
from riffdb_mlflow.generated.client import (
    RunStatus as RiffDBRunStatus,
)
from riffdb_mlflow.store import RiffDBTrackingStore


class FakeRunClient:
    def __init__(self, experiment_id: int = 7) -> None:
        self.experiment = Experiment(
            scope="global",
            experiment_id=experiment_id,
            name="training",
            name_digest=RiffDBTrackingStore._digest("training"),
            artifact_location=f"s3://artifacts/{experiment_id}",
            lifecycle_stage=LifecycleStage.ACTIVE,
            creation_time=Timestamp(seconds=1_800_000_000, nanos=0),
            last_update_time=Timestamp(seconds=1_800_000_000, nanos=0),
            packed_tags=b"{}",
            revision=0,
        )
        self.rows: dict[str, Run] = {}
        self.data: dict[tuple[str, RunDataKind, str, str], RunDatum] = {}
        self.search_calls: list[tuple[str, Any]] = []
        self.stale_once = False

    @staticmethod
    def _get_row(row: Run) -> GetRunFoundRun:
        return GetRunFoundRun(
            experiment_id=row.experiment_id,
            run_id=row.run_id,
            user_id=row.user_id,
            status=row.status,
            start_time=row.start_time,
            end_time=row.end_time,
            lifecycle_stage=row.lifecycle_stage,
            artifact_uri=row.artifact_uri,
            run_name=row.run_name,
            packed_tags=row.packed_tags,
            packed_params=row.packed_params,
            packed_latest_metrics=row.packed_latest_metrics,
            revision=row.revision,
            initialized=row.initialized,
        )

    def get_experiment(self, _query: Any) -> SimpleNamespace:
        row = self.experiment
        return SimpleNamespace(
            value=GetExperimentFound(
                experiment=GetExperimentFoundExperiment(
                    experiment_id=row.experiment_id,
                    name=row.name,
                    artifact_location=row.artifact_location,
                    lifecycle_stage=row.lifecycle_stage,
                    creation_time=row.creation_time,
                    last_update_time=row.last_update_time,
                    packed_tags=row.packed_tags,
                    revision=row.revision,
                )
            )
        )

    def create_run_with_tags(self, command: Any) -> SimpleNamespace:
        row = Run(
            experiment_id=command.experiment_id,
            run_id=command.run_id,
            user_id=command.user_id,
            status=RiffDBRunStatus.RUNNING,
            start_time=command.start_time,
            end_time=None,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_uri=command.artifact_uri,
            run_name=command.run_name,
            packed_tags=command.packed_tags,
            packed_params=command.packed_params,
            packed_latest_metrics=command.packed_latest_metrics,
            revision=0,
            initialized=True,
        )
        self.rows[row.run_id] = row
        return SimpleNamespace(outcome=CreateRunWithTagsRunCreated(run=row))

    def get_run(self, query: Any) -> SimpleNamespace:
        row = self.rows.get(query.run_id)
        if row is None:
            return SimpleNamespace(value=GetRunMissing())
        return SimpleNamespace(value=GetRunFound(run=self._get_row(row)))

    def validate_run_active(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.run_id)
        if row is None:
            return SimpleNamespace(outcome=ValidateRunActiveRunMissing())
        if row.lifecycle_stage != LifecycleStage.ACTIVE:
            return SimpleNamespace(outcome=ValidateRunActiveRunInactive())
        return SimpleNamespace(outcome=ValidateRunActiveRunActive())

    def log_batch(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.run_id)
        if row is None:
            return SimpleNamespace(outcome=LogBatchRunMissing())
        if row.lifecycle_stage != LifecycleStage.ACTIVE:
            return SimpleNamespace(outcome=LogBatchRunInactive())
        if row.revision != command.expected_revision:
            return SimpleNamespace(outcome=LogBatchRunStale())
        for item in command.items:
            key = (item.run_id, item.kind, item.item_key, item.entry_id)
            stored = self.data.get(key)
            if (
                item.kind == RunDataKind.PARAM
                and stored is not None
                and stored.text_value != item.text_value
            ):
                return SimpleNamespace(outcome=LogBatchParamConflict(param_key=item.logical_key))
        for item in command.items:
            key = (item.run_id, item.kind, item.item_key, item.entry_id)
            self.data[key] = replace(item, promote_latest=False)
            if item.kind == RunDataKind.METRIC and item.promote_latest:
                latest = replace(
                    item,
                    kind=RunDataKind.LATEST_METRIC,
                    item_key=item.latest_item_key,
                    entry_id="",
                    sort_key=item.latest_sort_key,
                    promote_latest=False,
                )
                self.data[(item.run_id, latest.kind, latest.item_key, "")] = latest
        updated = replace(
            row,
            packed_tags=command.packed_tags,
            packed_params=command.packed_params,
            packed_latest_metrics=command.packed_latest_metrics,
            revision=row.revision + 1,
        )
        self.rows[row.run_id] = updated
        return SimpleNamespace(outcome=LogBatchBatchLogged(run=updated))

    def get_metric_history(self, parameters: Any) -> SimpleNamespace:
        rows = sorted(
            (
                item
                for item in self.data.values()
                if item.experiment_id == parameters.experiment_id
                and item.kind == parameters.kind
                and item.item_key == parameters.item_key
                and item.value_digest == parameters.value_digest
            ),
            key=lambda item: (item.sort_key, item.run_id, item.entry_id),
        )
        offset = int(parameters.after) if parameters.after is not None else 0
        end = min(offset + parameters.limit, len(rows))
        values = tuple(
            GetMetricHistoryFoundMetrics(
                logical_key=item.logical_key,
                value_bits=item.value_bits,
                metric_timestamp=item.metric_timestamp,
                metric_step=item.metric_step,
                is_nan=item.is_nan,
            )
            for item in rows[offset:end]
        )
        return SimpleNamespace(
            value=GetMetricHistoryFound(metrics=values),
            next_cursor=str(end) if end < len(rows) else None,
        )

    def update_run_info(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.run_id)
        if row is None:
            return SimpleNamespace(outcome=UpdateRunInfoRunMissing())
        if row.lifecycle_stage != LifecycleStage.ACTIVE:
            return SimpleNamespace(outcome=UpdateRunInfoRunInactive())
        if self.stale_once:
            self.stale_once = False
            return SimpleNamespace(outcome=UpdateRunInfoRunStale())
        if row.revision != command.expected_revision:
            return SimpleNamespace(outcome=UpdateRunInfoRunStale())
        updated = replace(
            row,
            status=command.status,
            end_time=command.end_time,
            run_name=command.run_name,
            packed_tags=command.packed_tags,
            revision=row.revision + 1,
        )
        self.rows[row.run_id] = updated
        return SimpleNamespace(outcome=UpdateRunInfoRunUpdated(run=updated))

    def delete_run(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.run_id)
        if row is None:
            return SimpleNamespace(outcome=DeleteRunRunMissing())
        updated = replace(
            row,
            lifecycle_stage=LifecycleStage.DELETED,
            revision=row.revision + 1,
        )
        self.rows[row.run_id] = updated
        return SimpleNamespace(outcome=DeleteRunRunDeleted(run=updated))

    def restore_run(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.run_id)
        if row is None:
            return SimpleNamespace(outcome=RestoreRunRunMissing())
        updated = replace(
            row,
            lifecycle_stage=LifecycleStage.ACTIVE,
            revision=row.revision + 1,
        )
        self.rows[row.run_id] = updated
        return SimpleNamespace(outcome=RestoreRunRunRestored(run=updated))

    def _search(self, operation: str, parameters: Any) -> SimpleNamespace:
        self.search_calls.append((operation, parameters))
        rows = [
            row
            for row in self.rows.values()
            if row.experiment_id in parameters.experiment_ids
            and (
                not hasattr(parameters, "lifecycle_stage")
                or row.lifecycle_stage == parameters.lifecycle_stage
            )
        ]
        rows.sort(key=lambda row: (-row.start_time, row.run_id))
        offset = int(parameters.after) if parameters.after is not None else 0
        end = min(offset + parameters.limit, len(rows))
        return SimpleNamespace(
            value=SimpleNamespace(runs=tuple(rows[offset:end])),
            next_cursor=str(end) if end < len(rows) else None,
        )

    def search_runs(self, parameters: Any) -> SimpleNamespace:
        return self._search("SearchRuns", parameters)

    def search_runs_by_lifecycle(self, parameters: Any) -> SimpleNamespace:
        return self._search("SearchRunsByLifecycle", parameters)


@pytest.fixture
def store() -> RiffDBTrackingStore:
    value = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    value._client = FakeRunClient()
    value._default_ready = True
    return value


def test_run_create_get_update_and_lifecycle(
    store: RiffDBTrackingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("riffdb_mlflow.store.secrets.token_hex", lambda _: "0123456789abcdef")
    created = store.create_run(
        "7",
        "alice",
        1_700_000_000_123,
        [RunTag("team", "research")],
        "first-run",
    )
    assert created.info.run_id == "00000000000000070123456789abcdef"
    assert created.info.artifact_uri.endswith(f"/{created.info.run_id}/artifacts")
    assert created.data.tags == {"mlflow.runName": "first-run", "team": "research"}

    client = store._client
    assert isinstance(client, FakeRunClient)
    client.stale_once = True
    updated = store.update_run_info(
        created.info.run_id,
        RunStatus.FINISHED,
        1_700_000_001_000,
        "renamed-run",
    )
    assert updated.status == "FINISHED"
    assert updated.end_time == 1_700_000_001_000
    assert store.get_run(created.info.run_id).data.tags["mlflow.runName"] == "renamed-run"

    store.delete_run(created.info.run_id)
    assert store.get_run(created.info.run_id).info.lifecycle_stage == MLflowLifecycleStage.DELETED
    with pytest.raises(MlflowException, match="active state"):
        store.update_run_info(created.info.run_id, RunStatus.RUNNING, None, None)
    store.restore_run(created.info.run_id)
    assert store.get_run(created.info.run_id).info.lifecycle_stage == MLflowLifecycleStage.ACTIVE


def test_create_rejects_conflicting_run_names_before_write(store: RiffDBTrackingStore) -> None:
    client = store._client
    assert isinstance(client, FakeRunClient)
    with pytest.raises(MlflowException, match="different values"):
        store.create_run(
            "7",
            "alice",
            1,
            [RunTag("mlflow.runName", "tag-name")],
            "argument-name",
        )
    assert client.rows == {}


def test_run_search_coalesces_compiled_pages_and_dispatches_lifecycle() -> None:
    client = FakeRunClient()
    for index in range(120):
        run_id = f"0000000000000007{index:016x}"
        client.rows[run_id] = Run(
            experiment_id=7,
            run_id=run_id,
            user_id="alice",
            status=RiffDBRunStatus.RUNNING,
            start_time=120 - index,
            end_time=None,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_uri=f"s3://artifacts/7/{run_id}/artifacts",
            run_name=f"run-{index}",
            packed_tags=b"{}",
            packed_params=b"{}",
            packed_latest_metrics=b"{}",
            revision=0,
            initialized=True,
        )
    store = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    store._client = client
    store._default_ready = True

    first = store.search_runs(["7"], "", ViewType.ACTIVE_ONLY, max_results=1)
    assert len(first) == 1
    assert first.token == "1"
    second = store.search_runs(
        ["7"],
        "",
        ViewType.ACTIVE_ONLY,
        max_results=100,
        page_token=first.token,
    )
    assert len(second) == 100
    assert second.token == "101"
    assert [call[1].limit for call in client.search_calls] == [1] + [2] * 50
    assert {call[0] for call in client.search_calls} == {"SearchRunsByLifecycle"}

    before = len(client.search_calls)
    with pytest.raises(MlflowException, match="filters are not implemented"):
        store.search_runs(["7"], "attributes.status = 'RUNNING'", ViewType.ALL)
    assert len(client.search_calls) == before


def _seed_active_run(store: RiffDBTrackingStore) -> str:
    run_id = "00000000000000070000000000000042"
    client = store._client
    assert isinstance(client, FakeRunClient)
    client.rows[run_id] = Run(
        experiment_id=7,
        run_id=run_id,
        user_id="alice",
        status=RiffDBRunStatus.RUNNING,
        start_time=1,
        end_time=None,
        lifecycle_stage=LifecycleStage.ACTIVE,
        artifact_uri=f"s3://artifacts/7/{run_id}/artifacts",
        run_name="batch-run",
        packed_tags=b'{"mlflow.runName":"batch-run"}',
        packed_params=b"{}",
        packed_latest_metrics=b"{}",
        revision=0,
        initialized=True,
    )
    return run_id


def _float_from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _bits_from_float(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def test_log_batch_preserves_metric_payloads_latest_order_and_pages(
    store: RiffDBTrackingStore,
) -> None:
    run_id = _seed_active_run(store)
    first_nan_bits = 0x7FF8000000000042
    second_nan_bits = 0x7FF8000000000099
    first_nan = _float_from_bits(first_nan_bits)
    second_nan = _float_from_bits(second_nan_bits)

    store.log_batch(
        run_id,
        metrics=[
            Metric("loss", math.inf, 5, 1),
            Metric("loss", -0.0, 10, 0),
            Metric("loss", first_nan, 20, 2),
            Metric("loss", second_nan, 20, 2),
        ],
        params=[Param("optimizer", "adam")],
        tags=[RunTag("team", "one"), RunTag("team", "two")],
    )

    run = store.get_run(run_id)
    assert run.data.params == {"optimizer": "adam"}
    assert run.data.tags["team"] == "two"
    assert math.isnan(run.data.metrics["loss"])
    assert _bits_from_float(run.data.metrics["loss"]) == first_nan_bits

    first = store.get_metric_history(run_id, "loss", max_results=1)
    assert len(first) == 1
    assert first.token == "1"
    assert math.isinf(first[0].value)
    remainder = store.get_metric_history(run_id, "loss", max_results=100, page_token=first.token)
    assert len(remainder) == 2
    assert remainder.token is None
    assert _bits_from_float(remainder[0].value) == _bits_from_float(-0.0)
    assert _bits_from_float(remainder[1].value) == first_nan_bits

    store.log_batch(
        run_id,
        metrics=[Metric("loss", 99.0, 999, 1)],
        params=[],
        tags=[],
    )
    assert _bits_from_float(store.get_run(run_id).data.metrics["loss"]) == first_nan_bits


def test_log_batch_maps_transaction_current_param_conflict_and_empty_validation(
    store: RiffDBTrackingStore,
) -> None:
    run_id = _seed_active_run(store)
    store.log_batch(run_id, metrics=[], params=[Param("depth", "3")], tags=[])
    with pytest.raises(MlflowException, match="Changing param values is not allowed"):
        store.log_batch(run_id, metrics=[], params=[Param("depth", "5")], tags=[])
    assert store.get_run(run_id).data.params == {"depth": "3"}

    store.log_batch(run_id, metrics=[], params=[], tags=[])
    store.delete_run(run_id)
    with pytest.raises(MlflowException, match="active state"):
        store.log_batch(run_id, metrics=[], params=[], tags=[])
