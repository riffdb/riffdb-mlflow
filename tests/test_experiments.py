from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mlflow.entities import ExperimentTag
from mlflow.entities import LifecycleStage as MLflowLifecycleStage
from mlflow.exceptions import MlflowException
from riffdb_application import Timestamp

from riffdb_mlflow.generated.client import (
    CreateExperimentExperimentCreated,
    CreateExperimentExperimentExists,
    CreateExperimentWithTagsExperimentCreated,
    DeleteExperimentExperimentDeleted,
    DeleteExperimentExperimentMissing,
    Experiment,
    GetExperimentByNameFound,
    GetExperimentByNameFoundExperiment,
    GetExperimentByNameMissing,
    GetExperimentFound,
    GetExperimentFoundExperiment,
    GetExperimentMissing,
    LifecycleStage,
    RenameExperimentExperimentMissing,
    RenameExperimentExperimentRenamed,
    RestoreExperimentExperimentMissing,
    RestoreExperimentExperimentRestored,
)
from riffdb_mlflow.store import RiffDBTrackingStore


class FakeExperimentClient:
    def __init__(self) -> None:
        self.rows: dict[int, Experiment] = {}

    def create_experiment(self, command: Any) -> SimpleNamespace:
        if command.experiment_id in self.rows or any(
            row.name == command.name for row in self.rows.values()
        ):
            return SimpleNamespace(outcome=CreateExperimentExperimentExists())
        row = Experiment(
            scope="global",
            experiment_id=command.experiment_id,
            name=command.name,
            name_digest=command.name_digest,
            artifact_location=command.artifact_location,
            lifecycle_stage=LifecycleStage.ACTIVE,
            creation_time=Timestamp(seconds=1_800_000_000, nanos=123_000_000),
            last_update_time=Timestamp(seconds=1_800_000_000, nanos=123_000_000),
            packed_tags=command.packed_tags,
            is_gateway=getattr(command, "is_gateway", False),
            revision=0,
        )
        self.rows[command.experiment_id] = row
        return SimpleNamespace(outcome=CreateExperimentExperimentCreated(experiment=row))

    def create_experiment_with_tags(self, command: Any) -> SimpleNamespace:
        result = self.create_experiment(command)
        if isinstance(result.outcome, CreateExperimentExperimentCreated):
            return SimpleNamespace(
                outcome=CreateExperimentWithTagsExperimentCreated(
                    experiment=result.outcome.experiment
                )
            )
        return result

    def get_experiment(self, query: Any) -> SimpleNamespace:
        row = self.rows.get(query.experiment_id)
        if row is None:
            return SimpleNamespace(value=GetExperimentMissing())
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

    def get_experiment_by_name(self, query: Any) -> SimpleNamespace:
        for row in self.rows.values():
            if row.name == query.name:
                return SimpleNamespace(
                    value=GetExperimentByNameFound(
                        experiment=GetExperimentByNameFoundExperiment(
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
        return SimpleNamespace(value=GetExperimentByNameMissing())

    def delete_experiment(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.experiment_id)
        if row is None:
            return SimpleNamespace(outcome=DeleteExperimentExperimentMissing())
        updated = self._lifecycle(row, LifecycleStage.DELETED)
        self.rows[command.experiment_id] = updated
        return SimpleNamespace(outcome=DeleteExperimentExperimentDeleted(experiment=updated))

    def restore_experiment(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.experiment_id)
        if row is None:
            return SimpleNamespace(outcome=RestoreExperimentExperimentMissing())
        updated = self._lifecycle(row, LifecycleStage.ACTIVE)
        self.rows[command.experiment_id] = updated
        return SimpleNamespace(outcome=RestoreExperimentExperimentRestored(experiment=updated))

    def rename_experiment(self, command: Any) -> SimpleNamespace:
        row = self.rows.get(command.experiment_id)
        if row is None:
            return SimpleNamespace(outcome=RenameExperimentExperimentMissing())
        updated = Experiment(
            scope=row.scope,
            experiment_id=row.experiment_id,
            name=command.name,
            name_digest=command.name_digest,
            artifact_location=row.artifact_location,
            lifecycle_stage=row.lifecycle_stage,
            creation_time=row.creation_time,
            last_update_time=Timestamp(seconds=1_800_000_001, nanos=0),
            packed_tags=row.packed_tags,
            is_gateway=row.is_gateway,
            revision=row.revision + 1,
        )
        self.rows[command.experiment_id] = updated
        return SimpleNamespace(outcome=RenameExperimentExperimentRenamed(experiment=updated))

    @staticmethod
    def _lifecycle(row: Experiment, stage: LifecycleStage) -> Experiment:
        return Experiment(
            scope=row.scope,
            experiment_id=row.experiment_id,
            name=row.name,
            name_digest=row.name_digest,
            artifact_location=row.artifact_location,
            lifecycle_stage=stage,
            creation_time=row.creation_time,
            last_update_time=Timestamp(seconds=1_800_000_001, nanos=0),
            packed_tags=row.packed_tags,
            is_gateway=row.is_gateway,
            revision=row.revision + 1,
        )


class RecordingSearchClient:
    def __init__(self, count: int) -> None:
        self.rows = tuple(
            Experiment(
                scope="global",
                experiment_id=index,
                name=f"experiment-{index:03d}",
                name_digest=RiffDBTrackingStore._digest(f"experiment-{index:03d}"),
                artifact_location=f"s3://artifacts/{index}",
                lifecycle_stage=LifecycleStage.ACTIVE,
                creation_time=Timestamp(seconds=1_800_000_000 + index, nanos=0),
                last_update_time=Timestamp(seconds=1_800_000_000 + index, nanos=0),
                packed_tags=b"{}",
                is_gateway=False,
                revision=0,
            )
            for index in range(count)
        )
        self.calls: list[tuple[str, Any]] = []

    def _page(self, operation: str, parameters: Any) -> SimpleNamespace:
        self.calls.append((operation, parameters))
        offset = int(parameters.after) if parameters.after is not None else 0
        end = min(offset + parameters.limit, len(self.rows))
        next_cursor = str(end) if end < len(self.rows) else None
        return SimpleNamespace(
            value=SimpleNamespace(experiments=self.rows[offset:end]),
            next_cursor=next_cursor,
        )

    def search_experiments(self, parameters: Any) -> SimpleNamespace:
        return self._page("SearchExperiments", parameters)

    def search_experiments_by_two_tags(self, parameters: Any) -> SimpleNamespace:
        return self._page("SearchExperimentsByTwoTags", parameters)

    def search_experiments_by_tag_is_null(self, parameters: Any) -> SimpleNamespace:
        return self._page("SearchExperimentsByTagIsNull", parameters)

    def search_experiments_by_tag_is_not_null(self, parameters: Any) -> SimpleNamespace:
        return self._page("SearchExperimentsByTagIsNotNull", parameters)

    def search_experiments_by_tag_not_equal(self, parameters: Any) -> SimpleNamespace:
        return self._page("SearchExperimentsByTagNotEqual", parameters)

    def search_experiments_by_name_not_equal(self, parameters: Any) -> SimpleNamespace:
        return self._page("SearchExperimentsByNameNotEqual", parameters)

    def search_experiments_by_name_ilike_and_gateway_is_null(
        self, parameters: Any
    ) -> SimpleNamespace:
        return self._page("SearchExperimentsByNameILikeAndGatewayIsNull", parameters)

    def search_experiments_by_tag_and_gateway_is_null(self, parameters: Any) -> SimpleNamespace:
        return self._page("SearchExperimentsByTagAndGatewayIsNull", parameters)

    def search_experiments_by_tag_is_not_null_and_gateway_is_null(
        self, parameters: Any
    ) -> SimpleNamespace:
        return self._page("SearchExperimentsByTagIsNotNullAndGatewayIsNull", parameters)

    def search_experiments_by_name_ilike_and_tag_and_gateway_is_null(
        self, parameters: Any
    ) -> SimpleNamespace:
        return self._page("SearchExperimentsByNameILikeAndTagAndGatewayIsNull", parameters)


@pytest.fixture
def store() -> RiffDBTrackingStore:
    value = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    value._client = FakeExperimentClient()
    return value


def test_experiment_create_get_name_and_lifecycle(
    store: RiffDBTrackingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("riffdb_mlflow.store.secrets.randbelow", lambda _: 7)
    experiment_id = store.create_experiment(
        "training",
        tags=[ExperimentTag("team", "research"), ExperimentTag("stage", "dev")],
    )

    assert experiment_id == "100000007"
    experiment = store.get_experiment(experiment_id)
    assert experiment.name == "training"
    assert experiment.artifact_location == f"s3://artifacts/{experiment_id}"
    assert experiment.tags == {"stage": "dev", "team": "research"}
    assert experiment.creation_time == 1_800_000_000_123
    by_name = store.get_experiment_by_name("training")
    assert by_name is not None
    assert by_name.experiment_id == experiment.experiment_id
    assert by_name.tags == experiment.tags

    store.rename_experiment(experiment_id, "renamed")
    assert store.get_experiment_by_name("training") is None
    assert store.get_experiment_by_name("renamed").name == "renamed"  # type: ignore[union-attr]

    store.delete_experiment(experiment_id)
    assert store.get_experiment(experiment_id).lifecycle_stage == MLflowLifecycleStage.DELETED
    store.restore_experiment(experiment_id)
    assert store.get_experiment(experiment_id).lifecycle_stage == MLflowLifecycleStage.ACTIVE


def test_default_experiment_is_initialized_once(store: RiffDBTrackingStore) -> None:
    first = store.get_experiment("0")
    second = store.get_experiment("0")
    assert first.name == "Default"
    assert second.experiment_id == first.experiment_id
    assert second.creation_time == first.creation_time
    assert set(store._client.rows) == {0}  # type: ignore[union-attr]


def test_default_artifact_root_falls_back_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("_MLFLOW_SERVER_ARTIFACT_ROOT", raising=False)
    store = RiffDBTrackingStore("riffdb://mlflow-development")
    store._client = FakeExperimentClient()
    assert store.artifact_uri is not None
    assert store.artifact_uri.endswith("/mlruns") or "mlruns" in store.artifact_uri
    default_exp = store.get_experiment("0")
    assert default_exp.artifact_location.endswith("/0")


def test_duplicate_name_and_missing_id_map_to_mlflow_errors(
    store: RiffDBTrackingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = iter((1, 2, 3, 4))
    monkeypatch.setattr("riffdb_mlflow.store.secrets.randbelow", lambda _: next(values))
    store.create_experiment("duplicate")
    with pytest.raises(MlflowException, match="already exists"):
        store.create_experiment("duplicate")
    with pytest.raises(MlflowException, match="No Experiment"):
        store.get_experiment("99")


def test_search_coalesces_compiled_pages_without_skipping_rows() -> None:
    client = RecordingSearchClient(120)
    store = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    store._client = client
    store._default_ready = True

    first = store.search_experiments(max_results=1)
    assert [experiment.experiment_id for experiment in first] == ["0"]
    assert first.token == "1"

    second = store.search_experiments(max_results=110, page_token=first.token)
    assert [experiment.experiment_id for experiment in second] == [
        str(value) for value in range(1, 111)
    ]
    assert second.token == "111"
    assert [(call[1].limit, call[1].after) for call in client.calls] == [
        (1, None),
        (100, "1"),
        (10, "101"),
    ]


def test_two_exact_tag_filters_dispatch_one_compiled_intersection() -> None:
    client = RecordingSearchClient(1)
    store = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    store._client = client
    store._default_ready = True

    result = store.search_experiments(
        max_results=10,
        filter_string="tags.team = 'research' AND tags.stage = 'prod'",
        order_by=["name ASC", "experiment_id DESC"],
    )

    assert len(result) == 1
    assert [name for name, _ in client.calls] == ["SearchExperimentsByTwoTags"]
    parameters = client.calls[0][1]
    assert parameters.first_tag_key == "team"
    assert parameters.first_value_digest == RiffDBTrackingStore._digest("research")
    assert parameters.second_tag_key == "stage"
    assert parameters.second_value_digest == RiffDBTrackingStore._digest("prod")


def test_mlflow_ui_gateway_is_null_and_extended_experiment_filters_dispatch() -> None:
    client = RecordingSearchClient(1)
    store = RiffDBTrackingStore("riffdb://mlflow-development", "s3://artifacts")
    store._client = client
    store._default_ready = True

    # 1. Default MLflow 3.15 UI filter: tags.`mlflow.experiment.isGateway` IS NULL
    store.search_experiments(
        max_results=10,
        filter_string="tags.`mlflow.experiment.isGateway` IS NULL",
    )
    assert client.calls[-1][0] == "SearchExperimentsByTagIsNull"
    assert client.calls[-1][1].tag_key == "mlflow.experiment.isGateway"

    # 2. UI search bar + gateway filter
    store.search_experiments(
        max_results=10,
        filter_string="name ILIKE '%train%' AND tags.`mlflow.experiment.isGateway` IS NULL",
    )
    assert client.calls[-1][0] == "SearchExperimentsByNameILikeAndGatewayIsNull"
    assert client.calls[-1][1].pattern == "%train%"

    # 3. UI tag filter + gateway filter
    store.search_experiments(
        max_results=10,
        filter_string="tags.team = 'research' AND tags.`mlflow.experiment.isGateway` IS NULL",
    )
    assert client.calls[-1][0] == "SearchExperimentsByTagAndGatewayIsNull"
    assert client.calls[-1][1].tag_key == "team"

    # 4. UI search bar + tag filter + gateway filter (3 predicates)
    store.search_experiments(
        max_results=10,
        filter_string=(
            "name ILIKE '%train%' AND tags.team = 'research' "
            "AND tags.`mlflow.experiment.isGateway` IS NULL"
        ),
    )
    assert client.calls[-1][0] == "SearchExperimentsByNameILikeAndTagAndGatewayIsNull"

    # 5. IS NOT NULL, != on tag, != on name
    store.search_experiments(max_results=10, filter_string="tags.team IS NOT NULL")
    assert client.calls[-1][0] == "SearchExperimentsByTagIsNotNull"
    store.search_experiments(max_results=10, filter_string="tags.team != 'staging'")
    assert client.calls[-1][0] == "SearchExperimentsByTagNotEqual"
    store.search_experiments(max_results=10, filter_string="name != 'Default'")
    assert client.calls[-1][0] == "SearchExperimentsByNameNotEqual"
