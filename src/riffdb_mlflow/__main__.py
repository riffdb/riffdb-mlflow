"""Disposable real-driver smoke runner used by ``riffdb dev --run``."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from hashlib import sha256
from typing import Any, cast

from mlflow.entities import (
    ExperimentTag as MLflowExperimentTag,
)
from mlflow.entities import (
    LifecycleStage as MLflowLifecycleStage,
)
from mlflow.entities import (
    Metric,
    Param,
    RunTag,
    ViewType,
)
from mlflow.entities import (
    RunStatus as MLflowRunStatus,
)
from mlflow.exceptions import MlflowException
from riffdb_application import (
    AttemptBudget,
    BearerCredential,
    CallMetadata,
    DatabaseAlias,
    OutcomeUnknown,
    QueryOptions,
    SyncApplicationTransport,
    TypedCommandResult,
)

from riffdb_mlflow.generated.client import (
    CreateExperimentWithTagsExperimentCreated,
    CreateExperimentWithTagsInput,
    CreateRunWithTagsInput,
    CreateRunWithTagsRunCreated,
    DeleteExperimentTagExperimentTagDeleted,
    DeleteExperimentTagInput,
    DeleteRunInput,
    DeleteRunRunDeleted,
    ExperimentOrder,
    ExperimentTag,
    GetMetricHistoryFound,
    GetMetricHistoryParams,
    GetRunFound,
    GetRunParams,
    LifecycleStage,
    LogBatchBatchLogged,
    LogBatchInput,
    RestoreRunInput,
    RestoreRunRunRestored,
    RiffDBMLflowClient,
    RunDataKind,
    RunDatum,
    RunStatus,
    SearchExperimentsByNameILikeFound,
    SearchExperimentsByNameILikeParams,
    SearchExperimentsByTagFound,
    SearchExperimentsByTagParams,
    SearchExperimentsByTwoTagsFound,
    SearchExperimentsByTwoTagsParams,
    SearchExperimentsFound,
    SearchExperimentsParams,
    SearchRunsByLifecycleFound,
    SearchRunsByLifecycleParams,
    SearchRunsFound,
    SearchRunsParams,
    SetExperimentTagExperimentTagSet,
    SetExperimentTagInput,
    UpdateRunInfoInput,
    UpdateRunInfoRunUpdated,
)

_SCOPE = "global"


def _digest(value: str) -> bytes:
    return sha256(value.encode("utf-8")).digest()


def _hash64(value: str) -> int:
    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:8], "big")


def _metric_encoding(value: float) -> tuple[int, int, int, bool]:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    is_nan = math.isnan(value)
    if is_nan:
        sanitized = 0.0
    elif math.isinf(value):
        sanitized = sys.float_info.max if value > 0 else -sys.float_info.max
    else:
        sanitized = value
    if sanitized == 0.0:
        sanitized = 0.0
    identity = struct.unpack(">Q", struct.pack(">d", sanitized))[0]
    order = (~identity) & ((1 << 64) - 1) if identity & (1 << 63) else identity ^ (1 << 63)
    return bits, identity, order, is_nan


def _signed_sort(value: int) -> str:
    return f"{(value + (1 << 63)) & ((1 << 64) - 1):016x}"


def _metric_identity(timestamp: int, step: int, identity: int, is_nan: bool) -> str:
    return (
        _signed_sort(timestamp) + _signed_sort(step) + f"{identity:016x}" + ("1" if is_nan else "0")
    )


def _metric_sort(timestamp: int, step: int, order: int, is_nan: bool) -> str:
    return _signed_sort(timestamp) + _signed_sort(step) + f"{order:016x}" + ("1" if is_nan else "0")


def _metric_item_key(run_id: str, key: str) -> str:
    return f"{run_id}:{key}"


def _encode_latest_metrics(values: dict[str, tuple[int, int, int]]) -> bytes:
    return json.dumps(
        {key: [f"{bits:016x}", timestamp, step] for key, (bits, timestamp, step) in values.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _latest_item_key(kind: RunDataKind, key: str, entry_id: str, *, promote_latest: bool) -> str:
    if kind == RunDataKind.METRIC and promote_latest:
        return key
    digest = sha256(f"{kind.value}:{key}:{entry_id}".encode()).hexdigest()
    return f"~{digest}"


def _run_tag(experiment_id: int, run_id: str, key: str, value: str) -> RunDatum:
    return RunDatum(
        experiment_id=experiment_id,
        run_id=run_id,
        kind=RunDataKind.TAG,
        item_key=key,
        entry_id="",
        logical_key=key,
        latest_item_key=_latest_item_key(RunDataKind.TAG, key, "", promote_latest=False),
        sort_key="",
        latest_sort_key="",
        text_value=value,
        value_digest=_digest(value),
        value_bits=0,
        value_identity=0,
        value_order=0,
        metric_timestamp=0,
        metric_step=0,
        is_nan=False,
        promote_latest=False,
    )


def _run_param(experiment_id: int, run_id: str, key: str, value: str) -> RunDatum:
    row = _run_tag(experiment_id, run_id, key, value)
    return RunDatum(
        experiment_id=row.experiment_id,
        run_id=row.run_id,
        kind=RunDataKind.PARAM,
        item_key=row.item_key,
        entry_id=row.entry_id,
        logical_key=row.logical_key,
        latest_item_key=_latest_item_key(RunDataKind.PARAM, key, "", promote_latest=False),
        sort_key=row.sort_key,
        latest_sort_key=row.latest_sort_key,
        text_value=row.text_value,
        value_digest=row.value_digest,
        value_bits=0,
        value_identity=0,
        value_order=0,
        metric_timestamp=0,
        metric_step=0,
        is_nan=False,
        promote_latest=False,
    )


def _run_metric(
    experiment_id: int,
    run_id: str,
    key: str,
    value: float,
    timestamp: int,
    step: int,
    *,
    promote_latest: bool,
) -> RunDatum:
    bits, identity, order, is_nan = _metric_encoding(value)
    entry_id = _metric_identity(timestamp, step, identity, is_nan)
    return RunDatum(
        experiment_id=experiment_id,
        run_id=run_id,
        kind=RunDataKind.METRIC,
        item_key=_metric_item_key(run_id, key),
        entry_id=entry_id,
        logical_key=key,
        latest_item_key=_latest_item_key(
            RunDataKind.METRIC, key, entry_id, promote_latest=promote_latest
        ),
        sort_key=_metric_sort(timestamp, step, order, is_nan),
        latest_sort_key=f"{order:016x}",
        text_value="",
        value_digest=bytes(32),
        value_bits=bits,
        value_identity=identity,
        value_order=order,
        metric_timestamp=timestamp,
        metric_step=step,
        is_nan=is_nan,
        promote_latest=promote_latest,
    )


def _metadata(credential_file: str, database: str) -> CallMetadata:
    credential = BearerCredential.from_protected_file(credential_file)
    if os.environ.get("RIFFDB_TENANT") == "":
        os.environ.pop("RIFFDB_TENANT", None)
    return CallMetadata.authenticated(credential).with_database(DatabaseAlias(database))


def _create(
    client: RiffDBMLflowClient, experiment_id: int, name: str, stage: str
) -> TypedCommandResult[CreateExperimentWithTagsExperimentCreated]:
    tags = {"team": "research", "stage": stage}
    command = client.create_experiment_with_tags(
        CreateExperimentWithTagsInput(
            request_id=f"mlflow-loopback-create-{experiment_id}",
            experiment_id=experiment_id,
            name=name,
            name_digest=_digest(name),
            artifact_location=f"s3://artifacts/{experiment_id}",
            packed_tags=json.dumps(tags, sort_keys=True, separators=(",", ":")).encode(),
            is_gateway=False,
            tags=tuple(
                ExperimentTag(
                    scope=_SCOPE,
                    experiment_id=experiment_id,
                    tag_key=key,
                    value=value,
                    value_digest=_digest(value),
                    value_hash=_hash64(value),
                )
                for key, value in sorted(tags.items())
            ),
        )
    )
    if not isinstance(command.outcome, CreateExperimentWithTagsExperimentCreated):
        raise RuntimeError(f"unexpected create outcome: {command.outcome.outcome}")
    return cast(TypedCommandResult[CreateExperimentWithTagsExperimentCreated], command)


def run(endpoint: str, credential_file: str, database: str) -> dict[str, object]:
    with SyncApplicationTransport.connect_uri(
        endpoint, _metadata(credential_file, database)
    ) as transport:
        client = RiffDBMLflowClient(transport, AttemptBudget(3))
        first = _create(client, 910_000_000_000_000_001, "Alpha Training", "prod")
        second = _create(client, 910_000_000_000_000_002, "Beta Training", "dev")
        fence = max(
            sequence
            for sequence in (first.commit_sequence, second.commit_sequence)
            if sequence is not None
        )
        options = QueryOptions(read_after_commit=fence)

        page_one = client.search_experiments(
            SearchExperimentsParams(
                scope=_SCOPE,
                lifecycle_stages=(LifecycleStage.ACTIVE,),
                order=ExperimentOrder.ID_ASC,
                limit=1,
            ),
            options,
        )
        if not isinstance(page_one.value, SearchExperimentsFound):
            raise RuntimeError("unfiltered experiment search returned an integrity outcome")
        if len(page_one.value.experiments) != 1 or page_one.next_cursor is None:
            raise RuntimeError("first experiment page did not expose one row and a cursor")

        page_two = client.search_experiments(
            SearchExperimentsParams(
                scope=_SCOPE,
                lifecycle_stages=(LifecycleStage.ACTIVE,),
                order=ExperimentOrder.ID_ASC,
                limit=2,
                after=page_one.next_cursor,
            ),
            options,
        )
        if not isinstance(page_two.value, SearchExperimentsFound):
            raise RuntimeError("continued experiment search returned an integrity outcome")
        if [row.experiment_id for row in page_two.value.experiments] != [910_000_000_000_000_002]:
            raise RuntimeError("continued experiment search returned the wrong row")

        folded = client.search_experiments_by_name_ilike(
            SearchExperimentsByNameILikeParams(
                scope=_SCOPE,
                pattern="%TRAINING",
                lifecycle_stages=(LifecycleStage.ACTIVE,),
                order=ExperimentOrder.NAME_ASC_ID_ASC,
                limit=4,
            ),
            options,
        )
        if not isinstance(folded.value, SearchExperimentsByNameILikeFound):
            raise RuntimeError("folded name search returned an integrity outcome")

        intersection = client.search_experiments_by_two_tags(
            SearchExperimentsByTwoTagsParams(
                scope=_SCOPE,
                first_tag_key="team",
                first_value_digest=_digest("research"),
                second_tag_key="stage",
                second_value_digest=_digest("prod"),
                lifecycle_stages=(LifecycleStage.ACTIVE,),
                order=ExperimentOrder.NAME_ASC_ID_ASC,
                limit=4,
            ),
            options,
        )
        if not isinstance(intersection.value, SearchExperimentsByTwoTagsFound):
            raise RuntimeError("tag intersection returned an integrity outcome")

        first_row = first.outcome.experiment
        updated_tags = {"owner": "alice", "stage": "prod", "team": "research"}
        tag_set = client.set_experiment_tag(
            SetExperimentTagInput(
                request_id="mlflow-loopback-set-owner",
                experiment_id=first_row.experiment_id,
                tag_key="owner",
                value="alice",
                value_digest=_digest("alice"),
                value_hash=_hash64("alice"),
                packed_tags=json.dumps(
                    updated_tags, sort_keys=True, separators=(",", ":")
                ).encode(),
                is_gateway=False,
                expected_revision=first_row.revision,
            )
        )
        if not isinstance(tag_set.outcome, SetExperimentTagExperimentTagSet):
            raise RuntimeError(f"unexpected tag-set outcome: {tag_set.outcome.outcome}")

        owner_search = client.search_experiments_by_tag(
            SearchExperimentsByTagParams(
                scope=_SCOPE,
                tag_key="owner",
                value_digest=_digest("alice"),
                lifecycle_stages=(LifecycleStage.ACTIVE,),
                order=ExperimentOrder.ID_ASC,
                limit=4,
            ),
            QueryOptions(read_after_commit=tag_set.commit_sequence),
        )
        if not isinstance(owner_search.value, SearchExperimentsByTagFound):
            raise RuntimeError("exact tag search returned an integrity outcome")
        if [row.experiment_id for row in owner_search.value.experiments] != [
            first_row.experiment_id
        ]:
            raise RuntimeError("exact tag search did not observe the committed tag")

        remaining_tags = {"owner": "alice", "team": "research"}
        tag_delete = client.delete_experiment_tag(
            DeleteExperimentTagInput(
                request_id="mlflow-loopback-delete-stage",
                experiment_id=first_row.experiment_id,
                tag_key="stage",
                packed_tags=json.dumps(
                    remaining_tags, sort_keys=True, separators=(",", ":")
                ).encode(),
                is_gateway=False,
                expected_revision=tag_set.outcome.experiment.revision,
            )
        )
        if not isinstance(tag_delete.outcome, DeleteExperimentTagExperimentTagDeleted):
            raise RuntimeError(f"unexpected tag-delete outcome: {tag_delete.outcome.outcome}")

        run_specs = (
            (first_row.experiment_id, f"{first_row.experiment_id:016x}0000000000000001", 100),
            (
                second.outcome.experiment.experiment_id,
                f"{second.outcome.experiment.experiment_id:016x}0000000000000002",
                300,
            ),
            (first_row.experiment_id, f"{first_row.experiment_id:016x}0000000000000003", 200),
        )
        created_run_rows = []
        created_run_sequences = []
        for ordinal, (experiment_id, run_id, start_time) in enumerate(run_specs, start=1):
            run_name = f"loopback-run-{ordinal}"
            packed = json.dumps(
                {"mlflow.runName": run_name, "team": "research"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            created_run = client.create_run_with_tags(
                CreateRunWithTagsInput(
                    request_id=f"mlflow-loopback-create-run-{ordinal}",
                    experiment_id=experiment_id,
                    run_id=run_id,
                    user_id="loopback",
                    start_time=start_time,
                    artifact_uri=f"s3://artifacts/{experiment_id}/{run_id}/artifacts",
                    run_name=run_name,
                    packed_tags=packed,
                    packed_params=b"{}",
                    packed_latest_metrics=b"{}",
                    tags=tuple(
                        _run_tag(experiment_id, run_id, key, value)
                        for key, value in (
                            ("mlflow.runName", run_name),
                            ("team", "research"),
                        )
                    ),
                )
            )
            if not isinstance(created_run.outcome, CreateRunWithTagsRunCreated):
                raise RuntimeError(f"unexpected run-create outcome: {created_run.outcome.outcome}")
            created_run_rows.append(created_run.outcome.run)
            created_run_sequences.append(created_run.commit_sequence)

        run_fence = max(sequence for sequence in created_run_sequences if sequence is not None)
        run_options = QueryOptions(read_after_commit=run_fence)
        run_page_one = client.search_runs_by_lifecycle(
            SearchRunsByLifecycleParams(
                experiment_ids=(first_row.experiment_id, second.outcome.experiment.experiment_id),
                lifecycle_stage=LifecycleStage.ACTIVE,
                limit=1,
            ),
            run_options,
        )
        if not isinstance(run_page_one.value, SearchRunsByLifecycleFound):
            raise RuntimeError("first run page returned an integrity outcome")
        if len(run_page_one.value.runs) != 1 or run_page_one.next_cursor is None:
            raise RuntimeError("first run page did not expose one row and a cursor")
        if run_page_one.value.runs[0].run_id != run_specs[1][1]:
            raise RuntimeError("first run page violated global start-time ordering")

        run_page_two = client.search_runs_by_lifecycle(
            SearchRunsByLifecycleParams(
                experiment_ids=(
                    second.outcome.experiment.experiment_id,
                    first_row.experiment_id,
                    first_row.experiment_id,
                ),
                lifecycle_stage=LifecycleStage.ACTIVE,
                limit=2,
                after=run_page_one.next_cursor,
            ),
            run_options,
        )
        if not isinstance(run_page_two.value, SearchRunsByLifecycleFound):
            raise RuntimeError("continued run page returned an integrity outcome")
        if [row.run_id for row in run_page_two.value.runs] != [run_specs[2][1], run_specs[0][1]]:
            raise RuntimeError("continued run page skipped or reordered rows")

        first_run = created_run_rows[0]
        renamed_tags = {"mlflow.runName": "renamed-loopback", "team": "research"}
        updated_run = client.update_run_info(
            UpdateRunInfoInput(
                request_id="mlflow-loopback-update-run",
                experiment_id=first_run.experiment_id,
                run_id=first_run.run_id,
                status=RunStatus.FINISHED,
                end_time=400,
                run_name="renamed-loopback",
                run_name_tag_value="renamed-loopback",
                run_name_digest=_digest("renamed-loopback"),
                packed_tags=json.dumps(
                    renamed_tags, sort_keys=True, separators=(",", ":")
                ).encode(),
                expected_revision=first_run.revision,
            )
        )
        if not isinstance(updated_run.outcome, UpdateRunInfoRunUpdated):
            raise RuntimeError(f"unexpected run-update outcome: {updated_run.outcome.outcome}")

        fetched_run = client.get_run(
            GetRunParams(experiment_id=first_run.experiment_id, run_id=first_run.run_id),
            QueryOptions(read_after_commit=updated_run.commit_sequence),
        )
        if not isinstance(fetched_run.value, GetRunFound):
            raise RuntimeError("updated run was not found")
        if fetched_run.value.run.run_name != "renamed-loopback":
            raise RuntimeError("updated run name was not durable")

        first_metric = _run_metric(
            first_run.experiment_id,
            first_run.run_id,
            "loss",
            1.5,
            500,
            1,
            promote_latest=False,
        )
        latest_metric = _run_metric(
            first_run.experiment_id,
            first_run.run_id,
            "loss",
            2.5,
            501,
            2,
            promote_latest=True,
        )
        batch_items: tuple[RunDatum, ...] = (
            first_metric,
            latest_metric,
            _run_param(first_run.experiment_id, first_run.run_id, "optimizer", "adam"),
            _run_tag(first_run.experiment_id, first_run.run_id, "stage", "fit"),
        )
        packed_params = b'{"optimizer":"adam"}'
        packed_latest_metrics = _encode_latest_metrics({"loss": (latest_metric.value_bits, 501, 2)})
        batch_tags = {"mlflow.runName": "renamed-loopback", "stage": "fit", "team": "research"}
        batch_input = LogBatchInput(
            request_id="mlflow-loopback-log-batch",
            experiment_id=first_run.experiment_id,
            run_id=first_run.run_id,
            expected_revision=updated_run.outcome.run.revision,
            packed_tags=json.dumps(batch_tags, sort_keys=True, separators=(",", ":")).encode(),
            packed_params=packed_params,
            packed_latest_metrics=packed_latest_metrics,
            items=batch_items,
        )
        try:
            logged = client.log_batch(batch_input)
        except OutcomeUnknown:
            logged = client.log_batch(batch_input)
        if not isinstance(logged.outcome, LogBatchBatchLogged):
            raise RuntimeError(f"unexpected log-batch outcome: {logged.outcome.outcome}")

        history_page_one = client.get_metric_history(
            GetMetricHistoryParams(
                experiment_id=first_run.experiment_id,
                item_key=_metric_item_key(first_run.run_id, "loss"),
                kind=RunDataKind.METRIC,
                value_digest=bytes(32),
                limit=1,
            ),
            QueryOptions(read_after_commit=logged.commit_sequence),
        )
        if not isinstance(history_page_one.value, GetMetricHistoryFound):
            raise RuntimeError("first metric-history page returned an integrity outcome")
        if len(history_page_one.value.metrics) != 1 or history_page_one.next_cursor is None:
            raise RuntimeError("first metric-history page did not expose a cursor")
        history_page_two = client.get_metric_history(
            GetMetricHistoryParams(
                experiment_id=first_run.experiment_id,
                item_key=_metric_item_key(first_run.run_id, "loss"),
                kind=RunDataKind.METRIC,
                value_digest=bytes(32),
                limit=100,
                after=history_page_one.next_cursor,
            ),
            QueryOptions(read_after_commit=logged.commit_sequence),
        )
        if not isinstance(history_page_two.value, GetMetricHistoryFound):
            raise RuntimeError("continued metric-history page returned an integrity outcome")
        if len(history_page_two.value.metrics) != 1 or history_page_two.next_cursor is not None:
            raise RuntimeError("continued metric-history page skipped or duplicated rows")

        deleted_run = client.delete_run(
            DeleteRunInput(
                request_id="mlflow-loopback-delete-run",
                experiment_id=first_run.experiment_id,
                run_id=first_run.run_id,
            )
        )
        if not isinstance(deleted_run.outcome, DeleteRunRunDeleted):
            raise RuntimeError(f"unexpected run-delete outcome: {deleted_run.outcome.outcome}")
        deleted_search = client.search_runs_by_lifecycle(
            SearchRunsByLifecycleParams(
                experiment_ids=(first_run.experiment_id,),
                lifecycle_stage=LifecycleStage.DELETED,
                limit=2,
            ),
            QueryOptions(read_after_commit=deleted_run.commit_sequence),
        )
        if not isinstance(deleted_search.value, SearchRunsByLifecycleFound):
            raise RuntimeError("deleted run search returned an integrity outcome")
        if [row.run_id for row in deleted_search.value.runs] != [first_run.run_id]:
            raise RuntimeError("deleted run search did not return the deleted run")

        restored_run = client.restore_run(
            RestoreRunInput(
                request_id="mlflow-loopback-restore-run",
                experiment_id=first_run.experiment_id,
                run_id=first_run.run_id,
            )
        )
        if not isinstance(restored_run.outcome, RestoreRunRunRestored):
            raise RuntimeError(f"unexpected run-restore outcome: {restored_run.outcome.outcome}")
        all_runs_page_one = client.search_runs(
            SearchRunsParams(
                experiment_ids=(first_run.experiment_id, second.outcome.experiment.experiment_id),
                limit=2,
            ),
            QueryOptions(read_after_commit=restored_run.commit_sequence),
        )
        if (
            not isinstance(all_runs_page_one.value, SearchRunsFound)
            or len(all_runs_page_one.value.runs) != 2
            or all_runs_page_one.next_cursor is None
        ):
            raise RuntimeError("first all-lifecycle run page was incomplete")
        all_runs_page_two = client.search_runs(
            SearchRunsParams(
                experiment_ids=(first_run.experiment_id, second.outcome.experiment.experiment_id),
                limit=2,
                after=all_runs_page_one.next_cursor,
            ),
            QueryOptions(read_after_commit=restored_run.commit_sequence),
        )
        if not isinstance(all_runs_page_two.value, SearchRunsFound):
            raise RuntimeError("continued all-lifecycle run page returned an integrity outcome")
        run_total = len(all_runs_page_one.value.runs) + len(all_runs_page_two.value.runs)
        if run_total != 3:
            raise RuntimeError("all-lifecycle run pages did not return all restored rows")

        _verify_tracking_store(transport, database)

        return {
            "folded_matches": len(folded.value.experiments),
            "intersection_matches": len(intersection.value.experiments),
            "metric_history": len(history_page_one.value.metrics)
            + len(history_page_two.value.metrics),
            "owner_matches": len(owner_search.value.experiments),
            "page_one": len(page_one.value.experiments),
            "page_two": len(page_two.value.experiments),
            "run_deleted_matches": len(deleted_search.value.runs),
            "run_page_one": len(run_page_one.value.runs),
            "run_page_two": len(run_page_two.value.runs),
            "run_total": run_total,
            "tag_revision": tag_delete.outcome.experiment.revision,
            "schema": "riffdb-mlflow-loopback/v1",
        }


def _verify_tracking_store(transport: SyncApplicationTransport, database: str) -> None:
    from riffdb_mlflow.store import RiffDBTrackingStore

    store = RiffDBTrackingStore(f"riffdb://{database}", "s3://mlflow-artifacts")
    store._transport = transport
    store._client = RiffDBMLflowClient(transport, AttemptBudget(3))

    default_exp = store.get_experiment("0")
    if default_exp.name != "Default" or default_exp.experiment_id != "0":
        raise RuntimeError("default experiment was not initialized by RiffDBTrackingStore")

    exp_id = store.create_experiment(
        "Store Plugin Experiment",
        tags=[
            MLflowExperimentTag("team", "ml-platform"),  # type: ignore[no-untyped-call]
            MLflowExperimentTag("env", "ci"),  # type: ignore[no-untyped-call]
        ],
    )
    by_name = store.get_experiment_by_name("Store Plugin Experiment")
    if by_name is None or by_name.experiment_id != exp_id:
        raise RuntimeError("store get_experiment_by_name did not resolve created experiment")

    store.rename_experiment(exp_id, "Store Plugin Renamed")
    store.set_experiment_tag(exp_id, MLflowExperimentTag("owner", "alice"))  # type: ignore[no-untyped-call]
    if store.get_experiment(exp_id).tags.get("owner") != "alice":
        raise RuntimeError("store set_experiment_tag did not persist")
    store.delete_experiment_tag(exp_id, "owner")
    if "owner" in store.get_experiment(exp_id).tags:
        raise RuntimeError("store delete_experiment_tag did not delete tag")

    store.delete_experiment(exp_id)
    if store.get_experiment(exp_id).lifecycle_stage != MLflowLifecycleStage.DELETED:
        raise RuntimeError("store delete_experiment did not mark experiment deleted")
    store.restore_experiment(exp_id)
    if store.get_experiment(exp_id).lifecycle_stage != MLflowLifecycleStage.ACTIVE:
        raise RuntimeError("store restore_experiment did not restore experiment")

    matched = store.search_experiments(filter_string="tag.team = 'ml-platform' AND tag.env = 'ci'")
    if [item.experiment_id for item in matched] != [exp_id]:
        raise RuntimeError("store search_experiments two-tag filter did not match")

    # Verify MLflow 3.15 UI experiment filters (IS NULL, IS NOT NULL, !=, combined with ILIKE)
    ui_default = store.search_experiments(
        filter_string="tags.`mlflow.experiment.isGateway` IS NULL"
    )
    if exp_id not in [item.experiment_id for item in ui_default]:
        raise RuntimeError("store search_experiments UI isGateway IS NULL filter did not match")

    ui_search = store.search_experiments(
        filter_string="name ILIKE '%Renamed%' AND tags.`mlflow.experiment.isGateway` IS NULL"
    )
    if [item.experiment_id for item in ui_search] != [exp_id]:
        raise RuntimeError("store search_experiments UI ILIKE + isGateway IS NULL did not match")

    ui_tag_search = store.search_experiments(
        filter_string="tags.team = 'ml-platform' AND tags.`mlflow.experiment.isGateway` IS NULL"
    )
    if [item.experiment_id for item in ui_tag_search] != [exp_id]:
        raise RuntimeError("store search_experiments UI tag + isGateway IS NULL did not match")

    tag_not_null = store.search_experiments(filter_string="tags.team IS NOT NULL")
    if exp_id not in [item.experiment_id for item in tag_not_null]:
        raise RuntimeError("store search_experiments tag IS NOT NULL did not match")

    tag_ne = store.search_experiments(filter_string="tags.team != 'research'")
    if exp_id not in [item.experiment_id for item in tag_ne]:
        raise RuntimeError("store search_experiments tag != did not match")

    run_one = store.create_run(
        exp_id,
        user_id="alice",
        start_time=1000,
        tags=[RunTag("team", "ml-platform")],  # type: ignore[no-untyped-call]
        run_name="store-run-1",
    )
    run_two = store.create_run(
        exp_id,
        user_id="bob",
        start_time=2000,
        tags=[],
        run_name="store-run-2",
    )
    updated_info = store.update_run_info(
        run_one.info.run_id, MLflowRunStatus.FINISHED, 3000, "store-run-1-finished"
    )
    if updated_info.run_name != "store-run-1-finished" or updated_info.end_time != 3000:
        raise RuntimeError("store update_run_info did not update run")

    nan_bits = 0x7FF8000000000042
    nan_val = struct.unpack(">d", struct.pack(">Q", nan_bits))[0]
    store.log_batch(
        run_one.info.run_id,
        metrics=[
            Metric("loss", 1.5, 100, 1),
            Metric("loss", 0.5, 200, 2),
            Metric("special", nan_val, 300, 1),
        ],
        params=[
            Param("lr", "0.01"),  # type: ignore[no-untyped-call]
            Param("opt", "adam"),  # type: ignore[no-untyped-call]
        ],
        tags=[RunTag("stage", "eval")],  # type: ignore[no-untyped-call]
    )
    fetched = store.get_run(run_one.info.run_id)
    if (
        fetched.data.params != {"lr": "0.01", "opt": "adam"}
        or fetched.data.tags.get("stage") != "eval"
        or fetched.data.metrics.get("loss") != 0.5
        or not math.isnan(fetched.data.metrics["special"])
    ):
        raise RuntimeError("store log_batch did not persist metrics, params, and tags")

    try:
        store.log_batch(
            run_one.info.run_id,
            metrics=[],
            params=[Param("lr", "0.02")],  # type: ignore[no-untyped-call]
            tags=[],
        )
    except MlflowException as exc:
        if "Changing param values is not allowed" not in str(exc):
            raise RuntimeError(f"unexpected param conflict message: {exc}") from exc
    else:
        raise RuntimeError("store log_batch did not reject conflicting param value")

    hist_one = store.get_metric_history(run_one.info.run_id, "loss", max_results=1)
    if len(hist_one) != 1 or hist_one.token is None:
        raise RuntimeError("store get_metric_history first page was invalid")
    hist_two = store.get_metric_history(
        run_one.info.run_id, "loss", max_results=10, page_token=hist_one.token
    )
    if len(hist_two) != 1 or hist_two.token is not None:
        raise RuntimeError("store get_metric_history second page was invalid")
    if [hist_one[0].value, hist_two[0].value] != [1.5, 0.5]:
        raise RuntimeError("store get_metric_history values were out of order")

    store.delete_run(run_two.info.run_id)
    active_runs = store.search_runs([exp_id], "", ViewType.ACTIVE_ONLY)  # type: ignore[no-untyped-call]
    if [item.info.run_id for item in active_runs] != [run_one.info.run_id]:
        raise RuntimeError("store search_runs ACTIVE_ONLY did not exclude deleted run")
    deleted_runs = store.search_runs([exp_id], "", ViewType.DELETED_ONLY)  # type: ignore[no-untyped-call]
    if [item.info.run_id for item in deleted_runs] != [run_two.info.run_id]:
        raise RuntimeError("store search_runs DELETED_ONLY did not return deleted run")
    store.restore_run(run_two.info.run_id)
    all_runs = store.search_runs([exp_id], "", ViewType.ALL)  # type: ignore[no-untyped-call]
    if [item.info.run_id for item in all_runs] != [run_two.info.run_id, run_one.info.run_id]:
        raise RuntimeError("store search_runs ALL did not return restored runs in start_time DESC")

    _verify_extended_tracking_store(store, exp_id, run_one.info.run_id)


def _verify_extended_tracking_store(store: object, exp_id: str, run_id: str) -> None:
    import asyncio

    from mlflow.entities import (
        AssessmentError,
        AssessmentSource,
        AssessmentSourceType,
        Dataset,
        DatasetInput,
        Expectation,
        Feedback,
        InputTag,
        LoggedModelInput,
        LoggedModelOutput,
        LoggedModelParameter,
        LoggedModelStatus,
        LoggedModelTag,
        Span,
        SpanType,
        TraceInfo,
        TraceLocation,
        TraceState,
    )
    from mlflow.entities.model_registry import PromptVersion
    from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
    from mlflow.tracing.constant import (
        AssessmentMetricDimensionKey,
        AssessmentMetricKey,
        SpanMetricKey,
        TraceMetricKey,
    )

    from riffdb_mlflow.store import RiffDBTrackingStore

    s = cast(RiffDBTrackingStore, store)

    # 1. Run lineage: log_inputs, log_outputs, record_logged_model
    ds = Dataset(
        name="eval-ds",
        digest="d1234567",
        source_type="local",
        source='{"uri":"file:///tmp/eval.parquet"}',
        schema='{"col":"text"}',
        profile='{"rows":100}',
    )
    ds_in = DatasetInput(
        dataset=ds,
        tags=[InputTag(key="mlflow.data.context", value="eval")],
    )
    s.log_inputs(
        run_id,
        datasets=[ds_in],
        models=[LoggedModelInput(model_id="m-input-1")],
    )
    s.log_outputs(
        run_id,
        models=[LoggedModelOutput(model_id="m-output-1", step=3)],
    )
    s.record_logged_model(run_id, {"artifact_path": "model", "flavors": {"python_function": {}}})
    run_with_lineage = s.get_run(run_id)
    if len(run_with_lineage.inputs.dataset_inputs) != 1:
        raise RuntimeError("store log_inputs did not persist dataset_inputs")
    if len(run_with_lineage.inputs.model_inputs) != 1:
        raise RuntimeError("store log_inputs did not persist model_inputs")
    if len(run_with_lineage.outputs.model_outputs) != 1:
        raise RuntimeError("store log_outputs did not persist model_outputs")
    ds_summaries = s._search_datasets([exp_id])
    if len(ds_summaries) != 1 or ds_summaries[0].name != "eval-ds":
        raise RuntimeError(f"store _search_datasets mismatch: {ds_summaries}")
    s.set_tag(run_id, RunTag("mlflow.parentRunId", "parent-run-1"))  # type: ignore[no-untyped-call]
    if (
        len(
            s._search_runs(
                [exp_id],
                "tags.`mlflow.parentRunId` = 'parent-run-1'",
                ViewType.ALL,
                100,
                None,
                None,
            )[0]
        )
        != 1
    ):
        raise RuntimeError("store search_runs by tag digest mismatch")
    if len(s._search_runs([exp_id], "params.lr = '0.01'", ViewType.ALL, 100, None, None)[0]) != 1:
        raise RuntimeError("store search_runs by param digest mismatch")
    if (
        len(
            s._search_runs(
                [exp_id], f"attributes.run_id = '{run_id}'", ViewType.ALL, 100, None, None
            )[0]
        )
        != 1
    ):
        raise RuntimeError("store search_runs by run_id mismatch")
    if (
        len(
            s._search_runs(
                [exp_id],
                "attributes.run_name = 'store-run-1-finished'",
                ViewType.ALL,
                100,
                None,
                None,
            )[0]
        )
        != 1
    ):
        raise RuntimeError("store search_runs by run_name mismatch")
    if (
        len(
            s._search_runs(
                [exp_id], "attributes.status = 'FINISHED'", ViewType.ALL, 100, None, None
            )[0]
        )
        != 1
    ):
        raise RuntimeError("store search_runs by status mismatch")

    # 2. Out-of-order span ingestion: ingest child and root spans BEFORE start_trace!
    ooo_trace_id = f"tr-{int(exp_id):016x}0000000000000001"

    def _make_span(
        trace_id: str,
        span_id_int: int,
        name: str,
        parent_id_int: int | None,
        start_ns: int,
        end_ns: int,
        span_type: str = SpanType.LLM,
        model_name: str = "gpt-4o",
        input_tokens: int = 100,
        output_tokens: int = 50,
        cost_usd: float = 0.0025,
    ) -> Span:
        attrs: dict[str, Any] = {
            "mlflow.traceRequestId": trace_id,
            "mlflow.spanType": span_type,
            "mlflow.spanInputs": {"prompt": "hello"},
            "mlflow.spanOutputs": {"response": "world"},
            "mlflow.chat.model": model_name,
            "mlflow.chat.tokenUsage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "mlflow.llm.cost": {
                "input_cost": cost_usd * 0.5,
                "output_cost": cost_usd * 0.5,
                "total_cost": cost_usd,
            },
        }
        return Span.from_dict(
            {
                "trace_id": trace_id,
                "span_id": f"{span_id_int:016x}",
                "parent_span_id": f"{parent_id_int:016x}" if parent_id_int is not None else None,
                "name": name,
                "start_time_unix_nano": start_ns,
                "end_time_unix_nano": end_ns,
                "status": {"code": "STATUS_CODE_OK", "message": ""},
                "attributes": attrs,
                "events": [],
            }
        )

    # Child span arrives first, before parent span and before start_trace!
    child_span = _make_span(
        ooo_trace_id,
        0x2222222222222222,
        "llm_call_child",
        0x1111111111111111,
        1_700_000_000_100_000_000,
        1_700_000_000_400_000_000,
        SpanType.LLM,
        "gpt-4o",
        120,
        80,
        0.004,
    )
    logged_child = s.log_spans(exp_id, [child_span])
    if len(logged_child) != 1:
        raise RuntimeError("out-of-order child log_spans before start_trace failed")

    # Root span arrives second (async)
    root_span = _make_span(
        ooo_trace_id,
        0x1111111111111111,
        "agent_root",
        None,
        1_700_000_000_000_000_000,
        1_700_000_000_500_000_000,
        SpanType.AGENT,
        "gpt-4o",
        40,
        20,
        0.001,
    )
    asyncio.run(s.log_spans_async(exp_id, [root_span]))

    # Now start_trace arrives after spans were already ingested!
    ti = TraceInfo(
        trace_id=ooo_trace_id,
        trace_location=TraceLocation.from_experiment_id(exp_id),
        request_time=1_700_000_000_000,
        state=TraceState.OK,
        request_preview='{"prompt":"hello"}',
        response_preview='{"response":"world"}',
        client_request_id="req-ooo-1",
        execution_duration=500,
        trace_metadata={"mlflow.traceInputs": '{"prompt":"hello"}'},
        tags={"env": "prod"},
    )
    started_ti = s.start_trace(ti)
    if started_ti.trace_id != ooo_trace_id or started_ti.state != TraceState.OK:
        raise RuntimeError("start_trace after out-of-order spans did not finalize TraceInfo")

    # Verify batch_get_traces reconstructs parent-before-child ordering and trace info
    full_traces = s.batch_get_traces([ooo_trace_id], location=exp_id)
    if len(full_traces) != 1 or len(full_traces[0].data.spans) != 2:
        raise RuntimeError("batch_get_traces did not return both out-of-order spans")
    if full_traces[0].data.spans[0].name != "agent_root":
        raise RuntimeError("batch_get_traces did not order root span before child span")

    # Verify trace tag and lineage operations
    s.set_trace_tag(ooo_trace_id, "custom_tag", "v1")
    if s.get_trace_info(ooo_trace_id).tags.get("custom_tag") != "v1":
        raise RuntimeError("set_trace_tag did not persist tag")
    s.delete_trace_tag(ooo_trace_id, "custom_tag")
    if "custom_tag" in s.get_trace_info(ooo_trace_id).tags:
        raise RuntimeError("delete_trace_tag did not remove tag")

    s.link_traces_to_run([ooo_trace_id], run_id)
    if s.get_trace_info(ooo_trace_id).trace_metadata.get("mlflow.sourceRun") != run_id:
        raise RuntimeError("link_traces_to_run did not set sourceRun")
    s.link_prompts_to_trace(
        [PromptVersion(name="qa-prompt", version=2, template="Answer: {{q}}")],  # type: ignore[abstract]
        ooo_trace_id,
    )
    # Re-link run for search_traces by run_id test
    s.link_traces_to_run([ooo_trace_id], run_id)

    # Verify get_trace and batch_get_trace_infos with a cold trace_experiment_map
    s._trace_experiment_map.clear()
    fetched_full_trace = s.get_trace(ooo_trace_id, allow_partial=True)
    if fetched_full_trace.info.trace_id != ooo_trace_id or len(fetched_full_trace.data.spans) != 2:
        raise RuntimeError("get_trace with cold cache did not return full trace")
    if [ti.trace_id for ti in s.batch_get_trace_infos([ooo_trace_id])] != [ooo_trace_id]:
        raise RuntimeError("batch_get_trace_infos did not return trace info")

    # Verify query_trace_metrics(ASSESSMENTS) succeeds with [] when no assessments exist yet
    empty_ass_metrics = s.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
        aggregations=[MetricAggregation(AggregationType.COUNT)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )
    if empty_ass_metrics != []:
        raise RuntimeError(
            f"Expected empty assessment metrics before assessments exist, got {empty_ass_metrics}"
        )

    # 3. Assessments (Feedback and Expectation)
    fb = s.create_assessment(
        Feedback(
            name="relevance",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="judge-1",
            ),
            trace_id=ooo_trace_id,
            value=0.9,
            rationale="Highly relevant",
            span_id=root_span.span_id,
            metadata={"judge_model": "gpt-4o"},
        )
    )
    if fb.assessment_id is None:
        raise RuntimeError("create_assessment did not assign assessment_id")
    fetched_fb = s.get_assessment(ooo_trace_id, fb.assessment_id)
    if fetched_fb.name != "relevance":
        raise RuntimeError("get_assessment did not return created Feedback")
    updated_fb = s.update_assessment(
        trace_id=ooo_trace_id,
        assessment_id=fb.assessment_id,
        name="relevance_v2",
        feedback=Feedback(
            name="relevance_v2",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="judge-1",
            ),
            value=1.0,
            error=AssessmentError(error_code="NONE", error_message="ok"),
        ),
        rationale="Updated rationale",
        metadata={"judge_model": "gpt-4o", "updated": "true"},
    )
    if updated_fb.name != "relevance_v2" or updated_fb.rationale != "Updated rationale":
        raise RuntimeError("update_assessment did not update Feedback")

    exp_ass = s.create_assessment(
        Expectation(
            name="expected_answer",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="annotator-1",
            ),
            trace_id=ooo_trace_id,
            value="world",
        )
    )
    if exp_ass.assessment_id is None:
        raise RuntimeError("create_assessment did not assign assessment_id to Expectation")
    ti_with_assessments = s.get_trace_info(ooo_trace_id)
    if len(ti_with_assessments.assessments) != 2:
        raise RuntimeError("get_trace_info did not attach assessments")

    # 4. Trace Metrics & Analytical Rollups (ADR-0315 projections)
    trace_count_points = s.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[
            MetricAggregation(AggregationType.COUNT),
        ],
        dimensions=["trace_status"],
    )
    if len(trace_count_points) != 1 or trace_count_points[0].values["COUNT"] != 1.0:
        raise RuntimeError(f"unexpected TRACES metric points: {trace_count_points}")

    token_points = s.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TOTAL_TOKENS,
        aggregations=[
            MetricAggregation(AggregationType.SUM),
            MetricAggregation(AggregationType.AVG),
        ],
    )
    if len(token_points) != 1 or token_points[0].values["SUM"] != 260.0:
        raise RuntimeError(f"unexpected TOTAL_TOKENS metric points: {token_points}")

    span_cost_points = s.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.SPANS,
        metric_name=SpanMetricKey.TOTAL_COST,
        aggregations=[
            MetricAggregation(AggregationType.SUM),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=95.0),
        ],
        dimensions=["span_model_name"],
    )
    if len(span_cost_points) != 1 or abs(span_cost_points[0].values["SUM"] - 0.005) > 1e-6:
        raise RuntimeError(f"unexpected SPANS cost metric points: {span_cost_points}")

    assessment_points = s.query_trace_metrics(
        experiment_ids=[exp_id],
        view_type=MetricViewType.ASSESSMENTS,
        metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
        aggregations=[
            MetricAggregation(AggregationType.AVG),
            MetricAggregation(AggregationType.MAX),
        ],
        dimensions=["assessment_name"],
    )
    if len(assessment_points) < 1:
        raise RuntimeError("ASSESSMENTS query_trace_metrics returned no points")

    # Verify ADR-0315 event projection SpanModelRollup and columnar aggregate SpanAnalytics
    event_rollups_llm = s.get_span_model_rollups(exp_id, "LLM", "gpt-4o")
    event_rollups_agent = s.get_span_model_rollups(exp_id, "AGENT", "gpt-4o")
    if len(event_rollups_llm) != 1 or len(event_rollups_agent) != 1:
        raise RuntimeError(
            f"expected 1 LLM and 1 AGENT SpanModelRollup row, "
            f"got {event_rollups_llm}, {event_rollups_agent}"
        )
    columnar_rollups = s.get_columnar_span_model_rollups(exp_id)
    if len(columnar_rollups) != 2:
        raise RuntimeError(f"expected 2 columnar SpanAnalytics rollup rows, got {columnar_rollups}")

    # Search traces with filter
    found_traces, _token = s.search_traces(
        experiment_ids=[exp_id],
        filter_string="status = 'OK' AND tag.env = 'prod'",
        run_id=run_id,
    )
    if [t.trace_id for t in found_traces] != [ooo_trace_id]:
        raise RuntimeError("search_traces did not match expected trace")

    s.unlink_traces_from_run([ooo_trace_id], run_id)
    s.delete_assessment(ooo_trace_id, exp_ass.assessment_id)
    deleted_count = s.delete_traces(exp_id, trace_ids=[ooo_trace_id])
    if deleted_count != 1:
        raise RuntimeError("delete_traces did not delete the trace")

    # 5. Logged Models (MLflow 3.x)
    created_model = s.create_logged_model(
        experiment_id=exp_id,
        name="rag-assistant",
        source_run_id=run_id,
        tags=[LoggedModelTag("framework", "langchain")],  # type: ignore[no-untyped-call]
        params=[LoggedModelParameter("temperature", "0.2")],  # type: ignore[no-untyped-call]
        model_type="agent",
    )
    fetched_model = s.get_logged_model(created_model.model_id)
    if fetched_model.name != "rag-assistant" or fetched_model.status != LoggedModelStatus.PENDING:
        raise RuntimeError("create_logged_model/get_logged_model failed")

    finalized_model = s.finalize_logged_model(created_model.model_id, LoggedModelStatus.READY)
    if finalized_model.status != LoggedModelStatus.READY:
        raise RuntimeError("finalize_logged_model did not update status to READY")

    s.set_logged_model_tags(
        created_model.model_id,
        [LoggedModelTag("stage", "staging")],  # type: ignore[no-untyped-call]
    )
    s.log_logged_model_params(
        created_model.model_id,
        [LoggedModelParameter("top_k", "5")],  # type: ignore[no-untyped-call]
    )
    s.delete_logged_model_tag(created_model.model_id, "framework")
    updated_model = s.get_logged_model(created_model.model_id)
    if updated_model.tags.get("stage") != "staging" or "framework" in updated_model.tags:
        raise RuntimeError("logged model tag operations failed")
    if updated_model.params.get("top_k") != "5":
        raise RuntimeError("log_logged_model_params failed")

    searched_models = s.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="name = 'rag-assistant' AND status = 'READY'",
    )
    if [m.model_id for m in searched_models] != [created_model.model_id]:
        raise RuntimeError("search_logged_models did not return matching model")

    s.delete_logged_model(created_model.model_id)

    # 6. MCP Server Registry Mixin (MLflow 3.15)
    from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPStatus, MCPTool

    mcp_server = s.create_mcp_server(
        name="io.riffdb/analytics-mcp",
        description="RiffDB Analytics MCP Server",
        created_by="conformance",
    )
    if mcp_server.name != "io.riffdb/analytics-mcp":
        raise RuntimeError("create_mcp_server failed")
    s.set_mcp_server_tag("io.riffdb/analytics-mcp", "tier", "production")
    sv = s.create_mcp_server_version(
        server_json={"name": "io.riffdb/analytics-mcp", "version": "1.0.0"},
        source="https://github.com/riffdb/analytics-mcp",
        status=MCPStatus.DRAFT,
        tools=[MCPTool(name="query_traces", description="Query trace analytics")],
        created_by="conformance",
    )
    if sv.version != "1.0.0" or sv.status != MCPStatus.DRAFT:
        raise RuntimeError("create_mcp_server_version failed")
    s.update_mcp_server_version(
        "io.riffdb/analytics-mcp",
        "1.0.0",
        status=MCPStatus.ACTIVE,
    )
    s.set_mcp_server_alias("io.riffdb/analytics-mcp", "stable", "1.0.0")
    by_alias = s.get_mcp_server_version_by_alias("io.riffdb/analytics-mcp", "stable")
    if by_alias.version != "1.0.0" or by_alias.status != MCPStatus.ACTIVE:
        raise RuntimeError("get_mcp_server_version_by_alias failed")
    ep = s.create_mcp_access_endpoint(
        server_name="io.riffdb/analytics-mcp",
        url="https://mcp.example.com/v1",
        transport_type=MCPRemoteTransportType.STREAMABLE_HTTP,
        server_alias="stable",
    )
    found_mcp = s.search_mcp_servers(filter_string="status = 'active'")
    if [item.name for item in found_mcp] != ["io.riffdb/analytics-mcp"]:
        raise RuntimeError("search_mcp_servers did not return active server")
    s.delete_mcp_access_endpoint("io.riffdb/analytics-mcp", ep.id)
    s.update_mcp_server_version(
        "io.riffdb/analytics-mcp",
        "1.0.0",
        status=MCPStatus.DEPRECATED,
    )
    s.delete_mcp_server_version("io.riffdb/analytics-mcp", "1.0.0")
    s.delete_mcp_server("io.riffdb/analytics-mcp")

    # 7. Model Registry Store, Prompt Registry, and Webhooks
    from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
    from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent, WebhookStatus

    from riffdb_mlflow.registry_store import RiffDBModelRegistryStore

    reg_store = RiffDBModelRegistryStore(
        f"riffdb://{s.database}", tracking_uri=f"riffdb://{s.database}"
    )
    reg_store._transport = s._transport
    reg_store._client = s._client

    rm = reg_store.create_registered_model(
        "fraud-detector",
        tags=[RegisteredModelTag("domain", "risk")],  # type: ignore[no-untyped-call]
        description="Fraud detection model",
    )
    if rm.name != "fraud-detector":
        raise RuntimeError("create_registered_model failed")
    mv = reg_store.create_model_version(
        "fraud-detector",
        source=f"s3://mlflow-artifacts/{exp_id}/{run_id}/artifacts/model",
        run_id=run_id,
        tags=[ModelVersionTag("validated", "true")],  # type: ignore[no-untyped-call]
        description="Initial version",
    )
    if int(mv.version) != 1:
        raise RuntimeError("create_model_version did not assign version 1")
    reg_store.transition_model_version_stage("fraud-detector", 1, "Production", False)
    reg_store.set_registered_model_alias("fraud-detector", "champion", 1)
    by_model_alias = reg_store.get_model_version_by_alias("fraud-detector", "champion")
    if int(by_model_alias.version) != 1 or by_model_alias.current_stage != "Production":
        raise RuntimeError("model registry alias/stage resolution failed")
    searched_rms = reg_store.search_registered_models(filter_string="name = 'fraud-detector'")
    if [item.name for item in searched_rms] != ["fraud-detector"]:
        raise RuntimeError("search_registered_models failed")

    # Prompt Registry via AbstractStore concrete methods
    prompt = reg_store.create_prompt(
        name="system-greeting",
        description="System greeting prompt",
        tags={"team": "assistant"},
    )
    if prompt.name != "system-greeting":
        raise RuntimeError("create_prompt failed")
    prompt_ver = reg_store.create_prompt_version(
        name="system-greeting",
        template="Hello {{user}}!",
        description="Initial prompt",
    )
    if int(prompt_ver.version) != 1 or prompt_ver.template != "Hello {{user}}!":
        raise RuntimeError("create_prompt_version failed")
    searched_prompts = reg_store.search_prompts()
    if [p.name for p in searched_prompts] != ["system-greeting"]:
        raise RuntimeError("search_prompts failed")

    # Registry Webhooks
    wh = reg_store.create_webhook(
        name="prod-hook",
        url="https://93.184.216.34/mlflow",
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
        description="Notify on new model version",
        status=WebhookStatus.ACTIVE,
    )
    listed_wh = reg_store.list_webhooks()
    if [w.webhook_id for w in listed_wh] != [wh.webhook_id]:
        raise RuntimeError("list_webhooks failed")
    reg_store.delete_webhook(wh.webhook_id)
    reg_store.delete_model_version("fraud-detector", 1)
    reg_store.delete_registered_model("fraud-detector")
    reg_store.delete_registered_model("system-greeting")

    # 8. AI Gateway (GatewayStoreMixin + config_resolver), Scorers, Online Scoring, and Guardrails
    os.environ.setdefault("MLFLOW_CRYPTO_KEK_PASSPHRASE", "riffdb-conformance-kek-passphrase")
    from mlflow.entities import (
        GatewayEndpointModelConfig,
        GatewayEndpointTag,
        RoutingStrategy,
    )
    from mlflow.entities.gateway_budget_policy import (
        BudgetAction,
        BudgetDuration,
        BudgetDurationUnit,
        BudgetTargetScope,
        BudgetUnit,
    )
    from mlflow.entities.gateway_endpoint import GatewayModelLinkageType
    from mlflow.entities.gateway_guardrail import GuardrailAction, GuardrailStage
    from mlflow.store.tracking.gateway.config_resolver import (
        get_endpoint_config,
        get_resource_endpoint_configs,
    )

    gw_secret = s.create_gateway_secret(
        secret_name="openai-prod-key",
        secret_value={"api_key": "sk-test-secret-value-123456"},
        provider="openai",
        auth_config={"auth_mode": "api_key"},
        created_by="conformance",
    )
    if gw_secret.secret_name != "openai-prod-key":
        raise RuntimeError("create_gateway_secret failed")

    gw_model_def = s.create_gateway_model_definition(
        name="gpt-4o-primary",
        secret_id=gw_secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
        created_by="conformance",
    )
    if gw_model_def.secret_name != "openai-prod-key":
        raise RuntimeError("create_gateway_model_definition failed")

    gw_endpoint = s.create_gateway_endpoint(
        name="chat-prod",
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=gw_model_def.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1.0,
            )
        ],
        routing_strategy=RoutingStrategy.REQUEST_BASED_TRAFFIC_SPLIT,
        experiment_id=exp_id,
        usage_tracking=True,
        created_by="conformance",
    )
    if gw_endpoint.name != "chat-prod" or len(gw_endpoint.model_mappings) != 1:
        raise RuntimeError("create_gateway_endpoint failed")

    s.set_gateway_endpoint_tag(gw_endpoint.endpoint_id, GatewayEndpointTag(key="env", value="prod"))
    resolved_cfg = get_endpoint_config("chat-prod", store=s)  # type: ignore[arg-type]
    if (
        resolved_cfg.endpoint_id != gw_endpoint.endpoint_id
        or len(resolved_cfg.models) != 1
        or resolved_cfg.models[0].secret_value != {"api_key": "sk-test-secret-value-123456"}
    ):
        raise RuntimeError("get_endpoint_config failed to decrypt gateway secret")

    bp = s.create_budget_policy(
        budget_unit=BudgetUnit.USD,
        budget_amount=100.0,
        duration=BudgetDuration(unit=BudgetDurationUnit.DAYS, value=30),
        target_scope=BudgetTargetScope.ENDPOINT,
        target_value=gw_endpoint.endpoint_id,
        budget_action=BudgetAction.REJECT,
        created_by="conformance",
    )
    if bp.budget_amount != 100.0 or len(s.list_budget_policies()) != 1:
        raise RuntimeError("create/list_budget_policies failed")
    s.delete_budget_policy(bp.budget_policy_id)

    scorer_ver = s.register_scorer(
        experiment_id=exp_id,
        name="safety-judge",
        serialized_scorer=json.dumps(
            {
                "name": "safety-judge",
                "instructions_judge_pydantic_data": {
                    "instructions": "Evaluate response safety.",
                    "model": "gateway:/chat-prod",
                },
            }
        ),
    )
    if scorer_ver.scorer_version != 1 or scorer_ver.scorer_id is None:
        raise RuntimeError("register_scorer failed")

    res_cfgs = get_resource_endpoint_configs("scorer", scorer_ver.scorer_id, store=s)  # type: ignore[arg-type]
    if len(res_cfgs) != 1 or res_cfgs[0].endpoint_id != gw_endpoint.endpoint_id:
        raise RuntimeError("get_resource_endpoint_configs failed for bound scorer")

    online_cfg = s.upsert_online_scoring_config(
        experiment_id=exp_id,
        scorer_name="safety-judge",
        sample_rate=0.5,
    )
    if online_cfg.sample_rate != 0.5 or len(s.get_active_online_scorers()) != 1:
        raise RuntimeError("online scoring config registration failed")

    guardrail = s.create_gateway_guardrail(
        name="safety-input-guard",
        scorer_id=scorer_ver.scorer_id,
        scorer_version=scorer_ver.scorer_version,
        stage=GuardrailStage.BEFORE,
        action=GuardrailAction.VALIDATION,
        created_by="conformance",
    )
    gr_cfg = s.add_guardrail_to_endpoint(
        endpoint_id=gw_endpoint.endpoint_id,
        guardrail_id=guardrail.guardrail_id,
        execution_order=1,
    )
    if (
        gr_cfg.guardrail is None
        or gr_cfg.guardrail.guardrail_id != guardrail.guardrail_id
        or len(s.list_endpoint_guardrail_configs(gw_endpoint.endpoint_id)) != 1
    ):
        raise RuntimeError("endpoint guardrail configuration failed")

    s.remove_guardrail_from_endpoint(gw_endpoint.endpoint_id, guardrail.guardrail_id)
    s.delete_gateway_guardrail(guardrail.guardrail_id)
    s.delete_scorer(exp_id, "safety-judge")
    s.delete_gateway_endpoint(gw_endpoint.endpoint_id)
    s.delete_gateway_model_definition(gw_model_def.model_definition_id)
    s.delete_gateway_secret(gw_secret.secret_id)

    # 9. Server Job Store (RiffDBJobStore)
    from mlflow.entities._job_status import JobStatus

    from riffdb_mlflow.job_store import RiffDBJobStore

    job_store = RiffDBJobStore(f"riffdb://{s.database}")
    job_store._transport = s._transport
    job_store._client = s._client

    job = job_store.create_job(
        job_name="online_scoring_eval",
        params=json.dumps({"experiment_id": exp_id, "batch_size": 10}),
        timeout=60.0,
    )
    if job.status != JobStatus.PENDING:
        raise RuntimeError("create_job did not initialize PENDING status")
    job_store.start_job(job.job_id)
    job_store.update_status_details(job.job_id, {"worker_pid": 12345})
    job_store.finish_job(job.job_id, json.dumps({"evaluated": 10}))
    finished_job = job_store.get_job(job.job_id)
    if (
        finished_job.status != JobStatus.SUCCEEDED
        or finished_job.parsed_result != {"evaluated": 10}
        or finished_job.status_details != {"worker_pid": 12345}
    ):
        raise RuntimeError("RiffDBJobStore lifecycle verification failed")
    listed_jobs = list(
        job_store.list_jobs(
            job_name="online_scoring_eval",
            statuses=[JobStatus.SUCCEEDED],
            params={"experiment_id": exp_id},
        )
    )
    if [j.job_id for j in listed_jobs] != [job.job_id]:
        raise RuntimeError("RiffDBJobStore list_jobs failed")
    deleted_jobs = job_store.delete_jobs(job_ids=[job.job_id])
    if deleted_jobs != [job.job_id]:
        raise RuntimeError("RiffDBJobStore delete_jobs failed")

    # 10. Evaluation Datasets, Label Schemas, Review Queues, and Issues
    from mlflow.entities.issue import IssueSeverity, IssueStatus
    from mlflow.genai.label_schemas import InputCategorical, LabelSchemaType

    eval_ds = s.create_dataset(
        name="conformance-eval-ds",
        tags={"purpose": "conformance"},
        experiment_ids=[exp_id],
    )
    upsert_res = s.upsert_dataset_records(
        dataset_id=eval_ds.dataset_id,
        records=[{"inputs": {"q": "hello"}, "outputs": {"a": "world"}}],
    )
    if (upsert_res.get("inserted"), upsert_res.get("updated")) != (1, 0):
        raise RuntimeError("upsert_dataset_records did not insert record")
    fetched_ds = s.get_dataset(eval_ds.dataset_id)
    loaded_records, _ = s._load_dataset_records(eval_ds.dataset_id)
    if fetched_ds.name != "conformance-eval-ds" or len(loaded_records) != 1:
        raise RuntimeError("get_dataset / _load_dataset_records verification failed")
    if [d.dataset_id for d in s.search_datasets(experiment_ids=[exp_id])] != [eval_ds.dataset_id]:
        raise RuntimeError("search_datasets verification failed")
    s.delete_dataset(eval_ds.dataset_id)

    default_schemas = s.list_label_schemas(experiment_id=exp_id)
    if not any(ls.name == "Feedback" for ls in default_schemas):
        raise RuntimeError("list_label_schemas did not seed default Feedback schema")
    custom_ls = s.create_label_schema(
        experiment_id=exp_id,
        name="tone",
        type=LabelSchemaType.FEEDBACK,
        instruction="Tone",
        input=InputCategorical(options=["polite", "neutral"]),
    )
    s.delete_label_schema(custom_ls.schema_id)

    user_rq = s.get_or_create_user_queue(exp_id, user="reviewer-1")
    if str(user_rq.queue_type) != "user":
        raise RuntimeError("get_or_create_user_queue did not return user queue")
    added_rq_items = s.add_items_to_review_queue(
        user_rq.queue_id,
        item_ids=[ooo_trace_id],
    )
    if len(added_rq_items) != 1:
        raise RuntimeError("add_items_to_review_queue failed")
    updated_rq_item = s.set_review_queue_item_status(
        user_rq.queue_id,
        item_id=ooo_trace_id,
        status="complete",
        completed_by="reviewer-1",
    )
    if str(updated_rq_item.status) != "complete":
        raise RuntimeError("set_review_queue_item_status failed")
    s.delete_review_queue(user_rq.queue_id)

    issue = s.create_issue(
        experiment_id=exp_id,
        name="Latency regression",
        description="High latency on retrieval span",
        severity=IssueSeverity.HIGH,
    )
    updated_issue = s.update_issue(issue.issue_id, status=IssueStatus.RESOLVED)
    if (
        updated_issue.status != IssueStatus.RESOLVED
        or len(s.search_issues(experiment_id=exp_id)) != 1
    ):
        raise RuntimeError("Issue CRUD verification failed")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python -m riffdb_mlflow ENDPOINT CREDENTIAL_FILE DATABASE")
    print(json.dumps(run(*sys.argv[1:4]), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
