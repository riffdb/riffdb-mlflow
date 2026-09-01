"""Disposable real-driver smoke runner used by ``riffdb dev --run``."""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from typing import cast

from riffdb_application import (
    AttemptBudget,
    BearerCredential,
    CallMetadata,
    DatabaseAlias,
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
    GetRunFound,
    GetRunParams,
    LifecycleStage,
    RestoreRunInput,
    RestoreRunRunRestored,
    RiffDBMLflowClient,
    RunStatus,
    RunTag,
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


def _metadata(credential_file: str, database: str) -> CallMetadata:
    credential = BearerCredential.from_protected_file(credential_file)
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
            tags=tuple(
                ExperimentTag(
                    scope=_SCOPE,
                    experiment_id=experiment_id,
                    tag_key=key,
                    value=value,
                    value_digest=_digest(value),
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
                limit=4,
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
                packed_tags=json.dumps(
                    updated_tags, sort_keys=True, separators=(",", ":")
                ).encode(),
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
                    tags=tuple(
                        RunTag(
                            experiment_id=experiment_id,
                            run_id=run_id,
                            tag_key=key,
                            value=value,
                            value_digest=_digest(value),
                        )
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
                limit=4,
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
                limit=4,
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
        all_runs = client.search_runs(
            SearchRunsParams(
                experiment_ids=(first_run.experiment_id, second.outcome.experiment.experiment_id),
                limit=4,
            ),
            QueryOptions(read_after_commit=restored_run.commit_sequence),
        )
        if not isinstance(all_runs.value, SearchRunsFound) or len(all_runs.value.runs) != 3:
            raise RuntimeError("all-lifecycle run search did not return all restored rows")

        return {
            "folded_matches": len(folded.value.experiments),
            "intersection_matches": len(intersection.value.experiments),
            "owner_matches": len(owner_search.value.experiments),
            "page_one": len(page_one.value.experiments),
            "page_two": len(page_two.value.experiments),
            "run_deleted_matches": len(deleted_search.value.runs),
            "run_page_one": len(run_page_one.value.runs),
            "run_page_two": len(run_page_two.value.runs),
            "run_total": len(all_runs.value.runs),
            "tag_revision": tag_delete.outcome.experiment.revision,
            "schema": "riffdb-mlflow-loopback/v1",
        }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python -m riffdb_mlflow ENDPOINT CREDENTIAL_FILE DATABASE")
    print(json.dumps(run(*sys.argv[1:4]), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
