"""Disposable real-driver smoke runner used by ``riffdb dev --run``."""

from __future__ import annotations

import json
import sys
from hashlib import sha256

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
    CreateExperimentWithTagsOutcome,
    DeleteExperimentTagExperimentTagDeleted,
    DeleteExperimentTagInput,
    ExperimentOrder,
    ExperimentTag,
    LifecycleStage,
    RiffDBMLflowClient,
    SearchExperimentsByNameILikeFound,
    SearchExperimentsByNameILikeParams,
    SearchExperimentsByTagFound,
    SearchExperimentsByTagParams,
    SearchExperimentsByTwoTagsFound,
    SearchExperimentsByTwoTagsParams,
    SearchExperimentsFound,
    SearchExperimentsParams,
    SetExperimentTagExperimentTagSet,
    SetExperimentTagInput,
)

_SCOPE = "global"


def _digest(value: str) -> bytes:
    return sha256(value.encode("utf-8")).digest()


def _metadata(credential_file: str, database: str) -> CallMetadata:
    credential = BearerCredential.from_protected_file(credential_file)
    return CallMetadata.authenticated(credential).with_database(DatabaseAlias(database))


def _create(
    client: RiffDBMLflowClient, experiment_id: int, name: str, stage: str
) -> TypedCommandResult[CreateExperimentWithTagsOutcome]:
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
    return command


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

        if not isinstance(first.outcome, CreateExperimentWithTagsExperimentCreated):
            raise RuntimeError("first experiment create lost its declared outcome")
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

        return {
            "folded_matches": len(folded.value.experiments),
            "intersection_matches": len(intersection.value.experiments),
            "owner_matches": len(owner_search.value.experiments),
            "page_one": len(page_one.value.experiments),
            "page_two": len(page_two.value.experiments),
            "tag_revision": tag_delete.outcome.experiment.revision,
            "schema": "riffdb-mlflow-loopback/v1",
        }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python -m riffdb_mlflow ENDPOINT CREDENTIAL_FILE DATABASE")
    print(json.dumps(run(*sys.argv[1:4]), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
