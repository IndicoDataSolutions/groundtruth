import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import rich
from indico import IndicoClient, IndicoConfig
from indico.queries import (
    GetSubmission,
    RetrieveStorageObject,
    SubmissionResult,
    WorkflowSubmission,
)

from .utils import sanitize

logger = logging.getLogger(__name__)


def submit_documents(  # type: ignore[no-any-unimported]
    config: IndicoConfig,
    workflow_id: int,
    document_files: Iterable[Path],
) -> Iterator[int]:
    """
    Submit a collection of documents to a workflow and yield their submission IDs.
    """
    client = IndicoClient(config)

    for document_file in document_files:
        if document_file.is_file():
            (submission_id,) = client.call(
                WorkflowSubmission(workflow_id=workflow_id, files=[document_file])
            )
            yield submission_id

        elif document_file.is_dir():
            bundle_files = list(
                filter(
                    lambda file: file.is_file() and not file.name.startswith("."),
                    document_file.glob("*"),
                )
            )
            (submission_id,) = client.call(
                WorkflowSubmission(
                    workflow_id=workflow_id, files=bundle_files, bundle=True
                )
            )
            yield submission_id


def retrieve_results(  # type: ignore[no-any-unimported]
    config: IndicoConfig,
    results_folder: Path,
    submissions: Iterable[tuple[int, str]],
) -> None:
    """
    Retrieve result JSONs for submissions into a folder.
    Results are named the original file name with a `.json` extension.
    """
    results_folder.mkdir(parents=True, exist_ok=True)
    client = IndicoClient(config)

    for submission_id, file_name in submissions:
        submission = client.call(GetSubmission(submission_id))

        if submission.files_deleted:
            rich.print(
                "[yellow]"
                f"Submission {submission_id} {file_name!r} has been deleted. "
                "Skipping."
                "[/]"
            )
            continue

        submission_result = client.call(SubmissionResult(submission, wait=True))
        result = client.call(RetrieveStorageObject(submission_result.result))

        sanitized_file_name = sanitize(file_name)
        result_file = Path(sanitized_file_name + ".json")
        result_file = results_folder / result_file
        result_file.write_text(json.dumps(result))
