import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

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
        (submission_id,) = client.call(
            WorkflowSubmission(workflow_id=workflow_id, files=[document_file])
        )
        yield submission_id


def retrieve_results(  # type: ignore[no-any-unimported]
    config: IndicoConfig,
    results_folder: Path,
    submission_ids: Iterable[int],
) -> None:
    """
    Retrieve result JSONs for submissions into a folder.
    Results are named the original file name with a `.json` extension.
    """
    results_folder.mkdir(parents=True, exist_ok=True)
    client = IndicoClient(config)

    for submission_id in submission_ids:
        submission = client.call(GetSubmission(submission_id))
        submission_result = client.call(SubmissionResult(submission, wait=True))
        result = client.call(RetrieveStorageObject(submission_result.result))

        sanitized_file_name = sanitize(submission.input_filename)
        result_file = Path(sanitized_file_name).with_suffix(".json")
        result_file = results_folder / result_file
        result_file.write_text(json.dumps(result))
