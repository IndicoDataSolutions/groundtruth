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

logger = logging.getLogger(__name__)


def submit_documents(
    api_host: str,
    api_token_file: Path,
    workflow_id: int,
    document_files: Iterable[Path],
) -> Iterator[int]:
    """
    Submit a collection of documents to a workflow and yield their submission IDs.
    """
    client = IndicoClient(IndicoConfig(host=api_host, api_token_path=api_token_file))

    for document_file in document_files:
        (submission_id,) = client.call(
            WorkflowSubmission(workflow_id=workflow_id, files=[document_file])
        )
        yield submission_id


def retrieve_results(
    api_host: str,
    api_token_file: Path,
    results_folder: Path,
    submission_ids: Iterable[int],
) -> None:
    """
    Retrieve result JSONs for submissions into a folder.
    Results are named the original file name with a `.json` extension.
    """
    results_folder.mkdir(parents=True, exist_ok=True)
    client = IndicoClient(IndicoConfig(host=api_host, api_token_path=api_token_file))

    for submission_id in submission_ids:
        submission = client.call(GetSubmission(submission_id))
        submission_result = client.call(SubmissionResult(submission, wait=True))
        result = client.call(RetrieveStorageObject(submission_result.result))

        result_file = Path(submission.input_filename).with_suffix(".json")
        result_file = results_folder / result_file
        result_file.write_text(json.dumps(result))
