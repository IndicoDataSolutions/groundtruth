#!/usr/bin/env python

"""
Unattended document submission and retrieval script. Supports:

- N documents at a time
- Automatic retries with exponential backoff
- Processing timeouts

Usage: `submit_and_retrieve.py --help`
"""

import argparse
import asyncio
import functools
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiometer
from indico import IndicoClient, IndicoConfig
from indico.errors import IndicoRequestError, IndicoTimeoutError
from indico.queries import (
    GetSubmission,
    RetrieveStorageObject,
    SubmissionResult,
    WorkflowSubmission,
)
from indico_toolkit import retry  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="try.indico.io",
        help="Environment hostname",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=Path("indico_api_token.txt"),
        help="Token path",
    )
    parser.add_argument(
        "--workflow-id",
        type=int,
        required=True,
        help="Workflow ID",
    )
    parser.add_argument(
        "--documents-folder",
        type=Path,
        default=Path("."),
        help="Documents folder",
    )
    parser.add_argument(
        "--n-at-a-time",
        type=int,
        default=8,
        help="Number of documents to submit at a time",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Number of times to retry failed requests",
    )

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    documents = args.documents_folder.glob("*")
    await aiometer.run_on_each(
        functools.partial(submit_and_retrieve, args),
        documents,
        max_at_once=args.n_at_a_time,
        max_per_second=0.2,
    )


async def submit_and_retrieve(args: argparse.Namespace, document_path: Path) -> None:
    result_path = document_path.with_suffix(".json")

    # Skip result files and documents that have already been retrieved.
    if document_path.suffix.lower() == ".json" or result_path.exists():
        print("Skipping", document_path)
        return

    make_retry = retry(
        exceptions=(IndicoRequestError, IndicoTimeoutError),
        num_retries=args.n_at_a_time,
        wait=60,
    )
    client = await asyncio.to_thread(
        IndicoClient, IndicoConfig(host=args.host, api_token_path=args.token)
    )
    call: Callable[..., Any] = functools.partial(
        asyncio.to_thread, make_retry(client.call)
    )

    try:
        print("Submitting", document_path)

        (submission_id,) = await call(
            WorkflowSubmission(workflow_id=args.workflow_id, files=[document_path])
        )

        print("Queued", document_path, "with submission ID", submission_id)

        submission = await call(GetSubmission(submission_id))

        while submission.status not in ("PENDING_REVIEW", "COMPLETE", "FAILED"):
            await asyncio.sleep(random.randint(30, 60))
            submission = await call(GetSubmission(submission_id))

        if submission.status == "FAILED":
            print("Failed", document_path, "with submission ID", submission_id)
            return

        submission_result = await call(SubmissionResult(submission, wait=True))
        result = await call(RetrieveStorageObject(submission_result.result))
        result_path.write_text(json.dumps(result))

        print("Retrieved", result_path, "from submission ID", submission_id)
    except (IndicoRequestError, IndicoTimeoutError):
        print("Max retries exceeded for", document_path)
        return


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
