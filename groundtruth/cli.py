import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.set_defaults(command=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()
    add_submit_args(subparsers.add_parser("submit"))
    add_retrieve_args(subparsers.add_parser("retrieve"))
    add_extract_args(subparsers.add_parser("extract"))
    add_combine_args(subparsers.add_parser("combine"))
    add_analyze_args(subparsers.add_parser("analyze"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.command(args)


def add_submit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default="try.indico.io",
        help="Environment host name",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=Path("indico_api_token.txt"),
        help="API token path",
    )
    parser.add_argument(
        "--workflow-id",
        type=int,
        required=True,
        help="Workflow ID to submit into",
    )
    parser.add_argument(
        "--documents-folder",
        type=Path,
        default=Path("."),
        help="Folder of documents to submit",
    )
    parser.add_argument(
        "--submission-ids-file",
        type=Path,
        default=Path("submission_ids.csv"),
        help="Output CSV of submission IDs",
    )
    parser.set_defaults(command=submit)


def submit(args: argparse.Namespace) -> None:
    import polars
    import rich.progress

    from . import workflows
    from .utils import sanitize

    document_files = tuple(args.documents_folder.glob("*"))
    tracked_document_files = rich.progress.track(
        document_files, description="Submitting...", auto_refresh=False
    )
    submission_ids = workflows.submit_documents(
        api_host=args.host,
        api_token_file=args.token,
        workflow_id=args.workflow_id,
        document_files=tracked_document_files,
    )
    polars.DataFrame(
        {
            "submission_id": submission_id,
            "file_name": sanitize(document_file.name),
            "review_url": f"https://{args.host}/review/queues/{args.workflow_id}/submission/{submission_id}",  # noqa: E501
        }
        for submission_id, document_file in zip(submission_ids, document_files)
    ).write_csv(args.submission_ids_file)


def add_retrieve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default="try.indico.io",
        help="Environment host name",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=Path("indico_api_token.txt"),
        help="API token path",
    )
    parser.add_argument(
        "--submission-ids-file",
        type=Path,
        default=Path("submission_ids.csv"),
        help="CSV of submission IDs",
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        default=Path("."),
        help="Output folder of retrieved submission results",
    )
    parser.set_defaults(command=retrieve)


def retrieve(args: argparse.Namespace) -> None:
    import polars
    import rich.progress

    from . import workflows

    submission_ids = polars.read_csv(args.submission_ids_file)["submission_id"]
    tracked_submission_ids = rich.progress.track(
        submission_ids, description="Retrieving...", auto_refresh=False
    )
    workflows.retrieve_results(
        api_host=args.host,
        api_token_file=args.token,
        results_folder=args.results_folder,
        submission_ids=tracked_submission_ids,
    )


def add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-folder",
        type=Path,
        default=Path("."),
        help="Folder of submission results",
    )
    parser.add_argument(
        "--extractions-file",
        type=Path,
        default=Path("extractions.csv"),
        help="Output CSV of ground truths and predictions",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to extract ground truths and predictions from",
    )
    parser.add_argument(
        "fields",
        nargs="*",
        help=(
            "Fields to extract ground truths and predictions for "
            "(defaults to all fields)"
        ),
    )
    parser.set_defaults(command=extract)


def extract(args: argparse.Namespace) -> None:
    import rich.progress

    from . import extractions

    result_files = tuple(args.results_folder.glob("*.json"))

    if not args.fields:
        tracked_result_files = rich.progress.track(
            result_files, description="Discovering Fields...", auto_refresh=False
        )
        args.fields = extractions.all_fields_in_results(
            result_files=tracked_result_files, model=args.model
        )

    tracked_result_files = rich.progress.track(
        result_files, description="Extracting...", auto_refresh=False
    )
    extractions.results_to_csv(
        result_files=tracked_result_files,
        extractions_file=args.extractions_file,
        model=args.model,
        fields=args.fields,
    )


def add_combine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ground-truths-file",
        type=Path,
        default=Path("ground_truth_extractions.csv"),
        help="CSV of ground truths",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=Path("prediction_extractions.csv"),
        help="CSV of predictions",
    )
    parser.add_argument(
        "--combined-file",
        type=Path,
        default=Path("combined_extractions.csv"),
        help="Output CSV of ground truths and predictions",
    )
    parser.set_defaults(command=combine)


def combine(args: argparse.Namespace) -> None:
    from . import extractions

    extractions.combine_extractions(
        ground_truths_file=args.ground_truths_file,
        predictions_file=args.predictions_file,
        combined_file=args.combined_file,
    )


def add_analyze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extractions-file",
        type=Path,
        default=Path("extractions.csv"),
        help="CSV of ground truths and predictions",
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=Path("analysis.csv"),
        help="Output CSV of performance metrics",
    )
    parser.add_argument(
        "thresholds",
        type=float,
        nargs="+",
        help="Confidence thresholds to calculate accuracy, volume, and STP for",
    )
    parser.set_defaults(command=analyze)


def analyze(args: argparse.Namespace) -> None:
    from . import metrics

    metrics.extractions_to_analysis(
        extractions_file=args.extractions_file,
        analysis_file=args.analysis_file,
        thresholds=args.thresholds,
    )
