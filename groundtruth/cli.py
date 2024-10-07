from pathlib import Path
from typing import Annotated

import arguably
from indico import IndicoConfig

required = arguably.arg.required()


@arguably.command
def submit(
    *,
    host: Annotated[str, required],
    token: Annotated[Path, required],
    workflow_id: Annotated[int, required],
    documents_folder: Path = Path("documents"),
    submission_ids_file: Path = Path("submission_ids.csv"),
) -> None:
    import polars
    import rich.progress

    from . import workflows
    from .utils import sanitize

    config = IndicoConfig(host=host, api_token_path=token)

    document_files = tuple(
        filter(
            lambda file: not file.name.startswith("."),
            documents_folder.glob("*"),
        )
    )
    tracked_document_files = rich.progress.track(
        document_files, description="Submitting...", auto_refresh=False
    )
    submission_ids = workflows.submit_documents(
        config=config,
        workflow_id=workflow_id,
        document_files=tracked_document_files,
    )
    polars.DataFrame(
        {
            "submission_id": submission_id,
            "file_name": sanitize(document_file.name),
            "review_url": f"https://{config.host}/review/queues/{workflow_id}/submission/{submission_id}",  # noqa: E501
        }
        for submission_id, document_file in zip(submission_ids, document_files)
    ).write_csv(submission_ids_file)


@arguably.command
def retrieve(
    *,
    host: Annotated[str, required],
    token: Annotated[Path, required],
    submission_ids_file: Path = Path("submission_ids.csv"),
    results_folder: Path = Path("results"),
) -> None:
    import polars
    import rich.progress

    from . import workflows

    config = IndicoConfig(host=host, api_token_path=token)

    submission_ids = polars.read_csv(submission_ids_file)["submission_id"]
    tracked_submission_ids = rich.progress.track(
        submission_ids, description="Retrieving...", auto_refresh=False
    )
    workflows.retrieve_results(
        config=config,
        results_folder=results_folder,
        submission_ids=tracked_submission_ids,
    )


@arguably.command
def extract(
    *,
    results_folder: Path = Path("results"),
    samples_file: Path = Path("samples.csv"),
) -> None:
    pass
    import rich.progress

    from . import samples

    result_files = tuple(results_folder.glob("*.json"))
    tracked_result_files = rich.progress.track(
        result_files, description="Extracting...", auto_refresh=False
    )
    samples.results_to_csv(
        result_files=tracked_result_files,
        samples_file=samples_file,
    )


@arguably.command
def combine(
    *,
    ground_truths_file: Path = Path("ground_truths.csv"),
    predictions_file: Path = Path("predictions.csv"),
    combined_file: Path = Path("combined.csv"),
) -> None:
    from . import samples

    samples.combine_samples(
        ground_truths_file=ground_truths_file,
        predictions_file=predictions_file,
        combined_file=combined_file,
    )


@arguably.command
def analyze(
    *thresholds: Annotated[float, required],
    samples_file: Path = Path("samples.csv"),
    analysis_file: Path = Path("analysis.csv"),
) -> None:
    pass
    from . import metrics

    metrics.analyze_samples(
        samples_file=samples_file,
        analysis_file=analysis_file,
        thresholds=thresholds,
    )


def main() -> None:
    arguably.run()
