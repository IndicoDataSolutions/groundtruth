from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars
from Levenshtein import distance, ratio

from .utils import zip_match_longest

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """
    Plain Old Data class for a ground truth/prediction sample.
    """

    file_name: str
    field: str
    ground_truth_id: int | None
    prediction_id: int | None
    ground_truth: str | None
    prediction: str | None
    confidence: float | None
    edit_distance: int
    similarity: float
    accurate: bool

    @property
    def true_positive(self) -> bool:
        return self.prediction is not None and self.accurate

    @property
    def false_negative(self) -> bool:
        return self.prediction is None and not self.accurate

    @property
    def false_positive(self) -> bool:
        return self.prediction is not None and not self.accurate

    @property
    def true_negative(self) -> bool:
        return self.prediction is None and self.accurate

    @staticmethod
    def from_values(
        file_name: str,
        field: str,
        ground_truth_id: int | None,
        prediction_id: int | None,
        ground_truth: str | None,
        prediction: str | None,
        confidence: float | None,
    ) -> "Sample":
        return Sample(
            file_name=file_name,
            field=field,
            ground_truth_id=ground_truth_id,
            prediction_id=prediction_id,
            ground_truth=ground_truth,
            prediction=prediction,
            confidence=confidence,
            edit_distance=distance(normalize(ground_truth), normalize(prediction)),
            similarity=ratio(normalize(ground_truth), normalize(prediction)),
            accurate=normalize(ground_truth) == normalize(prediction),
        )


def normalize(value: str | None) -> str:
    """
    Normalize a ground truth or prediction value for the purposes of calculating edit
    distance and similarity.
    """
    value = value or ""
    value = value.casefold()
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def results_to_csv(result_files: Iterable[Path], samples_file: Path) -> None:
    """
    Convert result JSONs to a CSV of ground truth/prediction samples.
    """
    results_and_names = read_results(result_files)
    samples = samples_for_results(results_and_names)
    write_samples(samples, samples_file)


def read_results(result_files: Iterable[Path]) -> Iterator[tuple[str, Any]]:
    """
    Yield file names and parsed results from JSONs.
    """
    for result_file in result_files:
        result = json.loads(result_file.read_text())
        yield result_file.name, result


def samples_for_results(
    results_and_names: Iterable[tuple[str, Any]]
) -> Iterator[Sample]:
    """
    Yield ground truth/prediction samples for all models and fields.
    Results without auto-review will be skipped. Results without HITL-review will lack
    ground truth.
    """
    for result_file_name, result in results_and_names:
        for model in result["results"]["document"]["results"]:
            yield from samples_for_result(result_file_name, result, model)


def samples_for_result(
    result_file_name: str, result: Any, model: str
) -> Iterator[Sample]:
    try:
        submission_id = result["submission_id"]
        document_dict = result["results"]["document"]
        model_dict = document_dict["results"][model]
        post_reviews = model_dict["post_reviews"]
        auto_review = post_reviews[-1]
        assert auto_review is not None  # Rejected in review
    except (AssertionError, IndexError, KeyError, TypeError):
        logger.warning(
            f"Result '{result_file_name}' does not contain an auto review for model "
            f"'{model}'. Skipping."
        )
        return

    try:
        hitl_review = post_reviews[-2]
        assert hitl_review is not None  # Rejected in review
    except (AssertionError, IndexError):
        logger.warning(
            f"Result '{result_file_name}' does not contain a HITL review for model "
            f"'{model}'. Missing ground truth."
        )
        hitl_review = []

    ground_truths_by_field: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_field: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for ground_truth in hitl_review:
        ground_truths_by_field[ground_truth["label"]].append(ground_truth)

    for prediction in auto_review:
        predictions_by_field[prediction["label"]].append(prediction)

    fields = set(ground_truths_by_field.keys()) | set(predictions_by_field.keys())

    for field in sorted(fields):
        for ground_truth_dict, prediction_dict in zip_match_longest(
            left=ground_truths_by_field[field],
            right=predictions_by_field[field],
            left_key=lambda value: value["text"],
            right_key=lambda value: value["text"],
        ):
            ground_truth = ground_truth_dict["text"] if ground_truth_dict else None
            prediction = prediction_dict["text"] if prediction_dict else None
            confidence = (
                prediction_dict["confidence"][field] if prediction_dict else None
            )
            yield Sample.from_values(
                file_name=result_file_name,
                field=field,
                ground_truth_id=submission_id,
                prediction_id=submission_id,
                ground_truth=ground_truth,
                prediction=prediction,
                confidence=confidence,
            )


def write_samples(samples: Iterable[Sample], samples_file: Path) -> None:
    """
    Write samples to a CSV file.
    """
    dataframe = polars.DataFrame(map(dataclasses.asdict, samples))
    dataframe.write_csv(samples_file, null_value="__novalue__")


def combine_samples(
    ground_truths_file: Path,
    predictions_file: Path,
    combined_file: Path,
) -> None:
    """
    Combine the ground truths and predictions from two sample CSV files.
    Ground truths and predictions will be matched by file name, then field.
    The combined output will only contain samples for files and fields present in
    the predictions CSV file.
    """
    ground_truth_samples = read_samples(ground_truths_file)
    prediction_samples = read_samples(predictions_file)
    combined_samples = combine_samples_by_file_name(
        ground_truth_samples, prediction_samples
    )
    write_samples(combined_samples, combined_file)


def read_samples(samples_file: Path) -> Iterator[Sample]:
    """
    Read samples from a CSV file.
    """
    samples = polars.read_csv(samples_file, null_values=["__novalue__"])

    for sample in samples.iter_rows(named=True):
        yield Sample(**sample)


def combine_samples_by_file_name(
    ground_truth_samples: Iterable[Sample],
    prediction_samples: Iterable[Sample],
) -> Iterator[Sample]:
    """
    Combine ground truth and prediction samples by file name and field.
    """
    ground_truths_by_file_name: defaultdict[str, list[Sample]] = defaultdict(list)
    predictions_by_file_name: defaultdict[str, list[Sample]] = defaultdict(list)

    for sample in ground_truth_samples:
        ground_truths_by_file_name[sample.file_name].append(sample)

    for sample in prediction_samples:
        predictions_by_file_name[sample.file_name].append(sample)

    gt_file_names = ground_truths_by_file_name.keys()
    pred_file_names = predictions_by_file_name.keys()

    for file_name in set(gt_file_names) | set(pred_file_names):
        yield from combine_samples_by_field(
            ground_truths_by_file_name[file_name], predictions_by_file_name[file_name]
        )


def combine_samples_by_field(
    ground_truth_samples: Iterable[Sample],
    prediction_samples: Iterable[Sample],
) -> Iterator[Sample]:
    """
    Combine ground truth and prediction samples by field.
    """
    ground_truths_by_field: defaultdict[str, list[Sample]] = defaultdict(list)
    predictions_by_field: defaultdict[str, list[Sample]] = defaultdict(list)

    for sample in ground_truth_samples:
        ground_truths_by_field[sample.field].append(sample)

    for sample in prediction_samples:
        predictions_by_field[sample.field].append(sample)

    for field in set(ground_truths_by_field.keys()) | set(predictions_by_field.keys()):
        for ground_truth_extraction, prediction_extraction in zip_match_longest(
            left=ground_truths_by_field[field],
            right=predictions_by_field[field],
            left_key=lambda value: value.ground_truth,
            right_key=lambda value: value.prediction,
        ):
            if ground_truth_extraction and prediction_extraction:
                yield Sample.from_values(
                    file_name=prediction_extraction.file_name,
                    field=field,
                    ground_truth_id=ground_truth_extraction.ground_truth_id,
                    prediction_id=prediction_extraction.prediction_id,
                    ground_truth=ground_truth_extraction.ground_truth,
                    prediction=prediction_extraction.prediction,
                    confidence=prediction_extraction.confidence,
                )
            elif ground_truth_extraction:
                yield Sample.from_values(
                    file_name=ground_truth_extraction.file_name,
                    field=field,
                    ground_truth_id=ground_truth_extraction.ground_truth_id,
                    prediction_id=None,
                    ground_truth=ground_truth_extraction.ground_truth,
                    prediction=None,
                    confidence=None,
                )
            elif prediction_extraction:
                yield Sample.from_values(
                    file_name=prediction_extraction.file_name,
                    field=field,
                    ground_truth_id=None,
                    prediction_id=prediction_extraction.prediction_id,
                    ground_truth=None,
                    prediction=prediction_extraction.prediction,
                    confidence=prediction_extraction.confidence,
                )
            else:
                logger.error(
                    "Matched ground truth and prediction pair were both `None` for "
                    f"field '{field}'. This shouldn't be able to happen."
                )
