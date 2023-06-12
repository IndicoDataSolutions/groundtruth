from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars
from Levenshtein import distance, ratio

logger = logging.getLogger(__name__)


@dataclass
class Extraction:
    """
    Plain Old Data class for a ground truth/prediction pair.
    """

    file_name: str
    field: str
    ground_truth_id: int
    prediction_id: int
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
        ground_truth_id: int,
        prediction_id: int,
        ground_truth: str | None,
        prediction: str | None,
        confidence: float | None,
    ) -> "Extraction":
        return Extraction(
            file_name=file_name,
            field=field,
            ground_truth_id=ground_truth_id,
            prediction_id=prediction_id,
            ground_truth=ground_truth,
            prediction=prediction,
            confidence=confidence,
            edit_distance=distance(normalize(ground_truth), normalize(prediction)),
            similarity=ratio(normalize(ground_truth), normalize(prediction)),
            accurate=ground_truth == prediction,
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


def all_fields_in_results(
    result_files: Iterable[Path],
    model: str,
) -> set[str]:
    all_fields = set()

    for result_file_name, result in read_results(result_files):
        try:
            document_dict = result["results"]["document"]
            model_dict = document_dict["results"][model]
            post_reviews = model_dict["post_reviews"]
            auto_review = post_reviews[-1]
        except (IndexError, KeyError, TypeError):
            logger.warning(
                f"Result '{result_file_name}' does not contain an auto review for "
                f"model '{model}'."
            )
            raise

        try:
            hitl_review = post_reviews[-2]
        except IndexError:
            logger.warning(
                f"Result '{result_file_name}' does not contain a HITL review for "
                f"model '{model}'."
            )
            hitl_review = []

        for prediction in auto_review + hitl_review:
            all_fields.add(prediction["label"])

    return all_fields


def results_to_csv(
    result_files: Iterable[Path],
    extractions_file: Path,
    model: str,
    fields: Sequence[str],
) -> None:
    """
    Convert result JSONs to a CSV of ground truth/prediction samples.
    """
    results_and_names = read_results(result_files)
    extractions = extractions_for_results(results_and_names, model, fields)
    write_extractions(extractions, extractions_file)


def read_results(result_files: Iterable[Path]) -> Iterator[tuple[str, Any]]:
    """
    Yield file names and parsed results from JSONs.
    """
    for result_file in result_files:
        result = json.loads(result_file.read_text())
        yield result_file.name, result


def extractions_for_results(
    results_and_names: Iterable[tuple[str, Any]], model: str, fields: Sequence[str]
) -> Iterator[Extraction]:
    """
    Yield ground truth/prediction samples for specified model fields.
    Results without auto-review will be skipped. Results without HITL-review will be
    missing ground truth.
    """
    for result_file_name, result in results_and_names:
        yield from extractions_for_result(result_file_name, result, model, fields)


def extractions_for_result(
    result_file_name: str, result: Any, model: str, fields: Sequence[str]
) -> Iterator[Extraction]:
    try:
        submission_id = result["submission_id"]
        document_dict = result["results"]["document"]
        model_dict = document_dict["results"][model]
        post_reviews = model_dict["post_reviews"]
        auto_review = post_reviews[-1]
    except (IndexError, KeyError, TypeError):
        logger.warning(
            f"Result '{result_file_name}' does not contain an auto review for model "
            f"'{model}'. Skipping."
        )
        return

    try:
        hitl_review = post_reviews[-2]
    except IndexError:
        logger.warning(
            f"Result '{result_file_name}' does not contain a HITL review for model "
            f"'{model}'. Missing ground truth."
        )
        hitl_review = []

    ground_truths_by_field: defaultdict[str, list[Any]] = defaultdict(list)
    predictions_by_field: defaultdict[str, list[Any]] = defaultdict(list)

    for ground_truth in hitl_review:
        ground_truths_by_field[ground_truth["label"]].append(ground_truth)

    for prediction in auto_review:
        predictions_by_field[prediction["label"]].append(prediction)

    for field in fields:
        ground_truths = ground_truths_by_field[field]
        predictions = predictions_by_field[field]

        if len(ground_truths) > 1 or len(predictions) > 1:
            logger.warning(
                f"Result '{result_file_name}' contains multiple ground "
                f"truths/predictions for field '{field}'. "
                "Discarding all but the first."
            )

        ground_truth_dict = ground_truths[0] if ground_truths else None
        prediction_dict = predictions[0] if predictions else None
        ground_truth = ground_truth_dict["text"] if ground_truth_dict else None
        prediction = prediction_dict["text"] if prediction_dict else None
        confidence = prediction_dict["confidence"][field] if prediction_dict else None

        yield Extraction.from_values(
            file_name=result_file_name,
            field=field,
            ground_truth_id=submission_id,
            prediction_id=submission_id,
            ground_truth=ground_truth,
            prediction=prediction,
            confidence=confidence,
        )


def write_extractions(
    extractions: Iterable[Extraction], extractions_file: Path
) -> None:
    """
    Write extractions to a CSV file.
    """
    dataframe = polars.DataFrame(map(dataclasses.asdict, extractions))
    dataframe.write_csv(extractions_file, null_value="__novalue__")


def combine_extractions(
    ground_truths_file: Path,
    predictions_file: Path,
    combined_file: Path,
) -> None:
    """
    Combine the ground truths and predictions from two extraction CSV files.
    Ground truths and predictions will be matched by file name, then field.
    The combined output will only contain extractions for files and fields present in
    the predictions CSV file.
    """
    ground_truth_extractions = read_extractions(ground_truths_file)
    prediction_extractions = read_extractions(predictions_file)
    combined_extractions = combine_extractions_by_file_name(
        ground_truth_extractions, prediction_extractions
    )
    write_extractions(combined_extractions, combined_file)


def read_extractions(extractions_file: Path) -> Iterator[Extraction]:
    """
    Read extractions from a CSV file.
    """
    extractions = polars.read_csv(extractions_file, null_values=["__novalue__"])

    for extraction in extractions.iter_rows(named=True):
        yield Extraction(**extraction)


def combine_extractions_by_file_name(
    ground_truth_extractions: Iterable[Extraction],
    prediction_extractions: Iterable[Extraction],
) -> Iterator[Extraction]:
    """
    Combine ground truth and prediction extractions by file name and field.
    The combination will only contain extractions for files and fields present in
    the predictions.
    """
    ground_truths_by_file_name: defaultdict[str, list[Extraction]] = defaultdict(list)
    predictions_by_file_name: defaultdict[str, list[Extraction]] = defaultdict(list)

    for extraction in ground_truth_extractions:
        ground_truths_by_file_name[extraction.file_name].append(extraction)

    for extraction in prediction_extractions:
        predictions_by_file_name[extraction.file_name].append(extraction)

    for file_name in predictions_by_file_name.keys():
        yield from combine_extractions_by_field(
            ground_truths_by_file_name[file_name], predictions_by_file_name[file_name]
        )


def combine_extractions_by_field(
    ground_truth_extractions: Iterable[Extraction],
    prediction_extractions: Iterable[Extraction],
) -> Iterator[Extraction]:
    """
    Combine ground truth and prediction extractions by field.
    The combination will only contain extractions for fields present in predictions.
    """
    ground_truths_by_field: defaultdict[str, list[Extraction]] = defaultdict(list)
    predictions_by_field: defaultdict[str, list[Extraction]] = defaultdict(list)

    for extraction in ground_truth_extractions:
        ground_truths_by_field[extraction.field].append(extraction)

    for extraction in prediction_extractions:
        predictions_by_field[extraction.field].append(extraction)

    for field in predictions_by_field.keys():
        ground_truth_extractions = ground_truths_by_field[field]
        prediction_extractions = predictions_by_field[field]

        ground_truth_extraction = ground_truth_extractions[0]
        prediction_extraction = prediction_extractions[0]

        if len(ground_truth_extractions) > 1 or len(prediction_extractions) > 1:
            logger.warning(
                f"Result '{prediction_extraction.file_name}' contains multiple ground "
                f"truths/predictions for field '{field}'. "
                "All but the first were discarded."
            )

        yield Extraction.from_values(
            file_name=prediction_extraction.file_name,
            field=field,
            ground_truth_id=ground_truth_extraction.ground_truth_id,
            prediction_id=prediction_extraction.prediction_id,
            ground_truth=ground_truth_extraction.ground_truth,
            prediction=prediction_extraction.prediction,
            confidence=prediction_extraction.confidence,
        )
