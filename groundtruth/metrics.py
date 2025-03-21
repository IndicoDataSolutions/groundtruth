from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars

from .samples import Sample, read_samples
from .utils import group_by_attr


@dataclass
class Metric:
    """
    Plain Old Data class for a single ground truth metric.
    """

    name: str
    field: str
    threshold: float
    value: float | None

    @staticmethod
    def tabulate(
        metrics: Iterable["Metric"],
    ) -> Iterator[dict[str, str | float | None]]:
        """
        Tabulate an iterable of Metrics into a dictionary structure suitable for a
        DataFrame CSV export.

        E.g.
        >>> Metric.tabulate(
            Metric(name="A", field="B", threshold=0.5, value=0.4),
            Metric(name="A", field="B", threshold=0.6, value=0.5),
            Metric(name="A", field="B", threshold=0.7, value=0.6),
            Metric(name="A", field="C", threshold=0.5, value=0.8),
            Metric(name="A", field="C", threshold=0.6, value=0.9),
            Metric(name="A", field="C", threshold=0.7, value=1.0),
        )
        (
            {"Metric": "A", "Field": "B", "50": 0.4, "60": 0.5, "70": 0.6},
            {"Metric": "A", "Field": "C", "50": 0.8, "60": 0.9, "70": 1.0},
        )
        """
        for name, metrics_for_name in group_by_attr(metrics, "name"):
            for field, metrics_for_field in group_by_attr(metrics_for_name, "field"):
                yield {
                    "Metric": name,
                    "Field": field,
                    **{
                        f"{metric.threshold:g}": metric.value
                        for metric in metrics_for_field
                    },
                }


@dataclass
class ConfusionMatrix:
    """
    Confusion Matrix class to which multiple Extractions can be added.
    """

    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0

    def add(self, sample: Sample) -> None:
        self.true_positive += sample.true_positive
        self.false_negative += sample.false_negative
        self.false_positive += sample.false_positive
        self.true_negative += sample.true_negative

    @property
    def true(self) -> int:
        return self.true_positive + self.true_negative

    @property
    def false(self) -> int:
        return self.false_positive + self.false_negative

    @property
    def positive(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def negative(self) -> int:
        return self.false_negative + self.true_negative

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_negative
            + self.false_positive
            + self.true_negative
        )

    @property
    def accuracy(self) -> float | None:
        try:
            return self.true / self.total
        except ZeroDivisionError:
            return None

    @property
    def precision(self) -> float | None:
        try:
            return self.true_positive / self.positive
        except ZeroDivisionError:
            return None

    @property
    def recall(self) -> float | None:
        try:
            return self.true_positive / (self.true_positive + self.false_negative)
        except ZeroDivisionError:
            return None

    @property
    def f1(self) -> float | None:
        try:
            return (2 * self.true_positive) / (2 * self.true_positive + self.false)
        except ZeroDivisionError:
            return None


def analyze_samples(
    samples_file: Path,
    analysis_file: Path,
    thresholds: Sequence[float],
) -> None:
    """
    Calculate accuracy, volume, and STP performance metrics at specified thresholds and
    write to an analysis CSV.
    """
    samples = tuple(read_samples(samples_file))
    metrics = sample_metrics_at_thresholds(samples, thresholds)
    write_metrics(metrics, analysis_file)


def sample_metrics_at_thresholds(
    samples: Sequence[Sample],
    thresholds: Sequence[float],
) -> Iterator[Metric]:
    """
    Yields accuracy, volume, and STP performance metrics from samples at specified
    thresholds.
    """
    for field, field_samples in group_by_attr(samples, "field"):
        yield from accuracy_metrics(field, field_samples, thresholds)
        yield from volume_metrics(field, field_samples, thresholds)

    yield from stp_metrics(samples, thresholds)


def accuracy_metrics(
    field: str,
    samples: Sequence[Sample],
    thresholds: Iterable[float],
) -> Iterator[Metric]:
    """
    Yield accuracy metrics from samples at specified thresholds.
    """
    for threshold in thresholds:
        matrix = ConfusionMatrix()

        for sample in filter(confidence_above_threshold(threshold), samples):
            matrix.add(sample)

        yield Metric("Accuracy", field, threshold, matrix.accuracy)


def volume_metrics(
    field: str,
    samples: Sequence[Sample],
    thresholds: Iterable[float],
) -> Iterator[Metric]:
    """
    Yield volume metrics from samples at specified thresholds.
    """
    for threshold in thresholds:
        straight_through_processed = tuple(
            filter(confidence_above_threshold(threshold), samples)
        )

        try:
            volume = len(straight_through_processed) / len(samples)
        except ZeroDivisionError:
            volume = None

        yield Metric("Volume", field, threshold, volume)


def stp_metrics(
    samples: Iterable[Sample],
    thresholds: Iterable[float],
) -> Iterator[Metric]:
    """
    Yield STP metrics from samples at specified thresholds.
    """
    submission_groups = group_by_attr(samples, "file_name")

    for threshold in thresholds:
        straight_through_processed = tuple(
            file_name
            for file_name, samples in submission_groups
            if all(map(confidence_above_threshold(threshold), samples))
        )

        try:
            stp_rate = len(straight_through_processed) / len(submission_groups)
        except ZeroDivisionError:
            stp_rate = None

        yield Metric("STP", "All Fields", threshold, stp_rate)


def confidence_above_threshold(
    threshold: float,
) -> Callable[[Sample], bool]:
    """
    Produce a filter function for samples that have a confidence greater than or
    equal to a threshold. Extractions without a confidence also pass.
    """

    def threshold_filter(sample: Sample) -> bool:
        return sample.confidence is None or sample.confidence >= threshold

    return threshold_filter


def write_metrics(metrics: Iterable[Metric], analysis_file: Path) -> None:
    """
    Tabulate metrics and write to a CSV.
    """
    dataframe = polars.DataFrame(Metric.tabulate(metrics))
    dataframe.write_csv(analysis_file)
