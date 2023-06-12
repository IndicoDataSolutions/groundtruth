from __future__ import annotations

from groundtruth.extractions import Extraction
from groundtruth.metrics import (
    ConfusionMatrix,
    Metric,
    accuracy_metrics,
    stp_metrics,
    volume_metrics,
)


def default_extraction(
    file_name: str = "",
    field: str = "",
    ground_truth_id: int = 0,
    prediction_id: int = 0,
    ground_truth: str | None = None,
    prediction: str | None = None,
    confidence: float | None = None,
    edit_distance: int = 0,
    similarity: float = 0,
    accurate: bool = False,
) -> Extraction:
    return Extraction(
        file_name=file_name,
        field=field,
        ground_truth_id=ground_truth_id,
        prediction_id=prediction_id,
        ground_truth=ground_truth,
        prediction=prediction,
        confidence=confidence,
        edit_distance=edit_distance,
        similarity=similarity,
        accurate=accurate,
    )


class TestMetric:
    @staticmethod
    def test_tabulate() -> None:
        metrics = [
            Metric(name="A", field="B", threshold=0.1, value=0.4),
            Metric(name="A", field="B", threshold=0.2, value=0.5),
            Metric(name="A", field="B", threshold=0.3, value=0.6),
            Metric(name="A", field="C", threshold=0.5, value=0.8),
            Metric(name="A", field="C", threshold=0.6, value=0.9),
            Metric(name="A", field="C", threshold=0.7, value=1.0),
            Metric(name="D", field="E", threshold=0.4, value=0.1),
            Metric(name="D", field="E", threshold=0.5, value=0.2),
            Metric(name="D", field="E", threshold=0.6, value=0.3),
            Metric(name="D", field="F", threshold=0.8, value=0.5),
            Metric(name="D", field="F", threshold=0.9, value=0.6),
            Metric(name="D", field="F", threshold=1.0, value=0.7),
            Metric(name="G", field="H", threshold=0.9, value=0.1),
            Metric(name="G", field="H", threshold=0.99, value=0.2),
            Metric(name="G", field="H", threshold=0.99999, value=0.3),
        ]

        assert list(Metric.tabulate(metrics)) == [
            {"Metric": "A", "Field": "B", "0.1": 0.4, "0.2": 0.5, "0.3": 0.6},
            {"Metric": "A", "Field": "C", "0.5": 0.8, "0.6": 0.9, "0.7": 1.0},
            {"Metric": "D", "Field": "E", "0.4": 0.1, "0.5": 0.2, "0.6": 0.3},
            {"Metric": "D", "Field": "F", "0.8": 0.5, "0.9": 0.6, "1": 0.7},
            {"Metric": "G", "Field": "H", "0.9": 0.1, "0.99": 0.2, "0.99999": 0.3},
        ]


class TestConfusionMatrix:
    @staticmethod
    def test_add_true_positive() -> None:
        cm = ConfusionMatrix()
        cm.add(
            default_extraction(
                ground_truth="True Positive",
                prediction="True Positive",
                accurate=True,
            )
        )

        assert cm.true_positive == 1
        assert cm.false_negative == 0
        assert cm.false_positive == 0
        assert cm.true_negative == 0

    @staticmethod
    def test_add_false_negative() -> None:
        cm = ConfusionMatrix()
        cm.add(
            default_extraction(
                ground_truth="False Negative",
                prediction=None,
                accurate=False,
            )
        )

        assert cm.true_positive == 0
        assert cm.false_negative == 1
        assert cm.false_positive == 0
        assert cm.true_negative == 0

    @staticmethod
    def test_add_false_positive() -> None:
        cm = ConfusionMatrix()
        cm.add(
            default_extraction(
                ground_truth=None,
                prediction="False Positive",
                accurate=False,
            )
        )
        cm.add(
            default_extraction(
                ground_truth="Ground Truth",
                prediction="False Positive",
                accurate=False,
            )
        )

        assert cm.true_positive == 0
        assert cm.false_negative == 0
        assert cm.false_positive == 2
        assert cm.true_negative == 0

    @staticmethod
    def test_add_true_negative() -> None:
        cm = ConfusionMatrix()
        cm.add(
            default_extraction(
                ground_truth=None,
                prediction=None,
                accurate=True,
            )
        )

        assert cm.true_positive == 0
        assert cm.false_negative == 0
        assert cm.false_positive == 0
        assert cm.true_negative == 1

    @staticmethod
    def test_calculations() -> None:
        cm = ConfusionMatrix(
            true_positive=3,
            false_negative=5,
            false_positive=7,
            true_negative=11,
        )

        assert cm.true == 14
        assert cm.false == 12
        assert cm.positive == 10
        assert cm.negative == 16
        assert cm.total == 26
        assert cm.accuracy == 14 / 26
        assert cm.precision == 3 / 10
        assert cm.recall == 3 / 8
        assert cm.f1 == 6 / 18

    @staticmethod
    def test_undefined_calculations() -> None:
        cm = ConfusionMatrix(
            true_positive=0,
            false_negative=0,
            false_positive=0,
            true_negative=0,
        )

        assert cm.accuracy is None
        assert cm.precision is None
        assert cm.recall is None
        assert cm.f1 is None


class TestCalculations:
    @staticmethod
    def test_accuracy() -> None:
        field = "Field A"
        extractions = [
            default_extraction(field=field, confidence=0.5, accurate=True),
            default_extraction(field=field, confidence=0.5, accurate=True),
            default_extraction(field=field, confidence=0.9, accurate=True),
            default_extraction(field=field, confidence=0.9, accurate=False),
        ]
        thresholds = [0.3, 0.8, 1.0]
        metrics = list(accuracy_metrics(field, extractions, thresholds))

        assert metrics == [
            Metric(name="Accuracy", field=field, threshold=0.3, value=0.75),
            Metric(name="Accuracy", field=field, threshold=0.8, value=0.5),
            Metric(name="Accuracy", field=field, threshold=1.0, value=None),
        ]

    @staticmethod
    def test_accuracy_no_preds() -> None:
        field = "Field A"
        extractions: list[Extraction] = []
        thresholds = [1.0]
        metrics = list(accuracy_metrics(field, extractions, thresholds))

        assert metrics == [
            Metric(name="Accuracy", field=field, threshold=1.0, value=None),
        ]

    @staticmethod
    def test_volume() -> None:
        field = "Field A"
        extractions = [
            default_extraction(field=field, confidence=0.2),
            default_extraction(field=field, confidence=0.7),
            default_extraction(field=field, confidence=0.8),
            default_extraction(field=field, confidence=0.9),
        ]
        thresholds = [0.3, 0.8, 1.0]
        metrics = list(volume_metrics(field, extractions, thresholds))

        assert metrics == [
            Metric(name="Volume", field=field, threshold=0.3, value=0.75),
            Metric(name="Volume", field=field, threshold=0.8, value=0.5),
            Metric(name="Volume", field=field, threshold=1.0, value=0.0),
        ]

    @staticmethod
    def test_volume_no_preds() -> None:
        field = "Field A"
        extractions: list[Extraction] = []
        thresholds = [1.0]
        metrics = list(volume_metrics(field, extractions, thresholds))

        assert metrics == [
            Metric(name="Volume", field=field, threshold=1.0, value=None),
        ]

    @staticmethod
    def test_stp() -> None:
        extractions = [
            default_extraction(file_name="alpha.json", confidence=0.2),
            default_extraction(file_name="alpha.json", confidence=0.7),
            default_extraction(file_name="bravo.json", confidence=0.7),
            default_extraction(file_name="charlie.json", confidence=0.7),
            default_extraction(file_name="charlie.json", confidence=0.9),
            default_extraction(file_name="delta.json", confidence=0.9),
        ]
        thresholds = [0.3, 0.8, 1.0]
        metrics = list(stp_metrics(extractions, thresholds))

        assert metrics == [
            Metric(name="STP", field="All Fields", threshold=0.3, value=0.75),
            Metric(name="STP", field="All Fields", threshold=0.8, value=0.25),
            Metric(name="STP", field="All Fields", threshold=1.0, value=0.0),
        ]

    @staticmethod
    def test_stp_no_preds() -> None:
        extractions: list[Extraction] = []
        thresholds = [1.0]
        metrics = list(stp_metrics(extractions, thresholds))

        assert metrics == [
            Metric(name="STP", field="All Fields", threshold=1.0, value=None),
        ]
