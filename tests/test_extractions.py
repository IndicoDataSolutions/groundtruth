from __future__ import annotations

from groundtruth.extractions import (
    Extraction,
    combine_extractions_by_file_name,
    extractions_for_results,
)


def default_extraction(
    file_name: str = "",
    field: str = "",
    ground_truth_id: int | None = None,
    prediction_id: int | None = None,
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


class TestExtraction:
    @staticmethod
    def test_true_positive() -> None:
        extraction = default_extraction(
            ground_truth="True Positive",
            prediction="True Positive",
            accurate=True,
        )

        assert extraction.true_positive is True
        assert extraction.false_negative is False
        assert extraction.false_positive is False
        assert extraction.true_negative is False

    @staticmethod
    def test_false_negative() -> None:
        extraction = default_extraction(
            ground_truth="False Negative",
            prediction=None,
            accurate=False,
        )

        assert extraction.true_positive is False
        assert extraction.false_negative is True
        assert extraction.false_positive is False
        assert extraction.true_negative is False

    @staticmethod
    def test_false_positive() -> None:
        extraction = default_extraction(
            ground_truth=None,
            prediction="False Positive",
            accurate=False,
        )

        assert extraction.true_positive is False
        assert extraction.false_negative is False
        assert extraction.false_positive is True
        assert extraction.true_negative is False

    @staticmethod
    def test_true_negative() -> None:
        extraction = default_extraction(
            ground_truth=None,
            prediction=None,
            accurate=True,
        )

        assert extraction.true_positive is False
        assert extraction.false_negative is False
        assert extraction.false_positive is False
        assert extraction.true_negative is True

    @staticmethod
    def test_from_values() -> None:
        extraction = Extraction.from_values(
            file_name="Test File",
            field="Test Field",
            ground_truth_id=123,
            prediction_id=456,
            ground_truth="Ground Truth",
            prediction="Prediction",
            confidence=0.9,
        )

        assert extraction.file_name == "Test File"
        assert extraction.field == "Test Field"
        assert extraction.ground_truth_id == 123
        assert extraction.prediction_id == 456
        assert extraction.ground_truth == "Ground Truth"
        assert extraction.prediction == "Prediction"
        assert extraction.confidence == 0.9
        assert extraction.edit_distance == 10
        assert extraction.similarity == 0.2727272727272727
        assert extraction.accurate is False


def test_extract() -> None:
    model_name = "Test Model"
    fields = ["Alpha", "Bravo"]
    ground_truths = [
        {"label": "Alpha", "text": "Tres"},
        {"label": "Bravo", "text": "Duo"},
        {"label": "Bravo", "text": "Unus"},
        {"label": "Charlie", "text": "Nehil"},
        {"label": "Charlie", "text": "Nehil"},
        {"label": "Charlie", "text": "Nehil"},
    ]
    predictions = [
        {"label": "Alpha", "text": "Tres", "confidence": {"Alpha": 1.0}},
        {"label": "Bravo", "text": "Unus", "confidence": {"Bravo": 0.8}},
        {"label": "Bravo", "text": "Duodenum", "confidence": {"Bravo": 0.9}},
        {"label": "Charlie", "text": "Nehil", "confidence": {"Charlie": 0.7}},
        {"label": "Charlie", "text": "Nehil", "confidence": {"Charlie": 0.6}},
        {"label": "Charlie", "text": "Nehil", "confidence": {"Charlie": 0.5}},
    ]
    results_and_names = [
        (
            "abc.json",
            {
                "submission_id": 123,
                "results": {
                    "document": {
                        "results": {
                            model_name: {
                                "post_reviews": [
                                    ground_truths,
                                    predictions,
                                ]
                            }
                        }
                    }
                },
            },
        )
    ]

    alpha_extraction, bravo_extraction, bravo_2_extraction = list(
        extractions_for_results(results_and_names, model_name, fields)
    )

    assert alpha_extraction.true_positive is True
    assert bravo_extraction.false_positive is True
    assert bravo_2_extraction.true_positive is True


def test_combine() -> None:
    ground_truth_extractions = [
        default_extraction(
            file_name="alpha.json",
            field="Bravo",
            ground_truth_id=123,
            ground_truth="Bravo GT",
        ),
        default_extraction(
            file_name="alpha.json",
            field="Charlie",
            ground_truth_id=234,
            ground_truth="Charlie GT",
        ),
        default_extraction(
            file_name="alpha.json",
            field="Bravo",
            ground_truth_id=345,
            ground_truth="Another Bravo GT",
        ),
        default_extraction(
            file_name="delta.json",
            field="Bravo",
            ground_truth_id=987,
            ground_truth="Bravo GT",
        ),
    ]
    prediction_extractions = [
        default_extraction(
            file_name="alpha.json",
            field="Bravo",
            prediction_id=456,
            prediction="Bravo Prediction",
            confidence=0.4,
        ),
        default_extraction(
            file_name="alpha.json",
            field="Charlie",
            prediction_id=567,
            prediction="Charlie Prediction",
            confidence=0.5,
        ),
        default_extraction(
            file_name="alpha.json",
            field="Charlie",
            prediction_id=678,
            prediction="Another Charlie Prediction",
            confidence=0.6,
        ),
        default_extraction(
            file_name="delta.json",
            field="Bravo",
            prediction_id=654,
            prediction="Charlie Prediction",
            confidence=0.6,
        ),
    ]
    combined_extractions = [
        default_extraction(
            file_name="alpha.json",
            field="Bravo",
            ground_truth_id=123,
            ground_truth="Bravo GT",
            prediction_id=456,
            prediction="Bravo Prediction",
            confidence=0.4,
            edit_distance=9,
            similarity=0.5833333333333333,
            accurate=False,
        ),
        default_extraction(
            file_name="alpha.json",
            field="Charlie",
            ground_truth_id=234,
            ground_truth="Charlie GT",
            prediction_id=567,
            prediction="Charlie Prediction",
            confidence=0.5,
            edit_distance=9,
            similarity=0.6428571428571428,
            accurate=False,
        ),
        default_extraction(
            file_name="alpha.json",
            field="Bravo",
            ground_truth_id=345,
            ground_truth="Another Bravo GT",
            edit_distance=16,
            similarity=0.0,
            accurate=False,
        ),
        default_extraction(
            file_name="delta.json",
            field="Bravo",
            ground_truth_id=987,
            ground_truth="Bravo GT",
            prediction_id=654,
            prediction="Charlie Prediction",
            confidence=0.6,
            edit_distance=15,
            similarity=0.23076923076923073,
            accurate=False,
        ),
        default_extraction(
            file_name="alpha.json",
            field="Charlie",
            prediction_id=678,
            prediction="Another Charlie Prediction",
            confidence=0.6,
            edit_distance=26,
            similarity=0.0,
            accurate=False,
        ),
    ]

    assert (
        sorted(
            combine_extractions_by_file_name(
                ground_truth_extractions, prediction_extractions
            ),
            key=lambda value: str(value.ground_truth_id) + str(value.prediction_id),
        )
        == combined_extractions
    )


def test_mutliple_values() -> None:
    pass
