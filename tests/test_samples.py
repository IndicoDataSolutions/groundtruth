from __future__ import annotations

from groundtruth.samples import (
    Sample,
    combine_samples_by_file_name,
    samples_for_results,
)


def default_sample(
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
) -> Sample:
    return Sample(
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


class TestSample:
    @staticmethod
    def test_true_positive() -> None:
        sample = default_sample(
            ground_truth="True Positive",
            prediction="True Positive",
            accurate=True,
        )

        assert sample.true_positive is True
        assert sample.false_negative is False
        assert sample.false_positive is False
        assert sample.true_negative is False

    @staticmethod
    def test_false_negative() -> None:
        sample = default_sample(
            ground_truth="False Negative",
            prediction=None,
            accurate=False,
        )

        assert sample.true_positive is False
        assert sample.false_negative is True
        assert sample.false_positive is False
        assert sample.true_negative is False

    @staticmethod
    def test_false_positive() -> None:
        sample = default_sample(
            ground_truth=None,
            prediction="False Positive",
            accurate=False,
        )

        assert sample.true_positive is False
        assert sample.false_negative is False
        assert sample.false_positive is True
        assert sample.true_negative is False

    @staticmethod
    def test_true_negative() -> None:
        sample = default_sample(
            ground_truth=None,
            prediction=None,
            accurate=True,
        )

        assert sample.true_positive is False
        assert sample.false_negative is False
        assert sample.false_positive is False
        assert sample.true_negative is True

    @staticmethod
    def test_from_values() -> None:
        sample = Sample.from_values(
            file_name="Test File",
            field="Test Field",
            ground_truth_id=123,
            prediction_id=456,
            ground_truth="Ground Truth",
            prediction="Prediction",
            confidence=0.9,
        )

        assert sample.file_name == "Test File"
        assert sample.field == "Test Field"
        assert sample.ground_truth_id == 123
        assert sample.prediction_id == 456
        assert sample.ground_truth == "Ground Truth"
        assert sample.prediction == "Prediction"
        assert sample.confidence == 0.9
        assert sample.edit_distance == 10
        assert sample.similarity == 0.2727272727272727
        assert sample.accurate is False


def test_sample() -> None:
    ground_truths = [
        {"label": "Alpha", "text": "Tres"},
        {"label": "Bravo", "text": "Duo"},
        {"label": "Bravo", "text": "Unus"},
    ]
    predictions = [
        {"label": "Alpha", "text": "Tres", "confidence": {"Alpha": 1.0}},
        {"label": "Bravo", "text": "Unus", "confidence": {"Bravo": 0.8}},
        {"label": "Bravo", "text": "Duodenum", "confidence": {"Bravo": 0.9}},
    ]
    results_and_names = [
        (
            "abc.json",
            {
                "submission_id": 123,
                "results": {
                    "document": {
                        "results": {
                            "Test Model": {
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

    alpha_sample, bravo_sample, bravo_2_sample = list(
        samples_for_results(results_and_names)
    )

    assert alpha_sample.true_positive is True
    assert bravo_sample.false_positive is True
    assert bravo_2_sample.true_positive is True


def test_combine() -> None:
    ground_truth_samples = [
        default_sample(
            file_name="alpha.json",
            field="Bravo",
            ground_truth_id=123,
            ground_truth="Bravo GT",
        ),
        default_sample(
            file_name="alpha.json",
            field="Charlie",
            ground_truth_id=234,
            ground_truth="Charlie GT",
        ),
        default_sample(
            file_name="alpha.json",
            field="Bravo",
            ground_truth_id=345,
            ground_truth="Another Bravo GT",
        ),
        default_sample(
            file_name="delta.json",
            field="Bravo",
            ground_truth_id=987,
            ground_truth="Bravo GT",
        ),
    ]
    prediction_samples = [
        default_sample(
            file_name="alpha.json",
            field="Bravo",
            prediction_id=456,
            prediction="Bravo Prediction",
            confidence=0.4,
        ),
        default_sample(
            file_name="alpha.json",
            field="Charlie",
            prediction_id=567,
            prediction="Charlie Prediction",
            confidence=0.5,
        ),
        default_sample(
            file_name="alpha.json",
            field="Charlie",
            prediction_id=678,
            prediction="Another Charlie Prediction",
            confidence=0.6,
        ),
        default_sample(
            file_name="delta.json",
            field="Bravo",
            prediction_id=654,
            prediction="Charlie Prediction",
            confidence=0.6,
        ),
    ]
    combined_samples = [
        default_sample(
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
        default_sample(
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
        default_sample(
            file_name="alpha.json",
            field="Bravo",
            ground_truth_id=345,
            ground_truth="Another Bravo GT",
            edit_distance=16,
            similarity=0.0,
            accurate=False,
        ),
        default_sample(
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
        default_sample(
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
            combine_samples_by_file_name(ground_truth_samples, prediction_samples),
            key=lambda value: str(value.ground_truth_id) + str(value.prediction_id),
        )
        == combined_samples
    )


def test_mutliple_values() -> None:
    pass
