from dataclasses import dataclass

from groundtruth.utils import group_by_attr, sanitize, zip_match_longest


@dataclass
class Data:
    name: str
    number: int


def test_group_by_attr() -> None:
    values = [
        Data("A", 1),
        Data("B", 2),
        Data("A", 3),
        Data("C", 4),
        Data("A", 5),
        Data("C", 6),
    ]
    groups = (
        ("A", (Data("A", 1), Data("A", 3), Data("A", 5))),
        ("B", (Data("B", 2),)),
        ("C", (Data("C", 4), Data("C", 6))),
    )

    assert group_by_attr(values, "name") == groups


def test_sanitize() -> None:
    assert sanitize("06/09/2023.json") == "06_09_2023.json"


class TestZipMatchLongest:
    @staticmethod
    def test_left_smaller() -> None:
        left = [
            "Alpha",
            "Bravo",
        ]
        right = [
            "Brvao",
            "Charlie",
            "Delta",
            "Alhpa",
        ]

        assert list(zip_match_longest(left, right)) == [
            ("Alpha", "Alhpa"),
            ("Bravo", "Brvao"),
            (None, "Charlie"),
            (None, "Delta"),
        ]

    @staticmethod
    def test_right_smaller() -> None:
        left = [
            "Alpha",
            "Bravo",
            "Charlie",
            "Delta",
        ]
        right = [
            "Brvao",
            "Alhpa",
        ]

        assert list(zip_match_longest(left, right)) == [
            ("Alpha", "Alhpa"),
            ("Bravo", "Brvao"),
            ("Charlie", None),
            ("Delta", None),
        ]

    @staticmethod
    def test_same_size() -> None:
        left = [
            "Alpha",
            "Bravo",
            "Charlie",
            "Delta",
        ]
        right = [
            "Dorlta",
            "Brvao",
            "Chorlie",
            "Alhpa",
        ]

        assert list(zip_match_longest(left, right)) == [
            ("Alpha", "Alhpa"),
            ("Bravo", "Brvao"),
            ("Charlie", "Chorlie"),
            ("Delta", "Dorlta"),
        ]

    @staticmethod
    def test_zero() -> None:
        left: list[str] = []
        right: list[str] = []

        assert list(zip_match_longest(left, right)) == []
