from __future__ import annotations

import itertools
import operator
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

import Levenshtein
from munkres import Munkres
from pathvalidate import sanitize_filename

Value = TypeVar("Value")


def group_by_attr(
    values: Iterable[Value], attr_name: str
) -> tuple[tuple[str, tuple[Value, ...]], ...]:
    """
    Group an iterable of values by `attr_name`, producing an interator of
    (attr_value, (value_with_attr, ...)) pairs.

    E.g.
    >>> group_by_attr([range(1, 3), range(1, 4), range(5, 6)], "start")
    (
        (1, (range(1, 3), range(1, 4))),
        (5, (range(5, 6),)),
    )
    """
    return tuple(
        (attr_value, tuple(values))
        for attr_value, values in itertools.groupby(
            sorted(
                values,
                key=operator.attrgetter(attr_name),
            ),
            key=operator.attrgetter(attr_name),
        )
    )


def sanitize(file_name: str) -> str:
    return sanitize_filename(
        file_name,
        max_len=None,
        platform="auto",
        replacement_text="_",
    )


def zip_match_longest(
    left: Iterable[Value | None],
    right: Iterable[Value | None],
    *,
    left_key: Callable[[Value], str | None] = str,
    right_key: Callable[[Value], str | None] = str,
) -> Iterator[tuple[Value | None, Value | None]]:
    """
    Zip `left` and `right` iterables, reordering their values such that the total
    Levenshtein edit distance of paired values is minimized.

    `left_key` and `right_key` are used to produce the string for each value.

    Either `left` or `right` will be padded with `None` if it's shorter than the other.

    This is a version of the assignment problem where `left` and `right` values form
    a bipartite graph and edit distances are the costs assigned to edges connecting
    them. The Hungarian algorithm is used to determine the pairings that minimize the
    total cost of the graph.
    """
    left, right = list(left), list(right)
    max_len = max(len(left), len(right))
    left += [None] * (max_len - len(left))
    right += [None] * (max_len - len(right))

    left_values = [
        left_key(left_value) if left_value is not None else None for left_value in left
    ]
    right_values = [
        right_key(right_value) if right_value is not None else None
        for right_value in right
    ]

    edit_distance_graph = [
        [
            Levenshtein.distance(left_value, right_value)
            if left_value is not None and right_value is not None
            else 0
            for right_value in right_values
        ]
        for left_value in left_values
    ]
    matched_pair_indices = Munkres().compute(edit_distance_graph)

    for left_index, right_index in matched_pair_indices:
        yield left[left_index], right[right_index]
