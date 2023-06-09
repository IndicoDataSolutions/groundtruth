import itertools
import operator
from collections.abc import Iterable
from typing import TypeVar

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
