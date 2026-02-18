
"""
计算两个字符串数组的并集、交集及其长度.
"""
from __future__ import annotations

from typing import Iterable, List, Set, Tuple


def compute_union_intersection(
    first: Iterable[str],
    second: Iterable[str],
) -> Tuple[Set[str], Set[str]]:
    """返回并集与交集."""
    set_a = set(first)
    set_b = set(second)
    return set_a | set_b, set_a & set_b


def format_result(union_set: Set[str], intersection_set: Set[str]) -> str:
    """格式化输出."""
    union_sorted = sorted(union_set)
    intersection_sorted = sorted(intersection_set)
    return "\n".join(
        [
            f"并集({len(union_sorted)}): {union_sorted}",
            f"交集({len(intersection_sorted)}): {intersection_sorted}",
        ]
    )


if __name__ == "__main__":
    array_a = [
        "MIME type",
        "Other paths",
        "Rust panic",
        "arbitrary text",
        "error string",
        "frontend renderer",
        "panic strings",
        "plain text",
        "rich content",
        "tag name",
        "tag names",
        "valid name",
    ]
    array_b = [
        "MIME type",
        "error string",
        "panic strings",
        "tag names",
        "valid name",
        "new concept",
    ]
    union_set, intersection_set = compute_union_intersection(array_a, array_b)
    print(len(union_set))
    print(len(intersection_set))
    print(format_result(union_set, intersection_set))

