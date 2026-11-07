"""Task 03 - Recursive JSON Path Explorer.

find_paths(data, target) returns every path whose value equals ``target``.

Path format
-----------
* dictionary keys use dot notation:      a.b.c
* list/tuple positions use [index]:      a[0].x   or   [1][2]
* a match on the root object itself yields the empty path ""
* traversal is pre-order and keeps dict insertion order / list order
* ``True``/``False`` never match 1/0 (and vice-versa)
* circular references raise ValueError instead of recursing forever

Note: keys that themselves contain dots are not escaped (the notation is
ambiguous for them); this is documented rather than hidden.
"""
from __future__ import annotations

import unittest


def _equals(value, target):
    """Equality that refuses to treat bool as a number."""
    if isinstance(value, bool) != isinstance(target, bool):
        return False
    try:
        return bool(value == target)
    except Exception:                      # exotic objects with broken __eq__
        return False


def find_paths(data, target):
    results = []
    active = set()                          # ids of containers on current path

    def walk(node, path):
        if _equals(node, target):
            results.append(path)

        if isinstance(node, (dict, list, tuple)):
            if id(node) in active:
                raise ValueError(f"Circular reference detected at path {path or '<root>'!r}")
            active.add(id(node))
            try:
                if isinstance(node, dict):
                    for key, child in node.items():
                        segment = str(key)
                        walk(child, f"{path}.{segment}" if path else segment)
                else:
                    for index, child in enumerate(node):
                        walk(child, f"{path}[{index}]")
            finally:
                active.discard(id(node))

    walk(data, "")
    return results


def demo():
    data = {"a": [{"x": 5}, 5], "b": {"y": 5}}
    print(find_paths(data, 5))


class TestFindPaths(unittest.TestCase):
    def test_exam_sample(self):
        data = {"a": [{"x": 5}, 5], "b": {"y": 5}}
        self.assertEqual(find_paths(data, 5), ["a[0].x", "a[1]", "b.y"])

    def test_no_match(self):
        self.assertEqual(find_paths({"a": [1, 2]}, 99), [])

    def test_bool_not_numeric(self):
        data = {"flag": True, "one": 1, "zero": 0, "off": False}
        self.assertEqual(find_paths(data, 1), ["one"])
        self.assertEqual(find_paths(data, True), ["flag"])
        self.assertEqual(find_paths(data, 0), ["zero"])
        self.assertEqual(find_paths(data, False), ["off"])

    def test_nested_lists(self):
        self.assertEqual(find_paths([[5], [1, [5]]], 5), ["[0][0]", "[1][1][0]"])

    def test_none_and_strings(self):
        data = {"a": None, "b": ["x", None]}
        self.assertEqual(find_paths(data, None), ["a", "b[1]"])
        self.assertEqual(find_paths(data, "x"), ["b[0]"])

    def test_container_target_and_children(self):
        data = {"a": {"k": 1}, "b": [{"k": 1}]}
        self.assertEqual(find_paths(data, {"k": 1}), ["a", "b[0]"])

    def test_root_match_and_empty(self):
        self.assertEqual(find_paths(5, 5), [""])
        self.assertEqual(find_paths({}, 5), [])
        self.assertEqual(find_paths([], 5), [])

    def test_non_string_keys(self):
        self.assertEqual(find_paths({1: {2: "t"}}, "t"), ["1.2"])

    def test_traversal_order_preserved(self):
        data = {"z": 1, "a": 1, "m": [1, {"q": 1}]}
        self.assertEqual(find_paths(data, 1), ["z", "a", "m[0]", "m[1].q"])

    def test_circular_reference(self):
        loop = {"a": 1}
        loop["self"] = loop
        with self.assertRaises(ValueError):
            find_paths(loop, 1)

    def test_shared_but_not_circular_reference_is_fine(self):
        shared = {"v": 1}
        self.assertEqual(find_paths({"a": shared, "b": shared}, 1), ["a.v", "b.v"])

    def test_deep_nesting(self):
        node = 7
        for _ in range(200):
            node = [node]
        self.assertEqual(len(find_paths(node, 7)), 1)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
