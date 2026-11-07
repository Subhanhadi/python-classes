"""Task 12 - Custom Iterable Range Toolkit.

ReversibleRange(start, stop, step=1) mimics built-in range() semantics
(stop-exclusive, positive or negative non-zero step) but is implemented from
scratch with O(1) memory: no list is ever built.

Implemented: __iter__, __reversed__, __len__, __contains__ (all O(1) except
iteration itself), plus __getitem__, __eq__, __hash__ and __repr__.
ReversibleRange(n) with a single argument means ReversibleRange(0, n).
"""
from __future__ import annotations

import operator
import unittest


def _as_int(value, name):
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


class ReversibleRange:
    __slots__ = ("start", "stop", "step")

    def __init__(self, start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        self.start = _as_int(start, "start")
        self.stop = _as_int(stop, "stop")
        self.step = _as_int(step, "step")
        if self.step == 0:
            raise ValueError("step must not be zero")

    # -- size -------------------------------------------------------------
    def __len__(self):
        if self.step > 0:
            if self.start >= self.stop:
                return 0
            return (self.stop - self.start + self.step - 1) // self.step
        if self.start <= self.stop:
            return 0
        return (self.start - self.stop - self.step - 1) // (-self.step)

    # -- iteration ----------------------------------------------------------
    def __iter__(self):
        value = self.start
        if self.step > 0:
            while value < self.stop:
                yield value
                value += self.step
        else:
            while value > self.stop:
                yield value
                value += self.step

    def __reversed__(self):
        index = len(self) - 1
        while index >= 0:
            yield self.start + index * self.step
            index -= 1

    # -- membership ---------------------------------------------------------
    def __contains__(self, item):
        if isinstance(item, float) and item.is_integer():
            item = int(item)                 # range(5).__contains__(3.0) is True too
        if not isinstance(item, int):
            return False
        if self.step > 0:
            in_bounds = self.start <= item < self.stop
        else:
            in_bounds = self.stop < item <= self.start
        return in_bounds and (item - self.start) % self.step == 0

    # -- extras -------------------------------------------------------------
    def __getitem__(self, index):
        if isinstance(index, slice):
            r = range(self.start, self.stop, self.step)[index]
            return ReversibleRange(r.start, r.stop, r.step)
        index = _as_int(index, "index")
        size = len(self)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError("ReversibleRange index out of range")
        return self.start + index * self.step

    def _identity(self):
        size = len(self)
        if size == 0:
            return (0,)
        return (size, self.start, self.step if size > 1 else 0)

    def __eq__(self, other):
        return isinstance(other, ReversibleRange) and self._identity() == other._identity()

    def __hash__(self):
        return hash(self._identity())

    def __repr__(self):
        if self.step == 1:
            return f"ReversibleRange({self.start}, {self.stop})"
        return f"ReversibleRange({self.start}, {self.stop}, {self.step})"


def demo():
    r = ReversibleRange(2, 11, 3)
    print(r, list(r), list(reversed(r)), len(r), 5 in r, 6 in r)
    down = ReversibleRange(10, 0, -3)
    print(down, list(down), list(reversed(down)), len(down), 4 in down)
    print(len(ReversibleRange(0, 10**15, 7)), 7 * 10**9 in ReversibleRange(0, 10**15, 7))


class TestReversibleRange(unittest.TestCase):
    def test_exam_sample(self):
        r = ReversibleRange(2, 11, 3)
        self.assertEqual(list(r), [2, 5, 8])
        self.assertEqual(list(reversed(r)), [8, 5, 2])
        self.assertTrue(5 in r)
        self.assertFalse(6 in r)
        self.assertEqual(len(r), 3)

    def test_matches_builtin_range_exhaustively(self):
        for start in range(-6, 7):
            for stop in range(-6, 7):
                for step in [s for s in range(-4, 5) if s != 0]:
                    ref, mine = range(start, stop, step), ReversibleRange(start, stop, step)
                    label = (start, stop, step)
                    self.assertEqual(list(mine), list(ref), label)
                    self.assertEqual(list(reversed(mine)), list(reversed(ref)), label)
                    self.assertEqual(len(mine), len(ref), label)
                    for probe in range(-9, 10):
                        self.assertEqual(probe in mine, probe in ref, (label, probe))

    def test_empty_ranges(self):
        for args in [(5, 5), (5, 1), (1, 5, -1), (0, 0, 3)]:
            r = ReversibleRange(*args)
            self.assertEqual((len(r), list(r), list(reversed(r))), (0, [], []))
            self.assertFalse(3 in r)

    def test_zero_step_and_types(self):
        with self.assertRaises(ValueError):
            ReversibleRange(0, 5, 0)
        for bad in [(1.5, 5), (0, "5"), (0, 5, 0.5), (None, None)]:
            with self.assertRaises(TypeError):
                ReversibleRange(*bad)

    def test_single_argument(self):
        self.assertEqual(list(ReversibleRange(4)), [0, 1, 2, 3])

    def test_membership_non_ints(self):
        r = ReversibleRange(0, 10, 2)
        self.assertTrue(4.0 in r)
        self.assertFalse(4.5 in r)
        self.assertFalse("4" in r)
        self.assertFalse(None in r)

    def test_huge_range_is_constant_memory(self):
        r = ReversibleRange(0, 10 ** 18, 3)
        self.assertEqual(len(r), (10 ** 18 + 2) // 3)
        self.assertTrue(3 * 10 ** 17 in r)
        self.assertFalse(3 * 10 ** 17 + 1 in r)
        self.assertEqual(next(iter(r)), 0)
        self.assertEqual(next(reversed(r)), 3 * ((10 ** 18 - 1) // 3))

    def test_iterable_more_than_once_and_independent_iterators(self):
        r = ReversibleRange(3)
        self.assertEqual(list(r), list(r))
        a, b = iter(r), iter(r)
        self.assertEqual((next(a), next(a), next(b)), (0, 1, 0))

    def test_getitem_eq_hash_repr(self):
        r = ReversibleRange(2, 11, 3)
        self.assertEqual((r[0], r[-1], r[1]), (2, 8, 5))
        with self.assertRaises(IndexError):
            r[3]
        self.assertEqual(list(r[::-1]), [8, 5, 2])
        self.assertEqual(ReversibleRange(0, 3), ReversibleRange(0, 3, 1))
        self.assertEqual(ReversibleRange(0, 0), ReversibleRange(5, 5))
        self.assertEqual(hash(ReversibleRange(0, 3)), hash(ReversibleRange(0, 3, 1)))
        self.assertEqual(repr(r), "ReversibleRange(2, 11, 3)")

    def test_defines_required_dunders(self):
        for name in ("__iter__", "__reversed__", "__len__", "__contains__"):
            self.assertIn(name, vars(ReversibleRange))


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
