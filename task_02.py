"""Task 02 - Longest Valid Window (longest substring without repeating chars).

Sliding window with a "last seen index" dictionary: O(n) time, O(k) space
where k is the number of distinct characters. Spaces and punctuation are
ordinary characters. On equal-length windows the earliest one wins because we
only replace the best window when a strictly longer one is found.
"""
from __future__ import annotations

import random
import unittest


def longest_unique_window(text):
    """Return (substring, start_index, length) of the longest unique window."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    last_seen = {}
    left = best_start = best_len = 0
    for right, ch in enumerate(text):
        # If ch is already inside the current window, shrink past its last use.
        if last_seen.get(ch, -1) >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        if right - left + 1 > best_len:      # strict ">" keeps the earliest window
            best_start, best_len = left, right - left + 1
    return text[best_start:best_start + best_len], best_start, best_len


def _brute_force(text):
    best = ("", 0, 0)
    for i in range(len(text)):
        for j in range(i, len(text)):
            window = text[i:j + 1]
            if len(set(window)) != len(window):
                break
            if len(window) > best[2]:
                best = (window, i, len(window))
    return best


def demo():
    for s in ["abcabcbb", "", "bbbbb", "pwwkew", "a b!a b?"]:
        print(repr(s), "->", longest_unique_window(s))


class TestLongestUniqueWindow(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(longest_unique_window("abcabcbb"), ("abc", 0, 3))
        self.assertEqual(longest_unique_window(""), ("", 0, 0))

    def test_all_same(self):
        self.assertEqual(longest_unique_window("bbbbb"), ("b", 0, 1))

    def test_later_longer_window(self):
        self.assertEqual(longest_unique_window("pwwkew"), ("wke", 2, 3))

    def test_repeat_inside_window(self):
        self.assertEqual(longest_unique_window("abba"), ("ab", 0, 2))
        self.assertEqual(longest_unique_window("tmmzuxt"), ("mzuxt", 2, 5))

    def test_earliest_on_tie(self):
        self.assertEqual(longest_unique_window("abcdabcd"), ("abcd", 0, 4))

    def test_spaces_punctuation_and_case(self):
        self.assertEqual(longest_unique_window("a a"), ("a ", 0, 2))
        self.assertEqual(longest_unique_window("aA"), ("aA", 0, 2))
        self.assertEqual(longest_unique_window("!!@#!"), ("!@#", 1, 3))

    def test_unicode(self):
        self.assertEqual(longest_unique_window("héllo wörld"), ("o wörld", 4, 7))

    def test_single_char(self):
        self.assertEqual(longest_unique_window("x"), ("x", 0, 1))

    def test_type_error(self):
        for bad in (None, 123, ["a"]):
            with self.assertRaises(TypeError):
                longest_unique_window(bad)

    def test_matches_brute_force_random(self):
        rng = random.Random(7)
        for _ in range(300):
            s = "".join(rng.choice("abcd ") for _ in range(rng.randint(0, 25)))
            self.assertEqual(longest_unique_window(s), _brute_force(s), s)

    def test_linear_performance(self):
        text = "abcdefghij" * 20000          # 200k chars would be too slow if O(n^2)
        self.assertEqual(longest_unique_window(text)[2], 10)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
