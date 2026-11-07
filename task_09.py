"""Task 09 - Streaming Log Inspector.

read_errors(path)      generator yielding (line_number, timestamp, message) for
                       ERROR records only; the file is streamed line by line.
summarize_errors(path) counts ERROR records per date (YYYY-MM-DD).

Expected line format:  YYYY-MM-DD HH:MM:SS | LEVEL | message
* malformed lines (wrong field count, bad timestamp, non-alphabetic level,
  empty message) are skipped and counted; blank lines are ignored silently.
* the message may itself contain '|' characters.
* line numbers are 1-based physical line numbers in the file.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data" / "task_09"


@dataclass
class ParseStats:
    """Filled in while read_errors() is consumed."""
    lines_read: int = 0
    malformed: int = 0
    errors: int = 0


@dataclass
class ErrorSummary:
    counts_by_date: dict = field(default_factory=dict)
    total_errors: int = 0
    malformed_lines: int = 0


def _parse_line(line):
    parts = line.rstrip("\r\n").split("|", 2)
    if len(parts) != 3:
        return None
    timestamp, level, message = (p.strip() for p in parts)
    try:
        datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    except ValueError:
        return None
    if not level.isalpha() or not message:
        return None
    return timestamp, level.upper(), message


def read_errors(path, stats=None):
    """Lazily yield (line_number, timestamp, message) for ERROR records."""
    stats = stats if stats is not None else ParseStats()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stats.lines_read = line_number
            if not line.strip():
                continue
            parsed = _parse_line(line)
            if parsed is None:
                stats.malformed += 1
                continue
            timestamp, level, message = parsed
            if level == "ERROR":
                stats.errors += 1
                yield line_number, timestamp, message


def summarize_errors(path):
    stats = ParseStats()
    counts = Counter()
    for _, timestamp, _ in read_errors(path, stats):
        counts[timestamp[:10]] += 1
    return ErrorSummary(dict(sorted(counts.items())), stats.errors, stats.malformed)


def create_sample_log(path):
    lines = [
        "2026-11-07 10:30:00 | ERROR | Connection lost",
        "2026-11-07 10:31:12 | INFO | Reconnected",
        "this line is garbage",
        "2026-11-07 11:00:00 | ERROR | Disk almost full | usage=97%",
        "2026-11-08 09:15:30 | WARNING | Slow response",
        "2026-13-45 99:99:99 | ERROR | Impossible timestamp",
        "2026-11-08 09:45:00 | ERROR | Timeout calling payment API",
        "",
        "2026-11-09 08:00:00 | ERROR |",
        "2026-11-09 08:05:00 | ERROR | Null pointer in checkout",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def demo():
    log = create_sample_log(SAMPLE_DIR / "app.log")
    print("Sample log written to", log)
    stats = ParseStats()
    for record in read_errors(log, stats):
        print(record)
    print(stats)
    print(summarize_errors(log))


class TestLogInspector(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "test.log"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_exam_sample_line(self):
        self.write("2026-11-07 10:30:00 | ERROR | Connection lost\n")
        self.assertEqual(list(read_errors(self.path)), [(1, "2026-11-07 10:30:00", "Connection lost")])

    def test_is_a_lazy_generator(self):
        import inspect
        self.write("2026-11-07 10:30:00 | ERROR | x\n")
        gen = read_errors(self.path)
        self.assertTrue(inspect.isgenerator(gen))

    def test_only_errors_yielded_and_line_numbers_physical(self):
        self.write("2026-11-07 10:00:00 | INFO | a\njunk\n2026-11-07 10:01:00 | ERROR | b\n\n"
                   "2026-11-07 10:02:00 | ERROR | c\n")
        self.assertEqual([r[0] for r in read_errors(self.path)], [3, 5])

    def test_malformed_lines_counted_not_blank_lines(self):
        log = create_sample_log(self.path)
        stats = ParseStats()
        records = list(read_errors(log, stats))
        self.assertEqual(len(records), 4)
        self.assertEqual(stats.malformed, 3)         # garbage, bad timestamp, empty message
        self.assertEqual(stats.errors, 4)
        self.assertEqual(stats.lines_read, 10)

    def test_message_may_contain_pipe(self):
        self.write("2026-11-07 11:00:00 | ERROR | Disk full | usage=97%\n")
        self.assertEqual(list(read_errors(self.path))[0][2], "Disk full | usage=97%")

    def test_summary_by_date(self):
        summary = summarize_errors(create_sample_log(self.path))
        self.assertEqual(summary.counts_by_date,
                         {"2026-11-07": 2, "2026-11-08": 1, "2026-11-09": 1})
        self.assertEqual((summary.total_errors, summary.malformed_lines), (4, 3))

    def test_empty_file_and_no_errors(self):
        self.write("")
        self.assertEqual(list(read_errors(self.path)), [])
        self.assertEqual(summarize_errors(self.path), ErrorSummary({}, 0, 0))
        self.write("2026-11-07 10:00:00 | INFO | fine\n")
        self.assertEqual(summarize_errors(self.path).total_errors, 0)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            list(read_errors(Path(self.tmp.name) / "nope.log"))

    def test_crlf_and_last_line_without_newline(self):
        self.write("2026-11-07 10:00:00 | ERROR | one\r\n2026-11-07 10:00:01 | ERROR | two")
        self.assertEqual([r[2] for r in read_errors(self.path)], ["one", "two"])

    def test_streaming_large_file_without_loading_it(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            for i in range(50000):
                fh.write(f"2026-11-07 10:00:00 | {'ERROR' if i % 2 else 'INFO'} | msg {i}\n")
        self.assertEqual(summarize_errors(self.path).total_errors, 25000)

    def test_invalid_utf8_does_not_crash(self):
        self.path.write_bytes(b"2026-11-07 10:00:00 | ERROR | caf\xe9\n")
        self.assertEqual(len(list(read_errors(self.path))), 1)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
