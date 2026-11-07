"""Task 13 - Booking Conflict Scheduler.

Rooms hold their accepted bookings in start-sorted order together with
parallel lists of starts and ends. Because accepted bookings never overlap,
ends are sorted as well, so the bookings that clash with a new request form a
contiguous slice found with two binary searches (O(log n) per request plus
the cost of the insert).

Overlap rule (half-open intervals): a.start < b.end and b.start < a.end.
Adjacent bookings (one ends exactly when the next starts) are allowed.

book() returns a BookingResult. Rejections for data reasons (end <= start,
duplicate ID, time clash) are *returned* with a reason; wrong types or mixing
naive and timezone-aware datetimes raise (programming errors).
"""
from __future__ import annotations

import random
import unittest
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class Booking:
    booking_id: str
    room: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class BookingResult:
    accepted: bool
    booking: Booking
    conflicts: tuple = ()
    reason: str = ""


class _RoomCalendar:
    def __init__(self):
        self.bookings = []     # sorted by start
        self.starts = []
        self.ends = []

    def conflicts_with(self, start, end):
        upper = bisect_left(self.starts, end)        # bookings starting before `end`
        lower = bisect_right(self.ends, start)       # skip those finishing at/before `start`
        return self.bookings[lower:upper]

    def add(self, booking):
        index = bisect_right(self.starts, booking.start)
        self.bookings.insert(index, booking)
        self.starts.insert(index, booking.start)
        self.ends.insert(index, booking.end)

    def remove(self, booking):
        index = self.bookings.index(booking)
        for seq in (self.bookings, self.starts, self.ends):
            del seq[index]


class BookingScheduler:
    def __init__(self):
        self._rooms = {}
        self._by_id = {}
        self._aware = None          # becomes True/False with the first booking

    def book(self, booking_id, room, start, end):
        if not isinstance(booking_id, str) or not booking_id.strip():
            raise ValueError("booking_id must be a non-empty string")
        if not isinstance(room, str) or not room.strip():
            raise ValueError("room must be a non-empty string")
        for label, value in (("start", start), ("end", end)):
            if not isinstance(value, datetime):
                raise TypeError(f"{label} must be a datetime object")
        aware_flags = {start.tzinfo is not None and start.utcoffset() is not None,
                       end.tzinfo is not None and end.utcoffset() is not None}
        if len(aware_flags) != 1:
            raise ValueError("start and end must both be naive or both be timezone-aware")
        is_aware = aware_flags.pop()
        if self._aware is not None and self._aware != is_aware:
            raise ValueError("Cannot mix naive and timezone-aware datetimes in one scheduler")

        booking = Booking(booking_id.strip(), room.strip(), start, end)
        if end <= start:
            return BookingResult(False, booking, (), "end must be after start")
        if booking.booking_id in self._by_id:
            return BookingResult(False, booking, (), f"duplicate booking id {booking.booking_id!r}")

        self._aware = is_aware
        calendar = self._rooms.setdefault(booking.room, _RoomCalendar())
        clashes = tuple(calendar.conflicts_with(start, end))
        if clashes:
            return BookingResult(False, booking, clashes, "time conflict")

        calendar.add(booking)
        self._by_id[booking.booking_id] = booking
        return BookingResult(True, booking)

    def cancel(self, booking_id):
        booking = self._by_id.pop(booking_id, None)
        if booking is None:
            raise KeyError(f"Unknown booking id {booking_id!r}")
        self._rooms[booking.room].remove(booking)
        return booking

    def bookings_for(self, room):
        calendar = self._rooms.get(room)
        return list(calendar.bookings) if calendar else []


def demo():
    day = datetime(2026, 11, 7)
    at = lambda h, m=0: day.replace(hour=h, minute=m)
    scheduler = BookingScheduler()
    for bid, s, e in [("b1", at(9), at(10)), ("b2", at(10), at(11)), ("b3", at(9, 30), at(10, 30))]:
        result = scheduler.book(bid, "A", s, e)
        print(bid, "accepted" if result.accepted else f"rejected ({result.reason})",
              [c.booking_id for c in result.conflicts])


class TestScheduler(unittest.TestCase):
    base = datetime(2026, 11, 7)

    def t(self, hour, minute=0):
        return self.base.replace(hour=hour, minute=minute)

    def test_exam_sample(self):
        s = BookingScheduler()
        self.assertTrue(s.book("1", "A", self.t(9), self.t(10)).accepted)
        self.assertTrue(s.book("2", "A", self.t(10), self.t(11)).accepted)          # adjacent
        result = s.book("3", "A", self.t(9, 30), self.t(10, 30))
        self.assertFalse(result.accepted)
        self.assertEqual({c.booking_id for c in result.conflicts}, {"1", "2"})

    def test_same_time_other_room_ok(self):
        s = BookingScheduler()
        s.book("1", "A", self.t(9), self.t(10))
        self.assertTrue(s.book("2", "B", self.t(9), self.t(10)).accepted)

    def test_invalid_times_rejected(self):
        s = BookingScheduler()
        for end in (self.t(9), self.t(8)):
            r = s.book("x", "A", self.t(9), end)
            self.assertFalse(r.accepted)
            self.assertIn("end must be after start", r.reason)
        self.assertEqual(s.bookings_for("A"), [])

    def test_duplicate_id_rejected(self):
        s = BookingScheduler()
        s.book("1", "A", self.t(9), self.t(10))
        r = s.book("1", "B", self.t(12), self.t(13))
        self.assertFalse(r.accepted)
        self.assertIn("duplicate", r.reason)

    def test_rejected_id_may_be_reused(self):
        s = BookingScheduler()
        s.book("1", "A", self.t(9), self.t(10))
        self.assertFalse(s.book("2", "A", self.t(9), self.t(10)).accepted)
        self.assertTrue(s.book("2", "A", self.t(10), self.t(11)).accepted)

    def test_containing_and_contained(self):
        s = BookingScheduler()
        s.book("1", "A", self.t(9), self.t(12))
        self.assertFalse(s.book("2", "A", self.t(10), self.t(11)).accepted)
        self.assertFalse(s.book("3", "A", self.t(8), self.t(13)).accepted)
        self.assertTrue(s.book("4", "A", self.t(7), self.t(9)).accepted)
        self.assertTrue(s.book("5", "A", self.t(12), self.t(13)).accepted)

    def test_cancel_frees_slot(self):
        s = BookingScheduler()
        s.book("1", "A", self.t(9), self.t(10))
        s.cancel("1")
        self.assertTrue(s.book("2", "A", self.t(9), self.t(10)).accepted)
        with self.assertRaises(KeyError):
            s.cancel("nope")

    def test_bookings_sorted(self):
        s = BookingScheduler()
        for bid, h in [("c", 14), ("a", 8), ("b", 11)]:
            s.book(bid, "A", self.t(h), self.t(h + 1))
        self.assertEqual([b.booking_id for b in s.bookings_for("A")], ["a", "b", "c"])

    def test_type_and_timezone_validation(self):
        s = BookingScheduler()
        with self.assertRaises(TypeError):
            s.book("1", "A", "09:00", self.t(10))
        with self.assertRaises(ValueError):
            s.book("", "A", self.t(9), self.t(10))
        with self.assertRaises(ValueError):
            s.book("1", " ", self.t(9), self.t(10))
        with self.assertRaises(ValueError):
            s.book("1", "A", self.t(9), self.t(10).replace(tzinfo=timezone.utc))
        s.book("2", "A", self.t(9), self.t(10))
        with self.assertRaises(ValueError):
            s.book("3", "A", self.t(11).replace(tzinfo=timezone.utc), self.t(12).replace(tzinfo=timezone.utc))

    def test_timezone_aware_bookings(self):
        s = BookingScheduler()
        utc = timezone.utc
        plus5 = timezone(timedelta(hours=5))
        a = datetime(2026, 11, 7, 9, tzinfo=utc)
        # 14:00+05:00 == 09:00 UTC, so this overlaps the first booking
        self.assertTrue(s.book("1", "A", a, a + timedelta(hours=1)).accepted)
        b = datetime(2026, 11, 7, 14, tzinfo=plus5)
        self.assertFalse(s.book("2", "A", b, b + timedelta(hours=1)).accepted)

    def test_fuzz_against_bruteforce(self):
        rng = random.Random(42)
        for _ in range(50):
            s, accepted = BookingScheduler(), []
            for i in range(60):
                start = self.base + timedelta(minutes=rng.randrange(0, 600, 15))
                end = start + timedelta(minutes=rng.randrange(15, 120, 15))
                expected = [b for b in accepted if start < b[1] and b[0] < end]
                result = s.book(str(i), "R", start, end)
                self.assertEqual(result.accepted, not expected)
                self.assertEqual(sorted(c.booking_id for c in result.conflicts),
                                 sorted(str(b[2]) for b in expected))
                if result.accepted:
                    accepted.append((start, end, i))


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
