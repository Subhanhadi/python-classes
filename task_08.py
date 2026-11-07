"""Task 08 - Cache Decorator with Expiration.

@ttl_cache(seconds) memoises results for ``seconds`` using time.monotonic().

Behaviour
---------
* Cache key is built from the *bound* arguments (inspect.signature), so
  f(2, 3), f(x=2, y=3), f(y=3, x=2) and calls relying on defaults all share
  one entry.
* Expired entries are recomputed (and counted as a miss).
* Exceptions raised by the function are never cached.
* UNHASHABLE arguments (lists, dicts, ...): the cache is bypassed, the function
  is called normally, and the call is counted under ``uncacheable``
  (it is not a hit and not a miss).
* functools.wraps keeps __name__, __doc__, __wrapped__, etc.
* wrapper.cache_info() -> dict(hits, misses, uncacheable, size); wrapper.cache_clear()
* Thread-safe bookkeeping (the function itself runs outside the lock).
"""
from __future__ import annotations

import functools
import inspect
import math
import threading
import time
import unittest
from unittest import mock


def ttl_cache(seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError("seconds must be a number")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be a positive, finite number")

    def decorator(func):
        cache = {}                                   # key -> (value, expires_at)
        stats = {"hits": 0, "misses": 0, "uncacheable": 0}
        lock = threading.RLock()
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):              # some builtins have no signature
            signature = None

        def make_key(args, kwargs):
            if signature is None:
                return (args, tuple(sorted(kwargs.items())))
            bound = signature.bind(*args, **kwargs)  # TypeError for wrong calls
            bound.apply_defaults()
            parts = []
            for name, value in bound.arguments.items():
                if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    value = tuple(sorted(value.items()))
                parts.append((name, value))
            return tuple(parts)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                key = make_key(args, kwargs)
                hash(key)
            except TypeError:
                # Either unhashable arguments or a wrong call signature.
                # Calling func lets Python raise the natural TypeError if the call is wrong.
                result = func(*args, **kwargs)
                with lock:
                    stats["uncacheable"] += 1
                return result

            now = time.monotonic()
            with lock:
                entry = cache.get(key)
                if entry is not None:
                    if now < entry[1]:
                        stats["hits"] += 1
                        return entry[0]
                    del cache[key]                   # expired
                stats["misses"] += 1

            value = func(*args, **kwargs)            # exceptions propagate, nothing cached

            with lock:
                stored_at = time.monotonic()
                cache[key] = (value, stored_at + seconds)
                for old_key in [k for k, (_, exp) in cache.items() if exp <= stored_at]:
                    del cache[old_key]               # keep memory bounded by live entries
            return value

        def cache_info():
            with lock:
                return {**stats, "size": len(cache)}

        def cache_clear():
            with lock:
                cache.clear()
                for k in stats:
                    stats[k] = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator


def demo():
    @ttl_cache(seconds=0.2)
    def slow_add(x, y):
        """Add two numbers."""
        print(f"  computing {x}+{y}")
        return x + y

    print(slow_add(x=2, y=3))
    print(slow_add(y=3, x=2))        # hit - keyword order is irrelevant
    time.sleep(0.25)
    print(slow_add(2, 3))            # expired -> recomputed
    print(slow_add.cache_info())


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TestTtlCache(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        patcher = mock.patch("time.monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = 0

        @ttl_cache(10)
        def add(x, y=0):
            """Docstring."""
            self.calls += 1
            return x + y
        self.add = add

    def test_exam_keyword_order_hit(self):
        self.assertEqual(self.add(x=2, y=3), 5)
        self.assertEqual(self.add(y=3, x=2), 5)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.add.cache_info()["hits"], 1)
        self.assertEqual(self.add.cache_info()["misses"], 1)

    def test_positional_keyword_and_default_equivalence(self):
        self.add(2, 3)
        self.add(x=2, y=3)
        self.add(2, y=3)
        self.add(2)
        self.add(2, 0)
        self.assertEqual(self.calls, 2)

    def test_expiry_recomputes(self):
        self.add(1)
        self.clock.now += 9.99
        self.add(1)
        self.assertEqual(self.calls, 1)
        self.clock.now += 0.02                    # now past the 10 s TTL
        self.add(1)
        self.assertEqual(self.calls, 2)
        self.assertEqual(self.add.cache_info()["misses"], 2)

    def test_different_args_are_different_entries(self):
        self.add(1)
        self.add(2)
        self.assertEqual(self.calls, 2)
        self.assertEqual(self.add.cache_info()["size"], 2)

    def test_exceptions_not_cached(self):
        @ttl_cache(10)
        def flaky(x):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first call fails")
            return x
        with self.assertRaises(RuntimeError):
            flaky(1)
        self.assertEqual(flaky(1), 1)

    def test_unhashable_arguments_bypass_cache(self):
        @ttl_cache(10)
        def total(items):
            self.calls += 1
            return sum(items)
        self.assertEqual(total([1, 2]), 3)
        self.assertEqual(total([1, 2]), 3)
        self.assertEqual(self.calls, 2)
        info = total.cache_info()
        self.assertEqual((info["uncacheable"], info["hits"], info["misses"]), (2, 0, 0))

    def test_wrong_call_raises_natural_type_error(self):
        with self.assertRaises(TypeError):
            self.add()                            # missing required argument
        with self.assertRaises(TypeError):
            self.add(1, 2, 3)

    def test_var_args_and_kwargs(self):
        @ttl_cache(10)
        def f(*args, **kwargs):
            self.calls += 1
            return (args, sorted(kwargs.items()))
        f(1, a=1, b=2)
        f(1, b=2, a=1)
        self.assertEqual(self.calls, 1)

    def test_metadata_preserved(self):
        self.assertEqual(self.add.__name__, "add")
        self.assertEqual(self.add.__doc__, "Docstring.")
        self.assertTrue(hasattr(self.add, "__wrapped__"))

    def test_cache_clear(self):
        self.add(1)
        self.add.cache_clear()
        self.assertEqual(self.add.cache_info(), {"hits": 0, "misses": 0, "uncacheable": 0, "size": 0})
        self.add(1)
        self.assertEqual(self.calls, 2)

    def test_expired_entries_are_purged(self):
        for i in range(5):
            self.add(i)
        self.clock.now += 100
        self.add(99)
        self.assertEqual(self.add.cache_info()["size"], 1)

    def test_invalid_ttl(self):
        for bad in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                ttl_cache(bad)
        for bad in ("5", None, True):
            with self.assertRaises(TypeError):
                ttl_cache(bad)

    def test_uses_time_monotonic(self):
        with mock.patch("time.monotonic", wraps=self.clock) as spy:
            self.add(5)
            self.assertTrue(spy.called)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
