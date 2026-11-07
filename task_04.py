"""Task 04 - Dependency Execution Planner.

Topological ordering (Kahn's algorithm) using a min-heap so that, whenever
several tasks are available, the alphabetically first one is chosen.

* dependencies are pairs (A, B): A must finish before B
* unknown task names, duplicate task names and malformed pairs -> ValueError
* cycles (including self-dependencies) -> ValueError naming the cycle
"""
from __future__ import annotations

import heapq
import unittest


def _describe_cycle(remaining, predecessors):
    """Return one concrete cycle as 'a -> b -> a' from the unresolved tasks."""
    node = min(remaining)
    path, position = [], {}
    while node not in position:
        position[node] = len(path)
        path.append(node)
        # every unresolved node has at least one unresolved predecessor
        node = min(p for p in predecessors[node] if p in remaining)
    cycle = path[position[node]:]
    cycle.reverse()                         # we walked backwards along edges
    start = cycle.index(min(cycle))         # rotate so the message is deterministic
    cycle = cycle[start:] + cycle[:start]
    return " -> ".join(cycle + [cycle[0]])


def plan_execution(tasks, dependencies):
    if isinstance(tasks, (str, bytes)) or tasks is None:
        raise TypeError("tasks must be an iterable of task names")
    tasks = list(tasks)
    if not all(isinstance(t, str) and t for t in tasks):
        raise ValueError("Task names must be non-empty strings")
    if len(set(tasks)) != len(tasks):
        dupes = sorted({t for t in tasks if tasks.count(t) > 1})
        raise ValueError(f"Duplicate task names: {dupes}")
    known = set(tasks)

    successors = {t: set() for t in tasks}
    predecessors = {t: set() for t in tasks}
    for pair in dependencies or []:
        try:
            before, after = pair
        except (TypeError, ValueError):
            raise ValueError(f"Invalid dependency {pair!r}; expected (before, after)") from None
        for name in (before, after):
            if name not in known:
                raise ValueError(f"Unknown task in dependency {pair!r}: {name!r}")
        successors[before].add(after)       # sets make repeated pairs harmless
        predecessors[after].add(before)

    indegree = {t: len(predecessors[t]) for t in tasks}
    ready = [t for t in tasks if indegree[t] == 0]
    heapq.heapify(ready)

    order = []
    while ready:
        current = heapq.heappop(ready)
        order.append(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, nxt)

    if len(order) != len(tasks):
        remaining = set(tasks) - set(order)
        raise ValueError(
            f"Cycle detected in dependencies: {_describe_cycle(remaining, predecessors)}"
        )
    return order


def demo():
    print(plan_execution(["build", "test", "deploy"],
                         [("build", "test"), ("test", "deploy")]))
    try:
        plan_execution(["a", "b"], [("a", "b"), ("b", "a")])
    except ValueError as exc:
        print("Error:", exc)


class TestPlanExecution(unittest.TestCase):
    def test_exam_sample(self):
        self.assertEqual(
            plan_execution(["build", "test", "deploy"], [("build", "test"), ("test", "deploy")]),
            ["build", "test", "deploy"])

    def test_alphabetical_choice_among_available(self):
        self.assertEqual(plan_execution(["c", "a", "b"], []), ["a", "b", "c"])
        self.assertEqual(plan_execution(["d", "c", "b", "a"], [("d", "a")]), ["b", "c", "d", "a"])

    def test_diamond(self):
        deps = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        self.assertEqual(plan_execution(list("dcba"), deps), ["a", "b", "c", "d"])

    def test_cycle(self):
        with self.assertRaisesRegex(ValueError, "Cycle detected"):
            plan_execution(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])

    def test_cycle_message_names_cycle_not_downstream(self):
        with self.assertRaises(ValueError) as ctx:
            plan_execution(["a", "b", "c"], [("a", "b"), ("b", "a"), ("b", "c")])
        self.assertIn("a -> b -> a", str(ctx.exception))

    def test_self_dependency_is_cycle(self):
        with self.assertRaisesRegex(ValueError, "Cycle"):
            plan_execution(["a"], [("a", "a")])

    def test_unknown_task(self):
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            plan_execution(["a"], [("a", "ghost")])

    def test_duplicate_tasks_and_bad_pairs(self):
        with self.assertRaises(ValueError):
            plan_execution(["a", "a"], [])
        with self.assertRaises(ValueError):
            plan_execution(["a", "b"], [("a",)])
        with self.assertRaises(ValueError):
            plan_execution(["a", ""], [])

    def test_empty(self):
        self.assertEqual(plan_execution([], []), [])

    def test_duplicate_dependency_pairs_ok(self):
        self.assertEqual(plan_execution(["a", "b"], [("a", "b"), ("a", "b")]), ["a", "b"])

    def test_disconnected_components(self):
        self.assertEqual(plan_execution(list("abcd"), [("b", "a"), ("d", "c")]),
                         ["b", "a", "d", "c"])

    def test_type_error(self):
        with self.assertRaises(TypeError):
            plan_execution("abc", [])


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
