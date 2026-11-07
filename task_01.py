"""Task 01 - Nested Transaction Analyzer.

Builds a per-customer report from a list of transaction dictionaries.

Rules implemented
-----------------
* Only transactions whose status is "success" (case-insensitive) count.
* Duplicate transaction IDs are rejected with ValueError (checked across ALL
  transactions, including failed ones).
* Customers whose transactions all failed still appear with total 0.
* Result is sorted by total spending (descending), then customer name
  (ascending). ``rank`` is the 1-based position in that order.
* ``top_category`` is the category with the highest successful spending
  (ties broken alphabetically); ``None`` if the customer has no successes.
"""
from __future__ import annotations

import math
import unittest
from numbers import Real

REQUIRED_KEYS = frozenset({"id", "customer", "category", "amount", "status"})
SUCCESS_STATUS = "success"


def _validate_amount(amount, txn_id):
    if isinstance(amount, bool) or not isinstance(amount, Real):
        raise TypeError(f"Transaction {txn_id!r}: amount must be a number")
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"Transaction {txn_id!r}: amount must be finite and >= 0")
    return amount


def analyze_transactions(transactions):
    """Return a ranked list of dicts: customer, total, top_category, rank."""
    if transactions is None or isinstance(transactions, (str, bytes, dict)):
        raise TypeError("transactions must be a list of dictionaries")

    seen_ids = set()
    totals = {}            # customer -> total successful spending
    categories = {}        # customer -> {category: spending}

    for position, txn in enumerate(transactions):
        if not isinstance(txn, dict):
            raise TypeError(f"Transaction at position {position} must be a dict")
        missing = REQUIRED_KEYS - txn.keys()
        if missing:
            raise ValueError(
                f"Transaction at position {position} is missing keys: {sorted(missing)}"
            )

        txn_id = txn["id"]
        if txn_id in seen_ids:
            raise ValueError(f"Duplicate transaction id: {txn_id!r}")
        seen_ids.add(txn_id)

        customer = txn["customer"]
        if not isinstance(customer, str) or not customer.strip():
            raise ValueError(f"Transaction {txn_id!r}: customer must be a non-empty string")
        customer = customer.strip()

        # Every customer appears in the report, even with only failed payments.
        totals.setdefault(customer, 0)
        categories.setdefault(customer, {})

        if str(txn["status"]).strip().lower() != SUCCESS_STATUS:
            continue

        amount = _validate_amount(txn["amount"], txn_id)
        category = txn["category"]
        category = category.strip() if isinstance(category, str) and category.strip() else "Uncategorized"

        totals[customer] += amount
        categories[customer][category] = categories[customer].get(category, 0) + amount

    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))

    report = []
    for rank, (customer, total) in enumerate(ordered, start=1):
        by_category = categories[customer]
        top = (
            min(by_category.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if by_category
            else None
        )
        report.append(
            {"customer": customer, "total": total, "top_category": top, "rank": rank}
        )
    return report


def demo():
    sample = [
        {"id": 1, "customer": "Ali", "category": "Food", "amount": 500, "status": "success"},
        {"id": 2, "customer": "Ali", "category": "Travel", "amount": 200, "status": "failed"},
        {"id": 3, "customer": "Sara", "category": "Food", "amount": 700, "status": "success"},
    ]
    for row in analyze_transactions(sample):
        print(row)


class TestAnalyzeTransactions(unittest.TestCase):
    def txn(self, i, customer, category, amount, status="success"):
        return {"id": i, "customer": customer, "category": category,
                "amount": amount, "status": status}

    def test_sample_from_exam(self):
        data = [self.txn(1, "Ali", "Food", 500), self.txn(2, "Ali", "Travel", 200, "failed"),
                self.txn(3, "Sara", "Food", 700)]
        report = analyze_transactions(data)
        self.assertEqual([(r["customer"], r["total"]) for r in report],
                         [("Sara", 700), ("Ali", 500)])
        self.assertEqual([r["rank"] for r in report], [1, 2])

    def test_customer_with_only_failed_has_zero_total(self):
        report = analyze_transactions([self.txn(1, "Zed", "Food", 50, "failed")])
        self.assertEqual(report, [{"customer": "Zed", "total": 0,
                                   "top_category": None, "rank": 1}])

    def test_duplicate_ids_rejected_even_if_failed(self):
        with self.assertRaises(ValueError):
            analyze_transactions([self.txn(1, "A", "X", 1), self.txn(1, "B", "X", 1, "failed")])

    def test_tie_broken_by_name(self):
        report = analyze_transactions([self.txn(1, "Bob", "X", 10), self.txn(2, "Amy", "X", 10)])
        self.assertEqual([r["customer"] for r in report], ["Amy", "Bob"])

    def test_top_category_and_tie_break(self):
        data = [self.txn(1, "A", "Food", 30), self.txn(2, "A", "Travel", 90),
                self.txn(3, "B", "Zoo", 5), self.txn(4, "B", "Art", 5)]
        by_name = {r["customer"]: r for r in analyze_transactions(data)}
        self.assertEqual(by_name["A"]["top_category"], "Travel")
        self.assertEqual(by_name["B"]["top_category"], "Art")

    def test_empty_input(self):
        self.assertEqual(analyze_transactions([]), [])

    def test_status_case_insensitive(self):
        self.assertEqual(analyze_transactions([self.txn(1, "A", "X", 5, " SUCCESS ")])[0]["total"], 5)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            analyze_transactions([{"id": 1}])
        with self.assertRaises(ValueError):
            analyze_transactions([self.txn(1, "A", "X", -5)])
        with self.assertRaises(TypeError):
            analyze_transactions([self.txn(1, "A", "X", "10")])
        with self.assertRaises(TypeError):
            analyze_transactions(None)
        with self.assertRaises(TypeError):
            analyze_transactions(["not a dict"])

    def test_input_not_mutated(self):
        data = [self.txn(1, "A", "X", 5)]
        snapshot = [dict(d) for d in data]
        analyze_transactions(data)
        self.assertEqual(data, snapshot)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
