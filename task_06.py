"""Task 06 - Banking Ledger with OOP.

Classes
-------
Transaction : frozen (immutable) dataclass - id, action, amount, balance_after
Account     : encapsulated balance (read-only property) + immutable history
Bank        : owns accounts, performs atomic transfers

Money is handled with Decimal (max 2 decimal places). Amounts must be > 0.
A transfer validates everything first and also restores snapshots if anything
unexpected fails half-way, so either both balances change or neither does.
"""
from __future__ import annotations

import itertools
import unittest
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from unittest import mock

CENT = Decimal("0.01")


class BankError(Exception):
    """Base class for all banking errors."""


class InvalidAmountError(BankError, ValueError):
    pass


class InsufficientFundsError(BankError):
    pass


class AccountNotFoundError(BankError):
    pass


def to_money(value, *, allow_zero=False):
    """Convert int/float/Decimal to a 2-dp Decimal or raise InvalidAmountError."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise InvalidAmountError(f"Amount must be a number, got {value!r}")
    amount = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    if not amount.is_finite():
        raise InvalidAmountError("Amount must be finite")
    if amount != amount.quantize(CENT):
        raise InvalidAmountError("Amount cannot have more than 2 decimal places")
    if amount < 0 or (amount == 0 and not allow_zero):
        raise InvalidAmountError("Amount must be positive")
    return amount.quantize(CENT)


@dataclass(frozen=True)
class Transaction:
    transaction_id: int
    action: str
    amount: Decimal
    balance_after: Decimal
    counterparty: Optional[str] = None


class Account:
    def __init__(self, account_id, owner, opening_balance=0, _id_source=None):
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("account_id must be a non-empty string")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("owner must be a non-empty string")
        self._id = account_id.strip()
        self._owner = owner.strip()
        self._balance = Decimal("0.00")
        self._transactions = []
        self._ids = _id_source if _id_source is not None else itertools.count(1)
        opening = to_money(opening_balance, allow_zero=True)
        if opening > 0:
            self._credit(opening, "open")

    # -- read-only views ---------------------------------------------------
    @property
    def account_id(self):
        return self._id

    @property
    def owner(self):
        return self._owner

    @property
    def balance(self):
        return self._balance

    @property
    def transactions(self):
        return tuple(self._transactions)     # tuple of frozen records

    # -- public operations -------------------------------------------------
    def deposit(self, amount):
        return self._credit(to_money(amount), "deposit")

    def withdraw(self, amount):
        amount = to_money(amount)
        if amount > self._balance:
            raise InsufficientFundsError(
                f"Insufficient funds in {self._id}: balance {self._balance}, requested {amount}")
        return self._debit(amount, "withdrawal")

    def statement(self):
        lines = [f"Statement for {self._id} ({self._owner})",
                 f"{'ID':>4}  {'Action':<13}{'Amount':>12}{'Balance':>12}  Counterparty"]
        for t in self._transactions:
            lines.append(f"{t.transaction_id:>4}  {t.action:<13}{t.amount:>12,.2f}"
                         f"{t.balance_after:>12,.2f}  {t.counterparty or ''}")
        lines.append(f"Closing balance: {self._balance:,.2f}")
        return "\n".join(lines)

    # -- internals (amounts already validated) -------------------------------
    def _record(self, action, amount, counterparty):
        txn = Transaction(next(self._ids), action, amount, self._balance, counterparty)
        self._transactions.append(txn)
        return txn

    def _credit(self, amount, action, counterparty=None):
        self._balance += amount
        return self._record(action, amount, counterparty)

    def _debit(self, amount, action, counterparty=None):
        self._balance -= amount
        return self._record(action, amount, counterparty)

    def _snapshot(self):
        return self._balance, len(self._transactions)

    def _restore(self, snapshot):
        self._balance, size = snapshot
        del self._transactions[size:]

    def __repr__(self):
        return f"Account({self._id!r}, owner={self._owner!r}, balance={self._balance})"


class Bank:
    def __init__(self):
        self._accounts = {}
        self._ids = itertools.count(1)       # bank-wide unique transaction IDs
        self._auto = itertools.count(1)

    def open_account(self, owner, opening_balance=0, account_id=None):
        account_id = account_id or f"ACC-{next(self._auto):04d}"
        if account_id in self._accounts:
            raise BankError(f"Account {account_id!r} already exists")
        account = Account(account_id, owner, opening_balance, _id_source=self._ids)
        self._accounts[account.account_id] = account
        return account

    def get_account(self, account_id):
        try:
            return self._accounts[account_id]
        except KeyError:
            raise AccountNotFoundError(f"Account {account_id!r} does not exist") from None

    def deposit(self, account_id, amount):
        return self.get_account(account_id).deposit(amount)

    def withdraw(self, account_id, amount):
        return self.get_account(account_id).withdraw(amount)

    def transfer(self, from_id, to_id, amount):
        source, target = self.get_account(from_id), self.get_account(to_id)
        if source is target:
            raise BankError("Cannot transfer to the same account")
        amount = to_money(amount)
        if amount > source.balance:
            raise InsufficientFundsError(
                f"Insufficient funds in {from_id}: balance {source.balance}, requested {amount}")

        # All checks passed. Snapshots guarantee all-or-nothing even on surprises.
        before = (source._snapshot(), target._snapshot())
        try:
            out = source._debit(amount, "transfer_out", counterparty=to_id)
            inc = target._credit(amount, "transfer_in", counterparty=from_id)
        except Exception:
            source._restore(before[0])
            target._restore(before[1])
            raise
        return out, inc

    def statement(self, account_id):
        return self.get_account(account_id).statement()


def demo():
    bank = Bank()
    bank.open_account("Alice", 1000, "A")
    bank.open_account("Bob", 200, "B")
    bank.transfer("A", "B", 350)
    print(bank.statement("A"))
    print(bank.statement("B"))


class TestBank(unittest.TestCase):
    def setUp(self):
        self.bank = Bank()
        self.a = self.bank.open_account("Alice", 1000, "A")
        self.b = self.bank.open_account("Bob", 200, "B")

    def test_exam_transfer(self):
        self.bank.transfer("A", "B", 350)
        self.assertEqual((self.a.balance, self.b.balance), (650, 550))

    def test_deposit_withdraw_and_history(self):
        self.a.deposit(50.25)
        self.a.withdraw(25)
        self.assertEqual(self.a.balance, Decimal("1025.25"))
        self.assertEqual([t.action for t in self.a.transactions], ["open", "deposit", "withdrawal"])
        self.assertEqual(self.a.transactions[-1].balance_after, Decimal("1025.25"))

    def test_rejects_bad_amounts(self):
        for bad in (0, -5, True, "10", None, float("nan"), float("inf"), 1.005):
            with self.subTest(bad=bad), self.assertRaises(InvalidAmountError):
                self.a.deposit(bad)
            with self.subTest(bad=bad), self.assertRaises(InvalidAmountError):
                self.a.withdraw(bad)
        self.assertEqual(self.a.balance, 1000)

    def test_insufficient_funds(self):
        with self.assertRaises(InsufficientFundsError):
            self.a.withdraw(1000.01)
        with self.assertRaises(InsufficientFundsError):
            self.bank.transfer("B", "A", 500)
        self.assertEqual((self.a.balance, self.b.balance), (1000, 200))

    def test_withdraw_entire_balance_allowed(self):
        self.a.withdraw(1000)
        self.assertEqual(self.a.balance, 0)

    def test_missing_accounts(self):
        with self.assertRaises(AccountNotFoundError):
            self.bank.transfer("A", "ZZZ", 10)
        with self.assertRaises(AccountNotFoundError):
            self.bank.transfer("ZZZ", "A", 10)
        with self.assertRaises(AccountNotFoundError):
            self.bank.deposit("ZZZ", 10)
        self.assertEqual(self.a.balance, 1000)

    def test_transfer_to_self_rejected(self):
        with self.assertRaises(BankError):
            self.bank.transfer("A", "A", 10)

    def test_transfer_atomic_when_second_leg_fails(self):
        with mock.patch.object(Account, "_credit", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.bank.transfer("A", "B", 100)
        self.assertEqual((self.a.balance, self.b.balance), (1000, 200))
        self.assertEqual(len(self.a.transactions), 1)       # only the opening record
        self.assertEqual(len(self.b.transactions), 1)

    def test_transactions_immutable(self):
        txn = self.a.transactions[0]
        with self.assertRaises(Exception):
            txn.amount = Decimal("1")                        # FrozenInstanceError
        with self.assertRaises(AttributeError):
            self.a.balance = 5                               # read-only property
        self.assertIsInstance(self.a.transactions, tuple)

    def test_unique_transaction_ids_across_accounts(self):
        self.bank.transfer("A", "B", 10)
        self.a.deposit(5)
        ids = [t.transaction_id for acct in (self.a, self.b) for t in acct.transactions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_duplicate_account_and_bad_opening(self):
        with self.assertRaises(BankError):
            self.bank.open_account("X", 0, "A")
        with self.assertRaises(InvalidAmountError):
            self.bank.open_account("X", -1)
        with self.assertRaises(ValueError):
            self.bank.open_account("  ")

    def test_statement_text(self):
        self.bank.transfer("A", "B", 350)
        text = self.bank.statement("A")
        self.assertIn("transfer_out", text)
        self.assertIn("Closing balance: 650.00", text)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
