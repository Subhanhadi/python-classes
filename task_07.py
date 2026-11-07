"""Task 07 - Employee Payroll Engine.

Abstract base class ``Employee`` with three concrete pay strategies:

* SalariedEmployee    - fixed salary for the pay period
* HourlyEmployee      - rate * hours, hours above 40 paid at 1.5x
* CommissionEmployee  - base + sales * rate

Employee cannot be instantiated directly (abstract ``calculate_pay``).
All money uses Decimal rounded half-up to 2 places.
"""
from __future__ import annotations

import unittest
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
OVERTIME_THRESHOLD = Decimal("40")
OVERTIME_MULTIPLIER = Decimal("1.5")
MAX_WEEKLY_HOURS = Decimal("168")


def _number(value, name, *, minimum=Decimal("0"), maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError(f"{name} must be finite")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _money(amount):
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


class Employee(ABC):
    employee_type = "Employee"

    def __init__(self, employee_id, name):
        if not isinstance(employee_id, (str, int)) or isinstance(employee_id, bool) or str(employee_id).strip() == "":
            raise ValueError("employee_id must be a non-empty string or int")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self.employee_id = employee_id
        self.name = name.strip()

    @abstractmethod
    def calculate_pay(self) -> Decimal:
        """Gross pay for one pay period."""

    def __repr__(self):
        return f"{type(self).__name__}(id={self.employee_id!r}, name={self.name!r})"


class SalariedEmployee(Employee):
    employee_type = "Salaried"

    def __init__(self, employee_id, name, salary):
        super().__init__(employee_id, name)
        self.salary = _number(salary, "salary")

    def calculate_pay(self):
        return _money(self.salary)


class HourlyEmployee(Employee):
    employee_type = "Hourly"

    def __init__(self, employee_id, name, hourly_rate, hours_worked):
        super().__init__(employee_id, name)
        self.hourly_rate = _number(hourly_rate, "hourly_rate")
        self.hours_worked = _number(hours_worked, "hours_worked", maximum=MAX_WEEKLY_HOURS)

    def calculate_pay(self):
        regular = min(self.hours_worked, OVERTIME_THRESHOLD)
        overtime = max(self.hours_worked - OVERTIME_THRESHOLD, Decimal("0"))
        return _money(self.hourly_rate * regular
                      + self.hourly_rate * OVERTIME_MULTIPLIER * overtime)


class CommissionEmployee(Employee):
    employee_type = "Commission"

    def __init__(self, employee_id, name, base_salary, sales, commission_rate):
        super().__init__(employee_id, name)
        self.base_salary = _number(base_salary, "base_salary")
        self.sales = _number(sales, "sales")
        self.commission_rate = _number(commission_rate, "commission_rate", maximum=Decimal("1"))

    def calculate_pay(self):
        return _money(self.base_salary + self.sales * self.commission_rate)


def payroll_summary(employees):
    """Polymorphic payroll run -> {'lines': [...], 'total': Decimal, 'count': int}."""
    lines, seen, total = [], set(), Decimal("0.00")
    for employee in employees:
        if not isinstance(employee, Employee):
            raise TypeError(f"Expected Employee, got {type(employee).__name__}")
        if employee.employee_id in seen:
            raise ValueError(f"Duplicate employee id: {employee.employee_id!r}")
        seen.add(employee.employee_id)
        pay = employee.calculate_pay()            # polymorphic call
        total += pay
        lines.append({"id": employee.employee_id, "name": employee.name,
                      "type": employee.employee_type, "pay": pay})
    return {"lines": lines, "total": total, "count": len(lines)}


def format_payroll(summary):
    rows = [f"{'ID':<8}{'Name':<18}{'Type':<12}{'Pay':>12}"]
    rows += [f"{str(l['id']):<8}{l['name']:<18}{l['type']:<12}{l['pay']:>12,.2f}"
             for l in summary["lines"]]
    rows.append(f"{'TOTAL':<38}{summary['total']:>12,.2f}")
    return "\n".join(rows)


def demo():
    staff = [SalariedEmployee(1, "Ayesha", 3000),
             HourlyEmployee(2, "Bilal", 20, 45),
             CommissionEmployee(3, "Carla", 500, 4000, 0.10)]
    print(format_payroll(payroll_summary(staff)))


class TestPayroll(unittest.TestCase):
    def test_exam_samples(self):
        self.assertEqual(HourlyEmployee(1, "H", 20, 45).calculate_pay(), 950)
        self.assertEqual(CommissionEmployee(2, "C", 500, 4000, .10).calculate_pay(), 900)

    def test_hourly_boundaries(self):
        self.assertEqual(HourlyEmployee(1, "H", 20, 0).calculate_pay(), 0)
        self.assertEqual(HourlyEmployee(1, "H", 20, 40).calculate_pay(), 800)
        self.assertEqual(HourlyEmployee(1, "H", 20, 40.5).calculate_pay(), 815)
        self.assertEqual(HourlyEmployee(1, "H", 15.5, 10).calculate_pay(), Decimal("155.00"))

    def test_salaried_and_rounding(self):
        self.assertEqual(SalariedEmployee(1, "S", 2500.555).calculate_pay(), Decimal("2500.56"))

    def test_commission_edges(self):
        self.assertEqual(CommissionEmployee(1, "C", 0, 0, 0).calculate_pay(), 0)
        self.assertEqual(CommissionEmployee(1, "C", 100, 1000, 1).calculate_pay(), 1100)
        with self.assertRaises(ValueError):
            CommissionEmployee(1, "C", 100, 1000, 1.5)

    def test_cannot_instantiate_abstract_employee(self):
        with self.assertRaises(TypeError):
            Employee(1, "Nope")

    def test_subclass_without_calculate_pay_is_abstract(self):
        class Broken(Employee):
            pass
        with self.assertRaises(TypeError):
            Broken(1, "x")

    def test_validation(self):
        with self.assertRaises(ValueError):
            SalariedEmployee(1, "S", -1)
        with self.assertRaises(TypeError):
            SalariedEmployee(1, "S", "3000")
        with self.assertRaises(TypeError):
            HourlyEmployee(1, "H", True, 10)
        with self.assertRaises(ValueError):
            HourlyEmployee(1, "H", 10, 200)
        with self.assertRaises(ValueError):
            SalariedEmployee(1, "  ", 100)
        with self.assertRaises(ValueError):
            SalariedEmployee("", "S", 100)
        with self.assertRaises(ValueError):
            SalariedEmployee(1, "S", float("nan"))

    def test_payroll_summary(self):
        staff = [SalariedEmployee(1, "A", 3000), HourlyEmployee(2, "B", 20, 45),
                 CommissionEmployee(3, "C", 500, 4000, .1)]
        summary = payroll_summary(staff)
        self.assertEqual(summary["total"], 4850)
        self.assertEqual(summary["count"], 3)
        self.assertEqual([l["type"] for l in summary["lines"]], ["Salaried", "Hourly", "Commission"])
        self.assertIn("TOTAL", format_payroll(summary))

    def test_payroll_empty_duplicates_and_wrong_types(self):
        self.assertEqual(payroll_summary([])["total"], 0)
        with self.assertRaises(ValueError):
            payroll_summary([SalariedEmployee(1, "A", 1), SalariedEmployee(1, "B", 1)])
        with self.assertRaises(TypeError):
            payroll_summary(["not an employee"])


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
