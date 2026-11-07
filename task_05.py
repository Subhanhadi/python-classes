"""Task 05 - Expression Validator and Evaluator.

Integer expressions with + - * / ( ) and whitespace, evaluated with the
shunting-yard algorithm using two explicit stacks (values and operators).
No eval()/exec() and no third-party parsers.

* standard precedence (* / above + -), left-associative
* unary minus / plus supported (e.g. "-(3 + 2)", "2 * -3", "--4")
* malformed input -> ExpressionError (a ValueError subclass)
* division by zero -> ZeroDivisionError
* the result is always returned as a float (e.g. -14.0)
"""
from __future__ import annotations

import unittest

DIGITS = "0123456789"
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "pos": 3}
PREFIX_OPS = {"neg", "pos"}                 # right-associative, one operand


class ExpressionError(ValueError):
    """Raised for syntactically invalid expressions."""


def tokenize(expression):
    tokens, i, n = [], 0, len(expression)
    while i < n:
        ch = expression[i]
        if ch.isspace():
            i += 1
        elif ch in DIGITS:
            j = i
            while j < n and expression[j] in DIGITS:
                j += 1
            if j < n and expression[j] == ".":
                raise ExpressionError(f"Only integers are allowed (decimal point at position {j})")
            tokens.append(("num", int(expression[i:j])))
            i = j
        elif ch in "+-*/":
            tokens.append(("op", ch))
            i += 1
        elif ch in "()":
            tokens.append(("paren", ch))
            i += 1
        else:
            raise ExpressionError(f"Invalid character {ch!r} at position {i}")
    return tokens


def _apply(operator, values):
    if operator in PREFIX_OPS:
        if not values:
            raise ExpressionError("Missing operand")
        if operator == "neg":
            values.append(-values.pop())
        return
    if len(values) < 2:
        raise ExpressionError(f"Missing operand for {operator!r}")
    right, left = values.pop(), values.pop()
    if operator == "+":
        values.append(left + right)
    elif operator == "-":
        values.append(left - right)
    elif operator == "*":
        values.append(left * right)
    else:
        if right == 0:
            raise ZeroDivisionError("Division by zero")
        values.append(left / right)


def evaluate(expression):
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")

    values, operators = [], []
    expect_operand = True                   # tiny state machine validates syntax

    for kind, value in tokenize(expression):
        if expect_operand:
            if kind == "num":
                values.append(value)
                expect_operand = False
            elif kind == "paren" and value == "(":
                operators.append("(")
            elif kind == "op" and value in "+-":
                operators.append("neg" if value == "-" else "pos")
            else:
                raise ExpressionError(f"Unexpected {value!r}: expected a number, '(' or a sign")
        else:
            if kind == "op":
                while (operators and operators[-1] != "("
                       and PRECEDENCE[operators[-1]] >= PRECEDENCE[value]):
                    _apply(operators.pop(), values)
                operators.append(value)
                expect_operand = True
            elif kind == "paren" and value == ")":
                while operators and operators[-1] != "(":
                    _apply(operators.pop(), values)
                if not operators:
                    raise ExpressionError("Unbalanced parentheses: unexpected ')'")
                operators.pop()             # discard the matching "("
            else:
                raise ExpressionError(f"Missing operator before {value!r}")

    if expect_operand:
        raise ExpressionError("Incomplete expression" if values or operators else "Empty expression")

    while operators:
        operator = operators.pop()
        if operator == "(":
            raise ExpressionError("Unbalanced parentheses: missing ')'")
        _apply(operator, values)

    if len(values) != 1:                    # defensive; syntax checks make this unreachable
        raise ExpressionError("Malformed expression")
    return float(values[0])


def demo():
    for expr in ["-(3 + 2) * 4 + 18 / 3", "2 + 3 * 4", "10 - 4 - 3", "(1 + 2) * (3 + 4)"]:
        print(f"{expr} = {evaluate(expr)}")
    for bad in ["1 / 0", "3 +", "(3 + 4"]:
        try:
            evaluate(bad)
        except (ExpressionError, ZeroDivisionError) as exc:
            print(f"{bad!r} -> {type(exc).__name__}: {exc}")


class TestEvaluate(unittest.TestCase):
    def test_exam_sample(self):
        self.assertEqual(evaluate("-(3 + 2) * 4 + 18 / 3"), -14.0)

    def test_precedence_and_associativity(self):
        self.assertEqual(evaluate("2+3*4"), 14.0)
        self.assertEqual(evaluate("10-4-3"), 3.0)
        self.assertEqual(evaluate("100/10/5"), 2.0)
        self.assertEqual(evaluate("(2+3)*4"), 20.0)
        self.assertEqual(evaluate("2*(3+(4-1))*2"), 24.0)

    def test_unary(self):
        self.assertEqual(evaluate("-3"), -3.0)
        self.assertEqual(evaluate("--3"), 3.0)
        self.assertEqual(evaluate("2 * -3"), -6.0)
        self.assertEqual(evaluate("-3 * 2"), -6.0)
        self.assertEqual(evaluate("3 - -2"), 5.0)
        self.assertEqual(evaluate("+4 + +1"), 5.0)
        self.assertEqual(evaluate("-(-(2))"), 2.0)

    def test_whitespace_and_float_result(self):
        self.assertEqual(evaluate("  7 \t/ 2 "), 3.5)
        self.assertIsInstance(evaluate("1+1"), float)

    def test_division_by_zero(self):
        for expr in ("1/0", "5/(2-2)", "0/0"):
            with self.assertRaises(ZeroDivisionError):
                evaluate(expr)

    def test_malformed(self):
        bad = ["", "   ", "3 +", "+", "*3", "3 * * 4", "3 4", "(3+4", "3+4)", "()", "(+)",
               "2 (3)", "(2)(3)", "abc", "3 & 4", "3.5 + 1", "3 + a", "((1)", "1 2 +"]
        for expr in bad:
            with self.subTest(expr=expr), self.assertRaises(ExpressionError):
                evaluate(expr)

    def test_error_is_value_error(self):
        with self.assertRaises(ValueError):
            evaluate("3 +")

    def test_type_error(self):
        with self.assertRaises(TypeError):
            evaluate(42)

    def test_large_integers(self):
        self.assertEqual(evaluate("99999999999 * 99999999999"), float(99999999999 ** 2))

    def test_agrees_with_python_on_safe_expressions(self):
        for expr in ["1+2*3-4/2", "(1+2)*(3-4)/5", "-(2*3)+-4", "8/2/2*3"]:
            self.assertAlmostEqual(evaluate(expr), float(eval(expr)))  # test oracle only


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
