"""Task 15 - Integrated Order Fulfillment System.

Classes
-------
Product        frozen dataclass (id, name, price)
Inventory      products + stock, loaded from a JSON catalog; validates *before* changing
Cart           items + percentage discount, computes subtotal / discount / total
Order          immutable record of a placed order (prices copied at order time)
OrderStore     persists orders to a JSON file (atomic write, never silently discards data)
OrderProcessor places orders: validate everything -> deduct stock -> persist;
               if persistence fails the stock is rolled back and the error re-raised.

Money uses Decimal (ROUND_HALF_UP to 2 places). Order IDs are random and unique.

Catalog JSON format:
    [{"id": "P1", "name": "Keyboard", "price": 100, "stock": 5}, ...]
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data" / "task_15"
CENT = Decimal("0.01")


class OrderSystemError(Exception):
    pass


class CatalogError(OrderSystemError):
    pass


class InvalidQuantityError(OrderSystemError, ValueError):
    pass


class UnknownProductError(OrderSystemError):
    pass


class OutOfStockError(OrderSystemError):
    pass


class OrderPersistenceError(OrderSystemError):
    pass


def money(value, name="amount"):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{name} must be a number")
    try:
        amount = Decimal(str(value))
    except Exception:
        raise ValueError(f"{name} must be a number") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def _quantity(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidQuantityError(f"Quantity must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    price: Decimal


class Inventory:
    def __init__(self):
        self._products = {}
        self._stock = {}

    def add_product(self, product, stock):
        if product.product_id in self._products:
            raise CatalogError(f"Duplicate product id {product.product_id!r}")
        if isinstance(stock, bool) or not isinstance(stock, int) or stock < 0:
            raise CatalogError(f"Stock for {product.product_id!r} must be an integer >= 0")
        self._products[product.product_id] = product
        self._stock[product.product_id] = stock

    @classmethod
    def from_json(cls, path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise CatalogError(f"Catalog file not found: {path}") from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CatalogError(f"Catalog is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise CatalogError("Catalog must be a JSON list of products")
        inventory = cls()
        for entry in data:
            try:
                product = Product(str(entry["id"]).strip(), str(entry["name"]).strip(),
                                  money(entry["price"], "price"))
                if not product.product_id or not product.name:
                    raise ValueError("id and name must be non-empty")
                inventory.add_product(product, entry["stock"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CatalogError(f"Invalid catalog entry {entry!r}: {exc}") from exc
        return inventory

    def get_product(self, product_id):
        try:
            return self._products[product_id]
        except KeyError:
            raise UnknownProductError(f"Unknown product {product_id!r}") from None

    def stock_of(self, product_id):
        self.get_product(product_id)
        return self._stock[product_id]

    def check_available(self, requested):
        """Validate ALL requested items; raise one error describing every problem."""
        problems = []
        for product_id, qty in requested.items():
            if product_id not in self._products:
                problems.append(f"unknown product {product_id!r}")
            elif qty > self._stock[product_id]:
                problems.append(f"{product_id!r}: requested {qty}, only {self._stock[product_id]} in stock")
        if problems:
            raise OutOfStockError("Cannot fulfil order: " + "; ".join(problems))

    def deduct(self, requested):
        self.check_available(requested)             # all-or-nothing: validate first
        for product_id, qty in requested.items():
            self._stock[product_id] -= qty

    def snapshot(self):
        return dict(self._stock)

    def restore(self, snapshot):
        self._stock = dict(snapshot)


class Cart:
    def __init__(self, inventory):
        self._inventory = inventory
        self._items = {}
        self._discount = Decimal("0")

    def add(self, product_id, quantity):
        quantity = _quantity(quantity)
        self._inventory.get_product(product_id)
        self._items[product_id] = self._items.get(product_id, 0) + quantity

    def remove(self, product_id):
        if product_id not in self._items:
            raise UnknownProductError(f"{product_id!r} is not in the cart")
        del self._items[product_id]

    def clear(self):
        self._items.clear()

    def set_discount(self, percent):
        if isinstance(percent, bool) or not isinstance(percent, (int, float, str, Decimal)):
            raise ValueError("Discount must be a number")
        try:
            percent = Decimal(str(percent))
        except ArithmeticError:
            raise ValueError("Discount must be a number") from None
        if not percent.is_finite() or not Decimal(0) <= percent <= Decimal(100):
            raise ValueError("Discount must be between 0 and 100 percent")
        self._discount = percent

    @property
    def items(self):
        return dict(self._items)

    @property
    def discount_percent(self):
        return self._discount

    def is_empty(self):
        return not self._items

    def subtotal(self):
        return sum((self._inventory.get_product(p).price * q for p, q in self._items.items()),
                   Decimal("0.00")).quantize(CENT)

    def discount_amount(self):
        return (self.subtotal() * self._discount / Decimal(100)).quantize(CENT, rounding=ROUND_HALF_UP)

    def total(self):
        return self.subtotal() - self.discount_amount()


@dataclass(frozen=True)
class Order:
    order_id: str
    created_at: str
    lines: tuple              # ({"product_id", "name", "unit_price", "quantity", "line_total"}, ...)
    subtotal: Decimal
    discount_percent: Decimal
    discount_amount: Decimal
    total: Decimal

    def to_dict(self):
        return {"order_id": self.order_id, "created_at": self.created_at,
                "lines": [{**l, "unit_price": str(l["unit_price"]), "line_total": str(l["line_total"])}
                          for l in self.lines],
                "subtotal": str(self.subtotal), "discount_percent": str(self.discount_percent),
                "discount_amount": str(self.discount_amount), "total": str(self.total)}


class OrderStore:
    def __init__(self, path):
        self.path = Path(path)

    def load_all(self):
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OrderPersistenceError(f"Order file is corrupt, refusing to overwrite: {exc}") from exc
        if not isinstance(data, list):
            raise OrderPersistenceError("Order file must contain a JSON list")
        return data

    def order_ids(self):
        return {o.get("order_id") for o in self.load_all()}

    def append(self, order):
        orders = self.load_all()
        if any(o.get("order_id") == order.order_id for o in orders):
            raise OrderPersistenceError(f"Duplicate order id {order.order_id}")
        orders.append(order.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(orders, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise OrderPersistenceError(f"Could not save order: {exc}") from exc


class OrderProcessor:
    def __init__(self, inventory, store):
        self.inventory = inventory
        self.store = store

    def _new_order_id(self):
        existing = self.store.order_ids()
        while True:
            candidate = f"ORD-{uuid.uuid4().hex[:12].upper()}"
            if candidate not in existing:
                return candidate

    def place_order(self, cart):
        if cart.is_empty():
            raise InvalidQuantityError("Cannot place an order with an empty cart")
        requested = cart.items
        self.inventory.check_available(requested)             # 1. validate everything first

        lines = []
        for product_id, qty in requested.items():
            product = self.inventory.get_product(product_id)
            lines.append({"product_id": product_id, "name": product.name,
                          "unit_price": product.price, "quantity": qty,
                          "line_total": product.price * qty})
        order = Order(self._new_order_id(), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      tuple(lines), cart.subtotal(), cart.discount_percent,
                      cart.discount_amount(), cart.total())

        snapshot = self.inventory.snapshot()
        self.inventory.deduct(requested)                      # 2. update stock
        try:
            self.store.append(order)                          # 3. persist
        except Exception as exc:
            self.inventory.restore(snapshot)                  # roll back on ANY failure
            if isinstance(exc, OrderPersistenceError):
                raise
            raise OrderPersistenceError(f"Order could not be saved: {exc}") from exc
        cart.clear()
        return order


def create_sample_catalog(path=None):
    path = Path(path or SAMPLE_DIR / "catalog.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {"id": "P1", "name": "Mechanical Keyboard", "price": 100, "stock": 5},
        {"id": "P2", "name": "USB-C Cable", "price": 9.99, "stock": 50},
        {"id": "P3", "name": "Monitor Stand", "price": 45.5, "stock": 0},
    ], indent=2), encoding="utf-8")
    return path


def demo():
    catalog = create_sample_catalog()
    orders_path = SAMPLE_DIR / "orders.json"
    if orders_path.exists():
        orders_path.unlink()
    inventory = Inventory.from_json(catalog)
    cart = Cart(inventory)
    cart.add("P1", 2)
    cart.set_discount(10)
    order = OrderProcessor(inventory, OrderStore(orders_path)).place_order(cart)
    print(f"{order.order_id}: total={order.total}, P1 stock now {inventory.stock_of('P1')}")
    print("Orders saved to", orders_path)


class FailingStore:
    def order_ids(self):
        return set()

    def append(self, order):
        raise OSError("disk exploded")


class TestOrders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.inventory = Inventory()
        self.inventory.add_product(Product("P1", "Widget", Decimal("100.00")), 5)
        self.inventory.add_product(Product("P2", "Gadget", Decimal("9.99")), 3)
        self.store = OrderStore(self.dir / "orders.json")
        self.processor = OrderProcessor(self.inventory, self.store)

    def test_exam_sample_total_and_stock(self):
        cart = Cart(self.inventory)
        cart.add("P1", 2)
        cart.set_discount(10)
        order = self.processor.place_order(cart)
        self.assertEqual(order.total, Decimal("180.00"))
        self.assertEqual(self.inventory.stock_of("P1"), 3)
        saved = self.store.load_all()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["total"], "180.00")
        self.assertTrue(cart.is_empty())

    def test_out_of_stock_changes_nothing(self):
        cart = Cart(self.inventory)
        cart.add("P2", 1)
        cart.add("P1", 6)                                    # too many
        with self.assertRaises(OutOfStockError):
            self.processor.place_order(cart)
        self.assertEqual((self.inventory.stock_of("P1"), self.inventory.stock_of("P2")), (5, 3))
        self.assertEqual(self.store.load_all(), [])
        self.assertFalse(cart.is_empty())                    # cart kept so the user can adjust

    def test_persistence_failure_rolls_back_stock(self):
        cart = Cart(self.inventory)
        cart.add("P1", 2)
        cart.add("P2", 3)
        with self.assertRaises(OrderPersistenceError):
            OrderProcessor(self.inventory, FailingStore()).place_order(cart)
        self.assertEqual((self.inventory.stock_of("P1"), self.inventory.stock_of("P2")), (5, 3))
        self.assertFalse(cart.is_empty())

    def test_corrupt_orders_file_is_not_overwritten_and_rolls_back(self):
        (self.dir / "orders.json").write_text("{not json")
        cart = Cart(self.inventory)
        cart.add("P1", 1)
        with self.assertRaises(OrderPersistenceError):
            self.processor.place_order(cart)
        self.assertEqual((self.dir / "orders.json").read_text(), "{not json")
        self.assertEqual(self.inventory.stock_of("P1"), 5)

    def test_invalid_quantities(self):
        cart = Cart(self.inventory)
        for bad in (0, -1, 1.5, "2", None, True):
            with self.subTest(bad=bad), self.assertRaises(InvalidQuantityError):
                cart.add("P1", bad)
        with self.assertRaises(UnknownProductError):
            cart.add("NOPE", 1)
        with self.assertRaises(InvalidQuantityError):
            self.processor.place_order(cart)                 # empty cart

    def test_discount_validation_and_rounding(self):
        cart = Cart(self.inventory)
        cart.add("P2", 3)                                    # 29.97
        for bad in (-1, 100.1, "x", None, True, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cart.set_discount(bad)
        cart.set_discount(15)                                # 4.4955 -> 4.50
        self.assertEqual((cart.subtotal(), cart.discount_amount(), cart.total()),
                         (Decimal("29.97"), Decimal("4.50"), Decimal("25.47")))
        cart.set_discount(100)
        self.assertEqual(cart.total(), Decimal("0.00"))

    def test_cart_accumulates_and_combined_stock_is_checked(self):
        cart = Cart(self.inventory)
        cart.add("P1", 3)
        cart.add("P1", 3)                                    # now 6 > stock 5
        self.assertEqual(cart.items, {"P1": 6})
        with self.assertRaises(OutOfStockError):
            self.processor.place_order(cart)

    def test_unique_order_ids_and_appending(self):
        ids = set()
        for _ in range(5):
            cart = Cart(self.inventory)
            cart.add("P1", 1)
            ids.add(self.processor.place_order(cart).order_id)
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(self.store.load_all()), 5)
        self.assertEqual(self.inventory.stock_of("P1"), 0)

    def test_order_is_immutable_snapshot_of_prices(self):
        cart = Cart(self.inventory)
        cart.add("P1", 1)
        order = self.processor.place_order(cart)
        with self.assertRaises(Exception):
            order.total = Decimal("0")
        self.assertEqual(order.lines[0]["unit_price"], Decimal("100.00"))

    def test_catalog_loading(self):
        inventory = Inventory.from_json(create_sample_catalog(self.dir / "catalog.json"))
        self.assertEqual(inventory.stock_of("P1"), 5)
        self.assertEqual(inventory.get_product("P2").price, Decimal("9.99"))
        bad_files = {
            "missing": None,
            "notjson": "{{",
            "notlist": '{"a": 1}',
            "dup": '[{"id":"A","name":"x","price":1,"stock":1},{"id":"A","name":"y","price":1,"stock":1}]',
            "negprice": '[{"id":"A","name":"x","price":-1,"stock":1}]',
            "badstock": '[{"id":"A","name":"x","price":1,"stock":-2}]',
            "nokey": '[{"id":"A","name":"x","price":1}]',
        }
        for label, content in bad_files.items():
            path = self.dir / f"{label}.json"
            if content is not None:
                path.write_text(content)
            with self.subTest(label=label), self.assertRaises(CatalogError):
                Inventory.from_json(path)

    def test_unknown_product_in_stock_lookup(self):
        with self.assertRaises(UnknownProductError):
            self.inventory.stock_of("ZZZ")


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
