"""Task 10 - CSV Inventory Reconciliation.

Reads inventory.csv (sku,name,quantity) and movements.csv (sku,type,units),
applies IN/OUT movements in file order and writes:

* updated_inventory.csv   - sku,name,quantity (original SKU order)
* rejected_movements.csv  - line_number,sku,type,units,reason

A movement is rejected (and processing continues) when:
  unknown SKU | type not IN/OUT | units not a positive integer | OUT would make stock negative
Problems in inventory.csv itself (missing columns, duplicate SKU, bad quantity)
raise ValueError because reconciling against corrupt master data is unsafe.
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data" / "task_10"
INVENTORY_COLUMNS = ["sku", "name", "quantity"]
MOVEMENT_COLUMNS = ["sku", "type", "units"]
REJECT_COLUMNS = ["line_number", "sku", "type", "units", "reason"]


def _read_csv(path, required):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [f.strip().lower() for f in (reader.fieldnames or [])]
        missing = [c for c in required if c not in fields]
        if missing:
            raise ValueError(f"{Path(path).name}: missing required column(s): {missing}")
        reader.fieldnames = fields
        for row in reader:
            yield reader.line_num, {k: (v or "").strip() for k, v in row.items() if k is not None}


def load_inventory(path):
    """Return an ordered dict: sku -> {'name': str, 'quantity': int}."""
    inventory = {}
    for line, row in _read_csv(path, INVENTORY_COLUMNS):
        if not row["sku"]:
            raise ValueError(f"{Path(path).name} line {line}: empty SKU")
        if row["sku"] in inventory:
            raise ValueError(f"{Path(path).name} line {line}: duplicate SKU {row['sku']!r}")
        try:
            quantity = int(row["quantity"])
        except ValueError:
            raise ValueError(f"{Path(path).name} line {line}: invalid quantity {row['quantity']!r}") from None
        if quantity < 0:
            raise ValueError(f"{Path(path).name} line {line}: negative quantity")
        inventory[row["sku"]] = {"name": row["name"], "quantity": quantity}
    return inventory


def reconcile_inventory(inventory_path, movements_path,
                        updated_path="updated_inventory.csv",
                        rejected_path="rejected_movements.csv"):
    inventory = load_inventory(inventory_path)
    rejected, applied = [], 0

    for line, row in _read_csv(movements_path, MOVEMENT_COLUMNS):
        sku, kind = row["sku"], row["type"].upper()
        reason = None
        units = None
        try:
            units = int(row["units"])
        except ValueError:
            pass

        if sku not in inventory:
            reason = "unknown SKU"
        elif kind not in ("IN", "OUT"):
            reason = "invalid movement type"
        elif units is None:
            reason = "units is not an integer"
        elif units <= 0:
            reason = "units must be positive"
        elif kind == "OUT" and inventory[sku]["quantity"] < units:
            reason = (f"insufficient stock (available {inventory[sku]['quantity']}, "
                      f"requested {units})")

        if reason:
            rejected.append({"line_number": line, "sku": row["sku"], "type": row["type"],
                             "units": row["units"], "reason": reason})
            continue
        inventory[sku]["quantity"] += units if kind == "IN" else -units
        applied += 1

    for path, columns, rows in (
        (updated_path, INVENTORY_COLUMNS,
         [{"sku": s, "name": d["name"], "quantity": d["quantity"]} for s, d in inventory.items()]),
        (rejected_path, REJECT_COLUMNS, rejected),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    return {"applied": applied, "rejected": len(rejected),
            "final_quantities": {s: d["quantity"] for s, d in inventory.items()}}


def create_sample_files(directory=SAMPLE_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inventory.csv").write_text(
        "sku,name,quantity\nP01,Widget,10\nP02,Gadget,5\nP03,Gizmo,0\n", encoding="utf-8")
    (directory / "movements.csv").write_text(
        "sku,type,units\nP01,OUT,4\nP01,OUT,9\nP01,IN,2\nP99,IN,5\nP02,IN,0\n"
        "P02,SWAP,3\nP02,OUT,5\nP03,IN,-4\nP03,in,7\nP03,OUT,abc\n", encoding="utf-8")
    return directory


def demo():
    folder = create_sample_files()
    result = reconcile_inventory(folder / "inventory.csv", folder / "movements.csv",
                                 folder / "updated_inventory.csv", folder / "rejected_movements.csv")
    print(result)
    print((folder / "updated_inventory.csv").read_text())
    print((folder / "rejected_movements.csv").read_text())


class TestReconcile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_case(self, inventory, movements):
        (self.dir / "inventory.csv").write_text(inventory, encoding="utf-8")
        (self.dir / "movements.csv").write_text(movements, encoding="utf-8")
        result = reconcile_inventory(self.dir / "inventory.csv", self.dir / "movements.csv",
                                     self.dir / "u.csv", self.dir / "r.csv")
        with open(self.dir / "r.csv", newline="") as fh:
            rejected = list(csv.DictReader(fh))
        with open(self.dir / "u.csv", newline="") as fh:
            updated = list(csv.DictReader(fh))
        return result, updated, rejected

    def test_exam_sample(self):
        result, updated, rejected = self.run_case(
            "sku,name,quantity\nP01,Widget,10\n", "sku,type,units\nP01,OUT,4\nP01,OUT,9\nP01,IN,2\n")
        self.assertEqual(updated[0]["quantity"], "8")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["line_number"], "3")
        self.assertIn("insufficient stock", rejected[0]["reason"])
        self.assertEqual((result["applied"], result["rejected"]), (2, 1))

    def test_all_rejection_reasons_and_continue_processing(self):
        result, updated, rejected = self.run_case(
            "sku,name,quantity\nA,Alpha,5\n",
            "sku,type,units\nZ,IN,1\nA,SWAP,1\nA,IN,0\nA,IN,-3\nA,IN,x\nA,IN,2.5\nA,OUT,6\nA,IN,3\n")
        reasons = [r["reason"] for r in rejected]
        self.assertEqual(len(rejected), 7)
        self.assertEqual(reasons[0], "unknown SKU")
        self.assertEqual(reasons[1], "invalid movement type")
        self.assertEqual(updated[0]["quantity"], "8")          # the final valid IN 3 was applied

    def test_out_to_exactly_zero_allowed(self):
        _, updated, rejected = self.run_case("sku,name,quantity\nA,Alpha,5\n", "sku,type,units\nA,OUT,5\n")
        self.assertEqual((updated[0]["quantity"], rejected), ("0", []))

    def test_case_and_whitespace_tolerant(self):
        _, updated, rejected = self.run_case("sku,name,quantity\nA,Alpha,1\n", "sku,type,units\n A , in , 4 \n")
        self.assertEqual(updated[0]["quantity"], "5")

    def test_order_matters(self):
        _, updated, _ = self.run_case("sku,name,quantity\nA,Alpha,0\n", "sku,type,units\nA,IN,5\nA,OUT,5\n")
        self.assertEqual(updated[0]["quantity"], "0")
        _, updated, rejected = self.run_case("sku,name,quantity\nA,Alpha,0\n", "sku,type,units\nA,OUT,5\nA,IN,5\n")
        self.assertEqual((updated[0]["quantity"], len(rejected)), ("5", 1))

    def test_sku_order_preserved_and_empty_movements(self):
        _, updated, rejected = self.run_case("sku,name,quantity\nB,Beta,1\nA,Alpha,2\n", "sku,type,units\n")
        self.assertEqual([u["sku"] for u in updated], ["B", "A"])
        self.assertEqual(rejected, [])

    def test_bad_master_data_raises(self):
        with self.assertRaises(ValueError):
            self.run_case("sku,name,quantity\nA,x,1\nA,y,2\n", "sku,type,units\n")
        with self.assertRaises(ValueError):
            self.run_case("sku,name,quantity\nA,x,abc\n", "sku,type,units\n")
        with self.assertRaises(ValueError):
            self.run_case("sku,name,quantity\nA,x,-1\n", "sku,type,units\n")
        with self.assertRaises(ValueError):
            self.run_case("sku,name\nA,x\n", "sku,type,units\n")
        with self.assertRaises(ValueError):
            self.run_case("sku,name,quantity\nA,x,1\n", "sku,units\nA,1\n")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            reconcile_inventory(self.dir / "none.csv", self.dir / "none2.csv")

    def test_sample_generator(self):
        folder = create_sample_files(self.dir / "sample")
        result = reconcile_inventory(folder / "inventory.csv", folder / "movements.csv",
                                     folder / "u.csv", folder / "r.csv")
        self.assertEqual(result["final_quantities"], {"P01": 8, "P02": 0, "P03": 7})
        self.assertEqual(result["rejected"], 6)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
