"""Task 14 - Command-Line Contact Manager.

Layers (kept separate on purpose):
    JsonContactStorage  - reads/writes the JSON file (atomic writes, corruption checks)
    ContactService      - business logic: validation, CRUD, search
    run_cli()           - menu-driven user interface (input/output injectable for tests)

Safety rules
------------
* Missing or empty file  -> start with no contacts (file is created on first save).
* Corrupt / invalid JSON -> StorageError. The file is NEVER overwritten or
  deleted; the user is told to repair or move it.
* Saves go to a temp file first and are swapped in with os.replace(), so a crash
  cannot leave a half-written contacts file. The in-memory list is only updated
  after the save succeeded.
* Search is a case-insensitive substring match on the name.
* IDs are auto-incremented (max existing ID + 1) and unique.

Run interactively:  python task_14.py --cli [contacts.json]
Run demo + tests :  python task_14.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data" / "task_14"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?\d{7,15}$")
FIELDS = ("name", "phone", "email")


class StorageError(Exception):
    """The contacts file cannot be read or written safely."""


class ValidationError(ValueError):
    """User supplied invalid contact data."""


class ContactNotFoundError(LookupError):
    pass


# ----------------------------------------------------------------------------
# Storage layer
# ----------------------------------------------------------------------------
class JsonContactStorage:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StorageError(f"Cannot read {self.path}: {exc}") from exc
        if not text.strip():
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"{self.path} is not valid JSON ({exc}). The file was left untouched; "
                f"fix or move it and try again.") from exc
        return self._validate(data)

    def _validate(self, data):
        problem = None
        if not isinstance(data, list):
            problem = "top-level value must be a list"
        else:
            seen = set()
            for item in data:
                if not (isinstance(item, dict) and isinstance(item.get("id"), int)
                        and not isinstance(item.get("id"), bool)
                        and all(isinstance(item.get(f), str) for f in FIELDS)):
                    problem = f"invalid contact record: {item!r}"
                    break
                if item["id"] in seen:
                    problem = f"duplicate id {item['id']}"
                    break
                seen.add(item["id"])
        if problem:
            raise StorageError(f"{self.path} has an unexpected structure ({problem}). "
                               f"The file was left untouched.")
        return data

    def save(self, contacts):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(contacts, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except OSError as exc:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise StorageError(f"Cannot write {self.path}: {exc}") from exc


# ----------------------------------------------------------------------------
# Business logic
# ----------------------------------------------------------------------------
def clean_name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Name is required")
    value = " ".join(value.split())
    if len(value) > 100:
        raise ValidationError("Name must be at most 100 characters")
    return value


def clean_phone(value):
    if not isinstance(value, str):
        raise ValidationError("Phone must be text")
    value = value.strip()
    if not PHONE_RE.match(re.sub(r"[\s\-()]", "", value)):
        raise ValidationError("Phone must contain 7-15 digits (optional leading +)")
    return value


def clean_email(value):
    if not isinstance(value, str) or not EMAIL_RE.match(value.strip()):
        raise ValidationError("Email must look like name@example.com")
    return value.strip()


CLEANERS = {"name": clean_name, "phone": clean_phone, "email": clean_email}


class ContactService:
    def __init__(self, storage):
        self.storage = storage
        self._contacts = storage.load()          # raises StorageError if unsafe

    def _commit(self, new_contacts):
        self.storage.save(new_contacts)          # persist first ...
        self._contacts = new_contacts            # ... then update memory

    def _find(self, contact_id):
        for contact in self._contacts:
            if contact["id"] == contact_id:
                return contact
        raise ContactNotFoundError(f"No contact with id {contact_id}")

    def add(self, name, phone, email):
        contact = {"id": max((c["id"] for c in self._contacts), default=0) + 1,
                   "name": clean_name(name), "phone": clean_phone(phone), "email": clean_email(email)}
        self._commit(self._contacts + [contact])
        return dict(contact)

    def get(self, contact_id):
        return dict(self._find(contact_id))

    def list_all(self):
        return [dict(c) for c in sorted(self._contacts, key=lambda c: c["id"])]

    def search(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("Search text is required")
        needle = query.strip().casefold()
        return [dict(c) for c in self._contacts if needle in c["name"].casefold()]

    def update(self, contact_id, **changes):
        unknown = set(changes) - set(FIELDS)
        if unknown:
            raise ValidationError(f"Unknown field(s): {sorted(unknown)}")
        changes = {k: v for k, v in changes.items() if v is not None}
        if not changes:
            raise ValidationError("Nothing to update")
        current = self._find(contact_id)
        updated = {**current, **{k: CLEANERS[k](v) for k, v in changes.items()}}
        self._commit([updated if c["id"] == contact_id else c for c in self._contacts])
        return dict(updated)

    def delete(self, contact_id):
        removed = self._find(contact_id)
        self._commit([c for c in self._contacts if c["id"] != contact_id])
        return dict(removed)


# ----------------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------------
MENU = """
==== Contact Manager ====
1) Add contact
2) Search by name
3) Update contact
4) Delete contact
5) List all contacts
0) Exit"""


def _format(contact):
    return f"[{contact['id']}] {contact['name']} | {contact['phone']} | {contact['email']}"


def _ask_id(ask):
    text = ask("Contact ID: ").strip()
    if not text.isdigit():
        raise ValidationError("ID must be a whole number")
    return int(text)


def run_cli(service, input_fn=input, output_fn=print):
    while True:
        output_fn(MENU)
        try:
            choice = input_fn("Choose an option: ").strip()
            if choice == "0":
                output_fn("Goodbye!")
                return
            elif choice == "1":
                contact = service.add(input_fn("Name: "), input_fn("Phone: "), input_fn("Email: "))
                output_fn(f"Added: {_format(contact)}")
            elif choice == "2":
                matches = service.search(input_fn("Search name: "))
                output_fn("\n".join(_format(c) for c in matches) if matches else "No matches.")
            elif choice == "3":
                contact_id = _ask_id(input_fn)
                output_fn(f"Current: {_format(service.get(contact_id))}")
                output_fn("Press Enter to keep a value.")
                changes = {f: (input_fn(f"New {f}: ").strip() or None) for f in FIELDS}
                output_fn(f"Updated: {_format(service.update(contact_id, **changes))}")
            elif choice == "4":
                contact_id = _ask_id(input_fn)
                if input_fn(f"Delete {_format(service.get(contact_id))}? (y/N): ").strip().lower() == "y":
                    service.delete(contact_id)
                    output_fn("Deleted.")
                else:
                    output_fn("Cancelled.")
            elif choice == "5":
                contacts = service.list_all()
                output_fn("\n".join(_format(c) for c in contacts) if contacts else "No contacts yet.")
            else:
                output_fn("Invalid option, please choose 0-5.")
        except (ValidationError, ContactNotFoundError, StorageError) as exc:
            output_fn(f"Error: {exc}")
        except EOFError:
            output_fn("\nInput closed. Goodbye!")
            return


def main(path):
    try:
        service = ContactService(JsonContactStorage(path))
    except StorageError as exc:
        print(f"Cannot start: {exc}")
        return 1
    run_cli(service)
    return 0


def demo():
    path = SAMPLE_DIR / "contacts.json"
    if path.exists():
        path.unlink()
    service = ContactService(JsonContactStorage(path))
    service.add("Ayesha", "+92 300 1234567", "ayesha@example.com")
    restarted = ContactService(JsonContactStorage(path))          # "restart the application"
    print("search 'aye' after restart ->", restarted.search("aye"))
    print("Sample data written to", path)


class TestContactManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "contacts.json"

    def service(self):
        return ContactService(JsonContactStorage(self.path))

    def test_exam_persistence_and_case_insensitive_search(self):
        s = self.service()
        added = s.add("Ayesha", "03001234567", "ayesha@example.com")
        self.assertEqual(added["id"], 1)
        again = self.service()                                  # restart
        self.assertEqual([c["name"] for c in again.search("aye")], ["Ayesha"])
        self.assertEqual(len(again.search("AYESHA")), 1)
        self.assertEqual(again.search("zzz"), [])

    def test_missing_and_empty_file_start_empty(self):
        self.assertEqual(self.service().list_all(), [])
        self.path.write_text("   ")
        self.assertEqual(self.service().list_all(), [])

    def test_corrupt_json_is_not_overwritten(self):
        original = '{"id": 1, "name": "broken'
        self.path.write_text(original)
        with self.assertRaises(StorageError):
            self.service()
        self.assertEqual(self.path.read_text(), original)

    def test_wrong_structure_is_rejected_and_untouched(self):
        for payload in ('{"a": 1}', '[{"id": 1}]', '[{"id": 1, "name": "A", "phone": "1234567", "email": "a@b.co"},'
                        '{"id": 1, "name": "B", "phone": "1234567", "email": "b@b.co"}]',
                        '[{"id": true, "name": "A", "phone": "1234567", "email": "a@b.co"}]'):
            self.path.write_text(payload)
            with self.assertRaises(StorageError):
                self.service()
            self.assertEqual(self.path.read_text(), payload)

    def test_validation(self):
        s = self.service()
        for args in [("", "1234567", "a@b.co"), ("A", "abc", "a@b.co"), ("A", "123", "a@b.co"),
                     ("A", "1234567", "not-an-email"), ("A", "1234567", ""), (None, "1234567", "a@b.co")]:
            with self.subTest(args=args), self.assertRaises(ValidationError):
                s.add(*args)
        self.assertEqual(s.list_all(), [])
        self.assertFalse(self.path.exists())                    # nothing persisted on failure

    def test_update_and_delete(self):
        s = self.service()
        s.add("Ali", "+923001234567", "ali@x.com")
        s.add("Sara", "03111234567", "sara@x.com")
        s.update(2, phone="0300 9999999", name=None)
        self.assertEqual(self.service().get(2)["phone"], "0300 9999999")
        self.assertEqual(self.service().get(2)["name"], "Sara")
        with self.assertRaises(ValidationError):
            s.update(2)
        with self.assertRaises(ValidationError):
            s.update(2, email="bad")
        with self.assertRaises(ValidationError):
            s.update(2, nickname="x")
        with self.assertRaises(ContactNotFoundError):
            s.update(99, name="X")
        s.delete(1)
        self.assertEqual([c["id"] for c in self.service().list_all()], [2])
        with self.assertRaises(ContactNotFoundError):
            s.delete(1)

    def test_ids_unique_and_increasing(self):
        s = self.service()
        ids = [s.add(f"P{i}", "1234567", f"p{i}@x.com")["id"] for i in range(3)]
        s.delete(2)
        ids.append(s.add("New", "1234567", "n@x.com")["id"])
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_failed_save_keeps_memory_and_disk_consistent(self):
        s = self.service()
        s.add("Ali", "1234567", "ali@x.com")
        s.storage.save = lambda contacts: (_ for _ in ()).throw(StorageError("disk full"))
        with self.assertRaises(StorageError):
            s.add("Sara", "1234567", "sara@x.com")
        self.assertEqual(len(s.list_all()), 1)

    def test_unicode_names_survive(self):
        s = self.service()
        s.add("عائشہ", "1234567", "a@b.co")
        self.assertEqual(self.service().search("عائ")[0]["name"], "عائشہ")

    def test_search_requires_text(self):
        with self.assertRaises(ValidationError):
            self.service().search("  ")

    def run_script(self, answers):
        feed, out = iter(answers), []

        def read(prompt=""):
            try:
                return next(feed)
            except StopIteration:
                raise EOFError from None

        run_cli(self.service(), input_fn=read, output_fn=out.append)
        return "\n".join(out)

    def test_cli_flow(self):
        text = self.run_script([
            "1", "Ayesha", "03001234567", "ayesha@example.com",     # add
            "1", "Bad", "xx", "a@b.co",                              # invalid phone -> error, continues
            "2", "AYE",                                              # search
            "3", "1", "", "", "new@example.com",                     # update email only
            "5",                                                     # list
            "4", "1", "n",                                           # delete cancelled
            "4", "1", "y",                                           # delete confirmed
            "7",                                                     # invalid menu choice
            "0"])
        self.assertIn("Added: [1] Ayesha", text)
        self.assertIn("Error: Phone must contain", text)
        self.assertIn("new@example.com", text)
        self.assertIn("Cancelled.", text)
        self.assertIn("Deleted.", text)
        self.assertIn("Invalid option", text)
        self.assertEqual(self.service().list_all(), [])

    def test_cli_handles_bad_id_and_eof(self):
        text = self.run_script(["3", "abc", "4", "42", "5"])        # input ends -> EOFError (StopIteration)
        self.assertIn("ID must be a whole number", text)

    def test_cli_eof_exits_cleanly(self):
        def raise_eof(prompt=""):
            raise EOFError
        out = []
        run_cli(self.service(), input_fn=raise_eof, output_fn=out.append)
        self.assertIn("Goodbye", "\n".join(out))


if __name__ == "__main__":
    if "--cli" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--cli"]
        raise SystemExit(main(args[0] if args else SAMPLE_DIR / "contacts.json"))
    demo()
    unittest.main(verbosity=1)
