"""Task 11 - Multi-File Student Gradebook.

Merges students.csv (student_id,name) with scores.csv (student_id,subject,marks)
and exports report.csv: rank, student_id, name, subjects, average, grade.

Rules
-----
* marks must be numeric and within 0-100 (inclusive)
* unknown student IDs and duplicate student-subject entries (case-insensitive
  subject; the FIRST entry is kept) are rejected and written to
  rejected_scores.csv with a reason; processing continues
* grades: A >= 85, B >= 70, C >= 55, D >= 40, otherwise F
* rank: average descending, then student ID (numeric IDs sort numerically)
* students with no valid scores are listed with grade "N/A" and no rank
* duplicate IDs inside students.csv are a data error -> ValueError
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from decimal import Decimal, InvalidOperation
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data" / "task_11"
GRADE_BANDS = [(Decimal("85"), "A"), (Decimal("70"), "B"), (Decimal("55"), "C"), (Decimal("40"), "D")]
REPORT_COLUMNS = ["rank", "student_id", "name", "subjects", "average", "grade"]
REJECT_COLUMNS = ["line_number", "student_id", "subject", "marks", "reason"]


def grade_for(average):
    for threshold, letter in GRADE_BANDS:
        if average >= threshold:
            return letter
    return "F"


def _id_sort_key(student_id):
    return (0, int(student_id), "") if student_id.isdigit() else (1, 0, student_id)


def _read(path, required):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [f.strip().lower() for f in (reader.fieldnames or [])]
        missing = [c for c in required if c not in fields]
        if missing:
            raise ValueError(f"{Path(path).name}: missing required column(s): {missing}")
        reader.fieldnames = fields
        for row in reader:
            yield reader.line_num, {k: (v or "").strip() for k, v in row.items() if k is not None}


def _parse_marks(text):
    try:
        marks = Decimal(text)
    except InvalidOperation:
        return None, "marks is not a number"
    if not marks.is_finite():
        return None, "marks is not a finite number"
    if not Decimal(0) <= marks <= Decimal(100):
        return None, "marks outside 0-100"
    return marks, None


def build_report(students_path, scores_path, report_path="report.csv",
                 rejected_path="rejected_scores.csv"):
    students = {}
    for line, row in _read(students_path, ["student_id", "name"]):
        sid = row["student_id"]
        if not sid or not row["name"]:
            raise ValueError(f"{Path(students_path).name} line {line}: student_id and name are required")
        if sid in students:
            raise ValueError(f"{Path(students_path).name} line {line}: duplicate student_id {sid!r}")
        students[sid] = row["name"]

    marks_by_student = {sid: {} for sid in students}       # sid -> {subject_key: (subject, marks)}
    rejected = []

    for line, row in _read(scores_path, ["student_id", "subject", "marks"]):
        sid, subject = row["student_id"], row["subject"]
        reason, marks = None, None
        if sid not in students:
            reason = "unknown student_id"
        elif not subject:
            reason = "missing subject"
        else:
            marks, reason = _parse_marks(row["marks"])
            if reason is None and subject.casefold() in marks_by_student[sid]:
                reason = "duplicate student-subject entry"
        if reason:
            rejected.append({"line_number": line, "student_id": sid, "subject": subject,
                             "marks": row["marks"], "reason": reason})
        else:
            marks_by_student[sid][subject.casefold()] = (subject, marks)

    rows = []
    for sid, name in students.items():
        entries = marks_by_student[sid]
        if entries:
            average = sum(m for _, m in entries.values()) / len(entries)
            rows.append({"student_id": sid, "name": name, "subjects": len(entries),
                         "average": average, "grade": grade_for(average)})
        else:
            rows.append({"student_id": sid, "name": name, "subjects": 0,
                         "average": None, "grade": "N/A"})

    scored = sorted((r for r in rows if r["average"] is not None),
                    key=lambda r: (-r["average"], _id_sort_key(r["student_id"])))
    unscored = sorted((r for r in rows if r["average"] is None),
                      key=lambda r: _id_sort_key(r["student_id"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    for row in unscored:
        row["rank"] = None
    ordered = scored + unscored

    for path, columns, data in (
        (report_path, REPORT_COLUMNS,
         [{"rank": r["rank"] if r["rank"] else "", "student_id": r["student_id"], "name": r["name"],
           "subjects": r["subjects"],
           "average": f"{r['average']:.2f}" if r["average"] is not None else "",
           "grade": r["grade"]} for r in ordered]),
        (rejected_path, REJECT_COLUMNS, rejected),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)

    return {"report": ordered, "rejected": rejected}


def create_sample_files(directory=SAMPLE_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "students.csv").write_text(
        "student_id,name\n101,Ayesha Khan\n102,Bilal Ahmed\n103,Sara Malik\n104,Usman Ali\n",
        encoding="utf-8")
    (directory / "scores.csv").write_text(
        "student_id,subject,marks\n"
        "101,Math,90\n101,Physics,80\n101,English,85\n"
        "102,Math,72\n102,Physics,65\n102,Math,99\n"
        "103,Math,100\n103,Physics,101\n103,English,abc\n103,Chemistry,95\n"
        "999,Math,50\n", encoding="utf-8")
    return directory


def demo():
    folder = create_sample_files()
    build_report(folder / "students.csv", folder / "scores.csv",
                 folder / "report.csv", folder / "rejected_scores.csv")
    print((folder / "report.csv").read_text())
    print((folder / "rejected_scores.csv").read_text())


class TestGradebook(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_case(self, students, scores):
        (self.dir / "s.csv").write_text(students, encoding="utf-8")
        (self.dir / "sc.csv").write_text(scores, encoding="utf-8")
        return build_report(self.dir / "s.csv", self.dir / "sc.csv",
                            self.dir / "report.csv", self.dir / "rej.csv")

    def test_exam_sample(self):
        result = self.run_case("student_id,name\n101,Ayesha\n",
                               "student_id,subject,marks\n101,Math,90\n101,Phy,80\n101,Eng,85\n")
        row = result["report"][0]
        self.assertEqual((row["average"], row["grade"], row["rank"]), (Decimal("85"), "A", 1))
        with open(self.dir / "report.csv", newline="") as fh:
            self.assertEqual(list(csv.DictReader(fh))[0]["average"], "85.00")

    def test_grade_boundaries(self):
        for marks, grade in [(100, "A"), (85, "A"), (84.99, "B"), (70, "B"), (69.99, "C"),
                             (55, "C"), (54.99, "D"), (40, "D"), (39.99, "F"), (0, "F")]:
            self.assertEqual(grade_for(Decimal(str(marks))), grade, marks)

    def test_rejections(self):
        result = self.run_case(
            "student_id,name\n1,A\n2,B\n",
            "student_id,subject,marks\n1,Math,50\n1,MATH,60\n1,Sci,101\n1,Eng,-1\n1,Art,x\n"
            "9,Math,50\n2,Math,\n2,,50\n")
        reasons = [r["reason"] for r in result["rejected"]]
        self.assertEqual(len(reasons), 7)
        self.assertIn("duplicate student-subject entry", reasons)
        self.assertIn("unknown student_id", reasons)
        self.assertEqual(result["report"][0]["average"], Decimal("50"))     # first Math kept

    def test_boundary_marks_accepted(self):
        result = self.run_case("student_id,name\n1,A\n", "student_id,subject,marks\n1,X,0\n1,Y,100\n")
        self.assertEqual(result["report"][0]["average"], Decimal("50"))
        self.assertEqual(result["rejected"], [])

    def test_ranking_by_average_then_id(self):
        result = self.run_case(
            "student_id,name\n10,Ten\n9,Nine\n2,Two\n",
            "student_id,subject,marks\n10,M,80\n9,M,80\n2,M,95\n")
        self.assertEqual([(r["student_id"], r["rank"]) for r in result["report"]],
                         [("2", 1), ("9", 2), ("10", 3)])          # 9 before 10 numerically

    def test_student_without_scores_listed_unranked(self):
        result = self.run_case("student_id,name\n1,A\n2,B\n", "student_id,subject,marks\n1,M,50\n")
        self.assertEqual((result["report"][1]["grade"], result["report"][1]["rank"]), ("N/A", None))

    def test_bad_input_files(self):
        with self.assertRaises(ValueError):
            self.run_case("student_id,name\n1,A\n1,B\n", "student_id,subject,marks\n")
        with self.assertRaises(ValueError):
            self.run_case("student_id\n1\n", "student_id,subject,marks\n")
        with self.assertRaises(ValueError):
            self.run_case("student_id,name\n1,A\n", "student_id,marks\n1,5\n")
        with self.assertRaises(FileNotFoundError):
            build_report(self.dir / "x.csv", self.dir / "y.csv")

    def test_sample_generator(self):
        folder = create_sample_files(self.dir / "sample")
        result = build_report(folder / "students.csv", folder / "scores.csv",
                              folder / "r.csv", folder / "rej.csv")
        self.assertEqual([r["student_id"] for r in result["report"]], ["103", "101", "102", "104"])
        self.assertEqual(len(result["rejected"]), 4)


if __name__ == "__main__":
    demo()
    unittest.main(verbosity=1)
