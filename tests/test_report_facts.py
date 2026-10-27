"""Reject the erroneous size and winner seen in the received report revision."""
from pathlib import Path
import tempfile
import unittest

from src.report_facts import ROOT, START, END, checked_report_text, render_facts


class ReportFactsTests(unittest.TestCase):
    def test_rejects_a_report_with_tampered_model_size(self):
        facts = render_facts(ROOT / "results")
        corrupted = facts.replace("4.975993", "1.050000")
        self.assertNotEqual(facts, corrupted)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "checkpoint.md"
            report.write_text(f"{START}\n{corrupted}\n{END}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale or modified"):
                checked_report_text(report, ROOT / "results")

    def test_refresh_repairs_false_winner_without_replacing_review_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "checkpoint.md"
            report.write_text(f"Teacher feedback\n{START}\nBest is TPE 0.9734\n{END}\nPending approval\n",
                              encoding="utf-8")
            checked_report_text(report, ROOT / "results", write=True)
            checked_report_text(report, ROOT / "results")
            text = report.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("Teacher feedback\n"))
            self.assertTrue(text.endswith("\nPending approval\n"))
            self.assertNotIn("Best is TPE 0.9734", text)
            self.assertIn("thuộc **random**", text)


if __name__ == "__main__":
    unittest.main()
