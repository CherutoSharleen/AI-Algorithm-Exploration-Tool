import unittest
from pathlib import Path

from evaluate_queries import run


class EvaluateQueriesTests(unittest.TestCase):
    def test_report_shape(self) -> None:
        report = run(Path(__file__).resolve().parent)
        self.assertGreaterEqual(len(report), 1)
        self.assertIn("query", report[0])
        self.assertIn("top_title", report[0])


if __name__ == "__main__":
    unittest.main()
