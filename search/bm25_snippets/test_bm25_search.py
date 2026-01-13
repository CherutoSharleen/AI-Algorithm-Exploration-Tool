import unittest
from pathlib import Path

from bm25_search import load_documents, search


class BM25SnippetTests(unittest.TestCase):
    def test_robot_query_ranks_robotics_doc_first(self) -> None:
        docs = load_documents(Path(__file__).resolve().parent / "documents.json")
        top = search(docs, "robot search demo", top_k=1)[0][0]
        self.assertEqual(top, "Library Robotics Club")

    def test_synonym_expansion_supports_robotics_term(self) -> None:
        docs = load_documents(Path(__file__).resolve().parent / "documents.json")
        top = search(docs, "robotics club", top_k=1)[0][0]
        self.assertEqual(top, "Library Robotics Club")

    def test_min_score_filters_zero_results(self) -> None:
        docs = load_documents(Path(__file__).resolve().parent / "documents.json")
        results = search(docs, "unmatchedterm", top_k=3, min_score=0.0001)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
