import unittest
from pathlib import Path

from bm25_search import document_terms, expanded_query_tokens, load_documents, search, search_with_matched_terms


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

    def test_search_with_matched_terms(self) -> None:
        docs = load_documents(Path(__file__).resolve().parent / "documents.json")
        results = search_with_matched_terms(docs, "robot club", top_k=1)
        self.assertEqual(results[0]["title"], "Library Robotics Club")
        self.assertIn("club", results[0]["matched_terms"])

    def test_expanded_query_tokens_include_synonyms(self) -> None:
        tokens = expanded_query_tokens("robotics coding")
        self.assertIn("robot", tokens)
        self.assertIn("python", tokens)

    def test_document_terms(self) -> None:
        docs = load_documents(Path(__file__).resolve().parent / "documents.json")
        terms = document_terms(docs, "Library Robotics Club")
        self.assertIn("robots", terms)


if __name__ == "__main__":
    unittest.main()
